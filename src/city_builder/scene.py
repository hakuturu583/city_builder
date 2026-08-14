"""Blender side: turn surface groups into objects, and export.

``bpy`` is a dependency of this package, so this runs in-process — there is no
Blender executable to find and no JSON to shuttle. Everything above this module
is plain numpy/shapely and testable without Blender; this file is the only
place that touches the scene.
"""

from __future__ import annotations

import os
from collections.abc import Sequence

from . import classes
from .classes import SurfaceClass
from .geometry import Mesh, Polygon, Ribbon, merge_meshes, polygon_to_mesh, ribbon_to_mesh


def _material_nodes(name: str):
    """A fresh Principled material, wired to its output."""
    import bpy

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
    return mat, nodes, links, bsdf


def _material(name: str, colour, roughness: float, specular: float | None = None):
    import bpy

    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing

    mat, _nodes, _links, bsdf = _material_nodes(name)
    bsdf.inputs["Base Color"].default_value = (*colour, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    if specular is not None and "Specular IOR Level" in bsdf.inputs:
        bsdf.inputs["Specular IOR Level"].default_value = specular
    elif specular is not None and "Specular" in bsdf.inputs:
        bsdf.inputs["Specular"].default_value = specular
    return mat


def build_materials() -> dict[str, object]:
    return {
        "asphalt": _material("CityAsphalt", (0.055, 0.055, 0.058), 0.85, 0.15),
        "marking": _material("CityMarking", (0.90, 0.90, 0.88), 0.55),
        "concrete": _material("CityConcrete", (0.42, 0.42, 0.40), 0.90),
        "ground": _material("CityGround", (0.20, 0.19, 0.17), 1.00),
        "facade": _material("CityFacade", (0.52, 0.50, 0.47), 0.65),
        "roof": _material("CityRoof", (0.26, 0.26, 0.28), 0.80),
    }


def clear_scene() -> None:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)


def ensure_collection(name: str):
    import bpy

    existing = bpy.data.collections.get(name)
    if existing is not None:
        return existing
    collection = bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(collection)
    return collection


def group_to_mesh(shapes: Sequence[object], *, face_counts: list[int] | None = None) -> tuple[Mesh, int]:
    """Convert a group of ribbons/polygons into one mesh.

    ``face_counts``, if given, is filled with the face count each shape
    contributed, so a caller can map shapes back to face ranges.
    """
    meshes, skipped = [], 0
    for shape in shapes:
        if isinstance(shape, Ribbon):
            mesh, dropped = ribbon_to_mesh(shape)
        elif isinstance(shape, Polygon):
            mesh, dropped = polygon_to_mesh(shape)
        else:  # already a Mesh
            mesh, dropped = shape, 0
        skipped += dropped
        if face_counts is not None:
            face_counts.append(len(mesh.faces))
        if mesh.faces:
            meshes.append(mesh)
    return merge_meshes(meshes), skipped


def tag_object(obj, surface_class: SurfaceClass) -> None:
    """Record what this surface is, and whether its colour may be regenerated.

    Written three ways on purpose. Custom properties are the primary record and
    survive into glTF ``extras``; ``pass_index`` drives a Cycles ``IndexOB``
    pass for a segmentation render; and a colour attribute keeps the class on
    the faces themselves, so it survives a consumer that joins the objects or
    re-exports through a format with no object metadata.
    """
    obj["cb_class"] = surface_class.group
    obj["cb_label"] = surface_class.label
    obj["cb_paint"] = surface_class.paint
    obj["cb_pass_index"] = surface_class.pass_index
    obj.pass_index = surface_class.pass_index

    mesh = obj.data
    mesh["cb_class"] = surface_class.group
    mesh["cb_paint"] = surface_class.paint

    colour = (*surface_class.mask_colour, 1.0)
    # Leading underscore on purpose: glTF treats "_"-prefixed attributes as
    # application-specific, so a viewer ignores it. Exported as COLOR_0 it
    # would be multiplied into the base colour and tint the whole asset.
    attribute = mesh.color_attributes.new(name="_cb_mask", type="FLOAT_COLOR", domain="CORNER")
    for datum in attribute.data:
        datum.color = colour


def uv_from_xy(obj, tile_metres: float) -> None:
    """Planar UVs at a metric scale, so a tile repeats every ``tile_metres``.

    A generated tile is only useful if its size on the ground is known, and a
    real UV layer (rather than a procedural coordinate node) is also the only
    form that survives a glTF export.
    """
    mesh = obj.data
    layer = mesh.uv_layers.get("UVMap") or mesh.uv_layers.new(name="UVMap")
    scale = 1.0 / max(tile_metres, 1e-6)
    for loop in mesh.loops:
        x, y, _ = mesh.vertices[loop.vertex_index].co
        layer.data[loop.index].uv = (x * scale, y * scale)


def tiled_material(name: str, image_path: str, *, roughness: float = 0.95):
    """A material that repeats an image across the UVs it is given."""
    import bpy

    mat, nodes, links, bsdf = _material_nodes(name)
    bsdf.inputs["Roughness"].default_value = roughness

    texture = nodes.new("ShaderNodeTexImage")
    texture.location = (-400, 0)
    texture.image = bpy.data.images.load(os.path.abspath(image_path))
    texture.extension = "REPEAT"
    links.new(texture.outputs["Color"], bsdf.inputs["Base Color"])
    return mat


def add_object(name: str, mesh: Mesh, material=None, surface_class: SurfaceClass | None = None,
               materials: Sequence[object] | None = None, face_materials: Sequence[int] | None = None):
    """Create a Blender object from a mesh and link it into its collection.

    ``materials`` + ``face_materials`` give one object several materials — a
    facade sheet per building without one object per building, which would
    multiply the draw calls by a thousand.
    """
    import bpy

    if not mesh.faces:
        return None

    data = bpy.data.meshes.new(f"{name}_Mesh")
    data.from_pydata([tuple(v) for v in mesh.vertices], [], mesh.faces)
    data.update()

    if mesh.uvs:
        layer = data.uv_layers.new(name="UVMap")
        for loop in data.loops:
            layer.data[loop.index].uv = mesh.uvs[loop.vertex_index]

    obj = bpy.data.objects.new(name, data)
    for slot in (materials or ([material] if material is not None else [])):
        data.materials.append(slot)
    if face_materials is not None:
        for polygon, index in zip(data.polygons, face_materials):
            polygon.material_index = index

    ensure_collection(name).objects.link(obj)
    if surface_class is not None:
        tag_object(obj, surface_class)
    return obj


def apply_tiled_texture(obj, image_path: str, tile_metres: float, *, name: str | None = None):
    """Give an object a repeating image material at a known metric scale."""
    material = tiled_material(name or f"{obj.name}Tiled", image_path)
    obj.data.materials.clear()
    obj.data.materials.append(material)
    uv_from_xy(obj, tile_metres)
    return material


def assign_sheets(sheet_paths: Sequence[str], floors: Sequence[int] | None = None,
                  *, count: int | None = None, seed: int = 0) -> list[int]:
    """Which facade sheet each building wears.

    Not a free choice. The facade UV normalises V over the building's height,
    so a sheet drawn for six floors only reads correctly on a six-floor
    building — anywhere else its windows stretch off the storeys. Buildings are
    matched to sheets by floor count, and the shuffling for variety happens
    *within* a matching group.

    Falls back to the nearest available floor count when nothing matches
    exactly, and to a plain even spread when the sheets do not declare what
    they were drawn for (a hand-made tile, say).
    """
    import random

    from .facade_layout import sheet_floors

    rng = random.Random(seed)
    total = len(floors) if floors is not None else (count or 0)

    by_floor: dict[int, list[int]] = {}
    for index, path in enumerate(sheet_paths):
        declared = sheet_floors(path)
        if declared is not None:
            by_floor.setdefault(declared, []).append(index)

    if floors is None or not by_floor:
        order = [i % max(1, len(sheet_paths)) for i in range(total)]
        rng.shuffle(order)
        return order

    available = sorted(by_floor)
    return [
        rng.choice(by_floor[min(available, key=lambda f: (abs(f - wanted), f))])
        for wanted in floors
    ]


def apply_facade_sheets(obj, image_paths: Sequence[str], face_counts: Sequence[int],
                        floors: Sequence[int] | None = None, seed: int = 0):
    """Give one merged building object a different sheet per building.

    Material slots rather than objects: 1200 objects would be 1200 draw calls
    for no visual gain, while slots split into that many primitives only when
    a renderer needs them to.
    """
    materials = [tiled_material(f"CityFacade{i:03d}", path, roughness=0.6)
                 for i, path in enumerate(image_paths)]
    obj.data.materials.clear()
    for material in materials:
        obj.data.materials.append(material)

    choice = assign_sheets(image_paths, floors, count=len(face_counts), seed=seed)
    face_materials = []
    for shape_index, faces in enumerate(face_counts):
        face_materials.extend([choice[shape_index]] * faces)
    for polygon, index in zip(obj.data.polygons, face_materials):
        polygon.material_index = index
    return materials


def build(groups: dict[str, Sequence[object]], *, verbose: bool = True) -> dict[str, object]:
    """Build every group into its own object/collection. Returns the objects."""
    materials = build_materials()
    objects = {}
    face_counts: dict[str, list[int]] = {}
    for name, shapes in groups.items():
        counts: list[int] = []
        mesh, skipped = group_to_mesh(shapes, face_counts=counts)
        face_counts[name] = counts
        surface_class = classes.get(name)
        obj = add_object(name, mesh, materials.get(surface_class.material), surface_class)
        if obj is None:
            continue
        objects[name] = obj
        if verbose:
            note = f", {skipped} degenerate face(s) skipped" if skipped else ""
            print(f"[scene] {name}: {len(shapes)} shape(s) → {len(mesh.faces)} face(s) "
                  f"[{surface_class.label}/{surface_class.paint}]{note}")

    build.face_counts = face_counts  # for callers wiring per-shape materials
    return objects


def save(path: str) -> None:
    import bpy

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(path))
    print(f"[scene] wrote {path}")


def export_glb(path: str) -> None:
    import bpy

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    bpy.ops.export_scene.gltf(
        filepath=os.path.abspath(path),
        export_format="GLB",
        use_active_scene=True,
        export_extras=True,  # carries cb_class / cb_paint through to glTF extras
        export_attributes=True,  # …and _cb_mask as the _CB_MASK custom attribute
        export_all_vertex_colors=False,  # keep it out of COLOR_0, see tag_object
    )
    print(f"[scene] wrote {path}")


def verify_ground_clearance(*, samples: int = 6000, lift: float = 0.6, seed: int = 0) -> dict[str, float]:
    """Check that the ground never comes up through the carriageway.

    Casts a ray straight down at road vertices from just above them: if the
    first surface hit is the ground, the ground is on top of the road there.
    This is the regression the ground mesh exists to prevent, and eyeballing a
    render is not a test — three earlier approaches looked plausible and were
    burying a quarter of the network.
    """
    import random

    import bpy
    import mathutils

    depsgraph = bpy.context.evaluated_depsgraph_get()
    points = [
        tuple(obj.matrix_world @ v.co)
        for name in ("Roads", "Junctions")
        if (obj := bpy.data.objects.get(name)) is not None
        for v in obj.data.vertices
    ]
    if not points:
        raise RuntimeError("no Roads/Junctions objects in the scene")

    random.seed(seed)
    chosen = random.sample(points, min(samples, len(points)))

    down = mathutils.Vector((0, 0, -1))
    hits = above = 0
    worst = 0.0
    for x, y, z in chosen:
        ok, location, _, _, obj, _ = bpy.context.scene.ray_cast(
            depsgraph, mathutils.Vector((x, y, z + lift)), down
        )
        if not ok:
            continue
        hits += 1
        if obj.name == "Ground":
            above += 1
            worst = max(worst, location.z - z)

    return {
        "samples": hits,
        "ground_above_road": above,
        "ground_above_road_pct": round(above / max(1, hits) * 100, 2),
        "worst_m": round(worst, 4),
    }


# ---------------------------------------------------------------------------
# Driving through it
# ---------------------------------------------------------------------------


def sunlit(strength: float = 3.5, elevation: float = 48.0, azimuth: float = 35.0):
    """A sun and a sky, for a scene built without either.

    The build makes surfaces, not a photograph; a scene with no light renders
    black, and one lit only by a grey world renders flat.
    """
    import math

    import bpy

    sun_data = bpy.data.lights.new("Sun", type="SUN")
    sun_data.energy = strength
    sun_data.angle = math.radians(2.0)  # a soft edge to the shadows
    sun = bpy.data.objects.new("Sun", sun_data)
    bpy.context.scene.collection.objects.link(sun)
    sun.rotation_euler = (math.radians(elevation), 0.0, math.radians(azimuth))

    world = bpy.data.worlds.new("Sky")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.42, 0.52, 0.68, 1.0)
    background.inputs[1].default_value = 1.1
    bpy.context.scene.world = world
    return sun


def animate_camera(path, *, lens: float = 30.0, name: str = "DriveCam"):
    """Keyframe a camera along ``(position, target)`` pairs, one pair per frame.

    Rotations are keyed as quaternions with each one flipped into the same
    hemisphere as the last. Euler angles would be simpler and wrong: the yaw
    wraps through ±180° somewhere on any route that turns around, and the
    interpolation between the two keyframes either side of the wrap spins the
    camera all the way back round.
    """
    import bpy
    import mathutils

    # Set before any key is inserted: with a keyframe on every frame, Bezier
    # handles ease in and out of each one and the drive comes out stuttering.
    # Setting it here rather than editing the curves afterwards also avoids the
    # F-curve API, which moved when actions became slotted in Blender 4.4.
    bpy.context.preferences.edit.keyframe_new_interpolation_type = "LINEAR"

    data = bpy.data.cameras.new(name)
    data.lens = lens
    data.clip_end = 4000.0
    camera = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(camera)
    camera.rotation_mode = "QUATERNION"
    bpy.context.scene.camera = camera

    previous = None
    for frame, (position, target) in enumerate(path, start=1):
        camera.location = position
        direction = mathutils.Vector(target) - mathutils.Vector(position)
        rotation = direction.to_track_quat("-Z", "Y")
        if previous is not None and rotation.dot(previous) < 0.0:
            rotation.negate()
        previous = rotation.copy()

        camera.rotation_quaternion = rotation
        camera.keyframe_insert("location", frame=frame)
        camera.keyframe_insert("rotation_quaternion", frame=frame)

    bpy.context.scene.frame_start = 1
    bpy.context.scene.frame_end = len(path)
    return camera


def render_animation(output_path: str, *, frames: int, fps: int = 30,
                     resolution=(1280, 720), samples: int = 24,
                     engine: str = "BLENDER_EEVEE", keep_frames: bool = False,
                     verbose: bool = True) -> str:
    """Render the current camera animation and encode it to H.264.

    Through a PNG sequence and the system ffmpeg rather than Blender's own
    video writer: the ``bpy`` wheel on PyPI is built without FFmpeg, so its
    ``FFMPEG`` output format does not exist. Frames on disk are no loss anyway —
    a render that dies at frame 700 can be encoded from what it got.
    """
    import shutil
    import subprocess
    import tempfile

    import bpy

    scene = bpy.context.scene
    scene.render.engine = engine
    if engine == "CYCLES":
        scene.cycles.samples = samples
        scene.cycles.device = "GPU"
        preferences = bpy.context.preferences.addons["cycles"].preferences
        preferences.compute_device_type = "OPTIX"
        preferences.get_devices()
        for device in preferences.devices:
            device.use = device.type in ("OPTIX", "CUDA")
    else:
        scene.eevee.taa_render_samples = samples

    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.fps = fps
    scene.frame_end = frames
    scene.render.image_settings.file_format = "PNG"

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH; cannot encode the video")

    output_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    directory = tempfile.mkdtemp(prefix="city_builder_frames_")
    scene.render.filepath = os.path.join(directory, "frame_")

    if verbose:
        print(f"[drive] {frames} frames at {fps} fps ({frames / fps:.0f} s), "
              f"{resolution[0]}x{resolution[1]}, {engine}")
    try:
        bpy.ops.render.render(animation=True)
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-framerate", str(fps),
             "-i", os.path.join(directory, "frame_%04d.png"),
             "-c:v", "libx264", "-preset", "slow", "-crf", "20",
             "-pix_fmt", "yuv420p", output_path],
            check=True,
        )
    finally:
        if not keep_frames:
            shutil.rmtree(directory, ignore_errors=True)
        elif verbose:
            print(f"[drive] frames kept in {directory}")
    return output_path
