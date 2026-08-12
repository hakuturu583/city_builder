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


def _material(name: str, colour, roughness: float, specular: float | None = None):
    import bpy

    existing = bpy.data.materials.get(name)
    if existing is not None:
        return existing

    mat = bpy.data.materials.new(name=name)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    nodes.clear()
    output = nodes.new("ShaderNodeOutputMaterial")
    bsdf = nodes.new("ShaderNodeBsdfPrincipled")
    output.location = (400, 0)
    links.new(bsdf.outputs["BSDF"], output.inputs["Surface"])
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


def group_to_mesh(shapes: Sequence[object]) -> tuple[Mesh, int]:
    """Convert a group of ribbons/polygons into one mesh."""
    meshes, skipped = [], 0
    for shape in shapes:
        if isinstance(shape, Ribbon):
            mesh, dropped = ribbon_to_mesh(shape)
        elif isinstance(shape, Polygon):
            mesh, dropped = polygon_to_mesh(shape)
        else:  # already a Mesh
            mesh, dropped = shape, 0
        skipped += dropped
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


def add_object(name: str, mesh: Mesh, material=None, surface_class: SurfaceClass | None = None):
    """Create a Blender object from a mesh and link it into its collection."""
    import bpy

    if not mesh.faces:
        return None

    data = bpy.data.meshes.new(f"{name}_Mesh")
    data.from_pydata([tuple(v) for v in mesh.vertices], [], mesh.faces)
    data.update()

    obj = bpy.data.objects.new(name, data)
    if material is not None:
        data.materials.append(material)
    ensure_collection(name).objects.link(obj)
    if surface_class is not None:
        tag_object(obj, surface_class)
    return obj


def build(groups: dict[str, Sequence[object]], *, verbose: bool = True) -> dict[str, object]:
    """Build every group into its own object/collection. Returns the objects."""
    materials = build_materials()
    objects = {}
    for name, shapes in groups.items():
        mesh, skipped = group_to_mesh(shapes)
        surface_class = classes.get(name)
        obj = add_object(name, mesh, materials.get(surface_class.material), surface_class)
        if obj is None:
            continue
        objects[name] = obj
        if verbose:
            note = f", {skipped} degenerate face(s) skipped" if skipped else ""
            print(f"[scene] {name}: {len(shapes)} shape(s) → {len(mesh.faces)} face(s) "
                  f"[{surface_class.label}/{surface_class.paint}]{note}")
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
