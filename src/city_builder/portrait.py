"""One building, photographed out of the scene for a reconstruction to work from.

:mod:`city_builder.reconstruct` needs a picture of the building it is about to
model. Asking an image model for one does not work: measured over seven
promptings — metres, ratios, "a long rectangular slab" — the building that came
back had a plan aspect of 1.00 every time, against the 1.63 the plot wanted, and
the fitted footprint stalled at an IoU of 0.68. The reconstruction model is not
the problem: handed a picture of the *procedural* building instead, the same
model returned a mesh of plan aspect 1.91 and the fit reached 0.867.

So the shape comes from the plot, through a render, and only the surfaces come
from a model. That is the same division of labour the rest of this package
uses — the map is trusted for geometry and a diffusion model for appearance.
Over four plots the footprint IoU came out at 0.97 on average against a ceiling
of 1.0, and the plan aspect within a few per cent of what the plot asked for.

The elevation is the one setting that matters, and it was measured over those
four plots:

=========  ===================================================================
12°        0.983 0.969 **0.897** 0.965 — the roof is a sliver and the depth is
           guessed
35°        0.987 0.975 0.971 0.981 — the default
55°        0.971 0.987 0.974 0.983
=========  ===================================================================

Two things make this much simpler than a camera move.

**No mask pass.** Everything but the subject is deleted and the film is
transparent, so the alpha comes out of the render itself. That is also exactly
the form the reconstruction wants, and it is what keeps it from reaching for a
background-removal model.

**The view is chosen from the plot.** A three-quarter view has to show two
faces, and which direction that is depends on which way the building's long
axis runs — so the azimuth is measured off that axis rather than off north.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float, float]


# ---------------------------------------------------------------------------
# Where to stand
# ---------------------------------------------------------------------------


def long_axis_deg(footprint: Sequence[Sequence[float]]) -> float:
    """Which way the building's long side runs, in degrees anticlockwise from +X."""
    from shapely.geometry import Polygon as ShapelyPolygon

    plot = ShapelyPolygon([(float(x), float(y)) for x, y in footprint])
    if not plot.is_valid:
        plot = plot.buffer(0)
    corners = list(plot.minimum_rotated_rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        return 0.0
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    (ax, ay), (bx, by) = max(edges, key=lambda e: math.dist(e[0], e[1]))
    return math.degrees(math.atan2(by - ay, bx - ax)) % 180.0


def enclosing_circle(ring: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """Centre and radius of a circle covering a footprint, in plan.

    The centre of the bounding box and the furthest vertex from it — not the
    minimal circle. A framing a few per cent loose costs nothing; one a few per
    cent tight cuts the corner off the building, and a reconstruction of a
    building with a corner missing is a reconstruction of a different building.
    """
    if not ring:
        raise ValueError("an empty footprint has no circle")
    xs = [float(p[0]) for p in ring]
    ys = [float(p[1]) for p in ring]
    cx, cy = (min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0
    return cx, cy, max(math.dist((x, y), (cx, cy)) for x, y in zip(xs, ys))


def framing_distance(radius: float, height: float, *, lens: float = 50.0,
                     sensor: float = 36.0, resolution: tuple[int, int] = (1024, 1024),
                     elevation_deg: float = 35.0, margin: float = 1.25) -> float:
    """How far from the middle of the building the camera has to sit.

    The subject is treated as the cylinder that contains it, and the two
    constraints are separated: across the frame the plan circle has to fit, and
    up the frame the near bottom corner is the one that leaves first, because
    the camera is above the middle of the building looking down.

    Taking the bounding *sphere* instead is a line of arithmetic shorter and
    much too far back — the sphere round a wide, low building is as wide as the
    building — and the resulting picture is mostly empty.
    """
    if radius <= 0.0 or height <= 0.0:
        raise ValueError("a building with no extent cannot be framed")

    long_side, short_side = max(resolution), min(resolution)
    across = math.tan(math.atan(sensor / 2.0 / lens))
    up = math.tan(math.atan(sensor * short_side / long_side / 2.0 / lens))

    phi = math.radians(elevation_deg)
    half = height / 2.0
    wide = half * math.sin(phi) + radius * math.hypot(math.cos(phi), 1.0 / across)
    tall = max(
        side * radius * math.cos(phi) + rim * half * math.sin(phi)
        + abs(rim * half * math.cos(phi) - side * radius * math.sin(phi)) / up
        for side in (1.0, -1.0) for rim in (1.0, -1.0)
    )
    return margin * max(wide, tall)


@dataclass
class PortraitOptions:
    """The one view a reconstruction is given."""

    lens: float = 50.0
    # Measured off the building's own long axis, not off north: a
    # three-quarter view is one that shows a long face and a short one, and
    # 35 degrees off the long axis does that whichever way the street runs.
    azimuth_off_axis_deg: float = 35.0
    # High enough to read the plan. The reconstruction has to infer depth from
    # one picture, and at a low angle the roof is a sliver — the first run of
    # this, off a 12-degree orbit frame, came back 17 per cent too deep.
    elevation_deg: float = 35.0
    margin: float = 1.25
    size: int = 1024  # square, because the reconstruction squares it anyway
    samples: int = 64


def portrait_pose(plot: dict[str, Any],
                  options: PortraitOptions | None = None) -> tuple[Point, Point]:
    """``(position, target)`` for the one view, worked out from the plot."""
    options = options or PortraitOptions()
    cx, cy, radius = enclosing_circle(plot["footprint"])
    height = float(plot["height"])
    centre = (cx, cy, float(plot["base_z"]) + height / 2.0)

    distance = framing_distance(radius, height, lens=options.lens,
                                resolution=(options.size, options.size),
                                elevation_deg=options.elevation_deg,
                                margin=options.margin)
    theta = math.radians(long_axis_deg(plot["footprint"]) + options.azimuth_off_axis_deg)
    phi = math.radians(options.elevation_deg)
    return ((centre[0] + distance * math.cos(phi) * math.cos(theta),
             centre[1] + distance * math.cos(phi) * math.sin(theta),
             centre[2] + distance * math.sin(phi)), centre)


# ---------------------------------------------------------------------------
# Which faces are which building
# ---------------------------------------------------------------------------


def face_range(face_counts: Sequence[int], index: int) -> tuple[int, int]:
    """``(first, last+1)`` face of one building in its merged object.

    A scene has one Buildings object and one Roofs object however many
    buildings there are, so a building is a range of faces, in the order
    ``BuildResult.plots`` is in.
    """
    if not 0 <= index < len(face_counts):
        raise IndexError(f"no building {index}; the scene has {len(face_counts)}")
    start = sum(face_counts[:index])
    return start, start + face_counts[index]


def keep_only(obj, start: int, end: int) -> int:
    """Delete every face of ``obj`` outside ``[start, end)``. Returns what is left.

    Through bmesh, because the per-face material index is what carries the
    facade sheet this building wears, and bmesh keeps it.
    """
    import bmesh

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    doomed = [face for index, face in enumerate(bm.faces) if not start <= index < end]
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return len(mesh.polygons)


# ---------------------------------------------------------------------------
# The render
# ---------------------------------------------------------------------------


def render_portrait(scene, building: int, out_path: str, *,
                    options: PortraitOptions | None = None, facade_dir: str | None = None,
                    road_texture: str | None = None, ground_texture: str | None = None,
                    massing_options=None, massing_seed: int | None = None,
                    marking_options=None, verbose: bool = True) -> dict[str, Any]:
    """One RGBA picture of one building of a built scene.

    Everything else — the other buildings, the ground, the road — is taken out
    rather than masked, and the film is transparent, so what is written is the
    subject and nothing else. Dress it: the facade sheets are the only thing in
    the picture that says what the building is made of, and the reconstruction
    carries that appearance into the mesh's textures.

    ``massing_seed`` replaces the plot's extruded box with a building that has
    something going on — see :mod:`city_builder.massing`. Photograph the box and
    the reconstruction faithfully returns a box, which is the whole reason that
    module exists.
    """
    import shutil
    import tempfile
    import time

    options = options or PortraitOptions()
    plots = scene.result.plots
    if not plots:
        raise ValueError("this scene has no buildings; build it with buildings=True")
    if not 0 <= building < len(plots):
        raise IndexError(f"no building {building}; the scene has {len(plots)}")

    markings = tempfile.mkdtemp(prefix="city-markings-")
    started = time.time()
    try:
        features = _render(scene, building, out_path, options=options, facade_dir=facade_dir,
                           road_texture=road_texture, ground_texture=ground_texture,
                           massing_options=massing_options, massing_seed=massing_seed,
                           marking_options=marking_options, markings_dir=markings,
                           verbose=verbose)
    finally:
        shutil.rmtree(markings, ignore_errors=True)

    plot = plots[building]
    position, target = portrait_pose(plot, options)
    return {
        "image": out_path,
        "building": building,
        "features": features,
        "camera": [round(v, 2) for v in position],
        "looking_at": [round(v, 2) for v in target],
        "elevation_deg": options.elevation_deg,
        "took_seconds": round(time.time() - started, 1),
    }


def _replace_with_massing(plot: dict[str, Any], massing_options, seed: int, *,
                          facade_dir: str | None) -> list[str]:
    """Take the plot's box out of the scene and put a varied building in its place.

    The sheet is chosen by floor count, the same rule the whole-scene facade
    pass uses: the wall UV normalises over the building's height, so a sheet
    drawn for six floors only reads on a six-floor building.
    """
    import bpy

    from . import massing as massing_module
    from . import scene as scene_module
    from .classes import get as surface_class

    for name in ("Buildings", "Roofs"):
        obj = bpy.data.objects.get(name)
        if obj is not None:
            bpy.data.objects.remove(obj, do_unlink=True)

    built = massing_module.build(plot, massing_options, seed)
    material = None
    if facade_dir and os.path.isdir(facade_dir):
        sheets = sorted(os.path.join(facade_dir, f) for f in os.listdir(facade_dir)
                        if f.endswith(".png"))
        if sheets:
            choice = scene_module.assign_sheets(sheets, [int(plot.get("floors") or 1)],
                                                seed=seed)
            material = scene_module.tiled_material(f"MassingFacade{seed}",
                                                   sheets[choice[0]], roughness=0.6)

    for group, meshes in (("Buildings", built["Buildings"]), ("Roofs", built["Roofs"])):
        for index, mesh in enumerate(meshes):
            scene_module.add_object(f"{group}{index:03d}", mesh,
                                    material if group == "Buildings" else None,
                                    surface_class(group))
    return built["features"]


def _render(scene, building: int, out_path: str, *, options: PortraitOptions,
            facade_dir: str | None, road_texture: str | None, ground_texture: str | None,
            massing_options, massing_seed: int | None,
            marking_options, markings_dir: str, verbose: bool) -> list[str]:
    import bpy
    import mathutils

    from . import scene as scene_module
    from .build import build_scene

    build_scene(scene.result, facade_dir=facade_dir, road_texture=road_texture,
                ground_texture=ground_texture, road_tile_metres=4.0,
                marking_options=marking_options, markings_dir=markings_dir, verbose=False)
    if not any(obj.type == "LIGHT" for obj in bpy.data.objects):
        scene_module.sunlit()

    plot = scene.result.plots[building]
    features: list[str] = []
    if massing_seed is None:
        kept = 0
        for name in ("Buildings", "Roofs"):
            counts = scene_module.build.face_counts.get(name)
            obj = bpy.data.objects.get(name)
            if not counts or obj is None:
                continue
            kept += keep_only(obj, *face_range(counts, building))
        if not kept:
            raise RuntimeError("the built scene has no faces for this building")
    else:
        features = _replace_with_massing(plot, massing_options, massing_seed,
                                         facade_dir=facade_dir)

    # Everything that is not the subject goes. The ground and the road are what
    # a video model needed to know where it was; a reconstruction of one
    # building needs them gone, and gone is cleaner than masked.
    subject = {"Buildings", "Roofs"} if massing_seed is None else {
        o.name for o in bpy.data.objects
        if o.type == "MESH" and (o.name.startswith("Buildings") or o.name.startswith("Roofs"))}
    for obj in list(bpy.data.objects):
        if obj.type == "MESH" and obj.name not in subject:
            bpy.data.objects.remove(obj, do_unlink=True)

    position, target = portrait_pose(plot, options)

    data = bpy.data.cameras.new("Portrait")
    data.lens = options.lens
    data.clip_end = 10000.0
    camera = bpy.data.objects.new("Portrait", data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = position
    camera.rotation_euler = (
        mathutils.Vector(target) - mathutils.Vector(position)
    ).to_track_quat("-Z", "Y").to_euler()

    blender = bpy.context.scene
    blender.camera = camera
    blender.render.engine = "BLENDER_EEVEE"
    blender.eevee.taa_render_samples = options.samples
    blender.render.resolution_x = blender.render.resolution_y = options.size
    blender.render.film_transparent = True  # the alpha the reconstruction needs
    blender.render.image_settings.file_format = "PNG"
    blender.render.image_settings.color_mode = "RGBA"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    blender.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    if verbose:
        print(f"[portrait] building {building}: {', '.join(features) or 'the plot as built'}, "
              f"{options.elevation_deg:g} deg above, {options.size}px")
    return features
