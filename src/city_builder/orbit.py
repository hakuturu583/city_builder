"""One building, circled, and a mask saying which part of the frame may change.

The drive camera (:mod:`city_builder.route`) exists to look at the street. This
one exists to look at *one building*, because what sits downstream of it is not
a texturing pass but a reconstruction: a video model makes the procedural block
photoreal, and a mesh model turns that footage back into geometry. A
reconstruction wants the subject from all round it and wants the views to agree
with each other, which is the opposite of driving past.

Three things are decided here, and each of them is decided by something other
than taste.

**The frame count is not free.** H3 counts in 17k+5 — 5, 22, 39, 56, 73, 90,
107, 124 — and a closed turn divides 360° by the frame count. Only the counts
that are *also* a multiple of four land a frame exactly on each cardinal
azimuth, and those four are what a multiview reconstruction is conditioned on:
56, 124, 192. So 56, at 6.43° a frame, is the cheapest orbit that can hand over
four exact quadrant views, and it is the default. Asking for "about ninety
frames" and getting 90 would put the quadrants at frame 22.5.

**The distance comes from the framing.** A caller who names a radius has to
know the lens, the sensor and how tall the building came out; what they
actually want is "the whole building, in shot". So the subject is treated as
the sphere that contains it and the camera sits where that sphere fills the
narrower field of view. Doing it with a sphere rather than a box is what makes
the answer independent of the elevation the orbit is flown at.

**The mask is rendered, not derived.** Projecting the footprint would get the
silhouette wrong wherever anything stands in front of the building, and
something always does — a neighbour, a parapet, the deck of a viaduct. Rendering
the same camera path a second time with the subject white and everything else
black costs one more EEVEE pass and is right by construction, occlusion
included.

The mask matters because of where it is used. An H3 latent is
``[B,24,T,H/16,W/16]``: a mask meant for it has to be reduced 16-fold in space
and folded in time by the same non-uniform grouping ``_pixel_frames`` uses in
:mod:`city_builder.comfy_nodes` — not by a stride. That reduction belongs to the
sampler, not here; what belongs here is a mask that is exact at pixel scale and
frame-aligned with the clip it describes.

Nothing in this module imports bpy at import time.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

Point = tuple[float, float, float]

# The lengths H3 will accept: one sampling pass covers 5 pixel frames and each
# further latent token covers 17 more.
GRID_START = 5
GRID_STEP = 17


# ---------------------------------------------------------------------------
# How many frames, and which of them are the quadrants
# ---------------------------------------------------------------------------


def frame_counts(limit: int = 300, *, quadrants: bool = False) -> list[int]:
    """The frame counts the model accepts, up to ``limit``.

    With ``quadrants``, only those a closed 360° turn can divide into four —
    the counts that put a camera exactly on each cardinal azimuth.
    """
    counts = range(GRID_START, limit + 1, GRID_STEP)
    return [n for n in counts if not (quadrants and n % 4)]


def snap_frames(wanted: int, *, quadrants: bool = True) -> int:
    """The nearest usable frame count to ``wanted``, never zero.

    Ties go to the shorter orbit: the count is a cost, and a caller who says
    "about 90" and gets 56 rather than 124 has not been surprised expensively.
    """
    options = frame_counts(max(wanted, GRID_START) + 4 * GRID_STEP, quadrants=quadrants)
    return min(options, key=lambda n: (abs(n - wanted), n))


def quadrant_frames(frames: int) -> list[int]:
    """The four frame indices on the cardinal azimuths, for a closed orbit."""
    if frames % 4:
        raise ValueError(
            f"{frames} frames cannot be quartered; use snap_frames(quadrants=True) "
            f"— {', '.join(str(n) for n in frame_counts(200, quadrants=True))}")
    return [i * frames // 4 for i in range(4)]


# ---------------------------------------------------------------------------
# Where to put the camera
# ---------------------------------------------------------------------------


def enclosing_circle(ring: Sequence[Sequence[float]]) -> tuple[float, float, float]:
    """A circle covering a footprint in plan: ``(x, y, radius)``.

    The centre of the bounding box and the distance to the furthest vertex, not
    the minimal enclosing circle — for an L-shaped plot this is a few per cent
    wide. That is the right way to be wrong: a framing a few per cent loose
    costs nothing, and one a few per cent tight cuts the corner off the
    building in every frame of the orbit.
    """
    if not ring:
        raise ValueError("an empty footprint has no circle")
    xs = [float(p[0]) for p in ring]
    ys = [float(p[1]) for p in ring]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    radius = max(math.dist((x, y), (cx, cy)) for x, y in zip(xs, ys))
    return cx, cy, radius


def framing_distance(radius: float, height: float, *, lens: float = 35.0,
                     sensor: float = 36.0, resolution: tuple[int, int] = (832, 480),
                     elevation_deg: float = 12.0, margin: float = 1.15) -> float:
    """How far from the middle of the building the camera has to sit.

    The subject is the cylinder that contains it — ``radius`` about the vertical
    axis, ``height`` tall — which is the right envelope precisely because the
    camera goes all the way round: a cylinder looks the same from every azimuth,
    so one answer holds for the whole orbit.

    Taking the *sphere* instead would be a line of arithmetic shorter and much
    too far back. A building 35 m across and 10 m tall has a bounding sphere as
    wide as it is, and framing that sphere in the narrow field of view left the
    subject 13 % of the frame — most of the shot being ground and sky, and about
    a hundred pixels of building for a reconstruction to work from. The cylinder
    separates the two constraints that actually bind:

    * across the frame, the plan circle subtends ``r / sqrt(A² - r²cos²φ)``,
      whose worst case over the turn solves in closed form;
    * up the frame, the corner that leaves first is the *near bottom* one,
      because the camera is above the middle of the building looking down.

    Blender fits its 36 mm sensor to the longer side of the resolution, so a
    832x480 frame is 36 mm across and 20.8 mm tall.
    """
    if radius <= 0.0 or height <= 0.0:
        raise ValueError("a building with no extent cannot be framed")

    long_side, short_side = max(resolution), min(resolution)
    across = math.tan(math.atan(sensor / 2.0 / lens))
    up = math.tan(math.atan(sensor * short_side / long_side / 2.0 / lens))

    phi = math.radians(elevation_deg)
    half = height / 2.0
    wide = half * math.sin(phi) + radius * math.hypot(math.cos(phi), 1.0 / across)
    # The four extreme points of the cylinder in the plane of the orbit. Their
    # height in the image does not depend on the distance — only their depth
    # does — so each one states a distance directly.
    tall = max(
        side * radius * math.cos(phi) + rim * half * math.sin(phi)
        + abs(rim * half * math.cos(phi) - side * radius * math.sin(phi)) / up
        for side in (1.0, -1.0) for rim in (1.0, -1.0)
    )
    return margin * max(wide, tall)


def orbit_path(centre: Point, *, frames: int, distance: float,
               elevation_deg: float = 12.0, start_deg: float = 0.0,
               clockwise: bool = False) -> list[tuple[Point, Point]]:
    """``(position, target)`` per frame, once round ``centre``.

    Azimuth 0 puts the camera on the +X side of the building looking back along
    -X, and the turn advances anticlockwise seen from above unless
    ``clockwise``. The last frame stops one step short of the first, because
    the turn is closed: a frame at 360° is the frame at 0° again, and handing a
    video model a duplicate of its own first frame is asking for a stutter.
    """
    if frames < 1:
        raise ValueError("an orbit needs at least one frame")
    cx, cy, cz = centre
    step = 360.0 / frames * (-1.0 if clockwise else 1.0)
    phi = math.radians(elevation_deg)
    reach, lift = distance * math.cos(phi), distance * math.sin(phi)

    path = []
    for index in range(frames):
        theta = math.radians(start_deg + step * index)
        path.append((
            (cx + reach * math.cos(theta), cy + reach * math.sin(theta), cz + lift),
            (cx, cy, cz),
        ))
    return path


@dataclass
class OrbitOptions:
    """What to shoot, and how far round."""

    frames: int = 56  # the cheapest count that quarters exactly; see the module docstring
    lens: float = 35.0
    elevation_deg: float = 12.0  # a little above eye level, as a reconstruction wants
    start_deg: float = 0.0
    clockwise: bool = False
    margin: float = 1.15  # how much room round the building in shot

    width: int = 832  # what the refinement runs at, so nothing is resampled twice
    height: int = 480
    samples: int = 24
    fps: int = 24  # for looking at the clip; the model counts frames, not seconds

    # Which *buildings* stand besides the subject. The road and the ground are
    # never touched by any of these: they are what tells the video model what
    # kind of place this is, and they are also the only thing in frame that
    # says how big the building is.
    #
    # `hide` — the default — leaves the subject alone on the street. Every
    # other building in the clip is a building the reconstruction has to be
    # told to ignore, and the mask is a per-pixel instruction that a model
    # follows approximately; removing them is the version that cannot go wrong.
    #
    # `keep` leaves the whole block standing and merely forbids editing it.
    # Measured on a procedural block at 0.6 coverage it is unusable anyway: the
    # camera flies at the framing distance, which is inside the next block, and
    # 24 of 56 frames saw no part of the subject at all. `clear` is the middle
    # — the camera looks from a ring of radius `distance * cos(elevation)` at
    # the middle of that ring, so every sightline lies inside that disc; empty
    # the disc and the view cannot be blocked while the far city still stands.
    neighbours: str = "hide"

    def __post_init__(self) -> None:
        if self.neighbours not in ("keep", "clear", "hide"):
            raise ValueError(
                f"neighbours must be 'keep', 'clear' or 'hide', not {self.neighbours!r}")
        for name in ("width", "height"):
            if getattr(self, name) % 32:
                raise ValueError(f"orbit.{name} must be a multiple of 32 for the refinement")
        if self.frames not in frame_counts(self.frames):
            raise ValueError(
                f"orbit.frames must be 5, 22, 39, 56 … (17k+5); {self.frames} is not. "
                "snap_frames() picks the nearest.")


def plan_orbit(plot: dict[str, Any], options: OrbitOptions | None = None) -> dict[str, Any]:
    """The whole shot, worked out from one entry of ``BuildResult.plots``.

    Returned rather than rendered so it can be inspected — and so the numbers
    that the reconstruction downstream needs (the centre, the distance, which
    frames are the quadrants) exist before anything expensive runs.
    """
    options = options or OrbitOptions()
    ring = plot.get("footprint")
    if not ring:
        raise ValueError(
            "this plot has no footprint; it was built before buildings.generate kept one")

    cx, cy, radius = enclosing_circle(ring)
    height = float(plot["height"])
    centre = (cx, cy, float(plot["base_z"]) + height / 2.0)
    distance = framing_distance(radius, height, lens=options.lens,
                                resolution=(options.width, options.height),
                                elevation_deg=options.elevation_deg,
                                margin=options.margin)
    return {
        "centre": [round(v, 3) for v in centre],
        "plan_radius_m": round(radius, 3),
        "height_m": round(height, 3),
        "base_z": round(float(plot["base_z"]), 3),
        "distance_m": round(distance, 3),
        # The disc every sightline of this orbit stays inside.
        "reach_m": round(distance * math.cos(math.radians(options.elevation_deg)), 3),
        "frames": options.frames,
        "degrees_per_frame": round(360.0 / options.frames, 4),
        "quadrant_frames": quadrant_frames(options.frames) if options.frames % 4 == 0 else None,
        "elevation_deg": options.elevation_deg,
        "start_deg": options.start_deg,
        "clockwise": options.clockwise,
        "lens_mm": options.lens,
        "resolution": [options.width, options.height],
        "footprint": [list(p) for p in ring],
        "path": orbit_path(centre, frames=options.frames, distance=distance,
                           elevation_deg=options.elevation_deg,
                           start_deg=options.start_deg, clockwise=options.clockwise),
    }


# ---------------------------------------------------------------------------
# Which faces belong to which building
# ---------------------------------------------------------------------------


def blocking_buildings(plots: Sequence[dict[str, Any]], subject: int, reach: float) -> list[int]:
    """Every building that could get between this orbit's camera and its subject.

    The camera looks at the middle of a circle of radius ``reach`` from a point
    on it, so every sightline it ever has lies inside that disc. A building
    whose own circle touches the disc can be on one of those lines; one that
    does not, cannot — whatever the elevation, and whatever the frame. That is
    the whole test, and it is why clearing the disc is enough rather than a
    heuristic that has to be tuned per map.
    """
    centre = enclosing_circle(plots[subject]["footprint"])[:2]
    blocking = []
    for index, plot in enumerate(plots):
        if index == subject or not plot.get("footprint"):
            continue
        x, y, radius = enclosing_circle(plot["footprint"])
        if math.dist((x, y), centre) < reach + radius:
            blocking.append(index)
    return blocking


def face_range(face_counts: Sequence[int], index: int) -> tuple[int, int]:
    """``(first, last+1)`` face of one building in its merged object.

    A scene has one Buildings object and one Roofs object however many
    buildings there are — 1200 objects would be 1200 draw calls — so a building
    is a range of faces, and :func:`city_builder.scene.build` records the count
    each one contributed in the order ``BuildResult.plots`` is in.
    """
    if not 0 <= index < len(face_counts):
        raise IndexError(f"no building {index}; the scene has {len(face_counts)}")
    start = sum(face_counts[:index])
    return start, start + face_counts[index]


# ---------------------------------------------------------------------------
# Blender: the two passes
# ---------------------------------------------------------------------------


def remove_faces(obj, spans: Sequence[tuple[int, int]]) -> int:
    """Delete these face ranges from ``obj``. Returns how many faces are left.

    Through bmesh rather than by rebuilding the mesh, because the per-face
    material index is what carries the facade sheet each building wears, and
    bmesh keeps it.
    """
    import bmesh

    doomed_indices = {i for start, end in spans for i in range(start, end)}
    if not doomed_indices:
        return len(obj.data.polygons)

    mesh = obj.data
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    doomed = [face for index, face in enumerate(bm.faces) if index in doomed_indices]
    bmesh.ops.delete(bm, geom=doomed, context="FACES")
    bm.to_mesh(mesh)
    bm.free()
    mesh.update()
    return len(mesh.polygons)


def _emission(name: str, colour: tuple[float, float, float]):
    import bpy

    material = bpy.data.materials.new(name)
    material.use_nodes = True
    tree = material.node_tree
    tree.nodes.clear()
    emission = tree.nodes.new("ShaderNodeEmission")
    emission.inputs["Color"].default_value = (*colour, 1.0)
    emission.inputs["Strength"].default_value = 1.0
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    tree.links.new(emission.outputs["Emission"], output.inputs["Surface"])
    return material


def paint_mask(targets: dict[str, tuple[int, int] | None]) -> None:
    """Repaint the whole scene as a mask: the subject white, the rest black.

    Destructive, and meant to be. It runs after the beauty pass in the same
    Blender and off the same camera keys, because a mask rendered from geometry
    that is not exactly the geometry it describes is worse than no mask.

    ``targets`` maps object name to the face range that is the subject, or None
    for the whole object.

    Emission rather than a lit material, a black world, and the Standard view
    transform: with any of those wrong, white comes back somewhere near 0.8 and
    whoever consumes the mask has to guess a threshold. The sky is black
    because the sky is not the building.
    """
    import bpy

    black = _emission("MaskBlack", (0.0, 0.0, 0.0))
    white = _emission("MaskWhite", (1.0, 1.0, 1.0))

    for obj in bpy.data.objects:
        if obj.type != "MESH":
            continue
        mesh = obj.data
        mesh.materials.clear()
        mesh.materials.append(black)
        if obj.name not in targets:
            for polygon in mesh.polygons:
                polygon.material_index = 0
            continue
        span = targets[obj.name]
        mesh.materials.append(white)
        for index, polygon in enumerate(mesh.polygons):
            polygon.material_index = 1 if span is None or span[0] <= index < span[1] else 0

    world = bpy.data.worlds.new("MaskWorld")
    world.use_nodes = True
    background = world.node_tree.nodes["Background"]
    background.inputs[0].default_value = (0.0, 0.0, 0.0, 1.0)
    background.inputs[1].default_value = 0.0
    bpy.context.scene.world = world

    view = bpy.context.scene.view_settings
    view.view_transform = "Standard"
    view.look = "None"
    view.exposure = 0.0
    view.gamma = 1.0
    bpy.context.scene.render.film_transparent = False


def render_frames(directory: str, *, frames: int, resolution: tuple[int, int],
                  samples: int = 8, prefix: str = "mask_") -> list[str]:
    """Render the current camera animation to a PNG sequence.

    PNGs and not a video, for the mask. H.264 is 4:2:0 and lossy, and a mask
    that has been through it has grey where it had an edge; the whole value of
    this pass is that the answer at a pixel is not a matter of opinion.
    """
    import bpy

    os.makedirs(directory, exist_ok=True)
    blender = bpy.context.scene
    blender.render.engine = "BLENDER_EEVEE"
    blender.eevee.taa_render_samples = samples
    blender.render.resolution_x, blender.render.resolution_y = resolution
    blender.render.image_settings.file_format = "PNG"
    blender.render.image_settings.color_mode = "BW"
    blender.frame_start = 1
    blender.frame_end = frames
    blender.render.filepath = os.path.join(directory, prefix)
    bpy.ops.render.render(animation=True)

    return sorted(os.path.join(directory, name) for name in os.listdir(directory)
                  if name.startswith(prefix) and name.endswith(".png"))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def render_orbit(scene, building: int, out_dir: str, *, options: OrbitOptions | None = None,
                 facade_dir: str | None = None, road_texture: str | None = None,
                 ground_texture: str | None = None, tile_metres: float = 12.0,
                 road_tile_metres: float = 4.0, marking_options=None,
                 verbose: bool = True) -> dict[str, Any]:
    """Circle one building of a built scene: a clip, a mask, and the numbers.

    Writes ``orbit.mp4`` (what the refinement takes), ``mask/`` (a PNG per
    frame) and ``orbit.json`` (what the reconstruction downstream needs to put
    the mesh it makes back at the right size and heading).

    Dress it. ``facade_dir``, ``road_texture`` and ``ground_texture`` are the
    generated textures, and this is the one render where leaving them out costs
    something: a video model asked to make a grey box photoreal has nothing to
    tell it where the storeys are or which way the building faces, and at the
    low denoise this pipeline runs at it will not invent them. The provisional
    textures are the difference between "make this photoreal" and "imagine a
    building".

    ``road_tile_metres`` defaults to 4 rather than 12 for the same reason it
    does on the command line: asphalt aggregate is finer than paving, and a
    12 m road tile renders as cobblestone.
    """
    import shutil
    import tempfile

    options = options or OrbitOptions()
    plots = scene.result.plots
    if not plots:
        raise ValueError("this scene has no buildings; build it with buildings=True")
    if not 0 <= building < len(plots):
        raise IndexError(f"no building {building}; the scene has {len(plots)}")

    markings = tempfile.mkdtemp(prefix="city-markings-")
    try:
        return _render_orbit(scene, building, out_dir, options=options, facade_dir=facade_dir,
                             road_texture=road_texture, ground_texture=ground_texture,
                             tile_metres=tile_metres, road_tile_metres=road_tile_metres,
                             marking_options=marking_options, markings_dir=markings,
                             verbose=verbose)
    finally:
        shutil.rmtree(markings, ignore_errors=True)


def _render_orbit(scene, building: int, out_dir: str, *, options: OrbitOptions,
                  facade_dir: str | None, road_texture: str | None,
                  ground_texture: str | None, tile_metres: float, road_tile_metres: float,
                  marking_options, markings_dir: str, verbose: bool) -> dict[str, Any]:
    import time

    import bpy

    from . import scene as scene_module
    from .build import build_scene

    started = time.time()
    plot = scene.result.plots[building]
    shot = plan_orbit(plot, options)

    build_scene(scene.result, facade_dir=facade_dir, road_texture=road_texture,
                ground_texture=ground_texture, tile_metres=tile_metres,
                road_tile_metres=road_tile_metres, marking_options=marking_options,
                markings_dir=markings_dir, verbose=False)
    if not any(obj.type == "LIGHT" for obj in bpy.data.objects):
        scene_module.sunlit()

    # Which other buildings come out. Only buildings: Ground and Roads are not
    # in this loop, so the street the subject stands on survives every mode.
    plots = scene.result.plots
    if options.neighbours == "hide":
        removed = [i for i in range(len(plots)) if i != building]
    elif options.neighbours == "clear":
        removed = blocking_buildings(plots, building, shot["reach_m"])
    else:
        removed = []

    # Which faces are this building's, in each of the two merged objects it is
    # spread across. Roofs are generated in step with Buildings, so the index
    # is the same in both, and so is everyone else's.
    targets: dict[str, tuple[int, int] | None] = {}
    for name in ("Buildings", "Roofs"):
        counts = scene_module.build.face_counts.get(name)
        obj = bpy.data.objects.get(name)
        if not counts or obj is None:
            continue
        start, end = face_range(counts, building)
        if removed:
            remove_faces(obj, [face_range(counts, i) for i in removed])
            # The deletion renumbered the mesh, so the subject's range moves
            # down by whatever was taken out from in front of it.
            shift = sum(counts[i] for i in removed if i < building)
            start, end = start - shift, end - shift
        targets[name] = (start, end)
    if not targets:
        raise RuntimeError("the built scene has no Buildings object to circle")
    if verbose and removed:
        print(f"[orbit] {options.neighbours}: {len(removed)} of {len(plots) - 1} "
              f"neighbour(s) taken out of the shot")

    os.makedirs(out_dir, exist_ok=True)
    scene_module.animate_camera(shot["path"], lens=options.lens, name="OrbitCam")
    video = scene_module.render_animation(
        os.path.join(out_dir, "orbit.mp4"), frames=options.frames, fps=options.fps,
        resolution=(options.width, options.height), samples=options.samples,
        verbose=verbose)

    paint_mask(targets)
    masks = render_frames(os.path.join(out_dir, "mask"), frames=options.frames,
                          resolution=(options.width, options.height))

    report = {
        "building": building,
        "area_m2": plot.get("area"),
        "floors": plot.get("floors"),
        "neighbours": options.neighbours,
        "neighbours_removed": len(removed),
        "neighbours_standing": len(plots) - 1 - len(removed),
        # What it was wearing, because a clip refined from an undressed render
        # is a different experiment from one refined from a dressed one.
        "dressed": {"facades": bool(facade_dir), "road": bool(road_texture),
                    "ground": bool(ground_texture)},
        "video": video,
        "mask_dir": os.path.join(out_dir, "mask"),
        "mask_frames": len(masks),
        "took_seconds": round(time.time() - started, 1),
        **{k: v for k, v in shot.items() if k != "path"},
    }
    with open(os.path.join(out_dir, "orbit.json"), "w", encoding="utf-8") as handle:
        json.dump({**report, "path": [[list(p), list(t)] for p, t in shot["path"]]},
                  handle, indent=2)
    return report
