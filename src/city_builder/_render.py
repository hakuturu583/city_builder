"""Stills of a built scene, framed from the geometry rather than by hand.

Placing a camera by hand is fine when you can see what you got. A caller that
cannot has to be given a view that is right the first time, so the framing
comes from the scene's own extent and, for the street view, from the route the
drive would take.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import lanelet, route
from . import scene as scene_module


def _extent(result) -> tuple[np.ndarray, float]:
    """Centre and width of the road network, in scene coordinates."""
    points = [
        (p[0], p[1], p[2])
        for name in ("Roads", "Junctions")
        for ribbon in result.groups.get(name, ())
        for p in (*getattr(ribbon, "left", ()), *getattr(ribbon, "right", ()))
    ]
    if not points:
        return np.zeros(3), 100.0
    array = np.asarray(points, dtype=float)
    centre = array.mean(axis=0)
    span = float(max(np.ptp(array[:, 0]), np.ptp(array[:, 1])))
    return centre, max(span, 40.0)


def _street_pose(scene) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    _ll2, _projection, lmap = lanelet.load_map(
        scene.map_path, projector="utm",
        origin_lat=scene.result.frame.ref_lat, origin_lon=scene.result.frame.ref_lon)
    path = route.drive_path(scene.result.groups, lanelet.lanelet_end_keys(lmap), step=1.0)
    if not path:
        return None
    return path[min(len(path) // 8, len(path) - 1)]


def render_still(scene, *, view: str = "aerial", hide_buildings: bool = False,
                 out_path: str, resolution: tuple[int, int] = (900, 600),
                 samples: int = 32, facade_dir: str | None = None,
                 marking_options=None) -> dict[str, Any]:
    """Build the scene into Blender, frame it, and render one image."""
    import shutil
    import tempfile

    # The mask pages are loaded from disk by the material, so they have to be
    # written somewhere — but a render writes no scene file for them to sit
    # beside, and a server has no business leaving them in whatever cwd it
    # started in. They must outlive the render, not just the build.
    markings = tempfile.mkdtemp(prefix="city-markings-")
    try:
        return _render_still(scene, view=view, hide_buildings=hide_buildings,
                             out_path=out_path, resolution=resolution, samples=samples,
                             facade_dir=facade_dir, marking_options=marking_options,
                             markings_dir=markings)
    finally:
        shutil.rmtree(markings, ignore_errors=True)


def _render_still(scene, *, view: str, hide_buildings: bool, out_path: str,
                  resolution: tuple[int, int], samples: int, facade_dir: str | None,
                  marking_options, markings_dir: str) -> dict[str, Any]:
    import bpy
    import mathutils

    from .build import build_scene

    build_scene(scene.result, facade_dir=facade_dir, marking_options=marking_options,
                markings_dir=markings_dir, verbose=False)
    if not any(obj.type == "LIGHT" for obj in bpy.data.objects):
        scene_module.sunlit()

    if hide_buildings:
        for name in ("Buildings", "Roofs"):
            if name in bpy.data.collections:
                for obj in bpy.data.collections[name].objects:
                    obj.hide_render = True

    centre, span = _extent(scene.result)
    if view == "street":
        pose = _street_pose(scene)
        if pose is None:
            view = "aerial"
        else:
            location, target = pose
            lens = 30.0
    if view == "plan":
        location = (centre[0], centre[1] - 0.001, centre[2] + span * 1.4)
        target = tuple(centre)
        lens = 50.0
    elif view == "aerial":
        reach = span * 0.75 + 60.0
        location = (centre[0] - reach * 0.7, centre[1] - reach * 0.7, centre[2] + reach * 0.7)
        target = tuple(centre)
        lens = 35.0

    data = bpy.data.cameras.new("View")
    data.lens = lens
    data.clip_end = max(6000.0, span * 6)
    camera = bpy.data.objects.new("View", data)
    bpy.context.scene.collection.objects.link(camera)
    camera.location = location
    camera.rotation_euler = (
        mathutils.Vector(target) - mathutils.Vector(location)
    ).to_track_quat("-Z", "Y").to_euler()

    blender = bpy.context.scene
    blender.camera = camera
    blender.render.engine = "BLENDER_EEVEE"
    blender.eevee.taa_render_samples = samples
    blender.render.resolution_x, blender.render.resolution_y = resolution
    blender.render.image_settings.file_format = "PNG"
    blender.render.filepath = out_path
    bpy.ops.render.render(write_still=True)
    return {"path": out_path, "view": view,
            "camera": [round(float(v), 1) for v in location],
            "looking_at": [round(float(v), 1) for v in target],
            "span_m": round(span, 1)}
