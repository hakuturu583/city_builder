"""Putting a generated object back where the plot says it goes.

An image-to-3D model answers in a box of its own: unit-ish, centred on nothing
in particular, facing whichever way the picture faced. None of that is a defect
— it never saw a metre — but it does mean the mesh cannot be dropped into a
scene meant for testing a driving stack, where the plot boundary is a
specification and the road beside it is surveyed.

Everything needed to place it is already known before the model runs. The plot
has a footprint, and the footprint has a long side, a short side and a heading;
the plot has a storey count, and storeys have a height. So the fit is not an
estimation problem at all. It is a similarity transform read straight off the
plan, with one honest decision in it: whether to keep the generated
proportions or to force the plot's.

Forcing them is the default. A building that is 3 % wider than its plot is a
building that overhangs a pavement, and the pavement is the part that has to be
right.

**This is not the placement the pipeline runs.** That is
:func:`~city_builder.reconstruct.fit_to_footprint`, which *solves* the yaw over
the footprint's own ring rather than reading a heading off its rectangle, seats
the mesh with :func:`~city_builder.reconstruct.seat_z` instead of standing it on
its lowest vertex, and reports a footprint IoU. Where the two disagree the
reconstruct path is the one to believe. What is here is the analytic answer —
the transform read straight off the plan, with no search in it — which is worth
having as the thing a search can be checked against, and cheap enough to run on
a mesh nobody wants to reconstruct.

:func:`overhang` has no counterpart over there and is the part of this module
with nothing above it: an IoU conflates a building that spills over its
boundary with one that fails to fill it, and only one of those puts a wall in
the road.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass
class Placement:
    """Where a generated mesh has to go, and how much it had to be stretched."""

    scale: tuple[float, float, float]
    yaw: float               # radians, applied before translation
    translation: tuple[float, float, float]
    distortion: float        # how far from uniform the scaling had to be

    def to_dict(self) -> dict[str, Any]:
        return {"scale": [round(s, 4) for s in self.scale],
                "yaw_deg": round(math.degrees(self.yaw), 2),
                "translation": [round(t, 3) for t in self.translation],
                "distortion": round(self.distortion, 3)}


def footprint_frame(polygon) -> tuple[tuple[float, float], float, float, float]:
    """Centre, long side, short side and heading of a footprint's best rectangle.

    The minimum-area rectangle rather than the bounding box: a building at forty
    degrees to north has a bounding box half again its size, and fitting to that
    would put the wall across the boundary.

    Carried here rather than imported. The same function lives on ``main`` in
    ``elevations.py``, a module this branch does not have and should not grow —
    the rest of that module is camera placement, which is ``portrait.py``'s job
    here. A dozen lines duplicated is cheaper than two modules for photographs,
    and when the branches meet it is obvious which copy to drop.
    """
    import numpy as np

    rectangle = polygon.minimum_rotated_rectangle
    corners = np.asarray(rectangle.exterior.coords)[:4]
    edges = [(corners[(i + 1) % 4] - corners[i]) for i in range(4)]
    lengths = [float(np.linalg.norm(edge)) for edge in edges]
    longest = int(np.argmax(lengths[:2]))  # opposite sides are equal
    long_side = lengths[longest]
    short_side = lengths[1 - longest]
    heading = math.atan2(edges[longest][1], edges[longest][0])
    centre = corners.mean(axis=0)
    return (float(centre[0]), float(centre[1])), long_side, short_side, heading


def bounds(vertices) -> tuple[Any, Any]:
    import numpy as np

    points = np.asarray(vertices, dtype=float)
    # Percentiles, not extremes: a marching-cubes surface usually carries a few
    # stray vertices, and one of them fixing the scale would shrink the whole
    # building.
    return np.percentile(points, 0.5, axis=0), np.percentile(points, 99.5, axis=0)


def fit_to_plot(vertices, polygon, base_z: float, height: float, *,
                uniform: bool = False, up: str = "y") -> Placement:
    """The transform taking a generated mesh onto a plot's footprint.

    ``polygon`` is the shapely footprint the plot was extruded from; its
    minimum-area rectangle gives the heading to face and the two spans to fill.
    ``up`` names the generated mesh's up axis — Hunyuan3D answers in a
    y-up frame, Blender and this package are z-up.
    """
    import numpy as np

    low, high = bounds(vertices)
    span = np.maximum(high - low, 1e-6)

    order = {"y": (0, 2, 1), "z": (0, 1, 2)}[up]
    width, depth, tall = span[order[0]], span[order[1]], span[order[2]]

    (cx, cy), long_side, short_side, heading = footprint_frame(polygon)

    # The generated object's own long side in plan decides which way round it
    # goes: the model was shown a building, not a compass.
    if width >= depth:
        plan_long, plan_short, extra_yaw = width, depth, 0.0
    else:
        plan_long, plan_short, extra_yaw = depth, width, math.pi / 2

    scale_long = long_side / plan_long
    scale_short = short_side / plan_short
    scale_up = height / tall
    if uniform:
        one = float(np.mean([scale_long, scale_short, scale_up]))
        scale_long = scale_short = scale_up = one

    factors = [scale_long, scale_short, scale_up]
    distortion = max(factors) / max(min(factors), 1e-9) - 1.0

    return Placement(
        scale=(round(float(scale_long), 6), round(float(scale_short), 6),
               round(float(scale_up), 6)),
        yaw=heading + extra_yaw,
        # The mesh is centred in plan and stood on the pavement, not centred in
        # height: a building whose base floats is worse than one whose roof is
        # a little low, because the shadow gives it away immediately.
        translation=(float(cx), float(cy), float(base_z)),
        distortion=float(distortion),
    )


def apply(vertices, placement: Placement, *, up: str = "y"):
    """The transform, as arithmetic, for tests and for meshes outside Blender."""
    import numpy as np

    points = np.asarray(vertices, dtype=float)
    low, high = bounds(points)
    centre = (high + low) / 2.0

    if up == "y":
        # y-up, right-handed, looking down -z: `(x, y, z) -> (x, -z, y)` is the
        # rotation into this package's z-up. The tempting `(x, z, y)` is a
        # *reflection* — a building whose signage reads backwards, and whose fit
        # scores just as well, so nothing downstream catches it. Same convention
        # and same reason as `reconstruct.to_scene_axes`.
        local = np.stack([points[:, 0] - centre[0],
                          -(points[:, 2] - centre[2]),
                          points[:, 1] - low[1]], axis=1)
    else:
        local = np.stack([points[:, 0] - centre[0],
                          points[:, 1] - centre[1],
                          points[:, 2] - low[2]], axis=1)

    local *= np.asarray(placement.scale, dtype=float)
    angle = placement.yaw
    rotation = np.array([[math.cos(angle), -math.sin(angle), 0.0],
                         [math.sin(angle), math.cos(angle), 0.0],
                         [0.0, 0.0, 1.0]])
    return local @ rotation.T + np.asarray(placement.translation, dtype=float)


def overhang(vertices, polygon) -> dict[str, Any]:
    """How far the placed mesh leaves its plot, in metres.

    The number that decides whether a generated building can be used at all:
    the plot boundary is where the pavement starts.
    """
    import numpy as np
    from shapely.geometry import MultiPoint, Point

    points = np.asarray(vertices, dtype=float)[:, :2]
    hull = MultiPoint([tuple(p) for p in points]).convex_hull
    outside = hull.difference(polygon)
    strays = [Point(p) for p in points if not polygon.covers(Point(p))]
    worst = max((polygon.distance(p) for p in strays), default=0.0)
    return {"outside_m2": round(float(outside.area), 3),
            "worst_overhang_m": round(float(worst), 3),
            "vertices_outside": len(strays),
            "plot_m2": round(float(polygon.area), 2)}
