"""Running the roads off the edge of the map.

A Lanelet2 map is a rectangle cut out of a city, so its roads stop at an
arbitrary line with nothing beyond them. Everything downstream reads that
stopping point as ordinary ground: the terrain is interpolated across it, and
the building generator — which fills whatever the roads leave — puts a block
squarely across the end of the street. The result is a city with a wall around
it, and a drive that ends by heading into a wall.

So the dangling ends are run out to the edge instead. An end is dangling when
no lanelet starts where it finishes, which is the successor relation read off
the shared boundary point ids rather than through a routing graph. Each one is
continued straight, at the width and grade it had, until it either reaches the
map edge or comes within `clearance` of another lanelet — a stub pointing into
the side of a road that is already there gets no extension at all, which is the
same rule as "do not interfere with other lanelets", stated as geometry.

The extension is applied to the **boundary polylines**, keyed by the point id
they end at, not to the finished lane surfaces. That is what makes it uniform:
a lanelet bound, the painted line drawn from that same linestring, and the kerb
beside it are one polyline as far as the map is concerned, so lengthening it
once gives a road whose surface, lane markings and kerb all run off the edge
together. Widening, dashing and pairing then happen downstream as usual, so the
new dashes keep the phase of the old ones.

The map edge has to be decided *before* anything is extended, or it runs away:
the edge is a margin around the roads, and the roads are what is being moved.
So :func:`plan` fixes the box from the surveyed geometry and hands it back for
the ground to be built to, instead of the ground taking its own margin around
the extended roads and leaving a fresh ring of nothing beyond them.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

Point = tuple[float, float, float]


@dataclass
class ExtendOptions:
    """How far the roads run past the last thing the survey saw."""

    enabled: bool = True
    margin: float = 30.0  # how far beyond the surveyed roads the map edge sits
    clearance: float = 1.0  # keep an extension this far from any other lanelet
    min_length: float = 2.0  # shorter than this is not worth the geometry
    # A loose end only counts as cut off by the map if it leaves the surveyed
    # area within this distance. A road that stops in the middle of the city is
    # a dead end the survey meant, not an edge, and running it to the far
    # corner would draw a lane-wide scratch across half a kilometre of blocks.
    cut_off_within: float = 60.0
    step: float = 5.0  # sample spacing along the extension, for the ground grid
    slope_window: float = 15.0  # metres of lanelet used to measure the outgoing grade
    max_grade: float = 0.06  # cap on that grade, so a noisy end cannot dive or soar

    def __post_init__(self) -> None:
        for name in ("margin", "clearance", "min_length", "cut_off_within", "step",
                     "slope_window"):
            if getattr(self, name) < 0:
                raise ValueError(f"extend.{name} must not be negative")
        if self.step <= 0:
            raise ValueError("extend.step must be positive")


@dataclass
class Stub:
    """One dangling lanelet end, in scene coordinates.

    ``direction`` points out of the map and ``grade`` is the rise per metre the
    lanelet was on when it stopped.
    """

    lanelet_id: int
    left: Point
    right: Point
    direction: tuple[float, float]
    grade: float

    def outer(self, distance: float) -> tuple[Point, Point]:
        """The left and right bound points ``distance`` along the extension."""
        dx, dy = self.direction
        rise = self.grade * distance
        return (
            (self.left[0] + dx * distance, self.left[1] + dy * distance, self.left[2] + rise),
            (self.right[0] + dx * distance, self.right[1] + dy * distance, self.right[2] + rise),
        )

    def sweep(self, distance: float):
        """The plan-view quad the extension would cover."""
        from shapely.geometry import Polygon as ShapelyPolygon

        far_left, far_right = self.outer(distance)
        return ShapelyPolygon([self.left[:2], far_left[:2], far_right[:2], self.right[:2]])


# ---------------------------------------------------------------------------
# Which ends are loose
# ---------------------------------------------------------------------------


def dangling_ends(lanelet_ends: Sequence[tuple[int, Any, Any]]) -> set[tuple[int, str]]:
    """``(lanelet id, "start"|"end")`` for every end with nothing attached.

    Directional on purpose. Sharing an endpoint pair is not connectivity: two
    lanelets that *both start* at the same pair are a fork, and if nothing ends
    there the fork is as loose as a single lane would be. So a lanelet's end is
    attached only when some lanelet starts on it, and vice versa.
    """
    starts = {start for _id, start, _end in lanelet_ends}
    ends = {end for _id, _start, end in lanelet_ends}
    loose = set()
    for lanelet_id, start, end in lanelet_ends:
        if end not in starts:
            loose.add((lanelet_id, "end"))
        if start not in ends:
            loose.add((lanelet_id, "start"))
    return loose


def edge_box(points: Sequence[Sequence[float]], margin: float) -> tuple[float, float, float, float]:
    """The map edge: a margin around everything the survey covered."""
    arr = np.asarray(points, dtype=float)
    if len(arr) == 0:
        raise ValueError("no geometry to take an extent from")
    return (
        float(arr[:, 0].min() - margin), float(arr[:, 1].min() - margin),
        float(arr[:, 0].max() + margin), float(arr[:, 1].max() + margin),
    )


# ---------------------------------------------------------------------------
# How far a stub may go
# ---------------------------------------------------------------------------


def _heading(bound: Sequence[Point], window: float) -> tuple[np.ndarray, float] | None:
    """Direction and grade at the end of a polyline, over the last ``window`` m.

    Measured over a window rather than the final segment: the last segment of a
    surveyed boundary is often a few centimetres long, and a direction taken
    from it is noise pointed at the horizon.
    """
    arr = np.asarray(bound, dtype=float)
    if len(arr) < 2:
        return None

    steps = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    back = np.concatenate([[0.0], np.cumsum(steps[::-1])])
    index = int(np.searchsorted(back, max(window, 1e-6)))
    index = min(index, len(arr) - 1)
    start = arr[len(arr) - 1 - index]

    delta = arr[-1] - start
    run = float(np.hypot(delta[0], delta[1]))
    if run < 1e-6:
        return None
    return delta[:2] / run, float(delta[2] / run)


def stub_from_bounds(lanelet_id: int, left: Sequence[Point], right: Sequence[Point],
                     options: ExtendOptions) -> Stub | None:
    """A stub at the far end of a lanelet's two bounds, as given.

    Reverse both bounds to get the stub at the near end — which is why this
    takes the bounds rather than a lanelet.
    """
    headings = [h for h in (_heading(left, options.slope_window),
                            _heading(right, options.slope_window)) if h is not None]
    if not headings:
        return None

    # Average the two bounds. One of them can be a stub of two points with a
    # direction of its own; the pair is what the lane is actually doing.
    direction = np.mean([h[0] for h in headings], axis=0)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        direction = headings[0][0]
        norm = 1.0
    direction = direction / norm

    grade = float(np.clip(np.mean([h[1] for h in headings]), -options.max_grade, options.max_grade))
    return Stub(lanelet_id, tuple(left[-1]), tuple(right[-1]),
                (float(direction[0]), float(direction[1])), grade)


def to_edge(stub: Stub, box: Sequence[float]) -> float:
    """Distance along the stub before its centre leaves the map."""
    x = (stub.left[0] + stub.right[0]) / 2.0
    y = (stub.left[1] + stub.right[1]) / 2.0
    limit = math.inf
    for position, component, low, high in (
        (x, stub.direction[0], box[0], box[2]),
        (y, stub.direction[1], box[1], box[3]),
    ):
        if component > 1e-9:
            limit = min(limit, (high - position) / component)
        elif component < -1e-9:
            limit = min(limit, (low - position) / component)
    return max(0.0, limit if math.isfinite(limit) else 0.0)


def blocked_at(stub: Stub, distance: float, blockers, *, area_tolerance: float = 0.05) -> float:
    """How far the stub gets before it runs into something already there.

    ``blockers`` is anything with a shapely ``query``: the footprints of every
    other lanelet, plus the extensions granted so far.

    Contact along the side does not count. A lane that ends beside its
    neighbour touches it along a line of zero area, and refusing to extend
    there would rule out exactly the multi-lane roads that most need it — so
    only an overlap with real area stops anything.

    Returns infinity when the way is clear, so that a stub reaching the map
    edge is not pulled back by a clearance it owes nobody.
    """
    if distance <= 0:
        return 0.0
    sweep = stub.sweep(distance)
    origin = np.array([(stub.left[0] + stub.right[0]) / 2.0,
                       (stub.left[1] + stub.right[1]) / 2.0])
    axis = np.array(stub.direction)

    nearest = math.inf
    for other in _candidates(blockers, sweep):
        overlap = sweep.intersection(other)
        if overlap.is_empty or overlap.area <= area_tolerance:
            continue
        for part in getattr(overlap, "geoms", [overlap]):
            if part.geom_type != "Polygon":
                continue
            coords = np.asarray(part.exterior.coords, dtype=float)[:, :2]
            nearest = min(nearest, float(((coords - origin) @ axis).min()))
    return max(0.0, nearest)


def _candidates(blockers, sweep):
    if blockers is None:
        return []
    if hasattr(blockers, "query"):
        found = blockers.query(sweep)
        if len(found) and isinstance(found[0], (int, np.integer)):
            return [blockers.geometries[i] for i in found]
        return list(found)
    return list(blockers)


def leaves_within(stub: Stub, covered, distance: float) -> bool:
    """Does the stub get out of the surveyed area inside ``distance``?

    This is what separates a road the map cut from a road that ends. Both are
    loose ends in the graph; only one of them has open country in front of it.
    Measured against the road network's own outline rather than against the
    rectangle, because the rectangle is the bounding box of an irregular
    district: a dead end in the middle of it can be five hundred metres from
    the box and still be nowhere near the edge of the city.
    """
    if covered is None:
        return True
    from shapely.geometry import LineString, Point

    origin = ((stub.left[0] + stub.right[0]) / 2.0, (stub.left[1] + stub.right[1]) / 2.0)
    if not covered.contains(Point(origin)):
        return True
    far = (origin[0] + stub.direction[0] * distance, origin[1] + stub.direction[1] * distance)
    return not covered.contains(LineString([origin, far]))


def reach(stub: Stub, box: Sequence[float], blockers, options: ExtendOptions) -> float:
    """How long this stub's extension should be, in metres."""
    limit = to_edge(stub, box)
    if limit < options.min_length:
        return 0.0
    obstacle = blocked_at(stub, limit, blockers)
    length = limit if math.isinf(obstacle) else min(limit, obstacle - options.clearance)
    return length if length >= options.min_length else 0.0


def bound_points(stub: Stub, length: float, options: ExtendOptions) -> tuple[list[Point], list[Point]]:
    """The points to append to the left and right bounds, nearest first.

    Sampled along the way rather than one point at the end: the terrain grid
    bins road samples, and a single cross-section thirty metres out leaves
    every cell between them unsupported.
    """
    if length <= 0:
        return [], []
    count = max(1, math.ceil(length / options.step))
    distances = [length * (i + 1) / count for i in range(count)]
    outer = [stub.outer(d) for d in distances]
    return [o[0] for o in outer], [o[1] for o in outer]


# ---------------------------------------------------------------------------
# The whole map
# ---------------------------------------------------------------------------


@dataclass
class Plan:
    """What to lengthen, and the edge everything else should be built to."""

    box: tuple[float, float, float, float]
    # Boundary point id -> the points to continue that polyline with, ordered
    # away from it. Keyed by point rather than by linestring so a consumer can
    # apply it to a bound running either way without knowing which.
    points: dict[int, list[Point]]
    stats: dict[str, Any]

    def extended(self, polyline: Sequence[Point], first_id: int, last_id: int) -> list[Point]:
        """``polyline`` with whatever continues it at either end attached."""
        head = self.points.get(first_id)
        tail = self.points.get(last_id)
        if not head and not tail:
            return list(polyline)
        return [*reversed(head or []), *polyline, *(tail or [])]


def plan(bounds: dict[int, tuple[list[Point], list[Point]]],
         ends: Sequence[tuple[int, Any, Any]],
         end_point_ids: dict[tuple[int, str], tuple[int, int]],
         options: ExtendOptions | None = None) -> Plan:
    """Work out every extension, in scene coordinates.

    ``bounds`` maps a lanelet id to its left and right boundary polylines,
    ``ends`` is :func:`city_builder.lanelet.lanelet_end_keys`, and
    ``end_point_ids`` maps each ``(lanelet id, "start"|"end")`` to the pair of
    boundary point ids there. Nothing here touches lanelet2, so it can be
    tested on invented geometry.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.strtree import STRtree

    options = options or ExtendOptions()
    every_point = [p for left, right in bounds.values() for p in (*left, *right)]
    box = edge_box(every_point, options.margin)
    if not options.enabled:
        return Plan(box, {}, {"extended": 0, "reason": "disabled"})

    footprints: dict[int, Any] = {}
    for lanelet_id, (left, right) in bounds.items():
        if len(left) < 2 or len(right) < 2:
            continue
        polygon = ShapelyPolygon([p[:2] for p in (*left, *reversed(right))])
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 1e-6:
            footprints[lanelet_id] = polygon

    from shapely.ops import unary_union

    covered = (unary_union(list(footprints.values())).convex_hull.buffer(options.margin)
               if footprints else None)

    loose = dangling_ends(ends)
    # A point that some *attached* end also sits on must not move: the lanelet
    # continuing through it would be dragged off its successor.
    pinned = {
        point
        for key, pair in end_point_ids.items()
        if key not in loose
        for point in pair
    }

    granted: dict[int, tuple[float, list[Point]]] = {}
    lengths: list[float] = []
    extra: list[Any] = []
    inland = 0
    for lanelet_id, side in sorted(loose):
        pair = bounds.get(lanelet_id)
        if pair is None:
            continue
        left, right = pair
        if side == "start":
            left, right = list(reversed(left)), list(reversed(right))

        stub = stub_from_bounds(lanelet_id, left, right, options)
        if stub is None:
            continue
        if not leaves_within(stub, covered, options.cut_off_within):
            inland += 1
            continue

        others = [f for other_id, f in footprints.items() if other_id != lanelet_id]
        length = reach(stub, box, STRtree(others + extra) if others or extra else None, options)
        if length <= 0:
            continue

        ids = end_point_ids.get((lanelet_id, side))
        if ids is None or any(point in pinned for point in ids):
            continue

        left_points, right_points = bound_points(stub, length, options)
        for point_id, continuation in zip(ids, (left_points, right_points)):
            # Two loose ends can share a boundary point. The shorter claim wins:
            # one of them was stopped by something, and overshooting it is the
            # failure this whole pass exists to avoid.
            held = granted.get(point_id)
            if held is None or length < held[0]:
                granted[point_id] = (length, continuation)
        extra.append(stub.sweep(length))
        lengths.append(length)

    return Plan(box, {k: v[1] for k, v in granted.items()}, {
        "dangling_ends": len(loose),
        "extended": len(lengths),
        "inland_dead_ends": inland,
        "length_m": {
            "min": round(min(lengths), 1), "median": round(float(np.median(lengths)), 1),
            "max": round(max(lengths), 1), "total": round(float(np.sum(lengths)), 1),
        } if lengths else {},
    })


def from_map(ll2, projector, lmap, frame, options: ExtendOptions | None = None,
             *, converter=None) -> Plan:
    """The plan for a loaded map, in the scene frame."""
    from . import lanelet as ll

    convert = converter or ll.PointConverter(ll2, projector, frame)
    bounds: dict[int, tuple[list[Point], list[Point]]] = {}
    for lanelet in lmap.laneletLayer:
        if ll.attributes(lanelet).get("subtype") not in ll.ROAD_SUBTYPES:
            continue
        left, right = lanelet.leftBound, lanelet.rightBound
        if len(left) < 2 or len(right) < 2:
            continue
        bounds[lanelet.id] = (convert.polyline(left), convert.polyline(right))

    return plan(bounds, ll.lanelet_end_keys(lmap), ll.lanelet_end_points(lmap), options)
