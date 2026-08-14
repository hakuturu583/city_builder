"""A drivable route through the map, and a camera that follows it.

The point of building this scene is to drive through it, so the useful view of
it is not an aerial render — it is what a car windscreen sees. That needs a
path, and the map already contains one: lanelets are lanes, a lanelet's two
boundaries average to its centreline, and two lanelets are consecutive when one
ends on the pair of boundary points the other starts from.

Nothing here imports bpy. The path is geometry; putting a camera on it is
:func:`city_builder.scene.animate_camera`.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from itertools import pairwise

Point = tuple[float, float, float]


def centreline(ribbon) -> list[Point]:
    """The middle of a lane, from the two surveyed boundaries."""
    return [
        ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0, (a[2] + b[2]) / 2.0)
        for a, b in zip(ribbon.left, ribbon.right)
    ]


def successors(lanelet_ends) -> dict[int, list[int]]:
    """Directed lanelet graph: ``a -> b`` when b starts where a ends.

    :func:`city_builder.lanelet.build_adjacency` gives the undirected version,
    which is what the ground classifier wants — it only asks whether two
    lanelets touch. Driving has a direction, so the end pair has to match the
    *start* pair rather than either end.
    """
    starts: dict[frozenset, list[int]] = {}
    for lanelet_id, start, _ in lanelet_ends:
        starts.setdefault(start, []).append(lanelet_id)

    return {
        lanelet_id: [other for other in starts.get(end, ()) if other != lanelet_id]
        for lanelet_id, _, end in lanelet_ends
    }


def _length(points: Sequence[Point]) -> float:
    return sum(math.dist(a[:2], b[:2]) for a, b in pairwise(points))


def longest_route(succ: dict[int, list[int]], lines: dict[int, list[Point]], *,
                  attempts: int = 400, seed: int = 0) -> list[int]:
    """The longest chain of lanelets a bounded search finds.

    Random walks with restarts rather than an exact longest path: the road graph
    has cycles — every roundabout and every block does — so the exact problem is
    NP-hard, and a few hundred walks over 900 lanelets take milliseconds and
    find a route long enough to drive for a minute. Deterministic for a seed.
    """
    rng = random.Random(seed)
    starts = [i for i in succ if i in lines]
    if not starts:
        return []

    best: list[int] = []
    best_length = 0.0
    for _ in range(attempts):
        current = rng.choice(starts)
        walk, seen = [current], {current}
        while True:
            options = [n for n in succ.get(walk[-1], ()) if n not in seen and n in lines]
            if not options:
                break
            step = rng.choice(options)
            walk.append(step)
            seen.add(step)

        length = sum(_length(lines[i]) for i in walk)
        if length > best_length:
            best, best_length = walk, length
    return best


def route_polyline(route: Sequence[int], lines: dict[int, list[Point]]) -> list[Point]:
    """One polyline for the whole route, without the duplicated joins."""
    points: list[Point] = []
    for lanelet_id in route:
        segment = lines[lanelet_id]
        if points and segment and math.dist(points[-1][:2], segment[0][:2]) < 0.5:
            segment = segment[1:]
        points.extend(segment)
    return points


def resample(points: Sequence[Point], step: float) -> list[Point]:
    """Even spacing along the path, so speed is constant rather than per-vertex.

    A lanelet's vertices sit wherever the survey put them — dense on a curve,
    sparse on a straight — so animating one vertex per frame would race down
    the straights and crawl round the bends.
    """
    if len(points) < 2:
        return list(points)

    out = [points[0]]
    carry = 0.0
    for a, b in pairwise(points):
        span = math.dist(a[:2], b[:2])
        if span < 1e-9:
            continue
        travelled = carry
        while travelled + step <= span:
            travelled += step
            t = travelled / span
            out.append((a[0] + (b[0] - a[0]) * t,
                        a[1] + (b[1] - a[1]) * t,
                        a[2] + (b[2] - a[2]) * t))
        carry = travelled - span
    return out


def smooth(points: Sequence[Point], window: int) -> list[Point]:
    """A moving average, so the camera does not twitch at every survey vertex."""
    if window < 2 or len(points) < window:
        return list(points)
    half = window // 2
    out = []
    for i in range(len(points)):
        low, high = max(0, i - half), min(len(points), i + half + 1)
        chunk = points[low:high]
        out.append(tuple(sum(p[axis] for p in chunk) / len(chunk) for axis in range(3)))
    return out


def camera_path(points: Sequence[Point], *, eye_height: float = 1.4,
                look_ahead: float = 18.0, step: float = 0.4,
                smoothing: int = 9) -> list[tuple[Point, Point]]:
    """``(position, target)`` per frame, for a camera driving the route.

    The target is a point some way further along rather than the next sample:
    aiming at the next sample makes the camera yaw with every wobble in the
    centreline, while aiming well ahead is what a driver actually does and
    turns the same wobble into a steady approach.
    """
    path = smooth(resample(points, step), smoothing)
    if len(path) < 2:
        return []

    lead = max(1, round(look_ahead / step))
    frames = []
    for i in range(len(path) - 1):
        here = path[i]
        ahead = path[min(i + lead, len(path) - 1)]
        if here[:2] == ahead[:2]:
            continue
        frames.append((
            (here[0], here[1], here[2] + eye_height),
            (ahead[0], ahead[1], ahead[2] + eye_height * 0.8),
        ))
    return frames


def drive_path(groups: dict[str, list], lanelet_ends, *, seed: int = 0,
               **kwargs) -> list[tuple[Point, Point]]:
    """Route and camera path straight from a build result's groups.

    Junctions count as road: a route that refused to cross an intersection
    would run to the first corner and stop.
    """
    lines = {}
    for name in ("Roads", "Junctions"):
        for ribbon in groups.get(name, ()):
            line = centreline(ribbon)
            if len(line) >= 2:
                lines[ribbon.id] = line

    route = longest_route(successors(lanelet_ends), lines, seed=seed)
    if not route:
        return []
    return camera_path(route_polyline(route, lines), **kwargs)
