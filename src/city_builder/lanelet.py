"""Loading a Lanelet2 map and getting its geometry into the scene frame."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .frame import LocalFrame

ROAD_SUBTYPES = ("road", "road_shoulder", "highway")


def _import_lanelet2():
    try:
        import lanelet2
    except ImportError as e:  # pragma: no cover - dependency guard
        raise ImportError("lanelet2 bindings missing; install city-builder's dependencies") from e

    # Registers the Autoware-only regulatory elements (road_marking,
    # detection_area, ...). Without it, load() rejects every Autoware map.
    try:
        import autoware_lanelet2_extension_python.regulatory_elements  # noqa: F401
    except ImportError:  # pragma: no cover - optional
        print("[lanelet] autoware extension unavailable; Autoware maps may fail to load")

    import lanelet2

    return lanelet2


def make_projector(ll2, kind: str, origin_lat: float, origin_lon: float):
    origin = ll2.io.Origin(origin_lat, origin_lon)
    kinds = {
        "utm": ll2.projection.UtmProjector,
        "mercator": ll2.projection.MercatorProjector,
        "local-cartesian": ll2.projection.LocalCartesianProjector,
    }
    if kind not in kinds:
        raise ValueError(f"unknown projector {kind!r}; choose from {sorted(kinds)}")
    return kinds[kind](origin)


def load_map(path: str, *, projector: str = "utm", origin_lat: float = 0.0, origin_lon: float = 0.0):
    """Load a Lanelet2 ``.osm``, tolerating unknown regulatory elements."""
    ll2 = _import_lanelet2()
    proj = make_projector(ll2, projector, origin_lat, origin_lon)
    try:
        return ll2, proj, ll2.io.load(path, proj)
    except RuntimeError as e:
        print(f"[lanelet] strict load failed ({str(e).splitlines()[0]}); retrying with loadRobust")
        lmap, errors = ll2.io.loadRobust(path, proj)
        print(f"[lanelet] loaded with {len(errors)} parse error(s) ignored")
        return ll2, proj, lmap


class PointConverter:
    """Projected lanelet2 point → local-metre scene coordinates.

    Boundaries are shared between adjacent lanelets, so the same point id shows
    up many times; caching by id turns ~100k reverse projections into ~20k.
    """

    def __init__(self, ll2, projector, frame: LocalFrame):
        self._basic_point = ll2.core.BasicPoint3d
        self._projector = projector
        self._frame = frame
        self._cache: dict[int, tuple[float, float, float]] = {}

    def __call__(self, point) -> tuple[float, float, float]:
        cached = self._cache.get(point.id)
        if cached is not None:
            return cached
        gps = self._projector.reverse(self._basic_point(point.x, point.y, point.z))
        x, y = self._frame.to_local(gps.lat, gps.lon)
        result = (x, y, gps.ele)
        self._cache[point.id] = result
        return result

    def polyline(self, linestring) -> list[tuple[float, float, float]]:
        return [self(p) for p in linestring]


def map_centroid(ll2, projector, lmap) -> tuple[float, float]:
    """Geographic centre of the map, used when no anchor is given."""
    xs, ys = [], []
    for ls in lmap.lineStringLayer:
        for p in ls:
            xs.append(p.x)
            ys.append(p.y)
    if not xs:
        raise ValueError("map contains no geometry")
    centre = ll2.core.BasicPoint3d((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2, 0.0)
    gps = projector.reverse(centre)
    return gps.lat, gps.lon


def attributes(primitive) -> dict[str, str]:
    return {str(k): str(v) for k, v in dict(primitive.attributes).items()}


def lanelet_end_keys(lmap, subtypes: Sequence[str] = ROAD_SUBTYPES):
    """(id, start-pair, end-pair) of boundary point ids for each road lanelet."""
    ends = []
    for lanelet in lmap.laneletLayer:
        if attributes(lanelet).get("subtype") not in subtypes:
            continue
        left, right = lanelet.leftBound, lanelet.rightBound
        if len(left) < 2 or len(right) < 2:
            continue
        ends.append((
            lanelet.id,
            frozenset((left[0].id, right[0].id)),
            frozenset((left[-1].id, right[-1].id)),
        ))
    return ends


def build_adjacency(lanelet_ends) -> dict[int, set[int]]:
    """Lanelet graph from shared boundary endpoints.

    Two lanelets are connected when one's start pair of boundary points is the
    other's end pair — Lanelet2's successor relation read off the geometry,
    rather than through a routing graph that would need traffic rules.
    """
    buckets: dict[frozenset, list[int]] = {}
    for lanelet_id, start, end in lanelet_ends:
        for key in (start, end):
            buckets.setdefault(key, []).append(lanelet_id)

    adjacency: dict[int, set[int]] = {}
    for members in buckets.values():
        for a in members:
            for b in members:
                if a != b:
                    adjacency.setdefault(a, set()).add(b)
    return adjacency


def apply_z_datum(groups: dict[str, list], datum: float | None, z_offset: float) -> float:
    """Shift every surface so the scene sits near ``z_offset``.

    Lanelet2 elevations are absolute (≈29-46 m in Shinjuku) while a scene
    usually wants its ground near zero. Relative heights are preserved, so
    multi-level roads stay stacked.
    """
    if datum is None:
        zs = [z for shapes in groups.values() for s in shapes for z in s.z_values()]
        datum = min(zs) if zs else 0.0

    shift = z_offset - datum
    if shift:
        for shapes in groups.values():
            for shape in shapes:
                shape.shift_z(shift)
    return datum


def collect_points(shapes) -> np.ndarray:
    return np.array([p for s in shapes for p in (*s.left, *s.right)], dtype=float)
