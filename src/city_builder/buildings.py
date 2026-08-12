"""Procedural buildings in the space the roads leave behind.

A Lanelet2 map describes the carriageway and nothing else, so the blocks
between the streets are empty. This fills them: the buildable area is the
ground minus the road outline minus a setback, each block is cut into lots, and
each lot gets an extruded footprint standing on the reconstructed ground.

The layout is deliberately simple — recursive splitting along the long axis,
which is how a block of terraced plots actually reads from the air — and
deterministic given a seed. It is scaffolding for a texturing pass, not an
attempt to guess what is really there: nothing in the map says where the
buildings are, so the only honest claim is "something building-shaped, off the
road, on the ground".
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.errors import GEOSException

from .geometry import Mesh, signed_area_xy
from .ground import HeightMap


@dataclass
class BuildingOptions:
    """Everything that shapes the generated blocks."""

    setback: float = 3.0  # gap between the kerb line and any wall
    target_lot_area: float = 900.0  # split until a lot is about this big
    min_lot_area: float = 120.0  # anything smaller is left as open ground
    lot_margin: float = 1.5  # gap between neighbouring buildings
    split_jitter: float = 0.15  # 0 = split at the middle, 0.5 = anywhere

    min_height: float = 6.0
    max_height: float = 45.0
    floor_height: float = 3.5  # heights snap to whole floors
    tall_bias: float = 0.35  # 0 = every block low, 1 = every block tall

    skirt: float = 1.0  # how far the walls run below the ground
    max_buildings: int = 0  # 0 = unlimited
    seed: int = 0


# ---------------------------------------------------------------------------
# Where a building may stand
# ---------------------------------------------------------------------------


def exclusion_zone(groups: dict[str, Sequence[Any]]) -> Any:
    """Everything a building must not stand on.

    Every paved surface, *including the lanelets classified as elevated*. The
    ground layer excludes those on purpose — a viaduct wants ground underneath
    it — but a building does not: the deck's approach ramp is on the street,
    and building over it put walls on the carriageway (measured: 3.7 % of wall
    vertices before this).
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    polygons = []
    for name in ("Roads", "Junctions", "Crosswalks", "Walkways"):
        for shape in groups.get(name, ()):
            ring = getattr(shape, "ring", None)
            if ring is None:
                continue
            poly = ShapelyPolygon([(p[0], p[1]) for p in ring()])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.area > 1e-6:
                polygons.append(poly)
    return unary_union(polygons) if polygons else None


def buildable_area(road_union, bounds, *, setback: float):
    """The ground, minus the roads and a setback, as block polygons.

    Interior rings are kept. A road that runs through the middle of a region
    shows up as a hole, and taking only the exterior would hand that road back
    as buildable — measured, that put 449 of 2481 plots on the carriageway.
    """
    from shapely.geometry import box

    extent = box(*bounds)
    free = extent.difference(road_union.buffer(setback)) if road_union is not None else extent
    parts = list(free.geoms) if hasattr(free, "geoms") else [free]
    return [
        part for part in parts
        if part.geom_type == "Polygon" and not part.is_empty and part.area > 1.0
    ]


def split_lots(block, options: BuildingOptions, rng: random.Random) -> list:
    """Cut a block into lots by repeatedly halving its long axis."""
    from shapely.geometry import LineString
    from shapely.ops import split as shapely_split

    if block.area <= options.target_lot_area:
        return [block]

    rectangle = block.minimum_rotated_rectangle
    corners = list(rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        return [block]

    edges = [(np.array(corners[i]), np.array(corners[(i + 1) % 4])) for i in range(4)]
    (a, b) = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return [block]
    axis = axis / length

    # Cut across the long axis, a little off-centre so the lots vary.
    offset = 0.5 + rng.uniform(-options.split_jitter, options.split_jitter)
    centre = a + axis * (length * offset)
    normal = np.array([-axis[1], axis[0]])
    reach = float(np.hypot(*(np.array(block.bounds[2:]) - np.array(block.bounds[:2])))) + 10.0
    knife = LineString([centre - normal * reach, centre + normal * reach])

    try:
        pieces = list(shapely_split(block, knife).geoms)
    except (GEOSException, ValueError):  # pragma: no cover - degenerate cut
        return [block]
    if len(pieces) < 2:
        return [block]

    lots = []
    for piece in pieces:
        if piece.geom_type != "Polygon" or piece.area < options.min_lot_area:
            continue
        lots.extend(split_lots(piece, options, rng))
    return lots or [block]


def footprints(
    road_union,
    bounds,
    options: BuildingOptions | None = None,
) -> list:
    """Building footprints for the whole map, as shapely polygons."""
    options = options or BuildingOptions()
    rng = random.Random(options.seed)

    plots = []
    for block in buildable_area(road_union, bounds, setback=options.setback):
        for lot in split_lots(block, options, rng):
            if lot.area < options.min_lot_area:
                continue
            plot = lot.buffer(-options.lot_margin)
            parts = list(plot.geoms) if hasattr(plot, "geoms") else [plot]
            for part in parts:
                if part.geom_type == "Polygon" and part.area >= options.min_lot_area * 0.5:
                    plots.append(part)

    if options.max_buildings and len(plots) > options.max_buildings:
        plots.sort(key=lambda p: p.area, reverse=True)
        plots = plots[: options.max_buildings]
    return plots


def pick_height(area: float, options: BuildingOptions, rng: random.Random) -> float:
    """A plausible height for a plot, snapped to whole floors.

    Bigger plots lean taller — a tower needs a footprint — but the draw stays
    random so a street does not come out monotonic.
    """
    span = options.max_height - options.min_height
    footprint_pull = min(1.0, area / 2500.0)
    draw = rng.random() ** (1.0 - options.tall_bias * footprint_pull)
    height = options.min_height + span * draw * (0.35 + 0.65 * footprint_pull)
    floors = max(1, round(height / options.floor_height))
    return floors * options.floor_height


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _triangulate(polygon) -> list[list[tuple[float, float]]]:
    """Triangulate a simple polygon (Delaunay, keeping the inside triangles)."""
    from shapely.ops import triangulate
    from shapely.prepared import prep

    inside = prep(polygon)
    return [
        list(tri.exterior.coords)[:-1]
        for tri in triangulate(polygon)
        if inside.contains(tri.representative_point())
    ]


def extrude(polygon, base_z: float, height: float, *, skirt: float = 1.0) -> tuple[Mesh, Mesh]:
    """Walls and roof for one footprint.

    The walls start ``skirt`` below the base so a building on a slope has no gap
    under it — the ground is a coarse surface and the footprint is flat.
    """
    exterior = list(polygon.exterior.coords)[:-1]
    if len(exterior) < 3:
        return Mesh([], []), Mesh([], [])
    if signed_area_xy(exterior) < 0:
        exterior = list(reversed(exterior))  # counter-clockwise: walls face outward

    # A courtyard needs walls too, wound the other way so they face inward.
    rings = [exterior]
    for interior in polygon.interiors:
        ring = list(interior.coords)[:-1]
        if len(ring) < 3:
            continue
        rings.append(ring if signed_area_xy(ring) < 0 else list(reversed(ring)))

    top_z = base_z + height
    bottom_z = base_z - skirt

    wall_vertices: list[tuple[float, float, float]] = []
    wall_faces: list[list[int]] = []
    for ring in rings:
        for i, (x, y) in enumerate(ring):
            nx, ny = ring[(i + 1) % len(ring)]
            base = len(wall_vertices)
            wall_vertices.extend([
                (x, y, bottom_z), (nx, ny, bottom_z), (nx, ny, top_z), (x, y, top_z),
            ])
            wall_faces.append([base, base + 1, base + 2, base + 3])

    roof_vertices: list[tuple[float, float, float]] = []
    roof_faces: list[list[int]] = []
    index: dict[tuple[int, int], int] = {}
    for triangle in _triangulate(polygon):
        ordered = triangle if signed_area_xy(triangle) > 0 else list(reversed(triangle))
        face = []
        for x, y in ordered:
            key = (round(x * 1000), round(y * 1000))
            if key not in index:
                index[key] = len(roof_vertices)
                roof_vertices.append((x, y, top_z))
            face.append(index[key])
        if len(set(face)) == 3:
            roof_faces.append(face)

    return Mesh(wall_vertices, wall_faces), Mesh(roof_vertices, roof_faces)


def base_height(polygon, heightmap: HeightMap) -> float:
    """Ground height to stand a footprint on: the lowest of its corners.

    Taking the lowest means a building on a slope cuts into the hill on the
    high side rather than floating on the low side, which is what a real
    building does.
    """
    return min(heightmap.sample(x, y) for x, y in polygon.exterior.coords)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(
    heightmap: HeightMap,
    road_union,
    options: BuildingOptions | None = None,
    *,
    bounds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fill the empty ground with buildings.

    Returns ``{"Buildings": [Mesh], "Roofs": [Mesh], "plots": [...]}`` — the
    walls and roofs are separate because a texturing pass treats a facade and a
    roof completely differently.
    """
    options = options or BuildingOptions()
    if bounds is None:
        bounds = (
            heightmap.x0,
            heightmap.y0,
            heightmap.x0 + (heightmap.nx - 1) * heightmap.cell,
            heightmap.y0 + (heightmap.ny - 1) * heightmap.cell,
        )

    rng = random.Random(options.seed + 1)
    plots = footprints(road_union, bounds, options)

    walls, roofs, records = [], [], []
    for plot in plots:
        base = base_height(plot, heightmap)
        height = pick_height(plot.area, options, rng)
        wall_mesh, roof_mesh = extrude(plot, base, height, skirt=options.skirt)
        if not wall_mesh.faces or not roof_mesh.faces:
            continue
        walls.append(wall_mesh)
        roofs.append(roof_mesh)
        records.append({
            "area": round(plot.area, 2),
            "height": round(height, 2),
            "base_z": round(base, 3),
            "centroid": [round(plot.centroid.x, 3), round(plot.centroid.y, 3)],
        })

    return {"Buildings": walls, "Roofs": roofs, "plots": records}
