"""Ground surface, reconstructed from the road elevations of a Lanelet2 map.

A Lanelet2 map carries no terrain — elevation exists only on the carriageway.
The road network is nonetheless a dense set of ground samples, so a usable
ground surface can be interpolated from it:

1. **Drop what is not ground.** Taking every lanelet's z at face value puts the
   ground on top of the overpass, and these maps carry no ``bridge`` or
   ``layer`` tag. Elevated structure is found geometrically instead.
2. **Bin what is left into a grid**, as a lower envelope rather than a fit.
3. **Fill the gaps** by relaxation — most of a city block has no road in it.
4. **Clip to the road outline**, so the ground stops at the kerb rather than
   crossing the carriageway.

The result is deliberately coarse. It is a surface for buildings, props and
vegetation to sit on, not a survey product: accuracy falls off with distance
from the road network, and :attr:`HeightMap.support` records where the surface
is measured and where it is invented.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .geometry import Mesh, Ribbon, signed_area_xy

DEFAULT_CELL = 10.0
DEFAULT_SMOOTH = 1.0
DEFAULT_Z_GAP = 2.0  # m of separation before an overlap counts as stacked
DEFAULT_MIN_OVERLAP = 5.0  # m2, ignores slivers where two lanelets merely touch
DEFAULT_CLEARANCE = 1.5  # m above the local street before a ramp counts elevated


# ---------------------------------------------------------------------------
# Height map
# ---------------------------------------------------------------------------


@dataclass
class HeightMap:
    """Regular grid of ground heights in the scene frame."""

    x0: float
    y0: float
    cell: float
    z: np.ndarray  # (ny, nx)
    support: np.ndarray  # distance in cells to the nearest measured cell

    @property
    def ny(self) -> int:
        return self.z.shape[0]

    @property
    def nx(self) -> int:
        return self.z.shape[1]

    def sample(self, x: float, y: float) -> float:
        """Bilinear height at a scene position, clamped at the edges."""
        fx = (x - self.x0) / self.cell
        fy = (y - self.y0) / self.cell
        ix = min(max(math.floor(fx), 0), self.nx - 1)
        iy = min(max(math.floor(fy), 0), self.ny - 1)
        ix2, iy2 = min(ix + 1, self.nx - 1), min(iy + 1, self.ny - 1)
        tx = min(max(fx - ix, 0.0), 1.0)
        ty = min(max(fy - iy, 0.0), 1.0)

        z00, z10 = self.z[iy, ix], self.z[iy, ix2]
        z01, z11 = self.z[iy2, ix], self.z[iy2, ix2]
        return float((z00 * (1 - tx) + z10 * tx) * (1 - ty) + (z01 * (1 - tx) + z11 * tx) * ty)

    def to_json(self) -> dict[str, Any]:
        return {
            "x0": round(self.x0, 4),
            "y0": round(self.y0, 4),
            "cell": self.cell,
            "nx": self.nx,
            "ny": self.ny,
            "z": [round(float(v), 3) for v in self.z.ravel()],
            "support_cells": [int(v) for v in self.support.ravel()],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> HeightMap:
        ny, nx = int(data["ny"]), int(data["nx"])
        z = np.asarray(data["z"], dtype=float).reshape(ny, nx)
        support = np.asarray(data.get("support_cells", np.zeros(ny * nx)), dtype=float).reshape(ny, nx)
        return cls(float(data["x0"]), float(data["y0"]), float(data["cell"]), z, support)


def build_heightmap(
    points: Sequence[Sequence[float]],
    *,
    cell: float = DEFAULT_CELL,
    margin: float = 30.0,
    bounds: Sequence[float] | None = None,
    smooth: float = DEFAULT_SMOOTH,
    relax_iterations: int = 400,
    percentile: float = 10.0,
    drop: float = 0.05,
) -> HeightMap | None:
    """Bin ground samples into a grid and interpolate across the empty cells.

    The result is a *lower envelope*, not a fit. A surface that merely passes
    through the middle of its samples cuts through the road it was built from —
    measured on Nishi-Shinjuku, a median fit buried 28 % of the network, 9 % of
    it deeper than a kerb. Cells therefore take a low percentile, and after
    interpolation and smoothing the surface is clamped back under the lowest
    sample in every cell that has one.

    ``bounds`` fixes the extent instead of taking ``margin`` around the samples.
    That matters once the roads have been run out to the map edge: the edge is
    a margin around the roads, so taking a fresh margin around the *extended*
    roads would push it out again and leave another ring of ground with no road
    reaching it — which is the thing the extension was for.
    """
    arr = np.asarray(points, dtype=float)
    if len(arr) < 3:
        return None

    if bounds is not None:
        x0, y0, x1, y1 = (float(v) for v in bounds)
    else:
        x0 = float(arr[:, 0].min() - margin)
        y0 = float(arr[:, 1].min() - margin)
        x1 = float(arr[:, 0].max() + margin)
        y1 = float(arr[:, 1].max() + margin)
    nx = math.ceil((x1 - x0) / cell) + 1
    ny = math.ceil((y1 - y0) / cell) + 1

    ix = np.clip(((arr[:, 0] - x0) / cell).astype(int), 0, nx - 1)
    iy = np.clip(((arr[:, 1] - y0) / cell).astype(int), 0, ny - 1)
    flat = iy * nx + ix

    order = np.argsort(flat, kind="stable")
    flat_sorted, z_sorted = flat[order], arr[order, 2]
    cells = np.split(np.arange(len(flat_sorted)), np.flatnonzero(np.diff(flat_sorted)) + 1)

    z = np.full(nx * ny, np.nan)
    floor = np.full(nx * ny, np.nan)  # the ceiling the ground must stay under
    for group in cells:
        if len(group):
            target = flat_sorted[group[0]]
            z[target] = np.percentile(z_sorted[group], percentile)
            floor[target] = z_sorted[group].min()
    z = z.reshape(ny, nx)
    floor = floor.reshape(ny, nx) - drop

    known = ~np.isnan(z)
    if not known.any():
        return None

    support = _support_distance(known)
    filled = _relax(np.minimum(z, floor), known, relax_iterations)
    if smooth > 0:
        filled = _smooth(filled, smooth)

    # Relaxation and smoothing both average, and averaging can lift the surface
    # back over a sample. Clamp, let only the free cells re-settle, clamp again.
    constrained = known & ~np.isnan(floor)
    filled = np.where(constrained, np.minimum(filled, floor), filled)
    filled = _relax(filled, constrained, relax_iterations // 4)
    filled = np.where(constrained, np.minimum(filled, floor), filled)

    return HeightMap(x0, y0, cell, filled, support)


def _support_distance(known: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~known)


def _relax(z: np.ndarray, known: np.ndarray, iterations: int) -> np.ndarray:
    """Solve the unknown cells by averaging their neighbours (Laplace).

    Nearest-neighbour fill would step; this gives the gently varying surface a
    city block actually has between its bounding streets.
    """
    filled = np.array(z, dtype=float)
    if np.isnan(filled).any():
        filled = np.where(np.isnan(filled), np.nanmean(filled[known]), filled)
    for _ in range(iterations):
        padded = np.pad(filled, 1, mode="edge")
        neighbours = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.25
        updated = np.where(known, filled, neighbours)
        if np.max(np.abs(updated - filled)) < 1e-4:
            return updated
        filled = updated
    return filled


def _smooth(z: np.ndarray, sigma: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(z, sigma=sigma, mode="nearest")


# ---------------------------------------------------------------------------
# Elevated structure
# ---------------------------------------------------------------------------


@dataclass
class Footprint:
    id: int
    ring: list[tuple[float, float]]
    mean_z: float
    centroid: tuple[float, float]


def footprint_of(ribbon: Ribbon) -> Footprint | None:
    if len(ribbon.left) < 2 or len(ribbon.right) < 2:
        return None
    ring = [(p[0], p[1]) for p in ribbon.ring()]
    zs = [p[2] for p in (*ribbon.left, *ribbon.right)]
    cx = sum(p[0] for p in ring) / len(ring)
    cy = sum(p[1] for p in ring) / len(ring)
    return Footprint(ribbon.id, ring, float(np.mean(zs)), (cx, cy))


def find_stacked(
    footprints: Sequence[Footprint],
    *,
    z_gap: float = DEFAULT_Z_GAP,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
) -> set[int]:
    """Lanelets that pass over another one: the seeds of the elevated set."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.strtree import STRtree

    polygons, kept = [], []
    for fp in footprints:
        poly = ShapelyPolygon(fp.ring)
        if not poly.is_valid:
            poly = poly.buffer(0)
        if poly.is_empty or poly.area < 1e-6:
            continue
        polygons.append(poly)
        kept.append(fp)

    if not polygons:
        return set()

    tree = STRtree(polygons)
    elevated: set[int] = set()
    for i, poly in enumerate(polygons):
        for j in tree.query(poly):
            j = int(j)
            if j <= i or abs(kept[i].mean_z - kept[j].mean_z) < z_gap:
                continue
            if not poly.intersects(polygons[j]):
                continue
            if poly.intersection(polygons[j]).area < min_overlap:
                continue
            higher = kept[i] if kept[i].mean_z > kept[j].mean_z else kept[j]
            elevated.add(higher.id)
    return elevated


def grow_elevated(
    seeds: set[int],
    footprints: Sequence[Footprint],
    adjacency: dict[int, set[int]],
    *,
    clearance: float = DEFAULT_CLEARANCE,
    radius: float = 60.0,
    max_rounds: int = 8,
) -> set[int]:
    """Extend the elevated set along connectivity, down the approach ramps.

    A ramp overlaps nothing, so the overlap test cannot see it, and it climbs
    too gently for a local height test. It is continuous with the deck though,
    so walking the lanelet graph reaches it.

    A candidate is judged against the street level *beside* it — a low
    percentile over the nearby lanelets currently believed to be ground —
    because judging a ramp against a surface that still contains that ramp is
    self-referential and stops the walk at the first step. Removing a ramp
    lowers the street level around it and exposes the next one, so the rounds
    repeat until the set stops growing.
    """
    from scipy.spatial import cKDTree

    ids = [fp.id for fp in footprints]
    index = {fp.id: i for i, fp in enumerate(footprints)}
    centroids = np.array([fp.centroid for fp in footprints], dtype=float)
    heights = np.array([fp.mean_z for fp in footprints], dtype=float)

    elevated = set(seeds)
    for _ in range(max_rounds):
        ground_idx = np.array([i for i, lid in enumerate(ids) if lid not in elevated])
        if len(ground_idx) < 3:
            break
        tree = cKDTree(centroids[ground_idx])

        def street_level(i: int, tree=tree, ground_idx=ground_idx) -> float:
            near = tree.query_ball_point(centroids[i], r=radius)
            if not near:
                _, nearest = tree.query(centroids[i], k=min(5, len(ground_idx)))
                near = np.atleast_1d(nearest)
            return float(np.percentile(heights[ground_idx[near]], 10))

        added: set[int] = set()
        queue = list(elevated)
        while queue:
            for neighbour in adjacency.get(queue.pop(), ()):
                if neighbour in elevated or neighbour in added or neighbour not in index:
                    continue
                if heights[index[neighbour]] - street_level(index[neighbour]) < clearance:
                    continue
                added.add(neighbour)
                queue.append(neighbour)

        if not added:
            break
        elevated |= added

    return elevated


def classify(
    ribbons: Sequence[Ribbon],
    adjacency: dict[int, set[int]],
    *,
    cell: float = DEFAULT_CELL,
    z_gap: float = DEFAULT_Z_GAP,
    min_overlap: float = DEFAULT_MIN_OVERLAP,
    clearance: float = DEFAULT_CLEARANCE,
    smooth: float = DEFAULT_SMOOTH,
    drop: float = 0.05,
    bounds: Sequence[float] | None = None,
) -> tuple[set[int], HeightMap | None]:
    """Split lanelets into ground and elevated, and build the ground heightmap."""
    footprints = [fp for fp in (footprint_of(r) for r in ribbons) if fp is not None]
    if not footprints:
        return set(), None

    seeds = find_stacked(footprints, z_gap=z_gap, min_overlap=min_overlap)
    elevated = grow_elevated(seeds, footprints, adjacency, clearance=clearance)

    ground_points = [tuple(p) for r in ribbons if r.id not in elevated for p in (*r.left, *r.right)]
    return elevated, build_heightmap(ground_points, cell=cell, smooth=smooth, drop=drop,
                                     bounds=bounds)


# ---------------------------------------------------------------------------
# Ground mesh
# ---------------------------------------------------------------------------


def build_mesh(
    hm: HeightMap,
    ribbons: Sequence[Ribbon],
    elevated: set[int],
    *,
    close_gap: float = 0.5,
    fill_island: float = 0.0,
    snap_tolerance: float = 0.05,
    seam_radius: float = 1.0,
    drop: float = 0.05,
    return_road_union: bool = False,
) -> Mesh | tuple[Mesh, Any]:
    """Triangulate the ground *around* the roads, meeting them at their edges.

    The grid is clipped cell by cell against the dissolved road outline rather
    than handed to a mesh boolean. Two reasons, both measured: a grid surface
    and a road surface intersect wherever the grid cannot follow a carriageway
    that crosses a cell on a slope (a clamped lower envelope still cut 6 % of
    the network), and a boolean needs the outline as a closed solid — the
    dissolved outline is a several-thousand-vertex concave ring whose cap
    Blender could not tessellate, silently deleting the entire ground.

    Vertices that land on the outline take the road's own height, so the ground
    meets the kerb line exactly instead of passing over or under it.
    """
    from scipy.spatial import cKDTree
    from shapely.geometry import Point, box
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import triangulate, unary_union
    from shapely.prepared import prep

    footprints, boundary = [], []
    for ribbon in ribbons:
        if ribbon.id in elevated or len(ribbon.left) < 2 or len(ribbon.right) < 2:
            continue
        ring = ribbon.ring()
        boundary.extend(ring)
        poly = ShapelyPolygon([(p[0], p[1]) for p in ring])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 1e-6:
            footprints.append(poly)

    # Dissolve first: clipping against the lanelets one by one leaves the
    # hairline gaps between neighbouring lanes as ground, and those slivers
    # surface in the middle of the carriageway.
    roads = unary_union([p.buffer(close_gap) for p in footprints]).buffer(-close_gap) if footprints else None
    if roads is not None and fill_island > 0:
        # Turning lanelets do not tile a junction exactly. Their scraps can be
        # absorbed into the carriageway, but only where the road mesh actually
        # covers them — otherwise this trades a sliver for a hole in the scene.
        parts = list(roads.geoms) if hasattr(roads, "geoms") else [roads]
        patched = [
            ShapelyPolygon(part.exterior, [r for r in part.interiors if ShapelyPolygon(r).area >= fill_island])
            for part in parts
            if part.geom_type == "Polygon"
        ]
        roads = unary_union(patched) if patched else roads

    road_test = prep(roads) if roads is not None else None
    outline = roads.boundary if roads is not None else None
    samples = np.array(boundary, dtype=float) if boundary else None
    tree = cKDTree(samples[:, :2]) if samples is not None else None

    vertices: list[tuple[float, float, float]] = []
    lookup: dict[tuple[int, int], int] = {}

    def vertex(x: float, y: float) -> int:
        key = (round(x * 1000), round(y * 1000))
        found = lookup.get(key)
        if found is not None:
            return found
        if outline is not None and Point(x, y).distance(outline) <= snap_tolerance:
            # On the kerb line. Take the *lowest* road boundary point nearby
            # rather than the nearest one: where two carriageways at slightly
            # different heights meet, the nearest may belong to the higher of
            # them and the seam would stand a few centimetres proud of the
            # lower one.
            near = tree.query_ball_point([x, y], r=seam_radius)
            z = float(samples[near, 2].min()) if near else float(samples[tree.query([x, y])[1], 2])
            # …and hold it `drop` below. The clip only puts a vertex where the
            # outline crosses a cell edge, so a seam segment can be metres long
            # and its linear interpolation rides above a sloping carriageway in
            # between — measured at 3 cm. Dropping the seam absorbs that, and a
            # few centimetres at the kerb is what a kerb looks like anyway.
            z -= drop
        else:
            z = hm.sample(x, y)
        lookup[key] = len(vertices)
        vertices.append((x, y, z))
        return lookup[key]

    faces: list[list[int]] = []

    def add_face(ring: Sequence[Sequence[float]]) -> None:
        area = signed_area_xy(ring)
        if abs(area) < 5e-5:
            return
        ordered = list(ring) if area > 0 else list(reversed(ring))
        indices = [vertex(p[0], p[1]) for p in ordered]
        # Vertices are shared on a millimetre grid, so a clipped sliver can
        # collapse onto itself even though its outline had area.
        if len(set(indices)) < 3:
            return
        faces.append(indices)

    def emit(piece) -> None:
        if piece.is_empty or piece.area < 1e-6:
            return
        if not piece.interiors and len(piece.exterior.coords) == 5:
            add_face(list(piece.exterior.coords)[:-1])
            return
        # Delaunay over the vertices, keeping the triangles inside the piece —
        # the standard way to triangulate a concave polygon without new points.
        inside = prep(piece)
        for tri in triangulate(piece):
            if inside.contains(tri.representative_point()):
                add_face(list(tri.exterior.coords)[:-1])

    for iy in range(hm.ny - 1):
        for ix in range(hm.nx - 1):
            x0 = hm.x0 + ix * hm.cell
            y0 = hm.y0 + iy * hm.cell
            cell = box(x0, y0, x0 + hm.cell, y0 + hm.cell)
            if road_test is not None and road_test.contains(cell):
                continue  # entirely carriageway
            if road_test is None or not road_test.intersects(cell):
                add_face([(x0, y0), (x0 + hm.cell, y0), (x0 + hm.cell, y0 + hm.cell), (x0, y0 + hm.cell)])
                continue
            clipped = cell.difference(roads)
            for piece in (clipped.geoms if hasattr(clipped, "geoms") else [clipped]):
                if piece.geom_type == "Polygon":
                    emit(piece)

    mesh = Mesh(vertices, faces)
    # The dissolved outline is expensive to build and the building layer needs
    # exactly the same one, so hand it back rather than recomputing it.
    return (mesh, roads) if return_road_union else mesh
