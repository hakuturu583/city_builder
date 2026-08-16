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
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np

from .geometry import Mesh, Ribbon, cover_with_triangles, signed_area_xy

DEFAULT_CELL = 10.0
DEFAULT_ORDER = 2  # thin plate; see _solve for why 1 creases at every kerb
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
    order: int = DEFAULT_ORDER,
    guidance: np.ndarray | Callable[..., np.ndarray | None] | None = None,
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

    by_cell = np.argsort(flat, kind="stable")
    flat_sorted, z_sorted = flat[by_cell], arr[by_cell, 2]
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

    # The grid is only settled here, so a guidance source that has to be
    # fetched for this extent is given the extent rather than guessing it.
    if callable(guidance):
        guidance = guidance(x0, y0, nx, ny, cell)

    support = _support_distance(known)
    filled = _solve(np.minimum(z, floor), known, order=order, guidance=guidance)
    if smooth > 0:
        filled = _smooth(filled, smooth)

    # The solve honours its constraints exactly, but smoothing averages and
    # averaging can lift the surface back over a sample. Clamp, let only the
    # free cells re-settle, clamp again.
    constrained = known & ~np.isnan(floor)
    filled = np.where(constrained, np.minimum(filled, floor), filled)
    filled = _solve(filled, constrained, order=order, guidance=guidance)
    filled = np.where(constrained, np.minimum(filled, floor), filled)

    return HeightMap(x0, y0, cell, filled, support)


def _support_distance(known: np.ndarray) -> np.ndarray:
    from scipy.ndimage import distance_transform_edt

    return distance_transform_edt(~known)


def _laplacian(ny: int, nx: int):
    """Five-point Laplacian on the grid, ``mean(neighbours) - self``.

    Out-of-grid neighbours are taken as the cell itself, which is the discrete
    Neumann condition — and the same thing ``np.pad(..., mode="edge")`` did in
    the Jacobi loop this replaces, so the edges behave as they always have.
    """
    from scipy.sparse import coo_matrix

    index = np.arange(ny * nx).reshape(ny, nx)
    rows, cols, data = [], [], []
    for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        ys = np.clip(np.arange(ny) + dy, 0, ny - 1)
        xs = np.clip(np.arange(nx) + dx, 0, nx - 1)
        rows.append(index.ravel())
        cols.append(index[np.ix_(ys, xs)].ravel())
        data.append(np.full(ny * nx, 0.25))
    rows.append(index.ravel())
    cols.append(index.ravel())
    data.append(np.full(ny * nx, -1.0))
    return coo_matrix((np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
                      shape=(ny * nx, ny * nx)).tocsr()


def _solve(z: np.ndarray, known: np.ndarray, *, order: int = DEFAULT_ORDER,
           guidance: np.ndarray | None = None) -> np.ndarray:
    """Fill the unknown cells by minimising a surface energy under constraints.

    ``order`` is which energy:

    * ``1`` — the membrane, :math:`\\|\\nabla z\\|^2`, whose solution is the
      harmonic one: every free cell is the average of its neighbours.
    * ``2`` — the thin plate, :math:`\\|\\Delta z\\|^2`. **This is the default,
      and the difference is not cosmetic.** A harmonic surface is only C⁰ at its
      constraints, so it *creases along every kerb* by construction; a thin
      plate is C¹ there and rolls off the shoulder the way an embankment does.
      And harmonic functions obey the maximum principle — an interior value can
      never exceed its boundary — which makes relief inside a city block not
      merely unlikely but impossible: every block is a saddle between the
      streets around it. The thin plate has no such rule.

    ``guidance`` is a surface whose *shape* is copied while its heights are
    ignored — the flat energy :math:`\\|\\Delta z\\|^2` becomes
    :math:`\\|\\Delta z - \\Delta p\\|^2`, which is Poisson editing. This is how
    a published elevation model is used, and it is deliberately not the obvious
    way. Measured on Kashiwanoha, that model and the map's own road elevations
    disagree by p90 3.19 m after their datums are aligned, and the disagreement
    is a smooth tilt rather than noise. Screening against its *values* therefore
    fights the roads, and does: on held-out road samples it takes the error from
    0.32 m to 0.50 m at a weight of 0.1 and 0.88 m at 10. Matching its
    *curvature* does not — the roads keep their heights, the ground between them
    inherits the mound, the ditch and the fall the model measured, and the
    constant offset and the tilt both cancel because neither survives a
    Laplacian.

    Solved once, sparsely, rather than iterated: a city map is tens of
    thousands of cells, which is small for a direct sparse solve and was four
    hundred Jacobi sweeps before.
    """
    from scipy.sparse.linalg import spsolve

    ny, nx = z.shape
    filled = np.array(z, dtype=float)
    if not known.any():
        return filled
    filled[~known] = float(np.nanmean(filled[known]))
    if known.all():
        return filled

    laplace = _laplacian(ny, nx)
    # Positive semi-definite either way, so the reduced system is too.
    system = (laplace.T @ laplace) if order >= 2 else -laplace
    rhs = np.zeros(ny * nx)

    if guidance is not None:
        # Only the curvature is wanted, so gaps in the model are gaps in the
        # guidance — filled flat, which asks for no curvature there and leaves
        # those cells to the plain interpolation.
        shape = np.asarray(guidance, dtype=float)
        shape = np.where(np.isfinite(shape), shape, np.nanmean(shape) if
                         np.isfinite(shape).any() else 0.0)
        wanted = laplace @ shape.ravel()
        rhs = rhs + ((laplace.T @ wanted) if order >= 2 else -wanted)

    free = ~known.ravel()
    fixed_values = filled.ravel()[~free]
    reduced = system[free][:, free]
    coupling = system[free][:, ~free]
    answer = spsolve(reduced.tocsc(), rhs[free] - coupling @ fixed_values)

    out = filled.ravel().copy()
    out[free] = answer
    return out.reshape(ny, nx)


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
    order: int = DEFAULT_ORDER,
    guidance: np.ndarray | Callable[..., np.ndarray | None] | None = None,
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
                                     order=order, guidance=guidance, bounds=bounds)


# ---------------------------------------------------------------------------
# Ground mesh
# ---------------------------------------------------------------------------


_WELD = 1e-3
"""Metres at which two positions are one vertex, and at which a vertex counts
as lying *on* the road outline.

One number for both, because they are the same question. The clip computes its
seam vertices *from* the outline, so they land on it to within floating point:
on Kashiwanoha 1218 of the ground's 1714 vertices sit within 6e-15 m of it, one
more at 6e-4 m where a cell edge runs almost tangent to a kerb, and the next
nearest is 65 mm out. The threshold has room either side and nothing is moved
to reach the seam — which is what separates this from the 50 mm snap it
replaced. It classifies the same 1219 vertices that snap did; what changed is
the height they are given.
"""


class _Kerb:
    """The carriageway's own height, read off the edge it is asked about.

    Every ground lanelet's boundary ring, held as individual segments so a
    query lands on the segment it is actually on. The road mesh interpolates
    linearly between the same vertices, so a height taken here *is* the road
    surface at that point rather than an estimate of it — which is the whole
    difference between this and the nearest-neighbour snap it replaces.
    """

    def __init__(self, rings: Sequence[Sequence[Sequence[float]]]):
        from shapely.geometry import LineString
        from shapely.strtree import STRtree

        starts, ends, segments = [], [], []
        for ring in rings:
            for a, b in zip(ring, (*ring[1:], ring[0])):
                if a[0] == b[0] and a[1] == b[1]:
                    continue
                starts.append(a)
                ends.append(b)
                segments.append(LineString([(a[0], a[1]), (b[0], b[1])]))
        self.empty = not segments
        if self.empty:
            return
        self._a = np.asarray(starts, dtype=float)
        self._b = np.asarray(ends, dtype=float)
        self._tree = STRtree(segments)

    def height_at(self, x: float, y: float) -> float:
        from shapely.geometry import Point

        found = np.atleast_1d(self._tree.query_nearest(Point(x, y)))
        a, d = self._a[found], self._b[found] - self._a[found]
        along = (x - a[:, 0]) * d[:, 0] + (y - a[:, 1]) * d[:, 1]
        t = np.clip(along / np.maximum((d[:, :2] ** 2).sum(axis=1), 1e-18), 0.0, 1.0)
        # Lowest of the equally near ones. Where a lanelet ends against the
        # side of another a few centimetres lower, both edges pass through the
        # same point and the ground has to choose; standing proud of the lower
        # carriageway is the failure that shows, so it takes that one.
        return float((a[:, 2] + t * d[:, 2]).min())


def _rings_of(geometry) -> list[list[tuple[float, float]]]:
    """Every closed ring of a polygonal geometry, exteriors and holes alike."""
    parts = geometry.geoms if hasattr(geometry, "geoms") else [geometry]
    return [list(ring.coords)
            for part in parts if part.geom_type == "Polygon"
            for ring in (part.exterior, *part.interiors)]


def road_outline(ribbons: Sequence[Ribbon], elevated: set[int], *,
                 close_gap: float = 0.5, fill_island: float = 0.0):
    """The carriageway of the ground-level lanelets, dissolved into one shape.

    Dissolved rather than left as a list: clipping against the lanelets one by
    one leaves the hairline gaps between neighbouring lanes as ground, and
    those slivers surface in the middle of the carriageway. Closing by
    ``close_gap`` and opening again is what shuts them.

    Split out of :func:`build_mesh` because the water pass needs it *before*
    the ground is triangulated — a pond must not flood a street — and building
    it twice on a city-sized map is not free.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    footprints = []
    for ribbon in ribbons:
        if ribbon.id in elevated or len(ribbon.left) < 2 or len(ribbon.right) < 2:
            continue
        poly = ShapelyPolygon([(p[0], p[1]) for p in ribbon.ring()])
        if not poly.is_valid:
            poly = poly.buffer(0)
        if not poly.is_empty and poly.area > 1e-6:
            footprints.append(poly)
    if not footprints:
        return None

    roads = unary_union([p.buffer(close_gap) for p in footprints]).buffer(-close_gap)
    if fill_island > 0:
        # Turning lanelets do not tile a junction exactly. Their scraps can be
        # absorbed into the carriageway, but only where the road mesh actually
        # covers them — otherwise this trades a sliver for a hole in the scene.
        parts = list(roads.geoms) if hasattr(roads, "geoms") else [roads]
        patched = [ShapelyPolygon(part.exterior,
                                  [r for r in part.interiors
                                   if ShapelyPolygon(r).area >= fill_island])
                   for part in parts if part.geom_type == "Polygon"]
        roads = unary_union(patched) if patched else roads
    return roads


def build_mesh(
    hm: HeightMap,
    ribbons: Sequence[Ribbon],
    elevated: set[int],
    *,
    close_gap: float = 0.5,
    fill_island: float = 0.0,
    drop: float = 0.05,
    breaklines: Sequence[Any] = (),
    terraces: Sequence[tuple[Any, float]] = (),
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

    The seam is then **exact by construction** rather than by a tolerance. The
    kerb is what GIS calls a breakline, and each clipped cell is triangulated
    with the outline as a *forced edge* — a constrained Delaunay triangulation,
    the standard treatment, and what RoadRunner's terrain does when it shares
    nodes with the road network. Every triangle edge along the kerb is
    therefore a segment of the road outline, and every vertex on it is a point
    of the road outline carrying the carriageway's own height less ``drop``.
    Interior vertices take the height map. The exception is where ``close_gap``
    bridged the hairline gap between two lanelets: those arcs of the outline
    are over no carriageway at all — 71 of Kashiwanoha's 1219 seam vertices sit
    more than 5 cm outside every lanelet, the worst 0.36 m out into a junction
    — and they take the nearest lanelet edge's height, which is the best the
    map defines there.

    Two things follow that the nearest-neighbour snap this replaces could not
    give. The seam matches the carriageway to 9e-16 m on Kashiwanoha, against a
    worst case of 0.21 m before: the old rule looked for road boundary
    *samples* within a metre, a lanelet's boundary is only sampled where it has
    a vertex, and a straight lanelet has two — so the search usually found
    nothing and fell back to the nearest boundary vertex anywhere on the map.
    And the ground tiles its region exactly. The unconstrained triangulation
    that came before kept the triangles whose representative point fell inside
    the clipped piece — one point per triangle — so a triangle straddling a
    concavity could be kept with a third of itself out over the road, or dropped
    with most of itself in. Measured on two lanelets meeting at an oblique
    angle, it laid 15.7 m2 of ground over the carriageway.

    Cell by cell rather than one triangulation of the whole region, because a
    constrained Delaunay adds no points of its own: over the region as a whole
    it would span each city block with a few long triangles and the height map
    between the roads would never be sampled. The cell edges are constraints
    too, so the grid keeps its 10 m resolution where there is no road.
    """
    from shapely.geometry import LineString, Point, box
    from shapely.ops import polygonize, unary_union
    from shapely.prepared import prep
    from shapely.strtree import STRtree

    rings = [ribbon.ring() for ribbon in ribbons
             if ribbon.id not in elevated
             and len(ribbon.left) >= 2 and len(ribbon.right) >= 2]
    roads = road_outline(ribbons, elevated, close_gap=close_gap,
                         fill_island=fill_island)

    road_test = prep(roads) if roads is not None else None
    kerb = _Kerb(rings) if rings else None
    if kerb is not None and kerb.empty:
        kerb = None  # every lanelet collapsed to a point
    # The outline segment by segment, so asking whether a vertex is on the
    # seam is an indexed lookup rather than a walk of a several-thousand-vertex
    # ring: the ground of a city map has tens of thousands of vertices and each
    # one asks once.
    seam = STRtree([LineString([ring[i], ring[i + 1]])
                    for ring in _rings_of(roads) for i in range(len(ring) - 1)]) \
        if roads is not None else None

    vertices: list[tuple[float, float, float]] = []
    lookup: dict[tuple[int, int], int] = {}

    def vertex(x: float, y: float, lift: float = 0.0) -> int:
        # The lift is part of the key, so the two sides of a retaining wall do
        # not weld into one vertex and pull the step flat again.
        key = (round(x / _WELD), round(y / _WELD), round(lift / _WELD))
        found = lookup.get(key)
        if found is not None:
            return found
        on_seam = seam is not None and len(
            seam.query(Point(x, y), predicate="dwithin", distance=_WELD)) > 0
        if on_seam and kerb is not None:
            # The carriageway's height at this exact point, less `drop`. The
            # drop is not error absorption — the seam follows the kerb to
            # floating point now — but a kerb: a ground that meets the asphalt
            # dead level reads as a painted edge, and 5 cm is what a low kerb
            # measures.
            z = kerb.height_at(x, y) - drop
        else:
            z = hm.sample(x, y)
        lookup[key] = len(vertices)
        vertices.append((x, y, z + lift))
        return lookup[key]

    faces: list[list[int]] = []

    def add_face(ring: Sequence[Sequence[float]], lift: float = 0.0) -> None:
        area = signed_area_xy(ring)
        if abs(area) < 5e-5:
            return
        ordered = list(ring) if area > 0 else list(reversed(ring))
        indices = [vertex(p[0], p[1], lift) for p in ordered]
        # Vertices are welded at `_WELD`, so a clipped sliver can collapse
        # onto itself even though its outline had area.
        if len(set(indices)) < 3:
            return
        faces.append(indices)

    def lift_of(piece) -> float:
        """How far this piece of ground stands above the surface around it."""
        if not steps:
            return 0.0
        spot = piece.representative_point()
        for outline, rise in steps:
            if outline.contains(spot):
                return rise
        return 0.0

    def emit(piece) -> None:
        if piece.is_empty or piece.area < 1e-6:
            return
        lift = lift_of(piece)
        for tri in cover_with_triangles(piece):
            add_face(list(tri.exterior.coords)[:-1], lift)

    # A shoreline, a retaining wall, the toe of an embankment: a line the
    # ground is *meant* to break along. Splitting each cell by it puts it in
    # the triangulation as forced edges, exactly as the kerb is, so the bank
    # is a rim rather than whatever angle the grid happened to cut across it.
    edges = []
    for shape in breaklines:
        if shape is None or shape.is_empty:
            continue
        line = getattr(shape, "boundary", None)
        edges.append(line if line is not None and not line.is_empty else shape)
    # A terrace is a breakline that also carries a level: its outline is forced
    # into the triangulation *and* the ground inside it stands `rise` higher, so
    # the two sides of the line are two vertices at two heights rather than one
    # smeared between them. That is the difference between a retaining wall and
    # a ramp, and on a 10 m grid it is the only way to have the first.
    steps = [(outline, rise) for outline, rise in terraces
             if outline is not None and not outline.is_empty and rise]
    edges = edges + [outline.boundary for outline, _rise in steps]
    edge_tree = STRtree(edges) if edges else None

    def split(piece):
        """``piece`` cut along every breakline that crosses it."""
        if edge_tree is None:
            return [piece]
        near = [edges[int(i)] for i in edge_tree.query(piece)]
        near = [line for line in near if piece.intersects(line)]
        if not near:
            return [piece]
        cut = unary_union([piece.boundary, *near])
        return [part for part in polygonize(cut)
                if part.representative_point().within(piece)]

    for iy in range(hm.ny - 1):
        for ix in range(hm.nx - 1):
            x0 = hm.x0 + ix * hm.cell
            y0 = hm.y0 + iy * hm.cell
            cell = box(x0, y0, x0 + hm.cell, y0 + hm.cell)
            if road_test is not None and road_test.contains(cell):
                continue  # entirely carriageway
            if road_test is None or not road_test.intersects(cell):
                whole = box(x0, y0, x0 + hm.cell, y0 + hm.cell)
                for piece in split(whole):
                    if len(piece.exterior.coords) == 5 and not piece.interiors:
                        add_face(list(piece.exterior.coords)[:-1])
                    else:
                        emit(piece)
                continue
            clipped = cell.difference(roads)
            for piece in (clipped.geoms if hasattr(clipped, "geoms") else [clipped]):
                if piece.geom_type == "Polygon":
                    for part in split(piece):
                        emit(part)

    # And the wall that holds the terrace up. Without it the raised ground ends
    # in mid-air and the scene has a shelf floating over a hole.
    for outline, rise in steps:
        for part in (outline.geoms if hasattr(outline, "geoms") else [outline]):
            if part.geom_type != "Polygon":
                continue
            for ring in (part.exterior, *part.interiors):
                for (x0, y0), (x1, y1) in pairwise(list(ring.coords)):
                    if math.dist((x0, y0), (x1, y1)) < _WELD:
                        continue
                    low = [vertex(x0, y0), vertex(x1, y1)]
                    high = [vertex(x1, y1, rise), vertex(x0, y0, rise)]
                    face = low + high if rise > 0 else high + low
                    if len(set(face)) == 4:
                        faces.append(face)

    mesh = Mesh(vertices, faces)
    # The dissolved outline is expensive to build and the building layer needs
    # exactly the same one, so hand it back rather than recomputing it.
    return (mesh, roads) if return_road_union else mesh
