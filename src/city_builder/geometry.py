"""Geometry primitives shared by every surface this package builds.

Everything reduces to two shapes:

* a **ribbon** — two polylines of equal length, stitched into a quad strip.
  Lane surfaces, painted lines, stop lines and kerbs are all ribbons.
* a **polygon** — a closed ring, fan-triangulated. Zebra bars arrive this way.

Keeping the vocabulary that small is what lets the Blender layer stay thin.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class Ribbon:
    """A surface strip: matched left and right polylines."""

    id: int
    left: list[tuple[float, float, float]]
    right: list[tuple[float, float, float]]
    attributes: dict[str, str] = field(default_factory=dict)

    def shift_z(self, delta: float) -> None:
        self.left = [(x, y, z + delta) for x, y, z in self.left]
        self.right = [(x, y, z + delta) for x, y, z in self.right]

    def z_values(self) -> list[float]:
        return [p[2] for p in self.left]

    def ring(self) -> list[tuple[float, float, float]]:
        """Plan-view outline, for footprint tests."""
        return list(self.left) + list(reversed(self.right))


@dataclass
class Polygon:
    """A closed ring, triangulated as a fan."""

    id: int
    points: list[tuple[float, float, float]]
    attributes: dict[str, str] = field(default_factory=dict)

    def shift_z(self, delta: float) -> None:
        self.points = [(x, y, z + delta) for x, y, z in self.points]

    def z_values(self) -> list[float]:
        return [p[2] for p in self.points]


@dataclass
class Mesh:
    """A triangle/quad soup in scene coordinates, optionally with UVs."""

    vertices: list[tuple[float, float, float]]
    faces: list[list[int]]
    uvs: list[tuple[float, float]] | None = None  # per vertex, when authored

    def __bool__(self) -> bool:
        return bool(self.faces)


# ---------------------------------------------------------------------------
# Polylines
# ---------------------------------------------------------------------------


def resample_polyline(points: Sequence[Sequence[float]], count: int) -> np.ndarray:
    """Resample a 3D polyline to ``count`` points, evenly in arc length."""
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2 or count < 2:
        return arr

    seg = np.linalg.norm(np.diff(arr, axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = dist[-1]
    if total <= 1e-9:
        return np.repeat(arr[:1], count, axis=0)

    targets = np.linspace(0.0, total, count)
    return np.column_stack([np.interp(targets, dist, arr[:, i]) for i in range(arr.shape[1])])


def _sample_count(left: Sequence, right: Sequence, max_segment: float) -> int:
    count = max(len(left), len(right), 2)
    if max_segment > 0:
        for pts in (left, right):
            arr = np.asarray(pts, dtype=float)
            if len(arr) >= 2:
                length = float(np.linalg.norm(np.diff(arr, axis=0), axis=1).sum())
                count = max(count, math.ceil(length / max_segment) + 1)
    return count


def monotone_projection(left_samples: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Pair each left sample with its nearest point on the right bound, in order.

    Pairing two bounds by arc length assumes they are parameterised alike, and
    real maps break that: a crosswalk boundary can begin with a short stub
    pointing *back* along the lane, after which equal-arc-length samples on the
    two sides advance in opposite directions and the quad between them is
    twisted — it renders as a black bowtie.

    Projecting instead makes the connector the actual width, and forcing the
    projection to advance monotonically keeps the strip from folding back.
    """
    start = right[:-1]
    delta = right[1:] - start
    seg_len = np.linalg.norm(delta[:, :2], axis=1)
    seg_len2 = np.where(seg_len > 1e-12, seg_len**2, 1.0)
    cumulative = np.concatenate([[0.0], np.cumsum(seg_len)])[:-1]

    paired = np.empty((len(left_samples), right.shape[1]))
    reached = -np.inf

    for i, sample in enumerate(left_samples):
        offset = sample[:2] - start[:, :2]
        t = np.clip((offset * delta[:, :2]).sum(axis=1) / seg_len2, 0.0, 1.0)
        projected = start + t[:, None] * delta
        distance = np.linalg.norm(projected[:, :2] - sample[:2], axis=1)
        arc = cumulative + t * seg_len

        ahead = np.flatnonzero(arc >= reached - 1e-9)
        best = int(ahead[np.argmin(distance[ahead])]) if len(ahead) else int(np.argmin(distance))
        reached = arc[best]
        paired[i] = projected[best]

    paired[-1] = right[-1]  # the strip should span the full bound
    return paired


def pair_bounds(
    left: Sequence[Sequence[float]],
    right: Sequence[Sequence[float]],
    *,
    max_segment: float = 5.0,
) -> tuple[list, list] | None:
    """Resample a lanelet's two bounds onto a common parameterisation.

    Lanelet2 stores each boundary with its own vertex count (9 vs 7 is typical),
    so they have to be matched before they can be stitched into quads.
    """
    if len(left) < 2 or len(right) < 2:
        return None
    n = _sample_count(left, right, max_segment)
    ls = resample_polyline(left, n)
    rs = monotone_projection(ls, np.asarray(right, dtype=float))
    return [tuple(p) for p in ls], [tuple(p) for p in rs]


def ribbon_from_polyline(points: Sequence[Sequence[float]], width: float) -> tuple[list, list] | None:
    """Widen a polyline into a flat ribbon (paint, stop lines).

    The offset is taken in the XY plane and z carried through unchanged, so the
    marking follows the road's slope instead of floating over it.
    """
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2:
        return None

    half = width / 2.0
    left: list[tuple[float, float, float]] = []
    right: list[tuple[float, float, float]] = []

    for i in range(len(arr)):
        if i == 0:
            direction = arr[1, :2] - arr[0, :2]
        elif i == len(arr) - 1:
            direction = arr[-1, :2] - arr[-2, :2]
        else:
            d1 = arr[i, :2] - arr[i - 1, :2]
            d2 = arr[i + 1, :2] - arr[i, :2]
            n1, n2 = np.linalg.norm(d1), np.linalg.norm(d2)
            direction = (d1 / n1 if n1 > 1e-9 else 0) + (d2 / n2 if n2 > 1e-9 else 0)
            if np.isscalar(direction):
                direction = d1

        norm = np.linalg.norm(direction)
        if norm < 1e-9:
            continue
        direction = direction / norm
        perp = np.array([-direction[1], direction[0]])

        x, y, z = arr[i]
        left.append((x + perp[0] * half, y + perp[1] * half, z))
        right.append((x - perp[0] * half, y - perp[1] * half, z))

    if len(left) < 2:
        return None
    return left, right


def split_dashes(points: Sequence[Sequence[float]], dash_length: float, gap_length: float) -> list[np.ndarray]:
    """Cut a polyline into dash segments, preserving z along the way.

    A Lanelet2 map stores a dashed line as one continuous polyline, so the
    individual dashes are ours to invent.
    """
    arr = np.asarray(points, dtype=float)
    if len(arr) < 2 or dash_length <= 0 or gap_length < 0:
        return [arr]

    seg = np.linalg.norm(np.diff(arr[:, :2], axis=0), axis=1)
    dist = np.concatenate([[0.0], np.cumsum(seg)])
    total = dist[-1]
    if total <= dash_length:
        return [arr]

    def at(d: float) -> np.ndarray:
        return np.array([np.interp(d, dist, arr[:, i]) for i in range(arr.shape[1])])

    dashes = []
    start = 0.0
    period = dash_length + gap_length
    while start < total:
        end = min(start + dash_length, total)
        if end - start > 1e-6:
            inner = [arr[k] for k, d in enumerate(dist) if start < d < end]
            dashes.append(np.array([at(start), *inner, at(end)]))
        start += period
    return dashes


def extrude_curb(points: Sequence[Sequence[float]], height: float) -> tuple[list, list] | None:
    """Stand a kerb line up into a vertical strip.

    The base keeps the surveyed elevation of the border, so the kerb follows
    the road's slope instead of stepping.
    """
    base = [tuple(float(c) for c in p) for p in points]
    if len(base) < 2:
        return None
    return base, [(x, y, z + height) for x, y, z in base]


def close_ring(points: Sequence[Sequence[float]]) -> list[tuple[float, float, float]] | None:
    """Normalise a closed linestring into a ring without its repeated vertex."""
    ring = [tuple(float(c) for c in p) for p in points]
    if len(ring) > 2 and math.dist(ring[0][:2], ring[-1][:2]) < 1e-3:
        ring = ring[:-1]
    return ring if len(ring) >= 3 else None


# ---------------------------------------------------------------------------
# Faces
# ---------------------------------------------------------------------------


def normal(o: Sequence[float], a: Sequence[float], b: Sequence[float]) -> tuple[float, float, float]:
    """3D normal of triangle o-a-b, unnormalised."""
    ux, uy, uz = a[0] - o[0], a[1] - o[1], a[2] - o[2]
    vx, vy, vz = b[0] - o[0], b[1] - o[1], b[2] - o[2]
    return (uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx)


def quad_normals(a, b, c, d):
    return normal(a, b, c), normal(a, c, d)


def quad_is_sane(a, b, c, d) -> bool:
    """True if quad a-b-c-d is neither self-intersecting nor degenerate.

    The test is in 3D: kerbs are vertical strips whose faces have no area in
    the XY plane at all, so a planar test would discard every one of them.
    """
    n1, n2 = quad_normals(a, b, c, d)
    if sum(x * y for x, y in zip(n1, n2)) < 0:  # triangles wind opposite ways
        return False
    return sum(abs(x) for x in n1) + sum(abs(x) for x in n2) > 1e-9


def faces_up(normals) -> bool:
    """Whether this winding gives an upward surface.

    Vertical faces (kerbs) have no meaningful up direction; leave their winding
    alone rather than flipping on floating-point noise.
    """
    return sum(n[2] for n in normals) >= 0.0


def ribbon_to_mesh(ribbon: Ribbon) -> tuple[Mesh, int]:
    """Stitch a ribbon into a quad strip. Returns the mesh and skipped-quad count."""
    left, right = ribbon.left, ribbon.right
    n = min(len(left), len(right))
    if n < 2:
        return Mesh([], []), 0

    vertices: list[tuple[float, float, float]] = []
    for i in range(n):
        vertices.append(tuple(left[i]))
        vertices.append(tuple(right[i]))

    faces: list[list[int]] = []
    skipped = 0
    for i in range(n - 1):
        if not quad_is_sane(left[i], right[i], right[i + 1], left[i + 1]):
            # A hairpin in a painted line can still fold one quad; a bowtie face
            # renders as a black patch, and one dropped quad is invisible.
            skipped += 1
            continue
        a, b, c, d = i * 2, i * 2 + 1, i * 2 + 3, i * 2 + 2
        # Winding is decided per quad, not per ribbon: which side is 'left'
        # depends on the lanelet's direction of travel, and a short crosswalk can
        # change orientation partway along.
        upward = faces_up(quad_normals(left[i], right[i], right[i + 1], left[i + 1]))
        faces.append([a, b, c, d] if upward else [a, d, c, b])

    return Mesh(vertices, faces), skipped


def polygon_to_mesh(polygon: Polygon) -> tuple[Mesh, int]:
    """Fan-triangulate a small convex ring (a zebra bar)."""
    ring = list(polygon.points)
    if len(ring) > 2 and max(abs(a - b) for a, b in zip(ring[0], ring[-1])) < 1e-6:
        ring = ring[:-1]
    if len(ring) < 3:
        return Mesh([], []), 1

    normals = [normal(ring[0], ring[i], ring[i + 1]) for i in range(1, len(ring) - 1)]
    if sum(abs(n[0]) + abs(n[1]) + abs(n[2]) for n in normals) < 1e-9:
        return Mesh([], []), 1

    upward = faces_up(normals)
    faces = [[0, i, i + 1] if upward else [0, i + 1, i] for i in range(1, len(ring) - 1)]
    return Mesh([tuple(p) for p in ring], faces), 0


def merge_meshes(meshes: Sequence[Mesh]) -> Mesh:
    """Concatenate meshes, re-basing face indices and carrying UVs along."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    uvs: list[tuple[float, float]] = []
    authored = any(mesh.uvs for mesh in meshes)

    for mesh in meshes:
        offset = len(vertices)
        vertices.extend(mesh.vertices)
        faces.extend([[i + offset for i in face] for face in mesh.faces])
        if authored:
            uvs.extend(mesh.uvs or [(0.0, 0.0)] * len(mesh.vertices))
    return Mesh(vertices, faces, uvs if authored else None)


def signed_area_xy(ring: Sequence[Sequence[float]]) -> float:
    return 0.5 * sum(
        ring[i][0] * ring[(i + 1) % len(ring)][1] - ring[(i + 1) % len(ring)][0] * ring[i][1]
        for i in range(len(ring))
    )


def as_json(value: Any) -> Any:  # pragma: no cover - convenience for debugging dumps
    if isinstance(value, (Ribbon, Polygon)):
        return value.__dict__
    raise TypeError(type(value))
