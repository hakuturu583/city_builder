"""Geometry tests. No map, no Blender."""

from __future__ import annotations

import math

import numpy as np
import pytest

from city_builder import geometry as g
from city_builder.frame import EARTH_RADIUS_M, LocalFrame


def _ribbon(rid, x0, x1, y, z, width=3.0):
    left = [(x0, y + width / 2, z), (x1, y + width / 2, z)]
    right = [(x0, y - width / 2, z), (x1, y - width / 2, z)]
    return g.Ribbon(rid, left, right)


# --- frame -----------------------------------------------------------------


def test_frame_is_zero_at_its_anchor():
    assert LocalFrame(35.0, 139.0).to_local(35.0, 139.0) == (0.0, 0.0)


def test_frame_scales_longitude_by_latitude():
    frame = LocalFrame(35.0, 139.0)
    x, y = frame.to_local(35.0, 139.001)
    assert x == pytest.approx(EARTH_RADIUS_M * math.radians(0.001) * math.cos(math.radians(35.0)))
    assert y == pytest.approx(0.0)
    assert frame.to_local(35.001, 139.0)[1] == pytest.approx(111.19, abs=0.1)


def test_frame_round_trips():
    frame = LocalFrame(35.69, 139.69)
    lat, lon = frame.to_wgs84(*frame.to_local(35.6912, 139.6875))
    assert (lat, lon) == pytest.approx((35.6912, 139.6875), abs=1e-9)


# --- polylines -------------------------------------------------------------


def test_resample_preserves_endpoints():
    out = g.resample_polyline([(0, 0, 10), (10, 0, 12), (20, 0, 11)], 5)
    assert len(out) == 5
    assert tuple(out[0]) == pytest.approx((0, 0, 10))
    assert tuple(out[-1]) == pytest.approx((20, 0, 11))
    assert list(out[:, 0]) == sorted(out[:, 0])


def test_resample_is_uniform_in_3d_arc_length():
    out = g.resample_polyline([(0, 0, 0), (30, 0, 40)], 6)  # 50 m of slope
    steps = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert steps == pytest.approx([10.0] * 5)


def test_pair_bounds_equalises_unequal_vertex_counts():
    ls, rs = g.pair_bounds([(0, 1, 0), (5, 1, 1), (10, 1, 2)], [(0, -1, 0), (10, -1, 2)], max_segment=0)
    assert len(ls) == len(rs) == 3
    assert rs[1][0] == pytest.approx(5.0)


def test_pair_bounds_densifies_long_lanelets():
    ls, rs = g.pair_bounds([(0, 1, 0), (100, 1, 0)], [(0, -1, 0), (100, -1, 0)], max_segment=5.0)
    assert len(ls) == len(rs) >= 21


def test_pair_bounds_rejects_degenerate_bounds():
    assert g.pair_bounds([(0, 0, 0)], [(0, 1, 0), (1, 1, 0)]) is None


def test_monotone_pairing_survives_a_bound_that_starts_backwards():
    """Regression: real crosswalk bounds begin with a stub pointing back.

    Arc-length pairing then advances the two sides in opposite directions and
    the quad between them is twisted.
    """
    left = [(0.0, 2.0, 0.0), (10.0, 2.0, 0.0)]
    right = [(0.0, 0.0, 0.0), (-4.0, 0.0, 0.0), (10.0, 0.0, 0.0)]  # 4 m stub backwards
    ls, rs = g.pair_bounds(left, right, max_segment=0)
    for i in range(len(ls) - 1):
        a, b, c, d = ls[i], rs[i], rs[i + 1], ls[i + 1]
        n1, n2 = g.quad_normals(a, b, c, d)
        assert sum(x * y for x, y in zip(n1, n2)) >= 0, "quad must not be twisted"


# --- ribbons ---------------------------------------------------------------


def test_ribbon_from_polyline_has_the_width_and_keeps_z():
    left, right = g.ribbon_from_polyline([(0, 0, 5.0), (10, 0, 5.5)], 0.4)
    for i in range(2):
        assert math.dist(left[i][:2], right[i][:2]) == pytest.approx(0.4)
    assert [p[2] for p in left] == pytest.approx([5.0, 5.5])  # follows the slope


def test_ribbon_from_polyline_rejects_a_single_point():
    assert g.ribbon_from_polyline([(0, 0, 0)], 0.2) is None


# --- dashes ----------------------------------------------------------------


def test_split_dashes_produces_gapped_segments():
    dashes = g.split_dashes([(0, 0, 0), (24, 0, 0)], 3.0, 5.0)
    assert len(dashes) == 3
    for dash in dashes:
        assert float(np.linalg.norm(dash[-1][:2] - dash[0][:2])) == pytest.approx(3.0)
    assert dashes[1][0][0] == pytest.approx(8.0)


def test_split_dashes_keeps_short_lines_whole():
    assert len(g.split_dashes([(0, 0, 0), (2, 0, 0)], 3.0, 5.0)) == 1


def test_split_dashes_interpolates_elevation():
    dashes = g.split_dashes([(0, 0, 0.0), (16, 0, 16.0)], 4.0, 4.0)
    assert dashes[1][0][2] == pytest.approx(8.0)


# --- kerbs and rings -------------------------------------------------------


def test_extrude_curb_is_vertical_and_follows_the_slope():
    base, top = g.extrude_curb([(0, 0, 10.0), (10, 0, 10.5)], 0.15)
    assert [p[2] for p in top] == pytest.approx([10.15, 10.65])
    for b, t in zip(base, top):
        assert b[:2] == pytest.approx(t[:2])


def test_close_ring_drops_the_repeated_vertex():
    ring = g.close_ring([(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)])
    assert len(ring) == 4


def test_close_ring_rejects_degenerate_rings():
    assert g.close_ring([(0, 0, 0), (1, 0, 0), (0, 0, 0)]) is None


# --- faces -----------------------------------------------------------------


def test_quad_sanity_accepts_a_vertical_kerb_face():
    """Regression: an XY-area test discards every kerb face."""
    assert g.quad_is_sane((0, 0, 0), (10, 0, 0), (10, 0, 0.15), (0, 0, 0.15))


def test_quad_sanity_rejects_a_bowtie():
    assert not g.quad_is_sane((0, 1, 0), (0, -1, 0), (10, 5, 0), (10, 1, 0))


def test_ribbon_mesh_faces_up_for_either_bound_order():
    """Regression: the first implementation emitted downward normals."""
    a = _ribbon(1, 0, 10, 0, 0.0)
    b = g.Ribbon(2, a.right, a.left)
    for ribbon in (a, b):
        mesh, skipped = g.ribbon_to_mesh(ribbon)
        assert skipped == 0 and len(mesh.faces) == 1
        p0, p1, p2 = (np.array(mesh.vertices[i]) for i in mesh.faces[0][:3])
        assert np.cross(p1 - p0, p2 - p0)[2] > 0


def test_ribbon_mesh_decides_winding_per_quad():
    """Regression: a strip that changes orientation partway had black faces."""
    left = [(0.0, 1.0, 0.0), (10.0, 1.0, 0.0), (10.0, -3.0, 0.0), (0.0, -3.0, 0.0)]
    right = [(0.0, -1.0, 0.0), (10.0, -1.0, 0.0), (12.0, -3.0, 0.0), (0.0, -5.0, 0.0)]
    mesh, _ = g.ribbon_to_mesh(g.Ribbon(1, left, right))
    assert mesh.faces
    for face in mesh.faces:
        p0, p1, p2 = (np.array(mesh.vertices[i]) for i in face[:3])
        assert np.cross(p1 - p0, p2 - p0)[2] > 0


def test_kerb_ribbon_keeps_its_faces():
    base = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.5)]
    top = [(0.0, 0.0, 0.15), (10.0, 0.0, 0.65)]
    mesh, skipped = g.ribbon_to_mesh(g.Ribbon(1, base, top))
    assert skipped == 0 and len(mesh.faces) == 1


def test_zebra_ring_fans_into_upward_triangles():
    ring = [(0.0, 0.0, 3.0), (4.0, 0.0, 3.0), (4.0, 0.5, 3.0), (0.0, 0.5, 3.0), (0.0, 0.0, 3.0)]
    mesh, skipped = g.polygon_to_mesh(g.Polygon(1, ring))
    assert skipped == 0
    assert len(mesh.vertices) == 4 and len(mesh.faces) == 2
    for face in mesh.faces:
        p0, p1, p2 = (np.array(mesh.vertices[i]) for i in face)
        assert np.cross(p1 - p0, p2 - p0)[2] > 0


def test_zebra_ring_reversed_still_faces_up():
    ring = [(0.0, 0.5, 3.0), (4.0, 0.5, 3.0), (4.0, 0.0, 3.0), (0.0, 0.0, 3.0)]
    mesh, _ = g.polygon_to_mesh(g.Polygon(1, ring))
    for face in mesh.faces:
        p0, p1, p2 = (np.array(mesh.vertices[i]) for i in face)
        assert np.cross(p1 - p0, p2 - p0)[2] > 0


def test_merge_meshes_rebases_indices():
    a = g.Mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [[0, 1, 2]])
    b = g.Mesh([(5, 5, 0), (6, 5, 0), (6, 6, 0)], [[0, 1, 2]])
    merged = g.merge_meshes([a, b])
    assert len(merged.vertices) == 6
    assert merged.faces == [[0, 1, 2], [3, 4, 5]]
