"""Ground reconstruction tests. No map, no Blender."""

from __future__ import annotations

import numpy as np
import pytest

from city_builder import ground as gr
from city_builder.geometry import Ribbon
from city_builder.lanelet import build_adjacency


def _ribbon(rid, x0, x1, y, z, width=3.0):
    left = [(x0, y + width / 2, z), (x1, y + width / 2, z)]
    right = [(x0, y - width / 2, z), (x1, y - width / 2, z)]
    return Ribbon(rid, left, right)


def _flat_heightmap(value=0.0, nx=6, ny=6, cell=10.0):
    return gr.HeightMap(0.0, 0.0, cell, np.full((ny, nx), value), np.zeros((ny, nx)))


# --- height map ------------------------------------------------------------


def test_sample_is_exact_on_nodes_and_linear_between():
    hm = gr.HeightMap(0.0, 0.0, 10.0, np.array([[0.0, 10.0], [0.0, 10.0]]), np.zeros((2, 2)))
    assert hm.sample(0, 0) == pytest.approx(0.0)
    assert hm.sample(10, 0) == pytest.approx(10.0)
    assert hm.sample(5, 5) == pytest.approx(5.0)


def test_sample_clamps_outside_the_grid():
    hm = gr.HeightMap(0.0, 0.0, 10.0, np.array([[1.0, 2.0], [3.0, 4.0]]), np.zeros((2, 2)))
    assert hm.sample(-500, -500) == pytest.approx(1.0)
    assert hm.sample(500, 500) == pytest.approx(4.0)


def test_heightmap_json_round_trip():
    hm = gr.HeightMap(3.0, -7.0, 5.0, np.array([[1.0, 2.0], [3.0, 4.0]]), np.array([[0.0, 1.0], [2.0, 3.0]]))
    back = gr.HeightMap.from_json(hm.to_json())
    assert (back.x0, back.y0, back.cell) == (3.0, -7.0, 5.0)
    assert back.z == pytest.approx(hm.z)


def _binned_minimum(points, hm):
    lowest: dict[tuple[int, int], float] = {}
    for x, y, z in points:
        key = (int((x - hm.x0) / hm.cell), int((y - hm.y0) / hm.cell))
        lowest[key] = min(lowest.get(key, z), z)
    return lowest


def test_heightmap_nodes_stay_under_the_samples_binned_to_them():
    """The surface is a lower envelope, not a fit through the middle.

    A fit cuts through the road it was built from — measured on a real map, a
    median fit buried 28 % of the network.
    """
    rng = np.random.default_rng(0)
    points = [(float(x), float(y), 5.0 + rng.normal(0, 0.3))
              for x in range(0, 101, 4) for y in range(0, 101, 4)]
    hm = gr.build_heightmap(points, cell=10.0, smooth=0.5, margin=0.0, drop=0.05)
    assert hm is not None

    for (ix, iy), lowest in _binned_minimum(points, hm).items():
        assert hm.z[iy, ix] <= lowest - 0.05 + 1e-6


def test_bilinear_sampling_can_still_exceed_a_nearby_sample():
    """Why the ground mesh is clipped to the road outline rather than trusted.

    The clamp binds grid *nodes*; a sample near a cell corner is interpolated
    from four of them, and a higher neighbour can lift the result above it. No
    grid resolution removes this, so the mesh cuts the carriageway out instead.
    """
    rng = np.random.default_rng(0)
    points = [(float(x), float(y), 5.0 + rng.normal(0, 0.3))
              for x in range(0, 101, 4) for y in range(0, 101, 4)]
    hm = gr.build_heightmap(points, cell=10.0, smooth=0.5, margin=0.0, drop=0.05)
    assert any(hm.sample(x, y) > z for x, y, z in points)


def test_heightmap_fills_gaps_between_streets():
    points = [(x, 0.0, 0.0) for x in range(0, 101, 5)] + [(x, 100.0, 10.0) for x in range(0, 101, 5)]
    hm = gr.build_heightmap(points, cell=10.0, smooth=0.0, margin=0.0, drop=0.0)
    assert 0.0 < hm.sample(50, 50) < 10.0
    assert hm.support.max() > 0


def test_heightmap_rejects_too_few_points():
    assert gr.build_heightmap([(0, 0, 0)], cell=10.0) is None


# --- elevated structure ----------------------------------------------------


def test_find_stacked_seeds_the_upper_of_two_crossing_lanelets():
    lower = _ribbon(1, -20, 20, 0, 0.0, width=8.0)
    upper = Ribbon(2, [(0, -20, 9.0), (0, 20, 9.0)], [(4, -20, 9.0), (4, 20, 9.0)])
    assert gr.find_stacked([gr.footprint_of(lower), gr.footprint_of(upper)]) == {2}


def test_find_stacked_ignores_lanelets_at_the_same_level():
    a = _ribbon(1, -20, 20, 0, 0.0, width=8.0)
    b = Ribbon(2, [(0, -20, 0.1), (0, 20, 0.1)], [(4, -20, 0.1), (4, 20, 0.1)])
    assert gr.find_stacked([gr.footprint_of(a), gr.footprint_of(b)]) == set()


def test_grow_elevated_walks_down_the_ramp_and_stops_at_the_street():
    """A ramp overlaps nothing, so only connectivity can reach it."""
    street = [_ribbon(100 + i, i * 40, (i + 1) * 40, 60, 0.0) for i in range(8)]
    deck = _ribbon(1, 0, 40, 0, 9.0)
    ramp = [_ribbon(2 + i, 40 + i * 40, 80 + i * 40, 0, 9.0 - 3.0 * i) for i in range(4)]

    footprints = [gr.footprint_of(r) for r in (deck, *ramp, *street)]
    adjacency: dict[int, set[int]] = {}
    for a, b in zip([1, 2, 3, 4, 5], [2, 3, 4, 5, 100]):
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)

    elevated = gr.grow_elevated({1}, footprints, adjacency, clearance=1.5)
    assert {1, 2, 3} <= elevated
    assert 5 not in elevated
    assert not any(sid in elevated for sid in range(100, 108))


def test_build_adjacency_links_lanelets_sharing_an_end():
    adjacency = build_adjacency([
        (1, frozenset({10, 11}), frozenset({12, 13})),
        (2, frozenset({12, 13}), frozenset({14, 15})),
        (3, frozenset({90, 91}), frozenset({92, 93})),
    ])
    assert adjacency[1] == {2} and adjacency[2] == {1}
    assert 3 not in adjacency


# --- ground mesh -----------------------------------------------------------


def test_ground_mesh_leaves_a_hole_where_the_road_is():
    road = _ribbon(1, -5.0, 55.0, 25.0, 1.0, width=8.0)
    mesh = gr.build_mesh(_flat_heightmap(), [road], elevated=set())
    verts = np.array(mesh.vertices)
    assert mesh.faces
    inside = (verts[:, 0] > 5) & (verts[:, 0] < 45) & (np.abs(verts[:, 1] - 25.0) < 3.0)
    assert not inside.any(), "the carriageway must be cut out of the ground"


def test_ground_mesh_meets_the_road_edge_at_its_height():
    road = _ribbon(1, -5.0, 55.0, 25.0, 1.0, width=8.0)
    mesh = gr.build_mesh(_flat_heightmap(), [road], elevated=set(), drop=0.0)
    verts = np.array(mesh.vertices)

    on_edge = np.abs(np.abs(verts[:, 1] - 25.0) - 4.0) < 0.01
    assert on_edge.any()
    assert verts[on_edge, 2] == pytest.approx(1.0, abs=1e-6)

    far = np.abs(verts[:, 1] - 25.0) > 15.0
    assert verts[far, 2] == pytest.approx(0.0, abs=1e-6)


def test_ground_mesh_holds_the_seam_below_the_carriageway():
    """The clip only places a seam vertex where the outline crosses a cell edge.

    A seam segment can therefore be metres long, and its linear interpolation
    rides above a sloping carriageway in between — 3 cm on a real map. The drop
    absorbs that, and a few centimetres at the kerb is what a kerb looks like.
    """
    road = _ribbon(1, -5.0, 55.0, 25.0, 1.0, width=8.0)
    mesh = gr.build_mesh(_flat_heightmap(), [road], elevated=set(), drop=0.05)
    verts = np.array(mesh.vertices)
    on_edge = np.abs(np.abs(verts[:, 1] - 25.0) - 4.0) < 0.01
    assert verts[on_edge, 2] == pytest.approx(0.95, abs=1e-6)


def test_ground_mesh_runs_under_a_viaduct():
    deck = _ribbon(7, -5.0, 55.0, 25.0, 9.0, width=8.0)
    hm = _flat_heightmap()
    mesh = gr.build_mesh(hm, [deck], elevated={7})
    assert len(mesh.faces) == (hm.nx - 1) * (hm.ny - 1), "nothing is cut away"


def test_ground_mesh_faces_are_upward_and_non_degenerate():
    road = _ribbon(1, -5.0, 55.0, 25.0, 0.5, width=6.0)
    mesh = gr.build_mesh(_flat_heightmap(), [road], elevated=set())
    verts = np.array(mesh.vertices)
    for face in mesh.faces:
        pts = verts[face]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        assert np.linalg.norm(n) > 1e-9
        assert n[2] > 0


# ---------------------------------------------------------------------------
# Which energy the gaps are filled with
#
# The block interiors have no samples in them, so what fills them is entirely
# decided by the operator. Two properties separate the two choices, and both
# are visible in the result rather than only in the theory.
# ---------------------------------------------------------------------------


def _two_streets(z_low=0.0, z_high=0.0, span=90.0, step=6.0):
    """Samples along two parallel streets with an empty block between them."""
    points = []
    for x in np.arange(0.0, span + step, step):
        points.append((x, 0.0, z_low))
        points.append((x, span, z_high))
    return points


def _slope(step=6.0, span=90.0, rise=3.0):
    """Two streets, one higher than the other, with ground beyond both."""
    points = []
    for x in np.arange(0.0, span + step, step):
        points.append((x, 30.0, 0.0))
        points.append((x, 60.0, rise))
    return points


def test_a_harmonic_fill_is_trapped_inside_the_range_of_its_samples():
    """The maximum principle, which is why order 1 is not the default.

    A harmonic function never leaves the range of its constraints. Ground
    beyond the last street therefore goes *flat* at that street's height
    however steadily the streets were climbing, and no block interior can rise
    above the roads around it — not unlikely under this operator, impossible.
    """
    hm = gr.build_heightmap(_slope(), cell=10.0, margin=30.0, smooth=0.0, drop=0.0,
                            order=1)
    assert hm.z.max() <= 3.0 + 1e-6 and hm.z.min() >= -1e-6


def test_a_thin_plate_carries_the_grade_past_the_last_street():
    """Same samples: the hill goes on rising instead of stopping at the kerb."""
    hm = gr.build_heightmap(_slope(), cell=10.0, margin=30.0, smooth=0.0, drop=0.0,
                            order=2)
    assert hm.z.max() > 3.5, "the thin plate went flat at the last sample"
    assert hm.z.min() < -0.5


def test_both_orders_still_honour_the_samples_they_were_given():
    points = _two_streets(z_low=0.0, z_high=4.0)
    for order in (1, 2):
        hm = gr.build_heightmap(points, cell=10.0, margin=0.0, smooth=0.0, drop=0.0,
                                order=order)
        for x, y, z in points:
            assert hm.sample(x, y) == pytest.approx(z, abs=0.35)


def test_the_solve_is_not_iterative_and_converges_exactly():
    """400 Jacobi sweeps left a residual; a direct solve does not."""
    points = _two_streets(z_low=0.0, z_high=5.0)
    hm = gr.build_heightmap(points, cell=10.0, margin=0.0, smooth=0.0, drop=0.0, order=1)
    z, known = hm.z, hm.support == 0
    padded = np.pad(z, 1, mode="edge")
    mean = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.25
    assert np.abs((mean - z)[~known]).max() < 1e-8


# ---------------------------------------------------------------------------
# Taking the shape of an elevation model without taking its heights
# ---------------------------------------------------------------------------


def test_guidance_puts_the_models_relief_into_the_empty_block():
    points = _two_streets(z_low=0.0, z_high=0.0)
    plain = gr.build_heightmap(points, cell=10.0, margin=0.0, smooth=0.0, drop=0.0)
    # A model with a mound in the middle, and an absurd datum: neither its
    # offset nor its overall tilt should reach the answer.
    ys, xs = np.mgrid[0:plain.ny, 0:plain.nx]
    mound = 2.0 * np.exp(-(((xs - plain.nx / 2) ** 2 + (ys - plain.ny / 2) ** 2) / 6.0))
    model = mound + 137.0 + 0.4 * xs

    guided = gr.build_heightmap(points, cell=10.0, margin=0.0, smooth=0.0, drop=0.0,
                                guidance=model)
    middle = (plain.ny // 2, plain.nx // 2)
    assert guided.z[middle] > plain.z[middle] + 0.5, "the mound did not come through"
    assert abs(guided.z[middle]) < 10.0, "the model's datum came through with it"


def test_guidance_does_not_move_the_measured_cells():
    points = _two_streets(z_low=0.0, z_high=3.0)
    ys, xs = np.mgrid[0:20, 0:20]
    model = 55.0 + 3.0 * np.sin(xs / 2.0) + 2.0 * np.cos(ys / 3.0)
    plain = gr.build_heightmap(points, cell=10.0, margin=0.0, smooth=0.0, drop=0.0)
    model = model[: plain.ny, : plain.nx]
    guided = gr.build_heightmap(points, cell=10.0, margin=0.0, smooth=0.0, drop=0.0,
                                guidance=model)
    measured = plain.support == 0
    assert np.abs(guided.z[measured] - plain.z[measured]).max() < 1e-6


def test_the_grid_is_handed_to_a_guidance_source_that_has_to_fetch_it():
    """A model has to be fetched for an extent nobody knows until the solve."""
    asked = {}

    def source(x0, y0, nx, ny, cell):
        asked.update(x0=x0, y0=y0, nx=nx, ny=ny, cell=cell)

    hm = gr.build_heightmap(_two_streets(), cell=10.0, margin=0.0, smooth=0.0,
                            guidance=source)
    assert asked["nx"] == hm.nx and asked["ny"] == hm.ny
    assert asked["cell"] == 10.0 and asked["x0"] == hm.x0
