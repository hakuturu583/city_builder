"""Running the roads off the edge of the map. Geometry only — no map, no Blender."""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import extend as E
from city_builder.extend import ExtendOptions


def _bounds(x0, x1, y0=0.0, y1=3.0, *, z=0.0, rise=0.0, n=5):
    """Two straight bounds from x0 to x1, level or on a constant grade."""
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    zs = [z + rise * (x - x0) for x in xs]
    return ([(x, y0, h) for x, h in zip(xs, zs)], [(x, y1, h) for x, h in zip(xs, zs)])


def _stub(x=40.0, direction=(1.0, 0.0), grade=0.0, y0=0.0, y1=3.0, z=0.0):
    return E.Stub(1, (x, y0, z), (x, y1, z), direction, grade)


# ---------------------------------------------------------------------------
# Which ends are loose
# ---------------------------------------------------------------------------


def test_a_chain_is_loose_only_at_the_two_far_ends():
    a, b, c, d = frozenset({1, 2}), frozenset({3, 4}), frozenset({5, 6}), frozenset({7, 8})
    ends = [(1, a, b), (2, b, c), (3, c, d)]
    assert E.dangling_ends(ends) == {(1, "start"), (3, "end")}


def test_a_fork_with_nothing_feeding_it_is_loose_at_both_branches():
    # Two lanelets starting on the same pair is not connectivity: if nothing
    # ends there, the fork is as cut off as a single lane would be.
    shared, left, right = frozenset({1, 2}), frozenset({3, 4}), frozenset({5, 6})
    ends = [(1, shared, left), (2, shared, right)]
    loose = E.dangling_ends(ends)
    assert (1, "start") in loose
    assert (2, "start") in loose


def test_a_merge_feeding_a_successor_is_attached():
    join, out = frozenset({9, 10}), frozenset({11, 12})
    ends = [(1, frozenset({1, 2}), join), (2, frozenset({3, 4}), join), (3, join, out)]
    loose = E.dangling_ends(ends)
    assert (1, "end") not in loose
    assert (2, "end") not in loose


# ---------------------------------------------------------------------------
# Where a stub points
# ---------------------------------------------------------------------------


def test_the_heading_is_taken_over_a_window_not_the_last_segment():
    # A boundary that runs east for twenty metres and finishes with a two
    # centimetre survey wobble to the north. The last segment says north.
    bound = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0), (20.0, 0.0, 0.0), (20.0, 0.02, 0.0)]
    direction, _grade = E._heading(bound, window=15.0)
    assert direction[0] == pytest.approx(1.0, abs=0.01)
    assert abs(direction[1]) < 0.01


def test_the_grade_comes_from_the_window_and_is_capped():
    left, right = _bounds(0.0, 40.0, rise=0.2)
    stub = E.stub_from_bounds(1, left, right, ExtendOptions(max_grade=0.06))
    assert stub.grade == pytest.approx(0.06)
    assert stub.direction[0] == pytest.approx(1.0, abs=1e-6)


def test_a_stub_keeps_the_lane_width_as_it_goes():
    stub = _stub()
    far_left, far_right = stub.outer(25.0)
    assert far_left[0] == pytest.approx(65.0)
    assert math.dist(far_left[:2], far_right[:2]) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# How far it may go
# ---------------------------------------------------------------------------


def test_the_edge_is_the_nearest_side_along_the_heading():
    box = (-100.0, -100.0, 100.0, 100.0)
    assert E.to_edge(_stub(x=40.0, direction=(1.0, 0.0)), box) == pytest.approx(60.0)
    assert E.to_edge(_stub(x=40.0, direction=(-1.0, 0.0)), box) == pytest.approx(140.0)


def test_a_road_across_the_path_stops_the_extension_short_of_it():
    crossing = ShapelyPolygon([(60, -20), (70, -20), (70, 20), (60, 20)])
    assert E.blocked_at(_stub(), 100.0, [crossing]) == pytest.approx(20.0)


def test_touching_a_neighbour_along_the_side_does_not_stop_anything():
    # The lane that ends beside its neighbour shares an edge with it and no
    # area. Counting that as a collision would rule out every multi-lane road.
    beside = ShapelyPolygon([(0, 3), (40, 3), (40, 6), (0, 6)])
    assert math.isinf(E.blocked_at(_stub(), 100.0, [beside]))


def test_a_clear_run_is_not_docked_the_clearance_it_owes_nobody():
    box = (-100.0, -100.0, 100.0, 100.0)
    options = ExtendOptions(clearance=1.0)
    assert E.reach(_stub(), box, None, options) == pytest.approx(60.0)


def test_a_blocked_run_stops_a_clearance_short():
    box = (-100.0, -100.0, 100.0, 100.0)
    crossing = ShapelyPolygon([(60, -20), (70, -20), (70, 20), (60, 20)])
    length = E.reach(_stub(), box, [crossing], ExtendOptions(clearance=1.5))
    assert length == pytest.approx(18.5)


def test_a_stub_facing_a_wall_of_road_is_left_alone():
    box = (-100.0, -100.0, 100.0, 100.0)
    against = ShapelyPolygon([(41, -20), (60, -20), (60, 20), (41, 20)])
    assert E.reach(_stub(), box, [against], ExtendOptions(min_length=2.0)) == 0.0


def test_a_dead_end_in_the_middle_of_the_city_is_not_an_edge():
    # Both are loose in the graph. Only one has open country in front of it,
    # and running the other to the far corner would draw a lane-wide scratch
    # across half a kilometre of blocks.
    covered = ShapelyPolygon([(-30, -30), (500, -30), (500, 500), (-30, 500)])
    assert E.leaves_within(_stub(x=480.0), covered, 60.0)
    assert not E.leaves_within(_stub(x=100.0), covered, 60.0)


# ---------------------------------------------------------------------------
# What gets added
# ---------------------------------------------------------------------------


def test_the_extension_is_sampled_along_its_length_not_just_at_the_end():
    # The terrain grid bins road samples: one cross-section thirty metres out
    # leaves every cell in between with nothing holding it up.
    left, right = E.bound_points(_stub(), 30.0, ExtendOptions(step=5.0))
    assert len(left) == len(right) == 6
    assert [round(p[0], 1) for p in left] == [45.0, 50.0, 55.0, 60.0, 65.0, 70.0]


def test_the_extension_carries_the_grade():
    left, _right = E.bound_points(_stub(grade=0.05), 20.0, ExtendOptions(step=20.0))
    assert left[-1][2] == pytest.approx(1.0)


def test_nothing_is_added_for_a_zero_length():
    assert E.bound_points(_stub(), 0.0, ExtendOptions()) == ([], [])


# ---------------------------------------------------------------------------
# The whole map
# ---------------------------------------------------------------------------


def _chain():
    """Two lanelets end to end, sharing the boundary points between them."""
    bounds = {1: _bounds(0.0, 40.0), 2: _bounds(40.0, 80.0)}
    ends = [
        (1, frozenset({1, 3}), frozenset({2, 4})),
        (2, frozenset({2, 4}), frozenset({5, 6})),
    ]
    points = {
        (1, "start"): (1, 3), (1, "end"): (2, 4),
        (2, "start"): (2, 4), (2, "end"): (5, 6),
    }
    return bounds, ends, points


def test_a_chain_grows_at_both_ends_and_nowhere_in_the_middle():
    plan = E.plan(*_chain(), ExtendOptions(margin=30.0))
    assert plan.stats["extended"] == 2
    assert set(plan.points) == {1, 3, 5, 6}  # the two outer pairs, not the join
    assert plan.points[5][-1][0] == pytest.approx(110.0)  # out to the box
    assert plan.points[1][-1][0] == pytest.approx(-30.0)


def test_the_box_is_fixed_before_anything_moves():
    plan = E.plan(*_chain(), ExtendOptions(margin=30.0))
    assert plan.box == pytest.approx((-30.0, -30.0, 110.0, 33.0))
    # …and the roads now reach it, which is the point: taking a fresh margin
    # around the extended roads would leave another ring of nothing beyond them.
    assert plan.points[5][-1][0] == pytest.approx(plan.box[2])


def test_a_point_an_attached_end_sits_on_is_never_moved():
    bounds, ends, points = _chain()
    # Pretend the join is also lanelet 3's start, so it must not be dragged.
    ends.append((3, frozenset({2, 4}), frozenset({7, 8})))
    points[(3, "start")] = (2, 4)
    points[(3, "end")] = (7, 8)
    bounds[3] = _bounds(40.0, 80.0, y0=-3.0, y1=0.0)
    plan = E.plan(bounds, ends, points, ExtendOptions())
    assert 2 not in plan.points
    assert 4 not in plan.points


def test_disabled_still_reports_the_edge_it_would_have_used():
    plan = E.plan(*_chain(), ExtendOptions(enabled=False))
    assert plan.points == {}
    assert plan.box[2] == pytest.approx(110.0)


def test_a_polyline_is_continued_at_whichever_end_it_meets_the_extension():
    plan = E.Plan((0, 0, 1, 1), {7: [(1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]}, {})
    line = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    assert plan.extended(line, 99, 7)[-1][0] == pytest.approx(2.0)
    # Run the other way and the same continuation goes on the front, reversed,
    # so the polyline still reads from one end to the other.
    reversed_line = list(reversed(line))
    assert plan.extended(reversed_line, 7, 99)[0][0] == pytest.approx(2.0)
    assert plan.extended(reversed_line, 7, 99)[1][0] == pytest.approx(1.0)


def test_a_polyline_touching_nothing_comes_back_unchanged():
    plan = E.Plan((0, 0, 1, 1), {7: [(1.0, 0.0, 0.0)]}, {})
    line = [(0.0, 0.0, 0.0), (0.5, 0.0, 0.0)]
    assert plan.extended(line, 1, 2) == line
