"""The drivable route: geometry only, no Blender."""

from __future__ import annotations

import math
from itertools import pairwise

from city_builder import route
from city_builder.geometry import Ribbon


def _straight(lane_id, x0, x1, y=0.0, z=0.0, n=5):
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    return Ribbon(lane_id, [(x, y - 1.5, z) for x in xs], [(x, y + 1.5, z) for x in xs])


def test_centreline_is_the_middle_of_the_lane():
    line = route.centreline(_straight(1, 0, 10, y=4.0, z=2.0))
    assert all(abs(p[1] - 4.0) < 1e-9 and abs(p[2] - 2.0) < 1e-9 for p in line)


def test_successors_follow_the_direction_of_travel():
    """The undirected version would let a route run backwards up a one-way street."""
    ends = [
        (1, frozenset({10, 11}), frozenset({20, 21})),
        (2, frozenset({20, 21}), frozenset({30, 31})),   # follows 1
        (3, frozenset({40, 41}), frozenset({20, 21})),   # also *ends* where 1 does
    ]
    succ = route.successors(ends)
    assert succ[1] == [2]
    assert succ[3] == [2]
    assert succ[2] == []


def test_the_search_finds_the_longer_of_two_chains():
    lines = {i: route.centreline(_straight(i, i * 10, i * 10 + 10)) for i in range(1, 5)}
    ends = [
        (1, frozenset({1, 2}), frozenset({3, 4})),
        (2, frozenset({3, 4}), frozenset({5, 6})),
        (3, frozenset({5, 6}), frozenset({7, 8})),
        (4, frozenset({90, 91}), frozenset({92, 93})),   # an isolated stub
    ]
    best = route.longest_route(route.successors(ends), lines, attempts=50, seed=1)
    assert best == [1, 2, 3]


def test_a_route_is_deterministic_for_a_seed():
    lines = {i: route.centreline(_straight(i, i * 10, i * 10 + 10)) for i in range(1, 4)}
    ends = [(1, frozenset({1, 2}), frozenset({3, 4})),
            (2, frozenset({3, 4}), frozenset({5, 6})),
            (3, frozenset({5, 6}), frozenset({7, 8}))]
    succ = route.successors(ends)
    assert (route.longest_route(succ, lines, seed=5)
            == route.longest_route(succ, lines, seed=5))


def test_the_join_between_lanelets_is_not_doubled():
    lines = {1: [(0, 0, 0), (10, 0, 0)], 2: [(10, 0, 0), (20, 0, 0)]}
    assert route.route_polyline([1, 2], lines) == [(0, 0, 0), (10, 0, 0), (20, 0, 0)]


def test_resampling_makes_the_speed_constant():
    """Survey vertices are dense on curves; one per frame would race the straights."""
    uneven = [(0, 0, 0), (1, 0, 0), (1.5, 0, 0), (20, 0, 0)]
    even = route.resample(uneven, 0.5)
    gaps = [math.dist(a[:2], b[:2]) for a, b in pairwise(even)]
    assert max(gaps) - min(gaps) < 1e-6
    assert abs(gaps[0] - 0.5) < 1e-9


def test_resampling_follows_the_climb():
    climbed = route.resample([(0, 0, 0), (10, 0, 5)], 1.0)
    assert climbed[-1][2] > 4.0


def test_the_camera_sits_above_the_road_and_looks_along_it():
    points = [(x, 0.0, 0.0) for x in range(60)]
    path = route.camera_path(points, eye_height=1.4, look_ahead=18.0, step=0.5)
    assert path
    for position, target in path:
        assert abs(position[2] - 1.4) < 1e-6
        assert target[0] > position[0]  # aimed down the road, not at its feet


def test_the_camera_aims_ahead_rather_than_at_the_next_sample():
    """Aiming one sample ahead makes it yaw at every wobble in the centreline."""
    points = [(x, 0.0, 0.0) for x in range(80)]
    near = route.camera_path(points, look_ahead=1.0, step=0.5)
    far = route.camera_path(points, look_ahead=20.0, step=0.5)
    assert (far[0][1][0] - far[0][0][0]) > 10 * (near[0][1][0] - near[0][0][0])


def test_smoothing_keeps_the_ends_where_they_were():
    points = [(float(i), 0.0, 0.0) for i in range(20)]
    smoothed = route.smooth(points, 5)
    assert len(smoothed) == len(points)
    assert smoothed[0][0] < smoothed[-1][0]


def test_drive_path_crosses_junctions():
    """A route that refused to enter an intersection would stop at the first corner."""
    groups = {"Roads": [_straight(1, 0, 10), _straight(3, 20, 30)],
              "Junctions": [_straight(2, 10, 20)]}
    ends = [(1, frozenset({1, 2}), frozenset({3, 4})),
            (2, frozenset({3, 4}), frozenset({5, 6})),
            (3, frozenset({5, 6}), frozenset({7, 8}))]
    path = route.drive_path(groups, ends, step=0.5)
    assert path
    assert max(p[0][0] for p in path) > 20.0  # got past the junction


# ---------------------------------------------------------------------------
# Driving all of it, rather than the longest part of it


def _tee():
    """A crossbar with a stem hanging off the middle of it.

    ids 1,2 run along the bar; 3 turns off it; 4 runs down the stem.
    """
    lines = {
        1: [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)],
        2: [(10.0, 0.0, 0.0), (20.0, 0.0, 0.0)],
        3: [(10.0, 0.0, 0.0), (10.0, -5.0, 0.0)],
        4: [(10.0, -5.0, 0.0), (10.0, -15.0, 0.0)],
    }
    succ = {1: [2, 3], 2: [], 3: [4], 4: []}
    return succ, lines


def test_the_longest_route_misses_the_stem():
    # The thing that makes covering necessary: a T's longest drive is its bar.
    walk = route.longest_route(*_tee())
    assert set(walk) != {1, 2, 3, 4}


def test_covering_drives_every_lanelet():
    succ, lines = _tee()
    driven = set()
    for walk in route.routes_covering(succ, lines, least=1.0):
        driven |= set(walk)
    assert driven == set(lines)


def test_covering_takes_the_biggest_gain_first():
    succ, lines = _tee()
    walks = route.routes_covering(succ, lines, least=1.0)
    # 1-3-4 is twenty metres of new road; 1-2 is twenty as well but the walk
    # that reaches the stem is the one that has to happen at all.
    assert len(walks) >= 2
    assert len(walks[0]) >= len(walks[-1])


def test_a_spur_too_short_to_drive_is_left_alone():
    succ = {1: [2], 2: []}
    lines = {1: [(0.0, 0.0, 0.0), (30.0, 0.0, 0.0)],
             2: [(30.0, 0.0, 0.0), (30.5, 0.0, 0.0)]}
    walks = route.routes_covering(succ, lines, least=4.0)
    # The half-metre stub is not worth a drive of its own.
    assert all(set(w) != {2} for w in walks)


def test_covering_a_map_with_no_road_returns_nothing():
    assert route.routes_covering({}, {}) == []


def test_covering_is_the_same_for_the_same_seed():
    succ, lines = _tee()
    assert (route.routes_covering(succ, lines, seed=3)
            == route.routes_covering(succ, lines, seed=3))
