"""Driving the same street again, looking somewhere else.

Coverage is the fraction of the cloud that has actually been made: a splat no
camera reached keeps the flat colour the mesh gave it, and one forward camera
down one street left 45% of the t-junction in that state. What fixes it is not
more route — every seed on a map that size finds the same thirty-eight metres —
but more direction.

So what is checked here is that turning the camera turns the camera and not the
drive, that moving across the lane moves it the way it says, and that a sweep
stops for the two different reasons it can stop for.
"""

from __future__ import annotations

import math

import numpy as np

from city_builder.coverage import Pass, enough, turned


def _straight(length=5):
    """Driving along +x, looking ten metres ahead."""
    return [((float(i), 0.0, 1.5), (float(i) + 10.0, 0.0, 1.2))
            for i in range(length)]


def test_looking_ahead_changes_nothing():
    path = _straight()
    assert np.allclose(np.array(turned(path, Pass(yaw=0.0))), np.array(path))


def test_turning_the_camera_leaves_the_drive_where_it_was():
    # A driver who glances left is still in the same lane.
    path = _straight()
    for eye, _target in turned(path, Pass(yaw=-55.0)):
        assert eye in [p[0] for p in path]


def test_turning_left_puts_the_target_to_the_left():
    (eye, target), = turned([((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))], Pass(yaw=90.0))
    across = np.asarray(target) - np.asarray(eye)
    # +y is to the left of travel along +x.
    assert across[1] > 9.0 and abs(across[0]) < 1e-6


def test_turning_right_is_the_other_way():
    (eye, target), = turned([((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))], Pass(yaw=-90.0))
    assert (np.asarray(target) - np.asarray(eye))[1] < -9.0


def test_a_turn_keeps_the_distance_it_was_looking():
    path = [((0.0, 0.0, 1.5), (10.0, 0.0, 1.2))]
    for angle in (-110.0, -55.0, 55.0, 180.0):
        (eye, target), = turned(path, Pass(yaw=angle))
        reach = np.linalg.norm(np.asarray(target) - np.asarray(eye))
        assert abs(reach - math.dist((0, 0, 1.5), (10, 0, 1.2))) < 1e-6


def test_moving_across_the_lane_moves_it_to_the_side():
    (eye, _target), = turned([((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))],
                             Pass(yaw=0.0, sideways=2.0))
    assert abs(eye[1] - 2.0) < 1e-6 and abs(eye[2] - 1.5) < 1e-6


def test_the_sweep_works_outwards_and_alternates_sides():
    from city_builder.coverage import LOOKS

    assert LOOKS[0].yaw == 0.0, "the first look should be where the car is going"
    # Every turn comes with its mirror, immediately after it, so a sweep that
    # is stopped early leaves a balanced set rather than everything to one side.
    turns = [p.yaw for p in LOOKS if p.yaw not in (0.0, 180.0)]
    assert len(turns) % 2 == 0
    for left, right in zip(turns[::2], turns[1::2]):
        assert left == -right


def test_a_sweep_that_reached_the_target_says_so():
    why = enough([{"coverage": 91.0, "gained": 2.4}], target=90.0, least=1.5)
    assert why and "target reached" in why


def test_a_sweep_that_stopped_paying_says_something_different():
    # Not the same event: this one means looking has run out of new surfaces,
    # and the answer is a different map rather than another pass.
    why = enough([{"coverage": 70.0, "gained": 9.0},
                  {"coverage": 70.4, "gained": 0.4}], target=90.0, least=1.5)
    assert why and "target reached" not in why and "0.4" in why


def test_one_road_repeating_itself_does_not_stop_the_sweep():
    # Two routes down the same street in opposite directions see the same
    # walls, so the second adds nothing — and six directions are still to go.
    history = [{"coverage": 55.1, "gained": 55.1},
               {"coverage": 91.9, "gained": 36.8},
               {"coverage": 91.9, "gained": 0.0}]
    assert enough(history, target=96.0, least=0.8, routes=4) is None


def test_a_whole_round_finding_nothing_does_stop_it():
    history = [{"coverage": 91.9, "gained": 36.8}] + [
        {"coverage": 91.9, "gained": 0.0} for _ in range(4)]
    why = enough(history, target=96.0, least=0.8, routes=4)
    assert why and "a round of 4" in why


def test_a_first_pass_is_never_a_reason_to_stop():
    # However little it found, there is nothing yet to compare it against.
    assert enough([{"coverage": 0.9, "gained": 0.9}], target=90.0, least=1.5) is None


def test_a_sweep_with_nothing_in_it_keeps_going():
    assert enough([], target=90.0, least=1.5) is None


def test_a_sweep_drives_every_road_before_looking_twice_at_one():
    from city_builder.coverage import sweep

    passes = sweep(routes=3)
    # Whole streets left at the mesh's flat colour while the first is polished
    # is the failure this ordering exists to avoid.
    assert {p.route for p in passes[:3]} == {0, 1, 2}
    assert len({p.yaw for p in passes[:3]}) == 1


def test_a_sweep_covers_every_route_and_every_look():
    from city_builder.coverage import LOOKS, sweep

    passes = sweep(routes=2, both_ways=False)
    assert len(passes) == 2 * len(LOOKS)
    assert {p.route for p in passes} == {0, 1}


def test_a_raised_pass_is_the_only_one_that_looks_down():
    from city_builder.coverage import LOOKS

    assert any(p.height > 1.5 for p in LOOKS)
    assert all(p.height >= 1.5 for p in LOOKS)


def test_a_pass_is_named_by_everything_that_makes_it_different():
    from city_builder.coverage import Pass

    a = Pass(route=1, yaw=-55.0, height=4.0)
    b = Pass(route=1, yaw=-55.0, height=1.5)
    assert a.name != b.name, "two different drives would share a directory"


def test_both_directions_come_before_any_turning():
    from city_builder.coverage import sweep

    passes = sweep(routes=2)
    # A wall approached from one end only is fixed at how it looked from that
    # end, and the far face of a pole is never seen at all; that is worth more
    # than a second angle on what has already been seen.
    first_turn = next(i for i, p in enumerate(passes) if p.yaw != 0.0)
    assert {p.reverse for p in passes[:first_turn]} == {False, True}


def test_driving_both_ways_doubles_the_sweep():
    from city_builder.coverage import LOOKS, sweep

    assert len(sweep(routes=3)) == 3 * len(LOOKS) * 2
    assert len(sweep(routes=3, both_ways=False)) == 3 * len(LOOKS)


def test_the_two_directions_of_one_route_are_told_apart():
    from city_builder.coverage import Pass

    assert Pass(route=1).name != Pass(route=1, reverse=True).name
