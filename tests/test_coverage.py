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

from city_builder.coverage import SWEEP, Pass, enough, turned


def _straight(length=5):
    """Driving along +x, looking ten metres ahead."""
    return [((float(i), 0.0, 1.5), (float(i) + 10.0, 0.0, 1.2))
            for i in range(length)]


def test_looking_ahead_changes_nothing():
    path = _straight()
    assert np.allclose(np.array(turned(path, Pass(0.0))), np.array(path))


def test_turning_the_camera_leaves_the_drive_where_it_was():
    # A driver who glances left is still in the same lane.
    path = _straight()
    for eye, _target in turned(path, Pass(-55.0)):
        assert eye in [p[0] for p in path]


def test_turning_left_puts_the_target_to_the_left():
    (eye, target), = turned([((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))], Pass(90.0))
    across = np.asarray(target) - np.asarray(eye)
    # +y is to the left of travel along +x.
    assert across[1] > 9.0 and abs(across[0]) < 1e-6


def test_turning_right_is_the_other_way():
    (eye, target), = turned([((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))], Pass(-90.0))
    assert (np.asarray(target) - np.asarray(eye))[1] < -9.0


def test_a_turn_keeps_the_distance_it_was_looking():
    path = [((0.0, 0.0, 1.5), (10.0, 0.0, 1.2))]
    for angle in (-110.0, -55.0, 55.0, 180.0):
        (eye, target), = turned(path, Pass(angle))
        reach = np.linalg.norm(np.asarray(target) - np.asarray(eye))
        assert abs(reach - math.dist((0, 0, 1.5), (10, 0, 1.2))) < 1e-6


def test_moving_across_the_lane_moves_it_to_the_side():
    (eye, _target), = turned([((0.0, 0.0, 1.5), (10.0, 0.0, 1.5))],
                             Pass(0.0, sideways=2.0))
    assert abs(eye[1] - 2.0) < 1e-6 and abs(eye[2] - 1.5) < 1e-6


def test_the_sweep_works_outwards_and_alternates_sides():
    turns = [p.yaw for p in SWEEP if p.sideways == 0.0]
    assert turns[0] == 0.0
    # Stopping early should leave a balanced set, not everything to one side.
    for left, right in zip(turns[1::2], turns[2::2]):
        assert left == -right


def test_a_sweep_that_reached_the_target_says_so():
    why = enough([{"coverage": 91.0, "gained": 2.4}], target=90.0, least=1.5)
    assert why and "target reached" in why


def test_a_sweep_that_stopped_paying_says_something_different():
    # Not the same event: this one means turning the camera has run out of new
    # surfaces, and the answer is a different route rather than another look.
    why = enough([{"coverage": 70.0, "gained": 9.0},
                  {"coverage": 70.4, "gained": 0.4}], target=90.0, least=1.5)
    assert why and "target reached" not in why and "0.4" in why


def test_a_first_pass_is_never_a_reason_to_stop():
    # However little it found, there is nothing yet to compare it against.
    assert enough([{"coverage": 0.9, "gained": 0.9}], target=90.0, least=1.5) is None


def test_a_sweep_with_nothing_in_it_keeps_going():
    assert enough([], target=90.0, least=1.5) is None
