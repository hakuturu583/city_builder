"""The orbit camera and the mask ranges. No map, no Blender.

The claims worth testing here are not "the numbers came back" but the three
the module is built on: the frame count is one the model and a closed turn can
both use, the distance really does put the whole building in shot, and the face
range really is the building the caller asked for.
"""

from __future__ import annotations

import math

import pytest

from city_builder import orbit

# ---------------------------------------------------------------------------
# The frame grid
# ---------------------------------------------------------------------------


def test_the_grid_is_the_one_the_model_counts_in():
    assert orbit.frame_counts(130) == [5, 22, 39, 56, 73, 90, 107, 124]


def test_only_some_of_the_grid_can_be_quartered():
    assert orbit.frame_counts(200, quadrants=True) == [56, 124, 192]


def test_snapping_prefers_the_shorter_orbit_on_a_tie():
    # 90 is exactly 34 from both 56 and 124, and a frame is a cost.
    assert orbit.snap_frames(90) == 56
    assert orbit.snap_frames(100) == 124
    assert orbit.snap_frames(40, quadrants=False) == 39


def test_quadrant_frames_are_the_cardinal_azimuths():
    assert orbit.quadrant_frames(56) == [0, 14, 28, 42]
    path = orbit.orbit_path((0.0, 0.0, 0.0), frames=56, distance=10.0, elevation_deg=0.0)
    for index, expected in zip(orbit.quadrant_frames(56), [0.0, 90.0, 180.0, 270.0]):
        (x, y, _), _ = path[index]
        assert math.degrees(math.atan2(y, x)) % 360 == pytest.approx(expected, abs=1e-9)


def test_a_count_that_cannot_be_quartered_says_so():
    with pytest.raises(ValueError, match="quartered"):
        orbit.quadrant_frames(39)


# ---------------------------------------------------------------------------
# Framing
# ---------------------------------------------------------------------------


def _square(size: float, at=(0.0, 0.0)):
    half = size / 2.0
    return [(at[0] - half, at[1] - half), (at[0] + half, at[1] - half),
            (at[0] + half, at[1] + half), (at[0] - half, at[1] + half)]


def test_the_circle_covers_every_corner():
    cx, cy, radius = orbit.enclosing_circle(_square(20.0, at=(100.0, -40.0)))
    assert (cx, cy) == pytest.approx((100.0, -40.0))
    for x, y in _square(20.0, at=(100.0, -40.0)):
        assert math.dist((x, y), (cx, cy)) <= radius + 1e-9


def test_an_empty_footprint_is_refused():
    with pytest.raises(ValueError):
        orbit.enclosing_circle([])


def test_a_bigger_building_is_shot_from_further_back():
    small = orbit.framing_distance(8.0, 10.0)
    tall = orbit.framing_distance(8.0, 40.0)
    wide = orbit.framing_distance(30.0, 10.0)
    assert tall > small and wide > small


def _slack(point, position, target, *, lens, resolution):
    """How far out of the middle of the frame ``point`` lands: 1.0 is the edge.

    A projection done here rather than borrowed from the module, so the framing
    is checked against Blender's camera model and not against its own arithmetic.
    """
    import numpy as np

    forward = np.asarray(target, dtype=float) - np.asarray(position, dtype=float)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)

    offset = np.asarray(point, dtype=float) - np.asarray(position, dtype=float)
    depth = float(offset @ forward)
    if depth <= 0.0:
        return float("inf")

    long_side, short_side = max(resolution), min(resolution)
    across = 36.0 / 2.0 / lens
    upwards = 36.0 * short_side / long_side / 2.0 / lens
    return max(abs(offset @ right) / depth / across, abs(offset @ up) / depth / upwards)


def _in_shot(point, position, target, **kwargs):
    return _slack(point, position, target, **kwargs) <= 1.0


@pytest.mark.parametrize("size,height", [(12.0, 8.0), (40.0, 12.0), (10.0, 45.0)])
def test_the_whole_building_is_in_every_frame(size, height):
    """The claim the framing distance makes, checked by projecting the corners."""
    options = orbit.OrbitOptions(frames=56, elevation_deg=18.0)
    plot = {"footprint": _square(size), "height": height, "base_z": 3.0}
    shot = orbit.plan_orbit(plot, options)

    corners = [(x, y, z) for x, y in plot["footprint"]
               for z in (plot["base_z"], plot["base_z"] + height)]
    for position, target in shot["path"]:
        for corner in corners:
            assert _in_shot(corner, position, target, lens=options.lens,
                            resolution=(options.width, options.height)), \
                f"{corner} left the frame from {position}"


@pytest.mark.parametrize("size,height", [(12.0, 8.0), (40.0, 12.0), (10.0, 45.0)])
def test_at_margin_1_something_is_exactly_on_the_edge_of_the_frame(size, height):
    """The framing is tight: nothing is in shot that did not have to be."""
    options = orbit.OrbitOptions(frames=56, elevation_deg=18.0, margin=1.0)
    plot = {"footprint": _square(size), "height": height, "base_z": 3.0}
    shot = orbit.plan_orbit(plot, options)

    # The cylinder, not the footprint: that is the envelope the distance is for.
    radius = orbit.enclosing_circle(plot["footprint"])[2]
    rim = [(radius * math.cos(math.radians(a)), radius * math.sin(math.radians(a)), z)
           for a in range(0, 360, 5) for z in (3.0, 3.0 + height)]
    position, target = shot["path"][0]
    slack = max(_slack(point, position, target, lens=options.lens,
                       resolution=(options.width, options.height)) for point in rim)
    assert slack == pytest.approx(1.0, abs=0.02), "the cylinder is not touching the frame edge"


def test_a_squat_building_is_not_framed_as_if_it_were_a_sphere():
    """Regression: the bounding sphere put a wide, low building at 13 % of frame.

    The sphere round a 40 m wide, 10 m tall building is 40 m wide too, and
    framing that in the narrow field of view stands the camera nearly twice as
    far back as the building itself needs.
    """
    tight = orbit.framing_distance(20.0, 10.0, margin=1.0)
    sphere = math.hypot(20.0, 5.0) / math.sin(math.atan(36.0 * 480 / 832 / 2.0 / 35.0))
    assert tight < 0.7 * sphere  # measured: 0.68, so the subject is 2.2x the area


# ---------------------------------------------------------------------------
# The path
# ---------------------------------------------------------------------------


def test_the_camera_stays_on_the_sphere_and_looks_at_its_centre():
    centre = (5.0, -3.0, 12.0)
    path = orbit.orbit_path(centre, frames=22, distance=60.0, elevation_deg=15.0)
    assert len(path) == 22
    for position, target in path:
        assert target == centre
        assert math.dist(position, centre) == pytest.approx(60.0)
        assert position[2] == pytest.approx(centre[2] + 60.0 * math.sin(math.radians(15.0)))


def test_the_turn_is_closed_and_does_not_repeat_its_first_frame():
    path = orbit.orbit_path((0.0, 0.0, 0.0), frames=56, distance=10.0, elevation_deg=0.0)
    step = 360.0 / 56
    for index in (1, 27, 55):
        (x, y, _), _ = path[index]
        assert math.degrees(math.atan2(y, x)) % 360 == pytest.approx((step * index) % 360)
    assert math.dist(path[0][0], path[-1][0]) > 1e-6


def test_clockwise_goes_the_other_way():
    anti = orbit.orbit_path((0, 0, 0), frames=56, distance=10.0, elevation_deg=0.0)
    clock = orbit.orbit_path((0, 0, 0), frames=56, distance=10.0, elevation_deg=0.0,
                             clockwise=True)
    assert anti[1][0][1] > 0 > clock[1][0][1]


def test_a_start_angle_rotates_the_whole_orbit():
    path = orbit.orbit_path((0, 0, 0), frames=56, distance=10.0, elevation_deg=0.0,
                            start_deg=90.0)
    (x, y, _), _ = path[0]
    assert (x, y) == pytest.approx((0.0, 10.0), abs=1e-9)


# ---------------------------------------------------------------------------
# Options and the plan
# ---------------------------------------------------------------------------


def test_a_frame_count_off_the_grid_is_refused():
    with pytest.raises(ValueError, match="17k"):
        orbit.OrbitOptions(frames=60)


def test_a_resolution_the_refinement_cannot_take_is_refused():
    with pytest.raises(ValueError, match="multiple of 32"):
        orbit.OrbitOptions(width=830)


def test_neighbours_has_three_values():
    for mode in ("keep", "clear", "hide"):
        assert orbit.OrbitOptions(neighbours=mode).neighbours == mode
    with pytest.raises(ValueError, match="keep"):
        orbit.OrbitOptions(neighbours="delete")


def test_the_plan_carries_what_the_reconstruction_needs():
    plot = {"footprint": _square(18.0, at=(10.0, 20.0)), "height": 21.0, "base_z": 4.5}
    shot = orbit.plan_orbit(plot, orbit.OrbitOptions(frames=56))
    assert shot["centre"] == [10.0, 20.0, 4.5 + 10.5]
    assert shot["quadrant_frames"] == [0, 14, 28, 42]
    assert shot["degrees_per_frame"] == pytest.approx(360 / 56, abs=1e-4)
    assert len(shot["path"]) == 56
    # The footprint travels with the shot: it is the only statement of scale
    # anything downstream of the reconstruction will have.
    assert shot["footprint"] == [list(p) for p in plot["footprint"]]


def test_a_plot_built_before_footprints_were_kept_says_so():
    with pytest.raises(ValueError, match="footprint"):
        orbit.plan_orbit({"height": 10.0, "base_z": 0.0})


# ---------------------------------------------------------------------------
# Who is in the way
# ---------------------------------------------------------------------------


def _plot(at, size=10.0, height=10.0):
    return {"footprint": _square(size, at=at), "height": height, "base_z": 0.0,
            "area": size * size}


def test_only_what_can_get_between_the_camera_and_the_subject_is_cleared():
    """The disc is the whole test: the camera's sightlines never leave it."""
    plots = [
        _plot((0.0, 0.0)),        # the subject
        _plot((30.0, 0.0)),       # well inside the disc
        _plot((0.0, 200.0)),      # far outside it
        _plot((54.0, 0.0)),       # just outside 50, but 5 m of it reaches in
    ]
    assert orbit.blocking_buildings(plots, 0, reach=50.0) == [1, 3]


def test_the_subject_never_blocks_itself():
    plots = [_plot((0.0, 0.0)), _plot((300.0, 300.0))]
    assert orbit.blocking_buildings(plots, 0, reach=50.0) == []


def test_a_plot_with_no_footprint_is_left_alone():
    plots = [_plot((0.0, 0.0)), {"height": 9.0, "base_z": 0.0}]
    assert orbit.blocking_buildings(plots, 0, reach=500.0) == []


def test_the_reach_is_the_disc_the_camera_flies_over():
    plot = {"footprint": _square(20.0), "height": 15.0, "base_z": 0.0}
    shot = orbit.plan_orbit(plot, orbit.OrbitOptions(frames=56, elevation_deg=30.0))
    for position, _target in shot["path"]:
        assert math.dist(position[:2], shot["centre"][:2]) == pytest.approx(shot["reach_m"],
                                                                           abs=1e-3)


# ---------------------------------------------------------------------------
# Which faces are which building
# ---------------------------------------------------------------------------


def test_face_ranges_are_the_running_total():
    counts = [10, 4, 0, 7]
    assert orbit.face_range(counts, 0) == (0, 10)
    assert orbit.face_range(counts, 1) == (10, 14)
    assert orbit.face_range(counts, 2) == (14, 14)  # a degenerate plot contributed nothing
    assert orbit.face_range(counts, 3) == (14, 21)


def test_asking_for_a_building_that_is_not_there_says_how_many_are():
    with pytest.raises(IndexError, match="has 3"):
        orbit.face_range([1, 2, 3], 3)
