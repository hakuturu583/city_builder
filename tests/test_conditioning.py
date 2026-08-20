"""The camera a video model is told about, and the depth it is shown.

Everything here is the arithmetic between Blender's camera and the one every
rasteriser downstream expects. It is worth testing precisely because none of it
fails loudly: a pose with the wrong handedness, or a focal length off by the
aspect ratio, produces a depth that unprojects into a perfectly plausible scene
of the wrong shape, and the first symptom is a Gaussian cloud that will not
converge.

The render itself wants bpy and a scene, so it is checked end to end elsewhere;
what is checked here is that the numbers agree with each other.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from city_builder import conditioning as C


def blender_camera(position=(0.0, 0.0, 0.0), rotation=None) -> np.ndarray:
    """A ``matrix_world`` for a camera. Unrotated, it looks down world -Z."""
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
    matrix[:3, 3] = np.asarray(position, dtype=float)
    return matrix


def spin(axis: str, degrees: float) -> np.ndarray:
    """A rotation about a *world* axis.

    Worth being careful with: an unrotated Blender camera looks down its own
    -Z, and world +Z is up, so the camera that has had nothing done to it is
    pointing straight at the ground. Turning it about world Z is therefore a
    roll around its own view axis, not a yaw — which is why the driving camera
    below is built by standing it up with `spin("x", 90)` first.
    """
    angle = np.radians(degrees)
    cos, sin = np.cos(angle), np.sin(angle)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, cos, -sin], [0.0, sin, cos]])
    if axis == "y":
        return np.array([[cos, 0.0, sin], [0.0, 1.0, 0.0], [-sin, 0.0, cos]])
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


#: A camera standing up off the floor, looking along world +Y. What a drive is.
HORIZON = spin("x", 90.0)


# ---------------------------------------------------------------------------
# Intrinsics
# ---------------------------------------------------------------------------


def test_a_longer_lens_is_a_longer_focal_length_in_pixels():
    wide = C.intrinsics(1280, 720, 20.0)
    tight = C.intrinsics(1280, 720, 50.0)
    assert tight[0, 0] / wide[0, 0] == pytest.approx(2.5)


def test_the_principal_point_is_the_middle_of_the_image():
    matrix = C.intrinsics(1280, 720, 30.0)
    assert matrix[0, 2] == 640.0
    assert matrix[1, 2] == 360.0


def test_the_sensor_lies_across_the_longer_side_the_way_blender_fits_it():
    """`AUTO` sensor fit, not "always the width" — a portrait render differs."""
    landscape = C.intrinsics(1280, 720, 30.0)
    portrait = C.intrinsics(720, 1280, 30.0)
    assert landscape[0, 0] == pytest.approx(1280 * 30.0 / 36.0)
    assert portrait[0, 0] == pytest.approx(1280 * 30.0 / 36.0)
    # Square pixels either way: one focal length, not two.
    assert landscape[0, 0] == landscape[1, 1]


def test_the_field_of_view_is_the_one_the_lens_gives():
    width, lens = 1280, 30.0
    matrix = C.intrinsics(width, 720, lens)
    got = 2 * np.arctan(0.5 * C.SENSOR_MM / lens)
    assert 2 * np.arctan(0.5 * width / matrix[0, 0]) == pytest.approx(got)


# ---------------------------------------------------------------------------
# The pose
# ---------------------------------------------------------------------------


def test_blender_looks_down_minus_z_and_the_result_looks_down_plus_z():
    """The whole of the convention change, in one point."""
    view = C.world_to_camera(blender_camera())
    ahead = np.array([0.0, 0.0, -5.0, 1.0])          # 5 m in front of a Blender camera
    assert np.allclose((view @ ahead)[:3], [0.0, 0.0, 5.0])


def test_the_top_of_the_image_is_the_camera_own_up():
    """+Y down is what a pixel grid means, and the source of many bugs."""
    view = C.world_to_camera(blender_camera(rotation=HORIZON))
    overhead = np.array([0.0, 5.0, 2.0, 1.0])        # ahead, and higher up
    assert (view @ overhead)[1] < 0.0                # so: negative y, the top rows


def test_the_pose_never_mirrors():
    """Determinant +1, or the scene comes back handed the wrong way round."""
    for degrees in (0.0, 37.0, 90.0, 180.0, -125.0):
        rotation = spin("z", degrees) @ HORIZON
        view = C.world_to_camera(blender_camera((3.0, -2.0, 1.5), rotation))
        assert np.linalg.det(view[:3, :3]) == pytest.approx(1.0)


def test_the_camera_position_survives_the_round_trip():
    position = (12.0, -34.0, 5.6)
    rotation = spin("z", 28.0) @ HORIZON
    camera = C.Camera(frame=1, view=C.world_to_camera(blender_camera(position, rotation)),
                      intrinsics=C.intrinsics(640, 480, 35.0), width=640, height=480)
    assert np.allclose(camera.position, position)


def test_turning_the_camera_moves_the_world_the_other_way():
    straight = C.world_to_camera(blender_camera(rotation=HORIZON))
    turned = C.world_to_camera(blender_camera(rotation=spin("z", 90.0) @ HORIZON))
    ahead = np.array([0.0, 5.0, 0.0, 1.0])           # 5 m along +Y

    assert np.allclose((straight @ ahead)[:3], [0.0, 0.0, 5.0])
    # A quarter turn and that point is off to the side rather than in front.
    assert (turned @ ahead)[2] == pytest.approx(0.0, abs=1e-9)
    assert abs((turned @ ahead)[0]) == pytest.approx(5.0)


# ---------------------------------------------------------------------------
# Depth and the pose, together
# ---------------------------------------------------------------------------


def _camera(width=64, height=48, lens=30.0, position=(0.0, 0.0, 0.0), rotation=None):
    return C.Camera(frame=1,
                    view=C.world_to_camera(blender_camera(position, rotation)),
                    intrinsics=C.intrinsics(width, height, lens),
                    width=width, height=height)


def test_a_wall_at_a_known_distance_unprojects_to_that_wall():
    """Planar depth: a flat surface stays flat, corners included."""
    camera = _camera()
    depth = np.full((camera.height, camera.width), 25.0, dtype=np.float32)
    points = C.unproject(depth, camera)

    # The camera is at the origin looking down world -Z, so a plane 25 m along
    # the view axis is the plane z = -25 — everywhere, not just at the centre.
    assert np.allclose(points[..., 2], -25.0)


def test_the_centre_pixel_unprojects_straight_ahead():
    camera = _camera()
    depth = np.full((camera.height, camera.width), 10.0, dtype=np.float32)
    points = C.unproject(depth, camera)
    centre = points[camera.height // 2, camera.width // 2]
    assert np.allclose(centre[:2], 0.0, atol=1e-6)


def test_unprojection_follows_the_camera_when_it_moves_and_turns():
    """The pose is used, not ignored: the same depth lands somewhere else."""
    depth = np.full((48, 64), 12.0, dtype=np.float32)
    here = C.unproject(depth, _camera(rotation=HORIZON))
    moved = C.unproject(depth, _camera(rotation=HORIZON, position=(100.0, 0.0, 0.0)))
    assert np.allclose(moved - here, [100.0, 0.0, 0.0])

    # A camera looking along +Y puts the wall at y = 12; turned a quarter turn
    # it does not, and every point is still the same distance away.
    assert np.allclose(here[..., 1], 12.0)
    turned = C.unproject(depth, _camera(rotation=spin("z", 90.0) @ HORIZON))
    assert not np.allclose(turned[..., 1], 12.0)
    assert np.allclose(np.linalg.norm(turned, axis=-1),
                       np.linalg.norm(here, axis=-1))    # same distances


def test_a_pixel_off_centre_opens_by_the_focal_length():
    """The one place the intrinsics can be silently wrong by the aspect ratio."""
    camera = _camera(width=64, height=48, lens=30.0)
    depth = np.full((48, 64), 20.0, dtype=np.float32)
    points = C.unproject(depth, camera)

    fx = camera.intrinsics[0, 0]
    expected = (0 - camera.intrinsics[0, 2]) / fx * 20.0
    assert points[24, 0][0] == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Depth as something a model can be shown
# ---------------------------------------------------------------------------


def test_the_sky_is_where_nothing_was_hit():
    depth = np.array([[10.0, 0.0], [np.nan, 25.0]], dtype=np.float32)
    assert C.sky(depth).tolist() == [[False, True], [True, False]]


def test_the_sky_never_reads_as_the_nearest_thing_in_frame():
    """A depth of zero is 'no surface'. Read as metres it is the closest one."""
    depth = np.array([[5.0, 0.0, 50.0]], dtype=np.float32)
    image = C.to_control_image(depth)

    assert image[0, 0] == pytest.approx(1.0)      # 5 m: nearest, brightest
    assert image[0, 2] == pytest.approx(0.0)      # 50 m: furthest
    assert image[0, 1] == pytest.approx(0.0)      # sky sits with the far end


def test_near_is_bright_because_that_is_what_the_estimators_produce():
    depth = np.array([[2.0, 4.0, 8.0]], dtype=np.float32)
    inverted = C.to_control_image(depth)
    straight = C.to_control_image(depth, invert=False)

    assert inverted[0, 0] > inverted[0, 1] > inverted[0, 2]
    assert straight[0, 0] < straight[0, 1] < straight[0, 2]


def test_a_fixed_range_keeps_a_wall_the_same_brightness_between_frames():
    """Per-frame auto-ranging is flicker, and a video model reproduces flicker."""
    wall = 12.0
    first = np.array([[wall, 40.0]], dtype=np.float32)
    second = np.array([[wall, 90.0]], dtype=np.float32)   # something far arrives

    assert C.to_control_image(first)[0, 0] != C.to_control_image(second)[0, 0]
    fixed = {"near": 1.0, "far": 100.0}
    assert (C.to_control_image(first, **fixed)[0, 0]
            == pytest.approx(C.to_control_image(second, **fixed)[0, 0]))


def test_the_range_comes_from_the_depths_that_are_actually_there():
    """Guessing it wastes it: inverse depth spends its resolution near."""
    frames = [np.full((8, 8), 5.0, dtype=np.float32),
              np.full((8, 8), 80.0, dtype=np.float32)]
    near, far = C.depth_range(frames, low=0.0, high=100.0)
    assert near == pytest.approx(5.0)
    assert far == pytest.approx(80.0)


def test_the_range_ignores_the_sky_and_a_stray_outlier():
    """A percentile only outvotes an outlier that is rarer than the percentile.

    Sized like a real frame for that reason: one bad pixel in 1560 is 0.06 %,
    well inside the half a percent trimmed off each end.
    """
    frame = np.full((40, 40), 20.0, dtype=np.float32)
    frame[0, :] = 0.0                        # sky
    frame[1, 0] = 3900.0                     # one polygon at the clip plane
    near, far = C.depth_range([frame])
    assert near == pytest.approx(20.0)
    assert far == pytest.approx(20.0, abs=1.0)   # the stray does not set it


def test_a_drive_with_no_depth_at_all_says_so():
    with pytest.raises(ValueError, match="every frame is sky"):
        C.depth_range([np.zeros((4, 4), dtype=np.float32)])


def test_an_all_sky_frame_does_not_divide_by_nothing():
    empty = np.zeros((4, 4), dtype=np.float32)
    assert np.all(C.to_control_image(empty) == 0.0)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def test_the_cameras_round_trip_through_json(tmp_path):
    cameras = [_camera(position=(float(i), 0.0, 1.4), rotation=spin("z", i * 3.0) @ HORIZON)
               for i in range(5)]
    for i, camera in enumerate(cameras):
        camera.frame = i + 1

    path = C.write_cameras(str(tmp_path / "cameras.json"), cameras)
    back = C.read_cameras(path)

    assert [c.frame for c in back] == [1, 2, 3, 4, 5]
    for first, second in zip(cameras, back, strict=True):
        assert np.allclose(first.view, second.view, atol=1e-7)
        assert np.allclose(first.intrinsics, second.intrinsics, atol=1e-5)
        assert (first.width, first.height) == (second.width, second.height)


def test_the_file_says_which_convention_it_is_in(tmp_path):
    """A pose file without its convention is a puzzle, not data."""
    path = C.write_cameras(str(tmp_path / "cameras.json"), [_camera()])
    with open(path) as handle:
        payload = json.load(handle)
    assert "world-to-camera" in payload["convention"]
    assert "+Z forward" in payload["convention"]


# ---------------------------------------------------------------------------
# Carrying a picture from one camera to another
# ---------------------------------------------------------------------------


def _wall(camera, distance=20.0):
    """A flat wall square to the camera, at a known distance."""
    return np.full((camera.height, camera.width), distance, dtype=np.float32)


def test_a_view_of_the_same_wall_carries_its_colours_over():
    """The whole point: a picture taken there, shown here."""
    there = _camera(rotation=HORIZON)
    here = _camera(rotation=HORIZON, position=(0.0, 2.0, 0.0))   # 2 m along the wall's normal
    rgb = np.zeros((there.height, there.width, 3), np.float32)
    rgb[:, : there.width // 2] = (1.0, 0.0, 0.0)
    rgb[:, there.width // 2:] = (0.0, 0.0, 1.0)

    colour, valid = C.reproject(here, _wall(here, 18.0),
                                [(there, _wall(there, 20.0), rgb)])
    assert valid.mean() > 0.9
    # The camera moved straight at the wall, so left stays left.
    left = colour[valid][:, 0] > 0.5
    assert left.any() and (~left).any()
    assert np.allclose(colour[valid][left], (1.0, 0.0, 0.0))


def test_surface_no_source_ever_saw_comes_back_invalid():
    """"Not seen" has to be reported, not filled in with something plausible."""
    there = _camera(rotation=HORIZON)
    here = _camera(rotation=spin("z", 180.0) @ HORIZON)          # looking the other way
    rgb = np.full((there.height, there.width, 3), 0.5, np.float32)

    _colour, valid = C.reproject(here, _wall(here), [(there, _wall(there), rgb)])
    assert not valid.any()


def test_a_point_hidden_in_the_source_is_refused():
    """It lands inside the old frame, and the old frame saw something nearer."""
    there = _camera(rotation=HORIZON)
    here = _camera(rotation=HORIZON)
    rgb = np.full((there.height, there.width, 3), 0.5, np.float32)

    # The source saw a wall at 5 m; this view is asking about surface at 40 m,
    # in the same direction, which is therefore behind it.
    _colour, valid = C.reproject(here, _wall(here, 40.0),
                                 [(there, _wall(there, 5.0), rgb)])
    assert not valid.any()


def test_the_sky_is_never_carried():
    camera = _camera(rotation=HORIZON)
    depth = _wall(camera)
    depth[: camera.height // 3] = 0.0                            # no surface up there
    rgb = np.full((camera.height, camera.width, 3), 0.5, np.float32)

    _colour, valid = C.reproject(camera, depth, [(camera, depth, rgb)])
    assert not valid[: camera.height // 3].any()
    assert valid[camera.height // 3:].mean() > 0.9


def test_the_most_recent_source_wins():
    """Sources come newest first, so the freshest answer is the one taken."""
    camera = _camera(rotation=HORIZON)
    depth = _wall(camera)
    newest = np.full((camera.height, camera.width, 3), 0.9, np.float32)
    oldest = np.full((camera.height, camera.width, 3), 0.1, np.float32)

    colour, valid = C.reproject(camera, depth,
                                [(camera, depth, newest), (camera, depth, oldest)])
    assert np.allclose(colour[valid], 0.9)


# ---------------------------------------------------------------------------
# The surfaces the build paints into a texture, recovered from the map


def _straight_camera(width=64, height=48, height_m=1.5):
    """Looking down +X from the origin, eye above the road plane at z=0."""
    from city_builder.conditioning import Camera

    # world +X forward, +Y left, +Z up  ->  camera +X right, +Y down, +Z forward
    rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
    view = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ np.array([0.0, 0.0, height_m])
    focal = 60.0
    intrinsics = np.array([[focal, 0.0, width / 2],
                           [0.0, focal, height / 2],
                           [0.0, 0.0, 1.0]])
    return Camera(frame=0, view=view, intrinsics=intrinsics,
                  width=width, height=height)


def test_a_ribbon_becomes_one_ring_that_closes_round_it():
    from city_builder.conditioning import rings_of
    from city_builder.geometry import Ribbon

    ribbon = Ribbon(id=1, left=[(0, 1, 0), (5, 1, 0)], right=[(0, -1, 0), (5, -1, 0)])
    rings = rings_of([ribbon])
    assert len(rings) == 1
    # Down one bound and back up the other, so the ring encloses the strip.
    assert rings[0].shape == (4, 3)
    assert rings[0][0].tolist() == [0, 1, 0]
    assert rings[0][-1].tolist() == [0, -1, 0]


def test_an_infill_gives_one_ring_per_face_so_a_hole_stays_a_hole():
    from city_builder.conditioning import rings_of
    from city_builder.geometry import Mesh

    mesh = Mesh(vertices=[(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
                faces=[[0, 1, 2], [0, 2, 3]])
    assert len(rings_of([mesh])) == 2


def test_a_ring_too_small_to_be_a_surface_is_dropped():
    from city_builder.conditioning import rings_of
    from city_builder.geometry import Polygon

    assert rings_of([Polygon(id=1, points=[(0, 0, 0), (1, 0, 0)])]) == []


def test_paint_on_the_road_lands_below_the_horizon():
    from city_builder.conditioning import cover

    camera = _straight_camera()
    stripe = np.array([(4.0, 0.2, 0.0), (30.0, 0.2, 0.0),
                       (30.0, -0.2, 0.0), (4.0, -0.2, 0.0)])
    mask = cover([stripe], camera)
    assert mask.any()
    rows = np.nonzero(mask.any(axis=1))[0]
    # The road is under the camera, so every pixel of it is below centre.
    assert rows.min() >= camera.height // 2


def test_a_wall_in_the_way_hides_the_paint_behind_it():
    from city_builder.conditioning import cover

    camera = _straight_camera()
    stripe = np.array([(20.0, 0.2, 0.0), (30.0, 0.2, 0.0),
                       (30.0, -0.2, 0.0), (20.0, -0.2, 0.0)])
    open_road = np.full((camera.height, camera.width), 1000.0, np.float32)
    assert cover([stripe], camera, open_road).any()

    # Something solid at five metres: the paint is twenty metres further on.
    blocked = np.full((camera.height, camera.width), 5.0, np.float32)
    assert not cover([stripe], camera, blocked).any()


def test_the_near_end_of_a_long_line_survives_the_occlusion_test():
    from city_builder.conditioning import cover

    camera = _straight_camera()
    # Twenty metres long: judged by one distance for the whole ring, its near
    # end reads as further away than the road it is painted on and vanishes.
    stripe = np.array([(4.0, 0.2, 0.0), (24.0, 0.2, 0.0),
                       (24.0, -0.2, 0.0), (4.0, -0.2, 0.0)])
    ground = np.zeros((camera.height, camera.width), np.float32)
    u, v = np.meshgrid(np.arange(camera.width) + 0.5, np.arange(camera.height) + 0.5)
    rays = np.linalg.inv(camera.intrinsics) @ np.stack(
        [u.ravel(), v.ravel(), np.ones(u.size)])
    forward = (camera.view[:3, :3].T @ rays).T
    with np.errstate(divide="ignore", invalid="ignore"):
        along = (0.0 - 1.5) / forward[:, 2]
    ground = np.where(along > 0, along * rays[2], 1000.0).reshape(
        camera.height, camera.width).astype(np.float32)

    mask = cover([stripe], camera, ground)
    rows = np.nonzero(mask.any(axis=1))[0]
    assert rows.max() > camera.height * 0.75, "the near end of the line was dropped"


def test_a_mask_is_soft_at_the_edges_so_a_thin_line_does_not_dash():
    from city_builder.conditioning import cover

    camera = _straight_camera(width=64, height=48)
    stripe = np.array([(4.0, 0.05, 0.0), (40.0, 0.05, 0.0),
                       (40.0, -0.05, 0.0), (4.0, -0.05, 0.0)])
    mask = cover([stripe], camera)
    values = np.unique(mask[mask > 0])
    assert (values < 1.0).any(), "a sub-pixel line came back fully opaque"


# ---------------------------------------------------------------------------
# Carrying a frame forward: the sky, the gaps, and what must not be invented


def test_a_straight_run_carries_the_sky_pixel_for_pixel():
    from city_builder.conditioning import carry_sky

    here = _straight_camera()
    # Five metres further down the same road, pointing the same way.
    ahead = _straight_camera()
    ahead.view = here.view.copy()
    ahead.view[:3, 3] = -here.view[:3, :3] @ np.array([5.0, 0.0, 1.5])

    rgb = np.random.default_rng(0).random((here.height, here.width, 3)).astype(np.float32)
    where = np.zeros((here.height, here.width), bool)
    where[:10] = True

    colour, valid = carry_sky(ahead, here, rgb, where)
    assert valid[where].all()
    # Sky is at infinity: driving does not move it.
    assert np.allclose(colour[where], rgb[where], atol=1e-5)


def test_the_sky_is_only_taken_where_it_was_asked_for():
    from city_builder.conditioning import carry_sky

    camera = _straight_camera()
    rgb = np.ones((camera.height, camera.width, 3), np.float32)
    where = np.zeros((camera.height, camera.width), bool)
    where[0, 0] = True
    _colour, valid = carry_sky(camera, camera, rgb, where)
    assert valid.sum() == 1


def test_a_gap_is_filled_with_the_light_around_it_rather_than_with_black():
    from city_builder.conditioning import fill_gaps

    colour = np.full((32, 32, 3), 0.7, np.float32)
    valid = np.ones((32, 32), bool)
    valid[8:16, 8:16] = False
    colour[~valid] = 0.0

    filled = fill_gaps(colour, valid)
    # Blurring across a hole drags the mean down; if that is not put back the
    # gap comes out darker than the frame and the model paints shadow there.
    assert filled[~valid].mean() > 0.5
    assert np.allclose(filled[valid], 0.7)


def test_filling_a_frame_that_carried_nothing_does_not_divide_by_zero():
    from city_builder.conditioning import fill_gaps

    colour = np.zeros((8, 8, 3), np.float32)
    filled = fill_gaps(colour, np.zeros((8, 8), bool))
    assert np.isfinite(filled).all()


def test_repainting_follows_the_light_the_model_chose():
    from PIL import Image

    from city_builder.conditioning import repaint

    mask = np.zeros((16, 16), np.float32)
    mask[8] = 1.0

    dusk = Image.fromarray(np.full((16, 16, 3), 40, np.uint8))
    noon = Image.fromarray(np.full((16, 16, 3), 180, np.uint8))
    on_dusk = np.asarray(repaint(dusk, mask), np.float32)[8].mean()
    on_noon = np.asarray(repaint(noon, mask), np.float32)[8].mean()

    assert on_dusk > 40, "the paint is not brighter than the road it is on"
    assert on_noon > on_dusk, "the paint ignored the light in the frame"


def test_repainting_leaves_everything_the_mask_does_not_name():
    from PIL import Image

    from city_builder.conditioning import repaint

    pixels = np.full((16, 16, 3), 90, np.uint8)
    mask = np.zeros((16, 16), np.float32)
    mask[8] = 1.0
    out = np.asarray(repaint(Image.fromarray(pixels), mask))
    assert (out[:8] == 90).all() and (out[9:] == 90).all()


def test_a_surface_stretched_past_its_own_resolution_is_refused():
    from city_builder.conditioning import Camera, reproject

    # Two cameras on the same axis, one ten metres behind the other, so a wall
    # thirty metres from the first is twenty from the second: a modest stretch.
    def at(x):
        rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ np.array([x, 0.0, 0.0])
        return Camera(frame=0, view=view, width=16, height=12,
                      intrinsics=np.array([[40.0, 0.0, 8.0],
                                           [0.0, 40.0, 6.0],
                                           [0.0, 0.0, 1.0]]))

    behind, here = at(0.0), at(20.0)
    far = np.full((12, 16), 30.0, np.float32)
    near = np.full((12, 16), 10.0, np.float32)
    rgb = np.ones((12, 16, 3), np.float32)

    # Three times closer than it was seen: kept when nine pixels from one is
    # acceptable, refused when it is not.
    _c, loose = reproject(here, near, [(behind, far, rgb)], magnify=4.0)
    _c, tight = reproject(here, near, [(behind, far, rgb)], magnify=1.5)
    assert loose.sum() > tight.sum()
    assert tight.sum() == 0


def test_without_a_limit_nothing_is_refused_for_stretch():
    from city_builder.conditioning import Camera, reproject

    rotation = np.array([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]])

    def at(x):
        view = np.eye(4)
        view[:3, :3] = rotation
        view[:3, 3] = -rotation @ np.array([x, 0.0, 0.0])
        return Camera(frame=0, view=view, width=16, height=12,
                      intrinsics=np.array([[40.0, 0.0, 8.0],
                                           [0.0, 40.0, 6.0],
                                           [0.0, 0.0, 1.0]]))

    got, _valid = reproject(at(20.0), np.full((12, 16), 10.0, np.float32),
                            [(at(0.0), np.full((12, 16), 30.0, np.float32),
                              np.ones((12, 16, 3), np.float32))])
    assert got.any()
