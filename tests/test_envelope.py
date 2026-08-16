"""Generating inside the plot instead of fitting afterwards.

The generative model is not exercised here — it is sixteen gigabytes and a
card. What is testable is the part that decides what it is asked: the prism
that replaces its first sampling stage, and the one property of the
conditioning picture that predicts whether the result will be a building.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from shapely.affinity import rotate
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import reconstruct as R


def _footprint(long_side=20.0, short_side=12.0, angle=0.0):
    box = ShapelyPolygon([(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
                          (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)])
    return [list(p) for p in list(rotate(box, angle, origin=(0, 0)).exterior.coords)[:-1]]


def _extent(coords, axis):
    return coords[:, axis].max() - coords[:, axis].min() + 1


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_the_prism_has_the_plans_proportions():
    """What the whole thing is for: the plan is the plot's, not the model's."""
    coords = R.envelope_coords(_footprint(20.0, 10.0), 6.0, grid=32, eave_room=0.0)
    assert _extent(coords, 0) / _extent(coords, 1) == pytest.approx(2.0, rel=0.12)


def test_the_height_is_on_the_height_axis():
    """The axes are the identity, and every other mapping stands it on end."""
    tall = R.envelope_coords(_footprint(20.0, 20.0), 20.0, grid=32, eave_room=0.0)
    short = R.envelope_coords(_footprint(20.0, 20.0), 5.0, grid=32, eave_room=0.0)
    assert _extent(tall, 2) > _extent(short, 2) * 2
    assert _extent(tall, 0) == _extent(short, 0)


def test_a_turned_plot_turns_the_prism_rather_than_its_bounding_box():
    """A rectangle at 45 degrees is a diamond, not a bigger rectangle."""
    square = R.envelope_coords(_footprint(20.0, 20.0), 6.0, grid=32, eave_room=0.0)
    diamond = R.envelope_coords(_footprint(20.0, 20.0, angle=45.0), 6.0,
                                grid=32, eave_room=0.0)
    columns = lambda c: len({(i, j) for i, j, _k in c})
    assert columns(diamond) < columns(square) * 0.65


def test_room_for_the_eaves_grows_the_plan_and_not_the_height():
    """Measured: 0.6 m of room took the footprint IoU from 0.822 to 0.882."""
    tight = R.envelope_coords(_footprint(), 6.0, grid=32, eave_room=0.0)
    roomy = R.envelope_coords(_footprint(), 6.0, grid=32, eave_room=1.5)
    assert len({(i, j) for i, j, _k in roomy}) > len({(i, j) for i, j, _k in tight})
    assert _extent(roomy, 2) == _extent(tight, 2)


def test_the_prism_is_centred_in_the_cube_the_mesh_comes_back_in():
    """`to_glb` reads a cube centred on the origin, so an off-centre prism
    would place every building at an offset the fit then has to undo."""
    coords = R.envelope_coords(_footprint(20.0, 12.0), 8.0, grid=32, eave_room=0.0)
    for axis in (0, 1):
        middle = (coords[:, axis].max() + coords[:, axis].min()) / 2
        assert middle == pytest.approx(15.5, abs=0.6)


def test_a_plan_thinner_than_a_voxel_says_so():
    """The cube is sized by the *largest* dimension, so a tall enough building
    on a small enough plot has a plan below one cell and no columns at all.
    Silently returning no cells reaches the sampler as "generate nothing"."""
    with pytest.raises(ValueError, match="too small"):
        R.envelope_coords(_footprint(0.2, 0.2), 60.0, grid=32, eave_room=0.0)


def test_the_grid_is_the_one_the_flow_model_was_trained_on():
    """Handing the 512 model a 64 cube is a mismatch, not a finer envelope:
    it measured 0.743 against 0.822 for the same plot."""
    assert R._ENVELOPE_GRID["512"] == 32
    assert R._ENVELOPE_GRID["1024"] == 64


def test_a_cascade_has_no_single_set_of_coords_to_replace():
    with pytest.raises(ValueError, match="single-resolution"):
        R.to_mesh_in_envelope("unused.png", "out.glb", footprint=_footprint(), height=6.0,
                              options=R.MeshOptions(pipeline_type="1024_cascade"))


# ---------------------------------------------------------------------------
# The picture, which now only has to carry the material
# ---------------------------------------------------------------------------


def _framed(subject_fraction: float, size=128) -> Image.Image:
    """A building of that share of the frame, on a plain backdrop.

    The subject is textured, because a photograph is: a flat block would be
    keyed out as backdrop itself the moment it reached the border.
    """
    rng = np.random.default_rng(4)
    frame = np.full((size, size, 3), 150, dtype=np.uint8)
    side = round(size * subject_fraction ** 0.5)
    start = (size - side) // 2
    frame[start:start + side, start:start + side] = rng.integers(
        20, 110, (side, side, 3), dtype=np.uint8)
    return Image.fromarray(frame)


def test_a_subject_on_a_plain_field_reads_as_isolated():
    assert R.backdrop_share(_framed(0.25)) > 0.6


def _street(size=128) -> Image.Image:
    """A house in its setting: graded sky, building, garden. No backdrop at all.

    The failure this exists to catch. Asked for a photograph of a house, the
    image model returns the street it stands in, and TRELLIS takes the whole
    frame as the subject — sky, garden and the neighbours end up in the walls.
    """
    rng = np.random.default_rng(7)
    frame = np.empty((size, size, 3), dtype=np.uint8)
    sky = np.linspace(120, 235, size // 3).astype(np.uint8)
    frame[:size // 3] = sky[:, None, None]
    frame[size // 3:] = rng.integers(40, 120, (size - size // 3, size, 3), dtype=np.uint8)
    wall = frame[size // 3:size * 3 // 4, size // 5:size * 4 // 5]
    wall[:] = rng.integers(150, 210, wall.shape, dtype=np.uint8)
    return Image.fromarray(frame)


def test_a_street_scene_does_not_read_as_isolated():
    assert R.backdrop_share(_street()) < 0.25 <= R.backdrop_share(_framed(0.25))


def test_the_frame_is_asked_for_and_not_only_the_building():
    prompt = R.isolated_prompt("a house")
    assert "a house" in prompt
    for wanted in ("isolated", "no ground", "no sky", "whole building"):
        assert wanted in prompt
    for unwanted in ("street scene", "sky", "adjacent buildings", "cropped"):
        assert unwanted in R.ISOLATED_NEGATIVE
