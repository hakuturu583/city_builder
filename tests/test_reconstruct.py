"""Prompts from a plot, and putting a generated mesh back on it.

No GPU and no model: what is worth testing is the arithmetic that decides where
the building ends up, because a reconstruction that is the right shape and a
quarter turn out is the easiest thing to ship without noticing.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import reconstruct as R


def _rect(long_side=30.0, short_side=18.0, angle=0.0, at=(0.0, 0.0)):
    box = ShapelyPolygon([(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
                          (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)])
    return translate(rotate(box, angle, origin=(0, 0)), *at)


def _ring(polygon):
    return [list(p) for p in list(polygon.exterior.coords)[:-1]]


def _box_mesh(width=1.0, depth=1.0, height=1.0):
    """A box in glTF axes: Y up, centred on the origin."""
    x, y, z = width / 2, height / 2, depth / 2
    return np.array([(sx * x, sy * y, sz * z)
                     for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_plan_dimensions_do_not_depend_on_which_way_the_street_runs():
    """The axis-aligned bounds of a rotated plot are much bigger than the plot."""
    for angle in (0.0, 25.0, 40.0, 90.0):
        long_side, short_side = R.plan_dimensions(_ring(_rect(30.0, 18.0, angle)))
        assert long_side == pytest.approx(30.0, abs=1e-6)
        assert short_side == pytest.approx(18.0, abs=1e-6)


def test_the_prompt_carries_the_shape_the_model_can_act_on():
    plot = {"footprint": _ring(_rect(36.0, 12.0)), "floors": 4, "height": 14.0}
    prompt = R.describe(plot)
    assert "4 storeys" in prompt
    assert "36 by 12 metres" in prompt
    assert "3.0 times as wide as it is deep" in prompt
    # Without these the image is unusable however good the building is.
    assert "plain white background" in prompt and "no other buildings" in prompt


def test_a_square_plot_is_not_described_as_a_ratio():
    plot = {"footprint": _ring(_rect(20.0, 19.0)), "floors": 1, "height": 4.0}
    prompt = R.describe(plot)
    assert "almost square in plan" in prompt
    assert "single-storey" in prompt


def test_the_style_is_the_callers_and_the_framing_is_not():
    plot = {"footprint": _ring(_rect()), "floors": 2, "height": 7.0}
    prompt = R.describe(plot, "a weatherboard fishing shed")
    assert "weatherboard fishing shed" in prompt
    assert "whole building visible" in prompt


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


def test_the_axis_change_is_a_rotation_and_not_a_mirror():
    """A reflected building is one whose signage reads backwards and fits well."""
    basis = R.to_scene_axes(np.eye(3))
    assert np.linalg.det(basis) == pytest.approx(1.0)
    # Y up in glTF becomes Z up in the scene.
    assert R.to_scene_axes(np.array([[0.0, 1.0, 0.0]]))[0].tolist() == [0.0, 0.0, 1.0]


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def test_a_mesh_of_the_right_proportions_lands_on_the_plot():
    plot = _rect(30.0, 18.0, angle=25.0, at=(120.0, -40.0))
    mesh = R.to_scene_axes(_box_mesh(width=30 / 30, depth=18 / 30, height=0.4))

    fit = R.fit_to_footprint(mesh, _ring(plot), base_z=5.0, yaw_steps=180)

    assert fit["footprint_iou"] > 0.99
    assert fit["centre"] == [pytest.approx(120.0, abs=0.01), pytest.approx(-40.0, abs=0.01)]
    # A rectangle is its own half-turn, so the yaw is only defined modulo 180.
    assert fit["yaw_deg"] % 180 == pytest.approx(25.0, abs=0.5)


def test_the_scale_comes_out_in_metres():
    plot = _rect(40.0, 20.0)
    mesh = R.to_scene_axes(_box_mesh(width=2.0, depth=1.0, height=0.5))
    fit = R.fit_to_footprint(mesh, _ring(plot), base_z=0.0)
    # The mesh is 2 units wide and the plot is 40 m, so one unit is 20 m and the
    # half-unit height becomes 10.
    assert fit["scale"] == pytest.approx(20.0, rel=1e-3)
    assert fit["height_m"] == pytest.approx(10.0, rel=1e-3)


def test_a_building_of_the_wrong_proportions_says_so():
    """The IoU is the whole point: it is what tells you the fit is a lie."""
    plot = _rect(40.0, 10.0)
    tower = R.to_scene_axes(_box_mesh(width=1.0, depth=1.0, height=3.0))
    fit = R.fit_to_footprint(tower, _ring(plot), base_z=0.0)
    assert fit["footprint_iou"] < 0.75


def test_the_building_stands_on_the_ground_it_was_given():
    plot = _rect(20.0, 20.0)
    mesh = R.to_scene_axes(_box_mesh(1.0, 1.0, 1.0))
    fit = R.fit_to_footprint(mesh, _ring(plot), base_z=12.5)
    placed = R.place(mesh, yaw=math.radians(fit["yaw_deg"]), scale=fit["scale"],
                     centre=(fit["centre"][0], fit["centre"][1]), base_z=12.5)
    assert placed[:, 2].min() == pytest.approx(12.5)


def test_an_empty_plot_is_refused():
    with pytest.raises(ValueError, match="area"):
        R.fit_to_footprint(_box_mesh(), [[0, 0], [1, 0], [1, 0]], base_z=0.0)


# ---------------------------------------------------------------------------
# Writing it out
# ---------------------------------------------------------------------------


def test_the_obj_is_in_scene_metres_and_one_indexed(tmp_path):
    path = tmp_path / "b.obj"
    R.write_obj(str(path), np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]),
                np.array([[0, 1, 2]]))
    text = path.read_text()
    assert "v 1.0000 2.0000 3.0000" in text
    assert "f 1 2 3" in text  # OBJ counts from one; from zero it loads as a hole
