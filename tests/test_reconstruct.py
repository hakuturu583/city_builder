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
# Where the ground goes
#
# A model shown a building from above never sees its underside and closes it
# with a taper. Stood on its lowest vertex, the building hangs over a point of
# contact — one in six of a rebuilt street did — so the taper is buried.
# ---------------------------------------------------------------------------


def _shell(width=8.0, depth=6.0, height=6.0, dome=0.0, samples=40):
    """A closed box of points, standing on a domed underside.

    ``dome`` is how far that underside rises from its lowest point, at the
    centre, to the walls — which is what comes out of the pipeline. The model
    is shown the building from above, never sees the bottom, and closes it with
    a cap. ``dome=0`` is a building that sits flat.
    """
    points = []
    for z in np.linspace(dome, height, samples):
        for u in np.linspace(-1.0, 1.0, samples):
            for side in (-1.0, 1.0):
                points.append((side * width / 2, u * depth / 2, z))
                points.append((u * width / 2, side * depth / 2, z))
    for u in np.linspace(-1.0, 1.0, samples):
        for w in np.linspace(-1.0, 1.0, samples):
            points.append((u * width / 2, w * depth / 2, dome * max(abs(u), abs(w)) ** 2))
    return np.array(points, dtype=float)


def _reach(points, low, high):
    """How wide the mesh is between two heights."""
    slab = points[(points[:, 2] >= low) & (points[:, 2] < high)]
    return float(slab[:, 0].max() - slab[:, 0].min()) if len(slab) else 0.0


def test_a_mesh_that_is_flat_underneath_is_left_where_it_is():
    mesh = _shell()
    assert R.seat_z(mesh) == pytest.approx(mesh[:, 2].min())


def test_a_domed_underside_is_buried_rather_than_stood_on():
    mesh = _shell(dome=0.6)
    full = _reach(mesh, 3.0, 3.1)
    assert _reach(mesh, 0.0, 0.03) < 0.3 * full, "the mesh under test is not domed"
    seat = R.seat_z(mesh)
    assert seat > 0.0, "the tip of the dome was taken for the floor"
    assert _reach(mesh, seat, seat + 0.03) > 0.5 * full


def test_the_building_meets_the_ground_with_its_plan_and_not_a_point():
    """The measurement that named this problem: plan at the ground vs above it."""
    plot = _rect(8.0, 6.0)
    mesh = _shell(dome=0.6)
    fit = R.fit_to_footprint(mesh, _ring(plot), base_z=4.0)
    placed = R.place(mesh, yaw=math.radians(fit["yaw_deg"]), scale=fit["scale"],
                     centre=tuple(fit["centre"]), base_z=4.0)
    assert _reach(placed, 4.0, 4.05) > 0.5 * _reach(placed, 6.0, 6.05)


def test_the_fit_says_how_far_it_had_to_sink_it():
    plot = _rect(8.0, 6.0)
    flat = R.fit_to_footprint(_shell(), _ring(plot), base_z=0.0)
    domed = R.fit_to_footprint(_shell(dome=0.6), _ring(plot), base_z=0.0)
    assert flat["sunk_m"] == pytest.approx(0.0, abs=1e-6)
    assert domed["sunk_m"] > 0.05


def test_the_height_is_measured_from_the_ground_and_not_from_the_buried_tip():
    plot = _rect(8.0, 6.0)
    fit = R.fit_to_footprint(_shell(height=6.0, dome=0.6), _ring(plot), base_z=30.0)
    # Six metres of building, less however much of the dome went under.
    assert fit["height_m"] == pytest.approx(6.0 * fit["scale"] - fit["sunk_m"], rel=0.02)


def test_a_building_that_narrows_all_the_way_down_is_not_buried_whole():
    """A dome is a modelling artefact; a genuinely tapered building is not."""
    cone = _shell(height=6.0, dome=6.0)
    assert R.seat_z(cone) - cone[:, 2].min() <= 0.15 * 6.0 + 1e-6
