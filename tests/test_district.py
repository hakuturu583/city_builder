"""Deciding what to spend on a plot, and what to keep of what came back.

No GPU: what is testable here is the arithmetic that runs before and after the
model, which is where the street-scale decisions live.
"""

from __future__ import annotations

import pytest
from shapely.affinity import rotate
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import district as D


def _plot(long_side=20.0, short_side=12.0, angle=0.0, area=None):
    box = ShapelyPolygon([(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
                          (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)])
    ring = rotate(box, angle, origin=(0, 0))
    return {"footprint": [list(p) for p in list(ring.exterior.coords)[:-1]],
            "area": area if area is not None else long_side * short_side}


# ---------------------------------------------------------------------------
# What a plot is worth rebuilding
# ---------------------------------------------------------------------------


def test_a_plot_without_a_footprint_cannot_be_fitted_back_and_is_not_tried():
    assert not D.worth_rebuilding({"area": 400.0})
    assert D.worth_rebuilding(_plot())


def test_a_plot_below_the_area_asked_for_is_skipped():
    assert not D.worth_rebuilding(_plot(area=30.0), min_area=50.0)
    assert D.worth_rebuilding(_plot(area=80.0), min_area=50.0)


# ---------------------------------------------------------------------------
# How much licence the image model gets
#
# The measurement behind this: over 200 real plots the model returned a plan
# 1.4 to 1.5 times as long as it was deep whatever it was shown, so the drop
# rate tracked the plot's own aspect — 89 % kept under 1.5:1, 7 % over 3:1.
# ---------------------------------------------------------------------------


def test_a_square_plot_gets_the_whole_brush_up():
    assert D.licence(_plot(20.0, 18.0), 0.55) == pytest.approx(0.55)


def test_a_long_thin_plot_gets_less_of_it():
    assert D.licence(_plot(40.0, 10.0), 0.55) < 0.35


def test_the_licence_falls_as_the_plot_gets_longer():
    strengths = [D.licence(_plot(ratio * 10.0, 10.0), 0.55) for ratio in (1.0, 2.0, 3.0, 5.0)]
    assert strengths == sorted(strengths, reverse=True)
    assert strengths[0] > strengths[-1]


def test_the_licence_never_falls_below_the_floor():
    """Below it the massing comes back unchanged, which is a box on a street."""
    assert D.licence(_plot(100.0, 5.0), 0.55) == pytest.approx(0.30)


def test_a_caller_asking_for_little_licence_is_not_given_more():
    assert D.licence(_plot(40.0, 10.0), 0.2) == pytest.approx(0.2)


def test_the_aspect_is_the_plots_own_and_not_the_streets():
    """A plot on a street that does not run north is not thereby elongated."""
    for angle in (0.0, 27.0, 45.0, 63.0):
        assert D.licence(_plot(20.0, 18.0, angle), 0.55) == pytest.approx(0.55)


def test_a_plot_with_no_footprint_is_left_at_the_callers_strength():
    assert D.licence({"area": 90.0}, 0.55) == pytest.approx(0.55)
