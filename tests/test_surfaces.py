"""Zebra bars for a crossing the map located but did not draw.

Autoware maps routinely carry a crossing as a lanelet and no `pedestrian_marking`
rings at all — on the Kashiwanoha map, four crossings and zero bars — and the
baking pass then removes the crossing surface on the grounds that its paint is
in the road texture, where there is none. The crossing vanishes from a scene
whose own map inspector reported it. These are about the bars that replace it.
"""

from __future__ import annotations

import math

import pytest
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder.geometry import Ribbon
from city_builder.surfaces import SurfaceOptions, zebra_bars


def _crossing(length=8.0, width=4.0, angle=0.0, at=(0.0, 0.0), z=0.0, steps=2):
    """A crossing lanelet: two boundaries ``width`` apart, ``length`` long.

    The boundaries run along the kerbs, so people walk *across* it — which is
    the direction the bars have to span.
    """
    cos, sin = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    left, right = [], []
    for i in range(steps + 1):
        t = length * i / steps
        for side, out in ((-width / 2, left), (width / 2, right)):
            x, y = t, side
            out.append((at[0] + x * cos - y * sin, at[1] + x * sin + y * cos, z))
    return Ribbon(7, left, right)


def _ring(bar):
    return ShapelyPolygon([(p[0], p[1]) for p in bar.points])


def test_the_bars_span_the_crossing_rather_than_running_along_it():
    """A bar along the lanelet is a lane line; across it is a zebra."""
    bars = zebra_bars(_crossing(length=8.0, width=4.0))
    assert bars
    for bar in bars:
        ring = _ring(bar)
        rectangle = list(ring.minimum_rotated_rectangle.exterior.coords)[:4]
        sides = [math.dist(rectangle[i], rectangle[(i + 1) % 4]) for i in range(4)]
        # The long side of a bar is the width of the crossing.
        assert max(sides) == pytest.approx(4.0, abs=0.2)
        assert min(sides) == pytest.approx(SurfaceOptions().zebra_bar_width, abs=0.05)


def test_the_bars_stay_on_the_crossing():
    crossing = _crossing(length=9.0, width=3.5, angle=27.0, at=(120.0, -40.0))
    surface = ShapelyPolygon([(p[0], p[1]) for p in crossing.ring()]).buffer(0.01)
    for bar in zebra_bars(crossing):
        assert surface.contains(_ring(bar)), "a bar left the crossing it belongs to"


def test_a_longer_crossing_gets_more_bars():
    counts = [len(zebra_bars(_crossing(length=length))) for length in (4.0, 8.0, 16.0)]
    assert counts == sorted(counts) and counts[0] < counts[-1]


def test_the_paint_is_centred_on_the_crossing():
    """Otherwise one kerb has a bar flush against it and the other a gap."""
    crossing = _crossing(length=10.0, width=4.0)
    bars = zebra_bars(crossing)
    xs = [p[0] for bar in bars for p in bar.points]
    assert min(xs) == pytest.approx(10.0 - max(xs), abs=0.05)


def test_a_crossing_too_short_for_a_bar_gets_none():
    assert zebra_bars(_crossing(length=0.5, width=4.0)) == []


def test_the_bars_are_flat_on_the_crossing_they_cross():
    crossing = _crossing(length=8.0, width=4.0, z=3.25)
    for bar in zebra_bars(crossing):
        assert all(p[2] == pytest.approx(3.25) for p in bar.points)


def test_the_bar_size_is_a_setting():
    wide = zebra_bars(_crossing(length=12.0),
                      SurfaceOptions(zebra_bar_width=1.0, zebra_bar_gap=1.0))
    narrow = zebra_bars(_crossing(length=12.0),
                        SurfaceOptions(zebra_bar_width=0.2, zebra_bar_gap=0.2))
    assert len(narrow) > len(wide)


def test_a_crossing_with_mismatched_boundaries_is_refused():
    ragged = Ribbon(1, [(0, 0, 0), (5, 0, 0)], [(0, 2, 0)])
    assert zebra_bars(ragged) == []
