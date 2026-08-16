"""The one view a reconstruction is given. No Blender.

The claim worth testing is that the whole building is in the picture and that
the view is a three-quarter of *this* building — a view square-on to a slab
shows one face, and a reconstruction from it invents the depth.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import portrait as P


def _rect(long_side=30.0, short_side=18.0, angle=0.0, at=(0.0, 0.0)):
    box = ShapelyPolygon([(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
                          (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)])
    return [list(p) for p in
            list(translate(rotate(box, angle, origin=(0, 0)), *at).exterior.coords)[:-1]]


def _plot(long_side=30.0, short_side=18.0, angle=0.0, at=(0.0, 0.0), height=12.0, base=3.0):
    return {"footprint": _rect(long_side, short_side, angle, at), "height": height,
            "base_z": base, "floors": max(1, round(height / 3.5))}


# ---------------------------------------------------------------------------
# Where to stand
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 25.0, 90.0, 160.0])
def test_the_long_axis_is_found_whichever_way_the_street_runs(angle):
    assert P.long_axis_deg(_rect(30.0, 18.0, angle)) == pytest.approx(angle % 180, abs=1e-6)


def test_the_circle_covers_every_corner():
    ring = _rect(20.0, 20.0, at=(100.0, -40.0))
    cx, cy, radius = P.enclosing_circle(ring)
    for x, y in ring:
        assert math.dist((x, y), (cx, cy)) <= radius + 1e-9


def test_the_view_is_a_three_quarter_of_this_building():
    """Off the building's own axis, so it shows a long face and a short one."""
    for street in (0.0, 37.0, 115.0):
        plot = _plot(angle=street)
        (x, y, _z), (tx, ty, _tz) = P.portrait_pose(plot, P.PortraitOptions(
            azimuth_off_axis_deg=35.0))
        bearing = math.degrees(math.atan2(y - ty, x - tx)) % 180
        assert bearing == pytest.approx((street + 35.0) % 180, abs=1e-6)


def test_the_camera_is_above_the_middle_of_the_building():
    plot = _plot(height=12.0, base=3.0)
    (x, y, z), target = P.portrait_pose(plot, P.PortraitOptions(elevation_deg=35.0))
    assert target[2] == pytest.approx(3.0 + 6.0)
    distance = math.dist((x, y, z), target)
    assert z - target[2] == pytest.approx(distance * math.sin(math.radians(35.0)))


def _slack(point, position, target, *, lens, size):
    forward = np.asarray(target) - np.asarray(position)
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, (0.0, 0.0, 1.0))
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    offset = np.asarray(point, dtype=float) - np.asarray(position)
    depth = float(offset @ forward)
    if depth <= 0.0:
        return float("inf")
    half = 36.0 / 2.0 / lens  # square frame: the same both ways
    return max(abs(offset @ right) / depth / half, abs(offset @ up) / depth / half)


@pytest.mark.parametrize("long_side,short_side,height", [(30.0, 18.0, 12.0),
                                                         (60.0, 12.0, 8.0),
                                                         (14.0, 13.0, 40.0)])
def test_the_whole_building_is_in_the_picture(long_side, short_side, height):
    options = P.PortraitOptions()
    plot = _plot(long_side, short_side, angle=20.0, height=height)
    position, target = P.portrait_pose(plot, options)
    corners = [(x, y, z) for x, y in plot["footprint"]
               for z in (plot["base_z"], plot["base_z"] + height)]
    for corner in corners:
        assert _slack(corner, position, target, lens=options.lens, size=options.size) <= 1.0


def test_a_building_with_no_extent_is_refused():
    with pytest.raises(ValueError, match="extent"):
        P.framing_distance(0.0, 10.0)


# ---------------------------------------------------------------------------
# Which faces are which building
# ---------------------------------------------------------------------------


def test_face_ranges_are_the_running_total():
    counts = [10, 4, 0, 7]
    assert P.face_range(counts, 0) == (0, 10)
    assert P.face_range(counts, 2) == (14, 14)
    assert P.face_range(counts, 3) == (14, 21)


def test_asking_for_a_building_that_is_not_there_says_how_many_are():
    with pytest.raises(IndexError, match="has 3"):
        P.face_range([1, 2, 3], 3)
