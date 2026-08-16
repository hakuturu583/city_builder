"""Railings: where they go, and what they stand on.

The geometry is a post and three rails; what is worth testing is the *rule* —
that a railing goes where a person could fall, does not go where a person
cannot walk, and follows the ground rather than floating over it.
"""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest
from shapely.geometry import Point, box

from city_builder import fences as F
from city_builder.ground import HeightMap


class _Pond:
    """What `cover.flatten_water` hands back, as far as a fence cares."""

    def __init__(self, painted, level, polygon=None):
        self.painted = painted
        self.polygon = polygon if polygon is not None else painted
        self.level = level


def _flat(value=0.0, nx=30, ny=30, cell=5.0):
    return HeightMap(-40.0, -40.0, cell, np.full((ny, nx), value), np.zeros((ny, nx)))


def _length(lines):
    return sum(math.dist(a, b) for line in lines
               for a, b in pairwise(line))


# ---------------------------------------------------------------------------
# Where a railing goes
# ---------------------------------------------------------------------------


def test_standing_water_is_always_fenced():
    """No threshold: an open pond in a street is not left unfenced."""
    pond = _Pond(box(-10.0, -10.0, 10.0, 10.0), level=-1.0)
    lines = F.water_edges([pond], _flat().sample)
    assert lines and _length(lines) > 70.0


def test_the_railing_stands_on_the_bank_and_not_in_the_bowl():
    """A pond is dug, so a fixed setback puts the railing under the waterline."""
    hm = _flat()
    dug = box(-10.0, -10.0, 10.0, 10.0)
    # The ground inside the excavation is below the water; outside it is not.
    for j in range(hm.ny):
        for i in range(hm.nx):
            x, y = hm.x0 + i * hm.cell, hm.y0 + j * hm.cell
            if dug.buffer(3.0).contains(Point(x, y)):
                hm.z[j, i] = -2.0
    pond = _Pond(dug, level=-1.0)

    lines = F.water_edges([pond], hm.sample)
    heights = [hm.sample(x, y) for line in lines for x, y in line]
    assert np.mean(np.array(heights) >= -1.0) > 0.8, "most of the railing is under water"


def test_a_railing_never_crosses_the_carriageway():
    """A road beside a pond gets a guardrail, which is not this."""
    pond = _Pond(box(-10.0, -10.0, 10.0, 10.0), level=-1.0)
    road = box(-40.0, -2.0, 40.0, 2.0)
    lines = F.water_edges([pond], _flat().sample, road)
    assert lines
    for line in lines:
        for x, y in line:
            assert not road.buffer(-1e-6).contains(Point(x, y))


def test_a_stub_of_railing_is_not_worth_building():
    """A puddle rounds up to a 6 m ring; ten metres of railing is a decision."""
    tiny = _Pond(box(-0.4, -0.4, 0.4, 0.4), level=-1.0)
    assert F.water_edges([tiny], _flat().sample,
                         options=F.FenceOptions(min_length=10.0)) == []


# ---------------------------------------------------------------------------
# A drop needs a height
# ---------------------------------------------------------------------------


def test_a_kerb_high_platform_is_not_fenced():
    """Measured on a real map: the biggest drop beside a plot was 0.45 m.

    Fencing that would put a metre of railing round every front garden.
    """
    platform = box(-10.0, -10.0, 10.0, 10.0)
    assert F.terrace_edges([(platform, 0.3)], _flat().sample) == []


def test_a_wall_you_could_fall_off_is_fenced():
    platform = box(-10.0, -10.0, 10.0, 10.0)
    lines = F.terrace_edges([(platform, 1.6)], _flat().sample)
    assert lines and _length(lines) > 70.0


def test_the_drop_that_matters_is_a_setting():
    platform = box(-10.0, -10.0, 10.0, 10.0)
    assert F.terrace_edges([(platform, 0.6)], _flat().sample,
                           options=F.FenceOptions(min_drop=0.5))
    assert not F.terrace_edges([(platform, 0.6)], _flat().sample,
                               options=F.FenceOptions(min_drop=1.5))


def test_only_the_edges_that_drop_are_fenced():
    """A platform cut into a hillside is a wall on one side and a kerb on the other."""
    hm = _flat()
    # Ground falling away to the east only — the cliff starts back from the
    # platform edge, because the height map is bilinear and a step that lands
    # between two nodes is a ramp by the time it is sampled.
    for j in range(hm.ny):
        for i in range(hm.nx):
            hm.z[j, i] = -3.0 if hm.x0 + i * hm.cell > 5.0 else 0.0
    lines = F.terrace_edges([(box(-10.0, -10.0, 10.0, 10.0), 0.0)], hm.sample)
    assert lines
    xs = [x for line in lines for x, _y in line]
    assert min(xs) > 5.0, "an edge with no drop behind it was fenced"


# ---------------------------------------------------------------------------
# What it looks like
# ---------------------------------------------------------------------------


def test_the_railing_follows_the_ground_rather_than_floating_over_it():
    """A single ribbon at one height is what gives a fence away."""
    hm = _flat()
    for j in range(hm.ny):
        for i in range(hm.nx):
            hm.z[j, i] = (hm.x0 + i * hm.cell) * 0.05
    line = [(-30.0, 0.0), (30.0, 0.0)]
    mesh = F.along([line], hm.sample, F.FenceOptions())
    tops = [z for _x, _y, z in mesh.vertices]
    assert max(tops) - min(tops) > 2.0, "the railing ignored a 3 m fall"
    # Every post's top sits its own height above its own ground.
    for x, y, z in mesh.vertices:
        assert -0.3 <= z - hm.sample(x, y) <= 1.2


def test_a_post_is_sunk_rather_than_balanced_on_the_surface():
    mesh = F.along([[(-10.0, 0.0), (10.0, 0.0)]], _flat().sample)
    assert min(z for _x, _y, z in mesh.vertices) == pytest.approx(-0.15, abs=1e-6)


def test_the_posts_are_evenly_spaced_along_the_whole_run():
    """A corner must not reset the rhythm and leave two posts touching."""
    options = F.FenceOptions(post_spacing=2.0)
    mesh = F.along([[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0)]], _flat().sample, options)
    # The post centres, not the four corners of each post box.
    feet = sorted({(round(x / 0.5) * 0.5, round(y / 0.5) * 0.5)
                   for x, y, z in mesh.vertices if z < -0.1})
    gaps = [math.dist(a, b) for a, b in pairwise(feet)]
    assert min(gaps) > 1.0, "two posts landed on top of each other"
    assert max(gaps) < 3.0


def test_the_report_says_how_much_railing_and_why():
    pond = _Pond(box(-10.0, -10.0, 10.0, 10.0), level=-1.0)
    got = F.build(_flat(), water=[pond], terraces=[(box(20.0, 20.0, 30.0, 30.0), 0.2)])
    assert got["shoreline_runs"] == 1
    assert got["runs"] == 1, "the 0.2 m platform should not have been fenced"
    assert got["metres"] > 70.0
    assert got["posts"] == pytest.approx(got["metres"] / 1.8, rel=0.2)
    assert got["mesh"].faces
