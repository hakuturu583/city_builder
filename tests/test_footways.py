"""Putting pavements into a map that was surveyed without them.

The failure being guarded against is a street with no footway at all, which is
what every map to hand produces: carriageway from wall to wall. That is not
cosmetic — a segmentation-conditioned generator reads a twenty-metre road with
buildings either side as a pedestrian square and paves the middle.

What is checked here is the decision, not the file format: that a bound with
carriageway on both sides is left alone, that one with road on a single side is
paved on the other, that the pavement lands off the road rather than on it, and
that a stretch crossing a junction mouth is cut rather than laid across it.
"""

from __future__ import annotations

import numpy as np
import pytest

from city_builder import footways
from city_builder.footways import FootwayOptions

shapely = pytest.importorskip("shapely")
from shapely.geometry import Polygon
from shapely.ops import unary_union


def _lane(y_low, y_high, x_from=0.0, x_to=30.0):
    return Polygon([(x_from, y_low), (x_to, y_low), (x_to, y_high), (x_from, y_high)])


def _straight(y, x_from=0.0, x_to=30.0):
    return np.array([[x_from, y], [x_to, y]])


def test_a_bound_with_road_on_both_sides_is_left_alone():
    # Two lanes meeting at y = 0: the line between them is not an edge.
    road = unary_union([_lane(-3.5, 0.0), _lane(0.0, 3.5)])
    got = footways.plan({1: _straight(0.0)}, road)
    assert got == {}


def test_the_outer_edge_of_a_road_is_paved_on_the_side_away_from_it():
    road = _lane(0.0, 3.5)
    got = footways.plan({1: _straight(0.0)}, road)
    assert len(got) == 1
    inner, outer = next(iter(got.values()))
    # The carriageway is above; the pavement has to be below.
    assert inner[:, 1].mean() < 0 and outer[:, 1].mean() < inner[:, 1].mean()


def test_the_pavement_is_as_wide_as_it_was_asked_to_be():
    road = _lane(0.0, 3.5)
    options = FootwayOptions(offset=0.3, width=2.0)
    inner, outer = next(iter(footways.plan({1: _straight(0.0)}, road, options).values()))
    assert np.allclose(np.abs(inner[:, 1]), 0.3, atol=1e-6)
    assert np.allclose(np.abs(outer[:, 1]), 2.3, atol=1e-6)


def test_no_part_of_a_pavement_sits_on_the_carriageway():
    road = _lane(0.0, 3.5)
    inner, outer = next(iter(footways.plan({1: _straight(0.0)}, road).values()))
    laid = Polygon(np.vstack([inner, outer[::-1]])).buffer(0)
    assert laid.intersection(road).area < 1e-9


def test_a_junction_mouth_cuts_the_pavement_rather_than_losing_it():
    # A road along y in [0, 3.5], and a side road coming down from it between
    # x = 12 and x = 18. The pavement below runs, stops at the mouth, resumes.
    road = unary_union([_lane(0.0, 3.5), Polygon(
        [(12.0, -20.0), (18.0, -20.0), (18.0, 0.0), (12.0, 0.0)])])
    got = footways.plan({1: _straight(0.0)}, road)
    assert len(got) == 2, "the pavement should be cut in two, not laid or lost"
    for inner, outer in got.values():
        laid = Polygon(np.vstack([inner, outer[::-1]])).buffer(0)
        assert laid.intersection(road).area / laid.area < 0.05


def test_a_sliver_left_by_a_cut_is_not_worth_laying():
    road = unary_union([_lane(0.0, 3.5), Polygon(
        [(1.0, -20.0), (29.0, -20.0), (29.0, 0.0), (1.0, 0.0)])])
    assert footways.plan({1: _straight(0.0)}, road) == {}


def test_a_bound_with_nothing_either_side_is_not_an_edge_of_anything():
    road = _lane(0.0, 3.5, x_from=100.0, x_to=130.0)
    assert footways.plan({1: _straight(0.0)}, road) == {}


def test_a_bound_of_two_points_still_gets_a_decision():
    # Most bounds in these maps are a single straight segment, and probing at
    # their vertices gives an answer out of two samples that decides nothing.
    road = _lane(0.0, 3.5)
    assert len(footways.plan({1: _straight(0.0)}, road)) == 1


def test_the_offset_keeps_a_vertex_for_every_vertex_it_was_given():
    points = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 1.0]])
    assert footways._offset(points, 0.5).shape == points.shape


def test_writing_leaves_the_survey_untouched(tmp_path):
    import xml.etree.ElementTree as ET

    source = tmp_path / "in.osm"
    source.write_text(
        '<?xml version="1.0"?>\n<osm version="0.6">\n'
        '  <node id="1" lat="35.0" lon="139.0" />\n</osm>\n')
    target = tmp_path / "out.osm"

    paved = {(1, 0): (np.array([[0.0, 0.0], [1.0, 0.0]]),
                      np.array([[0.0, 2.0], [1.0, 2.0]]))}
    made = footways.write(str(target), str(source), paved,
                          lambda x, y: (35.0 + y * 1e-5, 139.0 + x * 1e-5))
    assert made == {"walkway": 1, "road_border": 1}

    root = ET.parse(target).getroot()
    assert root.find("node[@id='1']") is not None, "the survey's own node went missing"
    walkways = [r for r in root.findall("relation")
                if any(t.get("k") == "subtype" and t.get("v") == "walkway"
                       for t in r.findall("tag"))]
    assert len(walkways) == 1
    # New identifiers well clear of anything a hand-drawn map would use.
    assert all(int(e.get("id")) > 1000 for e in root.findall("way"))
