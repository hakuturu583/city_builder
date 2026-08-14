"""Scene handles, and the numbers a scene reports about itself.

Geometry only — no map, no Blender. The reports are what an agent driving the
MCP server sees instead of a render, so what matters is that each one is
faithful about the specific failure it was invented to catch.
"""

from __future__ import annotations

import numpy as np
import pytest

from city_builder import scenes
from city_builder.build import BuildResult
from city_builder.frame import LocalFrame
from city_builder.geometry import Ribbon
from city_builder.ground import HeightMap


def _lane(lane_id, y0, y1, *, z=0.0, x0=0.0, x1=40.0, n=9):
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    return Ribbon(lane_id, [(x, y0, z) for x in xs], [(x, y1, z) for x in xs])


def _ground(value=0.0, nx=21, ny=21, cell=10.0):
    return HeightMap(-100.0, -100.0, cell, np.full((ny, nx), value, dtype=float),
                     np.zeros((ny, nx), dtype=float))


def _result(roads=(), *, heightmap=None, elevated=(), **groups):
    return BuildResult(
        frame=LocalFrame(35.0, 139.0),
        groups={"Roads": list(roads), **groups},
        heightmap=heightmap,
        elevated=set(elevated),
        z_datum=0.0,
        stats={"cells_measured_pct": 42.0},
    )


def _scene(result, name="scene-1"):
    return scenes.Scene(name, "/nowhere/map.osm", result, buildings=False)


# ---------------------------------------------------------------------------
# The store
# ---------------------------------------------------------------------------


def test_handles_are_distinct_and_come_back():
    store = scenes.SceneStore()
    first = store.add("/a.osm", _result(), buildings=False)
    second = store.add("/b.osm", _result(), buildings=True)
    assert first.name != second.name
    assert store.get(second.name).map_path == "/b.osm"
    assert len(store.all()) == 2


def test_an_unknown_handle_says_what_is_known():
    store = scenes.SceneStore()
    store.add("/a.osm", _result(), buildings=False)
    with pytest.raises(KeyError) as caught:
        store.get("scene-9")
    # The agent that mistyped a handle needs the real ones in the error, not a
    # bare KeyError it has to go and list scenes to recover from.
    assert "scene-1" in str(caught.value)


def test_forgetting_twice_is_an_error_not_a_shrug():
    store = scenes.SceneStore()
    scene = store.add("/a.osm", _result(), buildings=False)
    store.drop(scene.name)
    with pytest.raises(KeyError):
        store.drop(scene.name)


def test_a_summary_carries_the_group_sizes_not_the_geometry():
    result = _result([_lane(1, 0.0, 3.0)], heightmap=_ground())
    summary = _scene(result).summary()
    assert summary["groups"]["Roads"] == 1
    assert summary["ground_cells_measured_pct"] == 42.0
    assert summary["ground_grid"] == "21x21"


# ---------------------------------------------------------------------------
# Holes in the drivable surface
# ---------------------------------------------------------------------------


def test_two_lanes_side_by_side_leave_no_hole():
    result = _result([_lane(1, 0.0, 3.0), _lane(2, 3.0, 6.0)])
    report = scenes.carriageway_holes(result)["levels"]["at_grade"]
    assert report["seams"] == 0
    assert report["openings"] == 0
    assert report["surface_m2"] == pytest.approx(240.0, rel=0.02)


def test_a_long_thin_slot_is_a_seam_however_much_area_it_covers():
    # 0.2 m of daylight down the middle over forty metres: eight square metres,
    # which by area alone would sort with the harmless islands. It is the gap a
    # wheel drops into, and the reason this metric measures width.
    result = _result([_lane(1, 0.0, 3.0), _lane(2, 3.2, 6.2),
                      _lane(3, 0.0, 6.2, x0=-6.0, x1=0.0, n=3),
                      _lane(4, 0.0, 6.2, x0=40.0, x1=46.0, n=3)])
    report = scenes.carriageway_holes(result, gap=0.05)["levels"]["at_grade"]
    assert report["seams"] == 1
    assert report["openings"] == 0
    assert report["seam_area_m2"] == pytest.approx(8.0, rel=0.05)


def test_a_hole_wide_enough_to_stand_in_is_an_opening_not_a_seam():
    ring = [_lane(1, 0.0, 3.0), _lane(2, 20.0, 23.0),
            _lane(3, 0.0, 23.0, x0=-3.0, x1=0.0, n=3),
            _lane(4, 0.0, 23.0, x0=40.0, x1=43.0, n=3)]
    report = scenes.carriageway_holes(_result(ring))["levels"]["at_grade"]
    assert report["openings"] == 1
    assert report["seams"] == 0
    assert report["largest_opening_m2"] == pytest.approx(680.0, rel=0.05)


def test_levels_are_judged_separately():
    # A deck ten metres over a street shares its plan view. Judged together,
    # the ground beside the deck reads as a hole through it.
    at_grade = _lane(1, 0.0, 8.0)
    deck = _lane(2, 2.0, 5.0, z=10.0)
    report = scenes.carriageway_holes(_result([at_grade, deck], elevated=[2]))["levels"]
    assert report["at_grade"]["seams"] == 0
    assert report["elevated"]["seams"] == 0
    assert report["elevated"]["surface_m2"] < report["at_grade"]["surface_m2"]


def test_a_scene_with_no_carriageway_says_so_rather_than_reporting_zero_holes():
    assert scenes.carriageway_holes(_result())["levels"] == {}


# ---------------------------------------------------------------------------
# Elevation
# ---------------------------------------------------------------------------


def test_clearance_is_measured_against_the_ground_under_each_lane():
    result = _result([_lane(1, 0.0, 3.0, z=8.0), _lane(2, 10.0, 13.0)],
                     heightmap=_ground(1.0), elevated=[1])
    report = scenes.elevation_report(result)
    assert report["elevated"] == 1
    assert report["clearance_m"]["median"] == pytest.approx(7.0, abs=0.01)


def test_without_a_ground_there_is_nothing_to_measure_against():
    result = _result([_lane(1, 0.0, 3.0, z=8.0)], elevated=[1])
    assert scenes.elevation_report(result)["elevated"] == 0
    assert "note" in scenes.elevation_report(result)


def test_structure_counts_ride_along_with_the_clearance():
    result = _result([_lane(1, 0.0, 3.0, z=8.0)], heightmap=_ground(), elevated=[1],
                     ViaductDecks=[object()], ViaductPiers=[object(), object()])
    report = scenes.elevation_report(result)
    assert report["decks"] == 1
    assert report["piers"] == 2
    assert report["parapets"] == 0


# ---------------------------------------------------------------------------
# Buildings
# ---------------------------------------------------------------------------


def test_floor_counts_are_reported_because_sheets_are_drawn_per_floor_count():
    result = _result()
    result.plots = [{"height": 7.0, "floors": 2}, {"height": 21.0, "floors": 6},
                    {"height": 21.5, "floors": 6}]
    report = scenes.building_report(result)
    assert report["floor_counts"] == [2, 6]
    assert report["height_m"]["max"] == 21.5


def test_no_buildings_is_zero_not_an_error():
    assert scenes.building_report(_result()) == {"buildings": 0}
