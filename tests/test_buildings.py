"""Procedural building tests. No map, no Blender."""

from __future__ import annotations

import random

import numpy as np
import pytest
from shapely.geometry import Polygon as ShapelyPolygon
from shapely.geometry import box
from shapely.ops import unary_union

from city_builder import buildings as B
from city_builder.ground import HeightMap


def _flat_heightmap(value=0.0, nx=21, ny=21, cell=10.0):
    return HeightMap(0.0, 0.0, cell, np.full((ny, nx), value), np.zeros((ny, nx)))


def _cross_roads():
    """A + of roads through a 200x200 m area, 10 m wide."""
    return unary_union([box(0, 95, 200, 105), box(95, 0, 105, 200)])


# --- where a building may stand ---------------------------------------------


def test_buildable_area_is_the_gap_between_the_roads():
    blocks = B.buildable_area(_cross_roads(), (0, 0, 200, 200), setback=3.0)
    assert len(blocks) == 4  # the four quadrants
    for block in blocks:
        assert block.intersection(_cross_roads()).area == pytest.approx(0.0, abs=1e-9)


def test_buildable_area_keeps_holes():
    """Regression: taking only the exterior handed the enclosed road back.

    A road running through the middle of a region is an interior ring; dropping
    it put 449 of 2481 plots on the carriageway on a real map.
    """
    ring_road = box(40, 40, 160, 160).difference(box(50, 50, 150, 150))
    blocks = B.buildable_area(ring_road, (0, 0, 200, 200), setback=0.0)
    outer = max(blocks, key=lambda b: b.area)
    assert outer.interiors, "the enclosed area must stay a hole"
    # Touching along the kerb is fine; overlapping it is not.
    assert outer.intersection(ring_road).area == pytest.approx(0.0, abs=1e-9)


def test_no_footprint_touches_a_road():
    roads = _cross_roads()
    plots = B.footprints(roads, (0, 0, 200, 200), B.BuildingOptions(seed=3))
    assert plots
    for plot in plots:
        assert plot.intersection(roads).area == pytest.approx(0.0, abs=1e-9)


def test_footprints_respect_the_setback():
    roads = _cross_roads()
    options = B.BuildingOptions(setback=5.0, lot_margin=0.0, seed=1)
    for plot in B.footprints(roads, (0, 0, 200, 200), options):
        assert plot.distance(roads) >= 5.0 - 1e-6


def test_footprints_do_not_overlap_each_other():
    plots = B.footprints(_cross_roads(), (0, 0, 200, 200), B.BuildingOptions(seed=7))
    for i, a in enumerate(plots):
        for b in plots[i + 1:]:
            assert a.intersection(b).area == pytest.approx(0.0, abs=1e-9)


def test_layout_is_deterministic_for_a_seed():
    roads = _cross_roads()
    a = B.footprints(roads, (0, 0, 200, 200), B.BuildingOptions(seed=42))
    b = B.footprints(roads, (0, 0, 200, 200), B.BuildingOptions(seed=42))
    c = B.footprints(roads, (0, 0, 200, 200), B.BuildingOptions(seed=43))
    assert [p.area for p in a] == [p.area for p in b]
    assert [p.area for p in a] != [p.area for p in c]


def test_lots_stay_near_the_target_area():
    options = B.BuildingOptions(target_lot_area=400.0, min_lot_area=50.0, seed=2)
    lots = B.split_lots(box(0, 0, 100, 100), options, random.Random(0))
    assert len(lots) > 1
    for lot in lots:
        assert lot.area <= options.target_lot_area * 2.5
    assert sum(lot.area for lot in lots) == pytest.approx(10000.0, rel=0.01)


def test_max_buildings_keeps_the_largest():
    options = B.BuildingOptions(max_buildings=5, seed=4)
    plots = B.footprints(_cross_roads(), (0, 0, 200, 200), options)
    assert len(plots) == 5
    assert plots == sorted(plots, key=lambda p: p.area, reverse=True)


# --- geometry ----------------------------------------------------------------


def test_extrude_makes_walls_and_a_roof():
    walls, roof = B.extrude(box(0, 0, 10, 20), base_z=5.0, height=12.0, skirt=1.0)
    assert len(walls.faces) == 4
    assert len(roof.faces) == 2

    wall_z = [v[2] for v in walls.vertices]
    assert min(wall_z) == pytest.approx(4.0)  # base minus skirt
    assert max(wall_z) == pytest.approx(17.0)
    assert all(v[2] == pytest.approx(17.0) for v in roof.vertices)


def test_roof_faces_up():
    _, roof = B.extrude(box(0, 0, 10, 20), base_z=0.0, height=9.0)
    verts = np.array(roof.vertices)
    for face in roof.faces:
        p0, p1, p2 = verts[face]
        assert np.cross(p1 - p0, p2 - p0)[2] > 0


def test_walls_face_outward():
    walls, _ = B.extrude(box(0, 0, 10, 10), base_z=0.0, height=5.0)
    verts = np.array(walls.vertices)
    centre = np.array([5.0, 5.0])
    for face in walls.faces:
        p0, p1, p2 = verts[face[:3]]
        normal = np.cross(p1 - p0, p2 - p0)
        outward = np.array([*(p0[:2] - centre), 0.0])
        assert float(np.dot(normal, outward)) > 0


def test_courtyard_gets_its_own_walls():
    footprint = ShapelyPolygon(
        [(0, 0), (30, 0), (30, 30), (0, 30)],
        [[(10, 10), (10, 20), (20, 20), (20, 10)]],
    )
    walls, roof = B.extrude(footprint, base_z=0.0, height=8.0)
    assert len(walls.faces) == 8, "four outside, four around the courtyard"
    roof_area = sum(
        abs(np.cross(*(np.array(roof.vertices)[face][1:] - np.array(roof.vertices)[face][0])[:, :2])) / 2
        for face in roof.faces
    )
    assert roof_area == pytest.approx(footprint.area, rel=0.01), "the courtyard is not roofed over"


def test_base_height_takes_the_lowest_corner():
    hm = HeightMap(0.0, 0.0, 10.0, np.array([[0.0, 0.0], [4.0, 4.0]]), np.zeros((2, 2)))
    assert B.base_height(box(0, 0, 10, 10), hm) == pytest.approx(0.0)


def test_heights_snap_to_whole_floors():
    options = B.BuildingOptions(floor_height=3.5, min_height=7.0, max_height=42.0)
    rng = random.Random(0)
    for _ in range(50):
        height = B.pick_height(800.0, options, rng)
        assert height % 3.5 == pytest.approx(0.0, abs=1e-9)
        assert height >= 3.5


# --- end to end (still no Blender) -------------------------------------------


def test_generate_stands_every_building_on_the_ground():
    hm = _flat_heightmap(12.0)
    result = B.generate(hm, _cross_roads(), B.BuildingOptions(seed=5), bounds=(0, 0, 200, 200))
    assert result["Buildings"] and len(result["Buildings"]) == len(result["Roofs"])

    for walls, record in zip(result["Buildings"], result["plots"]):
        wall_z = [v[2] for v in walls.vertices]
        assert min(wall_z) == pytest.approx(12.0 - 1.0)  # ground minus the skirt
        assert max(wall_z) == pytest.approx(12.0 + record["height"])


def test_every_record_carries_the_plot_it_was_built_on():
    """The footprint is the only statement of scale anything downstream has.

    A mesh reconstructed from footage of this building comes back normalised
    into a unit cube; this ring is what puts it back at the size it was.
    """
    hm = _flat_heightmap(12.0)
    result = B.generate(hm, _cross_roads(), B.BuildingOptions(seed=5), bounds=(0, 0, 200, 200))

    for walls, record in zip(result["Buildings"], result["plots"]):
        ring = record["footprint"]
        assert len(ring) >= 3
        assert ring[0] != ring[-1], "the ring is not closed; the last point would be a duplicate"
        assert ShapelyPolygon(ring).area == pytest.approx(record["area"], rel=1e-3)
        # The same plan the walls were extruded from, to the millimetre it was
        # rounded to.
        plan = {(round(x, 3), round(y, 3)) for x, y, _ in walls.vertices}
        assert {(x, y) for x, y in ring} <= plan


def test_generate_returns_nothing_when_there_is_no_room():
    hm = _flat_heightmap()
    covered = box(-10, -10, 210, 210)
    result = B.generate(hm, covered, B.BuildingOptions(), bounds=(0, 0, 200, 200))
    assert result["Buildings"] == []


# --- density -----------------------------------------------------------------


def test_coverage_sets_the_share_of_a_lot_that_is_built():
    lot = box(0, 0, 40, 40)
    for coverage in (0.3, 0.5, 0.8):
        plot = B.inset_to_coverage(lot, coverage, minimum_margin=0.0)
        assert plot.area / lot.area == pytest.approx(coverage, rel=0.02)


def test_coverage_is_independent_of_lot_size():
    """A fixed margin does not do this: it builds small lots less densely."""
    small = B.inset_to_coverage(box(0, 0, 20, 20), 0.6, 0.0)
    large = B.inset_to_coverage(box(0, 0, 50, 50), 0.6, 0.0)
    assert small.area / 400 == pytest.approx(large.area / 2500, rel=0.02)


def test_minimum_margin_wins_when_it_is_the_tighter_constraint():
    lot = box(0, 0, 20, 20)
    plot = B.inset_to_coverage(lot, coverage=0.99, minimum_margin=2.0)
    assert plot.bounds == pytest.approx((2.0, 2.0, 18.0, 18.0))


def test_density_responds_to_coverage_on_a_whole_layout():
    roads = _cross_roads()
    bounds = (0, 0, 200, 200)
    sparse = B.footprints(roads, bounds, B.BuildingOptions(coverage=0.3, seed=11))
    dense = B.footprints(roads, bounds, B.BuildingOptions(coverage=0.8, seed=11))
    assert sum(p.area for p in dense) > sum(p.area for p in sparse) * 2
    assert len(sparse) == len(dense), "coverage changes size, not the lot count"


def test_vacancy_empties_lots_without_moving_the_rest():
    roads = _cross_roads()
    bounds = (0, 0, 200, 200)
    full = B.footprints(roads, bounds, B.BuildingOptions(vacancy=0.0, seed=13))
    half = B.footprints(roads, bounds, B.BuildingOptions(vacancy=0.5, seed=13))
    assert 0 < len(half) < len(full)


def test_vacancy_is_deterministic():
    roads = _cross_roads()
    bounds = (0, 0, 200, 200)
    a = B.footprints(roads, bounds, B.BuildingOptions(vacancy=0.4, seed=21))
    b = B.footprints(roads, bounds, B.BuildingOptions(vacancy=0.4, seed=21))
    assert [p.area for p in a] == [p.area for p in b]


# ---------------------------------------------------------------------------
# Facade UVs
# ---------------------------------------------------------------------------


def test_wall_uvs_run_from_the_pavement_to_the_roofline():
    """V=0 is the ground and V=1 the top, so a sheet spans the building exactly.

    The skirt is the one thing below zero: it is buried in the hillside and
    whatever the sheet does down there is never seen.
    """
    walls, _ = B.extrude(box(0, 0, 12, 12), base_z=0.0, height=20.0, skirt=1.0)
    v = [uv[1] for uv in walls.uvs]
    assert max(v) == pytest.approx(1.0)
    assert min(v) == pytest.approx(-1.0 / 20.0)


def test_a_whole_number_of_sheets_goes_round_a_building():
    """Otherwise the last repeat is cut off mid-window at the closing corner."""
    for width, height in ((10, 20), (7, 7), (23, 4), (12, 12)):
        walls, _ = B.extrude(box(0, 0, width, height), base_z=0.0, height=15.0,
                             skirt=1.0, facade_width=12.0)
        u_max = max(uv[0] for uv in walls.uvs)
        assert u_max == pytest.approx(round(u_max)), f"{width}x{height} wraps mid-sheet"
        assert u_max >= 1.0


def test_the_stretch_needed_to_make_it_wrap_is_small():
    """The wrap is bought by scaling the sheet, so the price has to stay invisible."""
    for width, height in ((10, 20), (7, 7), (23, 4), (13, 9)):
        perimeter = 2 * (width + height)
        walls, _ = B.extrude(box(0, 0, width, height), base_z=0.0, height=15.0,
                             facade_width=12.0)
        metres_per_sheet = perimeter / max(uv[0] for uv in walls.uvs)
        assert abs(metres_per_sheet - 12.0) / 12.0 < 0.25


def test_a_courtyard_wall_wraps_on_its_own_ring():
    footprint = ShapelyPolygon(
        [(0, 0), (40, 0), (40, 40), (0, 40)],
        [[(10, 10), (10, 30), (30, 30), (30, 10)]],
    )
    walls, _ = B.extrude(footprint, base_z=0.0, height=12.0, facade_width=12.0)
    # Each ring restarts at U=0, so the maximum is the outer ring's whole count.
    u_max = max(uv[0] for uv in walls.uvs)
    assert u_max == pytest.approx(round(u_max))


def test_every_building_records_the_floor_count_its_sheet_needs():
    heightmap = _flat_heightmap()
    built = B.generate(heightmap, _cross_roads(),
                       B.BuildingOptions(seed=5, floor_height=3.5))
    assert built["plots"]
    for plot in built["plots"]:
        assert plot["floors"] == pytest.approx(plot["height"] / 3.5, abs=0.01)
        assert plot["floors"] >= 1


# --- the base course ---------------------------------------------------------


def test_the_plinth_stands_proud_of_the_wall_it_carries():
    """Without one, a wall meets the ground as a line and reads as card."""
    plot = box(0, 0, 10, 6)
    base = B.plinth(plot, 2.0, height=0.35, proud=0.12, skirt=1.0)
    xs = [v[0] for v in base.vertices]
    zs = [v[2] for v in base.vertices]
    assert min(xs) == pytest.approx(-0.12) and max(xs) == pytest.approx(10.12)
    assert min(zs) == pytest.approx(1.0)  # down to the skirt, like the walls
    assert max(zs) == pytest.approx(2.35)


def test_the_plinth_wears_the_bottom_of_the_sheet_and_not_a_storey():
    base = B.plinth(box(0, 0, 10, 6), 0.0)
    assert base.uvs and max(v for _u, v in base.uvs) <= 0.05


def test_a_building_is_still_one_wall_mesh_after_the_plinth_joins_it():
    """A scene has one Buildings object, and a building is a range of faces in it."""
    hm = _flat_heightmap()
    result = B.generate(hm, _cross_roads(), B.BuildingOptions(seed=5), bounds=(0, 0, 200, 200))
    assert len(result["Buildings"]) == len(result["Roofs"]) == len(result["plots"])


def test_the_plinth_can_be_turned_off():
    hm = _flat_heightmap()
    options = B.BuildingOptions(seed=5, plinth_height=0.0)
    result = B.generate(hm, _cross_roads(), options, bounds=(0, 0, 200, 200))
    with_base = B.generate(hm, _cross_roads(), B.BuildingOptions(seed=5),
                           bounds=(0, 0, 200, 200))
    assert len(result["Buildings"][0].faces) < len(with_base["Buildings"][0].faces)


# --- how much of the lot is used ---------------------------------------------


def test_a_building_stands_back_from_the_street_it_faces():
    """Insetting to coverage leaves the same gap on all four sides.

    A street of those reads as one continuous wall with slots in it. A real lot
    puts the building at the back and the front is a yard or a parking space.
    """
    roads = _cross_roads()
    options = {"seed": 5, "target_lot_area": 200.0, "min_lot_area": 70.0,
               "max_road_distance": 0.0}
    flush = B.footprints(roads, (0, 0, 200, 200), B.BuildingOptions(frontage=0.0, **options))
    stood_back = B.footprints(roads, (0, 0, 200, 200), B.BuildingOptions(frontage=3.5, **options))

    def frontline(plots):
        near = [p.distance(roads) for p in plots if p.distance(roads) < 25]
        return sorted(near)[len(near) // 2]

    assert frontline(stood_back) > frontline(flush) + 2.0
    # And without losing the street: a lot too small for both a frontage and a
    # house builds to the street rather than not at all.
    assert len(stood_back) == len(flush)


def test_a_lot_nothing_can_reach_is_not_built_on():
    roads = _cross_roads()
    options = {"seed": 5, "target_lot_area": 200.0, "min_lot_area": 70.0,
               "frontage": 0.0}
    everywhere = B.footprints(roads, (0, 0, 200, 200),
                              B.BuildingOptions(max_road_distance=0.0, **options))
    reachable = B.footprints(roads, (0, 0, 200, 200),
                             B.BuildingOptions(max_road_distance=30.0, **options))
    assert len(reachable) < len(everywhere)
    # The limit is on the *lot*; the building then sits inside it, and the
    # coverage inset moves it a metre or two further from the street.
    assert max(p.distance(roads) for p in reachable) <= 30.0 + 3.0


def test_the_frontage_is_taken_off_the_side_the_road_is_on():
    from shapely.geometry import box as shapely_box

    lot = shapely_box(0, 0, 20, 20)
    road = shapely_box(-10, -6, 30, -1)  # to the south
    back = B.give_up_the_frontage(lot, road, 5.0)
    assert back.area < lot.area
    assert back.centroid.y > lot.centroid.y, "the building moved towards the road"


def test_a_lot_with_no_road_keeps_all_of_itself():
    from shapely.geometry import box as shapely_box

    lot = shapely_box(0, 0, 20, 20)
    assert B.give_up_the_frontage(lot, None, 5.0) is lot
    assert B.give_up_the_frontage(lot, shapely_box(-10, -6, 30, -1), 0.0) is lot
