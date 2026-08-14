"""Structure under an elevated road. Geometry only — no Blender, no map."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
import pytest

from city_builder import viaduct as V
from city_builder.geometry import Ribbon
from city_builder.ground import HeightMap
from city_builder.viaduct import ViaductOptions


def _lane(lane_id, y0, y1, *, z=10.0, z_end=None, x0=0.0, x1=100.0, n=21):
    """A lane from x0 to x1 between y0 and y1, level at z or ramping to z_end."""
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    zs = [z if z_end is None else z + (z_end - z) * i / (n - 1) for i in range(n)]
    return Ribbon(lane_id, [(x, y0, h) for x, h in zip(xs, zs)],
                  [(x, y1, h) for x, h in zip(xs, zs)])


def _ground(value=0.0, nx=41, ny=41, cell=10.0):
    return HeightMap(-100.0, -100.0, cell, np.full((ny, nx), value, dtype=float),
                     np.zeros((ny, nx), dtype=float))


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_a_negative_dimension_is_refused():
    with pytest.raises(ValueError):
        ViaductOptions(parapet_height=-1.0)


def test_piers_need_a_spacing():
    with pytest.raises(ValueError):
        ViaductOptions(piers=True, pier_spacing=0.0)
    ViaductOptions(piers=False, pier_spacing=0.0)  # off: nobody cares


# ---------------------------------------------------------------------------
# Where the road is a bridge — the rule everything else hangs off
# ---------------------------------------------------------------------------


def test_clearance_is_measured_from_the_terrain_under_the_lane():
    profile = V.clearance_profile(_lane(1, -2, 2, z=10.0), _ground(4.0))
    assert all(c == pytest.approx(6.0) for c in profile)


def test_only_the_high_part_of_a_ramp_is_a_bridge():
    """The failure this rewrite exists for.

    A ramp is one lanelet running from deck height down to grade. Treating the
    lanelet as elevated and extruding all of it downwards drives the soffit
    through the street below — measured on the real map, girders resting on the
    ground-level carriageway.
    """
    ramp = _lane(1, -2, 2, z=10.0, z_end=0.0, x0=0.0, x1=100.0, n=21)
    options = ViaductOptions(bridge_clearance=2.0, min_bridge_length=5.0)
    runs = V.bridge_runs(ramp, _ground(0.0), options)

    assert len(runs) == 1
    start, end = runs[0]
    assert start == 0
    line = V.centreline(ramp)
    assert line[end][2] >= 2.0
    assert line[end + 1][2] < 2.0  # it stops as soon as the road reaches grade


def test_a_road_lying_on_the_ground_is_not_a_bridge():
    flat = _lane(1, -2, 2, z=0.2)
    assert V.bridge_runs(flat, _ground(0.0), ViaductOptions(bridge_clearance=2.0)) == []


def test_a_brief_hop_over_a_dip_is_not_a_viaduct():
    lane = _lane(1, -2, 2, z=5.0, x0=0.0, x1=100.0, n=21)
    terrain = _ground(0.0)
    short = ViaductOptions(bridge_clearance=2.0, min_bridge_length=500.0)
    long_enough = ViaductOptions(bridge_clearance=2.0, min_bridge_length=10.0)
    assert V.bridge_runs(lane, terrain, short) == []
    assert V.bridge_runs(lane, terrain, long_enough)


def test_runs_of_finds_the_stretches():
    assert V.runs_of([False, True, True, True, False, True, True]) == [(1, 3), (5, 6)]
    assert V.runs_of([True, False, True]) == []  # a lone sample is not a stretch
    assert V.runs_of([]) == []


def test_slicing_keeps_the_identity_of_the_lane():
    section = V.slice_ribbon(_lane(7, -2, 2, n=11), 3, 6)
    assert section.id == 7
    assert len(section.left) == 4 and len(section.right) == 4


# ---------------------------------------------------------------------------
# Which boundaries face open air
# ---------------------------------------------------------------------------


def test_only_the_outermost_lanes_have_an_outer_edge():
    """A parapet belongs on the edge of the carriageway, not between its lanes."""
    lanes = [_lane(1, -6, -2), _lane(2, -2, 2), _lane(3, 2, 6)]
    outline = V.deck_outline(lanes, {1, 2, 3})

    # The two ends of every lane really are on the outline — the deck stops
    # there — so the question is about the length in between.
    def middle(flags):
        return flags[1:-1]

    left, right = V.outer_flags(lanes[0], outline, 0.7)
    assert all(middle(left)) and not any(middle(right))   # only its far side is open

    left, right = V.outer_flags(lanes[1], outline, 0.7)
    assert not any(middle(left)) and not any(middle(right))  # the middle lane has none

    left, right = V.outer_flags(lanes[2], outline, 0.7)
    assert not any(middle(left)) and all(middle(right))


def test_a_lane_on_its_own_is_outer_on_both_sides():
    lane = _lane(1, -2, 2)
    left, right = V.outer_flags(lane, V.deck_outline([lane], {1}), 0.7)
    assert all(left) and all(right)


def test_a_survey_gap_between_lanes_is_not_open_air():
    """Neighbouring lanelets do not quite meet; the sliver must not read as an edge."""
    lanes = [_lane(1, -6.0, -2.0), _lane(2, -1.7, 2.0)]  # 30 cm apart
    outline = V.deck_outline(lanes, {1, 2}, close_gap=0.5)
    _left, right = V.outer_flags(lanes[0], outline, 0.7)
    assert not any(right[1:-1])


def test_a_deck_just_past_a_boundary_does_not_hide_the_edge():
    """Probing a fixed distance sideways under-reads: measured, it found 3323 m
    of edge where the outline finds 4199 m, because a separate structure passing
    within the probe distance stops it from saying so."""
    lanes = [_lane(1, -6.0, -2.0), _lane(2, -0.4, 3.0)]  # 1.6 m apart, not joined
    outline = V.deck_outline(lanes, {1, 2}, close_gap=0.5)
    _left, right = V.outer_flags(lanes[0], outline, 0.7)
    assert all(right), "the deck really does stop here"


def test_the_end_of_a_deck_is_on_the_outline_too():
    """It is: the structure stops there, whatever the lane beside it is doing."""
    lanes = [_lane(1, -6, -2), _lane(2, -2, 2)]
    left, _right = V.outer_flags(lanes[1], V.deck_outline(lanes, {1, 2}), 0.7)
    assert left[0] and left[-1]
    assert not any(left[1:-1])


def test_ground_level_lanes_are_not_part_of_the_structure():
    lanes = [_lane(1, -6, -2), _lane(2, -2, 2)]
    outline = V.deck_outline(lanes, {1})  # lane 2 is at grade
    _left, right = V.outer_flags(lanes[0], outline, 0.7)
    assert all(right), "a road at grade is not a neighbouring deck"


# ---------------------------------------------------------------------------
# Deck
# ---------------------------------------------------------------------------


def test_the_deck_hangs_below_the_surveyed_surface():
    mesh = V.deck_shell(_lane(1, -2, 2, z=10.0), 1.2)
    zs = [v[2] for v in mesh.vertices]
    assert max(zs) == pytest.approx(10.0)
    assert min(zs) == pytest.approx(8.8)


def test_a_shared_side_gets_no_face():
    lane = _lane(1, -2, 2, n=5)
    both = V.deck_shell(lane, 1.2)
    one = V.deck_shell(lane, 1.2, right_outer=[False] * 5)
    assert len(one.faces) < len(both.faces)


def test_a_joint_with_the_next_lanelet_is_not_capped():
    lane = _lane(1, -2, 2)
    capped = V.deck_shell(lane, 1.2)
    through = V.deck_shell(lane, 1.2, cap_start=False, cap_end=False)
    assert len(capped.faces) - len(through.faces) == 2


def test_a_deck_with_no_thickness_is_no_deck():
    assert not V.deck_shell(_lane(1, -2, 2), 0.0).faces


# ---------------------------------------------------------------------------
# Parapets
# ---------------------------------------------------------------------------


def test_a_parapet_stands_on_the_deck_edge():
    lane = _lane(1, -4, 4, z=10.0, n=11)
    walls = V.parapet_walls(lane, [True] * 11, [False] * 11,
                            ViaductOptions(parapet_height=1.1))
    assert len(walls) == 1
    zs = [v[2] for v in walls[0].vertices]
    assert min(zs) == pytest.approx(10.0)
    assert max(zs) == pytest.approx(11.1)
    # _lane puts the left boundary at y0, which is the negative side here
    assert all(v[1] < -3.0 for v in walls[0].vertices), "it should be on the left edge"


def test_no_parapet_where_the_lane_has_a_neighbour():
    lane = _lane(1, -4, 4, n=11)
    assert V.parapet_walls(lane, [False] * 11, [False] * 11, ViaductOptions()) == []


def test_a_parapet_follows_only_the_stretch_that_is_outermost():
    """Where a slip road peels away, the barrier starts there and not before."""
    lane = _lane(1, -4, 4, x0=0.0, x1=200.0, n=21)
    flags = [False] * 11 + [True] * 10
    walls = V.parapet_walls(lane, flags, [False] * 21,
                            ViaductOptions(parapet_min_length=1.0))
    assert len(walls) == 1
    assert min(v[0] for v in walls[0].vertices) >= 100.0


def test_a_stub_of_wall_is_worse_than_none():
    lane = _lane(1, -4, 4, x0=0.0, x1=200.0, n=21)
    flags = [False] * 19 + [True] * 2  # 10 m of edge
    assert V.parapet_walls(lane, flags, [False] * 21,
                           ViaductOptions(parapet_min_length=1.0))
    assert not V.parapet_walls(lane, flags, [False] * 21,
                               ViaductOptions(parapet_min_length=50.0))


def test_parapets_can_be_turned_off():
    lane = _lane(1, -4, 4, n=11)
    assert V.parapet_walls(lane, [True] * 11, [True] * 11,
                           ViaductOptions(parapets=False)) == []


# ---------------------------------------------------------------------------
# Piers
# ---------------------------------------------------------------------------


def test_piers_stand_from_the_ground_to_the_soffit():
    options = ViaductOptions(pier_spacing=25.0, deck_thickness=1.2, pier_embed=0.6)
    for pier in V.pier_boxes(_lane(1, -2, 2, z=10.0), _ground(0.0), options):
        zs = [v[2] for v in pier.vertices]
        assert min(zs) == pytest.approx(-0.6)  # embedded
        assert max(zs) == pytest.approx(8.8)   # the soffit, not the road


def test_piers_are_spaced_along_the_road():
    boxes = V.pier_boxes(_lane(1, -2, 2, x0=0.0, x1=200.0, n=41), _ground(),
                         ViaductOptions(pier_spacing=25.0))
    xs = sorted(sum(v[0] for v in b.vertices) / len(b.vertices) for b in boxes)
    gaps = [b - a for a, b in pairwise(xs)]
    assert gaps and all(abs(g - 25.0) < 2.0 for g in gaps), gaps


def test_no_pier_where_the_deck_is_nearly_at_grade():
    options = ViaductOptions(pier_spacing=20.0, pier_min_clearance=3.0, deck_thickness=1.2)
    assert not V.pier_boxes(_lane(1, -2, 2, z=3.0), _ground(1.5), options)


def test_a_box_is_closed_and_the_right_size():
    mesh = V.box((0.0, 0.0, 0.0), width=2.0, depth=3.0, bottom=-1.0, top=4.0, heading=0.0)
    assert len(mesh.vertices) == 8
    assert len(mesh.faces) == 6
    assert max(v[0] for v in mesh.vertices) - min(v[0] for v in mesh.vertices) == pytest.approx(3.0)
    assert max(v[1] for v in mesh.vertices) - min(v[1] for v in mesh.vertices) == pytest.approx(2.0)
    assert max(v[2] for v in mesh.vertices) - min(v[2] for v in mesh.vertices) == pytest.approx(5.0)


def test_a_box_turns_with_the_road():
    turned = V.box((0.0, 0.0, 0.0), 2.0, 3.0, 0.0, 1.0, math.pi / 2)
    xs = [v[0] for v in turned.vertices]
    assert max(xs) - min(xs) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_nothing_is_built_below_the_road_that_lies_on_the_ground():
    """The regression: a ramp's soffit must never end up under the terrain."""
    ramp = _lane(1, -2, 2, z=10.0, z_end=0.0, x0=0.0, x1=100.0, n=21)
    options = ViaductOptions(bridge_clearance=2.0, min_bridge_length=5.0, deck_thickness=1.2)
    built = V.build([ramp], {1}, _ground(0.0), options)

    for group, meshes in built.items():
        for mesh in meshes:
            if group == "ViaductPiers":
                continue  # a pier is meant to reach the ground
            assert min(v[2] for v in mesh.vertices) >= -1e-6, f"{group} went below the terrain"


def test_only_elevated_lanelets_get_structure():
    lanes = [_lane(1, -2, 2, z=10.0), _lane(2, 20, 24, z=10.0)]
    built = V.build(lanes, {1}, _ground(0.0), ViaductOptions())
    ys = [v[1] for mesh in built["ViaductDecks"] for v in mesh.vertices]
    assert ys and max(ys) < 10.0


def test_a_middle_lane_gets_a_deck_but_no_parapet():
    lanes = [_lane(1, -6, -2, z=10.0), _lane(2, -2, 2, z=10.0), _lane(3, 2, 6, z=10.0)]
    built = V.build(lanes, {1, 2, 3}, _ground(0.0), ViaductOptions())
    assert len(built["ViaductDecks"]) == 3

    ys = [v[1] for mesh in built["ViaductParapets"] for v in mesh.vertices]
    assert ys, "the carriageway edges should still have barriers"
    assert not [y for y in ys if -1.5 < y < 1.5], "a wall was built between the lanes"


def test_without_a_terrain_nothing_can_be_decided():
    built = V.build([_lane(1, -2, 2)], {1}, None, ViaductOptions())
    assert built == {"ViaductDecks": [], "ViaductParapets": [],
                     "ViaductPiers": [], "ViaductInfill": []}


# ---------------------------------------------------------------------------
# Infill — the slivers lanelets leave between themselves
# ---------------------------------------------------------------------------


def test_a_gap_between_lanes_is_patched():
    """On the ground a sliver shows the terrain; on a deck it is a slot to the street."""
    lanes = [_lane(1, -6.0, -2.0, z=10.0), _lane(2, -1.7, 2.0, z=10.0)]  # 30 cm apart
    patches = V.deck_infill(lanes, ViaductOptions(infill_gap=0.8))
    assert patches

    ys = [v[1] for mesh in patches for v in mesh.vertices]
    assert -2.1 < min(ys) and max(ys) < -1.6, "the patch should sit in the gap"
    zs = [v[2] for mesh in patches for v in mesh.vertices]
    assert all(z == pytest.approx(10.0) for z in zs), "and at the height of the deck"


def test_lanes_that_meet_need_no_patch():
    lanes = [_lane(1, -6.0, -2.0, z=10.0), _lane(2, -2.0, 2.0, z=10.0)]
    assert V.deck_infill(lanes, ViaductOptions(infill_gap=0.8)) == []


def test_a_real_opening_is_left_alone():
    """Two carriageways with a proper gap between them are not a survey artefact."""
    lanes = [_lane(1, -20.0, -10.0, z=10.0), _lane(2, 10.0, 20.0, z=10.0)]
    assert V.deck_infill(lanes, ViaductOptions(infill_gap=0.8)) == []


def test_infill_can_be_turned_off():
    lanes = [_lane(1, -6.0, -2.0, z=10.0), _lane(2, -1.7, 2.0, z=10.0)]
    assert V.deck_infill(lanes, ViaductOptions(infill=False)) == []


def test_the_patch_follows_a_sloping_deck():
    lanes = [_lane(1, -6.0, -2.0, z=10.0, z_end=16.0),
             _lane(2, -1.7, 2.0, z=10.0, z_end=16.0)]
    patches = V.deck_infill(lanes, ViaductOptions(infill_gap=0.8))
    zs = [v[2] for mesh in patches for v in mesh.vertices]
    assert min(zs) == pytest.approx(10.0, abs=0.5)
    assert max(zs) == pytest.approx(16.0, abs=0.5)


# ---------------------------------------------------------------------------
# Crossings sit on the carriageway rather than beside it
# ---------------------------------------------------------------------------


def test_a_crossing_is_cut_down_to_the_road_it_lies_on():
    """Measured on the map: 67 of 84 crossings overlapped the carriageway, a
    median 81 % of their area, at a median 7 cm apart in z."""
    from city_builder.build import clip_crosswalks

    road = _lane(1, -6.0, 6.0, z=10.0, x0=0.0, x1=100.0, n=11)
    crossing = _lane(2, -12.0, 12.0, z=9.93, x0=40.0, x1=45.0, n=3)  # spills over both kerbs
    groups = {"Roads": [road], "Crosswalks": [crossing]}
    clip_crosswalks(groups, 0.005)

    assert groups["Crosswalks"]
    ys = [v[1] for mesh in groups["Crosswalks"] for v in mesh.vertices]
    assert min(ys) >= -6.1 and max(ys) <= 6.1, "it should stop at the carriageway edge"

    zs = [v[2] for mesh in groups["Crosswalks"] for v in mesh.vertices]
    assert all(z == pytest.approx(10.005) for z in zs), "and sit just on top of the road"


def test_a_crossing_with_no_road_under_it_is_dropped():
    from city_builder.build import clip_crosswalks

    groups = {"Roads": [_lane(1, -6.0, 6.0, z=10.0)],
              "Crosswalks": [_lane(2, 40.0, 50.0, z=10.0, x0=0.0, x1=10.0, n=3)]}
    clip_crosswalks(groups, 0.005)
    assert groups["Crosswalks"] == []


# ---------------------------------------------------------------------------
# The barrier is continuous, and a kerb needs something to stop at
# ---------------------------------------------------------------------------


def test_a_flicker_in_the_probe_does_not_break_the_barrier():
    """Measured, 109 candidate runs carried 125 flips and 17 came out too short.

    The probe catches a neighbouring deck for a cross-section or two at a
    junction mouth. A barrier with holes punched in it by that is worse than
    one that ignores them.
    """
    stations = [float(i) for i in range(0, 40, 2)]  # 2 m apart
    flags = [True] * 20
    flags[9] = flags[10] = False  # a 2 m blink
    assert V.runs_of(flags) == [(0, 8), (11, 19)]
    assert V.runs_of(V.close_gaps(flags, stations, 5.0)) == [(0, 19)]


def test_a_real_opening_still_breaks_it():
    stations = [float(i) for i in range(0, 40, 2)]
    flags = [True] * 6 + [False] * 8 + [True] * 6  # 16 m of neighbour
    assert len(V.runs_of(V.close_gaps(flags, stations, 5.0))) == 2


def test_an_open_end_is_never_closed():
    stations = [float(i) for i in range(0, 20, 2)]
    flags = [False, False] + [True] * 8
    assert V.close_gaps(flags, stations, 50.0)[0] is False


def test_a_kerb_between_two_lanes_is_not_a_kerb():
    """A road_border is only a kerb where something stops at it."""
    from city_builder.build import clip_curbs
    from city_builder.geometry import Ribbon

    divider = [(x, 0.0, 0.0) for x in range(0, 31, 5)]
    groups = {
        "Roads": [_lane(1, -4.0, 0.0), _lane(2, 0.0, 4.0)],  # carriageway either side
        "Curbs": [Ribbon(9, divider, [(x, y, z + 0.15) for x, y, z in divider])],
    }
    clip_curbs(groups)
    assert groups["Curbs"] == []


def test_the_kerb_at_the_edge_of_the_road_survives():
    from city_builder.build import clip_curbs
    from city_builder.geometry import Ribbon

    edge = [(x, 4.0, 0.0) for x in range(0, 31, 5)]  # open ground on one side
    groups = {
        "Roads": [_lane(1, -4.0, 4.0)],
        "Curbs": [Ribbon(9, edge, [(x, y, z + 0.15) for x, y, z in edge])],
    }
    clip_curbs(groups)
    assert groups["Curbs"]
    assert len(groups["Curbs"][0].left) == len(edge)


def test_an_infilled_gap_is_deck_and_so_has_no_edge():
    """The barrier used to run all the way round the island between two turning
    lanes: the patch filled the gap, and the outline still had the hole."""
    ring = [_lane(1, -8.0, -4.0, x0=0.0, x1=60.0),
            _lane(2, 4.0, 8.0, x0=0.0, x1=60.0),
            _lane(3, -4.0, 4.0, x0=0.0, x1=6.0),
            _lane(4, -4.0, 4.0, x0=54.0, x1=60.0)]
    options = ViaductOptions(infill_gap=0.8, infill_max_area=1e6)
    patches, _covered = V.infill_polygons(ring, options)
    assert patches, "the island between the lanes should be patched"

    with_hole = V.deck_outline(ring, close_gap=0.5)
    filled = V.deck_outline(ring, close_gap=0.5, patches=patches)
    inner = np.array([[30.0, 0.0]])  # the middle of the island

    import shapely
    assert shapely.distance(with_hole, shapely.points(inner))[0] < 5.0
    assert shapely.distance(filled, shapely.points(inner))[0] > 7.0


def test_the_infill_is_the_same_geometry_the_outline_was_told_about():
    """Two descriptions of the same gap drift; one description cannot."""
    lanes = [_lane(1, -6.0, -2.0), _lane(2, -1.7, 2.0)]
    options = ViaductOptions(infill_gap=0.8)
    patches, _ = V.infill_polygons(lanes, options)
    meshes = V.deck_infill(lanes, options, patches)
    assert len(meshes) == len(patches)
