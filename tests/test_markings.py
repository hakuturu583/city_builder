"""Paint baked into the carriageway. Geometry and pixels only — no Blender."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from city_builder import markings as M
from city_builder.geometry import Polygon, Ribbon
from city_builder.markings import LaneFrame, MarkingOptions


def _lane(lane_id=1, y0=-1.5, y1=1.5, x0=0.0, x1=30.0, n=7, z=0.0):
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    return Ribbon(lane_id, [(x, y0, z) for x in xs], [(x, y1, z) for x in xs])


def _stripe(shape_id, x0, x1, y0, y1, z=0.0):
    """A painted bar, as the extractor builds one: a widened polyline."""
    return Ribbon(shape_id, [(x0, y0, z), (x1, y0, z)], [(x0, y1, z), (x1, y1, z)])


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_a_page_must_be_a_whole_number_of_steps():
    """Otherwise the last column runs off the edge and takes its lane with it."""
    with pytest.raises(ValueError):
        MarkingOptions(column_step=48, page_pixels=4096)
    MarkingOptions(column_step=32, page_pixels=4096)


def test_a_texel_has_to_have_a_size():
    with pytest.raises(ValueError):
        MarkingOptions(texel_metres=0.0)
    with pytest.raises(ValueError):
        MarkingOptions(column_step=4)


# ---------------------------------------------------------------------------
# A lane's own coordinates
# ---------------------------------------------------------------------------


def test_the_centreline_is_halfway_across():
    frame = LaneFrame(_lane())
    uv = frame.project(np.array([[15.0, 0.0]]))
    assert uv[0][0] == pytest.approx(0.5, abs=0.02)
    assert uv[0][1] == pytest.approx(0.5, abs=0.02)


def test_the_boundaries_are_the_edges():
    frame = LaneFrame(_lane())
    left = frame.project(np.array([[15.0, -1.5]]))[0]
    right = frame.project(np.array([[15.0, 1.5]]))[0]
    assert left[1] == pytest.approx(0.0, abs=0.02)
    assert right[1] == pytest.approx(1.0, abs=0.02)


def test_along_the_lane_runs_zero_to_one():
    frame = LaneFrame(_lane(x0=0.0, x1=30.0))
    assert frame.project(np.array([[0.0, 0.0]]))[0][0] == pytest.approx(0.0, abs=0.02)
    assert frame.project(np.array([[30.0, 0.0]]))[0][0] == pytest.approx(1.0, abs=0.02)


def test_a_point_beyond_the_kerb_lands_outside():
    """The rasteriser clips; nothing else has to know about the edge."""
    frame = LaneFrame(_lane())
    assert frame.project(np.array([[15.0, 4.0]]))[0][1] > 1.0


def test_the_frame_follows_a_curve():
    """A lane that turns still has its own straight coordinates."""
    points = [(np.cos(t) * 20, np.sin(t) * 20) for t in np.linspace(0, 1.2, 9)]
    ribbon = Ribbon(1, [(x, y, 0.0) for x, y in points],
                    [(x * 1.15, y * 1.15, 0.0) for x, y in points])
    frame = LaneFrame(ribbon)
    middle = [( (a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
              for a, b in zip(ribbon.left, ribbon.right)]
    uv = frame.project(np.asarray(middle))
    assert all(abs(v - 0.5) < 0.05 for _u, v in uv)
    assert list(uv[:, 0]) == sorted(uv[:, 0])


# ---------------------------------------------------------------------------
# Sizing and packing
# ---------------------------------------------------------------------------


def test_a_texel_is_the_same_size_on_every_lane():
    """A fixed count across a lane made a 15 cm line anywhere from 1.0 to 6.7
    texels wide across this map, and it visibly thinned and thickened."""
    options = MarkingOptions(texel_metres=0.05, column_step=32)
    narrow = M.lane_pixels(LaneFrame(_lane(y0=-1.0, y1=1.0, x0=0, x1=30)), options)
    wide = M.lane_pixels(LaneFrame(_lane(y0=-4.0, y1=4.0, x0=0, x1=30)), options)

    assert wide[0] > narrow[0], "a wider lane needs a wider strip"
    assert narrow[1] == wide[1] == pytest.approx(30 / 0.05, rel=0.02)


def test_a_strip_is_wide_enough_for_its_lane():
    options = MarkingOptions(texel_metres=0.05, column_step=32)
    width, _height = M.lane_pixels(LaneFrame(_lane(y0=-1.5, y1=1.5)), options)
    assert width >= 3.0 / 0.05
    assert width % options.column_step == 0


def test_strips_fill_a_column_then_start_the_next():
    options = MarkingOptions(column_step=32, page_pixels=256)
    places = M.pack([(64, 200), (64, 100), (64, 100)], options)
    assert (places[0].page, places[0].x, places[0].y) == (0, 0, 0)
    assert (places[1].page, places[1].x, places[1].y) == (0, 64, 0)
    assert (places[2].page, places[2].x, places[2].y) == (0, 64, 100)


def test_strips_of_different_widths_get_their_own_columns():
    """Widest first, so a column's strips share a width and the packing is
    shelf packing rather than the general rectangle problem."""
    options = MarkingOptions(column_step=32, page_pixels=512)
    places = M.pack([(64, 100), (128, 100), (64, 100)], options)
    assert places[1].x == 0 and places[1].width == 128   # the widest goes first
    assert places[0].x == places[2].x == 128
    assert {p.page for p in places} == {0}


def test_a_full_page_starts_another():
    options = MarkingOptions(column_step=32, page_pixels=128)  # two 64 columns
    places = M.pack([(64, 128), (64, 128), (64, 128)], options)
    assert sorted(p.page for p in places) == [0, 0, 1]


def test_a_lane_with_no_length_is_skipped():
    options = MarkingOptions(column_step=32, page_pixels=256)
    assert M.pack([(0, 0)], options) == [None]


# ---------------------------------------------------------------------------
# Rasterising
# ---------------------------------------------------------------------------


def test_paint_inside_the_lane_is_drawn():
    frame = LaneFrame(_lane(y0=-1.5, y1=1.5, x0=0, x1=30))
    stripe = _stripe(2, 10.0, 12.0, -1.4, 1.4)
    image = np.asarray(M.rasterise(frame, [stripe], (64, 640)))
    assert image.max() == 255
    rows = np.flatnonzero(image.max(axis=1) > 0)
    assert 200 < rows.mean() < 260, "the bar should land a third of the way along"


def test_paint_outside_the_lane_is_clipped_away():
    """A zebra bar runs across the footway too; the strip stops at the kerb."""
    frame = LaneFrame(_lane(y0=-1.5, y1=1.5))
    beyond = _stripe(2, 10.0, 12.0, 6.0, 9.0)
    image = np.asarray(M.rasterise(frame, [beyond], (64, 640)))
    assert image.max() == 0


def test_a_bar_that_overhangs_is_kept_only_where_the_road_is():
    frame = LaneFrame(_lane(y0=-1.5, y1=1.5))
    overhanging = _stripe(2, 10.0, 12.0, -8.0, 8.0)  # far past both kerbs
    image = np.asarray(M.rasterise(frame, [overhanging], (64, 640)))
    assert image.max() == 255
    # every column has paint in it, and there are only 64 columns to have
    assert (image > 0).any(axis=0).sum() == 64


def test_a_zebra_ring_is_filled():
    frame = LaneFrame(_lane())
    ring = Polygon(3, [(10.0, -1.0, 0.0), (12.0, -1.0, 0.0),
                       (12.0, 1.0, 0.0), (10.0, 1.0, 0.0), (10.0, -1.0, 0.0)])
    assert np.asarray(M.rasterise(frame, [ring], (64, 640))).max() == 255


# ---------------------------------------------------------------------------
# UVs onto the strip
# ---------------------------------------------------------------------------


def test_uvs_match_the_order_the_mesh_builds_vertices():
    """Left then right, one pair per cross-section — the same interleaving."""
    ribbon = _lane(n=7)
    frame = LaneFrame(ribbon)
    place = M.Placement(0, 64, 128, 64, 640)
    uvs = M.strip_uvs(frame, place, MarkingOptions(page_pixels=4096))
    assert len(uvs) == 2 * len(frame)
    assert all(0.0 <= u <= 1.0 and 0.0 <= v <= 1.0 for u, v in uvs)
    assert uvs[0][0] < uvs[1][0]        # left edge of the strip, then right
    assert uvs[0][1] > uvs[-1][1]       # and the lane runs down the page


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------


def test_baking_takes_the_paint_off_the_road_and_puts_it_in_the_texture():
    groups = {
        "Roads": [_lane(1, x0=0.0, x1=30.0)],
        "LaneMarkings": [_stripe(2, 0.0, 30.0, -1.45, -1.30)],
        "StopLines": [_stripe(3, 28.0, 28.4, -1.5, 1.5)],
    }
    pages, page_of_shape = M.bake(groups, MarkingOptions(page_pixels=1024))

    assert len(pages) == 1
    assert pages[0].max() == 255
    assert "LaneMarkings" not in groups and "StopLines" not in groups
    assert page_of_shape["Roads"] == [0]
    assert groups["Roads"][0].uvs is not None


def test_nothing_happens_without_paint():
    groups = {"Roads": [_lane()]}
    assert M.bake(groups, MarkingOptions()) == ([], {})
    assert groups["Roads"][0].uvs is None


def test_the_geometry_path_is_still_there():
    groups = {"Roads": [_lane()], "LaneMarkings": [_stripe(2, 0.0, 30.0, -1.4, -1.3)]}
    assert M.bake(groups, MarkingOptions(texture=False)) == ([], {})
    assert groups["LaneMarkings"], "the slabs should survive when texturing is off"


def test_the_mapping_is_the_exact_inverse_of_the_mesh_uv():
    """Projecting onto the centreline instead makes the painted line wander.

    A cross-section is the chord between two boundary points, and on a curve
    that chord is not perpendicular to the centreline, so a station taken by
    perpendicular projection is off by an amount that grows with curvature.
    Round-tripping the ribbon's own vertices is the check.
    """
    angles = np.linspace(0.0, 1.4, 12)
    inner = [(np.cos(t) * 18, np.sin(t) * 18) for t in angles]
    outer = [(np.cos(t) * 21, np.sin(t) * 21) for t in angles]
    ribbon = Ribbon(1, [(x, y, 0.0) for x, y in inner], [(x, y, 0.0) for x, y in outer])
    frame = LaneFrame(ribbon)

    expected_u = frame.station / frame.length
    for side, points, want_v in ((0, inner, 0.0), (1, outer, 1.0)):
        uv = frame.project(np.asarray(points))
        assert np.allclose(uv[:, 1], want_v, atol=1e-6), f"side {side} drifted across the lane"
        assert np.allclose(uv[:, 0], expected_u, atol=1e-6), f"side {side} drifted along it"


def test_a_line_painted_down_the_middle_stays_in_the_middle():
    angles = np.linspace(0.0, 1.4, 12)
    inner = [(np.cos(t) * 18, np.sin(t) * 18) for t in angles]
    outer = [(np.cos(t) * 21, np.sin(t) * 21) for t in angles]
    ribbon = Ribbon(1, [(x, y, 0.0) for x, y in inner], [(x, y, 0.0) for x, y in outer])
    frame = LaneFrame(ribbon)

    middle = np.asarray([((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in zip(inner, outer)])
    assert np.allclose(frame.project(middle)[:, 1], 0.5, atol=1e-6)


def test_supersampling_softens_the_stair_steps():
    """A lane line crosses the strip at an angle more often than not."""
    frame = LaneFrame(_lane(y0=-1.5, y1=1.5, x0=0.0, x1=30.0))
    diagonal = Ribbon(9, [(0.0, -1.4, 0.0), (30.0, 1.0, 0.0)],
                      [(0.0, -1.2, 0.0), (30.0, 1.2, 0.0)])
    hard = np.asarray(M.rasterise(frame, [diagonal], (64, 640), 1))
    soft = np.asarray(M.rasterise(frame, [diagonal], (64, 640), 2))

    assert set(np.unique(hard)) <= {0, 255}
    assert len(np.unique(soft)) > 2, "the edge should carry partial coverage"
    assert soft.max() > 200 and (soft > 0).sum() >= (hard > 0).sum()


def test_no_lane_line_is_painted_inside_a_junction():
    """A lane line stops at the intersection: inside it there is nothing to
    separate, because every turning path crosses every other one."""
    groups = {
        "Junctions": [_lane(1, x0=0.0, x1=30.0)],
        "LaneMarkings": [_stripe(2, 0.0, 30.0, -1.45, -1.30)],
    }
    pages, _ = M.bake(groups, MarkingOptions(page_pixels=1024))
    assert pages and pages[0].max() == 0


def test_a_stop_line_and_a_crossing_still_land_on_a_junction():
    groups = {
        "Junctions": [_lane(1, x0=0.0, x1=30.0)],
        "StopLines": [_stripe(2, 1.0, 1.4, -1.5, 1.5)],
    }
    pages, _ = M.bake(groups, MarkingOptions(page_pixels=1024))
    assert pages and pages[0].max() == 255


def test_the_same_line_is_still_painted_on_the_road_it_belongs_to():
    groups = {
        "Roads": [_lane(1, x0=0.0, x1=30.0)],
        "LaneMarkings": [_stripe(2, 0.0, 30.0, -1.45, -1.30)],
    }
    pages, _ = M.bake(groups, MarkingOptions(page_pixels=1024))
    assert pages and pages[0].max() == 255


def test_junction_lane_lines_can_be_asked_for():
    groups = {
        "Junctions": [_lane(1, x0=0.0, x1=30.0)],
        "LaneMarkings": [_stripe(2, 0.0, 30.0, -1.45, -1.30)],
    }
    pages, _ = M.bake(groups, MarkingOptions(page_pixels=1024, lane_lines_in_junctions=True))
    assert pages and pages[0].max() == 255


def test_the_carriageway_leaves_the_scene_as_asphalt_and_not_as_black():
    """The page is a mask; what a mesh format carries has to be a colour.

    A mix driven by the mask renders correctly in Blender and cannot be
    exported, so an exporter takes the mask itself and the road leaves as white
    lines on black. That is what happened: an exported carriageway texture came
    out at a mean of 0.001, the splat cloud built from it started black, and the
    generator, seeded with a render of that cloud, painted the black road it was
    being shown. The composite has to exist as pixels before export.
    """
    from PIL import Image

    from city_builder.scene import painted_page

    page = np.zeros((64, 64), dtype=np.uint8)
    page[30:34, :] = 255                      # one painted line across it
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "page00.png")
        Image.fromarray(page).save(path)

        asphalt = (0.055, 0.055, 0.058)
        out = painted_page(path, asphalt, (0.9, 0.9, 0.88))
        baked = np.asarray(Image.open(out).convert("RGB"), dtype=float) / 255.0

    # The page is an eight-bit file and a base colour socket is linear, so the
    # asphalt has to arrive encoded or it comes back twelve times too dark.
    from city_builder.conditioning import to_srgb

    expected = float(to_srgb(np.mean(asphalt)))
    unpainted = baked[0]
    assert unpainted.mean() == pytest.approx(expected, abs=0.01), \
        "off the paint the road has to be the asphalt, in the file's own space"
    assert baked[32].mean() > 0.8, "the line itself still has to be painted"
    assert baked.mean() > 0.2, "a road that averages near zero is the bug"
