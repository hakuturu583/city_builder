"""The facade layout: exact where it claims to be exact, and measurably so."""

from __future__ import annotations

import random
from itertools import pairwise

import numpy as np
import pytest

from city_builder.facade_layout import (
    FacadeLayout,
    bay_alignment,
    bays_for,
    bays_in,
    control_image,
    diversity,
    floor_alignment,
    procedural_facade,
    sample_layout,
    saturation,
    sheet_floors,
    sheet_name,
    wrap_seam,
)
from city_builder.texture import seam_error_axis

# ---------------------------------------------------------------------------
# Structure
# ---------------------------------------------------------------------------


def test_floor_lines_span_exactly_the_building():
    """V=0 is the pavement and V=1 the roofline, with nothing left over.

    The sheet is stretched over the wall by the UV, so an approximate total
    would show up as the top floor being clipped or a stripe of wall above it.
    """
    lines = FacadeLayout(floors=7).floor_lines()
    assert lines[0] == 0.0
    assert lines[-1] == 1.0
    assert len(lines) == 8
    assert all(b > a for a, b in pairwise(lines))


def test_ground_floor_is_the_tall_one():
    lines = FacadeLayout(floors=5, ground_floor_ratio=1.4).floor_lines()
    bands = [b - a for a, b in pairwise(lines)]
    assert bands[0] == max(bands)
    assert bands[1] == pytest.approx(bands[2])  # the flats above are uniform


def test_upper_floors_are_evenly_spaced():
    """The rhythm the whole design depends on: identical bands above the shop."""
    lines = FacadeLayout(floors=9).floor_lines()
    bands = [b - a for a, b in pairwise(lines[1:])]
    assert np.std(bands) < 1e-12


def test_a_single_storey_building_still_works():
    layout = FacadeLayout(floors=1)
    assert layout.floor_lines() == [0.0, 1.0]
    assert layout.windows()  # the shopfront, and nothing above it


def test_bays_wrap():
    """U=0 and U=1 land on the same place in the pattern, so the sheet repeats."""
    bays = FacadeLayout(floors=4, bays=5).bay_lines()
    assert bays[0] == 0.0
    assert bays[-1] == 1.0
    assert len(bays) == 6


def test_bays_come_from_metres_not_from_a_guess():
    assert bays_for(12.0, 3.0) == 4
    assert bays_for(6.0, 3.0) == 2
    assert bays_for(1.0, 3.0) == 1  # never zero


def test_windows_stay_inside_their_floor():
    layout = FacadeLayout(floors=6)
    lines = layout.floor_lines()
    for _, _, v0, v1 in layout.windows():
        floor = max(i for i, line in enumerate(lines) if line <= v0 + 1e-9)
        assert lines[floor] <= v0 < v1 <= lines[floor + 1] + 1e-9


def test_nothing_is_punched_through_the_parapet():
    layout = FacadeLayout(floors=3, parapet=0.05, window_height=0.95, window_base=0.04)
    assert max(v1 for *_, v1 in layout.windows()) <= 1.0 - layout.parapet + 1e-9


def test_shopfront_is_wider_than_a_window():
    layout = FacadeLayout(floors=4)
    rects = layout.windows()
    ground = [r for r in rects if r[2] < layout.floor_lines()[1]]
    upper = [r for r in rects if r[2] >= layout.floor_lines()[1]]
    assert (ground[0][1] - ground[0][0]) > (upper[0][1] - upper[0][0])


def test_a_floorless_building_is_refused():
    with pytest.raises(ValueError):
        FacadeLayout(floors=0)


# ---------------------------------------------------------------------------
# Sizing
# ---------------------------------------------------------------------------


def test_pixel_size_is_latent_friendly():
    """Diffusion VAEs need multiples of 8, whatever the floor count."""
    for floors in range(1, 25):
        width, height = FacadeLayout(floors=floors, bays=4).pixel_size(128, 128)
        assert width % 8 == 0 and height % 8 == 0
        assert width >= 8 and height >= 8


def test_texel_density_is_constant_across_floor_counts():
    """A three-storey wall and a twenty-storey one next to it must match.

    This is the reason the sheet size scales with the floor count instead of
    being fixed: a fixed sheet would give the tall building four times the
    metres per texel of its neighbour.
    """
    densities = []
    for floors in (3, 6, 12, 20):
        layout = FacadeLayout(floors=floors, bays=4)
        _, height = layout.pixel_size(128, 128)
        densities.append(layout.texel_metres(floors * 3.5, height))
    assert max(densities) / min(densities) < 1.15


# ---------------------------------------------------------------------------
# The images
# ---------------------------------------------------------------------------


def test_control_image_puts_the_shopfront_at_the_bottom():
    """V=0 is the pavement, and row 0 of an image is the top. Easy to get backwards.

    Two lines run the full width of the sheet — the fascia over the shopfront
    and the parapet at the roof — so where they land pins the V flip exactly.
    """
    layout = FacadeLayout(floors=6, bays=4)
    height = 512
    image = control_image(layout, 256, height)

    full_width = np.flatnonzero((image[:, :, 0] > 0).all(axis=1))
    fascia_v = 1.0 - full_width.max() / height
    parapet_v = 1.0 - full_width.min() / height

    assert fascia_v == pytest.approx(layout.floor_lines()[1], abs=0.02)
    assert parapet_v == pytest.approx(1.0 - layout.parapet, abs=0.02)


def test_control_image_is_a_line_drawing_not_a_render():
    """Mostly black: a conditioner is told where the edges are, not what fills them."""
    image = control_image(FacadeLayout(floors=8, bays=4), 256, 768)
    assert (image > 0).mean() < 0.35


def test_procedural_facade_wraps_horizontally():
    """A wall goes round the building, so the sheet must meet itself."""
    layout = FacadeLayout(floors=6, bays=4)
    sheet = procedural_facade(layout, 256, 512, seed=3)
    assert seam_error_axis(sheet, axis=1) < 2.0


def test_procedural_facade_does_not_wrap_vertically():
    """It must not: joining a roofline to a shopfront is the artefact to avoid."""
    layout = FacadeLayout(floors=6, bays=4)
    sheet = procedural_facade(layout, 256, 512, seed=3)
    assert seam_error_axis(sheet, axis=0) > 2.0


def test_procedural_facade_is_deterministic():
    layout = FacadeLayout(floors=5, bays=3)
    a = procedural_facade(layout, 128, 384, seed=7)
    b = procedural_facade(layout, 128, 384, seed=7)
    assert np.array_equal(a, b)


def test_two_seeds_are_alike_but_not_the_same():
    layout = FacadeLayout(floors=5, bays=3)
    a = procedural_facade(layout, 128, 384, seed=1).astype(float)
    b = procedural_facade(layout, 128, 384, seed=2).astype(float)
    assert not np.array_equal(a, b)
    assert np.abs(a - b).mean() < 40  # same building, different glazing


# ---------------------------------------------------------------------------
# The measurement — the point of Phase A
# ---------------------------------------------------------------------------


def test_a_drawn_facade_matches_its_own_layout():
    for floors in (2, 6, 12, 20):
        layout = FacadeLayout(floors=floors, bays=4)
        width, height = layout.pixel_size(64, 64)
        sheet = procedural_facade(layout, width, height, seed=5)
        assert floor_alignment(sheet, layout) > 0.6
        assert bay_alignment(sheet, layout) > 0.9


def test_noise_matches_nothing():
    """The metric has to be able to fail, or it is not a metric.

    This is the case the LCM sheets fell into: a wall-coloured image with no
    storeys in it, which a prompt alone cannot be blamed for and a glance at a
    contact sheet does not immediately reveal.
    """
    layout = FacadeLayout(floors=8, bays=4)
    rng = np.random.default_rng(0)
    noise = (rng.random((768, 256, 3)) * 255).astype(np.uint8)
    assert abs(floor_alignment(noise, layout)) < 0.2


def test_a_flat_wall_matches_nothing_either():
    layout = FacadeLayout(floors=8, bays=4)
    flat = np.full((768, 256, 3), 160, dtype=np.uint8)
    assert floor_alignment(flat, layout) == 0.0


def test_the_wrong_storey_count_is_caught():
    """A sheet drawn for 12 floors must not pass as one drawn for 6 or 24.

    This is the failure the metric exists for: a sheet with a perfectly good
    facade on it, stretched over a building with a different number of storeys,
    where every window is the wrong size and nothing looks obviously broken.
    """
    twelve = FacadeLayout(floors=12, bays=4)
    width, height = twelve.pixel_size(64, 64)
    sheet = procedural_facade(twelve, width, height, seed=5)

    assert floor_alignment(sheet, twelve) > 0.6
    for wrong in (6, 11, 13, 24):
        assert floor_alignment(sheet, FacadeLayout(floors=wrong, bays=4)) < 0.3


def test_the_wrong_bay_count_is_caught():
    layout = FacadeLayout(floors=6, bays=4)
    width, height = layout.pixel_size(64, 64)
    sheet = procedural_facade(layout, width, height, seed=2)

    assert bay_alignment(sheet, layout) > 0.9
    for wrong in (2, 3, 6):
        assert bay_alignment(sheet, FacadeLayout(floors=6, bays=wrong)) < 0.3


def test_windows_half_a_floor_out_are_caught():
    """Phase matters, not just pitch: a facade offset by half a storey is wrong."""
    layout = FacadeLayout(floors=8, bays=4)
    width, height = layout.pixel_size(64, 64)
    sheet = procedural_facade(layout, width, height, seed=5)
    shifted = np.roll(sheet, height // 16, axis=0)
    assert floor_alignment(shifted, layout) < 0.3


def test_alignment_survives_a_resize():
    """A generated sheet will not come back at the size we drew the layout at."""
    layout = FacadeLayout(floors=8, bays=4)
    width, height = layout.pixel_size(64, 64)
    small = procedural_facade(layout, width // 2, height // 2, seed=5)
    assert floor_alignment(small, layout) > 0.6


# ---------------------------------------------------------------------------
# Sheets on disk
# ---------------------------------------------------------------------------


def test_the_floor_count_survives_the_filename():
    name = sheet_name(6, 3)
    assert name == "facade_f06_003.png"
    assert sheet_floors(name) == 6
    assert sheet_floors("/tmp/sheets/" + name) == 6


def test_a_sheet_that_does_not_say_returns_none():
    assert sheet_floors("hand_made_tile.png") is None


def test_the_bay_count_is_read_off_the_drawing():
    """So the seam measurement cannot drift from the layout that made the sheet."""
    for bays in (2, 3, 4, 6):
        layout = FacadeLayout(floors=6, bays=bays)
        width, height = layout.pixel_size(64, 64)
        assert bays_in(control_image(layout, width, height)) == bays


def test_a_wrapping_sheet_looks_like_its_other_bay_divisions():
    """The wrap lands on a bay boundary, which is a pier, not a fault.

    Comparing it against the sheet's mean step compares a pier with blank
    wall — measured, that scored sheets which tile perfectly anywhere from 0.3
    to 11, and sent the diagnosis chasing the padding for an afternoon.
    """
    layout = FacadeLayout(floors=6, bays=4)
    width, height = layout.pixel_size(64, 64)
    sheet = procedural_facade(layout, width, height, seed=3)
    control = control_image(layout, width, height)
    assert wrap_seam(sheet, control) < 1.5


def test_a_real_seam_still_shows_up():
    layout = FacadeLayout(floors=6, bays=4)
    width, height = layout.pixel_size(64, 64)
    sheet = procedural_facade(layout, width, height, seed=3).astype(np.int16)
    sheet[:, : width // 2] = np.clip(sheet[:, : width // 2] + 60, 0, 255)
    control = control_image(layout, width, height)
    assert wrap_seam(sheet.astype(np.uint8), control) > 3.0


# ---------------------------------------------------------------------------
# Variety — the half the alignment score cannot see
# ---------------------------------------------------------------------------


def test_sampled_layouts_actually_differ():
    """One canonical drawing per floor count gives a city of one building."""
    rng = random.Random(0)
    layouts = [sample_layout(6, rng) for _ in range(24)]
    assert len({layout.bays for layout in layouts}) > 1
    assert np.std([layout.window_width for layout in layouts]) > 0.05
    assert np.std([layout.ground_floor_ratio for layout in layouts]) > 0.05


def test_a_sampled_layout_is_still_a_valid_layout():
    rng = random.Random(1)
    for floors in (1, 3, 12):
        layout = sample_layout(floors, rng)
        assert layout.floor_lines()[-1] == 1.0
        assert layout.windows()
        width, height = layout.pixel_size(64, 64)
        assert width % 8 == 0 and height % 8 == 0
        sheet = procedural_facade(layout, width, height, seed=2)
        assert floor_alignment(sheet, layout) > 0.4


def test_sampling_is_deterministic():
    assert sample_layout(6, random.Random(3)) == sample_layout(6, random.Random(3))


def test_wide_windows_read_as_a_ribbon():
    """The same drawing code has to cover punched openings and ribbon glazing."""
    punched = FacadeLayout(floors=6, bays=4, window_width=0.34)
    ribbon = FacadeLayout(floors=6, bays=4, window_width=0.88)
    glazed = lambda l: sum((u1 - u0) for u0, u1, *_ in l.windows())
    assert glazed(ribbon) > 2 * glazed(punched)


def test_diversity_separates_a_set_from_its_clones():
    layout = FacadeLayout(floors=6, bays=4)
    width, height = layout.pixel_size(64, 64)
    clones = [procedural_facade(layout, width, height, seed=1) for _ in range(4)]
    assorted = [
        procedural_facade(layout, width, height, seed=1, wall=w, glass=g)
        for w, g in (((0.62, 0.60, 0.57), (0.16, 0.20, 0.26)),
                     ((0.55, 0.28, 0.22), (0.30, 0.30, 0.28)),
                     ((0.88, 0.88, 0.86), (0.10, 0.14, 0.20)),
                     ((0.30, 0.42, 0.46), (0.42, 0.58, 0.60)))
    ]
    assert diversity(clones) == pytest.approx(0.0, abs=1e-9)
    assert diversity(assorted) > 0.1


def test_saturation_tells_grey_from_coloured():
    layout = FacadeLayout(floors=6, bays=4)
    width, height = layout.pixel_size(64, 64)
    grey = procedural_facade(layout, width, height, seed=1, wall=(0.6, 0.6, 0.6),
                             glass=(0.3, 0.3, 0.3))
    brick = procedural_facade(layout, width, height, seed=1, wall=(0.55, 0.24, 0.18),
                              glass=(0.2, 0.3, 0.4))
    assert saturation(grey) < 0.1 < saturation(brick)


# --- what the ground floor is ------------------------------------------------


def test_a_house_does_not_have_a_shop_under_it():
    """Where "the windows are enormous" came from.

    The numbers were written for a mid-rise and applied to houses. On a
    two-storey building a ground-floor ratio of 1.8 gives the shop nearly two
    thirds of the height, and a shopfront 0.92 of a bay wide glazes almost all
    of that.
    """
    for seed in range(20):
        house = sample_layout(2, random.Random(seed), facade_width=7.0, kind="house")
        shop = sample_layout(2, random.Random(seed), facade_width=7.0, kind="commercial")
        assert house.ground_floor_ratio < 1.2, "a house's ground floor is a floor"
        assert house.shopfront_width < shop.shopfront_width
        assert house.shopfront_height < shop.shopfront_height


def test_a_house_window_is_a_window_and_not_a_wall():
    for seed in range(20):
        house = sample_layout(2, random.Random(seed), facade_width=7.0, kind="house")
        bay = 7.0 / house.bays
        # 0.5 m is a stair or a bathroom, 2 m is a living room. Both are windows;
        # the commercial range runs past 3 m, which on a house is a shopfront.
        assert 0.5 <= house.window_width * bay <= 2.0


def test_a_house_bay_is_a_window_and_a_pier():
    """A structural span for a commercial frame, a window's worth for a house."""
    wide = [sample_layout(2, random.Random(s), facade_width=12.0, kind="commercial").bays
            for s in range(20)]
    narrow = [sample_layout(2, random.Random(s), facade_width=12.0, kind="house").bays
              for s in range(20)]
    assert sum(narrow) > sum(wide)


def test_a_kind_nobody_builds_is_refused():
    with pytest.raises(ValueError, match="facade kind"):
        sample_layout(2, random.Random(0), kind="cathedral")
