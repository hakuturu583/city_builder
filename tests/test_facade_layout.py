"""The facade layout: exact where it claims to be exact, and measurably so."""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from city_builder.facade_layout import (
    FacadeLayout,
    bay_alignment,
    bays_for,
    control_image,
    floor_alignment,
    procedural_facade,
    sheet_floors,
    sheet_name,
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
