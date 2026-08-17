"""Generating inside the plot instead of fitting afterwards.

The generative model is not exercised here — it is sixteen gigabytes and a
card. What is testable is the part that decides what it is asked: the prism
that replaces its first sampling stage, and the one property of the
conditioning picture that predicts whether the result will be a building.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from shapely.affinity import rotate
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import reconstruct as R


def _footprint(long_side=20.0, short_side=12.0, angle=0.0):
    box = ShapelyPolygon([(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
                          (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)])
    return [list(p) for p in list(rotate(box, angle, origin=(0, 0)).exterior.coords)[:-1]]


def _extent(coords, axis):
    return coords[:, axis].max() - coords[:, axis].min() + 1


# ---------------------------------------------------------------------------
# The envelope
# ---------------------------------------------------------------------------


def test_the_prism_has_the_plans_proportions():
    """What the whole thing is for: the plan is the plot's, not the model's."""
    coords = R.envelope_coords(_footprint(20.0, 10.0), 6.0, grid=32, eave_room=0.0)
    assert _extent(coords, 0) / _extent(coords, 1) == pytest.approx(2.0, rel=0.12)


def test_the_height_is_on_the_height_axis():
    """The axes are the identity, and every other mapping stands it on end."""
    at = {"grid": 32, "eave_room": 0.0, "roof_room": 0.0}
    tall = R.envelope_coords(_footprint(20.0, 20.0), 20.0, **at)
    short = R.envelope_coords(_footprint(20.0, 20.0), 5.0, **at)
    assert _extent(tall, 2) > _extent(short, 2) * 2
    # The plan is the same either way: the cube is sized by the largest
    # dimension, and at 20 m square that is the plan in both.
    assert _extent(tall, 0) == _extent(short, 0)


def test_a_turned_plot_turns_the_prism_rather_than_its_bounding_box():
    """A rectangle at 45 degrees is a diamond, not a bigger rectangle."""
    square = R.envelope_coords(_footprint(20.0, 20.0), 6.0, grid=32, eave_room=0.0)
    diamond = R.envelope_coords(_footprint(20.0, 20.0, angle=45.0), 6.0,
                                grid=32, eave_room=0.0)
    columns = lambda c: len({(i, j) for i, j, _k in c})
    assert columns(diamond) < columns(square) * 0.65


def test_room_for_the_eaves_grows_the_plan_and_not_the_height():
    """Measured: 0.6 m of room took the footprint IoU from 0.822 to 0.882."""
    at = {"grid": 32, "roof_room": 0.0}
    tight = R.envelope_coords(_footprint(), 6.0, eave_room=0.0, **at)
    roomy = R.envelope_coords(_footprint(), 6.0, eave_room=1.5, **at)
    assert len({(i, j) for i, j, _k in roomy}) > len({(i, j) for i, j, _k in tight})
    assert _extent(roomy, 2) == _extent(tight, 2)


def test_room_for_the_roof_grows_the_height_and_not_the_plan():
    """The other half of the same idea, and not a refinement: without it the
    shape model does not reach the top of its envelope and the buildings come
    out short — 0.81 of the block height over 185 of them, 174 more than a
    tenth short, against 1.25 for the path that invents its own massing."""
    at = {"grid": 32, "eave_room": 0.0}
    flat = R.envelope_coords(_footprint(20.0, 12.0), 6.0, roof_room=0.0, **at)
    pitched = R.envelope_coords(_footprint(20.0, 12.0), 6.0, roof_room=0.4, **at)
    assert _extent(pitched, 2) > _extent(flat, 2)
    assert len({(i, j) for i, j, _k in pitched}) == len({(i, j) for i, j, _k in flat})


def test_the_headroom_is_a_fraction_because_a_metre_means_different_things():
    """A metre of ridge on a shed is a different building and on an office
    block is nothing. Against a solid envelope, which the shape model fills to
    0.81, 0.4 lands at 1.23, 1.26 and 1.23 of the block on one, two and three
    storeys — the ratio the path inventing its own massing arrives at."""
    assert R.ROOF_ROOM == 0.4
    tall = R.envelope_coords(_footprint(60.0, 60.0), 30.0, grid=32,
                             eave_room=0.0, roof_room=0.4)
    # 30 m of block plus 12 m of roof, against 60 m of plan: 42/60 of the cube.
    assert _extent(tall, 2) / _extent(tall, 0) == pytest.approx(0.7, abs=0.05)


# ---------------------------------------------------------------------------
# Solid, unless it will not fit
#
# `sample_sparse_structure` returns a *surface*, so a solid prism is not the
# kind of object the shape model is used to — which is a real observation and
# was not a reason to hollow the envelope out. The whole map generated from
# surface envelopes came back a district of cages, walls you could see daylight
# through, while solid prisms had produced buildings. The footprint IoU is
# blind to it and was slightly better on the cages, which is how that run
# reached 189 buildings before anybody looked at it.
# ---------------------------------------------------------------------------


def test_the_envelope_is_solid_when_it_fits():
    # A house-sized plot: 159 of this map's 189 are inside the budget whole.
    coords = R.envelope_coords(_footprint(20.0, 12.0), 6.0, grid=32,
                               eave_room=0.0, roof_room=0.0)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    assert len(coords) == int(np.prod(hi - lo + 1)), "the envelope was hollowed out"


def test_a_prism_over_budget_is_peeled_until_it_fits():
    """Above the budget the run does not degrade, it throws: nineteen buildings
    on this map, three attempts each, twice, every one out of memory. Something
    has to give, and better the envelope than the building."""
    coords = R.envelope_coords(_footprint(20.0, 20.0), 20.0, grid=32,
                               eave_room=0.0, roof_room=0.0, budget=8000)
    assert 32768 > len(coords) <= 8000
    cells = {tuple(c) for c in coords}
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    assert tuple((lo + hi) // 2) not in cells, "the middle went last, not first"
    # The outside is what says where the building is, so it is what survives.
    for axis in range(3):
        for face in (lo[axis], hi[axis]):
            assert any(c[axis] == face for c in cells)


def test_the_budget_is_the_one_nothing_observed_has_failed_at():
    """The ceiling is not a property of the count alone: a square plot at
    20 000 asked for a 12.7 GB allocation and died where a long thin plot of
    the same count had not. Peeling costs a little detail; running out costs
    the whole building."""
    assert R.VOXEL_BUDGET <= 12_000


def test_no_budget_leaves_the_prism_alone():
    at = {"grid": 32, "eave_room": 0.0, "roof_room": 0.0}
    whole = R.envelope_coords(_footprint(20.0, 20.0), 20.0, budget=0, **at)
    assert len(whole) == 32 * 32 * 32


def test_the_prism_is_centred_in_the_cube_the_mesh_comes_back_in():
    """`to_glb` reads a cube centred on the origin, so an off-centre prism
    would place every building at an offset the fit then has to undo."""
    coords = R.envelope_coords(_footprint(20.0, 12.0), 8.0, grid=32, eave_room=0.0)
    for axis in (0, 1):
        middle = (coords[:, axis].max() + coords[:, axis].min()) / 2
        assert middle == pytest.approx(15.5, abs=0.6)


def test_a_plan_thinner_than_a_voxel_says_so():
    """The cube is sized by the *largest* dimension, so a tall enough building
    on a small enough plot has a plan below one cell and no columns at all.
    Silently returning no cells reaches the sampler as "generate nothing"."""
    with pytest.raises(ValueError, match="too small"):
        R.envelope_coords(_footprint(0.2, 0.2), 60.0, grid=32, eave_room=0.0)


def test_the_grid_is_the_one_the_flow_model_was_trained_on():
    """Handing the 512 model a 64 cube is a mismatch, not a finer envelope:
    it measured 0.743 against 0.822 for the same plot."""
    assert R._ENVELOPE_GRID["512"] == 32
    assert R._ENVELOPE_GRID["1024"] == 64


def test_a_cascade_has_no_single_set_of_coords_to_replace():
    with pytest.raises(ValueError, match="single-resolution"):
        R.to_mesh_in_envelope("unused.png", "out.glb", footprint=_footprint(), height=6.0,
                              options=R.MeshOptions(pipeline_type="1024_cascade"))


# ---------------------------------------------------------------------------
# The picture, which now only has to carry the material
# ---------------------------------------------------------------------------


def _framed(subject_fraction: float, size=128) -> Image.Image:
    """A building of that share of the frame, on a plain backdrop.

    The subject is textured, because a photograph is: a flat block would be
    keyed out as backdrop itself the moment it reached the border.
    """
    rng = np.random.default_rng(4)
    frame = np.full((size, size, 3), 150, dtype=np.uint8)
    side = round(size * subject_fraction ** 0.5)
    start = (size - side) // 2
    frame[start:start + side, start:start + side] = rng.integers(
        20, 110, (side, side, 3), dtype=np.uint8)
    return Image.fromarray(frame)


def test_a_subject_on_a_plain_field_reads_as_isolated():
    assert R.backdrop_share(_framed(0.25)) > 0.6


def _street(size=128) -> Image.Image:
    """A house in its setting: graded sky, building, garden. No backdrop at all.

    The failure this exists to catch. Asked for a photograph of a house, the
    image model returns the street it stands in, and TRELLIS takes the whole
    frame as the subject — sky, garden and the neighbours end up in the walls.
    """
    rng = np.random.default_rng(7)
    frame = np.empty((size, size, 3), dtype=np.uint8)
    sky = np.linspace(120, 235, size // 3).astype(np.uint8)
    frame[:size // 3] = sky[:, None, None]
    frame[size // 3:] = rng.integers(40, 120, (size - size // 3, size, 3), dtype=np.uint8)
    wall = frame[size // 3:size * 3 // 4, size // 5:size * 4 // 5]
    wall[:] = rng.integers(150, 210, wall.shape, dtype=np.uint8)
    return Image.fromarray(frame)


def test_a_street_scene_does_not_read_as_isolated():
    assert R.backdrop_share(_street()) < 0.25 <= R.backdrop_share(_framed(0.25))


def test_the_frame_is_asked_for_and_not_only_the_building():
    prompt = R.isolated_prompt("a house")
    assert "a house" in prompt
    for wanted in ("isolated", "no ground", "no sky", "whole building"):
        assert wanted in prompt
    for unwanted in ("street scene", "sky", "adjacent buildings", "cropped"):
        assert unwanted in R.ISOLATED_NEGATIVE


# ---------------------------------------------------------------------------
# The card, before anything has claimed it
# ---------------------------------------------------------------------------


def test_the_allocator_is_configured_before_torch_can_read_it():
    """`PYTORCH_CUDA_ALLOC_CONF` is read once, when the CUDA caching allocator
    is first built, and ignored ever after. Setting it on the way into TRELLIS
    is too late in a pipeline run — SDXL has already drawn the tiles and the
    photographs — and what that costs is buildings: the mesher wants one large
    contiguous block, and five plots in sixty died for want of one with 25 GB
    of the card free."""
    import os
    import subprocess
    import sys

    got = subprocess.run(
        [sys.executable, "-c",
         "import city_builder, os; print(os.environ.get('PYTORCH_CUDA_ALLOC_CONF'))"],
        capture_output=True, text=True, check=True,
        env={k: v for k, v in os.environ.items() if k != "PYTORCH_CUDA_ALLOC_CONF"})
    assert "expandable_segments:True" in got.stdout


def test_an_allocator_setting_the_caller_chose_is_left_alone():
    import os
    import subprocess
    import sys

    got = subprocess.run(
        [sys.executable, "-c",
         "import city_builder, os; print(os.environ['PYTORCH_CUDA_ALLOC_CONF'])"],
        capture_output=True, text=True, check=True,
        env={**os.environ, "PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:128"})
    assert got.stdout.strip() == "max_split_size_mb:128"


# ---------------------------------------------------------------------------
# The one thing about the shape the picture still has to agree with
# ---------------------------------------------------------------------------


def test_the_storey_count_is_said_to_the_image_model():
    """The envelope sets the height and the picture sets everything else, so a
    bungalow photographed for a three-storey plot comes back as a bungalow nine
    metres tall — one row of windows stretched over three."""
    assert "three-storey" in R.isolated_prompt("a Japanese house", 3)
    assert "single-storey" in R.isolated_prompt("a Japanese house", 1)
    assert "storey" not in R.isolated_prompt("a Japanese house")


def test_a_storey_count_nobody_wrote_down_is_still_sayable():
    assert "a 9-storey " == R.storeys_said(9)
    assert R.storeys_said(0) == "" and R.storeys_said(None) == ""


def test_the_subject_does_not_keep_its_own_article():
    """"a two-storey a small Japanese house" is not a prompt."""
    said = R.isolated_prompt("a small Japanese suburban house", 2)
    assert "a two-storey small Japanese suburban house" in said
    assert "storey a " not in said


def test_every_subject_in_the_catalogue_reads_as_english_with_a_count():
    for _name, subject in R.BUILDING_SUBJECTS:
        said = R.isolated_prompt(subject, 2)
        assert "a two-storey " in said
        assert " a a " not in said and "storey a " not in said


def test_a_photograph_of_a_wall_is_caught_even_with_room_around_it():
    """`backdrop_share` catches a picture with no background in it. This
    catches one where the background is ample but the building is still
    cropped — a photograph of a wall rather than of a house, which the image
    model returns often enough to matter and which reconstructs as a slab."""
    whole = _framed(0.25)
    assert R.touches_the_frame(whole) < 0.05
    assert R.touches_the_frame(_street()) > 0.5


def test_a_subject_that_runs_off_one_edge_only_still_counts():
    rng = np.random.default_rng(2)
    frame = np.full((128, 128, 3), 150, dtype=np.uint8)
    frame[20:128, 30:90] = rng.integers(20, 110, (108, 60, 3), dtype=np.uint8)
    assert R.touches_the_frame(Image.fromarray(frame)) > 0.05


# ---------------------------------------------------------------------------
# The prompt asks; the measurement decides
# ---------------------------------------------------------------------------


def _wide(width, height, size=160):
    rng = np.random.default_rng(3)
    frame = np.full((size, size, 3), 150, dtype=np.uint8)
    box = frame[(size-height)//2:(size+height)//2, (size-width)//2:(size+width)//2]
    box[:] = rng.integers(20, 110, box.shape, dtype=np.uint8)
    return Image.fromarray(frame)


def test_the_silhouette_aspect_is_what_was_drawn():
    assert R.silhouette_aspect(_wide(100, 50)) == pytest.approx(2.0, rel=0.1)
    assert R.silhouette_aspect(_wide(50, 100)) == pytest.approx(0.5, rel=0.1)


def test_the_proportion_is_relabelled_from_the_picture_not_the_prompt():
    """Told "long and narrow in plan", the image model returns the same roughly
    square house — on the same 3:1 plot, 0.876 against 0.879 for one drawn
    without the phrase. What it draws can still be measured."""
    drawn = [{"floors": 2, "asked": "elongated", "aspect": 0.9},
             {"floors": 2, "asked": "elongated", "aspect": 1.0},
             {"floors": 2, "asked": "compact", "aspect": 1.6},
             {"floors": 2, "asked": "compact", "aspect": 1.9}]
    got = R.classify_by_aspect(drawn)
    assert [row["proportion"] for row in got] == [
        "compact", "compact", "elongated", "elongated"]


def test_each_storey_count_is_split_on_its_own_median():
    """Self-calibrating: the plots are split the same way, so half of one meets
    half of the other however the model happened to draw that day."""
    drawn = ([{"floors": 1, "aspect": a} for a in (1.4, 1.5, 1.6, 1.7)]
             + [{"floors": 3, "aspect": a} for a in (0.8, 0.9, 1.0, 1.1)])
    got = R.classify_by_aspect(drawn)
    for floors in (1, 3):
        group = [row["proportion"] for row in got if row["floors"] == floors]
        assert group.count("compact") == 2 and group.count("elongated") == 2


def test_the_house_mix_is_mostly_ordinary_houses():
    """Given one photograph each, the six residential subjects come out even,
    and a suburb then has as many machiya, shopfronts and corrugated workshops
    as houses. What that reads as is a light industrial estate — which is what
    the twenty-building run came back as."""
    import collections

    counts = collections.Counter(R.HOUSE_MIX)
    assert counts["mortar"] + counts["siding"] > len(R.HOUSE_MIX) / 2
    assert counts["corrugated"] == 1 and counts["machiya"] == 1
    assert set(R.HOUSE_MIX) == set(R.HOUSE_SUBJECTS), "a subject went missing"


def test_a_short_run_still_gets_a_mix():
    """A caller asking for fewer takes a prefix, so grouping would give the
    first six nothing but rendered walls and siding."""
    for count in (4, 6, 8):
        assert len(set(R.HOUSE_MIX[:count])) >= 3
