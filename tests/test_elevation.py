"""Reading a published elevation model, and lining it up with a map.

No network: the tile fetch is stubbed, because what is worth testing is the
decoding and the datum arithmetic, and because a test that reaches the GSI
servers fails on a train.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from city_builder import elevation as el
from city_builder.frame import LocalFrame


def _encode(metres: np.ndarray) -> Image.Image:
    """The inverse of GSI's encoding, for building a tile to read back."""
    packed = np.round(np.asarray(metres) * 100).astype(np.int64) % 16777216
    rgb = np.stack([packed // 65536, (packed // 256) % 256, packed % 256], axis=-1)
    return Image.fromarray(rgb.astype(np.uint8))


def _tile(fill=12.34, size=16):
    return _encode(np.full((size, size), fill))


def _served(monkeypatch, tiles: dict[tuple[str, int], Image.Image | None]):
    """Serve the given tiles by template, and 404 everything else."""
    asked = []

    class _Response:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

    def urlopen(url, timeout=0):
        asked.append(url)
        template = url.split("/xyz/")[1].split("/")[0]
        image = tiles.get(template)
        if image is None:
            raise OSError("404")
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        return _Response(buffer.getvalue())

    monkeypatch.setattr(el.urllib.request, "urlopen", urlopen)
    return asked


# ---------------------------------------------------------------------------
# The encoding
# ---------------------------------------------------------------------------


def test_the_elevation_comes_out_of_the_colours():
    heights = np.array([[0.0, 1.0], [123.45, 8000.0]])
    assert _decoded(heights) == pytest.approx(heights, abs=1e-6)


def test_a_height_below_the_datum_decodes_negative():
    """The top bit is a sign; read unsigned, a beach becomes a mountain."""
    assert _decoded(np.array([[-3.5]]))[0, 0] == pytest.approx(-3.5)


def test_the_no_data_marker_is_not_read_as_a_height():
    """(128, 0, 0) decodes to -83886.08, which would flatten a whole map."""
    marker = Image.fromarray(np.array([[[128, 0, 0]]], dtype=np.uint8))
    assert np.isnan(el._decode(marker)[0, 0])


def _decoded(heights):
    return el._decode(_encode(heights))


# ---------------------------------------------------------------------------
# Choosing a source
# ---------------------------------------------------------------------------


def test_a_coarser_source_is_used_when_the_fine_one_is_absent(monkeypatch):
    _served(monkeypatch, {"dem_png": _tile(31.0)})
    frame = LocalFrame(35.9, 139.9)
    got = el.sample_grid(frame, 0.0, 0.0, 4, 4, 10.0)
    assert got is not None
    model, source, zoom, _tiles = got
    assert source == "dem_png" and zoom == 14
    assert model == pytest.approx(np.full((4, 4), 31.0))


def test_no_source_at_all_is_an_answer_and_not_a_failure(monkeypatch):
    """Outside Japan this is the ordinary case, and the ground is still built."""
    _served(monkeypatch, {})
    assert el.sample_grid(LocalFrame(48.85, 2.35), 0.0, 0.0, 4, 4, 10.0) is None


def test_a_source_that_only_reaches_a_corner_is_refused(monkeypatch):
    """A prior with a coverage edge running through the scene steps the ground."""
    pixels = np.asarray(_encode(np.full((16, 16), 20.0))).copy()
    pixels[1:, :] = [128, 0, 0]  # the fine product barely reaches this map
    _served(monkeypatch, {"dem5a_png": Image.fromarray(pixels),
                          "dem_png": _tile(19.0)})
    got = el.sample_grid(LocalFrame(35.9, 139.9), 0.0, 0.0, 4, 4, 10.0)
    assert got is not None and got[1] == "dem_png"


def test_tiles_are_fetched_once(monkeypatch):
    asked = _served(monkeypatch, {"dem5a_png": _tile(20.0)})
    el.sample_grid(LocalFrame(35.9, 139.9), 0.0, 0.0, 8, 8, 10.0)
    assert len(asked) == len(set(asked))


def test_a_cached_tile_is_not_fetched_again(monkeypatch, tmp_path):
    asked = _served(monkeypatch, {"dem5a_png": _tile(20.0)})
    for _ in range(2):
        el.sample_grid(LocalFrame(35.9, 139.9), 0.0, 0.0, 4, 4, 10.0,
                       cache_dir=str(tmp_path))
    assert len(asked) == len(set(asked)), "the cache was not read on the second run"


# ---------------------------------------------------------------------------
# The datum
# ---------------------------------------------------------------------------


def test_the_datum_offset_is_solved_rather_than_assumed():
    """Measured at 16.3 m on Kashiwanoha, and different on every map."""
    model = np.full((5, 5), 47.0)
    samples = np.array([[0.0, 0.0, 2.0], [10.0, 10.0, 2.0], [20.0, 20.0, 2.0]])
    offset, median, p90 = el.align(model, samples, 0.0, 0.0, 10.0)
    assert offset == pytest.approx(45.0)
    assert median == pytest.approx(0.0) and p90 == pytest.approx(0.0)


def test_a_few_wild_samples_do_not_drag_the_offset():
    """A median, not a mean: an embankment is an outlier, not a datum shift."""
    model = np.full((5, 5), 30.0)
    samples = np.array([[0.0, 0.0, 10.0], [10.0, 0.0, 10.0], [20.0, 0.0, 10.0],
                        [30.0, 0.0, 10.0], [40.0, 0.0, -90.0]])
    offset, _median, _p90 = el.align(model, samples, 0.0, 0.0, 10.0)
    assert offset == pytest.approx(20.0)


def test_the_residual_says_how_far_the_two_still_disagree():
    model = np.array([[10.0, 10.0, 10.0, 12.0]])
    samples = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0],
                        [30.0, 0.0, 0.0]])
    _offset, _median, p90 = el.align(model, samples, 0.0, 0.0, 10.0)
    assert p90 > 1.0


def test_a_model_that_reaches_none_of_the_roads_is_refused():
    with pytest.raises(ValueError, match="does not reach"):
        el.align(np.full((3, 3), np.nan), np.array([[0.0, 0.0, 1.0]]), 0.0, 0.0, 10.0)


def test_the_prior_comes_back_in_the_scenes_own_datum(monkeypatch):
    _served(monkeypatch, {"dem5a_png": _tile(28.5)})
    frame = LocalFrame(35.9, 139.9)
    samples = [(0.0, 0.0, 1.5), (10.0, 0.0, 1.5), (20.0, 0.0, 1.5)]
    got = el.terrain_for(frame, 0.0, 0.0, 4, 4, 10.0, samples)
    assert got is not None
    prior, coverage = got
    # 28.5 in the model's datum against 1.5 in the scene's: the prior must come
    # back at the scene's height, not 27 m over the buildings.
    assert prior == pytest.approx(np.full((4, 4), 1.5))
    assert coverage.datum_offset == pytest.approx(27.0)
    assert coverage.covered == pytest.approx(1.0)
    assert coverage.to_json()["source"] == "dem5a_png"


# ---------------------------------------------------------------------------
# Inventing one instead
#
# The procedural terrain is a *tile source*, not a separate path: it arrives
# through the same door as a downloaded one, so the datum solve, the coverage
# report and the guidance term are all one code path with two providers.
# ---------------------------------------------------------------------------


def test_invented_tiles_answer_the_same_questions_a_downloaded_one_does():
    source = el.InventedTiles(el.Relief(seed=3))
    assert isinstance(source.name, str) and isinstance(source.zoom, int)
    grid = source.grid(29121, 12878)
    assert grid.shape == (256, 256) and np.isfinite(grid).all()


def test_a_map_with_no_survey_over_it_still_gets_terrain(monkeypatch):
    """The point of the thing: a fully procedural city, anywhere on earth."""
    _served(monkeypatch, {})  # nothing published here
    frame = LocalFrame(48.85, 2.35)
    sources = [*el.web_sources(), el.InventedTiles(el.Relief(amplitude=3.0, seed=1))]
    got = el.sample_grid(frame, 0.0, 0.0, 12, 12, 10.0, sources=sources)
    assert got is not None and got[1] == "invented"
    assert np.isfinite(got[0]).all()


def test_neighbouring_tiles_join_along_their_edge():
    """The artefact this is built not to have: a step at every tile boundary.

    Both the noise and the amplitude normalisation are functions of absolute
    position, so two tiles agree at their seam without being asked to.
    """
    source = el.InventedTiles(el.Relief(amplitude=3.0, metres=140.0, seed=4))
    left, right = source.grid(29121, 12878), source.grid(29122, 12878)
    seam = float(np.abs(left[:, -1] - right[:, 0]).mean())
    ordinary = float(np.abs(np.diff(left, axis=1)).mean())
    assert seam < 2.0 * ordinary, "the terrain steps at the tile boundary"

    below = source.grid(29121, 12879)
    seam_y = float(np.abs(left[-1, :] - below[0, :]).mean())
    assert seam_y < 2.0 * float(np.abs(np.diff(left, axis=0)).mean())


def test_the_same_conditions_give_the_same_terrain():
    def made(seed):
        return el.InventedTiles(el.Relief(seed=seed)).grid(29121, 12878)

    assert np.array_equal(made(5), made(5))
    assert not np.array_equal(made(5), made(6))


def test_the_amplitude_is_what_was_asked_for():
    def spread(amplitude):
        source = el.InventedTiles(el.Relief(amplitude=amplitude, metres=140.0, seed=2))
        wide = np.concatenate([source.grid(29121 + i, 12878) for i in range(4)], axis=1)
        return float(np.ptp(wide))

    assert spread(6.0) == pytest.approx(2.0 * spread(3.0), rel=0.02)
    assert 0.3 < spread(3.0) / 3.0 <= 1.0, "a whole tile should see much of the range"


def test_a_bigger_feature_size_gives_smoother_ground():
    def roughness(metres):
        grid = el.InventedTiles(el.Relief(amplitude=3.0, metres=metres, seed=8)).grid(
            29121, 12878)
        return float(np.abs(np.diff(grid, axis=1)).mean())

    assert roughness(400.0) < roughness(80.0)


def test_the_solved_datum_absorbs_whatever_the_invented_terrain_calls_zero():
    """Its heights mean nothing, so they must not arrive as metres above sea."""
    frame = LocalFrame(35.9, 139.9)
    samples = [(0.0, 0.0, 4.0), (20.0, 0.0, 4.0), (40.0, 0.0, 4.0)]
    got = el.terrain_for(frame, 0.0, 0.0, 8, 8, 10.0, samples,
                       sources=[el.InventedTiles(el.Relief(amplitude=2.0, seed=9))])
    assert got is not None
    prior, _coverage = got
    assert abs(float(np.median(prior)) - 4.0) < 2.0
