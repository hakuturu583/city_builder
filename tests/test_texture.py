"""Tileable textures and how they land on the scene.

The diffusion path needs a GPU and a model, so it is not exercised here; the
procedural path, the seam metric and the scene wiring all are.
"""

from __future__ import annotations

import numpy as np
import pytest

from city_builder import scene, texture
from city_builder.geometry import Mesh
from city_builder.texture import TextureOptions, make_tile, procedural_tile, seam_error


def _quad(size=20.0):
    return Mesh(
        [(0, 0, 0), (size, 0, 0), (size, size, 0), (0, size, 0)],
        [[0, 1, 2, 3]],
    )


# --- the tile itself ---------------------------------------------------------


def test_procedural_tile_is_deterministic():
    assert np.array_equal(procedural_tile(64, seed=5), procedural_tile(64, seed=5))
    assert not np.array_equal(procedural_tile(64, seed=5), procedural_tile(64, seed=6))


def test_procedural_tile_wraps_exactly():
    """Filtering in the frequency domain makes it periodic by construction."""
    assert seam_error(procedural_tile(128, seed=1)) == pytest.approx(1.0, abs=0.25)


def test_seam_error_catches_a_hard_seam():
    """The metric has to fail on a texture that does not wrap, or it is useless."""
    tile = np.zeros((64, 64, 3), dtype=np.uint8)
    tile[:32] = 40  # a step across the middle, and a big one across the wrap
    tile[32:] = 200
    assert seam_error(tile) > 5.0


def test_make_tile_honours_the_procedural_flag(tmp_path):
    path = tmp_path / "tile.png"
    tile = make_tile("unused", TextureOptions(size=64, diffusion=False), path=str(path))
    assert path.exists()
    assert tile.shape == (64, 64, 3)


# --- how it lands on the scene ----------------------------------------------


def test_uvs_are_metric(tmp_path):
    scene.clear_scene()
    obj = scene.add_object("Ground", _quad(20.0))
    scene.uv_from_xy(obj, tile_metres=10.0)

    uvs = {tuple(round(c, 6) for c in loop.uv) for loop in obj.data.uv_layers["UVMap"].data}
    # 20 m of ground at 10 m per tile is two repeats.
    assert uvs == {(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)}


def test_applying_a_tile_replaces_only_that_object(tmp_path):
    path = str(tmp_path / "tile.png")
    make_tile("unused", TextureOptions(size=32, diffusion=False), path=path)

    scene.clear_scene()
    materials = scene.build_materials()
    ground = scene.add_object("Ground", _quad(), materials["ground"])
    markings = scene.add_object("LaneMarkings", _quad(), materials["marking"])
    before = markings.data.materials[0].name

    scene.apply_tiled_texture(ground, path, tile_metres=8.0)

    assert ground.data.materials[0].name != "CityGround"
    assert ground.data.uv_layers, "the tile needs UVs to repeat over"
    assert markings.data.materials[0].name == before, "a preserved surface must not change"
    assert not markings.data.uv_layers, "and must not be re-parameterised either"


# ---------------------------------------------------------------------------
# Choosing the material
# ---------------------------------------------------------------------------


def test_a_named_subset_of_styles_comes_back_in_the_order_asked_for():
    from city_builder.texture import styles_named

    assert [name for name, _ in styles_named(["brick", "concrete"])] == ["brick", "concrete"]


def test_a_misspelt_style_is_an_error_not_a_shrug():
    # Silently narrowing a street to one material is only visible in the render.
    import pytest

    from city_builder.texture import styles_named

    with pytest.raises(ValueError, match="unknown facade style"):
        styles_named(["brick", "brik"])


def test_prompts_cycle_so_a_short_run_still_covers_every_style():
    from city_builder.texture import styled_prompts, styles_named

    prompts = styled_prompts(6, styles=styles_named(["brick", "concrete"]))
    assert len({p for p in prompts}) == 2
    assert sum(1 for p in prompts if "brick" in p) == 3


def test_a_reference_without_the_adapter_loaded_is_refused():
    # The adapter is a gigabyte, so it is only loaded when asked for. Passing an
    # image anyway would otherwise be ignored in silence, and the sheets would
    # come back looking like the prompt alone — with nothing to say why.
    import numpy as np
    import pytest

    from city_builder.texture import FacadeOptions, facade_sheets

    control = np.zeros((64, 64, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="options.reference is off"):
        facade_sheets("a wall", control, FacadeOptions(reference=False),
                      reference=np.zeros((8, 8, 3), dtype=np.uint8),
                      pipeline=object())


def test_the_reference_default_quotes_the_material_not_the_content():
    # Measured: 0.4 takes the palette and panel material; at 0.7 a sheet came
    # back with the reference photograph's yellow road line across the facade.
    from city_builder.texture import FacadeOptions

    assert FacadeOptions().reference_strength <= 0.5


# --- what to sample at, which is not what the sheet is wanted at --------------


def _sizes(**kwargs):
    from city_builder.texture import FacadeOptions, _sampling_size

    options = FacadeOptions(**kwargs)
    return lambda w, h: _sampling_size((w, h), options)


def test_a_sheet_smaller_than_the_model_is_sampled_larger():
    """The defect: a two-storey shop front is 384x344, and SD1.5 wants 512.

    Measured over the same prompts and control images, sheets drawn for two to
    eight floors of a 12 m bay scored 0.74 for floor alignment at 512 wide,
    while one- and two-storey houses on a 7 m bay scored 0.36 at 384x344.
    Sampling the small ones at model resolution took them to 0.75.
    """
    at = _sizes()
    for wanted in ((384, 216), (384, 344), (384, 472), (512, 344)):
        width, height = at(*wanted)
        assert min(width, height) >= 512
        assert width / height == pytest.approx(wanted[0] / wanted[1], rel=0.03)


def test_a_sheet_the_model_can_already_draw_is_left_alone():
    at = _sizes()
    assert at(512, 856) == (512, 856)
    assert at(1024, 1024) == (1024, 1024)


def test_raising_the_resolution_never_lowers_it():
    """A tall narrow sheet hits the cap on its long side; that must not shrink it."""
    at = _sizes()
    for wanted in ((384, 1240), (512, 1240), (320, 2000)):
        width, height = at(*wanted)
        assert width >= wanted[0] and height >= wanted[1]


def test_both_sides_land_on_the_stride_the_vae_uses():
    at = _sizes()
    for wanted in ((384, 216), (391, 233), (500, 501)):
        assert all(side % 8 == 0 for side in at(*wanted))


# ---------------------------------------------------------------------------
# A tile fit to lay
#
# A ground tile goes over a whole town, so one bad draw is a town that looks
# like static. These are the numbers that decide, and the reason there is a
# number at all.
# ---------------------------------------------------------------------------


def _noise(size=64, mean=120.0, spread=12.0, seed=0):
    """A stand-in tile, in the uint8 a real one comes back as."""
    rng = np.random.default_rng(seed)
    return np.clip(rng.normal(mean, spread, (size, size, 3)), 0, 255).astype(np.uint8)


def test_a_photograph_of_ground_passes():
    assert texture.tile_is_usable(texture.tile_score(_noise()))


def test_clipart_on_a_white_background_is_refused():
    """The failure that was actually shipped: white gaps read as speckle."""
    tile = _noise()
    tile[:32] = 255  # half the frame is background
    score = texture.tile_score(tile)
    assert score["blown"] > 0.4
    assert not texture.tile_is_usable(score)


def test_an_illustration_is_refused_on_contrast():
    """The grass tile that had to be thrown away measured 99; ground measures 10-65."""
    harsh = _noise(spread=90.0)
    assert texture.tile_score(harsh)["sd"] > 70.0
    assert not texture.tile_is_usable(texture.tile_score(harsh))


def test_a_flat_grey_tile_is_refused_too():
    assert not texture.tile_is_usable(texture.tile_score(_noise(spread=0.2)))


def test_a_seed_from_a_name_is_the_same_in_every_process():
    """`hash()` is randomised per interpreter, so a tile set seeded that way
    cannot be reproduced — which is how one bad tile reached a whole town."""
    import subprocess
    import sys

    code = "from city_builder.texture import stable_seed; print(stable_seed('grass'))"
    seen = {subprocess.run([sys.executable, "-c", code], capture_output=True,
                           text=True, check=True).stdout.strip()
            for _ in range(3)}
    assert len(seen) == 1
    assert texture.stable_seed("grass") != texture.stable_seed("gravel")


def test_a_tile_that_misses_is_drawn_again(monkeypatch, tmp_path):
    draws = []

    def make_tile(prompt, options, *, negative_prompt=""):
        draws.append(options.seed)
        # The first two are clipart on white; the third is ground.
        tile = _noise(seed=len(draws))
        if len(draws) < 3:
            tile[:] = 255
        return tile

    monkeypatch.setattr(texture, "make_tile", make_tile)
    got = texture.ground_tile("grass", texture.TextureOptions(seed=5),
                              path=str(tmp_path / "g.png"), attempts=4)
    assert got["tries"] == 3 and got["usable"]
    assert len(set(draws)) == 3, "the same seed was drawn again"


def test_the_least_bad_draw_is_kept_when_none_pass(monkeypatch, tmp_path):
    def make_tile(prompt, options, *, negative_prompt=""):
        tile = _noise(seed=options.seed, spread=90.0)
        if options.seed % 2 == 0:
            tile[:] = 255          # worse: blown out entirely
        return tile

    monkeypatch.setattr(texture, "make_tile", make_tile)
    got = texture.ground_tile("grass", texture.TextureOptions(seed=1),
                              path=str(tmp_path / "g.png"), attempts=3)
    assert not got["usable"]
    assert got["blown"] < 0.5, "the blown-out draw was kept over the merely harsh one"
