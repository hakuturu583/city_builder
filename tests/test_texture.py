"""Tileable textures and how they land on the scene.

The diffusion path needs a GPU and a model, so it is not exercised here; the
procedural path, the seam metric and the scene wiring all are.
"""

from __future__ import annotations

import numpy as np
import pytest

from city_builder import scene
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
