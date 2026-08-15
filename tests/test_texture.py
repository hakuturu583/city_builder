"""Tileable textures and how they land on the scene.

The diffusion path needs a GPU and a model, so it is not exercised here; the
procedural path, the seam metric and the scene wiring all are.
"""

from __future__ import annotations

import os

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


def test_the_carriageway_gets_its_tile_with_no_paint_to_carry_it(tmp_path):
    """Regression: `road_texture` did nothing on a scene with no marking pages.

    The asphalt was only reached through the marking material, so a map with no
    paint in it — or one built with markings off — took the argument, reported
    the road as dressed, and rendered it flat.
    """
    import bpy
    import numpy as np

    from city_builder.build import BuildResult, build_scene
    from city_builder.frame import LocalFrame
    from city_builder.geometry import Ribbon
    from city_builder.ground import HeightMap

    path = str(tmp_path / "asphalt.png")
    make_tile("unused", TextureOptions(size=32, diffusion=False), path=path)

    lane = Ribbon(1, [(0, 4, 0), (40, 4, 0)], [(0, -4, 0), (40, -4, 0)])
    result = BuildResult(
        frame=LocalFrame(35.0, 139.0), groups={"Roads": [lane]},
        heightmap=HeightMap(0.0, 0.0, 10.0, np.zeros((5, 5)), np.zeros((5, 5))),
        elevated=set(), z_datum=0.0,
    )
    assert not result.marking_pages, "this scene is the case under test"

    build_scene(result, road_texture=path, road_tile_metres=4.0, verbose=False)

    roads = bpy.data.objects["Roads"]
    images = [node.image.filepath for material in roads.data.materials
              for node in material.node_tree.nodes if node.type == "TEX_IMAGE" and node.image]
    assert images, "the carriageway is still wearing a flat colour"
    assert os.path.basename(images[0]) == "asphalt.png"


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
