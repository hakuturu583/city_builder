"""The order the stages run in, and what they hand each other.

The stages themselves are a GPU each; what is testable — and what is easy to
get wrong by hand, which is why this module exists at all — is the wiring: that
the tiles are made before the massing is photographed wearing them, that the
facade families match the floor counts this map actually produced, and that a
stage whose output is already there is not paid for twice.
"""

from __future__ import annotations

import json
import os

import pytest

from city_builder import pipeline as P


class _Plot(dict):
    pass


def _result(floors=(1, 2, 3)):
    class _Mesh:
        faces = [[0, 1, 2]] * 7

    class _Frame:
        ref_lat, ref_lon = 35.9, 139.9

    return type("R", (), {
        "plots": [_Plot(floors=n, centroid=[0.0, 0.0], footprint=[[0, 0], [1, 0], [1, 1]])
                  for n in floors],
        "groups": {"Ground": [_Mesh()]},
        "frame": _Frame(),
        "stats": {},
    })()


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    """Every stage that costs a GPU, replaced by a note of what it was asked."""
    calls = []

    def build(map_path, config, *, buildings=True, verbose=True, cover_options=None):
        calls.append(("build", {"cover": cover_options is not None}))
        return _result()

    def manifest(result, path):
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{}")

    def make_tile(prompt, options):
        calls.append(("tile", {"prompt": prompt[:24], "seed": options.seed}))
        return __import__("numpy").zeros((8, 8, 3), dtype=float)

    def save_tile(tile, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as handle:
            handle.write(b"x")
        return path

    def draw_family(out, counts, **kwargs):
        calls.append(("layouts", {"counts": list(counts)}))
        control = os.path.join(out, "control")
        os.makedirs(control, exist_ok=True)
        return {"sheets": ["a"], "control_dir": control, "floor_alignment": 0.9}

    def paint_family(control_dir, out, **kwargs):
        calls.append(("facades", {"floors": list(kwargs.get("floors") or [])}))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "f_1f_00.png"), "wb") as handle:
            handle.write(b"x")
        return {"written": 1, "dropped": 0, "scores": [], "seconds": 1.0,
                "floor_alignment": 0.7, "diversity": 0.3, "saturation": [0.1, 0.2]}

    def rebuild(scene, out, **kwargs):
        calls.append(("reconstruct", {"facade_dir": kwargs.get("facade_dir"),
                                      "roof": kwargs.get("roof_texture"),
                                      "attempts": kwargs.get("attempts")}))
        os.makedirs(out, exist_ok=True)
        with open(os.path.join(out, "district.json"), "w", encoding="utf-8") as handle:
            json.dump({"used": 1, "attempted": 1, "buildings": []}, handle)
        return {"used": 1, "attempted": 1, "buildings": []}

    def place(scene, ledger, **kwargs):
        calls.append(("place", {"ledger": str(ledger)}))
        return {"placed": 1, "procedural": 0}

    import city_builder.build as build_module
    import city_builder.district as district_module
    import city_builder.facade_layout as layout_module
    import city_builder.texture as texture_module

    monkeypatch.setattr(build_module, "build_city_from_config", build)
    monkeypatch.setattr(build_module, "write_manifest", manifest)
    monkeypatch.setattr(texture_module, "make_tile", make_tile)
    monkeypatch.setattr(texture_module, "save_tile", save_tile)
    monkeypatch.setattr(texture_module, "seam_error", lambda tile: 0.5)
    monkeypatch.setattr(texture_module, "paint_family", paint_family)
    monkeypatch.setattr(layout_module, "draw_family", draw_family)
    monkeypatch.setattr(district_module, "rebuild", rebuild)
    monkeypatch.setattr(district_module, "place", place)
    return calls


def _named(calls, name):
    return [payload for kind, payload in calls if kind == name]


# ---------------------------------------------------------------------------
# The order
# ---------------------------------------------------------------------------


def test_every_stage_runs_and_in_the_order_they_depend_on_each_other(stubbed, tmp_path):
    report = P.run("map.osm", str(tmp_path), recipe=P.Recipe(renders=False, glb=False),
                   stages=("ground", "materials", "facades", "reconstruct"), verbose=False)
    assert list(report["stages"]) == ["ground", "materials", "facades", "reconstruct"]
    kinds = [kind for kind, _payload in stubbed]
    assert kinds.index("tile") < kinds.index("reconstruct")
    assert kinds.index("facades") < kinds.index("reconstruct")


def test_the_tiles_exist_before_the_massing_is_photographed_wearing_them(stubbed, tmp_path):
    """The picture handed to the 3D model is the whole of what it knows."""
    P.run("map.osm", str(tmp_path), recipe=P.Recipe(renders=False, glb=False),
          stages=("ground", "materials", "facades", "reconstruct"), verbose=False)
    asked = _named(stubbed, "reconstruct")[0]
    assert asked["roof"] and asked["roof"].endswith("kawara.png")
    assert os.path.exists(asked["roof"])
    assert asked["facade_dir"] and os.listdir(asked["facade_dir"])


def test_the_facade_families_are_the_floor_counts_this_map_produced(stubbed, tmp_path):
    """A sheet drawn for six storeys puts six rows of windows on a bungalow."""
    P.run("map.osm", str(tmp_path), recipe=P.Recipe(renders=False, glb=False),
          stages=("ground", "facades"), verbose=False)
    assert _named(stubbed, "layouts")[0]["counts"] == [1, 2, 3]
    assert _named(stubbed, "facades")[0]["floors"] == [1, 2, 3]


def test_the_recipe_reaches_the_stage_that_uses_it(stubbed, tmp_path):
    P.run("map.osm", str(tmp_path), recipe=P.Recipe(attempts=7, renders=False, glb=False),
          stages=("ground", "reconstruct"), verbose=False)
    assert _named(stubbed, "reconstruct")[0]["attempts"] == 7


def test_the_ground_settings_land_on_the_config_the_build_reads():
    from city_builder.config import CityConfig

    config = CityConfig()
    P._apply(P.Recipe(cell=1.5, relief_amplitude=9.0), config)
    assert config.ground.cell == 1.5 and config.ground.relief_amplitude == 9.0
    assert config.ground.relief is True


# ---------------------------------------------------------------------------
# Not paying twice
# ---------------------------------------------------------------------------


def test_a_tile_that_is_already_there_is_not_made_again(stubbed, tmp_path):
    for _ in range(2):
        P.run("map.osm", str(tmp_path), recipe=P.Recipe(renders=False, glb=False),
              stages=("ground", "materials"), verbose=False)
    made = len(_named(stubbed, "tile"))
    assert made == len(P.COVER_PROMPTS) + 1, "the second run paid for the tiles again"


def test_forcing_a_stage_makes_it_run_anyway(stubbed, tmp_path):
    for force in ((), ("materials",)):
        P.run("map.osm", str(tmp_path), recipe=P.Recipe(renders=False, glb=False),
              stages=("ground", "materials"), force=force, verbose=False)
    assert len(_named(stubbed, "tile")) == 2 * (len(P.COVER_PROMPTS) + 1)


def test_facades_already_drawn_are_left_alone(stubbed, tmp_path):
    for _ in range(2):
        P.run("map.osm", str(tmp_path), recipe=P.Recipe(renders=False, glb=False),
              stages=("ground", "facades"), verbose=False)
    assert len(_named(stubbed, "facades")) == 1


# ---------------------------------------------------------------------------
# Starting partway through
# ---------------------------------------------------------------------------


def test_a_stage_run_on_its_own_rebuilds_what_it_needs(stubbed, tmp_path):
    """`--stages reconstruct` has no ground in hand and has to make one."""
    P.run("map.osm", str(tmp_path), recipe=P.Recipe(renders=False, glb=False),
          stages=("reconstruct",), verbose=False)
    assert _named(stubbed, "build"), "the reconstruction ran with no ground under it"
    assert _named(stubbed, "reconstruct")


def test_the_run_writes_down_what_it_did(stubbed, tmp_path):
    P.run("map.osm", str(tmp_path), recipe=P.Recipe(renders=False, glb=False),
          stages=("ground", "materials"), verbose=False)
    with open(tmp_path / "pipeline.json", encoding="utf-8") as handle:
        report = json.load(handle)
    assert report["map"] == "map.osm"
    assert set(report["stages"]) == {"ground", "materials"}
    assert report["stages"]["ground"]["plots"] == 3
    assert all("seconds" in stage for stage in report["stages"].values())
