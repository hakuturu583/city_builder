"""Deciding what to spend on a plot, and what to keep of what came back.

No GPU: what is testable here is the arithmetic that runs before and after the
model, which is where the street-scale decisions live.
"""

from __future__ import annotations

import json

import pytest
from shapely.affinity import rotate
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import district as D


def _plot(long_side=20.0, short_side=12.0, angle=0.0, area=None):
    box = ShapelyPolygon([(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
                          (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)])
    ring = rotate(box, angle, origin=(0, 0))
    return {"footprint": [list(p) for p in list(ring.exterior.coords)[:-1]],
            "area": area if area is not None else long_side * short_side}


# ---------------------------------------------------------------------------
# What a plot is worth rebuilding
# ---------------------------------------------------------------------------


def test_a_plot_without_a_footprint_cannot_be_fitted_back_and_is_not_tried():
    assert not D.worth_rebuilding({"area": 400.0})
    assert D.worth_rebuilding(_plot())


def test_a_plot_below_the_area_asked_for_is_skipped():
    assert not D.worth_rebuilding(_plot(area=30.0), min_area=50.0)
    assert D.worth_rebuilding(_plot(area=80.0), min_area=50.0)


# ---------------------------------------------------------------------------
# How much licence the image model gets
#
# The measurement behind this: over 200 real plots the model returned a plan
# 1.4 to 1.5 times as long as it was deep whatever it was shown, so the drop
# rate tracked the plot's own aspect — 89 % kept under 1.5:1, 7 % over 3:1.
# ---------------------------------------------------------------------------


def test_a_square_plot_gets_the_whole_brush_up():
    assert D.licence(_plot(20.0, 18.0), 0.55) == pytest.approx(0.55)


def test_a_long_thin_plot_gets_less_of_it():
    assert D.licence(_plot(40.0, 10.0), 0.55) < 0.35


def test_the_licence_falls_as_the_plot_gets_longer():
    strengths = [D.licence(_plot(ratio * 10.0, 10.0), 0.55) for ratio in (1.0, 2.0, 3.0, 5.0)]
    assert strengths == sorted(strengths, reverse=True)
    assert strengths[0] > strengths[-1]


def test_the_licence_never_falls_below_the_floor():
    """Below it the massing comes back unchanged, which is a box on a street."""
    assert D.licence(_plot(100.0, 5.0), 0.55) == pytest.approx(0.30)


def test_a_caller_asking_for_little_licence_is_not_given_more():
    assert D.licence(_plot(40.0, 10.0), 0.2) == pytest.approx(0.2)


def test_the_aspect_is_the_plots_own_and_not_the_streets():
    """A plot on a street that does not run north is not thereby elongated."""
    for angle in (0.0, 27.0, 45.0, 63.0):
        assert D.licence(_plot(20.0, 18.0, angle), 0.55) == pytest.approx(0.55)


def test_a_plot_with_no_footprint_is_left_at_the_callers_strength():
    assert D.licence({"area": 90.0}, 0.55) == pytest.approx(0.55)


# ---------------------------------------------------------------------------
# Asking again
#
# A generation that comes back as the wrong shape is a draw from a
# distribution, not a verdict on the plot. These stub out the two GPU steps and
# check the loop around them.
# ---------------------------------------------------------------------------


class _Scene:
    name = "test"
    map_path = "test.osm"

    def __init__(self, plots):
        self.result = type("R", (), {"plots": plots})()


def _stub(monkeypatch, ious):
    """Make the pipeline hand back ``ious`` in order, and record the seeds."""
    from city_builder import portrait as portrait_module
    from city_builder import reconstruct as reconstruct_module

    seen = []

    def portrait(scene, index, path, **kwargs):
        return {"image": path}

    def reconstruct(plot, out_dir, *, image, name, mesh_options, restyle_prompt,
                    restyle_options):
        seen.append({"name": name, "seed": mesh_options.seed})
        got = ious[len(seen) - 1]
        if isinstance(got, Exception):
            raise got
        return {"footprint_iou": got, "glb": f"{name}.glb", "took_seconds": 1.0}

    monkeypatch.setattr(portrait_module, "render_portrait", portrait)
    monkeypatch.setattr(reconstruct_module, "reconstruct", reconstruct)
    return seen


def test_a_building_that_fits_first_time_is_not_asked_for_again(monkeypatch, tmp_path):
    seen = _stub(monkeypatch, [0.93])
    summary = D.rebuild(_Scene([_plot()]), str(tmp_path), attempts=3, verbose=False)
    assert len(seen) == 1
    assert summary["used"] == 1 and summary["generations"] == 1


def test_a_miss_is_drawn_again_and_the_second_draw_can_stand(monkeypatch, tmp_path):
    seen = _stub(monkeypatch, [0.61, 0.88])
    summary = D.rebuild(_Scene([_plot()]), str(tmp_path), attempts=3, verbose=False)
    assert len(seen) == 2
    assert summary["used"] == 1 and summary["retried"] == 1
    assert summary["buildings"][0]["footprint_iou"] == 0.88


def test_the_draws_differ_and_do_not_overwrite_each_other(monkeypatch, tmp_path):
    seen = _stub(monkeypatch, [0.61, 0.62, 0.63])
    D.rebuild(_Scene([_plot()]), str(tmp_path), attempts=3, verbose=False)
    assert len({row["seed"] for row in seen}) == 3, "the same seed draws the same building"
    assert len({row["name"] for row in seen}) == 3, "a worse try overwrote a better one"


def test_the_best_of_the_tries_is_the_one_kept(monkeypatch, tmp_path):
    _stub(monkeypatch, [0.61, 0.78, 0.55])
    summary = D.rebuild(_Scene([_plot()]), str(tmp_path), attempts=3, verbose=False)
    row = summary["buildings"][0]
    assert row["footprint_iou"] == 0.78 and not row["used"]
    assert row["glb"].endswith("_t1.glb"), "the ledger points at a mesh that lost"


def test_an_exception_is_a_miss_and_not_the_end_of_the_building(monkeypatch, tmp_path):
    _stub(monkeypatch, [RuntimeError("CUDA out of memory"), 0.91])
    summary = D.rebuild(_Scene([_plot()]), str(tmp_path), attempts=3, verbose=False)
    assert summary["used"] == 1
    assert "error" not in summary["buildings"][0]


def test_a_building_that_only_ever_fails_is_recorded_and_left_alone(monkeypatch, tmp_path):
    _stub(monkeypatch, [RuntimeError("boom")] * 3)
    summary = D.rebuild(_Scene([_plot()]), str(tmp_path), attempts=3, verbose=False)
    row = summary["buildings"][0]
    assert not row["used"] and "boom" in row["error"] and row["tries"] == 3


def test_one_attempt_is_the_old_behaviour(monkeypatch, tmp_path):
    seen = _stub(monkeypatch, [0.4, 0.9])
    D.rebuild(_Scene([_plot()]), str(tmp_path), attempts=1, verbose=False)
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# Generating inside the plot rather than fitting afterwards
# ---------------------------------------------------------------------------


def _stub_envelope(monkeypatch):
    """Record what the envelope route was asked for, and refuse the other one."""
    from city_builder import portrait as portrait_module
    from city_builder import reconstruct as reconstruct_module

    seen = []

    def envelope(plot, out_dir, *, image, name, mesh_options, **kwargs):
        seen.append({"name": name, "image": image, "seed": mesh_options.seed, **kwargs})
        return {"footprint_iou": 0.9, "glb": f"{name}.glb", "took_seconds": 1.0,
                "voxels": 4096, "grid": 32}

    def refuse(*_args, **_kwargs):
        raise AssertionError("the massing was photographed on the envelope route")

    monkeypatch.setattr(reconstruct_module, "reconstruct_in_envelope", envelope)
    monkeypatch.setattr(reconstruct_module, "reconstruct", refuse)
    monkeypatch.setattr(portrait_module, "render_portrait", refuse)
    return seen


def test_photographs_replace_the_massing_render_entirely(monkeypatch, tmp_path):
    """Nothing is photographed and nothing is brushed up: the picture is a
    photograph of a building, and the plot holds the shape."""
    seen = _stub_envelope(monkeypatch)
    summary = D.rebuild(_Scene([_plot()]), str(tmp_path), photos=["a.png"],
                        eave_room=0.9, verbose=False)
    assert len(seen) == 1 and seen[0]["image"] == "a.png"
    assert seen[0]["eave_room"] == 0.9
    assert summary["used"] == 1


def test_the_street_takes_the_photographs_in_turn(monkeypatch, tmp_path):
    """Assigning by area or at random clusters one material along a street."""
    seen = _stub_envelope(monkeypatch)
    D.rebuild(_Scene([_plot()] * 6), str(tmp_path), photos=["a.png", "b.png", "c.png"],
              verbose=False)
    assert [row["image"] for row in seen] == ["b.png", "c.png", "a.png"] * 2


def test_without_photographs_nothing_changes(monkeypatch, tmp_path):
    seen = _stub(monkeypatch, [0.93])
    D.rebuild(_Scene([_plot()]), str(tmp_path), verbose=False)
    assert len(seen) == 1


# ---------------------------------------------------------------------------
# The ledger as a type
#
# It is the only statement of how a reconstruction goes back into the world — a
# mesh out of TRELLIS.2 is normalised into a unit cube and knows neither its
# size nor its heading — so it is worth being something a consumer can read,
# write and hand on without losing anything.
# ---------------------------------------------------------------------------


def _row(**kwargs):
    fields = {"building": 1, "area_m2": 90.0, "roof": "hip", "used": True,
              "footprint_iou": 0.93, "yaw_deg": 12.0, "scale": 14.0,
              "scale_xy": [14.5, 13.5], "stretch": 1.07, "stretch_deg": 90.0,
              "centre": [10.0, 20.0], "base_z": 3.0, "glb": "b0001.glb"}
    return D.Rebuilt(**{**fields, **kwargs})


def test_a_row_round_trips_through_json():
    row = _row()
    assert D.Rebuilt.from_json(row.to_json()) == row


def test_a_key_from_a_newer_version_survives_a_round_trip():
    """An older reader must not silently drop what it does not understand."""
    raw = {**_row().to_json(), "provenance": "some-future-stage", "score": 0.5}
    back = D.Rebuilt.from_json(raw)
    assert back.extra == {"provenance": "some-future-stage", "score": 0.5}
    assert back.to_json() == raw


def test_the_empty_fields_are_left_out_rather_than_written_as_null():
    written = D.Rebuilt(building=7).to_json()
    assert "glb" not in written and "footprint_iou" not in written
    assert written["building"] == 7 and written["used"] is False


def test_a_district_counts_what_stands_and_what_it_cost():
    ledger = D.District(scene="s", map="m.osm", kept_above=0.8, buildings=[
        _row(), _row(building=2, used=False, footprint_iou=0.61, tries=3),
        _row(building=3, footprint_iou=0.85)])
    summary = ledger.to_json()
    assert summary["attempted"] == 3 and summary["used"] == 2
    assert summary["generations"] == 5 and summary["retried"] == 1
    assert summary["footprint_iou"]["min"] == pytest.approx(0.85)
    assert summary["footprint_iou"]["mean"] == pytest.approx(0.89)


def test_a_district_round_trips_through_a_file(tmp_path):
    ledger = D.District(scene="s", map="m.osm", buildings=[_row(), _row(building=2)])
    path = str(tmp_path / "district.json")
    ledger.write(path)
    back = D.District.read(path)
    assert back.scene == "s" and back.map == "m.osm"
    assert [r.building for r in back.buildings] == [1, 2]
    assert back.buildings[0] == ledger.buildings[0]


def test_a_ledger_written_before_this_was_a_type_still_reads():
    """Plain dicts on disk, from every run so far."""
    raw = {"scene": "old", "map": "m.osm", "kept_above": 0.8, "buildings": [
        {"building": 4, "area_m2": 80.0, "used": True, "footprint_iou": 0.9,
         "yaw_deg": 0.0, "scale": 10.0, "centre": [0.0, 0.0], "base_z": 1.0,
         "glb": "b0004.glb"}]}
    ledger = D.District.from_json(raw)
    assert len(ledger.standing) == 1
    assert ledger.standing[0].scale_xy is None, "a fit written before the stretch existed"


def test_rebuild_hands_back_a_summary_that_reads_the_same_as_the_file(monkeypatch, tmp_path):
    _stub(monkeypatch, [0.91])
    summary = D.rebuild(_Scene([_plot()]), str(tmp_path), verbose=False)
    with open(tmp_path / "district.json", encoding="utf-8") as handle:
        assert json.load(handle) == summary
