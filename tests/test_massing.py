"""A building with something going on, and the promise that it stays on its plot.

The one claim that has to hold whatever the dice say is containment: a courtyard,
a wing, a forecourt and a wall are all cut from the plot polygon, so none of them
can leave it. Everything else here is about the variety being real — a module
that draws the same building every time is a slower way to have a box.
"""

from __future__ import annotations

import math

import pytest
from shapely.affinity import rotate, translate
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import massing as M
from city_builder.buildings import pitched_roof


def _plot(long_side=30.0, short_side=18.0, angle=0.0, at=(0.0, 0.0), height=12.0, base=2.0):
    box = ShapelyPolygon([(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
                          (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)])
    ring = list(translate(rotate(box, angle, origin=(0, 0)), *at).exterior.coords)[:-1]
    return {"footprint": [list(p) for p in ring], "height": height, "base_z": base,
            "floors": max(1, round(height / 3.5))}


def _features(plot, seeds=range(40), **kwargs):
    options = M.MassingOptions(**kwargs) if kwargs else None
    return [tuple(sorted(M.plan(plot, options, seed)["features"])) for seed in seeds]


# ---------------------------------------------------------------------------
# The promise
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("angle", [0.0, 31.0, 90.0])
@pytest.mark.parametrize("shape", [(30.0, 18.0), (48.0, 11.0), (20.0, 19.0)])
def test_nothing_a_building_does_leaves_its_plot(angle, shape):
    plot = _plot(*shape, angle=angle, at=(120.0, -40.0))
    for seed in range(30):
        laid_out = M.plan(plot, seed=seed)
        assert M.within_plot(plot, laid_out), (
            f"seed {seed} put {laid_out['features']} outside the plot")


def test_every_piece_stands_on_the_ground_it_was_given_or_on_the_roof():
    plot = _plot(base=7.5, height=12.0)
    for seed in range(20):
        for _polygon, bottom, top in M.plan(plot, seed=seed)["parts"]:
            assert top > bottom, "a piece with no height is a piece with no faces"
            assert bottom >= 7.5 - 1e-9, "nothing may start below the plot's ground"


def test_a_plot_with_no_area_is_refused():
    with pytest.raises(ValueError, match="area"):
        M.plan({"footprint": [[0, 0], [1, 0], [2, 0]], "height": 10.0, "base_z": 0.0})


# ---------------------------------------------------------------------------
# The variety
# ---------------------------------------------------------------------------


def test_the_street_differs_from_itself():
    """Not "does it vary" but "does it vary enough to be worth the machinery"."""
    drawn = _features(_plot())
    assert len(set(drawn)) >= 5, f"only {len(set(drawn))} kinds of building in 40 plots"
    # And no single kind is most of the street.
    commonest = max(drawn.count(kind) for kind in set(drawn))
    assert commonest < len(drawn) * 0.5


def test_each_feature_actually_appears():
    drawn = _features(_plot())
    for feature in ("courtyard", "wing", "forecourt", "wall", "parapet"):
        assert any(feature in kind for kind in drawn), f"{feature} never drawn in 40"


def test_a_probability_of_zero_means_never():
    drawn = _features(_plot(), courtyard=0.0, wing=0.0, wall=0.0, parapet=0.0,
                      roof_forms=("flat",))
    assert set(drawn) == {()}


def test_the_same_seed_draws_the_same_building():
    plot = _plot()
    first, second = M.plan(plot, seed=11), M.plan(plot, seed=11)
    assert first["features"] == second["features"]
    assert [p.wkt for p, _b, _t in first["parts"]] == [p.wkt for p, _b, _t in second["parts"]]


def test_a_courtyard_is_a_hole_and_not_a_dent():
    """It has to be enclosed, or it is a notch in the plan and reads as one."""
    plot = _plot()
    for seed in range(40):
        laid_out = M.plan(plot, seed=seed)
        if "courtyard" not in laid_out["features"]:
            continue
        main = laid_out["parts"][0][0]
        assert main.interiors, "the courtyard did not come out as an interior ring"
        return
    pytest.fail("no courtyard in 40 seeds")


def test_a_wall_needs_a_forecourt_to_stand_in():
    """The plot already has the coverage ratio in it, so the yard has to be made."""
    for seed in range(40):
        features = M.plan(_plot(), seed=seed)["features"]
        if "wall" in features:
            assert "forecourt" in features


def test_a_probability_outside_zero_to_one_is_refused():
    with pytest.raises(ValueError, match="probability"):
        M.MassingOptions(courtyard=1.4)


# ---------------------------------------------------------------------------
# Into meshes
# ---------------------------------------------------------------------------


def test_the_pieces_come_out_as_walls_and_roofs():
    built = M.build(_plot(), seed=1)
    assert built["Buildings"] and built["Roofs"]
    assert all(mesh.faces for mesh in built["Buildings"])
    # The walls carry the UVs a facade sheet needs; without them every sheet
    # lands as a single stretched texel.
    assert all(mesh.uvs for mesh in built["Buildings"])


def test_the_building_stands_where_the_plot_says():
    plot = _plot(at=(90.0, 30.0), base=4.0, height=9.0)
    built = M.build(plot, seed=2)
    tops = [v[2] for mesh in built["Roofs"] for v in mesh.vertices]
    assert max(tops) == pytest.approx(4.0 + 9.0 + M.MassingOptions().parapet_height, abs=1e-6) \
        or max(tops) == pytest.approx(4.0 + 9.0, abs=1e-6)
    plan_x = [v[0] for mesh in built["Buildings"] for v in mesh.vertices]
    assert math.isclose(sum(plan_x) / len(plan_x), 90.0, abs_tol=6.0)


# ---------------------------------------------------------------------------
# Roofs
# ---------------------------------------------------------------------------


def _rect(long_side=30.0, short_side=18.0, angle=0.0):
    box = ShapelyPolygon([(-long_side / 2, -short_side / 2), (long_side / 2, -short_side / 2),
                          (long_side / 2, short_side / 2), (-long_side / 2, short_side / 2)])
    return rotate(box, angle, origin=(0, 0))


def test_a_flat_roof_is_the_absence_of_one():
    assert pitched_roof(_rect(), 10.0, "flat").faces == []


def test_a_form_nobody_builds_is_refused():
    with pytest.raises(ValueError, match="roof form"):
        pitched_roof(_rect(), 10.0, "onion")


@pytest.mark.parametrize("form", ["gable", "hip", "mono"])
@pytest.mark.parametrize("angle", [0.0, 37.0, 90.0])
def test_the_ridge_is_the_pitch_above_the_eave(form, angle):
    eave, pitch, top = 0.7, 0.5, 10.0
    mesh = pitched_roof(_rect(30.0, 18.0, angle), top, form, pitch=pitch, eave=eave)
    heights = [v[2] for v in mesh.vertices]
    rise = pitch * (18.0 / 2 + eave)
    assert min(heights) == pytest.approx(top)
    # mono carries the whole span on one slope, so it rises twice as far.
    assert max(heights) == pytest.approx(top + (2 * rise if form == "mono" else rise))


@pytest.mark.parametrize("form", ["gable", "hip", "mono"])
def test_the_roof_overhangs_the_walls_it_sits_on(form):
    plan_shape = _rect(30.0, 18.0, angle=20.0)
    mesh = pitched_roof(plan_shape, 10.0, form, eave=0.7)
    cover = ShapelyPolygon([(x, y) for x, y, _z in mesh.vertices]).convex_hull
    assert cover.contains(plan_shape), "a roof with no overhang is a lid"


def test_a_hip_has_a_shorter_ridge_than_a_gable():
    plan_shape = _rect(30.0, 18.0)
    ridges = {}
    for form in ("gable", "hip"):
        mesh = pitched_roof(plan_shape, 10.0, form)
        top = max(v[2] for v in mesh.vertices)
        at_top = [v for v in mesh.vertices if math.isclose(v[2], top, abs_tol=1e-6)]
        ridges[form] = math.dist(at_top[0][:2], at_top[-1][:2])
    assert ridges["hip"] < ridges["gable"]


def test_a_pitch_replaces_the_flat_cap_rather_than_covering_it():
    """Two roofs in the same place is a surface the reconstruction has to choose."""
    plot = _plot()
    seed = next(s for s in range(60) if M.plan(plot, seed=s)["roof"]["form"] == "gable")
    built = M.build(plot, seed=seed)
    main_top = built["parts"][0][2]
    flat_caps = [mesh for mesh in built["Roofs"]
                 if all(math.isclose(v[2], main_top, abs_tol=1e-6) for v in mesh.vertices)]
    assert not flat_caps, "the extruded cap is still under the pitch"


def test_a_pitched_roof_and_a_parapet_are_never_both_drawn():
    for seed in range(60):
        laid_out = M.plan(_plot(), seed=seed)
        if laid_out["roof"]["form"] != "flat":
            assert "parapet" not in laid_out["features"]
