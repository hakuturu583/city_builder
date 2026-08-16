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
def test_the_pitch_is_measured_from_the_wall_and_not_from_the_eave(form, angle):
    """The overhang is below the wall top, so the ridge is over the *span*."""
    eave, pitch, top = 0.7, 0.5, 10.0
    mesh = pitched_roof(_rect(30.0, 18.0, angle), top, form, pitch=pitch, eave=eave)
    heights = [v[2] for v in mesh.vertices]
    # A mono spreads the same rise over twice the run, so its slope — and with
    # it the drop of its overhang — is half.
    slope = pitch / (2.0 if form == "mono" else 1.0)
    assert min(heights) == pytest.approx(top - slope * eave)
    # The high point of a gable or a hip is its ridge, which is over the walls.
    # A mono's is the far eave, so its overhang goes on climbing past them.
    over = slope * eave if form == "mono" else 0.0
    assert max(heights) == pytest.approx(top + pitch * (18.0 / 2) + over)


def test_a_mono_pitch_does_not_tower_over_the_forms_beside_it():
    """The pitch is a rise over the half span, whichever way the roof runs.

    A mono climbs the whole span where a gable climbs half of it, so reading
    the same number as a slope made it exactly twice as tall. Measured over
    200 real plots before this: gables and hips rose a median 2.2 m above their
    walls and at most 4.0 m, monos a median 4.7 m and at most 8.4 m, and ten of
    the thirty-four monos stood taller than the whole building under them.
    """
    plan_shape = _rect(30.0, 18.0)
    tops = {form: max(v[2] for v in
                      pitched_roof(plan_shape, 10.0, form, pitch=0.5, eave=0.0).vertices)
            for form in ("gable", "hip", "mono")}
    assert tops["mono"] == pytest.approx(tops["gable"])
    assert tops["hip"] == pytest.approx(tops["gable"])


def test_a_wider_plot_still_gets_a_taller_mono_roof():
    """Halving the slope is not flattening it: the run still sets the rise."""
    narrow = max(v[2] for v in pitched_roof(_rect(30.0, 10.0), 10.0, "mono").vertices)
    wide = max(v[2] for v in pitched_roof(_rect(30.0, 20.0), 10.0, "mono").vertices)
    assert wide > narrow + 1.0


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
        # How far apart the highest points are, not how far apart the first and
        # last of them happen to be: the triangulation decides the order and a
        # ridge is the span of the whole set.
        ridges[form] = max(math.dist(a[:2], b[:2]) for a in at_top for b in at_top)
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


def test_the_roof_covers_the_footprint_and_no_more():
    """The defect this was rewritten for: a roof wider than the building.

    Carried on the footprint's bounding rectangle, an L-shaped or wedge-shaped
    plot got a roof standing clear of its walls on two sides — a lid resting on
    the building rather than its top. Plots cut out of the space between real
    roads are usually one of those.
    """
    from shapely.ops import unary_union

    shapes = {
        "L": ShapelyPolygon([(0, 0), (30, 0), (30, 10), (12, 10), (12, 18), (0, 18)]),
        "wedge": ShapelyPolygon([(0, 0), (30, 0), (22, 14), (0, 10)]),
        "triangle": ShapelyPolygon([(0, 0), (28, 0), (4, 16)]),
        "courtyard": ShapelyPolygon([(0, 0), (30, 0), (30, 18), (0, 18)],
                                    [[(10, 6), (20, 6), (20, 12), (10, 12)]]),
    }
    for name, plan_shape in shapes.items():
        want = plan_shape.buffer(0.5, join_style=2).area
        for form in ("gable", "hip", "mono"):
            mesh = pitched_roof(plan_shape, 10.0, form, pitch=0.5, eave=0.5)
            covered = unary_union([
                ShapelyPolygon([(mesh.vertices[i][0], mesh.vertices[i][1]) for i in face])
                .buffer(0) for face in mesh.faces])
            assert covered.area == pytest.approx(want, rel=1e-3), \
                f"{form} roof on {name} covers {covered.area:.0f} m2, wanted {want:.0f}"


def test_the_eave_is_the_only_thing_outside_the_walls():
    plan_shape = ShapelyPolygon([(0, 0), (30, 0), (30, 10), (12, 10), (12, 18), (0, 18)])
    mesh = pitched_roof(plan_shape, 10.0, "hip", eave=0.0)
    from shapely.ops import unary_union
    covered = unary_union([
        ShapelyPolygon([(mesh.vertices[i][0], mesh.vertices[i][1]) for i in face]).buffer(0)
        for face in mesh.faces])
    assert covered.area == pytest.approx(plan_shape.area, rel=1e-3)


def test_a_gable_has_one_ridge_and_a_hip_has_a_shorter_one():
    plan_shape = ShapelyPolygon([(0, 0), (30, 0), (30, 18), (0, 18)])
    spans = {}
    for form in ("gable", "hip"):
        mesh = pitched_roof(plan_shape, 10.0, form)
        top = max(v[2] for v in mesh.vertices)
        ridge = [v for v in mesh.vertices if math.isclose(v[2], top, abs_tol=1e-6)]
        spans[form] = max(math.dist(a[:2], b[:2]) for a in ridge for b in ridge)
    assert spans["hip"] < spans["gable"]


def test_the_roof_meets_the_wall_it_sits_on():
    """Where "the roof is floating" came from.

    The eave edge was put at the wall top, so the roof plane — which rises
    inward from the eave — stood `pitch * eave` clear of the building all the
    way round: at a 0.9 m eave and a 0.8 pitch, 0.72 m of daylight under it.
    The plane has to pass through the *wall* top and the overhang hang below.
    """

    plan_shape = ShapelyPolygon([(0, 0), (30, 0), (30, 18), (0, 18)])
    for form in ("gable", "hip", "mono"):
        mesh = pitched_roof(plan_shape, 10.0, form, pitch=0.8, eave=0.9)
        faces = [ShapelyPolygon([(mesh.vertices[i][0], mesh.vertices[i][1]) for i in face])
                 for face in mesh.faces]
        # Over the wall the roof has an eave on: the long sides for a gable or
        # a mono-pitch, all four for a hip. A gable *end* is a wall the roof
        # rises over, not an eave.
        probes = ((15.0, 0.0), (15.0, 18.0))
        if form == "hip":
            probes += ((0.0, 9.0), (30.0, 9.0))
        for probe in probes:
            height = _height_at(mesh, faces, probe)
            assert height is not None, f"{form}: no roof over the wall at {probe}"
            # A mono-pitch meets one wall at the top and climbs over the other.
            # …and a mono spreads its pitch over the whole span, so it climbs
            # the same as a gable would, not twice as far.
            want = 10.0 if form != "mono" or probe[1] < 9.0 else 10.0 + 0.8 * 18.0 / 2
            assert height == pytest.approx(want, abs=0.02), (
                f"{form}: the roof is {height - want:+.2f} m off the wall at {probe}")
        # And the eave really does hang below it.
        assert min(v[2] for v in mesh.vertices) < 10.0


def _height_at(mesh, faces, point):
    """Barycentric interpolation of the roof over one plan position."""
    from shapely.geometry import Point

    spot = Point(point)
    for face, shape in zip(mesh.faces, faces):
        if not shape.buffer(1e-6).contains(spot):
            continue
        (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = (mesh.vertices[i] for i in face[:3])
        det = (y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3)
        if abs(det) < 1e-12:
            continue
        a = ((y2 - y3) * (point[0] - x3) + (x3 - x2) * (point[1] - y3)) / det
        b = ((y3 - y1) * (point[0] - x3) + (x1 - x3) * (point[1] - y3)) / det
        return a * z1 + b * z2 + (1 - a - b) * z3
    return None


def test_a_gable_is_closed_by_a_wall_and_a_hip_closes_itself():
    """Where "you can see under the roof" came from.

    A hip slopes down to an eave on every side, so the roof meets the wall all
    the way round. A gable's two ends rise to the ridge and a mono-pitch's high
    side is the whole climb, and without a wall there the building is open
    under its own roof.
    """
    from city_builder.buildings import roof_walls

    plan_shape = ShapelyPolygon([(0, 0), (30, 0), (30, 18), (0, 18)])
    closing = {form: roof_walls(plan_shape, 10.0, form, pitch=0.8, eave=0.9)
               for form in ("gable", "hip", "mono")}
    assert not closing["hip"].faces, "a hip over a rectangle needs no closing wall"
    for form in ("gable", "mono"):
        mesh = closing[form]
        assert mesh.faces, f"{form} was left open"
        assert min(v[2] for v in mesh.vertices) == pytest.approx(10.0)
        assert max(v[2] for v in mesh.vertices) > 10.0


def test_the_closing_wall_reaches_the_ridge_and_not_the_corners():
    """Sampling only the ends of an edge misses the whole triangle.

    A rectangle's gable end has the eave at both corners and the ridge in the
    middle, so both ends read as "the roof already meets the wall here" — and
    the gable came back with no faces at all.
    """
    from city_builder.buildings import pitched_roof, roof_walls

    plan_shape = ShapelyPolygon([(0, 0), (30, 0), (30, 18), (0, 18)])
    roof = pitched_roof(plan_shape, 10.0, "gable", pitch=0.8, eave=0.9)
    wall = roof_walls(plan_shape, 10.0, "gable", pitch=0.8, eave=0.9)
    assert max(v[2] for v in wall.vertices) == pytest.approx(
        max(v[2] for v in roof.vertices), abs=0.01)


def test_an_l_shaped_hip_still_needs_closing():
    """It closes over a rectangle and not over an inside corner."""
    from city_builder.buildings import roof_walls

    ell = ShapelyPolygon([(0, 0), (30, 0), (30, 10), (12, 10), (12, 18), (0, 18)])
    assert roof_walls(ell, 10.0, "hip", pitch=0.8, eave=0.9).faces
