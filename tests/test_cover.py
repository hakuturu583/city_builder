"""What the ground is made of: the rules, and the image they paint.

No Blender. The painter writes a PNG and the assertions read it back, because
what matters about a ground texture is what colour it is where.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from shapely.geometry import Polygon as ShapelyPolygon

from city_builder import cover as cv
from city_builder.ground import HeightMap


def _flat(nx=11, ny=11, cell=10.0, z=0.0):
    return HeightMap(0.0, 0.0, cell, np.full((ny, nx), z), np.zeros((ny, nx)))


def _palette(*names):
    # Green channel only, so a test can read one number and know the class.
    shades = {"grass": (0.0, 1.0, 0.0), "gravel": (1.0, 0.1, 0.1),
              "ground": (0.5, 0.25, 0.0), "water": (0.0, 0.0, 1.0),
              "concrete": (1.0, 0.2, 0.0)}
    return [cv.Surface(n, colour=shades.get(n, (0.5, 0.5, 0.5))) for n in names]


def _road(y0=45.0, y1=55.0):
    return ShapelyPolygon([(-50, y0), (150, y0), (150, y1), (-50, y1)])


def _plot(x0=20.0, y0=10.0, x1=40.0, y1=30.0):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


# ---------------------------------------------------------------------------
# The rules
# ---------------------------------------------------------------------------


def test_the_last_rule_to_claim_a_cell_is_the_one_that_gets_it():
    """Painter's order, so a config reads as 'grass, except…'."""
    options = cv.CoverOptions(surfaces=_palette("grass", "gravel"),
                              rules=[cv.Everywhere("grass"), cv.Everywhere("gravel")])
    assert cv.classify(_flat(), options).at(50.0, 50.0) == "gravel"


def test_a_verge_follows_the_road_and_stops():
    options = cv.CoverOptions(
        surfaces=_palette("grass", "gravel"),
        rules=[cv.Everywhere("grass"), cv.NearRoad("gravel", metres=3.0)])
    got = cv.classify(_flat(), options, roads=_road())
    assert got.at(70.0, 43.0) == "gravel"   # 2 m off the kerb
    assert got.at(70.0, 39.0) == "grass"    # 6 m off it
    assert got.at(70.0, 57.0) == "gravel"   # and on the other side


def test_the_carriageway_itself_is_not_the_verge():
    """``inner=0`` means outside the road, not inside it."""
    options = cv.CoverOptions(
        surfaces=_palette("grass", "gravel"),
        rules=[cv.Everywhere("grass"), cv.NearRoad("gravel", metres=3.0)])
    assert cv.classify(_flat(), options, roads=_road()).at(70.0, 50.0) == "grass"


def test_a_plot_can_be_given_its_own_ground_and_a_pad_inside_it():
    options = cv.CoverOptions(
        surfaces=_palette("grass", "ground", "gravel"),
        rules=[cv.Everywhere("grass"), cv.InPlot("ground"),
               cv.InPlot("gravel", shrink=4.0)])
    got = cv.classify(_flat(), options, plots=[_plot()])
    assert got.at(30.0, 20.0) == "gravel"   # the middle
    assert got.at(21.5, 20.0) == "ground"   # the strip the shrink left
    assert got.at(60.0, 20.0) == "grass"    # outside


def test_a_region_is_painted_where_it_is_put():
    park = [(60.0, 60.0), (90.0, 60.0), (90.0, 90.0), (60.0, 90.0)]
    options = cv.CoverOptions(
        surfaces=_palette("grass", "water"),
        rules=[cv.Everywhere("grass"), cv.Region("water", polygon=park)])
    got = cv.classify(_flat(), options)
    assert got.at(75.0, 75.0) == "water" and got.at(20.0, 20.0) == "grass"


def test_patches_cover_the_fraction_they_were_asked_for():
    options = cv.CoverOptions(
        surfaces=_palette("grass", "ground"),
        rules=[cv.Everywhere("grass"), cv.Patches("ground", fraction=0.3, metres=20.0)])
    fractions = cv.classify(_flat(21, 21), options).fractions()
    assert fractions["ground"] == pytest.approx(0.3, abs=0.03)


def test_patches_are_the_same_patches_at_the_same_seed():
    def run(seed):
        options = cv.CoverOptions(
            surfaces=_palette("grass", "ground"),
            rules=[cv.Everywhere("grass"),
                   cv.Patches("ground", fraction=0.3, metres=20.0, seed=seed)])
        return cv.classify(_flat(21, 21), options).index

    assert np.array_equal(run(7), run(7))
    assert not np.array_equal(run(7), run(8))


def test_a_slope_rule_reads_the_height_map():
    hm = _flat(21, 21)
    hm.z[:, 10:] = np.arange(11) * 8.0  # a bank climbing steeply to the east
    options = cv.CoverOptions(
        surfaces=_palette("grass", "gravel"),
        rules=[cv.Everywhere("grass"), cv.Steeper("gravel", degrees=25.0)])
    got = cv.classify(hm, options)
    assert got.at(150.0, 100.0) == "gravel"
    assert got.at(30.0, 100.0) == "grass"


def test_a_rule_naming_a_surface_the_palette_lacks_is_an_error():
    options = cv.CoverOptions(surfaces=_palette("grass"),
                              rules=[cv.Everywhere("marmalade")])
    with pytest.raises(KeyError, match="marmalade"):
        cv.classify(_flat(), options)


def test_an_empty_palette_is_refused():
    with pytest.raises(ValueError, match="no surfaces"):
        cv.classify(_flat(), cv.CoverOptions())


def test_the_map_round_trips_through_json():
    options = cv.CoverOptions(surfaces=_palette("grass", "gravel"),
                              rules=[cv.Everywhere("grass"),
                                     cv.NearRoad("gravel", metres=3.0)])
    got = cv.classify(_flat(), options, roads=_road())
    back = cv.CoverMap.from_json(got.to_json())
    assert back.names == got.names and np.array_equal(back.index, got.index)
    assert back.at(70.0, 43.0) == got.at(70.0, 43.0)


# ---------------------------------------------------------------------------
# Painting it
# ---------------------------------------------------------------------------


def _painted(tmp_path, options, **kwargs):
    got = cv.classify(_flat(), options, **{k: v for k, v in kwargs.items()
                                           if k in ("roads", "plots")})
    report = cv.paint(got, options, str(tmp_path / "ground.png"),
                      texels_per_metre=kwargs.get("texels_per_metre", 4.0),
                      jitter=kwargs.get("jitter", 0.0),
                      softness=kwargs.get("softness", 0.0))
    return np.asarray(Image.open(report["path"]).convert("RGB")), report


def _colour_at(image, report, x, y):
    per_metre = report["texels_per_metre"]
    ix = int((x - report["origin"][0]) * per_metre)
    iy = int((y - report["origin"][1]) * per_metre)
    return image[iy, ix]


def test_the_painted_ground_is_the_colour_the_rules_asked_for(tmp_path):
    options = cv.CoverOptions(
        surfaces=_palette("grass", "gravel"),
        rules=[cv.Everywhere("grass"), cv.NearRoad("gravel", metres=3.0)])
    image, report = _painted(tmp_path, options, roads=_road())
    assert _colour_at(image, report, 70.0, 43.0)[1] < 100      # gravel: little green
    assert _colour_at(image, report, 70.0, 20.0)[1] > 150      # grass is green


def test_the_image_covers_the_map_at_the_asked_for_resolution(tmp_path):
    options = cv.CoverOptions(surfaces=_palette("grass"), rules=[cv.Everywhere("grass")])
    _image, report = _painted(tmp_path, options, texels_per_metre=6.0)
    assert report["texels_per_metre"] == pytest.approx(6.0, rel=0.02)
    assert report["size"][0] == pytest.approx(report["metres"][0] * 6.0, rel=0.02)


def test_a_huge_map_is_capped_rather_than_asking_for_a_gigapixel(tmp_path):
    options = cv.CoverOptions(surfaces=_palette("grass"), rules=[cv.Everywhere("grass")])
    got = cv.classify(_flat(201, 201, cell=10.0), options)
    report = cv.paint(got, options, str(tmp_path / "big.png"),
                      texels_per_metre=64.0, max_side=1024)
    assert max(report["size"]) <= 1024
    assert report["texels_per_metre"] < 64.0


def test_the_boundary_is_broken_up_rather_than_ruled(tmp_path):
    """A straight edge on a 2 m grid reads as a staircase; the jitter is what
    turns it into an interlocking one, and it has to act on the weights."""
    options = cv.CoverOptions(
        surfaces=_palette("grass", "gravel"),
        rules=[cv.Everywhere("grass"), cv.NearRoad("gravel", metres=3.0)])
    got = cv.classify(_flat(), options, roads=_road())

    def edge_rows(jitter):
        report = cv.paint(got, options, str(tmp_path / f"j{jitter}.png"),
                          texels_per_metre=8.0, jitter=jitter, softness=0.0)
        image = np.asarray(Image.open(report["path"]).convert("RGB"))
        band = image[:, :, 1] > 150  # where it is still grass
        # The row at which each column stops being grass, going up the image.
        return np.array([np.argmax(~band[:, i]) for i in range(image.shape[1])])

    assert edge_rows(0.0).std() < 1e-6, "the unjittered edge is not straight"
    assert edge_rows(2.0).std() > 1.0, "the jitter did not move the boundary"


def test_the_report_says_what_the_ground_turned_out_to_be(tmp_path):
    options = cv.CoverOptions(
        surfaces=_palette("grass", "gravel"),
        rules=[cv.Everywhere("grass"), cv.NearRoad("gravel", metres=3.0)])
    _image, report = _painted(tmp_path, options, roads=_road())
    assert set(report["surfaces"]) == {"grass", "gravel"}
    assert sum(report["surfaces"].values()) == pytest.approx(1.0, abs=0.01)
