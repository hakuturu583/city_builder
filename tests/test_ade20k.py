"""Saying this package's surfaces in a vocabulary a ControlNet was trained on.

The failure this guards against is quiet. A segmentation ControlNet does not
read a class render as an arbitrary index map; the colour *is* the class, so
handing it this package's own debug palette does not weaken the signal, it
replaces it with a confident wrong one. Nothing raises, and the model draws
whatever those colours happen to mean in ADE20K.

So what is checked here is that every group has a translation, that the
translation lands on ADE20K's real colours rather than near-misses, and that a
colour nobody recognises survives untouched — because a mistake that shows up as
this package's own pink is one somebody will notice.
"""

from __future__ import annotations

import numpy as np

from city_builder import ade20k
from city_builder.classes import CLASSES


def test_every_surface_class_has_something_to_be_called():
    assert set(ade20k.AS_ADE20K) == set(CLASSES)


def test_every_translation_names_a_colour_we_actually_have():
    for group, name in ade20k.AS_ADE20K.items():
        assert name in ade20k.ADE20K, f"{group} is called {name}, which is not ADE20K"


def test_the_palette_is_the_one_controlnet_reads():
    # Straight from ControlNet's own table. If these drift, every control image
    # this module makes is mislabelled and nothing will say so.
    assert ade20k.ADE20K["building"] == (180, 120, 120)
    assert ade20k.ADE20K["sky"] == (6, 230, 230)
    assert ade20k.ADE20K["road"] == (140, 140, 140)
    assert ade20k.ADE20K["sidewalk"] == (235, 255, 7)
    assert ade20k.ADE20K["tree"] == (4, 200, 3)


def test_a_render_of_one_class_comes_back_as_that_class():
    painted = np.tile(
        np.array([[round(c * 255) for c in CLASSES["Buildings"].mask_colour]],
                 dtype=np.uint8), (4, 4, 1))
    assert (ade20k.translate(painted) == ade20k.ADE20K["building"]).all()


def test_the_paint_on_a_road_is_still_road():
    # ADE20K has no class for lane markings, and calling them anything else
    # would put a different surface in the middle of the carriageway.
    for group in ("LaneMarkings", "StopLines", "CrosswalkStripes"):
        assert ade20k.AS_ADE20K[group] == "road"


def test_a_kerb_is_pavement_rather_than_road():
    assert ade20k.AS_ADE20K["Curbs"] == "sidewalk"
    assert ade20k.AS_ADE20K["Walkways"] == "sidewalk"


def test_the_render_comes_back_a_shade_off_and_is_still_recognised():
    exact = np.array([[round(c * 255) for c in CLASSES["Roads"].mask_colour]],
                     dtype=np.int16)
    painted = np.tile((exact + 3).astype(np.uint8), (2, 2, 1))
    assert (ade20k.translate(painted) == ade20k.ADE20K["road"]).all()


def test_a_colour_nobody_claims_is_left_alone():
    # Left as it was rather than assigned to the nearest class: a mistake that
    # survives as this package's own colours is one somebody sees.
    painted = np.tile(np.array([[[7, 200, 90]]], dtype=np.uint8), (2, 2, 1))
    assert (ade20k.translate(painted) == 
            np.array([7, 200, 90], dtype=np.uint8)).all()


def test_the_sky_is_painted_where_nothing_was_hit():
    painted = np.zeros((4, 4, 3), np.uint8)
    nothing = np.zeros((4, 4), bool)
    nothing[0] = True
    out = ade20k.translate(painted, sky=nothing)
    assert (out[0] == ade20k.ADE20K["sky"]).all()
    assert not (out[1] == ade20k.ADE20K["sky"]).all()


def test_coverage_adds_up_over_a_frame_of_two_classes():
    painted = np.zeros((4, 4, 3), np.uint8)
    painted[:2] = [round(c * 255) for c in CLASSES["Roads"].mask_colour]
    painted[2:] = [round(c * 255) for c in CLASSES["Buildings"].mask_colour]
    share = ade20k.coverage(ade20k.translate(painted))
    assert share["road"] == 0.5 and share["building"] == 0.5
