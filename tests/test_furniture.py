"""Standing the things that make a street read as a street.

The reason this exists is not decoration. A conditioned generator draws what
the geometry says is present, and a class render that never names a pole,
paired with a depth pass that says the kerb is beside a flat wall, does not
merely fail to suggest a lamp post — it forbids one. So they are built.

What is checked here is placement: on the pavement, clear of the carriageway,
spaced along it, and absent where there is no room.
"""

from __future__ import annotations

import numpy as np

from city_builder import furniture
from city_builder.furniture import FurnitureOptions
from city_builder.geometry import Ribbon


def _pavement(width=2.0, length=60.0, y=0.0):
    """A footway running along +x, kerb on the low-y side."""
    return Ribbon(id=1,
                  left=[(0.0, y, 0.0), (length, y, 0.0)],
                  right=[(0.0, y + width, 0.0), (length, y + width, 0.0)])


def test_a_pavement_gets_poles_along_it():
    made = furniture.build([_pavement()], FurnitureOptions(pole_spacing=20.0))
    assert made["stats"]["poles"] >= 2
    assert made["Poles"].faces


def test_nothing_stands_on_the_carriageway():
    # What must not cross the kerb is the foot of the thing. A canopy reaching
    # out over the road is what a street tree does, and a class render that
    # shows foliage above the carriageway is showing the truth.
    made = furniture.build([_pavement()])
    for group in ("Poles", "Trees", "TreeTrunks"):
        if group not in made:
            continue
        points = np.array(made[group].vertices)
        standing = points[points[:, 2] < 0.5]
        if not len(standing):
            continue
        assert standing[:, 1].min() >= -1e-6, f"{group} has its foot on the road"


def test_a_canopy_may_reach_over_the_road_but_its_trunk_may_not():
    made = furniture.build([_pavement(width=3.0)],
                           FurnitureOptions(pole_spacing=1e6, canopy_radius=2.0))
    assert np.array(made["TreeTrunks"].vertices)[:, 1].min() >= -1e-6
    assert np.array(made["Trees"].vertices)[:, 1].min() < 0.0, \
        "no street tree ever kept to the pavement"


def test_a_pole_stands_on_the_ground_and_reaches_its_height():
    options = FurnitureOptions(pole_height=7.5)
    made = furniture.build([_pavement()], options)
    z = np.array(made["Poles"].vertices)[:, 2]
    assert abs(z.min()) < 1e-6
    assert abs(z.max() - options.pole_height) < 1e-6


def test_a_pavement_too_narrow_for_a_tree_still_gets_poles():
    made = furniture.build([_pavement(width=1.0)],
                           FurnitureOptions(tree_needs=1.8))
    assert made["stats"]["poles"] >= 1
    assert made["stats"]["trees"] == 0


def test_spacing_is_along_the_kerb_rather_than_per_pavement():
    short = furniture.build([_pavement(length=30.0)], FurnitureOptions(pole_spacing=10.0))
    long = furniture.build([_pavement(length=90.0)], FurnitureOptions(pole_spacing=10.0))
    assert long["stats"]["poles"] > short["stats"]["poles"]


def test_a_pavement_shorter_than_the_first_gap_gets_nothing():
    made = furniture.build([_pavement(length=1.0)], FurnitureOptions(pole_spacing=22.0))
    assert made["stats"]["poles"] == 0


def test_a_tree_is_a_trunk_under_a_canopy_and_clears_the_ground():
    made = furniture.build([_pavement(width=3.0)],
                           FurnitureOptions(pole_spacing=1e6, tree_spacing=20.0,
                                            tree_height=6.0, canopy_radius=2.0))
    assert made["stats"]["trees"] >= 1
    # Separate meshes, because a texture is tiled onto all of one: sharing put
    # foliage down the trunk.
    trunk = np.array(made["TreeTrunks"].vertices)[:, 2]
    canopy = np.array(made["Trees"].vertices)[:, 2]
    assert abs(trunk.min()) < 1e-6, "the trunk does not reach the pavement"
    assert canopy.min() > trunk.min(), "the canopy starts at the ground"
    assert canopy.max() > 6.0, "the canopy is not above the trunk"


def test_the_same_seed_stands_them_in_the_same_places():
    a = furniture.build([_pavement()], FurnitureOptions(seed=7))
    b = furniture.build([_pavement()], FurnitureOptions(seed=7))
    assert np.allclose(np.array(a["Poles"].vertices), np.array(b["Poles"].vertices))


def test_a_ribbon_with_nothing_in_it_is_skipped_rather_than_raising():
    empty = Ribbon(id=1, left=[(0.0, 0.0, 0.0)], right=[(0.0, 2.0, 0.0)])
    assert furniture.build([empty])["stats"] == {"poles": 0, "trees": 0}


def test_every_group_it_makes_has_a_class_and_a_name_in_ade20k():
    from city_builder.ade20k import ADE20K, AS_ADE20K
    from city_builder.classes import CLASSES

    for group in ("Poles", "Trees", "TreeTrunks"):
        assert group in CLASSES, f"{group} has no class, so no control image names it"
        assert ADE20K[AS_ADE20K[group]]
