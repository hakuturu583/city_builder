"""This package's surface classes, said in the language a ControlNet speaks.

`classes.py` already carries a colour per class, but those were chosen to be
told apart by eye in a debug render. A segmentation ControlNet is not looking at
them that way: it was trained on ADE20K, where the colour *is* the class name.
Handing it this package's own palette does not give it a weak signal, it gives
it a confident wrong one — the pink that means "Buildings" here is nothing like
ADE20K's building, and the model will draw whatever that colour does mean.

So the render is recoloured. Several of this package's classes have no ADE20K
counterpart and are folded into the nearest one that does: the paint on a road
is road, a kerb is sidewalk, a viaduct deck is road. The alternative — inventing
a colour — would be the same mistake again.

The sky is not in the mesh at all and so is not in the render either. It is
recovered from the depth pass, where a ray that hit nothing is the sky, and
painted in. Without it the model is being told the upper half of a street scene
is unlabelled, and it fills that with anything.
"""
from __future__ import annotations

import numpy as np

#: Straight from ControlNet's own ADE20K table, not from memory.
ADE20K = {
    "wall": (120, 120, 120),
    "building": (180, 120, 120),
    "sky": (6, 230, 230),
    "tree": (4, 200, 3),
    "road": (140, 140, 140),
    "grass": (4, 250, 7),
    "sidewalk": (235, 255, 7),
    "earth": (120, 120, 70),
    "plant": (204, 255, 4),
    "water": (61, 230, 250),
    "fence": (255, 184, 6),
    "path": (255, 31, 0),
    "pole": (51, 0, 255),
}

#: Which ADE20K class each of this package's surface groups is said to be.
#: Lane paint is road because ADE20K has no class for it and the paint is a
#: texture on the carriageway, not a surface of its own.
AS_ADE20K = {
    "Ground": "earth",
    "Roads": "road",
    "Junctions": "road",
    "RoadInfill": "road",
    "Crosswalks": "road",
    "LaneMarkings": "road",
    "StopLines": "road",
    "CrosswalkStripes": "road",
    "Walkways": "sidewalk",
    "Curbs": "sidewalk",
    "Buildings": "building",
    # A roof seen from the street is the top of a building, and ADE20K has no
    # separate roof; calling it "house" instead would put a detached dwelling
    # on top of every block.
    "Roofs": "building",
    "ViaductDecks": "road",
    "ViaductParapets": "wall",
    "ViaductPiers": "wall",
    "Water": "water",
    "Fences": "fence",
    "Poles": "pole",
    "Trees": "tree",
}


def own_palette() -> dict[tuple[int, int, int], str]:
    """This package's debug colours as 8-bit RGB, keyed to their group."""
    from .classes import CLASSES

    return {tuple(round(c * 255) for c in surface.mask_colour): name
            for name, surface in CLASSES.items()}


def translate(painted: np.ndarray, *, sky: np.ndarray | None = None,
              tolerance: int = 12) -> np.ndarray:
    """Recolour a class render into ADE20K, and paint the sky if given.

    ``painted`` is the segmentation pass as 8-bit RGB. Matching is nearest
    colour rather than exact, because the render goes through a colour transform
    and comes back a shade or two off; anything further than ``tolerance`` from
    every known class is left as it was, so a mistake shows up as this package's
    own colours surviving into the output rather than as a silent reassignment.
    """
    lookup = own_palette()
    known = np.array(list(lookup.keys()), dtype=np.int16)
    groups = list(lookup.values())

    flat = painted.reshape(-1, 3).astype(np.int16)
    distance = np.abs(flat[:, None, :] - known[None, :, :]).sum(-1)
    nearest = distance.argmin(1)
    close = distance[np.arange(len(flat)), nearest] <= tolerance * 3

    out = flat.copy()
    for index, group in enumerate(groups):
        name = AS_ADE20K.get(group)
        if name is None:
            continue
        picked = close & (nearest == index)
        out[picked] = ADE20K[name]

    out = out.reshape(painted.shape).astype(np.uint8)
    if sky is not None:
        out[sky] = ADE20K["sky"]
    return out


def coverage(translated: np.ndarray) -> dict[str, float]:
    """What fraction of the image each ADE20K class ended up holding."""
    flat = translated.reshape(-1, 3)
    out = {}
    for name, colour in ADE20K.items():
        share = float((flat == np.array(colour, np.uint8)).all(-1).mean())
        if share > 0:
            out[name] = share
    return out
