"""Poles and street trees, stood on the pavement the map now has.

A street without them does not read as a street, and no amount of asking will
put them there. A conditioned generator draws what the geometry says is
present: the class render offers building, road, pavement, ground and sky, and
the depth pass says the wall beside the kerb is flat. Between them they do not
merely fail to suggest a lamp post — they forbid one. The same shape of problem
as the missing footway, one layer up.

So they are built rather than wished for. Both are the simplest solid that
carries the silhouette, because that is all the conditioning needs: a pole is a
narrow prism and a tree is a trunk under a lump of canopy. What matters is that
they stand in the right place, cast the right outline, and arrive with a class
of their own, so that ADE20K's `pole` and `tree` can be named in the control and
the generator has somewhere to put one.

**On the pavement, not on the road.** Spacing is along the kerb, and each piece
is set back from it by its own radius plus a margin, so nothing overhangs the
carriageway. A pavement too narrow to hold a tree gets poles only.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import Mesh


@dataclass(frozen=True)
class FurnitureOptions:
    """What to stand on the pavement, and how often."""

    #: Metres between poles along a pavement.
    pole_spacing: float = 22.0
    pole_height: float = 7.5
    pole_radius: float = 0.09
    #: Metres between trees. Offset from the poles so they do not collide.
    tree_spacing: float = 16.0
    tree_height: float = 6.0
    trunk_radius: float = 0.16
    canopy_radius: float = 2.0
    #: A pavement narrower than this gets no tree.
    tree_needs: float = 1.8
    #: Clearance from the kerb, beyond the piece's own radius.
    margin: float = 0.35
    seed: int = 0


def _prism(centre, radius, low, high, sides=8):
    """A vertical prism: enough of a cylinder to cast the right outline."""
    angles = np.linspace(0.0, 2.0 * math.pi, sides, endpoint=False)
    ring = np.column_stack([np.cos(angles) * radius, np.sin(angles) * radius])
    bottom = [(centre[0] + x, centre[1] + y, low) for x, y in ring]
    top = [(centre[0] + x, centre[1] + y, high) for x, y in ring]
    vertices = bottom + top
    faces = [[i, (i + 1) % sides, sides + (i + 1) % sides, sides + i]
             for i in range(sides)]
    faces.append(list(range(sides, 2 * sides)))
    return vertices, faces


def _blob(centre, radius, rings=4, sides=8):
    """A lump of canopy. Round enough to read as foliage from the street."""
    vertices, faces = [], []
    for row in range(rings + 1):
        theta = math.pi * row / rings
        r = radius * math.sin(theta)
        z = centre[2] + radius * math.cos(theta)
        for column in range(sides):
            phi = 2.0 * math.pi * column / sides
            vertices.append((centre[0] + r * math.cos(phi),
                             centre[1] + r * math.sin(phi), z))
    for row in range(rings):
        for column in range(sides):
            a = row * sides + column
            b = row * sides + (column + 1) % sides
            faces.append([a, b, b + sides, a + sides])
    return vertices, faces


def _along(points: np.ndarray, spacing: float, first: float):
    """Positions and headings at even intervals along a polyline."""
    step = np.linalg.norm(np.diff(points[:, :2], axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(step)])
    if along[-1] < first:
        return []
    out = []
    for distance in np.arange(first, along[-1], spacing):
        index = int(np.searchsorted(along, distance)) - 1
        index = min(max(index, 0), len(points) - 2)
        span = max(along[index + 1] - along[index], 1e-9)
        t = (distance - along[index]) / span
        point = points[index] + (points[index + 1] - points[index]) * t
        heading = points[index + 1][:2] - points[index][:2]
        out.append((point, heading / max(np.linalg.norm(heading), 1e-9)))
    return out


def build(walkways, options: FurnitureOptions | None = None) -> dict:
    """``{"Poles": Mesh, "Trees": Mesh}`` stood along the pavements given.

    ``walkways`` are the build's walkway ribbons: `left` is the kerb side and
    `right` the far side, which is how `surfaces.py` pairs a walkway lanelet
    whose left bound is the road_border.
    """
    options = options or FurnitureOptions()
    rng = np.random.default_rng(options.seed)
    made = {"Poles": ([], []), "Trees": ([], []), "TreeTrunks": ([], [])}
    counts = {"poles": 0, "trees": 0}

    def add(group, vertices, faces):
        store = made[group]
        base = len(store[0])
        store[0].extend(vertices)
        store[1].extend([[i + base for i in face] for face in faces])

    for ribbon in walkways:
        kerb = np.asarray(ribbon.left, dtype=float)
        far = np.asarray(ribbon.right, dtype=float)
        if len(kerb) < 2 or len(far) < 2:
            continue
        width = float(np.linalg.norm(far[:, :2] - kerb[:, :2], axis=1).mean())
        inward = far[:, :2] - kerb[:, :2]
        inward = inward / np.maximum(np.linalg.norm(inward, axis=1, keepdims=True), 1e-9)

        def foot(index, point, radius, inward=inward, width=width):
            """A point on the pavement, clear of the kerb by the radius."""
            step = inward[min(index, len(inward) - 1)]
            reach = min(radius + options.margin, max(width - radius, 0.0))
            return (point[0] + step[0] * reach, point[1] + step[1] * reach, point[2])

        for place, _heading in _along(kerb, options.pole_spacing,
                                      float(rng.uniform(2.0, options.pole_spacing))):
            index = int(np.argmin(np.linalg.norm(kerb[:, :2] - place[:2], axis=1)))
            base = foot(index, place, options.pole_radius)
            add("Poles", *_prism(base, options.pole_radius, base[2],
                                 base[2] + options.pole_height))
            counts["poles"] += 1

        if width < options.tree_needs:
            continue
        for place, _heading in _along(kerb, options.tree_spacing,
                                      float(rng.uniform(4.0, options.tree_spacing))):
            index = int(np.argmin(np.linalg.norm(kerb[:, :2] - place[:2], axis=1)))
            base = foot(index, place, options.canopy_radius * 0.35)
            trunk = options.tree_height * 0.55
            # Its own group, because a trunk is bark and a canopy is leaves, and
            # a texture can only be tiled onto a whole mesh at once: sharing one
            # put foliage down the trunk.
            add("TreeTrunks", *_prism(base, options.trunk_radius, base[2],
                                      base[2] + trunk, sides=6))
            canopy = options.canopy_radius * float(rng.uniform(0.8, 1.15))
            add("Trees", *_blob((base[0], base[1], base[2] + trunk + canopy * 0.7),
                                canopy))
            counts["trees"] += 1

    out = {name: Mesh(vertices=v, faces=f) for name, (v, f) in made.items() if f}
    out["stats"] = counts
    return out
