"""Structure under and beside an elevated road.

A Lanelet2 map surveys the driving surface and nothing else, so an elevated
lanelet arrives as a flat ribbon several metres in the air with no thickness,
no soffit and nothing holding it up. Measured on the Nishi-Shinjuku map, the
column under the deck held no geometry at all between 5 m and 9 m — from a car
on the deck the road ends in a knife edge with the city visible below it, which
reads as a height bug rather than as a missing model.

The rule for *where* a bridge exists is Galin et al., `Procedural Generation of
Roads <https://perso.liris.cnrs.fr/egalin/Articles/2010-roads.pdf>`_ (CGF 2010,
§6.1): sample the trajectory, take the difference between its height and the
terrain under it, and label each sample by that clearance. Only the stretches
labelled *bridge* get the bridge model.

That granularity is the point, and working per lanelet instead is what went
wrong first. An approach ramp is one lanelet running from deck height down to
grade, so extruding its whole length downwards drives the soffit through the
street below. A ramp is a bridge at one end and a road at the other.

Piers follow Kapu, `Procedural Generation of Bridges and Tunnels
<https://nccastaff.bournemouth.ac.uk/jmacey/MastersProject/MSc10/06ChaitanyaKapu/thesis.pdf>`_
(MSc, NCCA 2010): generated between the deck path and the same path projected
onto the terrain, then thinned.

Parapets go on the outer edge of the outermost lanelets and nowhere else. A
boundary is outer when there is no other deck just beyond it — probed against
the footprint of the elevated network, rather than inferred from lane
adjacency, because lanelets that share an edge do not always say so (this map
names 136 of its 165 elevated boundaries exactly once) and inside a junction
they overlap instead of tiling.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

from .geometry import Mesh, Ribbon

Point = tuple[float, float, float]


@dataclass
class ViaductOptions:
    """Everything about the structure of an elevated road.

    Metres throughout, and meant to be set from a config file rather than
    guessed at: a 0.15 m kerb and a 1.1 m parapet are different roads.
    """

    deck: bool = True
    parapets: bool = True
    piers: bool = True

    # Galin's h_B: where the road runs at least this far above the terrain it is
    # a bridge. Below it, the road lies on the ground and gets no structure.
    bridge_clearance: float = 2.0
    min_bridge_length: float = 10.0  # a brief hop over a dip is not a viaduct

    deck_thickness: float = 1.2  # slab plus girder depth

    parapet_height: float = 1.1  # above the deck surface
    parapet_width: float = 0.4
    parapet_min_length: float = 6.0  # shorter than this is a block, not a barrier

    pier_spacing: float = 28.0  # along the centreline
    pier_width: float = 1.8  # across the road
    pier_depth: float = 1.6  # along the road
    pier_embed: float = 0.6  # how far a pier sinks into the ground
    pier_min_clearance: float = 3.0  # below this the deck is close enough to grade

    neighbour_probe: float = 1.5  # how far past a boundary to look for another deck

    def __post_init__(self) -> None:
        for name in ("bridge_clearance", "deck_thickness", "parapet_height",
                     "parapet_width", "pier_width", "pier_depth", "neighbour_probe"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.piers and self.pier_spacing <= 0:
            raise ValueError("pier_spacing must be positive when piers are on")


# ---------------------------------------------------------------------------
# Where along a ribbon the road is actually a bridge
# ---------------------------------------------------------------------------


def centreline(ribbon) -> list[Point]:
    """The middle of a lane, from its two surveyed boundaries."""
    left, right = list(ribbon.left), list(ribbon.right)
    n = min(len(left), len(right))
    return [((left[i][0] + right[i][0]) / 2.0,
             (left[i][1] + right[i][1]) / 2.0,
             (left[i][2] + right[i][2]) / 2.0) for i in range(n)]


def clearance_profile(ribbon, heightmap) -> list[float]:
    """Height of the driving surface above the terrain, per cross-section."""
    return [z - heightmap.sample(x, y) for x, y, z in centreline(ribbon)]


def runs_of(flags: Sequence[bool]) -> list[tuple[int, int]]:
    """Inclusive index ranges of consecutive True, ignoring single samples."""
    runs, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append((start, i - 1))
            start = None
    if start is not None:
        runs.append((start, len(flags) - 1))
    return [(a, b) for a, b in runs if b > a]


def slice_ribbon(ribbon, start: int, end: int) -> Ribbon:
    """The part of a ribbon between two cross-sections, inclusive."""
    return Ribbon(ribbon.id, list(ribbon.left)[start:end + 1],
                  list(ribbon.right)[start:end + 1], dict(ribbon.attributes))


def polyline_length(points: Sequence[Sequence[float]]) -> float:
    return sum(math.dist(a[:2], b[:2]) for a, b in pairwise(points))


def bridge_runs(ribbon, heightmap, options: ViaductOptions) -> list[tuple[int, int]]:
    """The stretches of a ribbon high enough above the ground to be a bridge."""
    profile = clearance_profile(ribbon, heightmap)
    line = centreline(ribbon)
    return [
        (a, b) for a, b in runs_of([c >= options.bridge_clearance for c in profile])
        if polyline_length(line[a:b + 1]) >= options.min_bridge_length
    ]


# ---------------------------------------------------------------------------
# Which boundaries face open air
# ---------------------------------------------------------------------------


def _outward_normals(ribbon) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Unit vectors pointing away from the carriageway, per cross-section."""
    left, right = list(ribbon.left), list(ribbon.right)
    n = min(len(left), len(right))
    out_left, out_right = [], []
    for i in range(n):
        dx, dy = left[i][0] - right[i][0], left[i][1] - right[i][1]
        span = math.hypot(dx, dy)
        if span < 1e-9:
            out_left.append((0.0, 0.0))
            out_right.append((0.0, 0.0))
        else:
            out_left.append((dx / span, dy / span))
            out_right.append((-dx / span, -dy / span))
    return out_left, out_right


def occupancy(ribbons: Sequence[object], elevated: set[int], *, close_gap: float = 0.5):
    """A predicate: is this spot covered by the elevated network?

    Neighbouring lanelets are surveyed independently and do not quite meet, so
    each footprint is dilated a little before the union. Otherwise the sliver
    between two lanes reads as open air and both of them get a parapet.
    """
    from shapely.geometry import Point as ShapelyPoint
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union
    from shapely.prepared import prep

    polygons = []
    for ribbon in ribbons:
        if ribbon.id not in elevated:
            continue
        ring = [(p[0], p[1]) for p in ribbon.ring()]
        if len(ring) < 4:
            continue
        polygon = ShapelyPolygon(ring)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 1e-6:
            polygons.append(polygon.buffer(close_gap, join_style=2))

    if not polygons:
        return lambda _x, _y: False

    merged = prep(unary_union(polygons))
    return lambda x, y: merged.contains(ShapelyPoint(x, y))


def outer_flags(ribbon, occupied, probe: float) -> tuple[list[bool], list[bool]]:
    """Per cross-section, whether each boundary has open air just beyond it.

    Probing outwards is what makes this work where identity does not: it does
    not care whether two lanes share a linestring, only whether there is more
    deck on the other side of the line.
    """
    left, right = list(ribbon.left), list(ribbon.right)
    out_left, out_right = _outward_normals(ribbon)
    n = min(len(left), len(right))

    flags_left, flags_right = [], []
    for i in range(n):
        for line, normals, flags in ((left, out_left, flags_left),
                                     (right, out_right, flags_right)):
            nx, ny = normals[i]
            flags.append(not occupied(line[i][0] + nx * probe, line[i][1] + ny * probe))
    return flags_left, flags_right


# ---------------------------------------------------------------------------
# The three pieces
# ---------------------------------------------------------------------------


def deck_shell(ribbon, thickness: float, *, cap_start: bool = True, cap_end: bool = True,
               left_outer: Sequence[bool] | None = None,
               right_outer: Sequence[bool] | None = None) -> Mesh:
    """A ribbon given a soffit and sides, so it is a slab rather than a sheet.

    A side face is built only where that boundary faces open air. Between two
    lanes of one carriageway the faces would be buried against each other,
    which costs geometry and z-fights wherever the decks are not exactly level.
    """
    left, right = list(ribbon.left), list(ribbon.right)
    n = min(len(left), len(right))
    if n < 2 or thickness <= 0:
        return Mesh([], [])

    vertices: list[Point] = []
    for i in range(n):
        lx, ly, lz = left[i]
        rx, ry, rz = right[i]
        vertices.extend([
            (lx, ly, lz - thickness), (rx, ry, rz - thickness),  # soffit
            (lx, ly, lz), (rx, ry, rz),                          # deck level
        ])

    def wanted(flags, i):
        return flags is None or (flags[i] and flags[i + 1])

    faces = []
    for i in range(n - 1):
        a, b = i * 4, (i + 1) * 4
        faces.append([a + 1, a, b, b + 1])  # soffit, wound down
        if wanted(left_outer, i):
            faces.append([a + 2, a, b, b + 2])
        if wanted(right_outer, i):
            faces.append([a + 1, a + 3, b + 3, b + 1])
    if cap_start:
        faces.append([0, 1, 3, 2])
    if cap_end:
        last = (n - 1) * 4
        faces.append([last + 2, last + 3, last + 1, last])
    return Mesh(vertices, faces)


def wall(line: Sequence[Point], height: float, width: float) -> Mesh:
    """A solid wall standing on a polyline: a box swept along it.

    Solid rather than a single strip because a parapet is looked at from a few
    metres away, and a zero-thickness one shows its own back face through
    itself wherever the road curves.
    """
    if len(line) < 2 or height <= 0:
        return Mesh([], [])

    offsets = []
    for i in range(len(line)):
        before = line[max(i - 1, 0)]
        after = line[min(i + 1, len(line) - 1)]
        dx, dy = after[0] - before[0], after[1] - before[1]
        span = math.hypot(dx, dy)
        offsets.append((0.0, 0.0) if span < 1e-9 else (-dy / span * width, dx / span * width))

    vertices: list[Point] = []
    for (x, y, z), (ox, oy) in zip(line, offsets):
        vertices.extend([(x, y, z), (x + ox, y + oy, z),
                         (x, y, z + height), (x + ox, y + oy, z + height)])

    faces = []
    for i in range(len(line) - 1):
        a, b = i * 4, (i + 1) * 4
        faces.extend([
            [a + 2, a + 3, b + 3, b + 2],  # top
            [a, b, b + 2, a + 2],          # outer
            [a + 1, a + 3, b + 3, b + 1],  # inner
        ])
    last = (len(line) - 1) * 4
    faces.extend([[0, 2, 3, 1], [last + 1, last + 3, last + 2, last]])  # ends
    return Mesh(vertices, faces)


def parapet_walls(ribbon, left_outer: Sequence[bool], right_outer: Sequence[bool],
                  options: ViaductOptions) -> list[Mesh]:
    """Walls on the outer edge of this deck, and only there.

    Split into runs, so a lanelet that is outermost for part of its length —
    where a slip road peels away, say — gets a barrier over that part and
    nothing over the rest.
    """
    if not options.parapets or options.parapet_height <= 0:
        return []

    walls = []
    for line, flags, sign in ((list(ribbon.left), left_outer, 1.0),
                              (list(ribbon.right), right_outer, -1.0)):
        for start, end in runs_of(flags):
            run = line[start:end + 1]
            if polyline_length(run) < options.parapet_min_length:
                continue
            built = wall(run, options.parapet_height, sign * options.parapet_width)
            if built.faces:
                walls.append(built)
    return walls


def box(centre: Point, width: float, depth: float, bottom: float, top: float,
        heading: float) -> Mesh:
    """A box rotated about z, as a closed mesh."""
    cos, sin = math.cos(heading), math.sin(heading)
    half_d, half_w = depth / 2.0, width / 2.0
    corners = [
        (centre[0] + along * cos - across * sin, centre[1] + along * sin + across * cos)
        for along, across in ((-half_d, -half_w), (half_d, -half_w),
                              (half_d, half_w), (-half_d, half_w))
    ]
    vertices = [(x, y, bottom) for x, y in corners] + [(x, y, top) for x, y in corners]
    faces = [[3, 2, 1, 0], [4, 5, 6, 7],
             [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return Mesh(vertices, faces)


def pier_boxes(ribbon, heightmap, options: ViaductOptions) -> list[Mesh]:
    """Columns between the deck path and its projection on the terrain."""
    if not options.piers or heightmap is None:
        return []

    line = centreline(ribbon)
    if len(line) < 2:
        return []

    boxes = []
    travelled = options.pier_spacing / 2.0  # not one right on a joint
    for here, ahead in pairwise(line):
        span = math.dist(here[:2], ahead[:2])
        if span < 1e-9:
            continue
        travelled += span
        if travelled < options.pier_spacing:
            continue
        travelled -= options.pier_spacing

        soffit = here[2] - options.deck_thickness
        ground = heightmap.sample(here[0], here[1])
        if soffit - ground < options.pier_min_clearance:
            continue
        heading = math.atan2(ahead[1] - here[1], ahead[0] - here[0])
        boxes.append(box(here, options.pier_width, options.pier_depth,
                         ground - options.pier_embed, soffit, heading))
    return boxes


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def build(ribbons: Sequence[object], elevated: set[int], heightmap,
          options: ViaductOptions | None = None) -> dict[str, list[Mesh]]:
    """Deck, parapets and piers for the elevated parts of a road network.

    Needs the heightmap: with no terrain there is no clearance, and with no
    clearance there is no telling a viaduct from a road lying on the ground.
    """
    options = options or ViaductOptions()
    decks: list[Mesh] = []
    walls: list[Mesh] = []
    piers: list[Mesh] = []
    if heightmap is None:
        return {"ViaductDecks": decks, "ViaductParapets": walls, "ViaductPiers": piers}

    occupied = occupancy(ribbons, elevated)

    for ribbon in ribbons:
        if ribbon.id not in elevated:
            continue
        vertices = min(len(ribbon.left), len(ribbon.right))
        for start, end in bridge_runs(ribbon, heightmap, options):
            section = slice_ribbon(ribbon, start, end)
            left_outer, right_outer = outer_flags(section, occupied, options.neighbour_probe)

            if options.deck:
                shell = deck_shell(
                    section, options.deck_thickness,
                    # An end stopping mid-ribbon is where the deck meets grade:
                    # that is an abutment and wants a face. An end at the
                    # lanelet's own end is a joint with the next lanelet, and
                    # capping it would put a wall inside the beam.
                    cap_start=start > 0, cap_end=end < vertices - 1,
                    left_outer=left_outer, right_outer=right_outer,
                )
                if shell.faces:
                    decks.append(shell)

            walls.extend(parapet_walls(section, left_outer, right_outer, options))
            piers.extend(pier_boxes(section, heightmap, options))

    return {"ViaductDecks": decks, "ViaductParapets": walls, "ViaductPiers": piers}
