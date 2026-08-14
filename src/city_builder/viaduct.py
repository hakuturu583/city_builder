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

from .geometry import Mesh, Ribbon, height_lookup, triangulate_polygon

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
    parapet_bridge_gap: float = 5.0  # an inner stretch shorter than this does not break it

    pier_spacing: float = 28.0  # along the centreline
    pier_width: float = 1.8  # across the road
    pier_depth: float = 1.6  # along the road
    pier_embed: float = 0.6  # how far a pier sinks into the ground
    pier_min_clearance: float = 3.0  # below this the deck is close enough to grade

    # A boundary is outer when it lies on the outline of the whole elevated
    # network. Probing a fixed distance past it for a neighbour is the obvious
    # test and it under-reads: measured, it found 3323 m of edge where the
    # outline finds 4199 m, because a deck sitting 1.6 m away past a boundary
    # is not a neighbour but stops the probe from saying so.
    edge_buffer: float = 0.5  # dilate the footprints first, to close survey gaps
    edge_tolerance: float = 0.7  # how near the outline a boundary has to be

    infill: bool = True  # patch the slivers between lanelets
    infill_gap: float = 0.8  # widest sliver to treat as a survey gap rather than a hole
    infill_max_area: float = 80.0  # bigger than this is a real opening, not a gap

    def __post_init__(self) -> None:
        for name in ("bridge_clearance", "deck_thickness", "parapet_height",
                     "parapet_width", "pier_width", "pier_depth", "edge_tolerance"):
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


def close_gaps(flags: Sequence[bool], stations: Sequence[float], shortest: float) -> list[bool]:
    """Fill False stretches shorter than ``shortest`` metres.

    The neighbour probe flickers: a deck's edge passes within reach of another
    deck for a couple of cross-sections at a junction mouth, or the survey gap
    between two lanes happens to swallow the sample. Measured, 109 candidate
    barrier runs carried 125 flips between them, and 17 came out too short to
    build — which is a barrier with holes in it rather than a barrier.
    """
    closed = list(flags)
    for start, end in runs_of([not f for f in closed]):
        if start == 0 or end == len(closed) - 1:
            continue  # an open end is where the deck really stops
        if stations[end] - stations[start] < shortest:
            for i in range(start, end + 1):
                closed[i] = True
    return closed


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


def _footprints(ribbons: Sequence[object], elevated: set[int] | None):
    """Plan-view polygons of the given ribbons, made valid."""
    from shapely.geometry import Polygon as ShapelyPolygon

    polygons = []
    for ribbon in ribbons:
        if elevated is not None and ribbon.id not in elevated:
            continue
        ring = [(p[0], p[1]) for p in ribbon.ring()]
        if len(ring) < 4:
            continue
        polygon = ShapelyPolygon(ring)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if not polygon.is_empty and polygon.area > 1e-6:
            polygons.append(polygon)
    return polygons


def deck_outline(ribbons: Sequence[object], elevated: set[int] | None = None, *,
                 close_gap: float = 0.5, patches: Sequence[object] = ()):
    """The outline of the elevated network, as one geometry.

    ``patches`` are the infilled gaps, and they belong in here: they are deck,
    so nothing about them is an edge. Left out, the outline still has a hole
    where the patch went and the barrier runs all the way round the island
    between two turning lanes — which was there in the render and is exactly
    the thing the infill exists to remove.

    Footprints are dilated before the union because neighbouring lanelets are
    surveyed independently and do not quite meet; without it the sliver between
    two lanes reads as open air and both of them get a parapet.
    """
    from shapely.ops import unary_union

    polygons = [p.buffer(close_gap, join_style=2) for p in _footprints(ribbons, elevated)]
    polygons += [p.buffer(close_gap, join_style=2) for p in patches]
    if not polygons:
        return None
    return unary_union(polygons).boundary


def outer_flags(ribbon, outline, tolerance: float) -> tuple[list[bool], list[bool]]:
    """Per cross-section, whether each boundary lies on the edge of the structure.

    Asked of the outline rather than by probing sideways for a neighbour. The
    question is the same one — does the deck stop here — but the outline answers
    it without a distance to tune: a boundary in the middle of a carriageway is
    far from it however the lanes are split, and a boundary on the edge is on
    it however close the next structure happens to pass.
    """
    left, right = list(ribbon.left), list(ribbon.right)
    n = min(len(left), len(right))
    if outline is None or n == 0:
        return [True] * n, [True] * n

    import shapely

    flags = []
    for line in (left[:n], right[:n]):
        points = shapely.points([(p[0], p[1]) for p in line])
        flags.append(list(shapely.distance(outline, points) < tolerance))
    return flags[0], flags[1]


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

    stations = [0.0]
    for a, b in pairwise(centreline(ribbon)):
        stations.append(stations[-1] + math.dist(a[:2], b[:2]))

    walls = []
    for line, flags, sign in ((list(ribbon.left), left_outer, 1.0),
                              (list(ribbon.right), right_outer, -1.0)):
        for start, end in runs_of(close_gaps(flags, stations, options.parapet_bridge_gap)):
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


def _deck_samples(ribbons: Sequence[object]) -> list[Point]:
    return [p for ribbon in ribbons for p in list(ribbon.left) + list(ribbon.right)]


def infill_polygons(sections: Sequence[object], options: ViaductOptions):
    """``(patches, covered)``: the gaps to fill, and what the lanelets cover.

    The patch is the difference between the network's footprint and the same
    footprint with its gaps closed — dilate, erode, subtract — with anything
    larger than ``infill_max_area`` left alone, because a real opening between
    two carriageways is meant to be there.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    polygons = _footprints(sections, None)
    if not polygons:
        return [], None

    covered = unary_union(polygons)
    if not options.infill or options.infill_gap <= 0:
        return [], covered

    gap = options.infill_gap
    closed = covered.buffer(gap, join_style=2).buffer(-gap, join_style=2)
    # Explicitly close the interiors too: a slot ringed by lanelets survives the
    # erosion, because dilation cannot reach into a hole from outside it.
    without_holes = []
    for polygon in getattr(closed, "geoms", [closed]):
        if polygon.geom_type != "Polygon":
            continue
        keep = [ring for ring in polygon.interiors
                if ShapelyPolygon(ring).area >= options.infill_max_area]
        without_holes.append(ShapelyPolygon(polygon.exterior, keep))
    if not without_holes:
        return [], covered

    difference = unary_union(without_holes).difference(covered)
    patches = [
        patch for patch in getattr(difference, "geoms", [difference])
        if patch.geom_type == "Polygon" and 0.02 < patch.area < options.infill_max_area
    ]
    return patches, covered


def infill_meshes(shapes: Sequence[object], patches: Sequence[object],
                  *, reach: float = 1.0) -> list[Mesh]:
    """Patches as meshes, each standing at the height of the lanes around it.

    The height comes from the shapes that touch the patch rather than from the
    network as a whole: a viaduct passes directly over a street, so the nearest
    surveyed vertex in plan view can be seven metres above the gap it is being
    asked about.
    """
    from shapely.strtree import STRtree

    if not patches:
        return []

    footprints = _footprints(shapes, None)
    usable = [shape for shape in shapes if _footprints([shape], None)]
    tree = STRtree(footprints)

    meshes = []
    for patch in patches:
        neighbours = [usable[i] for i in tree.query(patch.buffer(reach))]
        samples = _deck_samples(neighbours or usable)
        if not samples:
            continue
        mesh = triangulate_polygon(patch, height_lookup(samples))
        if mesh.faces:
            meshes.append(mesh)
    return meshes


def build(ribbons: Sequence[object], elevated: set[int], heightmap,
          options: ViaductOptions | None = None) -> dict[str, list[Mesh]]:
    """Deck, parapets and piers for the elevated parts of a road network.

    Needs the heightmap: with no terrain there is no clearance, and with no
    clearance there is no telling a viaduct from a road lying on the ground.
    """
    options = options or ViaductOptions()
    empty = {"ViaductDecks": [], "ViaductParapets": [], "ViaductPiers": []}
    if heightmap is None:
        return empty

    # Which stretches are a bridge is decided first, and everything else is
    # decided against those stretches rather than against whole lanelets: a ramp
    # lying on the ground is not deck, so nothing may treat it as a neighbour.
    sections: list[tuple[object, bool, bool]] = []
    for ribbon in ribbons:
        if ribbon.id not in elevated:
            continue
        vertices = min(len(ribbon.left), len(ribbon.right))
        for start, end in bridge_runs(ribbon, heightmap, options):
            # An end stopping mid-ribbon is where the deck meets grade: that is
            # an abutment and wants a face. An end at the lanelet's own end is a
            # joint with the next lanelet, and capping it walls the beam.
            sections.append((slice_ribbon(ribbon, start, end),
                             start > 0, end < vertices - 1))
    if not sections:
        return empty

    only = [section for section, _s, _e in sections]
    patches, _covered = infill_polygons(only, options)
    outline = deck_outline(only, close_gap=options.edge_buffer, patches=patches)

    decks: list[Mesh] = []
    walls: list[Mesh] = []
    piers: list[Mesh] = []
    for section, cap_start, cap_end in sections:
        left_outer, right_outer = outer_flags(section, outline, options.edge_tolerance)
        if options.deck:
            shell = deck_shell(section, options.deck_thickness,
                               cap_start=cap_start, cap_end=cap_end,
                               left_outer=left_outer, right_outer=right_outer)
            if shell.faces:
                decks.append(shell)
        walls.extend(parapet_walls(section, left_outer, right_outer, options))
        piers.extend(pier_boxes(section, heightmap, options))

    return {"ViaductDecks": decks, "ViaductParapets": walls, "ViaductPiers": piers}
