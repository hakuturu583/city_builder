"""The railings that stand where the ground stops being walkable.

A pond in a residential street has a fence round it, and a wall you can fall
off has one on top. Neither is decoration: they are the reason a person can
stand next to the thing at all, and a scene without them reads as a model
rather than as a place. They belong to the *terrain* stage — the ground already
knows where its shoreline and its retaining walls are, because both are
breaklines in the mesh — so this takes those same lines and stands something on
them.

**What gets one is measured, not assumed.** Standing water always: there is no
threshold below which an open pond in a street is left unfenced. A drop needs a
height, and the one used here is the one Japanese practice uses — a fall of
about a metre is where a parapet or a railing becomes required, and below it a
kerb or a change of surface is the whole treatment. Measured on the Kashiwanoha
map, no plot edge reaches that: the platforms are 0.3 m and the biggest drop
just outside one is 0.45 m, so the terrace rule fires nowhere there. That is
the rule working, not the rule being useless — it is what stops a suburban
street growing a metre of railing round every front garden.

**Not on the carriageway.** A road running along a pond gets a guardrail, which
is a different object with a different profile and belongs to the road layer;
what this makes is the pedestrian railing on the ground beside it. So any part
of an edge that lies under the carriageway is cut out rather than fenced.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from .geometry import Mesh

#: A fall of about this much is where a railing stops being optional.
DEFAULT_MIN_DROP = 1.0


@dataclass
class FenceOptions:
    """A railing: how tall, how often, and how heavy."""

    height: float = 1.1          # m; the standard fall-protection height
    post_spacing: float = 1.8    # m between posts
    post_size: float = 0.06      # m square
    rails: int = 3               # horizontal members, evenly spread
    rail_size: float = 0.04      # m square
    setback: float = 0.25        # m back from the edge, so it stands on ground
    foot: float = 0.15           # m the post is sunk, so it is not floating
    min_drop: float = DEFAULT_MIN_DROP
    min_length: float = 3.0      # m; a stub of railing is worse than none


def _box(mesh_vertices: list, mesh_faces: list, centre, along, size, length,
         low: float, high: float) -> None:
    """A rectangular bar from ``low`` to ``high``, its long axis along ``along``."""
    ux, uy = along
    px, py = -uy, ux
    half = size / 2.0
    base = len(mesh_vertices)
    for z in (low, high):
        for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
            mesh_vertices.append((centre[0] + ux * sx * length / 2 + px * sy * half,
                                  centre[1] + uy * sx * length / 2 + py * sy * half,
                                  z))
    a, b, c, d, e, f, g, h = range(base, base + 8)
    mesh_faces.extend([[a, b, c, d], [h, g, f, e],
                       [a, e, f, b], [b, f, g, c], [c, g, h, d], [d, h, e, a]])


def along(lines: Sequence[Sequence[Sequence[float]]], height_at,
          options: FenceOptions | None = None) -> Mesh:
    """A railing following each line, standing on the ground under it.

    ``lines`` are plan polylines in scene metres and ``height_at(x, y)`` is the
    ground. The posts take the ground where they stand and the rails run
    between them, so a railing on a slope steps with the ground rather than
    floating off it — which is what a single extruded ribbon at one height
    would do, and is the thing that gives a fence away.
    """
    options = options or FenceOptions()
    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []

    for line in lines:
        points = [(float(x), float(y)) for x, y in line]
        if len(points) < 2:
            continue
        spans = [math.dist(a, b) for a, b in pairwise(points)]
        total = sum(spans)
        if total < options.min_length:
            continue

        # Posts at an even spacing along the *whole* line, so a corner does not
        # reset the rhythm and leave two posts a hand's width apart.
        count = max(round(total / options.post_spacing), 1)
        step = total / count
        walked, index, at = 0.0, 0, []
        for k in range(count + 1):
            want = min(k * step, total)
            while index < len(spans) - 1 and walked + spans[index] < want:
                walked += spans[index]
                index += 1
            span = spans[index] or 1.0
            t = (want - walked) / span
            a, b = points[index], points[index + 1]
            at.append(((a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t),
                       ((b[0] - a[0]) / span, (b[1] - a[1]) / span)))

        for (spot, direction) in at:
            ground = height_at(spot[0], spot[1])
            _box(vertices, faces, spot, direction, options.post_size,
                 options.post_size, ground - options.foot, ground + options.height)

        for (start, direction), (end, _next) in pairwise(at):
            length = math.dist(start, end)
            if length < 1e-6:
                continue
            middle = ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2)
            run = ((end[0] - start[0]) / length, (end[1] - start[1]) / length)
            ground = (height_at(*start) + height_at(*end)) / 2.0
            for rail in range(1, options.rails + 1):
                level = ground + options.height * rail / options.rails
                _box(vertices, faces, middle, run, options.rail_size, length,
                     level - options.rail_size / 2, level + options.rail_size / 2)
        del direction

    return Mesh(vertices, faces)


def water_edges(bodies: Sequence[Any], height_at=None, roads=None,
                options: FenceOptions | None = None) -> list[list[tuple[float, float]]]:
    """The bank of every body, less whatever the carriageway covers.

    The *bank*, not the waterline, and the bank is found rather than guessed. A
    pond is dug, so everything between the water's edge and the top of the
    excavation is the inside of a hole — measured on the pond this was built
    against, 6.6 m of it falling 1.1 m, and the levelling reaches half a height
    cell further still. A railing set back a fixed quarter of a metre from the
    water therefore stood 0.60 m *below* the waterline, down in the bowl,
    fencing nothing and half buried.

    So the ring is pushed outward until the ground under it is actually above
    the water. That is the definition of a bank, it needs no constant, and it
    comes out right whatever the height grid is.
    """
    options = options or FenceOptions()
    rims = []
    for body in bodies:
        rim = getattr(body, "painted", None) or getattr(body, "polygon", None)
        if rim is None or rim.is_empty:
            continue
        level = getattr(body, "level", None)
        if height_at is not None and level is not None:
            rim = _walk_to_the_bank(rim, level, height_at, options)
        rims.append(rim)
    return _edges(rims, roads, options)


def _walk_to_the_bank(rim, level: float, height_at, options: FenceOptions,
                      *, clearance: float = 0.1, step: float = 0.5,
                      reach: float = 10.0, clear: float = 0.9):
    """Push a ring outward until ``clear`` of the ground along it is above the water.

    A fraction rather than the minimum, and a fraction rather than the median.
    The minimum lets one notch in the bank drag the whole railing ten metres
    back into somebody's garden; the median stops as soon as half the ring is
    dry, which on the pond this was built against left a quarter of the railing
    still standing in the bowl, 0.60 m under the waterline. Nine tenths is the
    setting that cleared it there.
    """
    import numpy as np

    out = rim.buffer(options.setback)
    for extra in np.arange(0.0, reach + step, step):
        ring = rim.buffer(options.setback + float(extra))
        edge = ring.exterior if ring.geom_type == "Polygon" else None
        if edge is None or edge.length <= 0:
            continue
        heights = [height_at(*edge.interpolate(t, normalized=True).coords[0])
                   for t in np.linspace(0.0, 1.0, 48, endpoint=False)]
        out = ring
        if float(np.mean(np.asarray(heights) >= level + clearance)) >= clear:
            break
    return out


def terrace_edges(terraces: Sequence[tuple[Any, float]], height_at, roads=None,
                  options: FenceOptions | None = None) -> list[list[tuple[float, float]]]:
    """The edges of a platform that are a real drop, and only those.

    Measured a step outside the edge, against the platform's own level, because
    a terrace on flat ground is a kerb and a terrace cut into a hillside is a
    retaining wall, and the same rule has to tell them apart.
    """
    options = options or FenceOptions()
    from shapely.geometry import LineString, Point
    from shapely.prepared import prep

    steep = []
    for outline, level in terraces:
        for part in (outline.geoms if hasattr(outline, "geoms") else [outline]):
            if part.geom_type != "Polygon":
                continue
            inside = prep(part)
            for ring in (part.exterior, *part.interiors):
                coords = list(ring.coords)
                run: list[tuple[float, float]] = []
                for (x0, y0), (x1, y1) in pairwise(coords):
                    # Which way is *out* is decided by asking, not by the sign
                    # of a cross product: a ring's winding depends on where it
                    # came from, and an interior ring winds the other way from
                    # its exterior. Probing the wrong side samples the platform
                    # itself, which never drops, so nothing is ever fenced.
                    nx, ny = -(y1 - y0), (x1 - x0)
                    norm = math.hypot(nx, ny) or 1.0
                    reach = options.setback + 1.0
                    mid = ((x0 + x1) / 2, (y0 + y1) / 2)
                    probe = (mid[0] + nx / norm * reach, mid[1] + ny / norm * reach)
                    if inside.contains(Point(*probe)):
                        probe = (mid[0] - nx / norm * reach, mid[1] - ny / norm * reach)
                    if level - height_at(*probe) >= options.min_drop:
                        if not run:
                            run.append((x0, y0))
                        run.append((x1, y1))
                    elif run:
                        steep.append(LineString(run))
                        run = []
                if run:
                    steep.append(LineString(run))
    return _edges(steep, roads, options, already_lines=True)


def _edges(shapes, roads, options: FenceOptions, *, already_lines: bool = False):
    """Shapes to plan polylines, set back from the edge and off the carriageway."""
    from shapely.ops import unary_union

    if not shapes:
        return []
    lines = []
    for shape in shapes:
        if shape is None or shape.is_empty:
            continue
        # Set back onto the bank rather than sitting on the waterline itself:
        # a railing standing in the water is worse than no railing.
        line = shape if already_lines else shape.buffer(options.setback).boundary
        if roads is not None and not roads.is_empty:
            line = line.difference(roads)
        if line.is_empty:
            continue
        lines.append(line)

    out: list[list[tuple[float, float]]] = []
    for line in unary_union(lines).geoms if hasattr(unary_union(lines), "geoms") \
            else [unary_union(lines)]:
        if line.geom_type != "LineString" or line.length < options.min_length:
            continue
        out.append([(float(x), float(y)) for x, y in line.coords])
    return out


def build(heightmap, *, water: Sequence[Any] = (), terraces: Sequence = (),
          roads=None, keep_clear=None,
          options: FenceOptions | None = None) -> dict[str, Any]:
    """Every railing this scene wants, as one mesh, with a note of why.

    Returned as one mesh rather than one per run of railing: a suburban street
    is a few hundred posts, and a few hundred objects is a few hundred draw
    calls for something nobody selects individually.
    """
    options = options or FenceOptions()
    # The carriageway and the plots are both no-go. A railing across a street
    # is a barrier and a railing through somebody's house is a mistake; the
    # bank a fence stands on has to be the bank a person can stand on.
    blocked = roads
    if keep_clear is not None:
        from shapely.ops import unary_union

        blocked = unary_union([g for g in (roads, keep_clear) if g is not None])
    lines = water_edges(water, heightmap.sample, blocked, options)
    shore = len(lines)
    lines += terrace_edges(terraces, heightmap.sample, blocked, options)
    mesh = along(lines, heightmap.sample, options)
    return {
        "mesh": mesh,
        "runs": len(lines),
        "shoreline_runs": shore,
        "metres": round(sum(sum(math.dist(a, b) for a, b in pairwise(line))
                            for line in lines), 1),
        # One box is eight vertices, and a run of N posts carries N-1 spans of
        # `rails` bars each, so counting boxes would call the rails posts too.
        "posts": sum(max(round(sum(math.dist(a, b) for a, b in pairwise(line))
                               / options.post_spacing), 1) + 1 for line in lines),
    }
