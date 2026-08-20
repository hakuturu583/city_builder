"""Put pavements into a map that has none, by editing the map.

Every Lanelet2 map to hand here carries carriageway and nothing else: no
`walkway` lanelets, no `road_border` lines, no kerbs. A street built from one is
therefore tarmac from wall to wall, and that is not a cosmetic gap. Measured
with a segmentation-conditioned generator, a twenty-metre road with buildings on
both sides and no footway is read as a pedestrian square, and the model paves
the middle in pale slabs and puts the asphalt at the sides.

So the pavement is added to the map rather than to the build. The map is where
the rest of the geometry comes from, `surfaces.py` already knows what to do with
a `walkway` lanelet and a `road_border` line, and a map with pavements in it is
useful to everything downstream rather than to one renderer.

Which bounds get one is decided geometrically. Sharing does not work — in these
maps every lanelet carries its own copy of a shared edge, so no bound is used
twice — and a lanelet does not record which of its sides faces the world. What
does work is asking: step off this edge, and are you still on the carriageway?
A bound between two lanes says yes on both sides and is left alone; a bound at
the edge of the road says no on one, and that is the side the pavement goes.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FootwayOptions:
    """How wide a pavement, and how far off the carriageway it starts."""

    #: Metres from the lane edge to the face of the kerb.
    offset: float = 0.3
    #: Metres of walking surface beyond that.
    width: float = 2.0
    #: How far to step off a bound when asking which side faces the world.
    probe: float = 0.6
    #: Metres between probes. A bound is often two points long, and asking at
    #: its vertices gives an answer out of two samples that can only be nought,
    #: a half or one — which decides nothing.
    spacing: float = 0.5
    #: A bound is exterior when this fraction of its probes land off the road.
    clear: float = 0.7
    #: A stretch of pavement is cut where this much of it sits on carriageway.
    #: The bounds of a road running into a junction face the world along their
    #: length and then cross the mouth of it, which is precisely where a
    #: pedestrian is expected to be on the road rather than beside it. Refusing
    #: the whole bound would take the pavement off the entire street for the
    #: sake of four metres, so it is cut instead and picks up on the far side.
    overlap: float = 0.35
    #: Metres below which a surviving stretch is not worth laying.
    shortest: float = 2.0


def _offset(points: np.ndarray, distance: float) -> np.ndarray:
    """The polyline moved sideways, one normal per vertex.

    Not shapely's parallel offset: that returns a simplified curve with its own
    vertex count and sometimes its own direction, and a Lanelet2 bound has to
    stay paired with the one it was made from.
    """
    step = np.gradient(points, axis=0)
    length = np.linalg.norm(step, axis=1, keepdims=True)
    step = step / np.maximum(length, 1e-9)
    normal = np.column_stack([-step[:, 1], step[:, 0]])
    return points + normal * distance


def _walk(points: np.ndarray, spacing: float) -> np.ndarray:
    """The polyline resampled at roughly even spacing, ends included."""
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    along = np.concatenate([[0.0], np.cumsum(step)])
    if along[-1] < 1e-9:
        return points
    wanted = np.linspace(0.0, along[-1], max(int(along[-1] / spacing) + 1, 2))
    return np.column_stack([np.interp(wanted, along, points[:, axis])
                            for axis in (0, 1)])


def _side_facing_out(points, carriageway, options: FootwayOptions):
    """Which way is away from the road: +1, -1, or 0 when neither side is."""
    from shapely.geometry import Point

    walked = _walk(points, options.spacing)
    scores = {}
    for sign in (1.0, -1.0):
        probes = _offset(walked, sign * options.probe)
        outside = sum(not carriageway.contains(Point(x, y)) for x, y in probes)
        scores[sign] = outside / max(len(probes), 1)
    best = max(scores, key=scores.get)
    if scores[best] < options.clear:
        return 0.0
    # Both sides clear means a lone bound with no lanelet either side of it,
    # which is not an edge of anything and is left alone.
    other = -best
    if scores[other] >= options.clear:
        return 0.0
    return best


def plan(bounds, carriageway, options: FootwayOptions | None = None):
    """``(inner, outer)`` pairs for every stretch of bound that faces the world.

    ``bounds`` is ``{way id: polyline}`` in metres; ``carriageway`` is the union
    of the lanelet surfaces as a shapely geometry. One bound can yield more than
    one stretch, because a pavement stops at a junction mouth and starts again
    after it.
    """
    options = options or FootwayOptions()
    out = {}
    for key, points in bounds.items():
        if len(points) < 2:
            continue
        sign = _side_facing_out(points, carriageway, options)
        if sign == 0.0:
            continue

        walked = _walk(points, options.spacing)
        inner = _offset(walked, sign * options.offset)
        outer = _offset(walked, sign * (options.offset + options.width))

        clear = [_on_the_road(inner[i:i + 2], outer[i:i + 2], carriageway)
                 <= options.overlap for i in range(len(walked) - 1)]
        for piece, (start, stop) in enumerate(_runs(clear)):
            span = np.linalg.norm(
                np.diff(walked[start:stop + 1], axis=0), axis=1).sum()
            if span < options.shortest:
                continue
            out[(key, piece)] = (inner[start:stop + 1], outer[start:stop + 1])
    return out


def _runs(flags):
    """Start and stop vertex indices of each contiguous run of True."""
    spans, first = [], None
    for index, flag in enumerate(flags):
        if flag and first is None:
            first = index
        elif not flag and first is not None:
            spans.append((first, index))
            first = None
    if first is not None:
        spans.append((first, len(flags)))
    return spans


def _on_the_road(inner, outer, carriageway) -> float:
    """How much of a proposed pavement lies on carriageway."""
    from shapely.geometry import Polygon as Shape

    ring = np.vstack([inner, outer[::-1]])
    if len(ring) < 3:
        return 0.0
    shape = Shape(ring).buffer(0)
    if shape.is_empty or shape.area <= 0:
        return 0.0
    return float(shape.intersection(carriageway).area / shape.area)


def write(path: str, source: str, paved, reverse, *, start_id: int = 900000) -> dict:
    """Write a copy of the map with the pavements in it.

    New identifiers start well above anything a hand-drawn map is likely to use,
    and nothing existing is touched, so the result diffs cleanly against what it
    came from.
    """
    tree = ET.parse(source)
    root = tree.getroot()
    taken = {int(e.get("id")) for e in root.iter() if e.get("id", "").isdigit()}
    counter = max([start_id, *(i + 1 for i in taken if i >= start_id)])

    def fresh():
        nonlocal counter
        counter += 1
        return str(counter)

    def node(x, y):
        point = reverse(x, y)
        element = ET.SubElement(root, "node", id=fresh(), visible="true", version="1",
                                lat=f"{point[0]:.11f}", lon=f"{point[1]:.11f}")
        return element.get("id")

    def way(points, kind):
        element = ET.SubElement(root, "way", id=fresh(), visible="true", version="1")
        for x, y in points:
            ET.SubElement(element, "nd", ref=node(x, y))
        ET.SubElement(element, "tag", k="type", v=kind)
        return element.get("id")

    made = {"walkway": 0, "road_border": 0}
    for inner, outer in paved.values():
        kerb = way(inner, "road_border")
        # `virtual`, not `line_thin`: the far side of a pavement is where the
        # surface stops, not a line painted on the ground, and calling it paint
        # puts a white stripe along every kerb.
        edge = way(outer, "virtual")
        relation = ET.SubElement(root, "relation", id=fresh(), visible="true",
                                 version="1")
        ET.SubElement(relation, "member", type="way", ref=kerb, role="left")
        ET.SubElement(relation, "member", type="way", ref=edge, role="right")
        ET.SubElement(relation, "tag", k="location", v="urban")
        ET.SubElement(relation, "tag", k="subtype", v="walkway")
        ET.SubElement(relation, "tag", k="type", v="lanelet")
        made["walkway"] += 1
        made["road_border"] += 1

    ET.indent(tree, space="  ")
    tree.write(path, encoding="utf-8", xml_declaration=True)
    return made
