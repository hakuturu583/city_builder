"""Turning a Lanelet2 map into named groups of drawable surfaces.

A Lanelet2 map ships the surveyed left and right boundary of every lane in 3D,
so the road surface is *read* rather than inferred: true width, true curvature,
true elevation, and intersections that are real turning lanelets. Markings,
stop lines and zebra bars are likewise map features, not procedural decoration
— the only thing invented here is where a dashed line breaks, because the map
stores it as one continuous polyline.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from . import lanelet as ll
from .classes import MATERIALS  # noqa: F401  (re-exported; the registry owns it)
from .frame import LocalFrame
from .geometry import (
    Polygon,
    Ribbon,
    close_ring,
    extrude_curb,
    pair_bounds,
    ribbon_from_polyline,
    split_dashes,
)

# Linestrings that are actually painted on the road. `virtual` is a logical
# boundary and must never be drawn; `road_border` is a kerb, not paint.
MARKING_TYPES = ("line_thin", "line_thick")

# Stacking order above the carriageway, in metres. The gaps have to exceed the
# per-lanelet jitter below, or two layers land within ~1 mm of each other: a
# rasteriser z-fights, and a ray tracer has the upper surface shadow the lower,
# which renders as solid black patches.
Z_BIAS = {
    "Roads": 0.0,
    "Junctions": 0.002,
    "Crosswalks": 0.010,
    "LaneMarkings": 0.020,
    "StopLines": 0.022,
    "CrosswalkStripes": 0.026,
    "Walkways": 0.030,
    "Curbs": 0.0,
}

# Only the lane surfaces overlap each other (turning lanelets cross inside an
# intersection), so only they get the de-tie jitter.
JITTERED = ("Roads", "Junctions")




@dataclass
class SurfaceOptions:
    """Everything that changes what gets drawn, and how finely."""

    max_segment: float = 5.0
    marking_width: float = 0.15
    thick_marking_width: float = 0.3
    stop_line_width: float = 0.4
    dash_length: float = 3.0
    dash_gap: float = 5.0
    curb_height: float = 0.15
    crosswalk_lift: float = 0.005  # a crossing sits on the carriageway, not in it
    z_fight_bias: float = 0.0002

    crosswalks: bool = True
    walkways: bool = True
    markings: bool = True
    stop_lines: bool = True
    crosswalk_stripes: bool = True
    curbs: bool = True


def _bias(shapes, amount: float, jitter: float = 0.0) -> None:
    for shape in shapes:
        offset = amount + (shape.id % 8) * jitter
        if offset:
            shape.shift_z(offset)


def extract(
    ll2,
    projector,
    lmap,
    frame: LocalFrame,
    options: SurfaceOptions | None = None,
    *,
    road_subtypes: Sequence[str] = ll.ROAD_SUBTYPES,
) -> dict[str, list]:
    """Read a loaded map into named groups of ribbons and polygons."""
    options = options or SurfaceOptions()
    convert = ll.PointConverter(ll2, projector, frame)
    road_subtypes = tuple(road_subtypes)

    groups: dict[str, list] = {name: [] for name in Z_BIAS}

    for lanelet in lmap.laneletLayer:
        attrs = ll.attributes(lanelet)
        subtype = attrs.get("subtype", "")

        if subtype in road_subtypes:
            target = "Junctions" if "turn_direction" in attrs else "Roads"
        elif subtype == "crosswalk" and options.crosswalks:
            target = "Crosswalks"
        elif subtype == "walkway" and options.walkways:
            target = "Walkways"
        else:
            continue

        paired = pair_bounds(
            convert.polyline(lanelet.leftBound),
            convert.polyline(lanelet.rightBound),
            max_segment=options.max_segment,
        )
        if paired is not None:
            groups[target].append(Ribbon(lanelet.id, paired[0], paired[1], attrs))

    for linestring in lmap.lineStringLayer:
        attrs = ll.attributes(linestring)
        ls_type = attrs.get("type", "")

        if options.stop_lines and ls_type == "stop_line":
            widened = ribbon_from_polyline(convert.polyline(linestring), options.stop_line_width)
            if widened:
                groups["StopLines"].append(Ribbon(linestring.id, widened[0], widened[1], attrs))
            continue

        if options.crosswalk_stripes and ls_type == "pedestrian_marking":
            # One linestring per zebra bar, stored as a closed 4-corner ring.
            ring = close_ring(convert.polyline(linestring))
            if ring:
                groups["CrosswalkStripes"].append(Polygon(linestring.id, ring, attrs))
            continue

        if options.curbs and ls_type == "road_border":
            standing = extrude_curb(convert.polyline(linestring), options.curb_height)
            if standing:
                groups["Curbs"].append(Ribbon(linestring.id, standing[0], standing[1], attrs))
            continue

        if not options.markings or ls_type not in MARKING_TYPES:
            continue

        width = options.thick_marking_width if ls_type == "line_thick" else options.marking_width
        points = convert.polyline(linestring)
        pieces = (
            split_dashes(points, options.dash_length, options.dash_gap)
            if attrs.get("subtype") == "dashed"
            else [np.asarray(points, dtype=float)]
        )
        for piece in pieces:
            widened = ribbon_from_polyline(piece, width)
            if widened:
                groups["LaneMarkings"].append(Ribbon(linestring.id, widened[0], widened[1], attrs))

    for name, shapes in groups.items():
        _bias(shapes, Z_BIAS[name], options.z_fight_bias if name in JITTERED else 0.0)

    return {name: shapes for name, shapes in groups.items() if shapes}
