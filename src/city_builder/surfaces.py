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
    # Off. A walkway lanelet marks where people may walk, and drawing it makes
    # a slab that is not a road: it is lifted 3 cm clear of the carriageway so
    # it does not z-fight, its boundaries are generous at a junction mouth so
    # it laps onto the lane, and what a vehicle driving the scene then meets is
    # a step across its path. The scene this package builds is one to drive
    # through; a pedestrian surface that behaves like a kerb in the road is
    # worse than no pedestrian surface. Turn it on for a scene meant to be
    # looked at rather than driven — `clip_walkways` takes the lapped part off
    # either way.
    walkways: bool = False
    markings: bool = True
    stop_lines: bool = True
    crosswalk_stripes: bool = True
    curbs: bool = True

    # Drawn across a crossing lanelet when the map carries no
    # `pedestrian_marking` rings of its own. Japanese practice is a 45 cm bar
    # with a gap about the same, which is what these default to.
    zebra_bar_width: float = 0.45
    zebra_bar_gap: float = 0.50


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
    extensions=None,
    converter=None,
) -> dict[str, list]:
    """Read a loaded map into named groups of ribbons and polygons.

    ``extensions`` is a :class:`city_builder.extend.Plan`. It is applied here,
    to the boundary polylines, rather than to the finished surfaces: a lane
    bound, the line painted along it and the kerb beside it are one linestring
    as far as the map is concerned, so lengthening the polyline once gives all
    three the same continuation, and the widening, dashing and pairing below
    carry on as if the survey had gone that bit further.
    """
    options = options or SurfaceOptions()
    convert = converter or ll.PointConverter(ll2, projector, frame)
    road_subtypes = tuple(road_subtypes)

    def polyline(primitive) -> list:
        """The primitive's points, continued past the edge of the survey."""
        points = convert.polyline(primitive)
        if extensions is None or len(primitive) < 2:
            return points
        return extensions.extended(points, primitive[0].id, primitive[-1].id)

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

        # Only a road's bounds are continued: a crossing or a footway that
        # happens to end on the same point is not the thing running off the map.
        read = polyline if subtype in road_subtypes else convert.polyline
        paired = pair_bounds(
            read(lanelet.leftBound),
            read(lanelet.rightBound),
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
            standing = extrude_curb(polyline(linestring), options.curb_height)
            if standing:
                groups["Curbs"].append(Ribbon(linestring.id, standing[0], standing[1], attrs))
            continue

        if not options.markings or ls_type not in MARKING_TYPES:
            continue

        width = options.thick_marking_width if ls_type == "line_thick" else options.marking_width
        points = polyline(linestring)
        pieces = (
            split_dashes(points, options.dash_length, options.dash_gap)
            if attrs.get("subtype") == "dashed"
            else [np.asarray(points, dtype=float)]
        )
        for piece in pieces:
            widened = ribbon_from_polyline(piece, width)
            if widened:
                groups["LaneMarkings"].append(Ribbon(linestring.id, widened[0], widened[1], attrs))

    if options.crosswalk_stripes and groups["Crosswalks"] and not groups["CrosswalkStripes"]:
        # The map says where people cross and not what it looks like. Autoware
        # maps often carry the crossing as a lanelet and no `pedestrian_marking`
        # rings at all — measured on the Kashiwanoha map, four crossings and
        # zero bars — and the baking pass then removes the crossing surface on
        # the grounds that its paint is in the road texture, where there is
        # none. The crossing disappears from a scene whose own map inspector
        # reported it.
        #
        # So the bars are drawn across the crossing instead. It is the same
        # claim the rest of this package makes about buildings: the map does
        # not say, so what is generated is the plainest thing consistent with
        # what it does say.
        for crossing in groups["Crosswalks"]:
            groups["CrosswalkStripes"].extend(zebra_bars(crossing, options))

    for name, shapes in groups.items():
        _bias(shapes, Z_BIAS[name], options.z_fight_bias if name in JITTERED else 0.0)

    return {name: shapes for name, shapes in groups.items() if shapes}


def zebra_bars(crossing, options: SurfaceOptions | None = None) -> list:
    """Bars across a crossing lanelet, for a map that has none of its own.

    A crossing lanelet is a ribbon whose two boundaries run along the kerbs,
    so the direction people walk is *across* it: the bars run from one boundary
    to the other, spaced along its length. Which is why this can be built at
    all — the lanelet already carries the orientation the paint needs.
    """
    import numpy as np

    options = options or SurfaceOptions()
    left = np.asarray(crossing.left, dtype=float)
    right = np.asarray(crossing.right, dtype=float)
    if len(left) < 2 or len(left) != len(right):
        return []

    middle = (left + right) / 2.0
    run = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(middle[:, :2], axis=0),
                                                          axis=1))])
    length = float(run[-1])
    if length < options.zebra_bar_width * 2:
        return []

    step = options.zebra_bar_width + options.zebra_bar_gap
    # Centred, so the crossing has paint at both ends rather than a bar flush
    # against one kerb and a gap at the other.
    count = max(1, int((length + options.zebra_bar_gap) // step))
    margin = (length - (count * step - options.zebra_bar_gap)) / 2.0

    def edge(distance: float):
        at = float(np.clip(distance, 0.0, length))
        index = int(np.searchsorted(run, at, side="right") - 1)
        index = min(max(index, 0), len(run) - 2)
        span = run[index + 1] - run[index]
        t = 0.0 if span <= 0 else (at - run[index]) / span
        return (left[index] + (left[index + 1] - left[index]) * t,
                right[index] + (right[index + 1] - right[index]) * t)

    bars = []
    for i in range(count):
        start = margin + i * step
        (l0, r0), (l1, r1) = edge(start), edge(start + options.zebra_bar_width)
        ring = [tuple(l0), tuple(r0), tuple(r1), tuple(l1)]
        bars.append(Polygon(crossing.id * 1000 + i, ring, dict(crossing.attributes)))
    return bars
