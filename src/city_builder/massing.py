"""A building with something going on, inside the plot it is given.

The procedural block is one extruded footprint, which is honest — nothing in
the map says what stands there — and, put in front of a reconstruction model,
useless: it faithfully returns a box, and a street of boxes is what we already
had. Asking an image model to add the variety does not work either. Measured on
one massing at four strengths, the picture came back the same box at 0.35 and
0.50, grew a parapet at 0.65, and fell apart at 0.80; the model re-imagines
*surfaces* from a starting latent, not massing.

So the variety is built, and only the surfaces are asked for. Everything here
is a plan operation on the plot, which is what makes the guarantee cheap: a
courtyard is a hole, a wing is a piece of the plot standing lower, a wall is a
ring inside the boundary. None of them can leave the footprint, because none of
them is bigger than the polygon they are cut from.

Four things it does, each drawn or not per building:

**A courtyard.** An inner ring, buffered in from the plot and offset off
centre. :func:`city_builder.buildings.extrude` already walls interior rings the
other way round, so this costs nothing beyond choosing where the hole goes.

**A set-back wing.** The plot cut in two along its short axis; the smaller part
stands lower, which is what a shop with an office over half of it looks like.

**A perimeter wall.** A low ring just inside the boundary, around whatever the
building does not cover — the forecourt wall that most of these plots would
have in a Japanese street.

**A stepped parapet.** A thin band round the roof, so the roofline is not the
top of an extrusion.

Nothing here imports bpy.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any

from shapely.errors import GEOSException

from .geometry import Mesh


@dataclass
class MassingOptions:
    """How much variety, and how much of it at once."""

    # Each feature is drawn independently. Everything at once on every plot is
    # as monotonous as nothing at all — the point is that the street differs
    # from itself, not that every building is busy.
    courtyard: float = 0.35  # chance a plot gets one
    wing: float = 0.45  # chance the plot is split and half stands lower
    wall: float = 0.55  # chance of a forecourt wall
    parapet: float = 0.7  # chance of a band round the roof

    courtyard_share: float = 0.16  # of the plot's area, at most
    wing_share: float = 0.38  # of the plot's long side
    wing_drop: float = 0.45  # how much lower the wing is, as a share of height
    wall_height: float = 2.1  # metres
    wall_thickness: float = 0.35
    setback: float = 1.4  # gap between the wall and the plot boundary
    forecourt_share: float = 0.22  # of the plot's long side, taken as yard
    parapet_height: float = 0.9
    parapet_thickness: float = 0.4

    # How the top is finished. A parapet only makes sense on a flat roof, so
    # the two are chosen together: draw a pitch and the parapet is skipped.
    roof_forms: tuple[str, ...] = ("flat", "flat", "gable", "hip", "mono")
    roof_pitch: tuple[float, float] = (0.35, 0.65)  # rise over half the span
    roof_eave: float = 0.7

    def __post_init__(self) -> None:
        for name in ("courtyard", "wing", "wall", "parapet"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"massing.{name} is a probability, in [0, 1]")
        if not 0.0 < self.courtyard_share < 0.5:
            raise ValueError("massing.courtyard_share must leave a building around it")


ROOF_FORMS = ("flat", "gable", "hip", "mono")


def roof(polygon, top_z: float, form: str, *, pitch: float = 0.45,
         eave: float = 0.7) -> Mesh:
    """A pitched roof over a footprint, as a mesh. ``flat`` returns nothing.

    Built on the footprint's *minimum rotated rectangle* rather than on the
    outline itself. A roof over an arbitrary polygon is a straight-skeleton
    problem and this is not one: real roofs are simple forms carried across a
    plan, they overhang their walls anyway, and a hip that misses an inside
    corner by half a metre is a hip.

    Which is also why a flat roof is what an extruded footprint gives and why
    nothing before this had another one — a street with no pitched roof in it
    is not a Japanese street, and the reconstruction can only return the shapes
    it is shown.
    """
    if form not in ROOF_FORMS:
        raise ValueError(f"roof form must be one of {ROOF_FORMS}, not {form!r}")
    if form == "flat":
        return Mesh([], [])

    corners = list(polygon.minimum_rotated_rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        return Mesh([], [])
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    (ax, ay), (bx, by) = max(edges, key=lambda e: math.dist(e[0], e[1]))
    length = math.dist((ax, ay), (bx, by))
    if length < 1e-6:
        return Mesh([], [])
    ux, uy = (bx - ax) / length, (by - ay) / length  # along the ridge
    vx, vy = -uy, ux  # across it
    width = polygon.minimum_rotated_rectangle.area / length
    cx, cy = polygon.minimum_rotated_rectangle.centroid.coords[0]

    half_l, half_w = length / 2.0 + eave, width / 2.0 + eave

    def at(u: float, v: float, z: float) -> tuple[float, float, float]:
        return (cx + ux * u + vx * v, cy + uy * u + vy * v, z)

    rise = pitch * half_w
    a, b = at(-half_l, -half_w, top_z), at(half_l, -half_w, top_z)
    c, d = at(half_l, half_w, top_z), at(-half_l, half_w, top_z)

    if form == "mono":
        # One slope, the whole width. The low eave is the street side.
        high_c, high_d = at(half_l, half_w, top_z + 2 * rise), at(-half_l, half_w,
                                                                 top_z + 2 * rise)
        vertices = [a, b, high_c, high_d, c, d]
        faces = [[0, 1, 2, 3], [1, 4, 2], [5, 0, 3]]
        return Mesh(vertices, faces)

    inset = half_w if form == "hip" else 0.0
    if inset >= half_l:  # too short to hip: the ridge would be a point
        inset = half_l * 0.5
    r0, r1 = at(-half_l + inset, 0.0, top_z + rise), at(half_l - inset, 0.0, top_z + rise)

    vertices = [a, b, c, d, r0, r1]
    faces = [
        [0, 1, 5, 4],  # the slope on one side
        [2, 3, 4, 5],  # and on the other
        [0, 4, 3],  # the end: a gable wall, or a hip slope
        [1, 2, 5],
    ]
    return Mesh(vertices, faces)


def _polygon(ring):
    from shapely.geometry import Polygon as ShapelyPolygon

    plot = ShapelyPolygon([(float(x), float(y)) for x, y in ring])
    return plot if plot.is_valid else plot.buffer(0)


def _long_axis_deg(polygon) -> float:
    """Which way the long side runs, so cuts follow the plot rather than north."""
    corners = list(polygon.minimum_rotated_rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        return 0.0
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    (ax, ay), (bx, by) = max(edges, key=lambda e: math.dist(e[0], e[1]))
    return math.degrees(math.atan2(by - ay, bx - ax))


def _largest(shape):
    if shape.is_empty:
        return shape
    return max(shape.geoms, key=lambda g: g.area) if hasattr(shape, "geoms") else shape


def plan(plot: dict[str, Any], options: MassingOptions | None = None,
         seed: int = 0) -> dict[str, Any]:
    """What this building does, as plan geometry and heights.

    Returns the pieces rather than a mesh, so the choices can be inspected and
    tested without Blender: ``{"parts": [(polygon, base, top)], "features": [...]}``.
    """
    options = options or MassingOptions()
    rng = random.Random(seed)

    outline = _polygon(plot["footprint"])
    if outline.is_empty or outline.area <= 0:
        raise ValueError("this plot has no area to build on")
    height = float(plot["height"])
    base = float(plot["base_z"])

    features: list[str] = []
    footprint = outline
    parts: list[tuple[Any, float, float]] = []

    # A forecourt, taken off the street end of the plot before anything is
    # built. It has to come first: the plot already has the coverage ratio in
    # it — `inset_to_coverage` saw to that — so the building fills it, and
    # there is no yard for a wall to go round until one is made.
    forecourt = None
    if rng.random() < options.wall:
        from shapely.geometry import LineString
        from shapely.ops import split as shapely_split

        rectangle = outline.minimum_rotated_rectangle
        corners = list(rectangle.exterior.coords)[:4]
        edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        (ax, ay), (bx, by) = max(edges, key=lambda e: math.dist(e[0], e[1]))
        span = math.dist((ax, ay), (bx, by))
        ux, uy = (bx - ax) / span, (by - ay) / span
        depth = span * options.forecourt_share * rng.uniform(0.7, 1.3)
        # Off one of the two short ends, so the yard faces a street rather than
        # sitting in the middle of the block.
        if rng.random() < 0.5:
            cx, cy = ax + ux * depth, ay + uy * depth
        else:
            cx, cy = bx - ux * depth, by - uy * depth
        reach = span + outline.length
        knife = LineString([(cx + uy * reach, cy - ux * reach),
                            (cx - uy * reach, cy + ux * reach)])
        try:
            pieces = [p for p in shapely_split(outline, knife).geoms if p.area > 1.0]
        except (GEOSException, ValueError):  # pragma: no cover - degenerate cut
            pieces = []
        if len(pieces) == 2:
            pieces.sort(key=lambda p: -p.area)
            footprint, forecourt = pieces
            features.append("forecourt")

    # A wing: the plot split across its long axis, the smaller part lower.
    if rng.random() < options.wing:
        from shapely.geometry import LineString
        from shapely.ops import split as shapely_split

        rectangle = footprint.minimum_rotated_rectangle
        corners = list(rectangle.exterior.coords)[:4]
        edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
        (ax, ay), (bx, by) = max(edges, key=lambda e: math.dist(e[0], e[1]))
        length = math.dist((ax, ay), (bx, by))
        ux, uy = (bx - ax) / length, (by - ay) / length
        at = length * (1.0 - options.wing_share)
        cx, cy = ax + ux * at, ay + uy * at
        reach = length + footprint.length
        knife = LineString([(cx + uy * reach, cy - ux * reach),
                            (cx - uy * reach, cy + ux * reach)])
        try:
            pieces = [p for p in shapely_split(footprint, knife).geoms if p.area > 1.0]
        except (GEOSException, ValueError):  # pragma: no cover - degenerate cut
            pieces = []
        if len(pieces) == 2:
            pieces.sort(key=lambda p: -p.area)
            footprint, wing = pieces
            parts.append((wing, base, base + height * (1.0 - options.wing_drop)))
            features.append("wing")

    # A courtyard: a hole in whatever the main volume covers.
    if rng.random() < options.courtyard:
        target = footprint.area * options.courtyard_share
        inset = math.sqrt(footprint.area) * 0.18
        inner = _largest(footprint.buffer(-inset))
        if not inner.is_empty and inner.area > target * 0.5:
            radius = math.sqrt(target / math.pi)
            spot = inner.representative_point()
            # Off centre, but still inside what the walls can carry.
            wander = _largest(inner.buffer(-radius))
            if not wander.is_empty:
                bounds = wander.bounds
                for _ in range(12):
                    from shapely.geometry import Point

                    candidate = Point(rng.uniform(bounds[0], bounds[2]),
                                      rng.uniform(bounds[1], bounds[3]))
                    if wander.contains(candidate):
                        spot = candidate
                        break
            # Rectangular and turned with the plot, not a circle: a light well
            # is cut between walls that run parallel to the boundary, and a
            # round one reads as a hole punched by a machine.
            from shapely.affinity import rotate as rotate_shape
            from shapely.geometry import box as shapely_box

            angle = _long_axis_deg(outline)
            side = radius * math.sqrt(math.pi)
            stretch = rng.uniform(1.0, 1.9)
            hole = rotate_shape(
                shapely_box(spot.x - side * stretch / 2, spot.y - side / stretch / 2,
                            spot.x + side * stretch / 2, spot.y + side / stretch / 2),
                angle, origin=(spot.x, spot.y))
            pierced = footprint.difference(hole)
            if not pierced.is_empty and pierced.area > footprint.area * 0.4:
                footprint = _largest(pierced)
                features.append("courtyard")

    parts.insert(0, (footprint, base, base + height))

    # How the top is finished, and then a parapet only if it is flat.
    form = rng.choice(options.roof_forms)
    pitch = rng.uniform(*options.roof_pitch)
    if form != "flat":
        features.append(form)

    # A parapet: a band standing on the roof edge of the main volume.
    if form == "flat" and rng.random() < options.parapet:
        band = footprint.difference(footprint.buffer(-options.parapet_thickness))
        if not band.is_empty and band.area > 0.5:
            top = base + height
            parts.append((band, top, top + options.parapet_height))
            features.append("parapet")

    # A wall round the part of the plot the building does not stand on. Cut
    # against everything already built, so it stops at the walls rather than
    # running through them.
    if forecourt is not None:
        yard = _largest(forecourt.buffer(-options.setback * 0.3))
        if not yard.is_empty:
            ring = yard.difference(yard.buffer(-options.wall_thickness))
            for polygon, _bottom, _top in parts:
                ring = ring.difference(polygon)
            standing = [piece for piece in (ring.geoms if hasattr(ring, "geoms") else [ring])
                        if piece.area > 0.4]
            if standing:
                parts.extend((piece, base, base + options.wall_height) for piece in standing)
                features.append("wall")

    return {"parts": parts, "features": features,
            "roof": {"form": form, "pitch": round(pitch, 3), "eave": options.roof_eave},
            "plot_area": round(outline.area, 2),
            "covered_area": round(sum(p[0].area for p in parts), 2)}


def build(plot: dict[str, Any], options: MassingOptions | None = None,
          seed: int = 0, *, facade_width: float = 12.0) -> dict[str, Any]:
    """The varied massing as walls and roofs, ready to go into a scene."""
    from .buildings import extrude

    laid_out = plan(plot, options, seed)
    walls: list[Mesh] = []
    roofs: list[Mesh] = []
    pitched = laid_out["roof"]["form"] != "flat"
    for index, (polygon, bottom, top) in enumerate(laid_out["parts"]):
        for piece in (polygon.geoms if hasattr(polygon, "geoms") else [polygon]):
            if piece.is_empty or piece.area <= 1e-6:
                continue
            wall, cap = extrude(piece, bottom, max(top - bottom, 0.1),
                                skirt=0.4, facade_width=facade_width)
            if wall.faces:
                walls.append(wall)
            # The flat cap is dropped where a pitch goes over it: two roofs in
            # the same place is a surface the reconstruction has to choose
            # between, and it chooses the wrong one about half the time.
            if cap.faces and not (pitched and index == 0):
                roofs.append(cap)

    if pitched:
        main = laid_out["parts"][0][0]
        top = laid_out["parts"][0][2]
        pitch = roof(main, top, laid_out["roof"]["form"],
                            pitch=laid_out["roof"]["pitch"], eave=laid_out["roof"]["eave"])
        if pitch.faces:
            roofs.append(pitch)
    return {"Buildings": walls, "Roofs": roofs, **laid_out}


def within_plot(plot: dict[str, Any], laid_out: dict[str, Any],
                tolerance: float = 0.05) -> bool:
    """Does everything this building does fit inside the plot it was given?

    The guarantee the whole module rests on, checkable rather than argued: every
    piece is derived from the plot polygon by cutting, so none of them can leave
    it, and this is the assertion that says so.
    """
    outline = _polygon(plot["footprint"]).buffer(tolerance)
    return all(outline.contains(polygon) for polygon, _bottom, _top in laid_out["parts"])
