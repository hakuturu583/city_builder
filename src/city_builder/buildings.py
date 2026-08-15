"""Procedural buildings in the space the roads leave behind.

A Lanelet2 map describes the carriageway and nothing else, so the blocks
between the streets are empty. This fills them: the buildable area is the
ground minus the road outline minus a setback, each block is cut into lots, and
each lot gets an extruded footprint standing on the reconstructed ground.

The layout is deliberately simple — recursive splitting along the long axis,
which is how a block of terraced plots actually reads from the air — and
deterministic given a seed. It is scaffolding for a texturing pass, not an
attempt to guess what is really there: nothing in the map says where the
buildings are, so the only honest claim is "something building-shaped, off the
road, on the ground".
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.errors import GEOSException

from .geometry import Mesh, signed_area_xy
from .ground import HeightMap


@dataclass
class BuildingOptions:
    """Everything that shapes the generated blocks."""

    setback: float = 3.0  # gap between the kerb line and any wall
    target_lot_area: float = 900.0  # split until a lot is about this big
    min_lot_area: float = 120.0  # anything smaller is left as open ground
    split_jitter: float = 0.15  # 0 = split at the middle, 0.5 = anywhere

    # Density. `coverage` is the planner's one: the share of a lot its building
    # occupies, so the gap between neighbours falls out of the lot size rather
    # than being fixed. `lot_margin` is a floor under that gap, and `vacancy`
    # empties lots outright — car parks, yards, the plot nobody built on.
    coverage: float = 0.6
    lot_margin: float = 1.5  # minimum gap between neighbouring buildings
    vacancy: float = 0.0  # 0 = build on every lot, 0.3 = leave a third empty

    min_height: float = 6.0
    max_height: float = 45.0
    floor_height: float = 3.5  # heights snap to whole floors
    tall_bias: float = 0.35  # 0 = every block low, 1 = every block tall

    facade_width: float = 12.0  # how much wall one sheet spans horizontally
    skirt: float = 1.0  # how far the walls run below the ground

    # How each building is topped. Weighted by repetition rather than by
    # numbers: a flat roof twice for every pitch, which is roughly what a
    # Japanese block looks like once there is anything above two storeys on it.
    #
    # This is here rather than in a later pass because of what reads it. A
    # reconstruction model returns the shapes it is shown, so a scene whose
    # buildings are all flat-topped extrusions produces reconstructions that
    # are all flat-topped extrusions — the intent has to be in the massing
    # before anything downstream can pick it up.
    roof_forms: list[str] = field(
        default_factory=lambda: ["flat", "flat", "gable", "hip", "mono"])
    roof_pitch: tuple[float, float] = (0.35, 0.65)  # rise over half the span
    roof_eave: float = 0.7  # how far the roof oversails the walls

    # The base course. Small, and the thing that stops a building reading as a
    # card standing on a board when the camera is at eye level.
    plinth_height: float = 0.35
    plinth_proud: float = 0.12

    # How much of the lot's street side is left unbuilt — the parking space, the
    # bicycle, the strip of gravel. Taken off before the coverage inset, so the
    # building goes to the back of its lot rather than sitting in the middle of
    # it with an even gap all round.
    frontage: float = 3.5
    # How far a lot may be from the nearest road and still be built on. A real
    # map's roads cover a fraction of the ground, and filling the rest of the
    # bounding box puts houses where nothing can reach them.
    max_road_distance: float = 45.0

    max_buildings: int = 0  # 0 = unlimited
    seed: int = 0

    def __post_init__(self) -> None:
        self.roof_forms = list(self.roof_forms)
        unknown = set(self.roof_forms) - set(ROOF_FORMS)
        if unknown:
            raise ValueError(
                f"unknown roof form(s): {', '.join(sorted(unknown))}; "
                f"expected {', '.join(ROOF_FORMS)}")
        if not self.roof_forms:
            raise ValueError("buildings.roof_forms cannot be empty; use ['flat']")


# ---------------------------------------------------------------------------
# Where a building may stand
# ---------------------------------------------------------------------------


def exclusion_zone(groups: dict[str, Sequence[Any]]) -> Any:
    """Everything a building must not stand on.

    Every paved surface, *including the lanelets classified as elevated*. The
    ground layer excludes those on purpose — a viaduct wants ground underneath
    it — but a building does not: the deck's approach ramp is on the street,
    and building over it put walls on the carriageway (measured: 3.7 % of wall
    vertices before this).
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    polygons = []
    for name in ("Roads", "Junctions", "Crosswalks", "Walkways"):
        for shape in groups.get(name, ()):
            ring = getattr(shape, "ring", None)
            if ring is None:
                continue
            poly = ShapelyPolygon([(p[0], p[1]) for p in ring()])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if not poly.is_empty and poly.area > 1e-6:
                polygons.append(poly)
    return unary_union(polygons) if polygons else None


def buildable_area(road_union, bounds, *, setback: float):
    """The ground, minus the roads and a setback, as block polygons.

    Interior rings are kept. A road that runs through the middle of a region
    shows up as a hole, and taking only the exterior would hand that road back
    as buildable — measured, that put 449 of 2481 plots on the carriageway.
    """
    from shapely.geometry import box

    extent = box(*bounds)
    free = extent.difference(road_union.buffer(setback)) if road_union is not None else extent
    parts = list(free.geoms) if hasattr(free, "geoms") else [free]
    return [
        part for part in parts
        if part.geom_type == "Polygon" and not part.is_empty and part.area > 1.0
    ]


def split_lots(block, options: BuildingOptions, rng: random.Random) -> list:
    """Cut a block into lots by repeatedly halving its long axis."""
    from shapely.geometry import LineString
    from shapely.ops import split as shapely_split

    if block.area <= options.target_lot_area:
        return [block]

    rectangle = block.minimum_rotated_rectangle
    corners = list(rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        return [block]

    edges = [(np.array(corners[i]), np.array(corners[(i + 1) % 4])) for i in range(4)]
    (a, b) = max(edges, key=lambda e: np.linalg.norm(e[1] - e[0]))
    axis = b - a
    length = float(np.linalg.norm(axis))
    if length < 1e-6:
        return [block]
    axis = axis / length

    # Cut across the long axis, a little off-centre so the lots vary.
    offset = 0.5 + rng.uniform(-options.split_jitter, options.split_jitter)
    centre = a + axis * (length * offset)
    normal = np.array([-axis[1], axis[0]])
    reach = float(np.hypot(*(np.array(block.bounds[2:]) - np.array(block.bounds[:2])))) + 10.0
    knife = LineString([centre - normal * reach, centre + normal * reach])

    try:
        pieces = list(shapely_split(block, knife).geoms)
    except (GEOSException, ValueError):  # pragma: no cover - degenerate cut
        return [block]
    if len(pieces) < 2:
        return [block]

    lots = []
    for piece in pieces:
        if piece.geom_type != "Polygon" or piece.area < options.min_lot_area:
            continue
        lots.extend(split_lots(piece, options, rng))
    return lots or [block]


def inset_to_coverage(lot, coverage: float, minimum_margin: float):
    """Shrink a lot until its building covers ``coverage`` of it.

    Insetting by a fixed margin makes density depend on lot size — the same
    1.5 m gap leaves a 400 m2 lot 74 % built and a 2500 m2 lot 88 %. Solving for
    the inset instead makes the ratio the parameter, which is how a planner
    states it (建蔽率), and the gap between neighbours falls out of it.
    """
    target = lot.area * coverage
    low, high = minimum_margin, math.sqrt(lot.area) / 2.0
    if lot.buffer(-low).area <= target:
        return lot.buffer(-low)  # the minimum gap already gives us that density

    for _ in range(24):
        middle = (low + high) / 2.0
        if lot.buffer(-middle).area > target:
            low = middle
        else:
            high = middle
    return lot.buffer(-high)


def give_up_the_frontage(lot, road_union, depth: float):
    """Take a strip off the street side of a lot, so the building stands back.

    Insetting a lot to its coverage ratio shrinks it towards its own middle,
    which leaves the same thin gap on all four sides — and a street of those
    reads as one continuous wall with slots in it. A real lot is not used that
    way: the building goes to the back and the front is a yard, a parking space
    or a bicycle. So the frontage is taken off *before* the coverage inset, on
    whichever side is nearest the road.

    Which side that is has to be asked rather than assumed: a corner plot faces
    two streets and a plot behind another faces none.
    """
    from shapely.geometry import LineString
    from shapely.ops import nearest_points
    from shapely.ops import split as shapely_split

    if road_union is None or depth <= 0.0:
        return lot
    here, there = nearest_points(lot.centroid, road_union)
    towards = np.array([there.x - here.x, there.y - here.y])
    reach = float(np.linalg.norm(towards))
    if reach < 1e-6:
        return lot
    towards /= reach

    # The cut runs across that direction, `depth` back from the street-most
    # point of the lot.
    corners = np.asarray(lot.exterior.coords)[:, :2]
    front = corners @ towards
    # Never more than a third of the lot's own depth. A fixed frontage on a
    # small lot leaves nothing to build on — measured, a 3.5 m strip took a
    # street of 238 houses down to 165, because the remainder fell below the
    # minimum after the coverage inset.
    depth = min(depth, (front.max() - front.min()) / 3.0)
    at = front.max() - depth
    origin = np.array([here.x, here.y]) + towards * (at - here.x * towards[0]
                                                     - here.y * towards[1])
    across = np.array([-towards[1], towards[0]])
    span = lot.length + reach
    knife = LineString([tuple(origin - across * span), tuple(origin + across * span)])
    try:
        pieces = [p for p in shapely_split(lot, knife).geoms if p.area > 1.0]
    except (GEOSException, ValueError):  # pragma: no cover - degenerate cut
        return lot
    if len(pieces) < 2:
        return lot
    # Keep the piece furthest from the road: that is the back of the lot.
    return max(pieces, key=lambda p: -(np.asarray(p.centroid.coords)[0] @ towards))


def footprints(
    road_union,
    bounds,
    options: BuildingOptions | None = None,
) -> list:
    """Building footprints for the whole map, as shapely polygons."""
    options = options or BuildingOptions()
    rng = random.Random(options.seed)

    plots = []
    for block in buildable_area(road_union, bounds, setback=options.setback):
        for lot in split_lots(block, options, rng):
            if lot.area < options.min_lot_area:
                continue
            if options.vacancy and rng.random() < options.vacancy:
                continue  # left as open ground
            # A building nobody can reach is a building that is not there. On a
            # real map the roads cover a fraction of the ground, and filling
            # the rest of the bounding box puts houses in the middle of fields.
            if (road_union is not None and options.max_road_distance > 0
                    and lot.distance(road_union) > options.max_road_distance):
                continue
            floor = options.min_lot_area * 0.5

            def built(on, whole=lot, options=options):
                # `coverage` is the share of *the lot* that is built, so a
                # frontage already given up counts towards it: insetting the
                # remainder by the same ratio again would charge for it twice.
                share = min(1.0, whole.area * options.coverage / max(on.area, 1e-9))
                shape = inset_to_coverage(on, share, options.lot_margin)
                parts = shape.geoms if hasattr(shape, "geoms") else [shape]
                return [p for p in parts
                        if p.geom_type == "Polygon" and p.area >= options.min_lot_area * 0.5]

            standing = give_up_the_frontage(lot, road_union,
                                            options.frontage * rng.uniform(0.7, 1.3))
            made = built(standing) if standing.area >= floor else []
            if not made:
                # A lot too small to afford both a frontage and a house builds
                # to the street, which is what a narrow urban lot does. Tried
                # rather than predicted: measured, refusing the frontage only
                # after it fails keeps 236 of 238 houses instead of 167.
                made = built(lot)
            plots.extend(made)

    if options.max_buildings and len(plots) > options.max_buildings:
        plots.sort(key=lambda p: p.area, reverse=True)
        plots = plots[: options.max_buildings]
    return plots


def pick_height(area: float, options: BuildingOptions, rng: random.Random) -> float:
    """A plausible height for a plot, snapped to whole floors.

    Bigger plots lean taller — a tower needs a footprint — but the draw stays
    random so a street does not come out monotonic.
    """
    span = options.max_height - options.min_height
    footprint_pull = min(1.0, area / 2500.0)
    draw = rng.random() ** (1.0 - options.tall_bias * footprint_pull)
    height = options.min_height + span * draw * (0.35 + 0.65 * footprint_pull)
    floors = max(1, round(height / options.floor_height))
    return floors * options.floor_height


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------


def _triangulate(polygon) -> list[list[tuple[float, float]]]:
    """Triangulate a simple polygon (Delaunay, keeping the inside triangles)."""
    from shapely.ops import triangulate
    from shapely.prepared import prep

    inside = prep(polygon)
    return [
        list(tri.exterior.coords)[:-1]
        for tri in triangulate(polygon)
        if inside.contains(tri.representative_point())
    ]


def plinth(polygon, base_z: float, *, height: float = 0.35, proud: float = 0.12,
           skirt: float = 1.0) -> Mesh:
    """The base course a building stands on, a little proud of its walls.

    Without one the wall meets the ground as a line, and close to, the building
    reads as a card model standing on a board — which is exactly what it is.
    Every house in the street this generates has a concrete base about a third
    of a metre high and a hand's width wider than the wall above it, and the
    step is what tells the eye the building has a thickness.

    Merged into the wall mesh rather than added beside it: a scene has one
    Buildings object however many buildings are in it, and everything that
    picks one out — the facade sheet it wears, the portrait that photographs
    it — works from one face range per building.
    """
    outer = polygon.buffer(proud, join_style=2)
    outer = max(outer.geoms, key=lambda g: g.area) if hasattr(outer, "geoms") else outer
    if outer.is_empty:
        return Mesh([], [])

    rings = [list(outer.exterior.coords)[:-1]]
    if signed_area_xy(rings[0]) < 0:
        rings[0] = list(reversed(rings[0]))
    for interior in outer.interiors:
        ring = list(interior.coords)[:-1]
        if len(ring) >= 3:
            rings.append(ring if signed_area_xy(ring) < 0 else list(reversed(ring)))

    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    uvs: list[tuple[float, float]] = []
    top, bottom = base_z + height, base_z - skirt
    for ring in rings:
        for i, (x, y) in enumerate(ring):
            nx, ny = ring[(i + 1) % len(ring)]
            at = len(vertices)
            vertices.extend([(x, y, bottom), (nx, ny, bottom), (nx, ny, top), (x, y, top)])
            faces.append([at, at + 1, at + 2, at + 3])
            # The very bottom of the sheet, which is where a facade layout puts
            # its ground line. A band, not a stretch: the plinth is not a storey.
            uvs.extend([(0.0, 0.0), (1.0, 0.0), (1.0, 0.03), (0.0, 0.03)])

    cap = []
    for triangle in _triangulate(outer):
        ordered = triangle if signed_area_xy(triangle) > 0 else list(reversed(triangle))
        at = len(vertices)
        for x, y in ordered:
            vertices.append((x, y, top))
            uvs.append((0.0, 0.03))
        cap.append([at, at + 1, at + 2])
    faces.extend(cap)
    return Mesh(vertices, faces, uvs)


def extrude(polygon, base_z: float, height: float, *, skirt: float = 1.0,
            facade_width: float = 12.0, roof_tile: float = 12.0) -> tuple[Mesh, Mesh]:
    """Walls and roof for one footprint, with UVs.

    The walls start ``skirt`` below the base so a building on a slope has no gap
    under it — the ground is a coarse surface and the footprint is flat.

    The UVs are the reason to generate buildings rather than inherit them: U
    runs along the wall in metres, so a sheet repeats about every
    ``facade_width`` and reads at the same scale on every building; V runs 0 at
    the pavement to 1 at the roofline, so a sheet's shopfront lands on the
    ground floor and its roofline at the top whatever the building's height.

    That V normalisation is what makes a sheet belong to a *floor count* rather
    than to a height: stretched over a building with a different number of
    storeys, its windows stop lining up with anything. See
    :mod:`city_builder.facade_layout`.
    """
    exterior = list(polygon.exterior.coords)[:-1]
    if len(exterior) < 3:
        return Mesh([], []), Mesh([], [])
    if signed_area_xy(exterior) < 0:
        exterior = list(reversed(exterior))  # counter-clockwise: walls face outward

    # A courtyard needs walls too, wound the other way so they face inward.
    rings = [exterior]
    for interior in polygon.interiors:
        ring = list(interior.coords)[:-1]
        if len(ring) < 3:
            continue
        rings.append(ring if signed_area_xy(ring) < 0 else list(reversed(ring)))

    top_z = base_z + height
    bottom_z = base_z - skirt

    wall_vertices: list[tuple[float, float, float]] = []
    wall_faces: list[list[int]] = []
    wall_uvs: list[tuple[float, float]] = []
    for ring in rings:
        # Stretch the sheet slightly so a whole number of them goes round the
        # ring. Dividing by ``facade_width`` flat leaves the last repeat cut off
        # mid-window at the corner where the ring closes; absorbing the
        # remainder into the scale — at most a few per cent — makes the wrap
        # exact and puts no seam anywhere.
        perimeter = sum(math.dist(ring[i], ring[(i + 1) % len(ring)]) for i in range(len(ring)))
        repeats = max(1, round(perimeter / facade_width))
        u_scale = repeats / perimeter if perimeter > 1e-9 else 0.0

        run = 0.0  # metres travelled along this ring, for U
        for i, (x, y) in enumerate(ring):
            nx, ny = ring[(i + 1) % len(ring)]
            base = len(wall_vertices)
            wall_vertices.extend([
                (x, y, bottom_z), (nx, ny, bottom_z), (nx, ny, top_z), (x, y, top_z),
            ])
            wall_faces.append([base, base + 1, base + 2, base + 3])

            span = math.dist((x, y), (nx, ny))
            u0, u1 = run * u_scale, (run + span) * u_scale
            v_bottom = -skirt / height  # the buried part continues below V=0
            wall_uvs.extend([(u0, v_bottom), (u1, v_bottom), (u1, 1.0), (u0, 1.0)])
            run += span

    roof_vertices: list[tuple[float, float, float]] = []
    roof_faces: list[list[int]] = []
    index: dict[tuple[int, int], int] = {}
    for triangle in _triangulate(polygon):
        ordered = triangle if signed_area_xy(triangle) > 0 else list(reversed(triangle))
        face = []
        for x, y in ordered:
            key = (round(x * 1000), round(y * 1000))
            if key not in index:
                index[key] = len(roof_vertices)
                roof_vertices.append((x, y, top_z))
            face.append(index[key])
        if len(set(face)) == 3:
            roof_faces.append(face)

    roof_uvs = [(x / roof_tile, y / roof_tile) for x, y, _ in roof_vertices]
    return Mesh(wall_vertices, wall_faces, wall_uvs), Mesh(roof_vertices, roof_faces, roof_uvs)


ROOF_FORMS = ("flat", "gable", "hip", "mono")


def pitched_roof(polygon, top_z: float, form: str, *, pitch: float = 0.45,
                 eave: float = 0.5) -> Mesh:
    """A pitched roof that follows the footprint it stands on. ``flat`` gives none.

    The first version of this carried the roof on the footprint's *minimum
    rotated rectangle*, which is a slab as wide as the building's bounding box
    however the building is actually shaped. On an L-shaped or a wedge-shaped
    plot — and a plot cut out of the space between real roads is usually one of
    those — the roof stood well clear of the walls on two sides. It read as a
    lid resting on the building rather than as its top.

    So the roof is a *height field over the plan*. The plan is the footprint
    grown by the eave, which is what an overhang is; the height comes from the
    slope rising inward from each eave edge, and where two of those meet is a
    ridge or a hip. Those meetings are straight lines, so the plan is cut along
    them and each piece then lies in one plane exactly — no faceting on a
    crease, and no roof outside the building it belongs to.

    A flat roof is what an extrusion gives, and for a long time it was all this
    package had. That reads as a street of tofu blocks, and it carries: a
    reconstruction model can only return the shapes it is shown, so a scene
    with no pitched roof in it produces buildings with no pitched roof either.
    """
    from shapely.geometry import LineString
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import split as shapely_split

    if form not in ROOF_FORMS:
        raise ValueError(f"roof form must be one of {ROOF_FORMS}, not {form!r}")
    if form == "flat":
        return Mesh([], [])

    plan = polygon.buffer(eave, join_style=2)  # mitred, so a corner stays a corner
    plan = max(plan.geoms, key=lambda g: g.area) if hasattr(plan, "geoms") else plan
    if plan.is_empty or plan.area <= 0:
        return Mesh([], [])

    rectangle = polygon.minimum_rotated_rectangle
    corners = list(rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        return Mesh([], [])
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    (ax, ay), (bx, by) = max(edges, key=lambda e: math.dist(e[0], e[1]))
    length = math.dist((ax, ay), (bx, by))
    if length < 1e-6:
        return Mesh([], [])
    ux, uy = (bx - ax) / length, (by - ay) / length  # along the ridge
    cx, cy = rectangle.centroid.coords[0]
    width = rectangle.area / length
    half_l, half_w = length / 2.0 + eave, width / 2.0 + eave
    slope = pitch  # rise per metre travelled in from the eave
    # The roof plane has to pass through the *wall* top, not the eave's. Set
    # the eave to the wall top instead and the roof stands `slope * eave` clear
    # of the building all the way round — measured at a 0.9 m eave and a 0.8
    # pitch, 0.72 m of daylight under the roof. The overhang hangs below the
    # wall top, which is what an overhang does.
    hang = slope * eave

    def to_local(x: float, y: float) -> tuple[float, float]:
        dx, dy = x - cx, y - cy
        return dx * ux + dy * uy, -dx * uy + dy * ux

    def to_world(u: float, v: float) -> tuple[float, float]:
        return cx + ux * u - uy * v, cy + uy * u + ux * v

    local = ShapelyPolygon([to_local(x, y) for x, y in plan.exterior.coords[:-1]],
                           [[to_local(x, y) for x, y in ring.coords[:-1]]
                            for ring in plan.interiors])
    if not local.is_valid:
        local = local.buffer(0)

    # Each eave edge lifts the roof as it is left behind. A gable has two, a
    # hip has four, a mono-pitch has one that carries the whole width.
    if form == "mono":
        planes = [(0.0, slope, slope * half_w - hang)]
        creases: list[tuple[float, float, float]] = []
    elif form == "gable":
        planes = [(0.0, -slope, slope * half_w - hang), (0.0, slope, slope * half_w - hang)]
        creases = [(0.0, 1.0, 0.0)]  # v = 0
    else:  # hip
        planes = [(0.0, -slope, slope * half_w - hang), (0.0, slope, slope * half_w - hang),
                  (-slope, 0.0, slope * half_l - hang), (slope, 0.0, slope * half_l - hang)]
        offset = half_l - half_w
        creases = [(0.0, 1.0, 0.0),
                   (1.0, -1.0, -offset), (1.0, 1.0, -offset),
                   (1.0, 1.0, offset), (1.0, -1.0, offset)]

    reach = (half_l + half_w) * 4.0
    pieces = [local]
    for a, b, c in creases:
        # The line a*u + b*v = c, long enough to cross the whole plan.
        norm = math.hypot(a, b)
        px, py = a * c / norm**2, b * c / norm**2
        dx, dy = -b / norm, a / norm
        knife = LineString([(px - dx * reach, py - dy * reach),
                            (px + dx * reach, py + dy * reach)])
        cut = []
        for piece in pieces:
            try:
                cut.extend(p for p in shapely_split(piece, knife).geoms if p.area > 1e-6)
            except (GEOSException, ValueError):  # pragma: no cover - degenerate cut
                cut.append(piece)
        pieces = cut

    vertices: list[tuple[float, float, float]] = []
    faces: list[list[int]] = []
    for piece in pieces:
        if piece.geom_type != "Polygon" or piece.area <= 1e-6:
            continue
        # Which plane is lowest here is which one roofs this piece; inside one
        # crease-bounded piece that answer does not change.
        spot = piece.representative_point()
        a, b, c = min(planes, key=lambda p: p[0] * spot.x + p[1] * spot.y + p[2])
        for triangle in _triangulate(piece):
            ordered = triangle if signed_area_xy(triangle) > 0 else list(reversed(triangle))
            face = []
            for u, v in ordered:
                x, y = to_world(u, v)
                face.append(len(vertices))
                # Not clamped at the wall top: the eave hangs below it.
                vertices.append((x, y, top_z + a * u + b * v + c))
            if len(set(face)) == 3:
                faces.append(face)

    return Mesh(vertices, faces, [(v[0] / 3.0, v[1] / 3.0) for v in vertices])


def base_height(polygon, heightmap: HeightMap) -> float:
    """Ground height to stand a footprint on: the lowest of its corners.

    Taking the lowest means a building on a slope cuts into the hill on the
    high side rather than floating on the low side, which is what a real
    building does.
    """
    return min(heightmap.sample(x, y) for x, y in polygon.exterior.coords)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def generate(
    heightmap: HeightMap,
    road_union,
    options: BuildingOptions | None = None,
    *,
    bounds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Fill the empty ground with buildings.

    Returns ``{"Buildings": [Mesh], "Roofs": [Mesh], "plots": [...]}`` — the
    walls and roofs are separate because a texturing pass treats a facade and a
    roof completely differently.
    """
    options = options or BuildingOptions()
    if bounds is None:
        bounds = (
            heightmap.x0,
            heightmap.y0,
            heightmap.x0 + (heightmap.nx - 1) * heightmap.cell,
            heightmap.y0 + (heightmap.ny - 1) * heightmap.cell,
        )

    rng = random.Random(options.seed + 1)
    plots = footprints(road_union, bounds, options)

    walls, roofs, records = [], [], []
    for plot in plots:
        base = base_height(plot, heightmap)
        height = pick_height(plot.area, options, rng)
        wall_mesh, roof_mesh = extrude(plot, base, height, skirt=options.skirt,
                                      facade_width=options.facade_width)
        if options.plinth_height > 0:
            from .geometry import merge_meshes

            standing = plinth(plot, base, height=options.plinth_height,
                              proud=options.plinth_proud, skirt=options.skirt)
            if standing.faces:
                wall_mesh = merge_meshes([wall_mesh, standing])
        form = rng.choice(options.roof_forms)
        pitch = rng.uniform(*options.roof_pitch)
        if form != "flat":
            # The pitch replaces the extrusion's cap rather than sitting over
            # it: two roofs in the same place is a surface a renderer z-fights
            # over and a reconstruction has to choose between.
            pitched = pitched_roof(plot, base + height, form, pitch=pitch,
                                   eave=options.roof_eave)
            if pitched.faces:
                roof_mesh = pitched
            else:
                form = "flat"
        if not wall_mesh.faces or not roof_mesh.faces:
            continue
        walls.append(wall_mesh)
        roofs.append(roof_mesh)
        records.append({
            "area": round(plot.area, 2),
            "height": round(height, 2),
            # What tops it, so anything downstream photographing this building
            # knows what it is looking at without measuring the mesh.
            "roof": form,
            "roof_pitch": round(pitch, 3) if form != "flat" else 0.0,
            # The facade UV normalises V over the height, so a sheet belongs to
            # a floor count rather than to a height. Recorded per building
            # because that is what decides which sheet it may wear.
            "floors": max(1, round(height / options.floor_height)),
            "base_z": round(base, 3),
            "centroid": [round(plot.centroid.x, 3), round(plot.centroid.y, 3)],
            # The plot itself, not just where it is. A generated mesh comes back
            # from a reconstruction normalised into a unit cube with no memory
            # of how big the thing was, and this ring is the only statement of
            # scale this package holds — fitting one to the other is what puts
            # the model back in the scene at the right size. The exterior only:
            # a courtyard changes the area, not the extent being fitted.
            "footprint": [[round(x, 3), round(y, 3)]
                          for x, y in list(plot.exterior.coords)[:-1]],
        })

    return {"Buildings": walls, "Roofs": roofs, "plots": records}
