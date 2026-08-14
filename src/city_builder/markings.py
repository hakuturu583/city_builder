"""Road markings as a texture layer rather than as geometry.

Painted markings are not objects. Building them as coplanar slabs a couple of
millimetres above the carriageway is what a mesh pipeline does when it has
nowhere else to put them, and it costs: 2938 lane-line ribbons, 1327 zebra
bars and 207 stop lines on the Nishi-Shinjuku map, all fighting the road for
the same depth, and all of them free to hang past the edge of a viaduct deck
because nothing tells a stripe where the road stops.

Baking them into the road's own texture fixes all three at once. The paint is
clipped to the surface by construction, there is no second surface to fight,
and the face count drops to the carriageway alone.

**Resolution is the whole problem.** The carriageway is about 60 000 m²; one
image over the map at a resolution that keeps a 15 cm line crisp would be
hundreds of megatexels. But a lanelet is a *ribbon*, so it has a natural
parameterisation — along it and across it — and across it the useful resolution
is fixed by the lane width rather than by the map. Every lane is rasterised at
``across_pixels`` texels wide and however many long its length needs, and the
strips are packed into a few pages.

The class registry's distinction survives the move. ``preserve`` stops being a
property of a group of objects and becomes the mask channel itself: the road's
colour may be regenerated wherever the mask is zero, and must not be touched
where it is not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .geometry import Polygon, Ribbon

#: Groups whose geometry is paint on the carriageway rather than a surface.
PAINTED = ("LaneMarkings", "StopLines", "CrosswalkStripes")

#: Groups that carry the paint. Ribbons only: a lane's own along-and-across
#: coordinates are what makes the resolution tractable, and a patch of infill
#: or a clipped crossing has no such axis.
CARRIES_PAINT = ("Roads", "Junctions")


@dataclass
class MarkingOptions:
    """How the paint is baked, and how finely."""

    texture: bool = True  # False keeps the old coplanar geometry
    across_pixels: int = 64  # texels across a lane, whatever its width
    page_pixels: int = 4096
    reach: float = 0.6  # how far outside a lane to collect paint (m)
    supersample: int = 2  # draw this much larger, then shrink, for smooth edges
    colour: tuple[float, float, float] = (0.92, 0.92, 0.90)
    roughness: float = 0.55  # paint is smoother than the asphalt around it

    # Measured on the Nishi-Shinjuku carriageway, 887 lanes:
    #
    #   across   cm/texel   Mtexel   pages   a 15 cm line
    #       32       9.5      13.5       1        1.6 px
    #       64       4.7      52.5       4        3.2 px
    #      128       2.4     189.2      15        6.3 px
    #
    # 64 is the knee: a thin line still has three texels across it, and the
    # whole map's paint fits in four pages.

    def __post_init__(self) -> None:
        if self.across_pixels < 8:
            raise ValueError("across_pixels below 8 cannot draw a line")
        if self.page_pixels % self.across_pixels:
            raise ValueError(
                f"page_pixels ({self.page_pixels}) must be a whole number of "
                f"columns of across_pixels ({self.across_pixels})"
            )


# ---------------------------------------------------------------------------
# A lane's own coordinates
# ---------------------------------------------------------------------------


class LaneFrame:
    """Along-and-across coordinates for one ribbon.

    ``u`` runs 0 to 1 from the start of the lane to its end, ``v`` runs 0 to 1
    from the left boundary to the right. A point off the end or off the side
    lands outside that range; the rasteriser clips.

    The mapping is the ribbon's own triangles, interpolated barycentrically,
    which makes it the exact inverse of the UV the mesh gets. Projecting onto
    the centreline instead is the obvious thing and it is subtly wrong: a
    cross-section is the chord between two boundary points, and on a curve that
    chord is not perpendicular to the centreline, so the station comes out
    slightly off and the painted line wanders across the lane.
    """

    def __init__(self, ribbon):
        left = np.asarray(ribbon.left, dtype=float)[:, :2]
        right = np.asarray(ribbon.right, dtype=float)[:, :2]
        n = min(len(left), len(right))
        self.left, self.right = left[:n], right[:n]
        self.centre = (self.left + self.right) / 2.0

        steps = np.linalg.norm(np.diff(self.centre, axis=0), axis=1)
        self.station = np.concatenate([[0.0], np.cumsum(steps)])
        self.length = float(self.station[-1]) if len(self.station) else 0.0
        self.widths = np.linalg.norm(self.left - self.right, axis=1)
        self.width = float(np.median(self.widths)) if len(self.widths) else 0.0

        self._triangles, self._corner_uvs = self._mesh()

    def __len__(self) -> int:
        return len(self.centre)

    def _mesh(self):
        """Every cell of the ribbon as two triangles, with the UV of each corner."""
        if len(self.centre) < 2 or self.length <= 0:
            return np.zeros((0, 3, 2)), np.zeros((0, 3, 2))

        u = self.station / self.length
        triangles, uvs = [], []
        for i in range(len(self.centre) - 1):
            a, b = self.left[i], self.right[i]
            c, d = self.right[i + 1], self.left[i + 1]
            triangles.extend([[a, b, c], [a, c, d]])
            uvs.extend([
                [(u[i], 0.0), (u[i], 1.0), (u[i + 1], 1.0)],
                [(u[i], 0.0), (u[i + 1], 1.0), (u[i + 1], 0.0)],
            ])
        return np.asarray(triangles, dtype=float), np.asarray(uvs, dtype=float)

    def project(self, points: np.ndarray) -> np.ndarray:
        """``(u, v)`` for each xy point, both normalised."""
        triangles = self._triangles
        if not len(triangles) or not len(points):
            return np.zeros((len(points), 2))

        a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        e0, e1 = b - a, c - a
        d00 = np.einsum("ij,ij->i", e0, e0)
        d01 = np.einsum("ij,ij->i", e0, e1)
        d11 = np.einsum("ij,ij->i", e1, e1)
        denominator = d00 * d11 - d01 * d01
        denominator[np.abs(denominator) < 1e-12] = 1e-12

        offset = points[:, None, :] - a[None, :, :]
        d20 = np.einsum("pij,ij->pi", offset, e0)
        d21 = np.einsum("pij,ij->pi", offset, e1)
        beta = (d11 * d20 - d01 * d21) / denominator
        gamma = (d00 * d21 - d01 * d20) / denominator
        alpha = 1.0 - beta - gamma

        # The triangle a point is most inside; for a point outside every one,
        # the least-outside, whose extrapolation puts it off the strip.
        weights = np.stack([alpha, beta, gamma], axis=2)
        pick = np.argmax(weights.min(axis=2), axis=1)

        rows = np.arange(len(points))
        chosen = weights[rows, pick]
        return np.einsum("pk,pkj->pj", chosen, self._corner_uvs[pick])


def lane_pixels(frame: LaneFrame, options: MarkingOptions) -> tuple[int, int]:
    """``(width, height)`` of the strip for this lane, in texels.

    Across is fixed, so the texel size across a lane is set by its width; along
    is whatever keeps the texels square, bounded by the page.
    """
    if frame.width <= 0 or frame.length <= 0:
        return 0, 0
    texel = frame.width / options.across_pixels
    height = round(frame.length / texel)
    return options.across_pixels, max(2, min(height, options.page_pixels))


# ---------------------------------------------------------------------------
# Rasterising
# ---------------------------------------------------------------------------


def _outline(shape) -> np.ndarray | None:
    """The plan-view outline of a painted shape."""
    if isinstance(shape, Ribbon):
        points = list(shape.left) + list(reversed(list(shape.right)))
    elif isinstance(shape, Polygon):
        points = list(shape.points)
    else:
        return None
    return np.asarray([(p[0], p[1]) for p in points], dtype=float) if len(points) >= 3 else None


def rasterise(frame: LaneFrame, shapes: Sequence[object], size: tuple[int, int],
              supersample: int = 1):
    """Draw every painted shape into this lane's strip, white on black.

    Drawn larger and shrunk down, because polygon fill has no antialiasing and
    a lane line crosses the strip at an angle more often than not: without it
    the edges come out as visible stair steps at 4.7 cm per texel.
    """
    from PIL import Image, ImageDraw

    width, height = size
    if width < 2 or height < 2:
        return Image.new("L", (max(width, 1), max(height, 1)), 0)

    scale = max(1, int(supersample))
    image = Image.new("L", (width * scale, height * scale), 0)
    draw = ImageDraw.Draw(image)
    for shape in shapes:
        outline = _outline(shape)
        if outline is None:
            continue
        uv = frame.project(outline)
        # Row 0 is the start of the lane; the UV flip happens when the strip is
        # placed on the page, not here.
        polygon = [(float(v * width * scale), float(u * height * scale)) for u, v in uv]
        if len(polygon) >= 3:
            draw.polygon(polygon, fill=255)

    return image.resize((width, height), Image.LANCZOS) if scale > 1 else image


def paint_near(shapes: Sequence[object], reach: float):
    """An index of painted shapes, to ask which ones touch a lane."""
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.strtree import STRtree

    geometries, keep = [], []
    for shape in shapes:
        outline = _outline(shape)
        if outline is None:
            continue
        polygon = ShapelyPolygon(outline)
        if not polygon.is_valid:
            polygon = polygon.buffer(0)
        if polygon.is_empty or polygon.area <= 1e-9:
            continue
        geometries.append(polygon)
        keep.append(shape)

    if not geometries:
        return lambda _ribbon: []

    tree = STRtree(geometries)

    def near(ribbon):
        ring = [(p[0], p[1]) for p in ribbon.ring()]
        if len(ring) < 4:
            return []
        lane = ShapelyPolygon(ring)
        if not lane.is_valid:
            lane = lane.buffer(0)
        return [keep[i] for i in tree.query(lane.buffer(reach))]

    return near


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


@dataclass
class Placement:
    """Where one lane's strip ended up."""

    page: int
    x: int
    y: int
    width: int
    height: int


def pack(sizes: Sequence[tuple[int, int]], options: MarkingOptions) -> list[Placement | None]:
    """Place fixed-width strips into columns, columns into pages.

    Every strip is the same width, so the general rectangle-packing problem
    does not arise: a page is a fixed number of columns and a column is filled
    top to bottom.
    """
    columns = options.page_pixels // options.across_pixels
    placements: list[Placement | None] = []
    page = column = cursor = 0

    for width, height in sizes:
        if width <= 0 or height <= 0:
            placements.append(None)
            continue
        if cursor + height > options.page_pixels:
            column += 1
            cursor = 0
        if column >= columns:
            page += 1
            column = cursor = 0
        placements.append(Placement(page, column * options.across_pixels, cursor, width, height))
        cursor += height
    return placements


def compose(strips: Sequence[object], placements: Sequence[Placement | None],
            options: MarkingOptions) -> list[np.ndarray]:
    """Blit every strip onto its page."""
    pages = 1 + max((p.page for p in placements if p is not None), default=-1)
    if not pages:
        return []
    canvases = [np.zeros((options.page_pixels, options.page_pixels), dtype=np.uint8)
                for _ in range(pages)]
    for strip, place in zip(strips, placements):
        if place is None:
            continue
        canvases[place.page][place.y:place.y + place.height,
                             place.x:place.x + place.width] = np.asarray(strip)
    return canvases


def strip_uvs(frame: LaneFrame, place: Placement, options: MarkingOptions) -> list[tuple[float, float]]:
    """Atlas UV per vertex of the ribbon, in the order ``ribbon_to_mesh`` builds them.

    Left then right, one pair per cross-section — the same interleaving the
    mesh uses, so the list drops straight onto it.
    """
    page = float(options.page_pixels)
    uvs = []
    for i in range(len(frame)):
        along = frame.station[i] / frame.length if frame.length > 0 else 0.0
        row = place.y + along * place.height
        for side in (0.0, 1.0):  # left boundary, then right
            column = place.x + side * place.width
            uvs.append((column / page, 1.0 - row / page))
    return uvs


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def bake(groups: dict[str, list], options: MarkingOptions | None = None):
    """Bake the painted groups onto the surfaces that carry them.

    Returns ``(pages, page_of_shape)``: the mask images, and which page each
    shape of each carrying group landed on. The carrying ribbons come back with
    their atlas UVs set, and the painted groups are emptied — the paint is in
    the texture now.
    """
    options = options or MarkingOptions()
    painted = [shape for name in PAINTED for shape in groups.get(name, ())]
    carriers = [(name, shape) for name in CARRIES_PAINT for shape in groups.get(name, ())]
    if not options.texture or not painted or not carriers:
        return [], {}

    near = paint_near(painted, options.reach)
    frames, sizes = [], []
    for _name, ribbon in carriers:
        frame = LaneFrame(ribbon)
        frames.append(frame)
        sizes.append(lane_pixels(frame, options))

    placements = pack(sizes, options)
    strips = [
        rasterise(frame, near(ribbon), size, options.supersample)
        if place is not None else None
        for frame, size, place, (_n, ribbon) in zip(frames, sizes, placements, carriers)
    ]
    pages = compose([s for s in strips if s is not None],
                    [p for p, s in zip(placements, strips) if s is not None], options)

    page_of_shape: dict[str, list[int]] = {name: [] for name in CARRIES_PAINT}
    for (name, ribbon), frame, place in zip(carriers, frames, placements):
        page_of_shape[name].append(place.page if place else 0)
        if place is not None and hasattr(ribbon, "left"):
            ribbon.uvs = strip_uvs(frame, place, options)

    for name in PAINTED:
        groups.pop(name, None)
    return pages, {k: v for k, v in page_of_shape.items() if v}
