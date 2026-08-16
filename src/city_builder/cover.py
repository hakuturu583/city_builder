"""What the ground *is*, cell by cell — grass, gravel, paving, water, soil.

The height map says where the ground is. This says what it is made of, on a
grid of its own, and everything downstream reads from it: the texture that gets
painted, and eventually the mesh, since water is flat and a kerb is a
discontinuity where a lawn edge is not.

**Assigned by rule, not by classifier.** The obvious source is a land-cover
raster, and for a real site it is the right one — but three things argue
against making it the primary path here. A 10 m classifier cannot tell asphalt
from a rooftop (ESA WorldCover has no paved class at all: roads, car parks and
buildings are one `Built-up` label), while this package already knows where the
carriageway is to survey accuracy. Most of what it would classify is invented
by :mod:`city_builder.buildings` anyway and has no ground truth to look up. And
a rule set is *authorable* — a procedural city wants "gravel along the kerb,
yards inside the plots, a park in this block", which is a sentence, not a
download.

So the rules are painted in order, later over earlier, and the geometry the
package already has does most of the work: the dissolved carriageway, the
building plots, the height map's own slope. :class:`Region` is the escape hatch
for saying it by hand.

The class vocabulary is deliberately not invented here — :data:`SURFACES` is
OSM's ``surface=*`` values as OSM2World maps them, which is about twenty ground
materials that have already survived contact with global map data.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

DEFAULT_CELL = 2.0  # m; finer than the height grid, because a verge is 2 m wide

# The ground materials this package knows how to name, after OSM2World's
# surface map. Not every one needs a texture — an unpainted class falls back to
# its colour — and a caller is free to add its own.
SURFACES = (
    "asphalt", "concrete", "paving_stones", "sett", "cobblestone", "gravel",
    "pebblestone", "compacted", "ground", "earth", "sand", "grass",
    "grass_paver", "scrub", "wood", "woodchips", "rock", "water", "snow",
)


@dataclass
class Surface:
    """One ground material: what it is called and what it looks like."""

    name: str
    texture: str | None = None          # a tileable image; None uses the colour
    colour: tuple[float, float, float] = (0.5, 0.5, 0.5)
    tile_metres: float = 3.0
    # How assertively this material wins a boundary. Gravel poking through the
    # edge of a lawn is right; a lawn with a gravel-shaped hole in it is not.
    relief: float = 1.0


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass
class Rule:
    """Base: a rule names a surface and says which cells get it."""

    name: str

    def mask(self, board: _Board) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


@dataclass
class Everywhere(Rule):
    """The ground under everything else. Usually the first rule."""

    def mask(self, board: _Board) -> np.ndarray:
        return np.ones(board.shape, dtype=bool)


@dataclass
class NearRoad(Rule):
    """A band along the carriageway: the verge, the shoulder, the hard strip.

    Measured from the dissolved road outline, so it follows a junction round
    its corner rather than beading along each lanelet.
    """

    metres: float = 2.0
    inner: float = 0.0

    def mask(self, board: _Board) -> np.ndarray:
        if board.road_distance is None:
            return np.zeros(board.shape, dtype=bool)
        return (board.road_distance <= self.metres) & (board.road_distance > self.inner)


@dataclass
class InPlot(Rule):
    """Inside a building plot — the yard, the parking pad, the garden.

    ``shrink`` pulls the mask in from the plot boundary, which is what leaves a
    strip of whatever the plot stands on visible around each building instead
    of the yard running to the kerb.
    """

    shrink: float = 0.0

    def mask(self, board: _Board) -> np.ndarray:
        return board.inside(board.plots, shrink=self.shrink)


@dataclass
class Region(Rule):
    """A polygon in scene metres, said by hand. The escape hatch.

    This is the one that makes a park a park. Everything else here infers from
    geometry the map already carries; a pond, a car park, a plaza or a bare lot
    is a decision, and a decision belongs in the config.
    """

    polygon: Sequence[Sequence[float]] = ()

    def mask(self, board: _Board) -> np.ndarray:
        return board.inside([list(self.polygon)])


@dataclass
class Steeper(Rule):
    """Ground past a given slope. Grass does not hold on a batter; scree does."""

    degrees: float = 25.0

    def mask(self, board: _Board) -> np.ndarray:
        return board.slope_deg >= self.degrees


@dataclass
class Patches(Rule):
    """Procedural blobs covering a fraction of what is left.

    Value noise thresholded at whatever level gives the asked-for fraction, so
    ``fraction`` means what it says whatever the noise happened to do. This is
    the "some of this block is bare earth" rule, and it is the only one that is
    not derived from something — hence the seed.
    """

    fraction: float = 0.2
    metres: float = 25.0
    seed: int = 0

    def mask(self, board: _Board) -> np.ndarray:
        if self.fraction <= 0:
            return np.zeros(board.shape, dtype=bool)
        if self.fraction >= 1:
            return np.ones(board.shape, dtype=bool)
        noise = board.noise(self.metres, self.seed)
        return noise >= np.quantile(noise, 1.0 - self.fraction)


# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


@dataclass
class CoverOptions:
    """The palette and the rules that paint it."""

    surfaces: list[Surface] = field(default_factory=list)
    rules: list[Rule] = field(default_factory=list)
    cell: float = DEFAULT_CELL

    def surface(self, name: str) -> Surface:
        for entry in self.surfaces:
            if entry.name == name:
                return entry
        raise KeyError(f"no surface named {name!r} in the palette")


def japanese_suburb(*, textures: dict[str, str] | None = None) -> CoverOptions:
    """A palette and rule set for the streets this package generates.

    Gravel and paving where the plots meet the road, packed earth in the yards,
    grass in what is left over, and a scatter of bare ground. It is a starting
    point to edit, not a claim about any real place.
    """
    art = textures or {}

    def paint(name: str, colour: tuple[float, float, float], tile: float,
              relief: float = 1.0) -> Surface:
        return Surface(name, texture=art.get(name), colour=colour,
                       tile_metres=tile, relief=relief)

    return CoverOptions(
        surfaces=[
            paint("grass", (0.36, 0.42, 0.24), 2.5, relief=0.6),
            paint("ground", (0.42, 0.37, 0.30), 3.0, relief=0.8),
            paint("gravel", (0.55, 0.53, 0.50), 1.5, relief=1.4),
            paint("paving_stones", (0.62, 0.61, 0.59), 2.0, relief=1.2),
            paint("concrete", (0.66, 0.66, 0.65), 4.0, relief=1.0),
        ],
        rules=[
            Everywhere("grass"),
            Patches("ground", fraction=0.25, metres=30.0, seed=1),
            InPlot("ground", shrink=0.0),
            InPlot("gravel", shrink=1.2),
            NearRoad("paving_stones", metres=2.0),
            NearRoad("concrete", metres=0.6),
        ],
    )


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


@dataclass
class CoverMap:
    """Which surface each cell is, on a grid in the scene frame."""

    x0: float
    y0: float
    cell: float
    index: np.ndarray            # (ny, nx) into ``names``
    names: list[str]

    @property
    def ny(self) -> int:
        return self.index.shape[0]

    @property
    def nx(self) -> int:
        return self.index.shape[1]

    def at(self, x: float, y: float) -> str:
        ix = min(max(int((x - self.x0) / self.cell), 0), self.nx - 1)
        iy = min(max(int((y - self.y0) / self.cell), 0), self.ny - 1)
        return self.names[int(self.index[iy, ix])]

    def fractions(self) -> dict[str, float]:
        """How much of the ground each surface ended up as."""
        counts = np.bincount(self.index.ravel(), minlength=len(self.names))
        total = max(int(counts.sum()), 1)
        return {name: round(float(n) / total, 4)
                for name, n in zip(self.names, counts) if n}

    def to_json(self) -> dict[str, Any]:
        return {
            "x0": round(self.x0, 4), "y0": round(self.y0, 4), "cell": self.cell,
            "nx": self.nx, "ny": self.ny, "names": list(self.names),
            "index": [int(v) for v in self.index.ravel()],
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> CoverMap:
        index = np.asarray(data["index"], dtype=np.int16).reshape(int(data["ny"]),
                                                                  int(data["nx"]))
        return cls(float(data["x0"]), float(data["y0"]), float(data["cell"]),
                   index, list(data["names"]))


class _Board:
    """The geometry a rule is allowed to ask about, on the cover grid."""

    def __init__(self, x0: float, y0: float, nx: int, ny: int, cell: float,
                 heightmap=None, roads=None, plots: Sequence[Sequence[Sequence[float]]] = ()):
        self.x0, self.y0, self.cell = x0, y0, cell
        self.shape = (ny, nx)
        self.plots = [list(p) for p in plots]
        self.xs = x0 + (np.arange(nx) + 0.5) * cell
        self.ys = y0 + (np.arange(ny) + 0.5) * cell
        self.grid_x, self.grid_y = np.meshgrid(self.xs, self.ys)
        self._roads = roads
        self._heightmap = heightmap
        self._road_distance: np.ndarray | None = None
        self._slope: np.ndarray | None = None

    @property
    def road_distance(self) -> np.ndarray | None:
        """Metres from each cell centre to the carriageway, zero inside it."""
        if self._roads is None:
            return None
        if self._road_distance is None:
            from shapely.geometry import Point
            from shapely.prepared import prep

            inside = prep(self._roads)
            edge = self._roads.boundary
            out = np.empty(self.shape)
            for j in range(self.shape[0]):
                for i in range(self.shape[1]):
                    point = Point(self.grid_x[j, i], self.grid_y[j, i])
                    out[j, i] = 0.0 if inside.contains(point) else point.distance(edge)
            self._road_distance = out
        return self._road_distance

    @property
    def slope_deg(self) -> np.ndarray:
        if self._slope is None:
            if self._heightmap is None:
                self._slope = np.zeros(self.shape)
            else:
                z = np.array([[self._heightmap.sample(x, y) for x in self.xs]
                              for y in self.ys])
                dy, dx = np.gradient(z, self.cell)
                self._slope = np.degrees(np.arctan(np.hypot(dx, dy)))
        return self._slope

    def inside(self, polygons: Sequence[Sequence[Sequence[float]]], *,
               shrink: float = 0.0) -> np.ndarray:
        from shapely.geometry import Point
        from shapely.geometry import Polygon as ShapelyPolygon
        from shapely.ops import unary_union
        from shapely.prepared import prep

        shapes = []
        for ring in polygons:
            if len(ring) < 3:
                continue
            poly = ShapelyPolygon([(float(x), float(y)) for x, y in ring])
            if not poly.is_valid:
                poly = poly.buffer(0)
            if shrink:
                poly = poly.buffer(-shrink)
            if not poly.is_empty and poly.area > 0:
                shapes.append(poly)
        if not shapes:
            return np.zeros(self.shape, dtype=bool)

        merged = prep(unary_union(shapes))
        out = np.zeros(self.shape, dtype=bool)
        for j in range(self.shape[0]):
            for i in range(self.shape[1]):
                out[j, i] = merged.contains(Point(self.grid_x[j, i], self.grid_y[j, i]))
        return out

    def noise(self, metres: float, seed: int) -> np.ndarray:
        """Smooth value noise at a given feature size, in scene metres."""
        from scipy.ndimage import zoom

        rng = np.random.default_rng(seed)
        step = max(metres / self.cell, 1.0)
        coarse = rng.random((max(int(self.shape[0] / step) + 2, 2),
                             max(int(self.shape[1] / step) + 2, 2)))
        grown = zoom(coarse, (self.shape[0] / coarse.shape[0] + 1e-9,
                              self.shape[1] / coarse.shape[1] + 1e-9), order=3)
        return grown[: self.shape[0], : self.shape[1]]


def classify(heightmap, options: CoverOptions, *, roads=None,
             plots: Sequence[Sequence[Sequence[float]]] = (),
             bounds: Sequence[float] | None = None) -> CoverMap:
    """Paint the rules onto a grid and return what each cell became.

    Later rules paint over earlier ones — the painter's model, so a config
    reads top to bottom as "the ground is grass, except…". A cell no rule
    claims takes the first surface in the palette, so a palette with no
    ``Everywhere`` rule still produces a complete map rather than holes.
    """
    if not options.surfaces:
        raise ValueError("this cover has no surfaces in its palette")

    if bounds is not None:
        x0, y0, x1, y1 = (float(v) for v in bounds)
    else:
        x0, y0 = heightmap.x0, heightmap.y0
        x1 = x0 + (heightmap.nx - 1) * heightmap.cell
        y1 = y0 + (heightmap.ny - 1) * heightmap.cell
    nx = max(math.ceil((x1 - x0) / options.cell), 1)
    ny = max(math.ceil((y1 - y0) / options.cell), 1)

    board = _Board(x0, y0, nx, ny, options.cell, heightmap=heightmap,
                   roads=roads, plots=plots)
    names = [s.name for s in options.surfaces]
    index = np.zeros((ny, nx), dtype=np.int16)
    for rule in options.rules:
        if rule.name not in names:
            raise KeyError(f"rule paints {rule.name!r}, which is not in the palette")
        index = np.where(rule.mask(board), names.index(rule.name), index)
    return CoverMap(x0, y0, options.cell, index, names)


# ---------------------------------------------------------------------------
# Painting it
# ---------------------------------------------------------------------------


def paint(cover: CoverMap, options: CoverOptions, path: str, *,
          texels_per_metre: float = 8.0, softness: float = 0.35,
          jitter: float = 1.6, seed: int = 0, max_side: int = 8192) -> dict[str, Any]:
    """Bake the whole ground into one image, and write it to ``path``.

    Baked rather than blended at render time, for the same reason the road
    paint is baked into the carriageway: one texture is one sample per pixel
    whatever the class count, it survives a glTF export, and it puts the
    boundaries under the same measured texels-per-metre policy as everything
    else this package draws.

    Three things happen at every texel, and the order of the last two is the
    whole trick.

    **Each class contributes its own tile at its own metric scale.** A gravel
    tile that repeats every 1.5 m and a grass tile every 2.5 m; the class map
    only decides *which*, never how big.

    **The class weights are perturbed by noise before they are compared, not
    after.** Perturbing the result gives a fuzzy edge; perturbing the weight
    gives an interlocking one, and at a 2 m class grid that is the difference
    between a lawn that frays into gravel and a lawn with a 2 m staircase cut
    out of it.

    **Then the winner is chosen by weight times the tile's own brightness**,
    which is height blending in Mishkinis' formulation with luminance standing
    in for a height map. Gravel pokes through the edge of the grass along its
    own silhouette instead of cross-fading with it. ``softness`` widens the
    band where the two mix; at zero the boundary is a hard cut.
    """
    from PIL import Image

    width = round(cover.nx * cover.cell * texels_per_metre)
    height = round(cover.ny * cover.cell * texels_per_metre)
    scale = min(1.0, max_side / max(width, height, 1))
    width, height = max(int(width * scale), 1), max(int(height * scale), 1)
    per_metre = width / max(cover.nx * cover.cell, 1e-6)

    # Scene metres at every texel, from the cover grid's own origin.
    xs = cover.x0 + (np.arange(width) + 0.5) / per_metre
    ys = cover.y0 + (np.arange(height) + 0.5) / per_metre
    grid_x, grid_y = np.meshgrid(xs, ys)

    used = sorted({int(v) for v in np.unique(cover.index)})
    weights = np.zeros((len(used), height, width), dtype=np.float32)
    colours = np.zeros((len(used), height, width, 3), dtype=np.float32)
    rng = np.random.default_rng(seed)

    for slot, class_index in enumerate(used):
        surface = options.surface(cover.names[class_index])
        weights[slot] = _upsampled(cover.index == class_index, height, width)
        colours[slot] = _sampled(surface, grid_x, grid_y)
        # Brightness stands in for the tile's own height. It is not a height
        # map, but on the stochastic ground textures this generates it tracks
        # one closely enough to break the boundary up, which is the job.
        relief = colours[slot].mean(axis=-1)
        weights[slot] *= 1.0 + surface.relief * (relief - float(relief.mean()))

    if jitter > 0 and len(used) > 1:
        # One noise field, added to every class with a different offset: the
        # boundary wanders, and no class is systematically favoured.
        for slot in range(len(used)):
            weights[slot] += jitter * 0.15 * _noise(height, width,
                                                    per_metre / max(jitter, 1e-6),
                                                    int(rng.integers(1 << 30)))

    if softness <= 0:
        chosen = np.argmax(weights, axis=0)
        image = np.take_along_axis(colours, chosen[None, ..., None], axis=0)[0]
    else:
        top = weights.max(axis=0, keepdims=True)
        blend = np.clip((weights - top) / softness + 1.0, 0.0, None)
        blend /= np.maximum(blend.sum(axis=0, keepdims=True), 1e-6)
        image = (colours * blend[..., None]).sum(axis=0)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8)).save(path)
    return {
        "path": path,
        "size": [width, height],
        "texels_per_metre": round(per_metre, 3),
        "metres": [round(cover.nx * cover.cell, 2), round(cover.ny * cover.cell, 2)],
        "origin": [round(cover.x0, 3), round(cover.y0, 3)],
        "surfaces": cover.fractions(),
    }


def _upsampled(mask: np.ndarray, height: int, width: int) -> np.ndarray:
    """The class grid at texel resolution, smoothed across cell boundaries.

    Nearest-neighbour would put a 2 m staircase on every diagonal edge; a
    bilinear ramp is what the jitter and the height comparison then break up.
    """
    from scipy.ndimage import zoom

    grown = zoom(mask.astype(np.float32),
                 (height / mask.shape[0], width / mask.shape[1]), order=1)
    out = np.zeros((height, width), dtype=np.float32)
    take = grown[:height, :width]
    out[: take.shape[0], : take.shape[1]] = take
    return out


def _sampled(surface: Surface, grid_x: np.ndarray, grid_y: np.ndarray) -> np.ndarray:
    """A surface's tile repeated across the map, or its flat colour."""
    if not surface.texture or not os.path.exists(surface.texture):
        return np.broadcast_to(np.asarray(surface.colour, dtype=np.float32),
                               (*grid_x.shape, 3)).copy()
    from PIL import Image

    tile = np.asarray(Image.open(surface.texture).convert("RGB"),
                      dtype=np.float32) / 255.0
    th, tw = tile.shape[:2]
    per_metre_u = tw / max(surface.tile_metres, 1e-6)
    per_metre_v = th / max(surface.tile_metres, 1e-6)
    u = np.mod((grid_x * per_metre_u).astype(np.int64), tw)
    v = np.mod((grid_y * per_metre_v).astype(np.int64), th)
    return tile[v, u]


def _noise(height: int, width: int, cells_per_feature: float, seed: int) -> np.ndarray:
    from scipy.ndimage import zoom

    rng = np.random.default_rng(seed)
    step = max(cells_per_feature, 2.0)
    coarse = rng.random((max(int(height / step) + 2, 2), max(int(width / step) + 2, 2)))
    grown = zoom(coarse, (height / coarse.shape[0] + 1e-9,
                          width / coarse.shape[1] + 1e-9), order=3)
    field = grown[:height, :width]
    return (field - field.mean()) / max(field.std(), 1e-6)
