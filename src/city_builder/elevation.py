"""What shape the ground is, between the roads — measured or invented.

Two sources for one thing. A published elevation model where one covers the
map, and a procedural relief where none does or none is wanted, and both come
out as the same object: a grid handed to
:func:`city_builder.ground.build_heightmap` as ``guidance``, whose *curvature*
is copied while its heights are ignored.

That last point is what makes the two interchangeable. The road elevations stay
hard constraints either way, so neither source can move a carriageway; all
either one decides is what the ground does between them.

A Lanelet2 map knows the height of its carriageway and nothing else, so
:mod:`city_builder.ground` reconstructs the rest by interpolation. Where a
national elevation model exists, most of that is invention where a measurement
was available — and where none does, inventing it deliberately beats letting
the interpolation invent a flat one by default.

Measured
--------

This fetches a published model as ordinary XYZ tiles. The only source wired up is Japan's
Geospatial Information Authority, because the maps this package is aimed at —
Autoware's — are largely Japanese, and because the GSI tiles need no key, no
registration and no account: they are the same tiles the 地理院地図 viewer uses,
under the 国土地理院コンテンツ利用規約.

**The two elevations will not agree, and the disagreement is the point.**
Measured over the Kashiwanoha map: after removing a constant 16.31 m datum
offset the residual between the lanelet z and the 5 m lidar model is p90
3.19 m — the lanelets claim 5.97 m of relief over 163 x 143 m where the lidar
sees 1.36 m. That is not sample noise; road vertices a metre apart in plan
agree to a centimetre. It is a smooth, low-frequency tilt.

So neither source is simply right. The lanelets are exact locally and drift
globally; the elevation model is coarse locally and correct globally. What this
module returns is therefore a *prior*, to be given to
:func:`city_builder.ground.build_heightmap` as a screening term with the road
elevations still hard constraints — the roads keep their own heights, and the
prior shapes everything between them.
"""

from __future__ import annotations

import io
import math
import os
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

# GSI publishes several elevation products on the same scheme, coarsest last.
# The 5 m airborne-lidar model is the one worth having and does not cover the
# whole country, so the list is tried in order and the first hit wins.
GSI_TILES = (
    ("dem5a_png", 15),  # DEM5A, airborne lidar, 5 m
    ("dem5b_png", 15),  # DEM5B, photogrammetry, 5 m
    ("dem5c_png", 15),
    ("dem_png", 14),    # DEM10B, 10 m, nationwide
)

NO_DATA = -83886.08  # what (128, 0, 0) decodes to under the GSI encoding


# ---------------------------------------------------------------------------
# Inventing one instead
# ---------------------------------------------------------------------------


@dataclass
class Relief:
    """Terrain to invent where none is published, stated as conditions.

    The knobs are the ones somebody asking for terrain actually has an opinion
    about: how much it moves, over what distance, and how rough it is up close.

    Note what is *not* here: an overall grade. A constant slope has no
    curvature, and curvature is all that is taken from this — so a tilt would
    be silently discarded. It is also the one part of the terrain the map
    already knows, since the road elevations carry it exactly.
    """

    amplitude: float = 2.5      # metres between the lowest and highest ground
    metres: float = 140.0       # how far across the largest feature is
    octaves: int = 4            # how many halvings of that size are added
    roughness: float = 0.45     # how much each octave keeps of the last
    warp: float = 0.35          # bends the features; 0 is plain, dull fBm
    seed: int = 0


class InventedTiles:
    """A tile source that generates the terrain instead of downloading it.

    Deliberately shaped as a *tile source*. Once the invented terrain arrives
    through the same door as a published one, everything downstream — the datum
    solve, the coverage report, the caching, the guidance term — is one code
    path with two providers behind it, and a caller swaps them with a config
    line. It also means an invented terrain can be written out and looked at as
    ordinary elevation tiles.

    Seamless across tile boundaries by construction: the noise is a function of
    absolute position on the globe, not of an array the size of one tile, so
    neighbouring tiles agree along their edge without being asked to.
    """

    #: Metres per radian on the sphere.
    RADIUS = 6378137.0

    def __init__(self, relief: Relief | None = None, *, zoom: int = 15,
                 size: int = 256, latitude: float = 35.0):
        self.relief = relief or Relief()
        self.zoom = zoom
        self.size = size
        # A *constant* for the whole source, and that matters. Mercator metres
        # are stretched by 1/cos(lat), so undoing that per tile would give two
        # vertically adjacent tiles different scales and put a step along their
        # seam — the artefact this source exists not to have. One factor,
        # correct at the map's own latitude, is continuous everywhere.
        self.latitude = latitude
        self.name = "invented"
        self.fetched = 0

    def grid(self, tx: int, ty: int) -> np.ndarray:
        step = (np.arange(self.size) + 0.5) / self.size
        n = 2.0 ** self.zoom
        lon = ((tx + step) / n * 2.0 - 1.0) * math.pi
        lat = np.arctan(np.sinh(math.pi * (1.0 - 2.0 * (ty + step) / n)))

        # Ground metres, from coordinates that are continuous across any tile
        # boundary because they are functions of position on the globe.
        east = self.RADIUS * math.cos(math.radians(self.latitude)) * lon
        north = self.RADIUS * lat
        world_x, world_y = np.meshgrid(east, north)

        self.fetched += 1
        return _fbm(world_x, world_y, self.relief) * _scale(self.relief)


_SCALES: dict[tuple, float] = {}


def _scale(relief: Relief) -> float:
    """What to multiply the raw field by to get ``amplitude`` peak to trough.

    Measured once over a wide reference patch rather than over the tile in
    hand. Normalising each tile by its own range would be the obvious thing and
    is wrong: two neighbouring tiles would be divided by different numbers and
    the terrain would step at every tile boundary — the artefact this source is
    built to not have.
    """
    key = (relief.metres, relief.octaves, relief.roughness, relief.warp, relief.seed)
    if key not in _SCALES:
        step = np.linspace(0.0, relief.metres * 24.0, 192)
        x, y = np.meshgrid(step, step)
        field = _fbm(x, y, relief)
        spread = float(field.max() - field.min())
        _SCALES[key] = (1.0 / spread) if spread > 1e-9 else 0.0
    return _SCALES[key] * relief.amplitude


def _lattice(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """A repeatable random value per integer lattice point, in [0, 1).

    Hashed rather than drawn from a generator, because the whole point is that
    the value at a position does not depend on which tile asked for it.
    """
    h = (ix.astype(np.int64) * 374761393 + iy.astype(np.int64) * 668265263
         + seed * 1442695041) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 1274126177) & 0xFFFFFFFF
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF)


def _value_noise(x: np.ndarray, y: np.ndarray, cell: float, seed: int) -> np.ndarray:
    """Smooth noise sampled at absolute positions, one feature per ``cell``."""
    fx, fy = x / cell, y / cell
    ix, iy = np.floor(fx), np.floor(fy)
    tx, ty = fx - ix, fy - iy
    # Quintic smoothstep: C2, so the second derivative this is read through
    # does not step at every lattice line.
    sx = tx * tx * tx * (tx * (tx * 6 - 15) + 10)
    sy = ty * ty * ty * (ty * (ty * 6 - 15) + 10)
    c00 = _lattice(ix, iy, seed)
    c10 = _lattice(ix + 1, iy, seed)
    c01 = _lattice(ix, iy + 1, seed)
    c11 = _lattice(ix + 1, iy + 1, seed)
    return ((c00 * (1 - sx) + c10 * sx) * (1 - sy)
            + (c01 * (1 - sx) + c11 * sx) * sy) * 2.0 - 1.0


def _fbm(x: np.ndarray, y: np.ndarray, relief: Relief) -> np.ndarray:
    """Fractional Brownian motion, optionally warped through itself."""
    if relief.warp:
        # Bending the sample positions is what turns the round blobs of plain
        # fBm into ridges and hollows that look like ground.
        reach = relief.metres * relief.warp
        x = x + reach * _value_noise(x, y, relief.metres * 2.0, relief.seed + 101)
        y = y + reach * _value_noise(x, y, relief.metres * 2.0, relief.seed + 202)

    total = np.zeros_like(x, dtype=float)
    amplitude, cell = 1.0, max(relief.metres, 1e-6)
    for octave in range(max(relief.octaves, 1)):
        total += amplitude * _value_noise(x, y, cell, relief.seed + 7 * octave)
        amplitude *= relief.roughness
        cell /= 2.0
    return total



@dataclass
class Coverage:
    """What came back, so a caller can decide whether to trust it."""

    source: str
    zoom: int
    metres_per_pixel: float
    covered: float          # fraction of the grid the model reaches
    datum_offset: float     # metres to add to scene z to land in the model's datum
    residual_p90: float     # how far the two still disagree after that
    tiles: int              # HTTP requests it took

    def to_json(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "zoom": self.zoom,
            "metres_per_pixel": round(self.metres_per_pixel, 3),
            "covered": round(self.covered, 4),
            "datum_offset_m": round(self.datum_offset, 3),
            "residual_p90_m": round(self.residual_p90, 3),
            "tiles": self.tiles,
        }


def _tile_of(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    """Fractional slippy-map tile coordinates."""
    n = 2.0 ** zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
    return x, y


def _decode(image) -> np.ndarray:
    """GSI's 24-bit elevation PNG: ``h = (R*65536 + G*256 + B) * 0.01`` metres.

    The top bit is a sign — values at or above 2^23 are negative — and
    ``(128, 0, 0)`` exactly is the no-data marker, which decodes to a very
    negative number rather than to anything plausible.
    """
    raw = np.asarray(image.convert("RGB")).astype(np.int64)
    packed = raw[..., 0] * 65536 + raw[..., 1] * 256 + raw[..., 2]
    metres = np.where(packed < 8388608, packed, packed - 16777216) * 0.01
    missing = (raw[..., 0] == 128) & (raw[..., 1] == 0) & (raw[..., 2] == 0)
    return np.where(missing, np.nan, metres)


class WebTiles:
    """A tile source that downloads. Fetches and remembers: one request each."""

    def __init__(self, template: str, zoom: int, cache_dir: str | None = None,
                 timeout: float = 30.0):
        self.name = template
        self.template = template
        self.zoom = zoom
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.held: dict[tuple[int, int], np.ndarray | None] = {}
        self.fetched = 0

    def grid(self, tx: int, ty: int) -> np.ndarray | None:
        if (tx, ty) in self.held:
            return self.held[(tx, ty)]
        from PIL import Image

        path = (os.path.join(self.cache_dir, f"{self.template}_{self.zoom}_{tx}_{ty}.png")
                if self.cache_dir else None)
        data: bytes | None = None
        if path and os.path.exists(path):
            with open(path, "rb") as handle:
                data = handle.read()
        else:
            url = f"https://cyberjapandata.gsi.go.jp/xyz/{self.template}/{self.zoom}/{tx}/{ty}.png"
            try:
                with urllib.request.urlopen(url, timeout=self.timeout) as response:
                    data = response.read()
                self.fetched += 1
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError):
                # A 404 is the ordinary answer for a tile outside the product's
                # coverage, not a failure, so it is recorded as absent.
                self.held[(tx, ty)] = None
                return None
            if path:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(path, "wb") as handle:
                    handle.write(data)
        try:
            self.held[(tx, ty)] = _decode(Image.open(io.BytesIO(data)))
        except Exception:  # noqa: BLE001 - a corrupt tile is an absent tile
            self.held[(tx, ty)] = None
        return self.held[(tx, ty)]


def web_sources(cache_dir: str | None = None, timeout: float = 30.0,
                specs: tuple[tuple[str, int], ...] = GSI_TILES) -> list[WebTiles]:
    """The published sources, finest first."""
    return [WebTiles(template, zoom, cache_dir, timeout) for template, zoom in specs]


def sample_grid(frame, x0: float, y0: float, nx: int, ny: int, cell: float, *,
                sources: Sequence[Any] | None = None,
                cache_dir: str | None = None,
                timeout: float = 30.0) -> tuple[np.ndarray, str, int, int] | None:
    """An elevation model over a heightmap's grid, in the model's own datum.

    Returns ``(z, source, zoom, tiles_read)`` with NaN where the model has no
    data, or ``None`` if no source covers the map at all.

    ``sources`` is anything with ``name``, ``zoom`` and ``grid(tx, ty)``, which
    is how a downloaded model and an invented one become interchangeable —
    :class:`WebTiles` and :class:`InventedTiles` both qualify. They are tried
    in order and the first that reaches more than half the grid is taken: a
    source covering a corner is worse than a coarser one covering everything,
    because a coverage boundary running through the scene puts a step in the
    ground. Put an :class:`InventedTiles` last and it becomes the fallback for
    a map nobody has surveyed.
    """
    xs = x0 + np.arange(nx) * cell
    ys = y0 + np.arange(ny) * cell
    grid_x, grid_y = np.meshgrid(xs, ys)
    lat = np.empty_like(grid_x)
    lon = np.empty_like(grid_x)
    for j in range(ny):
        for i in range(nx):
            lat[j, i], lon[j, i] = frame.to_wgs84(grid_x[j, i], grid_y[j, i])

    for tiles in (sources if sources is not None else web_sources(cache_dir, timeout)):
        out = np.full((ny, nx), np.nan)
        n = 2 ** tiles.zoom
        fx = (lon + 180.0) / 360.0 * n
        fy = (1.0 - np.arcsinh(np.tan(np.radians(lat))) / np.pi) / 2.0 * n
        tx, ty = np.floor(fx).astype(int), np.floor(fy).astype(int)
        for key in {(int(a), int(b)) for a, b in zip(tx.ravel(), ty.ravel())}:
            grid = tiles.grid(*key)
            if grid is None:
                continue
            here = (tx == key[0]) & (ty == key[1])
            px = np.clip(((fx[here] - key[0]) * grid.shape[1]).astype(int), 0, grid.shape[1] - 1)
            py = np.clip(((fy[here] - key[1]) * grid.shape[0]).astype(int), 0, grid.shape[0] - 1)
            out[here] = grid[py, px]
        if np.isfinite(out).mean() > 0.5:
            return out, tiles.name, tiles.zoom, tiles.fetched
    return None


def align(model: np.ndarray, samples: np.ndarray, frame_x0: float, frame_y0: float,
          cell: float) -> tuple[float, float, float]:
    """Solve the datum offset between a model and the map's own elevations.

    Returns ``(offset, median_absolute_residual, p90_residual)``. The offset is
    a **median**, not a mean: the two datums differ by a constant, and the
    places where they differ by more than that — an embankment, a building the
    model has and the map does not — are outliers that a mean would chase.

    It is solved rather than assumed because it is different on every map, and
    because a vertical datum mismatch is the classic silent failure here: the
    prior lands tens of metres out and quietly flattens everything.
    """
    if not len(samples):
        raise ValueError("no map elevations to align against")
    ny, nx = model.shape
    ix = np.clip(((samples[:, 0] - frame_x0) / cell).round().astype(int), 0, nx - 1)
    iy = np.clip(((samples[:, 1] - frame_y0) / cell).round().astype(int), 0, ny - 1)
    got = model[iy, ix]
    usable = np.isfinite(got)
    if not usable.any():
        raise ValueError("the elevation model does not reach any of the map's roads")
    difference = got[usable] - samples[usable, 2]
    offset = float(np.median(difference))
    residual = np.abs(difference - offset)
    return offset, float(np.median(residual)), float(np.percentile(residual, 90))


def prior_for(frame, x0: float, y0: float, nx: int, ny: int, cell: float,
              samples: Sequence[Sequence[float]], *,
              sources: Sequence[Any] | None = None,
              cache_dir: str | None = None,
              timeout: float = 30.0) -> tuple[np.ndarray, Coverage] | None:
    """An elevation model over a grid, brought into the scene's own datum.

    The whole module in one call: read it, solve the offset, subtract it, and
    hand back something that goes straight into ``build_heightmap`` as
    ``guidance=``. ``None`` when nothing covers the map — the ordinary answer
    outside Japan with no invented source in the list, and not an error.
    """
    found = sample_grid(frame, x0, y0, nx, ny, cell, sources=sources,
                        cache_dir=cache_dir, timeout=timeout)
    if found is None:
        return None
    model, source, zoom, fetched = found
    offset, _median, p90 = align(model, np.asarray(samples, dtype=float), x0, y0, cell)
    lat, _lon = frame.to_wgs84(x0, y0)
    return model - offset, Coverage(
        source=source, zoom=zoom,
        # A slippy tile is 256 px across, so this is the ground size of one.
        metres_per_pixel=156543.03392 * math.cos(math.radians(lat)) / 2 ** zoom,
        covered=float(np.isfinite(model).mean()), datum_offset=offset,
        residual_p90=p90, tiles=fetched)
