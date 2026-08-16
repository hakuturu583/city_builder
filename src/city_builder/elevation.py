"""A measured elevation model to put under a map, where one is published.

A Lanelet2 map knows the height of its carriageway and nothing else, so
:mod:`city_builder.ground` reconstructs the rest by interpolation. Where a
national elevation model exists, most of that is invention where a measurement
was available.

This fetches one as ordinary XYZ tiles. The only source wired up is Japan's
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


class _Tiles:
    """Fetches and remembers tiles. One HTTP request per tile per run."""

    def __init__(self, template: str, zoom: int, cache_dir: str | None, timeout: float):
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


def sample_grid(frame, x0: float, y0: float, nx: int, ny: int, cell: float, *,
                sources: tuple[tuple[str, int], ...] = GSI_TILES,
                cache_dir: str | None = None,
                timeout: float = 30.0) -> tuple[np.ndarray, str, int, int] | None:
    """The published elevation over a heightmap's grid, in the model's own datum.

    Returns ``(z, source, zoom, tiles_fetched)`` with NaN where the model has
    no data, or
    ``None`` if no source covers the map at all. Sources are tried finest
    first, and the first one that reaches more than half the grid is taken —
    a source that covers a corner is worse than a coarser one that covers
    everything, because a prior with a coverage boundary running through the
    scene puts a step in the ground.
    """
    xs = x0 + np.arange(nx) * cell
    ys = y0 + np.arange(ny) * cell
    grid_x, grid_y = np.meshgrid(xs, ys)
    lat = np.empty_like(grid_x)
    lon = np.empty_like(grid_x)
    for j in range(ny):
        for i in range(nx):
            lat[j, i], lon[j, i] = frame.to_wgs84(grid_x[j, i], grid_y[j, i])

    for template, zoom in sources:
        tiles = _Tiles(template, zoom, cache_dir, timeout)
        out = np.full((ny, nx), np.nan)
        n = 2 ** zoom
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
            return out, template, zoom, tiles.fetched
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
              sources: tuple[tuple[str, int], ...] = GSI_TILES,
              cache_dir: str | None = None,
              timeout: float = 30.0) -> tuple[np.ndarray, Coverage] | None:
    """A published elevation model over a grid, brought into the scene's datum.

    This is the whole module in one call: fetch, solve the offset, subtract it,
    and hand back something that can be added to ``build_heightmap`` as
    ``prior=``. ``None`` when nothing covers the map, which is the ordinary
    answer outside Japan and is not an error.
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
