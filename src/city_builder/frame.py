"""Scene coordinate frame: WGS84 → local metres."""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6371000.0


class LocalFrame:
    """Equirectangular projection about a reference point.

    Every layer this package emits — road surfaces, markings, ground — shares
    one frame so they land on top of each other, and so anything produced
    elsewhere (buildings, props) can be placed in it by using the same anchor.
    """

    def __init__(self, ref_lat: float, ref_lon: float):
        self.ref_lat = ref_lat
        self.ref_lon = ref_lon
        self._ref_lat_rad = math.radians(ref_lat)
        self._ref_lon_rad = math.radians(ref_lon)
        self._cos_ref_lat = math.cos(self._ref_lat_rad)

    def to_local(self, lat: float, lon: float) -> tuple[float, float]:
        x = EARTH_RADIUS_M * (math.radians(lon) - self._ref_lon_rad) * self._cos_ref_lat
        y = EARTH_RADIUS_M * (math.radians(lat) - self._ref_lat_rad)
        return x, y

    def to_wgs84(self, x: float, y: float) -> tuple[float, float]:
        lat = math.degrees(y / EARTH_RADIUS_M + self._ref_lat_rad)
        lon = math.degrees(x / (EARTH_RADIUS_M * self._cos_ref_lat) + self._ref_lon_rad)
        return lat, lon

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"LocalFrame(ref_lat={self.ref_lat!r}, ref_lon={self.ref_lon!r})"
