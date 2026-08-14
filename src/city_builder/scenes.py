"""Built scenes an agent can keep referring to, and what can be said about them.

Two things a caller driving this from the outside needs that the CLI does not.

**A handle.** Building a map takes twenty seconds and produces geometry that
export, measurement and rendering all want. Passing the map path to every call
would rebuild it every time, so a build is kept and named, and later calls
refer to it.

**A survey.** An agent cannot look at a scene the way a person looks at a
render; it needs the scene to describe itself. :func:`survey` is the session's
worth of debugging distilled into numbers — how much of the ground is measured
rather than guessed, how much of the carriageway is holes, how many lanelets
are elevated and how far above what — because those are the questions that
turned out to matter, and each of them was a bug before it was a metric.

Blender is a singleton: one process holds one scene, so a handle stores the
geometry (which is plain numpy and shapely) and anything wanting Blender
rebuilds into it on demand. That is why exporting twice costs twice.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any

import numpy as np

from .build import BuildResult


@dataclass
class Scene:
    """One built map, kept so it can be exported, measured and rendered."""

    name: str
    map_path: str
    result: BuildResult
    buildings: bool
    options: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        heightmap = self.result.heightmap
        return {
            "scene": self.name,
            "map": self.map_path,
            "buildings": self.buildings,
            "groups": {name: len(shapes) for name, shapes in self.result.groups.items()},
            "elevated_lanelets": len(self.result.elevated),
            "ground_cells_measured_pct": self.result.stats.get("cells_measured_pct"),
            "ground_grid": None if heightmap is None else f"{heightmap.nx}x{heightmap.ny}",
            "marking_pages": len(self.result.marking_pages),
            "anchor": [self.result.frame.ref_lat, self.result.frame.ref_lon],
        }


class SceneStore:
    """The scenes this process is holding."""

    def __init__(self) -> None:
        self._scenes: dict[str, Scene] = {}
        self._next = 1

    def add(self, map_path: str, result: BuildResult, *, buildings: bool,
            options: dict[str, Any] | None = None) -> Scene:
        name = f"scene-{self._next}"
        self._next += 1
        scene = Scene(name, map_path, result, buildings, options or {})
        self._scenes[name] = scene
        return scene

    def get(self, name: str) -> Scene:
        try:
            return self._scenes[name]
        except KeyError:
            known = ", ".join(sorted(self._scenes)) or "none"
            raise KeyError(f"no scene named {name!r}; built so far: {known}") from None

    def drop(self, name: str) -> None:
        self.get(name)
        del self._scenes[name]

    def all(self) -> list[Scene]:
        return list(self._scenes.values())


# ---------------------------------------------------------------------------
# What can be said about a scene without looking at it
# ---------------------------------------------------------------------------


def _footprint_union(shapes):
    from shapely.ops import unary_union

    from .viaduct import _footprints

    polygons = _footprints(shapes, None)
    return unary_union(polygons) if polygons else None


def _mesh_polygons(meshes):
    from shapely.geometry import Polygon as ShapelyPolygon

    out = []
    for mesh in meshes:
        for face in mesh.faces:
            ring = [(mesh.vertices[i][0], mesh.vertices[i][1]) for i in face]
            polygon = ShapelyPolygon(ring)
            if polygon.is_valid and polygon.area > 1e-9:
                out.append(polygon)
    return out


def carriageway_holes(result: BuildResult, *, gap: float = 0.8,
                      narrow: float = 1.0) -> dict[str, Any]:
    """Holes left in the drivable surface, per level and split by width.

    Per level because a viaduct and the street under it share no plan view: a
    single union puts the space beside the viaduct in the same bucket as a slot
    through the deck.

    Split by width, not by area, because width is what makes a hole a defect. A
    forty-metre seam a handspan wide is eight square metres and a wheel drops
    into it; a roundabout's central island is three hundred and is meant to be
    there. So a hole that nowhere admits a circle of diameter ``narrow`` is a
    *seam*, and the rest are *openings* — the seam count is the one to drive to
    zero.
    """
    from shapely.geometry import Polygon as ShapelyPolygon
    from shapely.ops import unary_union

    carriageway = [s for name in ("Roads", "Junctions")
                   for s in result.groups.get(name, ()) if hasattr(s, "ring")]
    if not carriageway:
        return {"levels": {}, "note": "no carriageway in this scene"}

    patches = _mesh_polygons(result.groups.get("RoadInfill", []))
    levels = {
        "at_grade": [s for s in carriageway if s.id not in result.elevated],
        "elevated": [s for s in carriageway if s.id in result.elevated],
    }

    report: dict[str, Any] = {}
    for label, shapes in levels.items():
        covered = _footprint_union(shapes)
        if covered is None:
            continue
        area = unary_union([covered, *patches]) if patches else covered
        closed = area.buffer(gap, join_style=2).buffer(-gap, join_style=2)
        seams, openings = [], []
        for polygon in getattr(closed, "geoms", [closed]):
            if polygon.geom_type != "Polygon":
                continue
            for ring in polygon.interiors:
                hole = ShapelyPolygon(ring)
                target = seams if hole.buffer(-narrow / 2.0).is_empty else openings
                target.append(hole.area)
        report[label] = {
            "surface_m2": round(float(covered.area), 1),
            "seams": len(seams),
            "seam_area_m2": round(float(np.sum(seams)), 1) if seams else 0.0,
            "openings": len(openings),
            "largest_opening_m2": round(float(max(openings)), 1) if openings else 0.0,
        }
    return {
        "levels": report,
        "patches": len(result.groups.get("RoadInfill", [])),
        "note": (f"a seam is a hole nowhere {narrow} m wide — a wheel drops into it, and it "
                 "should be zero; an opening is a traffic island or a gap that belongs"),
    }


def elevation_report(result: BuildResult) -> dict[str, Any]:
    """How high the elevated lanelets run, and over what."""
    from .viaduct import centreline

    if result.heightmap is None:
        return {"elevated": 0, "note": "no ground was reconstructed"}

    clearances = []
    for name in ("Roads", "Junctions"):
        for ribbon in result.groups.get(name, ()):
            if ribbon.id not in result.elevated:
                continue
            line = centreline(ribbon)
            if line:
                clearances.append(float(np.median(
                    [p[2] - result.heightmap.sample(p[0], p[1]) for p in line])))
    if not clearances:
        return {"elevated": 0}

    values = np.array(clearances)
    return {
        "elevated": len(values),
        "clearance_m": {
            "min": round(float(values.min()), 2),
            "median": round(float(np.median(values)), 2),
            "max": round(float(values.max()), 2),
        },
        "decks": len(result.groups.get("ViaductDecks", [])),
        "parapets": len(result.groups.get("ViaductParapets", [])),
        "piers": len(result.groups.get("ViaductPiers", [])),
    }


def building_report(result: BuildResult) -> dict[str, Any]:
    if not result.plots:
        return {"buildings": 0}
    heights = np.array([p["height"] for p in result.plots])
    floors = sorted({p["floors"] for p in result.plots})
    return {
        "buildings": len(result.plots),
        "height_m": {"min": round(float(heights.min()), 1),
                     "median": round(float(np.median(heights)), 1),
                     "max": round(float(heights.max()), 1)},
        "floor_counts": floors,
        "note": "one facade sheet family is needed per floor count",
    }


def route_report(result: BuildResult, map_path: str, *, seed: int = 0) -> dict[str, Any]:
    """How far a camera can drive without repeating a lanelet."""
    from . import lanelet, route

    _ll2, _projection, lmap = lanelet.load_map(
        map_path, projector="utm",
        origin_lat=result.frame.ref_lat, origin_lon=result.frame.ref_lon)
    lines = {}
    for name in ("Roads", "Junctions"):
        for ribbon in result.groups.get(name, ()):
            line = route.centreline(ribbon)
            if len(line) >= 2:
                lines[ribbon.id] = line
    if not lines:
        return {"lanelets": 0, "length_m": 0.0}

    chain = route.longest_route(route.successors(lanelet.lanelet_end_keys(lmap)),
                                lines, seed=seed)
    if not chain:
        return {"lanelets": 0, "length_m": 0.0, "note": "no drivable chain"}
    points = route.route_polyline(chain, lines)
    length = sum(math.dist(a[:2], b[:2]) for a, b in pairwise(points))
    return {
        "lanelets": len(chain),
        "length_m": round(length, 1),
        "seconds_at_11ms": round(length / 11.0, 1),
        "z_range_m": [round(min(p[2] for p in points), 2), round(max(p[2] for p in points), 2)],
    }


def survey(scene: Scene) -> dict[str, Any]:
    """Everything worth knowing about a built scene, in numbers.

    Each of these was a bug before it was a metric, which is why they are the
    ones an agent is handed rather than a face count.
    """
    return {
        **scene.summary(),
        "carriageway": carriageway_holes(scene.result),
        "elevation": elevation_report(scene.result),
        "buildings_detail": building_report(scene.result),
        "route": route_report(scene.result, scene.map_path),
    }
