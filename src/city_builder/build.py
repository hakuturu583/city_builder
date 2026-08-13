"""End-to-end: Lanelet2 map → ground and road-surface meshes."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from . import buildings as buildings_module
from . import classes, facade_layout, lanelet, scene
from . import ground as ground_module
from .buildings import BuildingOptions
from .frame import LocalFrame
from .surfaces import SurfaceOptions, extract


@dataclass
class BuildResult:
    frame: LocalFrame
    groups: dict[str, list]
    heightmap: ground_module.HeightMap | None
    elevated: set[int]
    z_datum: float
    stats: dict[str, Any] = field(default_factory=dict)
    # One record per generated building, in the order of ``groups["Buildings"]``.
    # Carries the floor count, which decides which facade sheet it may wear.
    plots: list[dict[str, Any]] = field(default_factory=list)


def build_city(
    input_path: str,
    *,
    ref_lat: float | None = None,
    ref_lon: float | None = None,
    projector: str = "utm",
    z_datum: float | None = None,
    z_offset: float = 0.0,
    surface_options: SurfaceOptions | None = None,
    ground: bool = True,
    cell: float = ground_module.DEFAULT_CELL,
    smooth: float = ground_module.DEFAULT_SMOOTH,
    z_gap: float = ground_module.DEFAULT_Z_GAP,
    min_overlap: float = ground_module.DEFAULT_MIN_OVERLAP,
    clearance: float = ground_module.DEFAULT_CLEARANCE,
    ground_drop: float = 0.05,
    fill_island: float = 0.0,
    buildings: bool = False,
    building_options: BuildingOptions | None = None,
    verbose: bool = True,
) -> BuildResult:
    """Read a map and produce every surface, without touching Blender.

    Split out from the scene building so the geometry can be inspected, tested
    or fed somewhere other than Blender.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Lanelet2 map not found: {input_path}")

    ll2, projection, lmap = lanelet.load_map(
        input_path, projector=projector, origin_lat=ref_lat or 0.0, origin_lon=ref_lon or 0.0
    )
    if ref_lat is None or ref_lon is None:
        ref_lat, ref_lon = lanelet.map_centroid(ll2, projection, lmap)
        if verbose:
            print(f"[build] anchoring the scene at the map centroid {ref_lat:.7f},{ref_lon:.7f}")

    frame = LocalFrame(ref_lat, ref_lon)
    groups = extract(ll2, projection, lmap, frame, surface_options)
    datum = lanelet.apply_z_datum(groups, z_datum, z_offset)

    stats = {name: len(shapes) for name, shapes in groups.items()}
    if verbose:
        print(f"[build] surfaces: {stats}")
        print(f"[build] z datum {datum:.2f} m → scene ground at {z_offset:.2f} m")

    elevated: set[int] = set()
    heightmap = None
    if ground:
        surfaces = list(groups.get("Roads", [])) + list(groups.get("Junctions", []))
        adjacency = lanelet.build_adjacency(lanelet.lanelet_end_keys(lmap))
        elevated, heightmap = ground_module.classify(
            surfaces, adjacency,
            cell=cell, z_gap=z_gap, min_overlap=min_overlap,
            clearance=clearance, smooth=smooth, drop=ground_drop,
        )
        if heightmap is None:
            raise RuntimeError("no ground-level road surfaces found; cannot build a ground surface")

        mesh, road_union = ground_module.build_mesh(heightmap, surfaces, elevated,
                                                    fill_island=fill_island, drop=ground_drop,
                                                    return_road_union=True)
        groups["Ground"] = [mesh]

        measured = float((heightmap.support == 0).mean() * 100)
        stats.update({
            "elevated_lanelets": len(elevated),
            "ground_faces": len(mesh.faces),
            "cells_measured_pct": round(measured, 1),
        })
        if verbose:
            print(f"[build] elevated lanelets: {len(elevated)} / {len(surfaces)}")
            print(f"[build] ground: {heightmap.nx}x{heightmap.ny} @ {cell:.1f} m, "
                  f"{measured:.1f}% of cells measured, {len(mesh.faces)} faces")

    plots: list[dict[str, Any]] = []
    if buildings:
        if heightmap is None:
            raise RuntimeError("buildings need the ground; do not pass ground=False with buildings=True")
        keep_clear = buildings_module.exclusion_zone(groups) or road_union
        built = buildings_module.generate(heightmap, keep_clear, building_options)
        if built["Buildings"]:
            groups["Buildings"] = built["Buildings"]
            groups["Roofs"] = built["Roofs"]
        plots = built["plots"]
        stats["buildings"] = len(built["Buildings"])
        if verbose:
            heights = [p["height"] for p in plots]
            span = f"{min(heights):.0f}-{max(heights):.0f} m" if heights else "none"
            floors = sorted({p["floors"] for p in plots})
            print(f"[build] buildings: {len(built['Buildings'])} on the open ground, {span} tall")
            if floors:
                print(f"[build] floor counts: {floors[0]}-{floors[-1]} "
                      f"({len(floors)} distinct, one facade sheet family each)")

    return BuildResult(frame, groups, heightmap, elevated, datum, stats, plots)


def write_manifest(result: BuildResult, path: str) -> None:
    """Describe every surface: what it is, and whether its colour may change.

    A texturing pass reads this to decide where it may invent and where it must
    not. The same information is on the objects themselves (``cb_class`` /
    ``cb_paint``, a ``cb_mask`` colour attribute, and ``pass_index``); this file
    is for a consumer that wants the whole picture before opening the scene.
    """
    payload = classes.manifest()
    payload["scene"] = {
        "ref_lat": result.frame.ref_lat,
        "ref_lon": result.frame.ref_lon,
        "z_datum": result.z_datum,
    }
    payload["groups"] = [
        {
            **classes.get(name).to_json(),
            "shapes": len(shapes),
            "present": True,
        }
        for name, shapes in result.groups.items()
    ]
    payload["preserve_groups"] = [
        name for name in result.groups if classes.get(name).preserved
    ]

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[build] wrote {path}")


def write_heightmap(result: BuildResult, path: str) -> None:
    """Dump the heightmap so other tools can place things on this ground."""
    if result.heightmap is None:
        raise ValueError("this build has no heightmap")
    payload = {
        "meta": {
            "ref_lat": result.frame.ref_lat,
            "ref_lon": result.frame.ref_lon,
            "z_datum": result.z_datum,
            "elevated_lanelets": sorted(result.elevated),
            "stats": result.stats,
        },
        "heightmap": result.heightmap.to_json(),
    }
    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    print(f"[build] wrote {path}")


def build_scene(result: BuildResult, *, blend: str | None = None, glb: str | None = None,
                ground_texture: str | None = None, tile_metres: float = 12.0,
                facade_dir: str | None = None, verbose: bool = True) -> None:
    """Build the result into Blender and export it.

    ``ground_texture`` is a tile image to repeat across the ground. Only the
    ground: every lanelet-derived surface keeps the material it was built with,
    because the map already says what those look like.
    """
    scene.clear_scene()
    objects = scene.build(result.groups, verbose=verbose)

    if facade_dir:
        sheets = sorted(
            os.path.join(facade_dir, f) for f in os.listdir(facade_dir) if f.endswith(".png")
        )
        walls = objects.get("Buildings")
        if sheets and walls is not None:
            counts = scene.build.face_counts["Buildings"]
            floors = [plot["floors"] for plot in result.plots] or None
            scene.apply_facade_sheets(walls, sheets, counts, floors)
            if verbose:
                matched = sum(1 for path in sheets if facade_layout.sheet_floors(path) is not None)
                print(f"[scene] Buildings: {len(sheets)} facade sheet(s) "
                      f"({matched} floor-matched) across {len(counts)} buildings")

    if ground_texture:
        ground = objects.get("Ground")
        if ground is None:
            raise RuntimeError("no Ground object to texture; build with ground=True")
        scene.apply_tiled_texture(ground, ground_texture, tile_metres)
        if verbose:
            print(f"[scene] Ground: tiled {os.path.basename(ground_texture)} every {tile_metres:g} m")
    if blend:
        scene.save(blend)
    if glb:
        scene.export_glb(glb)


def options_from_kwargs(**kwargs) -> SurfaceOptions:
    known = {f for f in asdict(SurfaceOptions())}
    return SurfaceOptions(**{k: v for k, v in kwargs.items() if k in known and v is not None})
