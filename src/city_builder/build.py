"""End-to-end: Lanelet2 map → ground and road-surface meshes."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Any

from . import ground as ground_module
from . import lanelet, scene
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

        mesh = ground_module.build_mesh(heightmap, surfaces, elevated,
                                        fill_island=fill_island, drop=ground_drop)
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

    return BuildResult(frame, groups, heightmap, elevated, datum, stats)


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
                verbose: bool = True) -> None:
    """Build the result into Blender and export it."""
    scene.clear_scene()
    scene.build(result.groups, verbose=verbose)
    if blend:
        scene.save(blend)
    if glb:
        scene.export_glb(glb)


def options_from_kwargs(**kwargs) -> SurfaceOptions:
    known = {f for f in asdict(SurfaceOptions())}
    return SurfaceOptions(**{k: v for k, v in kwargs.items() if k in known and v is not None})
