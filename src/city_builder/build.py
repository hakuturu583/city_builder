"""End-to-end: Lanelet2 map → ground and road-surface meshes."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

from . import buildings as buildings_module
from . import classes, facade_layout, lanelet, scene
from . import extend as extend_module
from . import ground as ground_module
from . import viaduct as viaduct_module
from .buildings import BuildingOptions
from .extend import ExtendOptions
from .frame import LocalFrame
from .markings import MarkingOptions
from .surfaces import SurfaceOptions, extract
from .viaduct import ViaductOptions


@dataclass
class BuildResult:
    frame: LocalFrame
    groups: dict[str, list]
    heightmap: ground_module.HeightMap | None
    elevated: set[int]
    z_datum: float
    stats: dict[str, Any] = field(default_factory=dict)
    # Paint baked off the carriageway: one mask page per image, and which page
    # each shape of each carrying group landed on.
    marking_pages: list[Any] = field(default_factory=list)
    marking_page_of_shape: dict[str, list[int]] = field(default_factory=dict)
    # One record per generated building, in the order of ``groups["Buildings"]``.
    # Carries the floor count, which decides which facade sheet it may wear.
    plots: list[dict[str, Any]] = field(default_factory=list)
    # The dissolved carriageway. Expensive to build and wanted again by the
    # ground cover, which measures its verges from it.
    road_union: Any = None


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
    ground_order: int = ground_module.DEFAULT_ORDER,
    elevation_model: bool = False,
    elevation_cache: str | None = None,
    relief=None,
    buildings: bool = False,
    building_options: BuildingOptions | None = None,
    viaduct_options: ViaductOptions | None = None,
    marking_options: MarkingOptions | None = None,
    extend_options: ExtendOptions | None = None,
    cover_options=None,
    verbose: bool = True,
) -> BuildResult:
    """Read a map and produce every surface, without touching Blender.

    Split out from the scene building so the geometry can be inspected, tested
    or fed somewhere other than Blender.

    ``cover_options`` is read here for one thing only: standing water, which is
    the single land-cover class that changes the *shape* of the ground rather
    than its colour. The rest of the cover is a texture and is applied at scene
    time. A palette with no water rule in it is never even classified, so a
    build without water is the build it always was.
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

    # Where the map ends, decided before anything moves. A lanelet that stops
    # inside the map leaves ground behind it, and the building generator fills
    # ground — which is how a street comes to end in the side of a block.
    converter = lanelet.PointConverter(ll2, projection, frame)
    plan = extend_module.from_map(ll2, projection, lmap, frame, extend_options,
                                  converter=converter)
    if verbose and plan.stats.get("extended"):
        span = plan.stats["length_m"]
        print(f"[build] roads run to the map edge: {plan.stats['extended']} of "
              f"{plan.stats['dangling_ends']} loose end(s) extended, "
              f"{span['min']:.0f}-{span['max']:.0f} m each")

    groups = extract(ll2, projection, lmap, frame, surface_options,
                     extensions=plan, converter=converter)
    datum = lanelet.apply_z_datum(groups, z_datum, z_offset)

    clipped = clip_crosswalks(groups, surface_options.crosswalk_lift
                              if surface_options else SurfaceOptions().crosswalk_lift)
    buried = clip_curbs(groups)
    paved_over = clip_walkways(groups)

    stats = {name: len(shapes) for name, shapes in groups.items()}
    stats["extended_ends"] = plan.stats.get("extended", 0)
    stats["dangling_ends"] = plan.stats.get("dangling_ends", 0)
    if verbose:
        print(f"[build] surfaces: {stats}")
        if paved_over:
            print(f"[build] walkways: {paved_over:.0f} m2 lapping the carriageway removed")
        if clipped:
            print(f"[build] crossings clipped to the carriageway "
                  f"({clipped:+d} surface(s) after the cut)")
        if buried:
            print(f"[build] kerbs: {buried} vertex/vertices had carriageway on both "
                  f"sides and were dropped")
        print(f"[build] z datum {datum:.2f} m → scene ground at {z_offset:.2f} m")

    elevated: set[int] = set()
    heightmap = None
    road_union = None
    water: list[Any] = []
    if ground:
        surfaces = list(groups.get("Roads", [])) + list(groups.get("Junctions", []))
        adjacency = lanelet.build_adjacency(lanelet.lanelet_end_keys(lmap))
        # Where the shape of the ground comes from: a downloaded survey, an
        # invented terrain, or both with the survey first and the invention
        # behind it. Only the curvature is taken — see city_builder.elevation.
        # The grid is not settled until the heightmap is being built, so this
        # goes in as a callback and the extent comes back to it.
        found: dict[str, Any] = {}

        def terrain_shape(x0, y0, nx, ny, cell_size, found=found, surfaces=surfaces):
            from . import elevation as elevation_module

            sources: list[Any] = []
            if elevation_model:
                sources += elevation_module.web_sources(elevation_cache)
            if relief is not None:
                # The map's own latitude, so "140 m across" is 140 m here.
                sources.append(elevation_module.InventedTiles(
                    relief, latitude=frame.ref_lat))
            if not sources:
                return None

            samples = [tuple(p) for r in surfaces for p in (*r.left, *r.right)]
            got = elevation_module.terrain_for(frame, x0, y0, nx, ny, cell_size, samples,
                                             sources=sources)
            if got is None:
                return None
            shape, coverage = got
            found["coverage"] = coverage
            return shape

        elevated, heightmap = ground_module.classify(
            surfaces, adjacency,
            cell=cell, z_gap=z_gap, min_overlap=min_overlap,
            clearance=clearance, smooth=smooth, drop=ground_drop,
            order=ground_order,
            guidance=terrain_shape if (elevation_model or relief is not None) else None,
            bounds=plan.box if plan.points else None,
        )
        if heightmap is None:
            raise RuntimeError("no ground-level road surfaces found; cannot build a ground surface")

        # Before the mesh, because this changes the heights the mesh is built
        # from. The class grid it reads is the one the painter will build again
        # at scene time, minus the road and plot geometry, which do not exist
        # yet — neither is produced until this mesh is. A rule that painted
        # water relative to a road would therefore not be honoured here; none
        # does, because standing water is said with Region.
        water_meshes: list[Any] = []
        # The class grid is wanted for two things — where the water is and
        # where the ground steps — so it is built once here and read twice
        # rather than by each pass that needs it.
        painted = None
        road_plan = None
        steps: list[tuple[Any, float]] = []
        if cover_options is not None:
            from . import cover as cover_module

            if cover_module.paints_water(cover_options) or any(
                    surface.step for surface in cover_options.surfaces):
                # The carriageway, dissolved here rather than waiting for
                # build_mesh to do it: the water has to be settled before the
                # ground is triangulated, and it must not flood a street.
                road_plan = ground_module.road_outline(surfaces, elevated,
                                                       close_gap=0.5)
                painted = cover_module.classify(heightmap, cover_options,
                                                roads=road_plan)
                if cover_module.paints_water(cover_options):
                    water = cover_module.flatten_water(heightmap, painted,
                                                       roads=road_plan)
                    water_meshes = cover_module.water_surface(water)
                if verbose:
                    for body in water:
                        report = body.to_json()
                        print(f"[build] water: {body.area:.0f} m2 at {body.level:+.2f} m, "
                              f"its bank at {body.bank:+.2f} m; the interpolation had it "
                              f"falling {body.fall:.2f} m across")
                        if report["held"] is not None and report["held"] < 0.9:
                            print(f"[build] water: only {report['held'] * 100:.0f}% of the "
                                  f"{report['asked_for']:.0f} m2 painted holds water — the "
                                  f"ground grid is {cell:g} m and cannot cut a bank inside "
                                  f"one cell. Try ground.cell of about a tenth of the pond.")

            if painted is not None:
                steps += [(outline, rise) for outline, rise
                          in cover_module.terraces(painted, cover_options)]

        # The buildings before the ground they stand on, which is the order the
        # work actually has: a plot is cut and filled level behind a retaining
        # wall *before* anything is built on it, so the ground has to be told
        # where those platforms are. Nothing here needs the ground *mesh* —
        # only the height map and where the carriageway is — so the dependency
        # was the other way round all along.
        plots = []
        if buildings:
            keep_clear = buildings_module.exclusion_zone(groups) or road_plan \
                or ground_module.road_outline(surfaces, elevated, close_gap=0.5)
            if water:
                # The plot generator reads a pond as open ground with no road
                # anywhere near it, which is its idea of a good place to build.
                # What is kept clear is what was *painted*, not the sheet that
                # ended up on it: the dry margin of a pond the grid could not
                # cut is the inside of an excavation, and a house belongs there
                # even less than one in the water.
                from shapely.ops import unary_union

                keep_clear = unary_union([keep_clear,
                                          *(b.painted if b.painted is not None
                                            else b.polygon for b in water)])
            built = buildings_module.generate(heightmap, keep_clear, building_options)
            if built["Buildings"]:
                groups["Buildings"] = built["Buildings"]
                groups["Roofs"] = built["Roofs"]
            plots = built["plots"]
            stats["buildings"] = len(built["Buildings"])
            steps += [(ring, building_options.terrace if building_options else
                       buildings_module.BuildingOptions().terrace)
                      for ring in built.get("terraces", ())]
            if steps and verbose:
                print(f"[build] terraces: {len(steps)} platform(s), the ground "
                      f"stepped and walled round each")

        # The shoreline and the retaining walls go into the triangulation as
        # breaklines. Without them the bank is cut at whatever angle the grid
        # crosses it, and the water meets a rim of triangles that belong to the
        # grid rather than to the pond.
        mesh, road_union = ground_module.build_mesh(
            heightmap, surfaces, elevated, fill_island=fill_island, drop=ground_drop,
            breaklines=[b.polygon for b in water if b.polygon is not None],
            terraces=steps, return_road_union=True)
        groups["Ground"] = [mesh]
        if water_meshes:
            groups["Water"] = water_meshes
            stats["water"] = [body.to_json() for body in water]

        # An elevated lanelet is surveyed as a driving surface and nothing
        # else: no slab, no soffit, no columns. Which *parts* of it are a bridge
        # is decided by clearance to the terrain, per cross-section — a ramp is
        # a bridge at one end and a road at the other.
        structure = viaduct_module.build(surfaces, elevated, heightmap, viaduct_options)
        for name, meshes in structure.items():
            if meshes:
                groups[name] = meshes
                stats[name] = len(meshes)
        if verbose and any(structure.values()):
            print("[build] viaduct: "
                  + ", ".join(f"{len(v)} {k[7:].lower()}" for k, v in structure.items() if v))

        measured = float((heightmap.support == 0).mean() * 100)
        if found.get("coverage") is not None:
            stats["terrain"] = found["coverage"].to_json()
            if verbose:
                c = found["coverage"]
                print(f"[build] terrain: {c.source} at {c.metres_per_pixel:.1f} m/px, "
                      f"{c.covered*100:.0f}% coverage, datum offset {c.datum_offset:+.2f} m, "
                      f"disagrees with the roads by {c.residual_p90:.2f} m (p90) — "
                      f"its shape is used, its heights are not")
        elif (elevation_model or relief is not None) and verbose:
            print("[build] terrain: no source covers this map; "
                  "the ground is interpolated from the roads alone")
        stats.update({
            "elevated_lanelets": len(elevated),
            "ground_faces": len(mesh.faces),
            "cells_measured_pct": round(measured, 1),
        })
        if verbose:
            print(f"[build] elevated lanelets: {len(elevated)} / {len(surfaces)}")
            print(f"[build] ground: {heightmap.nx}x{heightmap.ny} @ {cell:.1f} m, "
                  f"{measured:.1f}% of cells measured, {len(mesh.faces)} faces")

    patched, patched_area = infill_roads(groups, viaduct_options or ViaductOptions(),
                                         elevated)
    if patched and verbose:
        print(f"[build] road infill: {patched} patch(es), {patched_area:.0f} m2 of gap "
              f"between lanelets closed")

        if verbose:
            heights = [p["height"] for p in plots]
            span = f"{min(heights):.0f}-{max(heights):.0f} m" if heights else "none"
            floors = sorted({p["floors"] for p in plots})
            print(f"[build] buildings: {len(built['Buildings'])} on the open ground, {span} tall")
            if floors:
                print(f"[build] floor counts: {floors[0]}-{floors[-1]} "
                      f"({len(floors)} distinct, one facade sheet family each)")

    from . import markings as markings_module

    pages, page_of_shape = markings_module.bake(groups, marking_options)
    if pages:
        # The crossing surface was kept as a semantic region sitting a few
        # millimetres over the road. With the zebra in the road's own texture,
        # that slab now hides the paint it was meant to sit beside.
        groups.pop("Crosswalks", None)
        stats["marking_pages"] = len(pages)
        if verbose:
            painted = sum(len(v) for v in page_of_shape.values())
            print(f"[build] paint baked onto {painted} lane(s) in {len(pages)} "
                  f"atlas page(s); the marking geometry is gone")

    return BuildResult(frame, groups, heightmap, elevated, datum, stats=stats, plots=plots,
                       marking_pages=pages, marking_page_of_shape=page_of_shape,
                       road_union=road_union)


def clip_crosswalks(groups: dict[str, list], lift_by: float) -> int:
    """Cut each crossing surface down to the carriageway that carries it.

    A crossing lanelet covers the road *and* the footway either side, and the
    road is already there — measured on this map, 67 of 84 crossing surfaces
    overlapped the carriageway, a median 81 % of their area, 4646 m2 in total,
    at a median 7 cm apart in z. Two slabs of asphalt in the same place is a
    z-fight at best; on a viaduct the overhanging part is worse, because it
    sails past the deck edge with nothing under it.

    So the surface is intersected with the carriageway and lifted a few
    millimetres onto it. The crossing survives as a region a consumer can
    select, without being a second road.
    """
    from shapely.ops import unary_union

    from .geometry import height_lookup, triangulate_polygon
    from .viaduct import _footprints

    crossings = groups.get("Crosswalks")
    carriageway = list(groups.get("Roads", [])) + list(groups.get("Junctions", []))
    if not crossings or not carriageway:
        return 0

    road = unary_union(_footprints(carriageway, None))
    lift = height_lookup([p for r in carriageway for p in list(r.left) + list(r.right)])

    kept = []
    for crossing in crossings:
        footprint = _footprints([crossing], None)
        if not footprint:
            continue
        on_road = footprint[0].intersection(road)
        if on_road.is_empty:
            continue
        for part in getattr(on_road, "geoms", [on_road]):
            if part.geom_type != "Polygon" or part.area < 0.5:
                continue
            mesh = triangulate_polygon(part, lift, offset=lift_by)
            if mesh.faces:
                kept.append(mesh)

    dropped = len(crossings) - len(kept)
    groups["Crosswalks"] = kept
    return dropped


def clip_walkways(groups: dict[str, list]) -> float:
    """Take the carriageway back out of every footway. Returns the area removed.

    The mirror of :func:`clip_crosswalks`, and needed for the same reason from
    the other side. A crossing belongs on the road, so it is cut *down* to it;
    a walkway does not, so it is cut *away* from it. A walkway lanelet is
    surveyed along the footway but its boundaries are generous at a junction
    mouth, and the surface then laps onto the carriageway — measured on the
    Kashiwanoha map, 9 of one walkway's 91 m2.

    That is not a cosmetic overlap. Every pedestrian surface is lifted 3 cm
    clear of the road so it does not z-fight with it, so the lapped part is a
    3 cm lip lying across a lane: a vehicle driving the scene hits a step, and
    a perception stack sees a kerb where the map says there is none.
    """
    from shapely.ops import unary_union

    from .geometry import height_lookup, triangulate_polygon
    from .viaduct import _footprints

    walkways = groups.get("Walkways")
    carriageway = list(groups.get("Roads", [])) + list(groups.get("Junctions", []))
    if not walkways or not carriageway:
        return 0.0

    road = unary_union(_footprints(carriageway, None))
    lift = height_lookup([p for r in walkways for p in list(r.left) + list(r.right)])

    kept, removed = [], 0.0
    for walkway in walkways:
        footprint = _footprints([walkway], None)
        if not footprint:
            continue
        off_road = footprint[0].difference(road)
        removed += footprint[0].area - off_road.area
        if off_road.is_empty:
            continue
        for part in getattr(off_road, "geoms", [off_road]):
            if part.geom_type != "Polygon" or part.area < 0.5:
                continue
            mesh = triangulate_polygon(part, lift)
            if mesh.faces:
                kept.append(mesh)

    groups["Walkways"] = kept
    return removed


def clip_curbs(groups: dict[str, list], probe: float = 0.6,
               shortest: float = 1.5) -> int:
    """Drop the parts of a kerb line that have carriageway on both sides.

    A road_border is only a kerb where something stops at it. Measured on this
    map, 689 of 8055 kerb vertices had lanelet surface on either side of them —
    a lane divider, a give-way line, the seam between a carriageway and its
    slip road — and standing those up puts a 15 cm wall down the middle of the
    road for traffic to drive through.
    """
    import numpy as np
    from shapely.geometry import Point as ShapelyPoint
    from shapely.ops import unary_union
    from shapely.prepared import prep

    from .geometry import Ribbon
    from .viaduct import _footprints, polyline_length, runs_of

    curbs = groups.get("Curbs")
    surfaces = [shape for name in ("Roads", "Junctions", "Crosswalks", "Walkways")
                for shape in groups.get(name, ()) if hasattr(shape, "ring")]
    if not curbs or not surfaces:
        return 0

    covered = prep(unary_union([p.buffer(probe / 2.0) for p in _footprints(surfaces, None)]))

    kept, dropped = [], 0
    for curb in curbs:
        base, raised = list(curb.left), list(curb.right)
        points = np.asarray(base, dtype=float)[:, :2]
        if len(points) < 2:
            continue
        step = np.gradient(points, axis=0)
        normal = np.stack([-step[:, 1], step[:, 0]], axis=1)
        normal /= np.linalg.norm(normal, axis=1, keepdims=True) + 1e-12

        edge = [
            not (covered.contains(ShapelyPoint(*(p + n * probe)))
                 and covered.contains(ShapelyPoint(*(p - n * probe))))
            for p, n in zip(points, normal)
        ]
        for start, end in runs_of(edge):
            if polyline_length(base[start:end + 1]) < shortest:
                continue
            kept.append(Ribbon(curb.id, base[start:end + 1], raised[start:end + 1],
                               dict(curb.attributes)))
        dropped += sum(1 for flag in edge if not flag)

    groups["Curbs"] = kept
    return dropped


def infill_roads(groups: dict[str, list], options, elevated: set[int] | None = None
                 ) -> tuple[int, float]:
    """Patch the gaps the lanelets leave between themselves, level by level.

    Lanelets are surveyed one at a time and do not tile exactly. Measured on
    this map, 306 gaps totalling 2922 m2 — 3 % of a 98 120 m2 carriageway, most
    of them a hand's breadth wide. On a viaduct each one is a slot through to
    the street below; at grade the ground shows through, and since the ground
    is held 5 cm under the road a wheel crossing a sliver drops into it. Either
    way the drivable surface is not a surface.

    Each level is patched against itself. A single plan-view union cannot do
    this: a viaduct and the street beneath it leave a long strip between their
    footprints, which is not a gap in anything — it is the space beside the
    viaduct, and filling it paves over the street at deck height.
    """
    from .viaduct import infill_meshes, infill_polygons

    carriageway = [shape for name in ("Roads", "Junctions")
                   for shape in groups.get(name, ()) if hasattr(shape, "ring")]
    if not carriageway:
        return 0, 0.0

    elevated = elevated or set()
    levels = ([shape for shape in carriageway if shape.id not in elevated],
              [shape for shape in carriageway if shape.id in elevated])

    meshes, area = [], 0.0
    for level in levels:
        if not level:
            continue
        patches, _covered = infill_polygons(level, options)
        meshes.extend(infill_meshes(level, patches))
        area += sum(patch.area for patch in patches)

    if meshes:
        groups["RoadInfill"] = meshes
    return len(meshes), area


def _relief_of(g):
    """The invented-terrain settings, as the dataclass that generates it."""
    if not getattr(g, "relief", False):
        return None
    from .elevation import Relief

    return Relief(amplitude=g.relief_amplitude, metres=g.relief_metres,
                  octaves=g.relief_octaves, roughness=g.relief_roughness,
                  warp=g.relief_warp, seed=g.relief_seed)


def build_city_from_config(input_path: str, config, *, buildings: bool = False,
                           ref_lat: float | None = None, ref_lon: float | None = None,
                           projector: str = "utm", z_datum: float | None = None,
                           z_offset: float = 0.0, cover_options=None,
                           verbose: bool = True) -> BuildResult:
    """:func:`build_city`, with every option group taken from a
    :class:`city_builder.config.CityConfig`.

    ``cover_options`` stays an argument rather than a config section because
    the cover is not in the config file yet — it is authored in Python, and
    :func:`build_scene` takes it the same way.
    """
    g = config.ground
    return build_city(
        input_path, ref_lat=ref_lat, ref_lon=ref_lon, projector=projector,
        z_datum=z_datum, z_offset=z_offset,
        surface_options=config.surfaces, marking_options=config.markings,
        ground=g.enabled, cell=g.cell, smooth=g.smooth, z_gap=g.z_gap,
        min_overlap=g.min_overlap, clearance=g.clearance,
        ground_drop=g.drop, fill_island=g.fill_island, ground_order=g.order,
        elevation_model=g.elevation_model, elevation_cache=g.elevation_cache,
        relief=_relief_of(g),
        buildings=buildings, building_options=config.buildings,
        viaduct_options=config.viaduct, extend_options=config.extend,
        cover_options=cover_options, verbose=verbose,
    )


def render_drive(result: BuildResult, input_path: str, scene_path: str, output_path: str, *,
                 seconds: float = 30.0, speed: float = 11.0, fps: int = 30,
                 eye_height: float = 1.4, look_ahead: float = 18.0, lens: float = 30.0,
                 resolution: tuple[int, int] = (1280, 720), samples: int = 32,
                 engine: str = "eevee", route_seed: int = 0,
                 verbose: bool = True) -> str | None:
    """Drive a camera through a built scene and render it.

    The route comes from the map rather than from the scene, because the scene
    only kept the surfaces. Returns the video path, or None if the map has no
    drivable route in it.
    """
    import bpy

    from . import lanelet, route
    from . import scene as scene_module

    _ll2, _projection, lmap = lanelet.load_map(
        input_path, projector="utm",
        origin_lat=result.frame.ref_lat, origin_lon=result.frame.ref_lon)

    path = route.drive_path(
        result.groups, lanelet.lanelet_end_keys(lmap), seed=route_seed,
        eye_height=eye_height, look_ahead=look_ahead, step=speed / fps)
    if not path:
        if verbose:
            print("[drive] no drivable route in this map; skipping the video")
        return None

    wanted = int(seconds * fps)
    if len(path) < wanted and verbose:
        print(f"[drive] the route runs out after {len(path) / fps:.0f} s "
              f"at {speed:g} m/s; rendering that")
    path = path[:wanted]

    bpy.ops.wm.open_mainfile(filepath=os.path.abspath(scene_path))
    if not any(obj.type == "LIGHT" for obj in bpy.data.objects):
        scene_module.sunlit()
    scene_module.animate_camera(path, lens=lens)
    return scene_module.render_animation(
        output_path, frames=len(path), fps=fps, resolution=resolution,
        samples=samples, engine="CYCLES" if engine == "cycles" else "BLENDER_EEVEE",
        verbose=verbose)


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
    if result.marking_pages:
        # `preserve` stops being a property of a group of objects and becomes
        # the mask channel: the carriageway's colour may be regenerated where
        # the mask is zero and must not be touched where it is not.
        payload["markings"] = {
            "mode": "texture",
            "policy": classes.PRESERVE,
            "pages": len(result.marking_pages),
            "carried_by": sorted(result.marking_page_of_shape),
            "uv_layer": "UVMap",
            "note": "road markings are baked into the carriageway texture; the mask "
                    "is where the colour is authored and must survive a texturing pass",
        }

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


def write_marking_pages(result: BuildResult, directory: str) -> list[str]:
    """Save the mask pages beside the scene, so Blender has files to load."""
    from PIL import Image

    os.makedirs(directory, exist_ok=True)
    paths = []
    for i, page in enumerate(result.marking_pages):
        path = os.path.join(directory, f"markings_{i:02d}.png")
        Image.fromarray(page, mode="L").save(path)
        paths.append(path)
    return paths


def build_scene(result: BuildResult, *, blend: str | None = None, glb: str | None = None,
                fbx: str | None = None,
                ground_texture: str | None = None, tile_metres: float = 12.0,
                road_tile_metres: float | None = None,
                roof_texture: str | None = None, roof_tile_metres: float = 3.0,
                facade_dir: str | None = None, road_texture: str | None = None,
                marking_options: MarkingOptions | None = None,
                cover_options=None, cover_path: str | None = None,
                cover_texels_per_metre: float = 8.0,
                markings_dir: str | None = None, verbose: bool = True) -> None:
    """Build the result into Blender and export it.

    ``ground_texture`` is a tile image to repeat across the ground. Only the
    ground: every lanelet-derived surface keeps the material it was built with,
    because the map already says what those look like.

    ``cover_options`` replaces that one tile with a painted map — grass, gravel,
    yards and paving where :mod:`city_builder.cover` says they are — baked once
    into a single image anchored to the scene. It takes precedence over
    ``ground_texture``, which then serves only as the ``grass`` fallback tile a
    palette may not have supplied.

    Water is the exception, and it is not handled here at all: it is geometry
    rather than colour, so :func:`build_city` levelled the ground under it and
    gave it a surface of its own, which keeps its own material.

    The carriageway gets its own metric scale. Paving slabs and road aggregate
    are not the same size: one tile spanning the twelve metres that suits a
    pavement gives asphalt a grain a foot across, and the road comes out
    cobbled.
    """
    road_tile_metres = tile_metres if road_tile_metres is None else road_tile_metres
    scene.clear_scene()
    objects = scene.build(result.groups, verbose=verbose)

    if result.marking_pages:
        # The pages live beside whatever is being written. Nothing being
        # written — a render, say — means the caller has to say where, or the
        # scene would leave a directory behind in whatever cwd it ran in.
        anchor = blend or glb or fbx
        directory = markings_dir or (
            os.path.splitext(os.path.abspath(anchor))[0] + "_markings" if anchor
            else os.path.abspath("scene_markings"))
        pages = write_marking_pages(result, directory)
        options = marking_options or MarkingOptions()
        for name, page_of_shape in result.marking_page_of_shape.items():
            carrier = objects.get(name)
            if carrier is None:
                continue
            if road_texture:
                scene.uv_from_xy(carrier, road_tile_metres, name="AsphaltUV")
            scene.apply_marking_pages(carrier, pages, scene.build.face_counts[name],
                                      page_of_shape, options,
                                      asphalt_image=road_texture,
                                      tile_metres=road_tile_metres)
        if verbose:
            print(f"[scene] paint: {len(pages)} page(s) over "
                  f"{', '.join(result.marking_page_of_shape)}")

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

    if roof_texture:
        # Its own scale, and a small one. A roof tile is a hand's width and the
        # pitched roofs this generates are the largest single surfaces in a
        # low-rise street — at the ground's twelve metres a kawara roof reads as
        # a tarpaulin.
        roofs = objects.get("Roofs")
        if roofs is None:
            raise RuntimeError("no Roofs object to texture; build with buildings=True")
        scene.apply_tiled_texture(roofs, roof_texture, roof_tile_metres)
        if verbose:
            print(f"[scene] Roofs: tiled {os.path.basename(roof_texture)} every "
                  f"{roof_tile_metres:g} m")

    if cover_options is not None:
        from . import cover as cover_module

        ground = objects.get("Ground")
        if ground is None:
            raise RuntimeError("no Ground object to texture; build with ground=True")
        if result.heightmap is None:
            raise RuntimeError("the ground cover needs the height map")
        painted = cover_module.classify(
            result.heightmap, cover_options, roads=result.road_union,
            plots=[plot["footprint"] for plot in result.plots if plot.get("footprint")])
        anchor = blend or glb or fbx
        target = cover_path or (
            os.path.splitext(os.path.abspath(anchor))[0] + "_ground.png" if anchor
            else os.path.abspath("ground_cover.png"))
        report = cover_module.paint(painted, cover_options, target,
                                    texels_per_metre=cover_texels_per_metre)
        # One image over the whole map, so the UV spans it once and is anchored
        # to it — and clamps at the edge rather than wrapping the town round.
        scene.apply_tiled_texture(ground, target, report["metres"][0],
                                  origin=tuple(report["origin"]), extend=True)
        if verbose:
            spread = ", ".join(f"{k} {v * 100:.0f}%" for k, v in report["surfaces"].items())
            print(f"[scene] Ground: painted {report['size'][0]}x{report['size'][1]} at "
                  f"{report['texels_per_metre']:.1f} texels/m — {spread}")
    elif ground_texture:
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
    if fbx:
        scene.export_fbx(fbx)
