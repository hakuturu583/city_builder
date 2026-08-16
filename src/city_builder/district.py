"""Every building in a scene, reconstructed and put back where it stood.

:mod:`city_builder.reconstruct` does one building. This does a street, which is
a different problem in three ways.

**It has to keep going.** A reconstruction is a generative model and a fraction
of them come back as something else — measured on ten real plots, one in five
fitted its footprint worse than 0.85. A run over two hundred buildings that
stops at the first of those is not a pipeline. Each building is tried, its fit
is recorded, and the ones that came back badly are *left as they were*: the
procedural massing is a poor building but it is the right shape, and a scene
where one house in five is the wrong shape is worse than a scene of boxes.

**It has to be resumable.** An hour and a half of GPU is long enough that
something will interrupt it. Every building's mesh and fit are written as they
are made, and a second run reads what is already there rather than paying for
it again.

**bpy and torch have to share a process.** They can — Blender ships its own
numpy and torch tolerates it — but the scene is rebuilt into Blender for every
portrait, so the rebuild happens once and the camera moves instead.
"""

from __future__ import annotations

import json
import math
import os
import time
from typing import Any

DEFAULT_STYLE = (
    "colour photograph of an ordinary Japanese suburban house, overcast afternoon, "
    "grey kawara tiled roof with ridge caps, cream mortar and ceramic siding walls, "
    "aluminium sliding windows, a downpipe, an air-conditioning unit, plain and modern"
)

# Half the work, and none of it obvious in advance. Asked for a Japanese house
# at a brush-up of 0.6, the image model returned a half-timbered cottage —
# the prompt named the country and the model's prior named the architecture,
# and the prior won. These are the shapes it reaches for when it is not told
# not to.
DEFAULT_NEGATIVE = (
    "half-timbered, tudor, european, alpine, cottage, chalet, brick, stone cottage, "
    "thatch, ornate, medieval, render, cgi, line drawing, sketch, "
    "street scene, adjacent buildings, cropped, people, cars, text, watermark"
)


def licence(plot: dict[str, Any], brush_up: float, *, plain: float = 1.5,
            elongated: float = 3.0, floor: float = 0.30) -> float:
    """How far the image model may redraw *this* plot's massing.

    Brushing the massing up is what makes the reconstruction look like a
    building rather than a box, and it is also what loses the plot's shape. The
    image model has a strong prior about the proportions of a house, and at a
    strength of 0.55 it acts on it: over 200 real plots the mesh came back
    between 1.4 and 1.5 times as long as it was deep whatever it was shown,
    and the drop rate went with the plot's own aspect — 89 % of plots under
    1.5:1 were kept against 7 % of those over 3:1.

    So the licence is spent where there is shape to spare. A square-ish plot
    gets the full ``brush_up``; by ``elongated`` it is down to ``floor``, which
    is enough to change the materials and not enough to square the plan up.
    """
    from .reconstruct import plan_dimensions

    if brush_up <= floor or not plot.get("footprint"):
        return brush_up
    long_side, short_side = plan_dimensions(plot["footprint"])
    ratio = long_side / max(short_side, 1e-6)
    if ratio <= plain:
        return brush_up
    towards = min((ratio - plain) / max(elongated - plain, 1e-6), 1.0)
    return round(brush_up + (floor - brush_up) * towards, 4)


def worth_rebuilding(plot: dict[str, Any], *, min_area: float = 0.0) -> bool:
    """Whether a plot is worth half a minute of GPU.

    Area, because that is what decides how much of the street a building is,
    and a footprint, because a plot recorded before ``buildings.generate`` kept
    one cannot be fitted back afterwards.
    """
    return bool(plot.get("footprint")) and plot.get("area", 0.0) >= min_area


def rebuild(scene, out_dir: str, *, buildings: list[int] | None = None,
            min_area: float = 0.0, limit: int = 0,
            facade_dir: str | None = None, roof_texture: str | None = None,
            roof_tile_metres: float = 0.45, style: str = DEFAULT_STYLE,
            brush_up: float = 0.55, resolution: str = "512", seed: int = 0,
            negative: str = DEFAULT_NEGATIVE,
            keep_below: float = 0.80, attempts: int = 3, resume: bool = True,
            marking_options=None, verbose: bool = True) -> dict[str, Any]:
    """Reconstruct a scene's buildings and write the models and the ledger.

    Writes ``<name>.png`` (the massing photographed), ``<name>_styled.png``
    (that picture brushed up), ``<name>.glb`` (the model with its textures) and
    ``<name>.obj`` (the same mesh in scene metres) per building, and one
    ``district.json`` recording every fit.

    ``keep_below`` is the line under which a reconstruction is not used. It is
    not an error — the model returned a building, just not this one — so the
    plot keeps its procedural massing and the ledger says why.

    ``attempts`` is how many times a building may be *asked for*. A generation
    that comes back as the wrong shape is a draw from a distribution, not a
    deterministic failure, and the cheapest thing to do about it is to draw
    again: only the buildings that missed pay for it, and the best of the tries
    is the one kept. It also covers the other kind of failure — an exception
    out of the pipeline, which on a run this long is usually the card being
    momentarily out of memory rather than anything about this building.
    """
    # Blender first: it brings its own numpy, and torch is content with that
    # while the reverse can leave one of them unable to load its own C module.
    from . import portrait as portrait_module
    from . import reconstruct as reconstruct_module

    plots = scene.result.plots
    if not plots:
        raise ValueError("this scene has no buildings; build it with buildings=True")

    if buildings is None:
        buildings = [i for i, plot in enumerate(plots)
                     if worth_rebuilding(plot, min_area=min_area)]
        buildings.sort(key=lambda i: -plots[i]["area"])
    if limit:
        buildings = buildings[:limit]

    os.makedirs(out_dir, exist_ok=True)
    ledger_path = os.path.join(out_dir, "district.json")
    done: dict[str, Any] = {}
    if resume and os.path.exists(ledger_path):
        with open(ledger_path, encoding="utf-8") as handle:
            done = {str(row["building"]): row for row in json.load(handle)["buildings"]}
        if verbose and done:
            print(f"[district] resuming: {len(done)} building(s) already modelled")

    options = portrait_module.PortraitOptions()
    started = time.time()
    rows: list[dict[str, Any]] = []
    for count, index in enumerate(buildings, start=1):
        if str(index) in done:
            rows.append(done[str(index)])
            continue
        name = f"b{index:04d}"
        strength = licence(plots[index], brush_up)
        row: dict[str, Any] = {"building": index, "area_m2": plots[index]["area"],
                               "roof": plots[index].get("roof"), "brush_up": strength}

        # The massing render is deterministic, so it is shot once and the tries
        # differ in what the image model and the mesh sampler make of it.
        best: dict[str, Any] | None = None
        shot: dict[str, Any] | None = None
        for attempt in range(max(1, attempts)):
            draw = seed + index + attempt * 10_007  # coprime with anything here
            try:
                if shot is None:
                    shot = portrait_module.render_portrait(
                        scene, index, os.path.join(out_dir, f"{name}.png"), options=options,
                        facade_dir=facade_dir, roof_texture=roof_texture,
                        roof_tile_metres=roof_tile_metres,
                        marking_options=marking_options, verbose=False)
                report = reconstruct_module.reconstruct(
                    plots[index], out_dir, image=shot["image"],
                    # Each try writes its own mesh, and the row that wins
                    # carries the paths of the one that won. Renaming the
                    # winner over the loser instead would leave a ledger and a
                    # directory that disagree if the run is interrupted.
                    name=name if attempt == 0 else f"{name}_t{attempt}",
                    mesh_options=reconstruct_module.MeshOptions(
                        seed=draw, pipeline_type=resolution, texture_size=1024,
                        decimation_target=80_000, tex_guidance=3.0),
                    restyle_prompt=style,
                    restyle_options=(reconstruct_module.RestyleOptions(strength=strength,
                                                                      seed=draw,
                                                                      negative=negative)
                                     if strength > 0 else None))
                got = {k: v for k, v in report.items() if k != "source"}
            except Exception as error:  # noqa: BLE001 - one building must not stop a street
                # A generative model has failure modes a caller cannot
                # enumerate, and the answer to all of them is the same: try
                # again, and if that is the last try leave the massing alone
                # and carry on. What matters is that the ledger says so.
                got = {"error": f"{type(error).__name__}: {error}"}
            if best is None or got.get("footprint_iou", 0.0) > best.get("footprint_iou", 0.0):
                best = got
            if best.get("footprint_iou", 0.0) >= keep_below:
                break
        row.update(best or {})
        row["tries"] = attempt + 1
        row["used"] = row.get("footprint_iou", 0.0) >= keep_below
        rows.append(row)

        if verbose:
            elapsed = time.time() - started
            fresh = sum(1 for r in rows if "took_seconds" in r)
            left = (len(buildings) - count) * (elapsed / max(fresh, 1))
            mark = "ok " if row.get("used") else "skip"
            again = f" x{row['tries']}" if row["tries"] > 1 else ""
            print(f"[district] {count}/{len(buildings)} {mark} {name}{again} "
                  f"iou={row.get('footprint_iou', float('nan')):.3f} "
                  f"~{left / 60:.0f} min left")
        _write(ledger_path, scene, rows, keep_below)

    return _write(ledger_path, scene, rows, keep_below)


def _turn(radians: float):
    """The plan rotation :mod:`city_builder.reconstruct` applies, as a matrix."""
    import numpy as np

    cos, sin = math.cos(radians), math.sin(radians)
    return np.array([[cos, sin], [-sin, cos]])


def _write(path: str, scene, rows: list[dict[str, Any]], keep_below: float) -> dict[str, Any]:
    used = [r for r in rows if r.get("used")]
    ious = [r["footprint_iou"] for r in used]
    summary = {
        "scene": scene.name,
        "map": scene.map_path,
        "attempted": len(rows),
        "used": len(used),
        "kept_above": keep_below,
        # What the retries cost and what they bought.
        "generations": sum(r.get("tries", 1) for r in rows),
        "retried": sum(1 for r in rows if r.get("tries", 1) > 1),
        "footprint_iou": {
            "mean": round(sum(ious) / len(ious), 4) if ious else None,
            "min": round(min(ious), 4) if ious else None,
        },
        "buildings": rows,
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


# ---------------------------------------------------------------------------
# Putting them back
# ---------------------------------------------------------------------------


def place(scene, ledger: str | dict[str, Any], *, facade_dir: str | None = None,
          roof_texture: str | None = None, roof_tile_metres: float = 0.45,
          ground_texture: str | None = None, tile_metres: float = 3.0,
          road_texture: str | None = None, road_tile_metres: float = 2.5,
          marking_options=None, verbose: bool = True) -> dict[str, Any]:
    """Build the scene into Blender with the reconstructions standing in it.

    The plots that were rebuilt lose their procedural massing; the ones that
    were not keep it, which is what makes a partial run useful rather than a
    scene with holes in it.
    """
    import bmesh
    import bpy
    import numpy as np

    from . import reconstruct as reconstruct_module
    from . import scene as scene_module
    from .build import build_scene
    from .portrait import face_range

    if isinstance(ledger, str):
        with open(ledger, encoding="utf-8") as handle:
            ledger = json.load(handle)
    rows = [r for r in ledger["buildings"] if r.get("used")]

    build_scene(scene.result, facade_dir=facade_dir, roof_texture=roof_texture,
                roof_tile_metres=roof_tile_metres, ground_texture=ground_texture,
                tile_metres=tile_metres, road_texture=road_texture,
                road_tile_metres=road_tile_metres, marking_options=marking_options,
                verbose=False)

    rebuilt = sorted({r["building"] for r in rows})
    for group in ("Buildings", "Roofs"):
        counts = scene_module.build.face_counts.get(group)
        obj = bpy.data.objects.get(group)
        if obj is None or not counts:
            continue
        doomed = {face for index in rebuilt
                  for face in range(*face_range(counts, index))}
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)
        bm.faces.ensure_lookup_table()
        bmesh.ops.delete(bm, geom=[f for i, f in enumerate(bm.faces) if i in doomed],
                         context="FACES")
        bm.to_mesh(mesh)
        bm.free()
        mesh.update()

    placed = 0
    for row in rows:
        if not os.path.exists(row.get("glb", "")):
            continue
        before = set(bpy.data.objects)
        bpy.ops.import_scene.gltf(filepath=os.path.abspath(row["glb"]))
        added = [o for o in bpy.data.objects if o not in before and o.type == "MESH"]
        parts = []
        for obj in added:
            obj.data.transform(obj.matrix_world)
            obj.matrix_world.identity()
            coords = np.empty(len(obj.data.vertices) * 3)
            obj.data.vertices.foreach_get("co", coords)
            parts.append(coords.reshape(-1, 3))
        if not parts:
            continue

        # The whole import shares one transform, so the centroid and the base
        # are measured over all of its parts together — and the base is the
        # height the building is full width at rather than its lowest vertex,
        # the same rule the fit was made under. Recomputed here rather than
        # read from the ledger so a ledger written before that rule existed
        # still places its buildings on the ground.
        everything = np.concatenate(parts)
        centroid = everything[:, :2].mean(axis=0)
        floor = reconstruct_module.seat_z(everything)
        # The two plan axes if the fit stretched it, one number if it did not;
        # ``stretch_deg`` is which way they point, the height always takes
        # their mean, and a ledger written before any of that still reads.
        axes = np.asarray(row.get("scale_xy") or [row["scale"], row["scale"]], dtype=float)
        rise = math.sqrt(axes[0] * axes[1])
        phi = math.radians(row.get("stretch_deg", 0.0))
        into = _turn(math.radians(row["yaw_deg"]) - phi)
        back = _turn(phi)
        for obj, coords in zip(added, parts):
            plan = ((coords[:, :2] - centroid) @ into * axes) @ back
            height = (coords[:, 2] - floor) * rise + row["base_z"]
            obj.data.vertices.foreach_set("co", np.column_stack(
                [plan[:, 0] + row["centre"][0], plan[:, 1] + row["centre"][1],
                 height]).reshape(-1))
            obj.data.update()
            obj.name = f"rebuilt_b{row['building']:04d}_{obj.name}"
        placed += 1

    if not any(obj.type == "LIGHT" for obj in bpy.data.objects):
        scene_module.sunlit()
    if verbose:
        print(f"[district] {placed} reconstruction(s) standing, "
              f"{len(scene.result.plots) - placed} still procedural")
    return {"placed": placed, "procedural": len(scene.result.plots) - placed}
