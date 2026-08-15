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
            keep_below: float = 0.80, resume: bool = True,
            marking_options=None, verbose: bool = True) -> dict[str, Any]:
    """Reconstruct a scene's buildings and write the models and the ledger.

    Writes ``<name>.png`` (the massing photographed), ``<name>_styled.png``
    (that picture brushed up), ``<name>.glb`` (the model with its textures) and
    ``<name>.obj`` (the same mesh in scene metres) per building, and one
    ``district.json`` recording every fit.

    ``keep_below`` is the line under which a reconstruction is not used. It is
    not an error — the model returned a building, just not this one — so the
    plot keeps its procedural massing and the ledger says why.
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
    mesh_options = reconstruct_module.MeshOptions(seed=seed, pipeline_type=resolution,
                                                  texture_size=1024,
                                                  decimation_target=80_000,
                                                  tex_guidance=3.0)
    started = time.time()
    rows: list[dict[str, Any]] = []
    for count, index in enumerate(buildings, start=1):
        if str(index) in done:
            rows.append(done[str(index)])
            continue
        name = f"b{index:04d}"
        row: dict[str, Any] = {"building": index, "area_m2": plots[index]["area"],
                               "roof": plots[index].get("roof")}
        try:
            shot = portrait_module.render_portrait(
                scene, index, os.path.join(out_dir, f"{name}.png"), options=options,
                facade_dir=facade_dir, roof_texture=roof_texture,
                roof_tile_metres=roof_tile_metres, marking_options=marking_options,
                verbose=False)
            report = reconstruct_module.reconstruct(
                plots[index], out_dir, image=shot["image"], name=name,
                mesh_options=mesh_options, restyle_prompt=style,
                restyle_options=(reconstruct_module.RestyleOptions(strength=brush_up,
                                                                  seed=seed + index,
                                                                  negative=negative)
                                 if brush_up > 0 else None))
            row.update({k: v for k, v in report.items() if k != "source"})
            row["used"] = report["footprint_iou"] >= keep_below
        except Exception as error:  # noqa: BLE001 - one building must not stop a street
            # A generative model has failure modes a caller cannot enumerate,
            # and the answer to all of them is the same: leave the massing
            # alone and carry on. What matters is that the ledger says so.
            row.update({"used": False, "error": f"{type(error).__name__}: {error}"})
        rows.append(row)

        if verbose:
            elapsed = time.time() - started
            fresh = sum(1 for r in rows if "took_seconds" in r)
            left = (len(buildings) - count) * (elapsed / max(fresh, 1))
            mark = "ok " if row.get("used") else "skip"
            print(f"[district] {count}/{len(buildings)} {mark} {name} "
                  f"iou={row.get('footprint_iou', float('nan')):.3f} "
                  f"~{left / 60:.0f} min left")
        _write(ledger_path, scene, rows, keep_below)

    return _write(ledger_path, scene, rows, keep_below)


def _write(path: str, scene, rows: list[dict[str, Any]], keep_below: float) -> dict[str, Any]:
    used = [r for r in rows if r.get("used")]
    ious = [r["footprint_iou"] for r in used]
    summary = {
        "scene": scene.name,
        "map": scene.map_path,
        "attempted": len(rows),
        "used": len(used),
        "kept_above": keep_below,
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
        # are measured over all of its parts together.
        everything = np.concatenate(parts)
        centroid, floor = everything[:, :2].mean(axis=0), everything[:, 2].min()
        yaw = math.radians(row["yaw_deg"])
        turn = np.array([[math.cos(yaw), math.sin(yaw)], [-math.sin(yaw), math.cos(yaw)]])
        for obj, coords in zip(added, parts):
            plan = (coords[:, :2] - centroid) @ turn * row["scale"]
            height = (coords[:, 2] - floor) * row["scale"] + row["base_z"]
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
