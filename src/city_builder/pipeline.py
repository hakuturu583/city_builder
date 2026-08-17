"""A Lanelet2 map to a textured, reconstructed city, in one run.

Every stage of this package exists on its own and can be run on its own; this
is the one that runs them in order, with each stage's output wired into the
next. There is nothing clever in it and that is the point — the value is in the
*order* and in the handover, both of which are easy to get subtly wrong by
hand:

1. **Ground.** The map's road surfaces, the terrain between them, and the
   plots. The plots come out of this stage, and the two stages after it are
   both parameterised by what the plots turned out to be — how many storeys the
   buildings have, and how many of them there are.
2. **Materials.** A tile per ground cover class and a roof tile, from prompts.
   These are what the *massing* wears when it is photographed, so they have to
   exist before the reconstruction and not after it: the picture handed to the
   3D model is the whole of what it knows.
3. **Facades.** A family of sheets per floor count present in the plots — not a
   fixed set, because the facade UV normalises V over the building's height and
   a sheet drawn for six storeys does not read on a three-storey house.
4. **Reconstruction.** Every plot photographed, brushed up, turned into a mesh
   by TRELLIS.2 and fitted back to its own footprint.
5. **Scene.** The reconstructions placed on the ground they came from, the
   ground painted, and the whole thing written out.

**Resumable at every stage**, because stages 2 to 4 are GPU hours and something
will interrupt them. A stage is skipped when its output is already there; pass
``force`` to make it run anyway. Stage 4 keeps its own per-building ledger and
resumes inside itself.

Nothing here imports torch or bpy at module level: importing this module costs
nothing, and a caller running only stage 1 never loads a diffusion stack.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

STAGES = ("ground", "materials", "facades", "reconstruct", "scene")

# What each ground cover class looks like. Written here rather than in
# `cover.py` because they are prompts for one particular image model, and the
# cover module is about what the ground *is*, not about how a picture of it is
# obtained.
COVER_PROMPTS: dict[str, str] = {
    "grass": "top-down photograph of short mown lawn grass, overcast daylight, "
             "seamless texture",
    "ground": "top-down photograph of bare packed brown earth with fine gravel, dry, "
              "overcast daylight, seamless texture",
    "gravel": "top-down photograph of light grey crushed stone gravel driveway, "
              "overcast daylight, seamless texture",
    "paving_stones": "top-down photograph of grey concrete paving slabs laid in a grid, "
                     "narrow joints, overcast daylight, seamless texture",
    "concrete": "top-down photograph of a plain grey concrete slab, faint stains, "
                "overcast daylight, seamless texture",
}
# The carriageway is deliberately absent. The ground cover is what lies
# *between* the roads; the road surface is the map's own, it already carries
# its paint baked into an atlas at its own texel density, and a tile generated
# from the same "seamless ground texture" prompt as a lawn came out as a field
# of speckle with lane lines through it. A road wants a road texture, given to
# `build_scene` as `road_texture`, not a ground cover class.

ROOF_PROMPT = ("top-down photograph of grey Japanese kawara roof tiles in even courses, "
               "overcast daylight, seamless texture")

# "Seamless texture" is the phrase that gets a tileable image out of the model
# and also the phrase that pulls it towards stock clipart, which arrives on a
# white background. Saying no to the background is cheaper than losing the
# phrase that makes the tile tile.
TILE_NEGATIVE = ("illustration, clipart, vector art, cartoon, drawing, "
                 "white background, isolated objects, cut out, border, frame, "
                 "watermark, text, high contrast, black outlines")


@dataclass
class Recipe:
    """What to make, as opposed to how the stages work.

    The defaults are the ones measured on the Kashiwanoha map through this
    session; each is a decision rather than a preference, and the ones worth
    knowing carry their reason.
    """

    # --- ground ---------------------------------------------------------
    # 2.5 m rather than the 10 m default, because the plot platforms and any
    # water body are cut by moving grid nodes: the sharpest bank the ground can
    # take is one cell wide. A 28 m pond holds 40 % of itself at 10 m and 100 %
    # at 2.5 m.
    cell: float = 2.5
    relief: bool = True
    relief_amplitude: float = 4.0
    # About a city block. Terrain much larger than one is cancelled by the road
    # constraints, which pin the low frequencies: measured, 0.18 m of relief
    # survives at 200 m features against 0.45 m at 60 m.
    relief_metres: float = 80.0
    relief_seed: int = 3
    elevation_model: bool = False

    # --- materials ------------------------------------------------------
    tile_size: int = 768
    tile_steps: int = 30
    tile_attempts: int = 3       # a tile is a draw; score it and draw again
    roof_tile_metres: float = 0.45
    cover_texels_per_metre: float = 12.0

    # --- facades --------------------------------------------------------
    facade_variants: int = 4
    facade_count: int = 4
    facade_keep_below: float | None = 0.45
    house_floors: int = 3        # at or below this a building is a house
    # How hard the line drawing holds the windows on the storeys. 1.2 rather
    # than the library's 0.9: on this map's 1-3 storey controls, the share of
    # sheets scoring at least 0.45 goes 47 % -> 64 %, and pushing on to 1.5 or
    # 1.8 buys nothing more (61 %, 64 %) while the mean keeps drifting up on
    # sheets that were already good. Diversity is unmoved at 0.28-0.29
    # throughout, so this is not bought by making every wall the same.
    control_scale: float = 1.2

    # --- reconstruction -------------------------------------------------
    # Where the massing comes from. Left off, the plot's procedural block is
    # photographed, brushed up by an image model, and the mesh that comes back
    # is fitted onto the plot — 184 of 189 buildings at a mean footprint IoU of
    # 0.917, bought with a yaw sweep, an anisotropic stretch and a seating pass.
    # Turned on, the plot's own prism *is* the occupancy grid TRELLIS generates
    # in, and the picture is only asked what the building is made of: 0.88
    # uniform, nothing to stretch, and half the time per building. It is also
    # the only one of the two where a photograph can be photoreal, because the
    # picture no longer has to be a render of this particular block.
    envelope: bool = False
    envelope_subjects: int = 10   # how many photographs the street shares
    eave_room: float = 0.6
    roof_room: float = 0.4
    resolution: str = "512"
    brush_up: float = 0.55
    keep_below: float = 0.80
    attempts: int = 3
    limit: int = 0
    min_area: float = 0.0
    seed: int = 0

    # --- scene ----------------------------------------------------------
    renders: bool = True
    glb: bool = True
    # Off by default and glTF on, because glTF is what this package's own
    # readers take and it carries the surface mask that FBX cannot. FBX is for
    # handing the district to Unreal or Unity, which is a thing you ask for.
    fbx: bool = False

    extra: dict[str, Any] = field(default_factory=dict)


def run(map_path: str, out_dir: str, *, config=None, recipe: Recipe | None = None,
        stages: Sequence[str] = STAGES, force: Sequence[str] = (),
        verbose: bool = True) -> dict[str, Any]:
    """The whole thing. Returns a report, and writes ``pipeline.json`` beside it.

    ``stages`` is which of :data:`STAGES` to run and ``force`` which to run
    again even though their output is there. Both take stage names, so
    ``stages=("ground", "scene")`` re-places an existing reconstruction without
    touching the GPU.
    """
    from .config import CityConfig

    recipe = recipe or Recipe()
    config = config or CityConfig()
    _apply(recipe, config)
    os.makedirs(out_dir, exist_ok=True)

    report: dict[str, Any] = {"map": map_path, "out": out_dir, "stages": {}}
    started = time.time()
    state: dict[str, Any] = {}

    for stage in STAGES:
        if stage not in stages:
            continue
        mark = time.time()
        if verbose:
            print(f"\n=== {stage}")
        report["stages"][stage] = _STAGES[stage](
            map_path, out_dir, config, recipe, state,
            force=stage in force, verbose=verbose)
        report["stages"][stage]["seconds"] = round(time.time() - mark, 1)

    report["seconds"] = round(time.time() - started, 1)
    with open(os.path.join(out_dir, "pipeline.json"), "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, default=str)
    if verbose:
        print(f"\n=== done in {report['seconds'] / 60:.1f} min → {out_dir}")
    return report


def _apply(recipe: Recipe, config) -> None:
    """The recipe's ground settings, onto the config the build will read."""
    config.ground.cell = recipe.cell
    config.ground.relief = recipe.relief
    config.ground.relief_amplitude = recipe.relief_amplitude
    config.ground.relief_metres = recipe.relief_metres
    config.ground.relief_seed = recipe.relief_seed
    config.ground.elevation_model = recipe.elevation_model


# ---------------------------------------------------------------------------
# 1. Ground
# ---------------------------------------------------------------------------


def _ground(map_path, out_dir, config, recipe, state, *, force, verbose):
    """Roads, terrain, cover and plots. Everything after this reads the plots."""
    from .build import build_city_from_config, write_manifest

    result = build_city_from_config(map_path, config, buildings=True, verbose=verbose,
                                    cover_options=_cover(out_dir, config))
    state["result"] = result
    write_manifest(result, os.path.join(out_dir, "manifest.json"))
    floors = sorted({plot["floors"] for plot in result.plots})
    state["floors"] = floors
    if verbose:
        print(f"[pipeline] {len(result.plots)} plot(s), floor counts {floors}")
    return {"plots": len(result.plots), "floors": floors,
            "ground_faces": len(result.groups["Ground"][0].faces)}


def _cover(out_dir: str, config):
    """The ground cover palette, wearing whatever tiles have been made so far.

    Called before the materials stage as well as after it, and that is
    deliberate: a class with no tile yet falls back to its flat colour, so the
    first pass through still produces a scene rather than an error.
    """
    from . import cover as cover_module

    art = {}
    tiles = os.path.join(out_dir, "tiles")
    for name in COVER_PROMPTS:
        path = os.path.join(tiles, f"{name}.png")
        if os.path.exists(path):
            art[name] = path
    return cover_module.japanese_suburb(textures=art)


# ---------------------------------------------------------------------------
# 2. Materials
# ---------------------------------------------------------------------------


def _materials(map_path, out_dir, config, recipe, state, *, force, verbose):
    """A tile per ground class, and one for the roofs."""
    from .texture import TextureOptions, ground_tile, stable_seed

    tiles = os.path.join(out_dir, "tiles")
    os.makedirs(tiles, exist_ok=True)
    wanted = {**COVER_PROMPTS, "kawara": ROOF_PROMPT}
    made, skipped, scores, poor = [], 0, {}, []
    for name, prompt in wanted.items():
        path = os.path.join(tiles, f"{name}.png")
        if os.path.exists(path) and not force:
            skipped += 1
            continue
        # A stable seed, because `hash()` is randomised per process and a tile
        # set seeded that way is a different set every run — which is how one
        # bad grass tile got laid over a whole town with nothing to reproduce.
        options = TextureOptions(size=recipe.tile_size, steps=recipe.tile_steps,
                                 seed=stable_seed(name) + recipe.seed)
        got = ground_tile(prompt, options, path=path,
                          negative_prompt=TILE_NEGATIVE,
                          attempts=recipe.tile_attempts)
        scores[name] = {k: got[k] for k in ("blown", "sd", "seam", "tries")}
        made.append(path)
        if not got["usable"]:
            poor.append(name)
        if verbose:
            mark = "" if got["usable"] else "  <- still not usable"
            print(f"[pipeline] {name}: blown {got['blown']:.3f}, sd {got['sd']:.1f}, "
                  f"seam {got['seam']:.2f}, x{got['tries']}{mark}")
    state["roof_texture"] = os.path.join(tiles, "kawara.png")
    return {"made": len(made), "kept": skipped, "unusable": poor, "tiles": scores}


# ---------------------------------------------------------------------------
# 3. Facades
# ---------------------------------------------------------------------------


def _facades(map_path, out_dir, config, recipe, state, *, force, verbose):
    """Sheets for the floor counts this map actually produced, and no others.

    The facade UV normalises V over the building's height, so a sheet belongs
    to a floor count: drawing the usual 4-6-9 for a street of two-storey houses
    puts six rows of windows on three storeys.
    """
    from .facade_layout import draw_family
    from .texture import FacadeOptions, paint_family, styles_named

    floors = state.get("floors") or [1, 2, 3]
    layouts = os.path.join(out_dir, "layouts")
    sheets = os.path.join(out_dir, "facades")
    if os.path.isdir(sheets) and os.listdir(sheets) and not force:
        return {"kept": len(os.listdir(sheets)), "floors": floors}

    # The wall the sheet will be worn by, not a default. `config.buildings`
    # already says how wide a sheet spans and how tall a storey is; drawing for
    # 24 m and wrapping the result round a 12 m house is what put a hundred
    # small windows on every one of them.
    homes = config.buildings
    drawn = draw_family(layouts, floors, variants=recipe.facade_variants,
                        facade_width=homes.facade_width,
                        floor_height=homes.floor_height,
                        kind=lambda n: "house" if n <= recipe.house_floors else "commercial",
                        seed=recipe.seed, control=True)
    if verbose:
        print(f"[pipeline] {len(drawn['sheets'])} layout(s), "
              f"floor alignment {drawn['floor_alignment']:.2f}")
    # What a wall is faced with depends on how tall the building is. Below
    # `recipe.house_floors` it is a house and gets the residential half of the
    # set; above it, the commercial mid-rise half. Handed the whole set for a
    # street of two-storey houses, the model returns an office block on every
    # plot — which is what the first run through this pipeline produced.
    def styles_for(count: int):
        from .texture import RESIDENTIAL_STYLES

        return styles_named(RESIDENTIAL_STYLES) if count <= recipe.house_floors else None

    painted = paint_family(drawn["control_dir"], sheets, floors=floors,
                           keep_below=recipe.facade_keep_below, styles=styles_for,
                           options=FacadeOptions(count=recipe.facade_count,
                                                 seed=recipe.seed,
                                                 control_scale=recipe.control_scale))
    if verbose:
        print(f"[pipeline] {painted['written']} sheet(s) in {painted['seconds']:.0f}s, "
              f"floor alignment {painted['floor_alignment']:.2f}, "
              f"diversity {painted['diversity']:.3f}")
    return {"floors": floors, "layouts": len(drawn["sheets"]),
            **{k: v for k, v in painted.items() if k != "scores"}}


# ---------------------------------------------------------------------------
# 4. Reconstruction
# ---------------------------------------------------------------------------


def _reconstruct(map_path, out_dir, config, recipe, state, *, force, verbose):
    """Every plot through TRELLIS.2, fitted back to its own footprint."""
    from . import district, scenes

    result = state.get("result") or _rebuild_ground(map_path, config, out_dir, state, verbose)
    handle = scenes.Scene("pipeline", map_path, result, buildings=True,
                          options=config.to_dict())
    models = os.path.join(out_dir, "models")
    ledger = os.path.join(models, "district.json")
    if force and os.path.exists(ledger):
        os.remove(ledger)

    summary = district.rebuild(
        handle, models, facade_dir=_dir(out_dir, "facades"),
        roof_texture=_file(out_dir, "tiles", "kawara.png"),
        roof_tile_metres=recipe.roof_tile_metres, brush_up=recipe.brush_up,
        resolution=recipe.resolution, seed=recipe.seed, keep_below=recipe.keep_below,
        attempts=recipe.attempts, limit=recipe.limit, min_area=recipe.min_area,
        photos=_photographs(out_dir, recipe, state, force=force, verbose=verbose),
        eave_room=recipe.eave_room, roof_room=recipe.roof_room,
        marking_options=config.markings, verbose=verbose)
    state["ledger"] = ledger
    return {k: v for k, v in summary.items() if k != "buildings"}


def _photographs(out_dir: str, recipe: Recipe, state, *, force, verbose):
    """The street's shared photographs, drawn once — or ``None`` for the render path.

    Cheap enough to be a step inside the reconstruction rather than a stage of
    its own: a few dozen pictures against a hundred and eighty buildings. They
    are kept on disk because the whole point is that they are shared, and a
    resumed run that redrew them would change the material of every building it
    had not got to yet.

    One family per floor count, exactly as the facade sheets are and for the
    same reason: the envelope sets the height and the picture sets everything
    else, so a bungalow photographed for a three-storey plot comes back as a
    bungalow nine metres tall.
    """
    if not recipe.envelope:
        return None
    from . import reconstruct as reconstruct_module

    # Houses first, and in the proportion a suburb of 200 m2 plots is built
    # in — see HOUSE_MIX, which repeats a subject to ask for more of it. The
    # commercial subjects only come in if more are asked for than the
    # residential mix has entries.
    known = dict(reconstruct_module.BUILDING_SUBJECTS)
    ordered = ([known[name] for name in reconstruct_module.HOUSE_MIX]
               + [text for name, text in reconstruct_module.BUILDING_SUBJECTS
                  if name not in reconstruct_module.HOUSE_SUBJECTS])
    wanted = ordered[:max(1, recipe.envelope_subjects)]
    floors = sorted(state.get("floors") or [1, 2, 3])
    shapes = [name for name, _ratio, _phrase in reconstruct_module.PROPORTIONS]

    where = os.path.join(out_dir, "subjects")
    families: dict[str, list[str]] = {}
    for row, (storeys, shape, _subject) in enumerate(
            (n, p, s) for n in floors for p in shapes for s in wanted):
        families.setdefault(f"{storeys}f_{shape}", []).append(
            os.path.join(where, f"subject_{storeys}f_{shape}_{row:02d}.png"))
    if all(os.path.exists(p) for group in families.values() for p in group) and not force:
        # Re-measured rather than read off the filename: the name records what
        # the prompt asked for and the family is what the picture turned out to
        # be. Thirty-six PNGs is nothing to measure.
        from PIL import Image

        already = []
        for key, group in families.items():
            for path in group:
                seen = Image.open(path).convert("RGB")
                backdrop = reconstruct_module.backdrop_share(seen)
                edge = reconstruct_module.touches_the_frame(seen)
                already.append({
                    "path": path, "floors": int(key.split("f_", 1)[0]),
                    "aspect": reconstruct_module.silhouette_aspect(seen),
                    "backdrop": backdrop, "edge": edge,
                    # Re-checked, not remembered: the whole point of keeping
                    # them on disk is that nothing about them is written down.
                    "isolated": backdrop >= 0.25 and edge <= 0.08})
        if verbose:
            print(f"[pipeline] {len(already)} photograph(s) already drawn")
        return _families(reconstruct_module.classify_by_aspect(already))

    drawn = reconstruct_module.photographs(
        wanted, where, floors=floors, proportions=shapes,
        options=reconstruct_module.ImageOptions(seed=recipe.seed))
    if verbose:
        thin = [row for row in drawn if not row["isolated"]]
        print(f"[pipeline] {len(drawn)} photograph(s) over floor counts {floors} "
              f"and {len(shapes)} plan shape(s), backdrop "
              f"{min(r['backdrop'] for r in drawn):.2f}-"
              f"{max(r['backdrop'] for r in drawn):.2f}"
              + (f", {len(thin)} still framed badly" if thin else ""))
    state["photographs"] = drawn
    return _families(drawn)


def _families(drawn) -> dict[str, list[str]]:
    """The drawn photographs, grouped by the family each turned out to be in.

    A picture that never passed the framing checks is dropped, so long as its
    family keeps somebody. :func:`~city_builder.reconstruct.photographs` keeps
    the least bad of its tries rather than returning nothing, which is right
    for one picture and wrong for a street: seven of thirty-six failed, and
    what they reconstruct as is confetti, or a building with a lit interior
    showing through the walls. A sibling of the same storey count and plan
    shape is a much better answer than the least bad draw.
    """
    made: dict[str, list[str]] = {}
    dropped: dict[str, list[str]] = {}
    for row in drawn:
        key = f"{int(row['floors'] or 0)}f_{row['proportion']}"
        (made if row.get("isolated", True) else dropped).setdefault(
            key, []).append(row["path"])
    for key, group in dropped.items():
        if key not in made:
            made[key] = group
    return made


# ---------------------------------------------------------------------------
# 5. Scene
# ---------------------------------------------------------------------------


def _scene(map_path, out_dir, config, recipe, state, *, force, verbose):
    """The reconstructions standing on their own ground, painted and written out."""
    from . import district, scenes
    from .build import build_scene

    result = state.get("result") or _rebuild_ground(map_path, config, out_dir, state, verbose)
    handle = scenes.Scene("pipeline", map_path, result, buildings=True,
                          options=config.to_dict())
    ledger = state.get("ledger") or os.path.join(out_dir, "models", "district.json")
    placed = {}
    if os.path.exists(ledger):
        # The cover goes in here too. Placing the reconstructions rebuilds the
        # scene from scratch, so a ground painted by an earlier stage is gone
        # unless it is painted again — which is how a run came out with every
        # building reconstructed and the ground a flat grey.
        placed = district.place(
            handle, ledger, facade_dir=_dir(out_dir, "facades"),
            roof_texture=_file(out_dir, "tiles", "kawara.png"),
            roof_tile_metres=recipe.roof_tile_metres,
            cover_options=_cover(out_dir, config),
            cover_path=os.path.join(out_dir, "ground.png"),
            cover_texels_per_metre=recipe.cover_texels_per_metre,
            marking_options=config.markings, verbose=verbose)
    else:
        build_scene(result, facade_dir=_dir(out_dir, "facades"),
                    roof_texture=_file(out_dir, "tiles", "kawara.png"),
                    roof_tile_metres=recipe.roof_tile_metres,
                    cover_options=_cover(out_dir, config),
                    cover_path=os.path.join(out_dir, "ground.png"),
                    cover_texels_per_metre=recipe.cover_texels_per_metre,
                    marking_options=config.markings, verbose=verbose)

    written = _write(out_dir, recipe)
    shots = _shoot(result, out_dir, map_path) if recipe.renders else []
    return {**placed, "files": written, "renders": shots}


def _write(out_dir: str, recipe: Recipe) -> list[str]:
    from . import scene as scene_module

    blend = os.path.join(out_dir, "scene.blend")
    scene_module.save(blend)
    made = [blend]
    if recipe.glb:
        glb = os.path.join(out_dir, "scene.glb")
        scene_module.export_glb(glb)
        made.append(glb)
    if recipe.fbx:
        # Last, and its failure is not the run's. A district of reconstructions
        # is a few hundred megabytes of texture and the FBX exporter packs a
        # copy of every page into the file; on a full map that is the largest
        # thing this pipeline writes, and the disk running out at the very end
        # would otherwise throw away the scene the run just spent two hours on.
        fbx = os.path.join(out_dir, "scene.fbx")
        try:
            scene_module.export_fbx(fbx)
            made.append(fbx)
        except (RuntimeError, OSError) as error:
            print(f"[pipeline] the FBX was not written: {error}")
    return made


def _shoot(result, out_dir: str, map_path: str = "") -> list[str]:
    """An aerial, a block and a view from the carriageway."""
    import bpy
    import mathutils
    import numpy as np

    from . import lanelet as lanelet_module
    from . import route as route_module
    from . import scene as scene_module

    if not any(obj.type == "LIGHT" for obj in bpy.data.objects):
        scene_module.sunlit(elevation=32.0, azimuth=130.0)
    data = bpy.data.cameras.new("pipeline_cam")
    data.clip_end = 8000.0
    cam = bpy.data.objects.new("pipeline_cam", data)
    bpy.context.scene.collection.objects.link(cam)
    board = bpy.context.scene
    board.camera = cam
    board.render.engine = "BLENDER_EEVEE"
    board.eevee.taa_render_samples = 96
    board.render.film_transparent = False
    board.render.image_settings.file_format = "PNG"
    board.render.resolution_x, board.render.resolution_y = 1600, 900

    def shoot(name, location, look, lens):
        data.lens = lens
        cam.location = location
        target = mathutils.Vector(look)
        cam.rotation_euler = (target - mathutils.Vector(location)).to_track_quat(
            "-Z", "Y").to_euler()
        board.render.filepath = os.path.join(out_dir, name)
        bpy.ops.render.render(write_still=True)
        return board.render.filepath

    centres = np.array([[p["centroid"][0], p["centroid"][1]] for p in result.plots])
    mid = centres.mean(axis=0) if len(centres) else np.zeros(2)
    made = [shoot("aerial.png", (mid[0] - 150, mid[1] - 150, 105.0), (mid[0], mid[1], 2.0), 42.0),
            shoot("block.png", (mid[0] - 60, mid[1] - 60, 32.0), (mid[0] - 12, mid[1] - 12, 3.0), 40.0)]
    try:
        # From the carriageway, because a scene for a driving simulator is
        # wrong in ways an aerial cannot show.
        _ll2, _projector, lmap = lanelet_module.load_map(
            map_path, projector="utm",
            origin_lat=result.frame.ref_lat, origin_lon=result.frame.ref_lon)
        path = route_module.drive_path(result.groups,
                                       lanelet_module.lanelet_end_keys(lmap), step=1.0)
        position, look = path[int(len(path) * 0.3)]
        made.append(shoot("street.png", position, look, 30.0))
    except Exception as error:  # noqa: BLE001 - a view is not worth failing a run over
        print(f"[pipeline] no street view: {type(error).__name__}: {error}")
    return made


# ---------------------------------------------------------------------------


def _rebuild_ground(map_path, config, out_dir, state, verbose):
    """Stage 1 again, for a run that starts partway through."""
    from .build import build_city_from_config

    result = build_city_from_config(map_path, config, buildings=True, verbose=verbose,
                                    cover_options=_cover(out_dir, config))
    state["result"] = result
    state["floors"] = sorted({plot["floors"] for plot in result.plots})
    return result


def _dir(out_dir: str, name: str) -> str | None:
    path = os.path.join(out_dir, name)
    return path if os.path.isdir(path) and os.listdir(path) else None


def _file(out_dir: str, *parts: str) -> str | None:
    path = os.path.join(out_dir, *parts)
    return path if os.path.exists(path) else None


_STAGES = {
    "ground": _ground,
    "materials": _materials,
    "facades": _facades,
    "reconstruct": _reconstruct,
    "scene": _scene,
}
