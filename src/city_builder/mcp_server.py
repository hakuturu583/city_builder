"""city_builder as an MCP server, for an agent to drive.

The tools are not the CLI with a different coat on. Two things change when the
caller is a language model rather than a person at a shell.

**It cannot see.** A person runs a build, opens the .blend and knows in a
second whether the road has holes in it. An agent gets a JSON blob. So every
tool that changes something answers with measurements, ``survey`` exists at
all, and the render tools hand back an actual image the agent can look at
rather than a path it has to take on trust.

**It cannot afford to rebuild.** Building a map is twenty seconds and the
result is wanted by export, measurement and rendering alike, so a build is
named and kept. Blender is a singleton — one process, one scene — so the handle
holds the geometry and anything wanting Blender rebuilds into it on demand.
Exporting twice costs twice, and that is worth knowing before asking.

Long and expensive operations say so in their descriptions, because the agent
is choosing between them without a wall clock in front of it: ``generate_facades``
wants a GPU and minutes, ``render_drive`` is minutes, everything else is seconds.

    uv run --extra mcp city-builder-mcp
"""

from __future__ import annotations

import os
import time
from typing import Annotated, Any, Literal

from pydantic import Field

from .scenes import SceneStore, survey

STORE = SceneStore()


def _server():
    from mcp.server.mcpserver import MCPServer

    return MCPServer(
        name="city-builder",
        instructions=(
            "Builds textured 3D city scenes from Lanelet2 HD maps.\n\n"
            "Start with inspect_map to see what a map contains, then build to get a "
            "scene handle. Everything else takes that handle. survey is how you check "
            "your own work — it reports holes in the drivable surface, how much of the "
            "ground is measured rather than guessed, and how far the elevated roads "
            "run above what. render_view returns a picture you can look at; prefer it "
            "over guessing from numbers alone.\n\n"
            "Facade textures are optional and come in two steps: make_layouts (fast, "
            "no GPU) draws the structure, generate_facades (GPU, minutes) paints it. "
            "A scene exports perfectly well without them, in flat colours.\n\n"
            "refine_render takes a drive rendered from a scene and makes it photoreal "
            "with a video model, keeping the geometry. Its frames are reference for a "
            "second pass at the textures, which is the loop: build, render, refine, "
            "re-texture.\n\n"
            "To work on one building rather than the street: list_buildings says which "
            "are worth the time, render_orbit circles one and masks it, and the clip it "
            "writes is what refine_render takes."
        ),
    )


server = _server()


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


@server.tool()
def inspect_map(
    map_path: Annotated[str, Field(description="Path to a Lanelet2 .osm map")],
) -> dict[str, Any]:
    """What a Lanelet2 map contains, without building anything. Fast.

    Read this before building: it tells you whether the map has the markings,
    kerbs and crossings the scene can draw, and whether its roads sit at more
    than one height, which is what decides if there is any elevated structure
    to build.
    """
    import collections

    import numpy as np

    from . import lanelet

    ll2, projection, lmap = lanelet.load_map(map_path)
    lanelets = collections.Counter(
        lanelet.attributes(x).get("subtype", "?") for x in lmap.laneletLayer)
    linestrings = collections.Counter(
        lanelet.attributes(x).get("type", "?") for x in lmap.lineStringLayer)
    lat, lon = lanelet.map_centroid(ll2, projection, lmap)

    heights = [p.z for p in lmap.pointLayer]
    spread = (round(float(np.percentile(heights, 90) - np.percentile(heights, 10)), 1)
              if heights else 0.0)
    return {
        "map": map_path,
        "centroid": [round(lat, 7), round(lon, 7)],
        "lanelets": dict(lanelets.most_common()),
        "linestrings": dict(linestrings.most_common()),
        "height_spread_p10_p90_m": spread,
        "draws": {
            "lane_markings": linestrings.get("line_thin", 0) + linestrings.get("line_thick", 0),
            "stop_lines": linestrings.get("stop_line", 0),
            "crossings": lanelets.get("crosswalk", 0),
            "kerbs": linestrings.get("road_border", 0),
        },
        "note": ("a spread over about 5 m usually means more than one road level, "
                 "so expect elevated structure"),
    }


@server.tool()
def describe_options() -> list[dict[str, Any]]:
    """Every build option: section, key, type and default. Fast.

    Pass the ones you want to `build` as `options`, keyed by section — for
    example {"buildings": {"coverage": 0.4}, "viaduct": {"parapet_height": 1.2}}.
    An unknown key is an error rather than a shrug.
    """
    from .config import describe

    return [{"section": s, "key": k, "type": str(t), "default": d} for s, k, t, d in describe()]


@server.tool()
def list_styles() -> list[dict[str, str]]:
    """The facade characters `generate_facades` spreads its sheets across. Fast."""
    from .texture import FACADE_STYLES

    return [{"name": name, "prompt": prompt} for name, prompt in FACADE_STYLES]


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


@server.tool()
def build(
    map_path: Annotated[str, Field(description="Path to a Lanelet2 .osm map")],
    buildings: Annotated[bool, Field(description="Fill the empty blocks with procedural buildings")] = True,
    options: Annotated[dict[str, Any] | None,
                       Field(description="Overrides by section; see describe_options")] = None,
) -> dict[str, Any]:
    """Build a map into geometry and keep it. Seconds to a minute.

    Returns a scene handle every other tool takes, plus the survey. Nothing
    touches Blender here — this is numpy and shapely, so it is cheap to hold
    and cheap to measure.
    """
    from .build import build_city_from_config
    from .config import CityConfig

    config = CityConfig.from_dict(options)
    started = time.time()
    result = build_city_from_config(map_path, config, buildings=buildings, verbose=False)
    scene = STORE.add(map_path, result, buildings=buildings, options=options or {})
    return {"took_seconds": round(time.time() - started, 1), **survey(scene)}


@server.tool()
def list_scenes() -> list[dict[str, Any]]:
    """The scenes this process is holding. Fast."""
    return [scene.summary() for scene in STORE.all()]


@server.tool()
def survey_scene(
    scene: Annotated[str, Field(description="A handle from build")],
) -> dict[str, Any]:
    """Everything worth knowing about a built scene, in numbers. Fast.

    Holes in the drivable surface split by level and by width — a *seam* is a
    hole nowhere a metre across, which is a defect and should be zero; an
    *opening* is a traffic island or a gap that belongs. How much of the ground
    is measured rather than interpolated. How far the elevated roads run above
    what. How far a camera can drive.
    """
    return survey(STORE.get(scene))


@server.tool()
def forget_scene(
    scene: Annotated[str, Field(description="A handle from build")],
) -> dict[str, str]:
    """Drop a scene from memory. Fast."""
    STORE.drop(scene)
    return {"dropped": scene}


# ---------------------------------------------------------------------------
# Textures
# ---------------------------------------------------------------------------


def _require_diffusion() -> None:
    """Fail with the reason rather than with a missing module.

    The published container leaves the diffusion stack out — it is several
    gigabytes of CUDA that can do nothing without a GPU on the host — so an
    agent calling these tools there gets an ImportError from three frames down
    unless it is told what is actually going on.
    """
    try:
        import diffusers  # noqa: F401
        import torch  # noqa: F401
    except ImportError as missing:
        raise RuntimeError(
            f"the diffusion stack is not installed here ({missing.name}). "
            "This tool needs a GPU and the `texture` extra: `uv sync --extra texture`, "
            'or build the container with --build-arg EXTRAS="mcp texture" and run it '
            "with --gpus all. Everything else in this server works without it, and "
            "`make_layouts` draws usable stand-in sheets with no model at all."
        ) from missing


@server.tool()
def make_layouts(
    out_dir: Annotated[str, Field(description="Directory to write the layouts into")],
    floors: Annotated[str, Field(description="Floor counts, e.g. '2-8' or '4,6,9'")] = "2-8",
    variants: Annotated[int, Field(description="Structurally different layouts per floor count")] = 3,
    facade_width: Annotated[float, Field(description="Wall one sheet spans (m)")] = 12.0,
) -> dict[str, Any]:
    """Draw facade layouts and their control images. Seconds, no GPU.

    One family per floor count, because the facade UV normalises over a
    building's height: a sheet drawn for six floors only reads correctly on a
    six-floor building. Ask `survey_scene` for the floor counts a scene needs.

    The sheets this writes are plain stand-ins — correct, not photographic —
    and a scene can be exported with them as they are. `generate_facades` turns
    the control images into painted ones.
    """
    import random

    from .facade_layout import (
        bay_alignment,
        bays_for,
        control_image,
        floor_alignment,
        procedural_facade,
        sample_layout,
        sheet_name,
    )
    from .texture import save_tile

    counts = []
    for part in floors.split(","):
        part = part.strip()
        if "-" in part:
            low, high = (int(v) for v in part.split("-", 1))
            counts.extend(range(low, high + 1))
        elif part:
            counts.append(int(part))
    if not counts:
        raise ValueError(f"no floor counts in {floors!r}")

    control_dir = os.path.join(out_dir, "control")
    os.makedirs(control_dir, exist_ok=True)
    written, scores = 0, []
    for count in sorted(set(counts)):
        for variant in range(variants):
            rng = random.Random(1000 * count + variant)
            layout = sample_layout(count, rng, facade_width=facade_width)
            width, height = layout.pixel_size()
            save_tile(control_image(layout, width, height),
                      os.path.join(control_dir, sheet_name(count, variant, "control")))
            sheet = procedural_facade(layout, width, height, seed=1000 * count + variant)
            save_tile(sheet, os.path.join(out_dir, sheet_name(count, variant)))
            scores.append((floor_alignment(sheet, layout), bay_alignment(sheet, layout)))
            written += 1

    return {
        "dir": out_dir,
        "control_dir": control_dir,
        "sheets": written,
        "floor_counts": sorted(set(counts)),
        "bays_per_sheet": bays_for(facade_width),
        "worst_floor_alignment": round(min(s[0] for s in scores), 2),
        "worst_bay_alignment": round(min(s[1] for s in scores), 2),
        "note": "alignment above 0.6 means the windows sit on the storeys as drawn",
    }


@server.tool()
def generate_facades(
    layouts_dir: Annotated[str, Field(description="Directory from make_layouts")],
    out_dir: Annotated[str, Field(description="Directory to write the painted sheets into")],
    prompts: Annotated[list[str] | None,
                       Field(description="What the buildings are made of, one prompt per "
                                         "material; spread over the sheets. Beats `styles`")] = None,
    styles: Annotated[list[str] | None,
                      Field(description="Names from list_styles, to narrow the built-in "
                                        "spread instead of writing prompts")] = None,
    negative: Annotated[str | None, Field(description="What to keep out of the sheets")] = None,
    reference_images: Annotated[list[str] | None,
                                Field(description="Photographs to take the material from — "
                                                  "refine_render's frames are the intended "
                                                  "source. One is used per control image, "
                                                  "cycled")] = None,
    reference_strength: Annotated[float,
                                  Field(description="How literally to quote them; 0.4 takes "
                                                    "the palette and material, 0.7 starts "
                                                    "copying content")] = 0.4,
    count: Annotated[int, Field(description="Sheets per floor count")] = 4,
    variation: Annotated[float, Field(description="0 = identical siblings, 1 = strangers")] = 0.45,
    seed: Annotated[int, Field(description="Same seed and prompts give the same sheets")] = 0,
    family: Annotated[Literal["sd15", "sdxl"], Field(description="Which model stack")] = "sd15",
    controlnet: Annotated[Literal["canny", "mlsd", "none"],
                          Field(description="What holds the windows on the storeys")] = "mlsd",
    keep_below: Annotated[float, Field(description="Discard sheets scoring under this")] = 0.5,
    vram_budget_gb: Annotated[float, Field(description="Cap, for a shared card")] = 10.0,
):
    """Paint the layouts with a diffusion model. **Needs a GPU. Minutes.**

    About a second a sheet once the model is loaded, plus ten seconds to load.
    Check `city-builder models` has the weights before calling, and that the
    card is free — this is the only tool here that competes for one.

    **The prompt is the whole of the material** — unless you pass photographs.
    The control image fixes the architecture, where the floors and windows are,
    so what is left is what the building is *made of*. One prompt gives a street
    built entirely of one material; pass several and they are spread across the
    sheets. `reference_images` answers the same question with a picture instead
    of words, which is the second half of the loop: refine a render into
    photoreal frames, then hand those frames back here. Each is given a suffix that keeps the result
    usable as a texture (flat elevation, overcast, no sky, no perspective),
    so write the material, not the photograph: "photograph of a red brick
    warehouse facade, steel window frames" rather than "a street at sunset".

    `controlnet` is not optional in practice: without it the model returns a
    wall with windows somewhere, scoring about zero against the layout. Every
    sheet is scored before it is written, and `keep_below` drops the ones that
    lost the structure.

    Answers with a contact sheet of what it kept, one row per floor count, so
    you can see the street you asked for rather than infer it from a diversity
    number.
    """
    _require_diffusion()

    import numpy as np
    from mcp.server.mcpserver import Image
    from PIL import Image as PILImage

    from .facade_layout import alignment, diversity, saturation, sheet_floors, sheet_name
    from .texture import (
        COMMON_PROMPT,
        FacadeOptions,
        facade_sheets,
        load_facade_pipeline,
        save_tile,
        styled_prompts,
        styles_named,
    )

    control_dir = os.path.join(layouts_dir, "control")
    if not os.path.isdir(control_dir):
        raise ValueError(f"no control images in {control_dir}; call make_layouts first")

    controls = [(sheet_floors(f), os.path.join(control_dir, f))
                for f in sorted(os.listdir(control_dir)) if f.endswith(".png")]
    controls = [(n, p) for n, p in controls if n is not None]
    if not controls:
        raise ValueError(f"no usable control images in {control_dir}")

    asked = [f"{p}, {COMMON_PROMPT}" for p in prompts] if prompts else None
    palette = styles_named(styles) if styles else None

    options = FacadeOptions(count=count, family=family, vram_budget_gb=vram_budget_gb,
                            variation=variation, seed=seed,
                            reference=bool(reference_images),
                            reference_strength=reference_strength,
                            controlnet="" if controlnet == "none" else controlnet)
    os.makedirs(out_dir, exist_ok=True)

    started = time.time()
    pipeline = load_facade_pipeline(options)
    loaded = time.time()

    written, dropped, scores, kept = 0, 0, [], []
    rows: dict[int, list] = {}
    for index, (floors, path) in enumerate(controls):
        control = np.asarray(PILImage.open(path).convert("RGB"))
        if asked:
            # Cycled rather than repeated, so a small run still covers every
            # material asked for instead of drawing the first one every time.
            wanted = [asked[(index * count + k) % len(asked)] for k in range(count)]
        elif palette:
            wanted = styled_prompts(count, seed=seed + index * count, styles=palette)
        else:
            wanted = styled_prompts(count, seed=seed + index * count)
        reference = (reference_images[index % len(reference_images)]
                     if reference_images else None)
        for sheet in facade_sheets(wanted, control, options, pipeline=pipeline,
                                   negative_prompt=negative or "", reference=reference):
            score = alignment(sheet, control, axis=0)
            scores.append(score)
            if score < keep_below:
                dropped += 1
                continue
            save_tile(sheet, os.path.join(out_dir, sheet_name(floors, written)))
            kept.append(sheet)
            rows.setdefault(floors, []).append(sheet)
            written += 1

    report = {
        "dir": out_dir,
        "sheets": written,
        "dropped": dropped,
        "prompts": len(asked) if asked else len(palette or ()) or "the built-in style set",
        "load_seconds": round(loaded - started, 1),
        "paint_seconds": round(time.time() - loaded, 1),
        "mean_floor_alignment": round(float(np.mean(scores)), 2) if scores else 0.0,
        "diversity": round(diversity(kept), 3) if kept else 0.0,
        "saturation": [round(min(saturation(s) for s in kept), 2),
                       round(max(saturation(s) for s in kept), 2)] if kept else [0, 0],
        "note": ("alignment above 0.6 is drawn-to-spec, near 0 is a wall with windows "
                 "somewhere; diversity near 0.05 is one material, 0.4 is the whole set"),
    }
    contact = _contact_sheet(rows)
    return [report, Image(data=contact, format="png")] if contact else report


def _contact_sheet(rows: dict[int, list], *, cell=(160, 240), across: int = 12) -> bytes | None:
    """One row per floor count, so a whole run can be looked at in one image."""
    import io

    from PIL import Image as PILImage

    if not rows:
        return None
    wide = min(across, max(len(v) for v in rows.values()))
    sheet = PILImage.new("RGB", (wide * cell[0], len(rows) * cell[1]), "white")
    for row, floors in enumerate(sorted(rows)):
        for column, tile in enumerate(rows[floors][:wide]):
            sheet.paste(PILImage.fromarray(tile).resize(cell), (column * cell[0], row * cell[1]))
    buffer = io.BytesIO()
    sheet.save(buffer, format="PNG")
    return buffer.getvalue()


@server.tool()
def make_tile(
    out_path: Annotated[str, Field(description="Where to write the tile PNG")],
    prompt: Annotated[str, Field(description="What the surface is made of")] =
        "seamless top-down photograph of urban asphalt with fine gravel",
    negative: Annotated[str | None, Field(description="What to keep out of it")] = None,
    procedural: Annotated[bool, Field(description="Skip the model; filtered noise, no GPU")] = False,
    size: Annotated[int, Field(description="Pixels, square")] = 1024,
    steps: Annotated[int, Field(description="Sampling steps")] = 24,
    seed: Annotated[int, Field(description="Same seed and prompt give the same tile")] = 0,
    vram_budget_gb: Annotated[float, Field(description="Cap, for a shared card")] = 6.0,
):
    """A tileable surface texture, for the ground or the carriageway.

    **With a model: GPU, about a minute.** With `procedural=True`: instant, no
    GPU, and genuinely seamless since it is periodic by construction.

    Write the *material at the scale it is seen*, and say how far the tile
    spans when you use it: `export` repeats the ground tile every 12 m and the
    carriageway tile every 4, so "fine gravel aggregate" is right for a road
    and "paving slabs" for a pavement. Getting that backwards is how asphalt
    ends up with a grain a foot across and the road renders as cobblestone.

    Answers with the tile itself as well as its seam score — a wrap that scores
    well can still be the wrong material.
    """
    if not procedural:
        _require_diffusion()

    from mcp.server.mcpserver import Image

    from .texture import TextureOptions, seam_error
    from .texture import make_tile as build_tile

    options = TextureOptions(size=size, steps=steps, seed=seed,
                             vram_budget_gb=vram_budget_gb, diffusion=not procedural)
    tile = build_tile(prompt, options, path=out_path, negative_prompt=negative or "")
    report = {
        "path": out_path,
        "seam_error": round(seam_error(tile), 2),
        "note": "1.0 means the wrap looks like the texture's own variation",
    }
    return [report, Image(data=_as_png(tile, 512), format="png")]


def _as_png(tile, size: int) -> bytes:
    import io

    from PIL import Image as PILImage

    buffer = io.BytesIO()
    PILImage.fromarray(tile).resize((size, size)).save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Photoreal refinement
# ---------------------------------------------------------------------------


@server.tool()
def refine_render(
    video: Annotated[str, Field(description="A rendered clip, e.g. what render_drive wrote")],
    out_dir: Annotated[str, Field(description="Directory to write the refined frames into")],
    prompt: Annotated[str | None,
                      Field(description="What the street should look like; the default is "
                                        "an overcast Japanese city street")] = None,
    denoise: Annotated[float,
                       Field(description="How far from the render to go. See the table in "
                                         "the description — 0.25 keeps this street")] = 0.25,
    length: Annotated[int, Field(description="Frames; must be 5, 22, 39 … (17k+5)")] = 5,
    width: Annotated[int, Field(description="Pixels, multiple of 32")] = 832,
    height: Annotated[int, Field(description="Pixels, multiple of 32")] = 480,
    seed: Annotated[int, Field(description="Same seed and prompt give the same frames")] = 0,
):
    """Make a render photoreal with MiniMax H3. **Needs a GPU. A minute or two.**

    The scene has its geometry right and its appearance approximate. This puts
    the render in as the *starting latent* and returns only part of the noise,
    so the geometry is held by what is already there and the model supplies the
    surfaces. About 95 s for five frames at 832x480 on a 24 GB card.

    `denoise` is the whole dial, and these are measurements, not guesses:

    | 0.25 | the same street, photoreal — masses, road width, vanishing point |
    | 0.35 | more convincing, and the buildings start becoming generic |
    | 0.45+ | a real street, but no longer *this* one |

    Keep it low when the frames are wanted as reference for `generate_facades`:
    a reference whose windows sit on different floors than the building it
    describes is worse than no reference at all.

    Answers with the frames themselves, because whether the street survived is
    not something the numbers can tell you.
    """
    _require_comfy()

    from mcp.server.mcpserver import Image

    from .refine import DEFAULT_PROMPT, RefineOptions, refine

    options = RefineOptions(length=length, width=width, height=height,
                            denoise=denoise, seed=seed)
    report = refine(video, out_dir, prompt=prompt or DEFAULT_PROMPT, options=options)
    strip = _film_strip(report["frames"])
    return [report, Image(data=strip, format="png")] if strip else report


def _require_comfy() -> None:
    """Fail with the reason rather than with a missing directory."""
    import os.path

    from .refine import Comfy

    root = Comfy().root
    if not os.path.exists(os.path.join(root, "main.py")):
        raise RuntimeError(
            f"no ComfyUI at {root}. The published container ships one; from a source "
            "install, clone ComfyUI and the ComfyUI-GGUF node, link "
            "src/city_builder/comfy_nodes into its custom_nodes, and point COMFYUI_PATH "
            "at it. The weights are a further ~35 GB; see the README.")


def _film_strip(paths, *, height: int = 200, across: int = 5) -> bytes | None:
    """The frames in a row, so a caller can see what came back."""
    import io

    from PIL import Image as PILImage

    frames = [PILImage.open(p).convert("RGB") for p in paths[:across]]
    if not frames:
        return None
    scaled = [f.resize((round(f.width * height / f.height), height)) for f in frames]
    strip = PILImage.new("RGB", (sum(f.width for f in scaled), height), "white")
    x = 0
    for frame in scaled:
        strip.paste(frame, (x, 0))
        x += frame.width
    buffer = io.BytesIO()
    strip.save(buffer, format="PNG")
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@server.tool()
def export(
    scene: Annotated[str, Field(description="A handle from build")],
    out_dir: Annotated[str, Field(description="Directory to write into")],
    formats: Annotated[list[Literal["blend", "glb", "fbx"]] | None,
                       Field(description="Which files to write; glb if omitted")] = None,
    facade_dir: Annotated[str | None, Field(description="Painted or drawn facade sheets")] = None,
    road_texture: Annotated[str | None, Field(description="Tile for the carriageway")] = None,
    ground_texture: Annotated[str | None, Field(description="Tile for the terrain")] = None,
) -> dict[str, Any]:
    """Write a scene out. Tens of seconds.

    This is where Blender is entered, so it costs the same every time — the
    handle holds geometry, not a Blender scene.

    glTF carries the surface class of every object as extras and the mask as a
    custom attribute, so a texturing pass downstream can tell a lane line from
    asphalt. FBX carries the class as user properties but not the mask, because
    FBX has one vertex colour set and using it would tint the asset.
    """
    from .build import build_scene
    from .config import CityConfig

    formats = formats or ["glb"]
    held = STORE.get(scene)
    config = CityConfig.from_dict(held.options)
    name = os.path.splitext(os.path.basename(held.map_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    paths = {kind: os.path.join(out_dir, f"{name}.{kind}") for kind in formats}
    started = time.time()
    build_scene(held.result, blend=paths.get("blend"), glb=paths.get("glb"),
                fbx=paths.get("fbx"), facade_dir=facade_dir, road_texture=road_texture,
                ground_texture=ground_texture, marking_options=config.markings, verbose=False)
    return {
        "took_seconds": round(time.time() - started, 1),
        "files": {kind: {"path": p, "bytes": os.path.getsize(p) if os.path.exists(p) else 0}
                  for kind, p in paths.items()},
    }


# ---------------------------------------------------------------------------
# Look at it
# ---------------------------------------------------------------------------


@server.tool()
def render_view(
    scene: Annotated[str, Field(description="A handle from build")],
    view: Annotated[Literal["aerial", "plan", "street"],
                    Field(description="aerial: oblique over the map. plan: straight down, "
                                      "good for road layout. street: eye height on the route")] = "aerial",
    hide_buildings: Annotated[bool, Field(description="Leave them out, to see the roads")] = False,
    facade_dir: Annotated[str | None,
                          Field(description="Facade sheets to dress it with; flat colours without")] = None,
    width: Annotated[int, Field(description="Pixels")] = 900,
    height: Annotated[int, Field(description="Pixels")] = 600,
    samples: Annotated[int, Field(description="More is slower and cleaner")] = 32,
):
    """Render a scene and hand back the picture. Tens of seconds.

    Use this. Numbers say whether the road has holes; only a look says whether
    it reads as a street. `plan` with `hide_buildings` is the clearest view of
    what the road surface actually became.

    Answers with the framing alongside the image, since a `street` view falls
    back to `aerial` on a map with no drivable chain and you would otherwise be
    reading the wrong picture.
    """
    import tempfile

    from mcp.server.mcpserver import Image

    from ._render import render_still
    from .config import CityConfig

    held = STORE.get(scene)
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as handle:
        path = handle.name
    framing = render_still(held, view=view, hide_buildings=hide_buildings, out_path=path,
                           resolution=(width, height), samples=samples, facade_dir=facade_dir,
                           marking_options=CityConfig.from_dict(held.options).markings)
    with open(path, "rb") as handle:
        data = handle.read()
    os.unlink(path)
    framing.pop("path", None)
    framing["scene"] = held.name
    return [framing, Image(data=data, format="png")]


@server.tool()
def render_drive(
    scene: Annotated[str, Field(description="A handle from build")],
    out_path: Annotated[str, Field(description="Where to write the .mp4")],
    seconds: Annotated[float, Field(description="Clip length; the route may be shorter")] = 15.0,
    speed: Annotated[float, Field(description="Metres per second")] = 11.0,
    facade_dir: Annotated[str | None, Field(description="Facade sheets to dress it with")] = None,
    width: Annotated[int, Field(description="Pixels")] = 1280,
    height: Annotated[int, Field(description="Pixels")] = 720,
) -> dict[str, Any]:
    """Drive a camera along the roads and render it. **Minutes.**

    About a second a frame at 720p, so fifteen seconds of footage is a few
    minutes. `survey_scene` reports how long the route actually is; asking for
    more than that just renders what there is.
    """
    import shutil
    import tempfile

    from .build import build_scene
    from .build import render_drive as drive
    from .config import CityConfig

    held = STORE.get(scene)
    config = CityConfig.from_dict(held.options)
    workdir = tempfile.mkdtemp(prefix="city-drive-")
    blend = os.path.join(workdir, "scene.blend")
    # The drive renders in a Blender subprocess that reads this .blend, so the
    # mask pages have to still be on disk when it does — hence one directory
    # holding both, removed once the video exists.
    build_scene(held.result, blend=blend, facade_dir=facade_dir,
                marking_options=config.markings, verbose=False)

    started = time.time()
    try:
        written = drive(held.result, held.map_path, blend, out_path, seconds=seconds,
                        speed=speed, resolution=(width, height), verbose=False)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
    if written is None:
        return {"video": None, "note": "no drivable route in this map"}
    return {
        "video": written,
        "bytes": os.path.getsize(written),
        "took_seconds": round(time.time() - started, 1),
    }


@server.tool()
def list_buildings(
    scene: Annotated[str, Field(description="A handle from build")],
    limit: Annotated[int, Field(description="How many to list, biggest first")] = 20,
    min_area: Annotated[float, Field(description="Skip plots smaller than this, in m2")] = 0.0,
) -> dict[str, Any]:
    """Every building in a scene, biggest first, with what circling it would cost.

    Which building to reconstruct is a choice, and this is what it is made
    from. A map yields hundreds of plots and most of them are not worth a video
    model's time — `area_m2` and `floors` are how to tell, and `orbit_distance_m`
    is how far back the camera would have to stand.
    """
    from . import orbit as orbit_module

    held = STORE.get(scene)
    plots = held.result.plots
    if not plots:
        return {"scene": held.name, "buildings": 0,
                "note": "this scene was built without buildings"}

    rows = []
    for index, plot in enumerate(plots):
        if plot["area"] < min_area:
            continue
        row = {"building": index, "area_m2": plot["area"], "height_m": plot["height"],
               "floors": plot["floors"], "centroid": plot["centroid"]}
        if plot.get("footprint"):
            shot = orbit_module.plan_orbit(plot)
            row["orbit_distance_m"] = shot["distance_m"]
        rows.append(row)
    rows.sort(key=lambda row: row["area_m2"], reverse=True)
    return {
        "scene": held.name,
        "buildings": len(plots),
        "listed": len(rows[:limit]),
        "plots": rows[:limit],
    }


@server.tool()
def render_orbit(
    scene: Annotated[str, Field(description="A handle from build")],
    building: Annotated[int, Field(description="Which one; see list_buildings")],
    out_dir: Annotated[str, Field(description="Directory for the clip, the masks and the plan")],
    frames: Annotated[int, Field(description="Must be 5, 22, 39, 56 … (17k+5). 56 and 124 "
                                             "are the ones that quarter exactly")] = 56,
    neighbours: Annotated[Literal["clear", "keep", "hide"],
                          Field(description="clear: empty the disc the camera flies over. "
                                            "keep: leave the block standing. "
                                            "hide: this building alone")] = "clear",
    elevation: Annotated[float, Field(description="Degrees above the horizon")] = 12.0,
    facade_dir: Annotated[str | None, Field(description="Facade sheets to dress it with")] = None,
    width: Annotated[int, Field(description="Pixels, multiple of 32")] = 832,
    height: Annotated[int, Field(description="Pixels, multiple of 32")] = 480,
):
    """Circle one building and mask it, for a reconstruction. **A minute or two.**

    Writes `orbit.mp4` (the clip `refine_render` takes), `mask/` (a PNG per
    frame, white where the building is) and `orbit.json` (the centre, the
    camera distance, the quadrant frames and the footprint — everything needed
    to put a reconstructed mesh back at the right size).

    `frames` is not free: H3 counts in 17k+5 and a closed turn divides 360° by
    the count, so only 56 and 124 land a frame exactly on each of the four
    cardinal views a multiview reconstruction wants. 56 is the cheap one.

    `neighbours` was decided by measurement, not taste. With the block left
    standing, the camera flies at the framing distance — which is inside the
    next block — and 29 of 56 frames saw no part of the subject at all. `clear`
    empties the disc every sightline lies inside and leaves the rest of the
    city standing: measured on the same scene, the subject was in all 56 frames
    at the same size as if it stood alone, with 37 of 53 neighbours still there.

    Answers with the four quadrant views, because whether the building reads as
    a building is not something the numbers can tell you.
    """
    import subprocess

    from mcp.server.mcpserver import Image

    from .config import CityConfig
    from .orbit import OrbitOptions
    from .orbit import render_orbit as run

    held = STORE.get(scene)
    options = OrbitOptions(frames=frames, neighbours=neighbours, elevation_deg=elevation,
                           width=width, height=height)
    report = run(held, building, out_dir, options=options, facade_dir=facade_dir,
                 marking_options=CityConfig.from_dict(held.options).markings, verbose=False)

    picks = report.get("quadrant_frames")
    if not picks:
        return report
    views = os.path.join(out_dir, "views")
    os.makedirs(views, exist_ok=True)
    select = "+".join(rf"eq(n\,{n})" for n in picks)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", report["video"],
                    "-vf", f"select='{select}'", "-vsync", "0",
                    os.path.join(views, "view_%02d.png")], check=True)
    paths = sorted(os.path.join(views, f) for f in os.listdir(views) if f.endswith(".png"))
    report["views"] = paths
    strip = _film_strip(paths, across=4)
    return [report, Image(data=strip, format="png")] if strip else report


def main() -> None:
    """Run the server on stdio."""
    import anyio

    anyio.run(server.run_stdio_async)


if __name__ == "__main__":
    main()
