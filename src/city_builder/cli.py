"""Command line interface."""

from __future__ import annotations

import click

from . import ground as ground_module


@click.group()
@click.version_option(package_name="city-builder")
def main():
    """Build ground and road-surface meshes from a Lanelet2 HD map."""


@main.command("build")
@click.option("--input", "input_path", required=True, help="Lanelet2 HD map (.osm)")
@click.option("--output", "blend", default=None, help="Output .blend")
@click.option("--glb", default=None, help="Also export a .glb")
@click.option("--heightmap", "heightmap_path", default=None, help="Also write the ground heightmap JSON")
@click.option("--manifest", "manifest_path", default=None,
              help="Also write the surface manifest: class and paint policy per group")
@click.option("--ref-lat", type=float, default=None, help="Scene anchor; defaults to the map centroid")
@click.option("--ref-lon", type=float, default=None)
@click.option("--projector", type=click.Choice(["utm", "mercator", "local-cartesian"]), default="utm")
@click.option("--z-datum", type=float, default=None, help="Map elevation placed at --z-offset")
@click.option("--z-offset", type=float, default=0.0, help="Scene height of the datum")
# surfaces
@click.option("--max-segment", type=float, default=None, help="Resample spacing along a lanelet (m)")
@click.option("--marking-width", type=float, default=None)
@click.option("--stop-line-width", type=float, default=None)
@click.option("--dash-length", type=float, default=None)
@click.option("--dash-gap", type=float, default=None)
@click.option("--curb-height", type=float, default=None)
@click.option("--crosswalks/--no-crosswalks", default=None)
@click.option("--walkways/--no-walkways", default=None)
@click.option("--markings/--no-markings", default=None)
@click.option("--stop-lines/--no-stop-lines", default=None)
@click.option("--crosswalk-stripes/--no-crosswalk-stripes", default=None,
              help="Zebra bars from the map's pedestrian_marking rings")
@click.option("--curbs/--no-curbs", default=None, help="Stand road_border lines up into kerbs")
# ground
@click.option("--ground/--no-ground", default=True)
@click.option("--cell", type=float, default=ground_module.DEFAULT_CELL, help="Ground grid cell (m)")
@click.option("--smooth", type=float, default=ground_module.DEFAULT_SMOOTH, help="Smoothing radius in cells")
@click.option("--z-gap", type=float, default=ground_module.DEFAULT_Z_GAP,
              help="Separation before two overlapping lanelets count as stacked")
@click.option("--min-overlap", type=float, default=ground_module.DEFAULT_MIN_OVERLAP,
              help="Minimum overlap area (m2) for the stacked test")
@click.option("--clearance", type=float, default=ground_module.DEFAULT_CLEARANCE,
              help="Height above the local street before a connected ramp counts as elevated")
@click.option("--ground-drop", type=float, default=0.05, help="Hold the ground this far under the road")
@click.option("--fill-island", type=float, default=0.0,
              help="Absorb junction scraps below this area into the carriageway (can leave holes)")
# buildings
@click.option("--buildings/--no-buildings", default=False,
              help="Fill the open ground with procedural buildings")
@click.option("--setback", type=float, default=None, help="Gap between the kerb line and any wall (m)")
@click.option("--lot-area", "target_lot_area", type=float, default=None, help="Split blocks to about this (m2)")
@click.option("--min-lot-area", type=float, default=None)
@click.option("--coverage", type=float, default=None,
              help="Share of each lot its building occupies (0-1); the density knob")
@click.option("--vacancy", type=float, default=None,
              help="Share of lots left as open ground (0-1)")
@click.option("--lot-margin", type=float, default=None,
              help="Minimum gap between neighbouring buildings (m)")
@click.option("--min-height", type=float, default=None)
@click.option("--max-height", type=float, default=None)
@click.option("--floor-height", type=float, default=None)
@click.option("--facade-width", type=float, default=None,
              help="Wall one facade sheet spans (m); must match `city-builder layouts`")
@click.option("--tall-bias", type=float, default=None, help="0 = every block low, 1 = every block tall")
@click.option("--max-buildings", type=int, default=None)
@click.option("--seed", type=int, default=None, help="Building layout is deterministic for a given seed")
@click.option("--ground-texture", default=None,
              help="Tile image to repeat across the ground (see `city-builder tile`)")
@click.option("--tile-metres", type=float, default=12.0, help="How far one tile spans")
@click.option("--facade-dir", default=None,
              help="Directory of facade sheets (see `city-builder layouts` / `facades`)")
@click.option("--quiet", is_flag=True)
def build_command(input_path, blend, glb, heightmap_path, manifest_path, ground_texture,
                  tile_metres, facade_dir, quiet, **kwargs):
    """Build a scene from a Lanelet2 map."""
    from .build import build_city, build_scene, options_from_kwargs, write_heightmap, write_manifest

    if not any((blend, glb, heightmap_path, manifest_path)):
        raise click.UsageError("nothing to write: pass --output, --glb, --heightmap or --manifest")

    building_keys = (
        "setback", "target_lot_area", "min_lot_area", "lot_margin", "coverage", "vacancy",
        "min_height", "max_height", "floor_height", "facade_width", "tall_bias",
        "max_buildings", "seed",
    )
    building_values = {k: kwargs.pop(k) for k in building_keys}

    surface_keys = (
        "max_segment", "marking_width", "stop_line_width", "dash_length", "dash_gap", "curb_height",
        "crosswalks", "walkways", "markings", "stop_lines", "crosswalk_stripes", "curbs",
    )
    options = options_from_kwargs(**{k: kwargs.pop(k) for k in surface_keys})

    from .buildings import BuildingOptions

    building_options = BuildingOptions(**{k: v for k, v in building_values.items() if v is not None})
    result = build_city(input_path, surface_options=options, building_options=building_options,
                        verbose=not quiet, **kwargs)

    if heightmap_path:
        write_heightmap(result, heightmap_path)
    if manifest_path:
        write_manifest(result, manifest_path)
    if blend or glb:
        build_scene(result, blend=blend, glb=glb, ground_texture=ground_texture,
                    tile_metres=tile_metres, facade_dir=facade_dir, verbose=not quiet)


@main.command("inspect")
@click.option("--input", "input_path", required=True, help="Lanelet2 HD map (.osm)")
def inspect_command(input_path):
    """Report what a map contains, without building anything."""
    import collections

    from . import lanelet

    ll2, projection, lmap = lanelet.load_map(input_path)
    lanelets = collections.Counter(
        lanelet.attributes(x).get("subtype", "?") for x in lmap.laneletLayer
    )
    linestrings = collections.Counter(
        lanelet.attributes(x).get("type", "?") for x in lmap.lineStringLayer
    )
    lat, lon = lanelet.map_centroid(ll2, projection, lmap)

    click.echo(f"centroid: {lat:.7f},{lon:.7f}")
    click.echo(f"lanelets ({sum(lanelets.values())}): {dict(lanelets.most_common())}")
    click.echo(f"linestrings ({sum(linestrings.values())}): {dict(linestrings.most_common())}")


@main.command("verify")
@click.option("--scene", "scene_path", required=True, help="A .blend produced by `city-builder build`")
@click.option("--samples", type=int, default=6000)
@click.option("--tolerance", type=float, default=0.02,
              help="Allowed overlap at the seam where ground meets kerb (m)")
def verify_command(scene_path, samples, tolerance):
    """Check that the ground does not come up through the carriageway."""
    import bpy

    from .scene import verify_ground_clearance

    bpy.ops.wm.open_mainfile(filepath=scene_path)
    report = verify_ground_clearance(samples=samples)
    click.echo(
        f"samples={report['samples']} ground_above_road={report['ground_above_road']} "
        f"({report['ground_above_road_pct']}%) worst={report['worst_m']} m"
    )
    if report["worst_m"] > tolerance:
        raise SystemExit(f"ground rises {report['worst_m']} m above the road (tolerance {tolerance} m)")
    click.echo("OK: the ground stays under the carriageway")


@main.command("classes")
def classes_command():
    """List the surface classes and their texturing policy."""
    from .classes import CLASSES

    width = max(len(name) for name in CLASSES)
    for name, surface in CLASSES.items():
        click.echo(f"{name:<{width}}  {surface.label:<13} {surface.paint:<9} "
                   f"idx={surface.pass_index}  {surface.note}")


@main.command("tile")
@click.option("--prompt", default="seamless top-down asphalt and dirt ground texture, "
                                  "urban street surface, uniform lighting, photographic")
@click.option("--output", "output_path", required=True, help="Where to write the tile PNG")
@click.option("--size", type=int, default=1024)
@click.option("--steps", type=int, default=24)
@click.option("--seed", type=int, default=0)
@click.option("--model", default=None, help="Diffusion model id [default: SDXL base]")
@click.option("--vram-budget-gb", type=float, default=6.0,
              help="Hard cap, so a shared card keeps working for its other tenant")
@click.option("--diffusion/--procedural", default=True,
              help="--procedural makes a tileable noise texture with no GPU at all")
def tile_command(prompt, output_path, size, steps, seed, model, vram_budget_gb, diffusion):
    """Generate one tileable ground texture."""
    from .texture import TextureOptions, make_tile, seam_error

    options = TextureOptions(size=size, steps=steps, seed=seed, vram_budget_gb=vram_budget_gb,
                             diffusion=diffusion)
    if model:
        options.model = model

    tile = make_tile(prompt, options, path=output_path)
    click.echo(f"wrote {output_path}  seam_error={seam_error(tile):.2f} "
               f"(1.0 = the wrap looks like the texture's own variation)")


@main.command("models")
@click.option("--family", type=click.Choice(["sd15", "sdxl", "all"]), default="all",
              help="Which stack to look at")
@click.option("--download", "do_download", is_flag=True, help="Fetch whatever is missing")
def models_command(family, do_download):
    """What the texturing path needs, and whether it is on this machine.

    Reads the Hugging Face cache and nothing else — no model is loaded, no CUDA
    context is created, so this is safe to run while the card is busy.
    """
    from .weights import cache_root, download, missing, report, size_on_disk, variant

    click.echo(f"cache: {cache_root()}")

    if do_download:
        wanted = missing(family)
        if not wanted:
            click.echo("nothing missing")
        for weight in wanted:
            click.echo(f"fetching {weight.repo} ...")
            download(weight)

    width = max(len(weight.key) for weight, _ in report(family))
    absent = 0
    for weight, path in report(family):
        if path:
            mark = "ok     "
            detail = f"{variant(weight) or 'fp32':>4}  {size_on_disk(weight) / 1e9:5.1f} GB"
        else:
            mark, detail, absent = "MISSING", "  run with --download", absent + 1
        click.echo(f"{mark}  {weight.key:<{width}}  {detail:>21}  {weight.note}")

    if absent:
        click.echo(f"\n{absent} missing")
        raise SystemExit(1)


def _floor_spec(text: str) -> list[int]:
    """``"3-12"`` or ``"4,6,9"`` or ``"3-6,10"`` into a list of floor counts."""
    counts: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            low, high = (int(x) for x in part.split("-", 1))
            counts.extend(range(low, high + 1))
        else:
            counts.append(int(part))
    if not counts:
        raise click.BadParameter(f"no floor counts in {text!r}")
    return sorted(set(counts))


@main.command("layouts")
@click.option("--output", "output_dir", required=True, help="Directory to write the sheets into")
@click.option("--floors", "floor_spec", default="2-14",
              help="Floor counts to draw a sheet family for, e.g. 3-12 or 4,6,9")
@click.option("--variants", type=int, default=2, help="Sheets per floor count")
@click.option("--facade-width", type=float, default=12.0,
              help="Wall each sheet spans; must match the build's --facade-width")
@click.option("--bay-metres", type=float, default=3.0, help="Spacing of the window columns")
@click.option("--floor-height", type=float, default=3.5, help="Only used to report texel density")
@click.option("--px-per-floor", type=int, default=128)
@click.option("--px-per-bay", type=int, default=128)
@click.option("--seed", type=int, default=0)
@click.option("--control/--no-control", default=True,
              help="Also write the line drawings a structural conditioner needs")
def layouts_command(output_dir, floor_spec, variants, facade_width, bay_metres, floor_height,
                    px_per_floor, px_per_bay, seed, control):
    """Draw facade sheets from the geometry alone — no model, no GPU.

    One family per floor count, because the facade UV normalises V over the
    building's height: a sheet only reads correctly on a building with the
    number of storeys it was drawn for. The sheets are plain stand-ins meant to
    finish and check the UV path; the control images beside them are what a
    diffusion pass is conditioned on so its windows land on the same floors.
    """
    import os
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
    from .texture import save_tile, seam_error_axis

    counts = _floor_spec(floor_spec)
    bays = bays_for(facade_width, bay_metres)
    control_dir = os.path.join(output_dir, "control")
    os.makedirs(output_dir, exist_ok=True)
    if control:
        os.makedirs(control_dir, exist_ok=True)

    written, seams, floor_scores, bay_scores, densities = 0, [], [], [], []
    for floors in counts:
        for variant in range(variants):
            # A drawing per variant, not per floor count. One canonical layout
            # gives every building in the city the same window proportions and
            # the same bay rhythm, and the conditioner then holds the model to
            # it — structure is the half of a facade's variety no prompt supplies.
            rng = random.Random(seed + 1000 * floors + variant)
            layout = sample_layout(floors, rng, facade_width=facade_width,
                                   bay_metres=bay_metres)
            width, height = layout.pixel_size(px_per_floor, px_per_bay)
            if control:
                save_tile(control_image(layout, width, height),
                          os.path.join(control_dir, sheet_name(floors, variant, "control")))
            sheet = procedural_facade(layout, width, height, seed=seed + 1000 * floors + variant)
            save_tile(sheet, os.path.join(output_dir, sheet_name(floors, variant)))
            written += 1
            seams.append(seam_error_axis(sheet, axis=1))
            floor_scores.append(floor_alignment(sheet, layout))
            bay_scores.append(bay_alignment(sheet, layout))
            densities.append(100.0 * layout.texel_metres(floors * floor_height, height))

    click.echo(f"wrote {written} sheet(s) to {output_dir}"
               + (f" (+ control images in {control_dir})" if control else ""))
    click.echo(f"floor counts {counts[0]}-{counts[-1]}, "
               f"about {bays} bays per {facade_width:g} m of wall, {variants} layout(s) each")
    click.echo(f"horizontal seam  {sum(seams)/len(seams):.2f} (wants ~1: a wall wraps)")
    click.echo(f"floor alignment  {min(floor_scores):.2f} (worst; 1 = windows exactly on the storeys)")
    click.echo(f"bay alignment    {min(bay_scores):.2f} (worst)")
    click.echo(f"texel density    {min(densities):.1f}-{max(densities):.1f} cm/texel vertically")


@main.command("facades")
@click.option("--layouts", "layouts_dir", required=True,
              help="Directory from `city-builder layouts`; its control/ images are the input")
@click.option("--output", "output_dir", required=True, help="Directory to write the sheets into")
@click.option("--prompt", default=None,
              help="One prompt for every sheet; by default they are spread across "
                   "`city-builder styles`, which is what gives a street more than one material")
@click.option("--negative", "negative_prompt", default="")
@click.option("--family", type=click.Choice(["sd15", "sdxl"]), default="sd15")
@click.option("--controlnet", type=click.Choice(["canny", "mlsd", "none"]), default="canny",
              help="What holds the windows on the storeys; 'none' is the ablation")
@click.option("--control-scale", type=float, default=0.9)
@click.option("--lcm/--no-lcm", default=True, help="--no-lcm is the quality ceiling, ~4x slower")
@click.option("--count", type=int, default=4, help="Sheets per floor count")
@click.option("--floors", "floor_spec", default=None, help="Only these floor counts, e.g. 4,6,9")
@click.option("--steps", type=int, default=6, help="LCM needs very few")
@click.option("--guidance", type=float, default=1.5)
@click.option("--variation", type=float, default=0.45,
              help="0 = identical siblings, 1 = unrelated strangers")
@click.option("--seed", type=int, default=0)
@click.option("--batch", type=int, default=1)
@click.option("--vram-budget-gb", type=float, default=10.0)
@click.option("--offload", is_flag=True, help="One module on the GPU at a time, for a shared card")
@click.option("--keep-below", type=float, default=None,
              help="Discard sheets whose floor alignment falls under this")
def facades_command(layouts_dir, output_dir, prompt, negative_prompt, floor_spec, keep_below,
                    controlnet, **kwargs):
    """Generate facade sheets conditioned on the layouts' control images.

    One family per floor count, taking `layouts`' line drawing as the structure
    and letting the model decide only the materials. Each sheet is scored
    against the drawing it was given before it is written, so a run that lost
    the storeys says so instead of leaving it to a glance at a contact sheet.
    """
    import os
    import time

    import numpy as np

    from .facade_layout import alignment, diversity, saturation, sheet_floors, sheet_name, wrap_seam
    from .texture import FacadeOptions, facade_sheets, load_facade_pipeline, save_tile, styled_prompts

    control_dir = os.path.join(layouts_dir, "control")
    if not os.path.isdir(control_dir):
        raise click.UsageError(f"no control images in {control_dir}; run `city-builder layouts` first")

    wanted = set(_floor_spec(floor_spec)) if floor_spec else None
    controls = []
    for name in sorted(os.listdir(control_dir)):
        floors = sheet_floors(name) if name.endswith(".png") else None
        if floors is not None and (wanted is None or floors in wanted):
            controls.append((floors, os.path.join(control_dir, name)))
    if not controls:
        raise click.UsageError(f"no control images matching {floor_spec or 'anything'}")

    options = FacadeOptions(controlnet="" if controlnet == "none" else controlnet, **kwargs)
    os.makedirs(output_dir, exist_ok=True)

    from PIL import Image

    started = time.time()
    pipeline = load_facade_pipeline(options)
    loaded = time.time()
    click.echo(f"{options.family} + {controlnet}{' + lcm' if options.lcm else ''}: "
               f"loaded in {loaded - started:.0f}s")

    written, dropped, scores, kept = 0, 0, [], []
    for index, (floors, path) in enumerate(controls):
        control = np.asarray(Image.open(path).convert("RGB"))
        # A different material per sheet. The alignment score cannot see colour,
        # so leaving the prompt fixed quietly produces a city of one material.
        prompts = prompt or styled_prompts(options.count,
                                           seed=options.seed + index * options.count)
        sheets = facade_sheets(prompts, control, options,
                               negative_prompt=negative_prompt, pipeline=pipeline)
        for variant, sheet in enumerate(sheets):
            floor_score = alignment(sheet, control, axis=0)
            bay_score = alignment(sheet, control, axis=1)
            seam = wrap_seam(sheet, control)
            scores.append((floors, floor_score, bay_score, seam))
            if keep_below is not None and floor_score < keep_below:
                dropped += 1
                continue
            save_tile(sheet, os.path.join(output_dir, sheet_name(floors, written)))
            kept.append(sheet)
            written += 1

    elapsed = time.time() - loaded
    click.echo(f"{len(scores)} sheet(s) in {elapsed:.0f}s ({elapsed / max(1, len(scores)):.1f}s each)"
               f" — wrote {written}" + (f", dropped {dropped}" if dropped else ""))
    click.echo(f"{'floors':>6} {'align':>7} {'bays':>7} {'seam':>7}")
    for floors, floor_score, bay_score, seam in scores:
        click.echo(f"{floors:>6} {floor_score:>7.2f} {bay_score:>7.2f} {seam:>7.2f}")
    mean = sum(s[1] for s in scores) / len(scores)
    click.echo(f"mean floor alignment {mean:.2f} "
               f"(a sheet drawn to spec scores >0.6; noise scores ~0)")
    if kept:
        sats = [saturation(sheet) for sheet in kept]
        click.echo(f"diversity {diversity(kept):.3f} (one prompt gives ~0.05, the whole style set ~0.4)")
        click.echo(f"saturation {min(sats):.2f}-{max(sats):.2f} "
                   f"(concrete sits near 0.05, brick near 0.25)")


@main.command("styles")
def styles_command():
    """The facade characters a `facades` run spreads its sheets across."""
    from .texture import FACADE_STYLES

    width = max(len(name) for name, _ in FACADE_STYLES)
    for name, prompt in FACADE_STYLES:
        click.echo(f"{name:<{width}}  {prompt}")
