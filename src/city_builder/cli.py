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
@click.option("--tall-bias", type=float, default=None, help="0 = every block low, 1 = every block tall")
@click.option("--max-buildings", type=int, default=None)
@click.option("--seed", type=int, default=None, help="Building layout is deterministic for a given seed")
@click.option("--ground-texture", default=None,
              help="Tile image to repeat across the ground (see `city-builder tile`)")
@click.option("--tile-metres", type=float, default=12.0, help="How far one tile spans")
@click.option("--quiet", is_flag=True)
def build_command(input_path, blend, glb, heightmap_path, manifest_path, ground_texture,
                  tile_metres, quiet, **kwargs):
    """Build a scene from a Lanelet2 map."""
    from .build import build_city, build_scene, options_from_kwargs, write_heightmap, write_manifest

    if not any((blend, glb, heightmap_path, manifest_path)):
        raise click.UsageError("nothing to write: pass --output, --glb, --heightmap or --manifest")

    building_keys = (
        "setback", "target_lot_area", "min_lot_area", "lot_margin", "coverage", "vacancy",
        "min_height", "max_height", "floor_height", "tall_bias", "max_buildings", "seed",
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
                    tile_metres=tile_metres, verbose=not quiet)


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
