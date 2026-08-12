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
@click.option("--quiet", is_flag=True)
def build_command(input_path, blend, glb, heightmap_path, quiet, **kwargs):
    """Build a scene from a Lanelet2 map."""
    from .build import build_city, build_scene, options_from_kwargs, write_heightmap

    if not blend and not glb and not heightmap_path:
        raise click.UsageError("nothing to write: pass --output, --glb or --heightmap")

    surface_keys = (
        "max_segment", "marking_width", "stop_line_width", "dash_length", "dash_gap", "curb_height",
        "crosswalks", "walkways", "markings", "stop_lines", "crosswalk_stripes", "curbs",
    )
    options = options_from_kwargs(**{k: kwargs.pop(k) for k in surface_keys})

    result = build_city(input_path, surface_options=options, verbose=not quiet, **kwargs)

    if heightmap_path:
        write_heightmap(result, heightmap_path)
    if blend or glb:
        build_scene(result, blend=blend, glb=glb, verbose=not quiet)


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
