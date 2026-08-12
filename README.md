# city_builder

Build **ground and road-surface meshes from a Lanelet2 HD map**, in Blender,
from Python. No Blender install to find: `bpy` is a dependency, so the scene is
built in-process.

```bash
uv sync
uv run city-builder build --input lanelet2_map.osm --output scene.blend --glb scene.glb
```

```python
from city_builder import build_city, build_scene

result = build_city("lanelet2_map.osm")          # geometry only, no Blender
build_scene(result, blend="scene.blend")         # …now put it in a scene
```

## What it makes

| Collection | Source in the map |
|---|---|
| `Roads` | `lanelet` subtype `road` / `road_shoulder` / `highway` |
| `Junctions` | the same, where the lanelet carries `turn_direction` |
| `Crosswalks`, `Walkways` | `lanelet` subtype `crosswalk` / `walkway` |
| `LaneMarkings` | linestring `type=line_thin\|line_thick`, `subtype=solid\|dashed` |
| `StopLines` | linestring `type=stop_line` |
| `CrosswalkStripes` | linestring `type=pedestrian_marking` (one closed ring per zebra bar) |
| `Curbs` | linestring `type=road_border`, extruded `--curb-height` (0.15 m) |
| `Ground` | reconstructed — see below |

`virtual` linestrings are logical boundaries and are never drawn. Dashes are
cut from the map's continuous polyline with `--dash-length` / `--dash-gap`:
the map stores the line, not the individual dashes.

Why read the map rather than OSM centrelines: a Lanelet2 map ships the surveyed
left and right boundary of every lane in 3D, so width, curvature and elevation
are measured instead of inferred from highway tags, and intersections are real
turning lanelets rather than a hull over the approaching roads.

## Surface classes and the texturing policy

Every object carries what it *is* and whether its colour may be regenerated —
a diffusion pass must not reinvent the colour of a stop line, but asphalt and
terrain are placeholders and repainting them is the point.

```bash
uv run city-builder classes                       # the registry
uv run city-builder build ... --manifest out.json # …and per-build, as JSON
```

| group | label | policy |
|---|---|---|
| `LaneMarkings`, `StopLines`, `CrosswalkStripes` | `road_marking` | **preserve** |
| `Roads`, `Junctions`, `Crosswalks` | `road` | generate |
| `Walkways` | `sidewalk` | generate |
| `Curbs` | `curb` | generate |
| `Ground` | `terrain` | generate |

The tags reach a consumer three ways, because pipelines read different things:

* **object custom properties** — `cb_class`, `cb_label`, `cb_paint`,
  `cb_pass_index`; kept in the `.blend` and exported to glTF node `extras`;
* **`pass_index`** — for a Cycles `IndexOB` pass, i.e. a segmentation render;
* **`_cb_mask`**, a flat per-corner colour attribute from a well-separated
  palette (worst pairwise separation 0.33 in L1), so the class survives a
  consumer that joins the objects. It exports as the glTF `_CB_MASK` custom
  attribute rather than `COLOR_0`: a viewer multiplies `COLOR_0` into the base
  colour, which would tint the whole asset with the mask palette.

```python
import bpy
for obj in bpy.data.objects:
    if obj.get("cb_paint") == "preserve":
        ...  # lock this one before running the texturing pass
```

## The ground

A Lanelet2 map has no terrain — elevation exists only on the carriageway. The
road network is a dense set of ground samples, so the ground is interpolated
from it and then **clipped to the road outline**, meeting the carriageway at
the kerb line.

Elevated structure has to come out first or the ground ends up on top of the
overpass, and these maps carry no `bridge` or `layer` tag. Lanelets that
overlap in plan view but sit apart in z seed the elevated set; connectivity
then carries it down the approach ramps, which overlap nothing and climb too
gently for a local height test to catch. Candidates are judged against the
street level *beside* them, and the rounds repeat until the set stops growing.

Measured on the Nishi-Shinjuku Autoware map (979 lanelets, 1.1 km across):

| | |
|---|---|
| elevated lanelets | 97 of 887 |
| ground surface vs the road | never above it by more than **5 mm** (ray-cast at every road vertex) |
| heightmap vs the street | ±0.35 m (p90) |
| viaduct above the ground | 5.9 m (median) |
| cells containing a road sample | ~12 % |

Note that most of that map's 17 m spread is *real* relief — the plateau edge —
not structure.

Three earlier approaches were rejected by measurement, and are worth knowing
about before "simplifying" this:

* a median fit through the samples buried 28 % of the road network, 9 % of it
  deeper than a kerb;
* a clamped lower envelope still cut 6 % — a 10 m cell cannot follow a
  carriageway crossing it on a slope;
* cutting the grid with mesh booleans, per lanelet, left the gaps between
  neighbouring lanes as slivers up the middle of the carriageway; dissolving
  the outline first produced a concave ring whose cap Blender could not
  tessellate, and it silently deleted the whole ground.

### Known limitations

* Lanelets do not tile a junction exactly, so a few square metres of ground
  show through between turning lanes. `--fill-island` absorbs them into the
  carriageway, but only where the road mesh covers them — otherwise it trades
  a sliver for a hole.
* Coverage is uneven: the heightmap records `support_cells`, the distance in
  cells to the nearest measured cell, so a consumer can tell where the surface
  is measured and where it is invented. A block interior far from any street is
  an educated guess.

## Placing other things on this ground

`--heightmap out.json` writes the grid plus the scene anchor, so a building or
prop generator can sit on the same surface:

```python
import json
from city_builder import HeightMap, LocalFrame

data = json.load(open("out.json"))
frame = LocalFrame(data["meta"]["ref_lat"], data["meta"]["ref_lon"])
ground = HeightMap.from_json(data["heightmap"])

x, y = frame.to_local(35.6902, 139.6914)
z = ground.sample(x, y)
```

## Requirements

Python **3.11** — `bpy` publishes cp311 wheels up to 5.0.1 and cp313 from 5.1,
and pinning the 3.11 line keeps `bpy`, `simple-lanelet2`, `shapely` and `scipy`
co-installable (bpy links against the numpy 1.x ABI).

Autoware maps use regulatory elements core Lanelet2 does not define
(`road_marking`, `detection_area`); the loader registers them via
`autoware_lanelet2_extension_python` and falls back to `loadRobust`.

## Development

```bash
uv sync --group dev
uv run pytest          # geometry tests; no map and no Blender needed
uv run ruff check src/ tests/
```
