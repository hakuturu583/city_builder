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

## Procedural buildings

The map describes the carriageway and nothing else, so the blocks between the
streets are empty. `--buildings` fills them:

```bash
uv run city-builder build --input map.osm --output scene.blend \
    --buildings --setback 4 --lot-area 700 --max-height 60 --seed 7
```

The steps:

1. **Buildable area** — the ground minus every paved surface, minus `--setback`.
   Interior rings are kept, so a road enclosed by a block stays a hole.
2. **Lots** — each block is cut by a line across its longest axis, a little
   off-centre (`--split-jitter`), recursing until a lot is about `--lot-area`.
   Below `--min-lot-area` it is left as open ground.
3. **Vacancy** — `--vacancy` of the lots are dropped outright: car parks, yards,
   the plot nobody built on.
4. **Footprint** — each lot is inset until the building covers `--coverage` of
   it, never closer than `--lot-margin` to its neighbour.
5. **Extrusion** — base is the *lowest* heightmap sample under the footprint, so
   a building on a slope cuts into the hill rather than floating; walls run
   `--skirt` below that. The height is drawn per lot, snapped to whole floors
   (`--floor-height`) and biased taller on bigger plots (`--tall-bias`).

Deterministic for a `--seed`.

Density has two independent knobs, and a third for grain:

| | | Nishi-Shinjuku, share of open ground built |
|---|---|---|
| `--coverage 0.3` | how much of a lot is building | 28.8 % |
| `--coverage 0.6` (default) | | 57.6 % |
| `--coverage 0.8` | | 73.4 % |
| `--vacancy 0.4` | lots left empty | 34.0 % (1148 buildings, from 1962) |
| `--lot-area 300` | *grain*, not density | 55.4 % across 5692 buildings |
| `--lot-area 2500` | | 57.6 % across 713 buildings |

`--coverage` is the planner's ratio (建蔽率) rather than a fixed gap on purpose:
a fixed 1.5 m margin leaves a 400 m² lot 74 % built and a 2500 m² lot 88 %, so
density would drift with lot size. Solving for the inset keeps the ratio the
parameter and lets the gap between neighbours fall out of it.

Walls (`Buildings`) and roofs (`Roofs`) are separate objects because a
texturing pass treats a facade and a roof completely differently. Both are
`generate` — nothing in the map says what stands here, so there is no authored
colour worth preserving.

On the Nishi-Shinjuku map: 1983 buildings, 913,000 m² of footprint, none of it
overlapping the 99,700 m² of paved surface (exact 2-D test, not sampled).

Two things this deliberately does **not** claim: the buildings are not the real
ones, and the layout has no alleys or courtyards beyond what the road network
implies. It is scaffolding of about the right bulk in about the right place.

Two bugs worth knowing about, since both produced plausible-looking output:

* the exclusion zone must include the lanelets classified as *elevated*. The
  ground layer excludes them on purpose — a viaduct wants ground underneath —
  but a building does not, and reusing that union put 3.7 % of wall feet on the
  carriageway;
* the buildable area must keep its interior rings. A road through the middle of
  a region is a hole, and taking only the exterior handed it back as buildable:
  449 of 2481 plots ended up on the road.

## Textures

Roads and markings come from the map and are left exactly as built — that is
what `paint = preserve` in the class registry means, and the texturing path
filters on it. What has no authored colour is the ground (and, later, the
buildings).

A 1.2 km² ground mesh cannot be painted the way a model paints an *object*:
render a few views, diffuse, project back, and you get about half a metre per
texel. A large, statistically uniform surface wants a small **tileable** image
repeated at a metric scale.

```bash
# one tile, wrapping, from SDXL — or --procedural for no GPU at all
uv run city-builder tile --output ground.png --size 1024 --vram-budget-gb 6 \
    --prompt "seamless top-down photograph of urban asphalt with fine gravel"

uv run city-builder build --input map.osm --output scene.blend \
    --buildings --ground-texture ground.png --tile-metres 10
```

Every convolution in the UNet and VAE is switched to circular padding so the
result wraps, and the wrap is **measured rather than assumed**: `seam_error`
compares the step across the wrap with the steps inside the texture, so 1.0
means the seam looks like the texture's own variation. The SDXL tile above
scores 0.97; the procedural fall-back — noise filtered in the frequency domain,
periodic by construction — scores 0.99.

`--vram-budget-gb` caps the process and the pipeline runs with model CPU
offload, VAE tiling and attention slicing, so this stays usable on a card
shared with something else.

Install the extra for the diffusion path; the geometry half never imports it:

```bash
uv sync --extra texture
```

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
