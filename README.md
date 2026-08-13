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

## Facades

A texturing model is bad at exactly the thing a facade is made of: a regular
grid of windows on evenly spaced storeys. Asked in words for "six floors", it
returns a wall with windows somewhere. That is not a weak prompt, it is what
samplers do to periodic structure.

We do not have to ask. These buildings are *generated*, so their floor count is
known rather than guessed — every height snaps to whole floors — and a wall is
flat, so its UV layout and its front elevation are the same picture. Rasterise
the structure, and a model paints materials onto it instead of inventing an
architecture.

```bash
# sheets and their control images, from the geometry alone — no model, no GPU
uv run city-builder layouts --output sheets/ --floors 2-8 --facade-width 12

uv run city-builder build --input map.osm --output scene.blend \
    --buildings --facade-dir sheets/ --facade-width 12
```

`build` prints the floor counts it produced (`floor counts: 2-7`), which is what
`--floors` should be given. `--facade-width` has to match between the two.

**A sheet belongs to a floor count, not to a height.** The wall UV runs V=0 at
the pavement to V=1 at the roofline whatever the building is, so a sheet drawn
for six floors stretches its windows off the storeys of anything else.
Buildings are matched to sheets by floor count — the count travels in the
filename, `facade_f06_003.png` — and shuffled only within a matching family.
That is also why the sheets are sized at a fixed number of texels per floor
rather than to a fixed resolution: at 128 px per floor every wall in the scene
comes out at about 2.7 cm per texel, whether it is three storeys or twenty.

Two more decisions fall out of the same UV. The sheet must tile **horizontally**
— a wall goes round the building — and must not tile vertically, since joining a
roofline to a shopfront is the artefact to avoid. And the sheet is stretched by
a few per cent so a whole number of repeats goes round the ring, which puts no
seam at the closing corner instead of cutting the last repeat mid-window.

### Measuring it

`floor_alignment` is the check that matters: it correlates the sheet's edges
against the control image's, so it answers "did the windows land on *this*
building's floors". A sheet drawn to spec scores above 0.6; noise, a flat wall,
the right facade shifted half a storey, and a sheet drawn for a different floor
count all score below 0.3. `bay_alignment` is the same across the sheet.

This is deliberately not a measure of periodicity, which is what it was first
built as and does not work: a facade profile has two impulses per storey — a
head and a sill — so it carries a strong half-floor beat of its own, and "does
this repeat once per floor" scores a correct sheet no better than one with twice
the storeys.

`wrap_seam` is the other check, and it needs the same care. A facade repeats
every bay, so its wrap lands on a bay boundary — a pier, which is *supposed* to
be a hard edge. The ground tile's `seam_error` compares that against the mean
step over the whole sheet, which compares a pier with blank wall: measured, it
scored sheets that tile perfectly anywhere from 0.3 to 11, and the false alarm
cost an afternoon of hunting a padding bug that was not there. `wrap_seam`
compares the wrap against the *other bay boundaries*, reading the bay count off
the control image so it cannot drift from the layout.

`procedural_facade` draws the sheets with no model at all. They are not
photographic, they are *correct*, which is what lets the UV, the texel density,
the material slots and the export be finished and tested on a machine with no
card in it — and what gives a generated sheet something to be compared against.

### Model weights

```bash
uv sync --extra texture
uv run city-builder models                        # what is here, and where
uv run city-builder models --family sd15 --download
```

The report reads the Hugging Face cache and nothing else — no model is loaded
and no CUDA context is created, so it is safe to run while the card is busy.
It prints whether each repo is cached at half or full precision, because
passing `variant="fp16"` when only the full-precision files are there fails
outright.

Two stacks are declared, because the choice between them was a measurement
rather than a preference. Six sheets each, 3090, floor counts 4/6/8:

| | floor alignment | per sheet | note |
|---|---|---|---|
| sd15, **no ControlNet** | **−0.01** | 0.8 s | a wall with windows somewhere |
| sd15 + canny | 0.84 | 1.0 s | |
| sd15 + mlsd | **0.85** | 1.0 s | most consistent bay alignment (0.97–0.99) |
| sdxl + canny | 0.80 | 1.6 s | more varied materials; needs ~12 GB |

Two things settled. **ControlNet is not optional** — without it the score is
indistinguishable from noise, which is exactly the failure that prompted all
of this. And the reported interference between LCM-LoRA and ControlNet on SDXL
**did not show up**: 0.80 is perfectly usable, so the choice is resolution and
VRAM against speed, not one stack working and the other not.

More sampling steps do not help either. LCM at 6 steps scored 0.84; the same
model at 12 scored 0.70, and a full 25-step non-LCM run at guidance 7 scored
0.60 while taking three times as long. LCM-LoRA is tuned for four to eight
steps, and higher guidance simply lets the model wander further from the lines
it was given.

`--download` fetches fp16 safetensors only. These repos ship the same tensors
three or four ways over — `.bin`, full-precision `.safetensors`, and on the
SD1.5 base two single-file checkpoints as well — so a bare `*.safetensors`
costs several times what is needed to run them.

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
