# city_builder

Build **ground and road-surface meshes from a Lanelet2 HD map**, in Blender,
from Python. No Blender install to find: `bpy` is a dependency, so the scene is
built in-process.

```bash
uv sync
uv run city-builder make --input lanelet2_map.osm --out-dir out/    # everything
uv run city-builder build --input lanelet2_map.osm --output scene.blend --glb scene.glb
```

`make` takes one map or a directory of them and writes the scene, a `.glb`, an
`.fbx`, the manifest and a drive video for each, in one pass — so a map is
never half-built and the video always comes from the scene that was just
written rather than from whatever .blend was lying around under that name.

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

## Road markings

Painted markings are not objects. Building them as coplanar slabs a couple of
millimetres above the carriageway is what a mesh pipeline does when it has
nowhere else to put them, and it costs: 2938 lane-line ribbons, 1327 zebra bars
and 207 stop lines on this map, all fighting the road for the same depth, and
all of them free to hang past the edge of a viaduct deck because nothing tells
a stripe where the road stops.

They are baked into the carriageway's own texture instead. The paint is clipped
to the surface by construction, there is no second surface to fight, and 12 483
faces go away.

**Resolution is the whole problem.** The carriageway is 124 452 m²; one image
over the map at a resolution that keeps a 15 cm line crisp would be hundreds of
megatexels. But a lanelet is a *ribbon*, so it has a natural parameterisation —
along it and across it — and each lane can be rasterised into its own strip and
packed with the others.

One texel is `texel_metres` on the ground, everywhere. A fixed *count* of
texels across a lane was the first version and it is wrong in a way that shows:
lane widths here run 1.43 m to 9.42 m, so a 15 cm line came out anywhere
between 1.0 and 6.7 texels across and visibly thinned and thickened along a
drive. Strips then have different widths, so they are packed widest first —
which keeps a column's strips the same width as each other and stops the
general rectangle-packing problem from turning up in a road builder.

| `texel_metres` | total | pages | fill | a 15 cm line |
|---|---|---|---|---|
| 0.04 | 92.7 Mtexel | 7 | 79 % | 3.8 texels |
| **0.05** | **58.4 Mtexel** | **5** | **70 %** | **3.0 texels** |
| 0.06 | 43.9 Mtexel | 4 | 65 % | 2.5 texels |

```bash
uv run city-builder build --input map.osm --output scene.blend \
    --marking-pixels 64 --road-texture asphalt.png
uv run city-builder build ... --marking-geometry   # the old coplanar slabs
```

A lane line stops at the intersection. Inside one there is nothing to separate
— every turning path crosses every other one — so a junction lanelet's boundary
carries no paint, while the stop line at its mouth and the crossing over it
plainly do (`lane_lines_in_junctions` puts them back).

The class registry's distinction survives the move. `preserve` stops being a
property of a group of objects and becomes **the mask channel itself**: the
carriageway's colour may be regenerated wherever the mask is zero, and must not
be touched where it is not. The manifest records it under `markings`, and the
material mixes the generated asphalt with the paint colour using the mask as
the factor — so a texturing pass can replace the asphalt image without ever
being able to touch a lane line.

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

## The edge of the map

A Lanelet2 map is cut out of a city, so its roads stop at an arbitrary line
with nothing beyond them. Everything downstream reads that stopping point as
ordinary ground: the terrain is interpolated across it, and the building
generator — which fills whatever the roads leave — puts a block squarely across
the end of the street. The scene comes out as a city with a wall around it.

So the ends the map cut are run out to its edge. An end is loose when no
lanelet *starts* where it finishes — the successor relation read off the shared
boundary point ids, and directional, because two lanelets starting on the same
pair is a fork rather than a connection. Each loose end continues straight, at
the width and grade it had, until it reaches the edge or comes within
`clearance` of another lanelet. A stub pointing into the side of a road that is
already there gets nothing, which is "do not interfere with other lanelets"
stated as geometry.

Two things stop this from doing damage:

* **A loose end is not always an edge.** A road that stops in the middle of the
  city is a dead end the survey meant, and running it to the far corner draws a
  lane-wide scratch across half a kilometre of blocks — 578 m of it, measured,
  on Nishi-Shinjuku. So an end only counts as cut off if it leaves the road
  network's own outline within `cut_off_within` metres. On that map this is the
  difference between extending 73 loose ends and extending 23 of them; the
  other 112 are inland dead ends left alone.
* **The edge is fixed before anything moves.** It is a margin around the roads,
  and the roads are what is being lengthened, so the ground is built to the box
  the *surveyed* geometry gave. Taking a fresh margin around the extended roads
  would push the edge out again and leave another ring of ground with no road
  reaching it, which is the thing being fixed.

The extension is applied to the **boundary polylines**, not to the finished
lane surfaces. A lane bound, the line painted along it and the kerb beside it
are one linestring as far as the map is concerned, so lengthening it once gives
all three the same continuation — and the widening, dashing and pairing happen
downstream as usual, so new dashes keep the phase of the old ones.

```yaml
extend:
  enabled: true
  margin: 30.0           # how far past the surveyed roads the edge sits
  cut_off_within: 60.0   # farther inland than this and it is a dead end, not an edge
  clearance: 1.0         # keep this far from any other lanelet
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

## Elevated roads

A Lanelet2 map surveys the driving surface and nothing else, so an elevated
lanelet arrives as a flat ribbon in the air with no thickness, no soffit and
nothing holding it up. Measured on the Nishi-Shinjuku map, the column under the
deck held **no geometry at all between 5 m and 9 m** — from a car on the deck
the road ends in a knife edge with the city visible below it, which reads as a
height bug rather than as a missing model.

Where a bridge exists follows [Galin et al., *Procedural Generation of Roads*
(CGF 2010)](https://perso.liris.cnrs.fr/egalin/Articles/2010-roads.pdf), §6.1:
sample the trajectory, take the difference between its height and the terrain
under it, and label each sample by that clearance. Only the stretches above
`bridge_clearance` get the bridge model.

**The granularity is the whole point.** Working per lanelet — "this lanelet is
in the elevated set, extrude all of it downwards" — puts girders on the street
below, because an approach ramp is one lanelet running from deck height down to
grade. A ramp is a bridge at one end and a road at the other.

Piers follow [Kapu, *Procedural Generation of Bridges and Tunnels* (MSc, NCCA
2010)](https://nccastaff.bournemouth.ac.uk/jmacey/MastersProject/MSc10/06ChaitanyaKapu/thesis.pdf):
generated between the deck path and the same path projected onto the terrain,
then thinned by spacing and a minimum clearance.

Parapets go on the **outer edge of the outermost lanelets and nowhere else**. A
boundary counts as outer when there is no other deck just beyond it, probed
against the footprint of the elevated network. Three cheaper rules were tried
first and all of them build a wall down the middle of the carriageway:

| rule | why it fails |
|---|---|
| a parapet on every lanelet boundary | a carriageway is several lanelets wide; each lane ends up in its own trench |
| skip boundaries shared by linestring id | this map names 136 of its 165 elevated boundaries exactly once |
| skip boundaries that coincide geometrically | inside a junction, turning lanelets *overlap* rather than tile |
| the outline of the union of all decks | picks up the outline of every shoulder strip inside the carriageway |

What works is asking whether a **lanelet boundary lies on** that outline, rather
than building the outline itself. It keeps the parapet on the surveyed lines —
smooth, and at the right height — while the outline decides which of them are
edges. Probing a fixed distance sideways for a neighbour was the version before
it, and it under-reads: measured, the probe found 3323 m of edge where the
outline finds 4199 m, because a separate structure passing 1.6 m away stops the
probe from saying the deck ends. The barrier is then closed over inner
stretches shorter than `parapet_bridge_gap`, since the test flickers where a
deck brushes past another at a junction mouth.

A road_border is only a kerb where something stops at it. Measured, 689 of
8055 kerb vertices had lanelet surface on *either* side of them — a lane
divider, a give-way line, the seam between a carriageway and its slip road —
and standing those up puts a 15 cm wall down the middle of the road. They are
dropped, and the kerb is split into the stretches that still have open ground
on one side.

The barrier is also closed over short gaps. The neighbour probe flickers where
a deck passes within reach of another one at a junction mouth: measured, 109
candidate runs carried 125 flips between outer and inner, and 17 came out too
short to build, which is a barrier with holes punched in it. An inner stretch
shorter than `parapet_bridge_gap` no longer interrupts one; a real opening
still does.

Two more things a deck needs that the ground does not.

The infilled gaps count as deck for that test, which is the point of filling
them: without it the outline still has a hole where the patch went, and the
barrier runs all the way round the island between two turning lanes. One
description of the gap, used by both.

**Clipped crossings.** A crossing lanelet covers the road *and* the footway
either side, and the road is already there. Measured, 67 of 84 crossing
surfaces overlapped the carriageway, a median 81 % of their area, 4646 m² in
total, at a median 7 cm apart in z — a z-fight at best, and on a viaduct the
overhang sails past the deck edge with nothing under it. Each crossing is
intersected with the carriageway and lifted `crosswalk_lift` onto it, so it
survives as a region a consumer can select without being a second road.

```bash
uv run city-builder build --input map.osm --output scene.blend \
    --parapet-height 1.1 --deck-thickness 1.2 --pier-spacing 28
uv run city-builder build ... --no-viaduct     # just the driving surfaces
```

## Gaps between lanelets

Lanelets are surveyed one at a time and do not tile exactly. Measured here, 306
gaps totalling 2922 m² — 3 % of a 98 120 m² carriageway, most of them a hand's
breadth wide. On a viaduct each one is a slot through to the street below; at
grade the ground shows through, and since the ground is held 5 cm under the
road, **a wheel crossing a sliver drops into it**. Either way the drivable
surface is not a surface, which matters more to a simulator than to a camera.

The patch is the difference between the network's footprint and the same
footprint with its gaps closed — dilate, erode, subtract — with anything larger
than `infill_max_area` left alone, because a real opening between two
carriageways is meant to be there. Each patch takes its height from the lanes
that *touch* it rather than from the nearest surveyed vertex in plan view: a
viaduct passes directly over a street, and the gap in the street is not seven
metres up.

**Each level is patched against itself.** A single plan-view union cannot do
it: a viaduct and the street beneath it leave a long strip between their
footprints, which is not a gap in anything — it is the space beside the
viaduct, and filling it paves over the street at deck height.

What counts as a gap was measured per level rather than guessed:

| | survey artefacts | then | which is |
|---|---|---|---|
| elevated | 5 holes up to 24 m² | 736 m² | a real opening between two carriageways — on a deck, filling it would pave over the sky |
| at grade | 35 holes up to 99 m² | 219 m² and up | traffic islands and medians, which have ground under them and a kerb round them |

`infill_max_area` sits at 150 m², in both gaps. Holes below it: 892 m² at
grade and 69 m² on the deck, both to **zero**. Everything above it survives.
The floor matters too — 2933 of the pinholes here are a few cm² each, and a
patch area cut-off of 0.02 m² threw all of them away.

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

### Variety

Structure and material vary independently, and both have to be asked for.

**Structure** comes from the drawing: `layouts --variants N` samples a layout
per variant rather than one per floor count — bay rhythm, window proportions,
shopfront height, parapet. The window width does most of the work, since narrow
openings in a wide pier read as a punched-window block and wide ones thin the
piers to mullions and draw a ribbon window. One canonical drawing per floor
count was right while the mechanism was being verified and wrong for a city:
the conditioner holds the model to whatever it is given, so identical drawings
mean identical architecture.

**Material** comes from the prompt, and `facades` spreads its sheets across
`city-builder styles` instead of repeating one. This matters more than it
sounds: `floor_alignment` cannot see colour, so ranking configurations by it
alone picked the most literal output there was — saturation 0.06, a city of
grey concrete, which nobody noticed until it was rendered. `diversity` is the
other half of the measurement. One prompt scores about 0.05; the style set
scores about 0.4.

    120 sheets, sd15 + mlsd, 78 s   diversity 0.406   saturation 0.02-0.83

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

## Driving through it

The point of building this scene is to drive through it, so the useful view is
not an aerial render — it is what a windscreen sees.

```bash
uv run city-builder drive --input map.osm --scene scene.blend \
    --output drive.mp4 --seconds 30 --speed 11
```

The route comes from the map, not from the scene: lanelets are lanes, a
lanelet's two boundaries average to its centreline, and one lanelet follows
another when it starts on the pair of boundary points the other ends on.
`lanelet.build_adjacency` gives the *undirected* version of that, which is what
the ground classifier wants; driving needs the directed one, or a route can run
backwards up a one-way street. The road graph has cycles, so the longest route
is found by bounded random walks rather than exactly — on the Nishi-Shinjuku
map that finds 3.0 km across 60 lanelets, including the climb over the viaduct.

Two details decide whether it looks like driving or like a camera on a rail:

* **Resample before animating.** A lanelet's vertices sit wherever the survey
  put them, dense on a curve and sparse on a straight, so one vertex per frame
  races the straights and crawls round the bends.
* **Aim well ahead, not at the next sample.** Aiming at the next sample makes
  the camera yaw with every wobble in the centreline; aiming 18 m down the road
  is what a driver does and turns the same wobble into a steady approach.

Rotations are keyed as quaternions, because the yaw wraps through ±180° on any
route that turns around and Euler interpolation spins the camera the long way
at that frame. Keyframe interpolation is set to linear before the first key is
inserted — with a key on every frame, Bezier handles ease in and out of each
one and the drive comes out stuttering.

Output goes through a PNG sequence and the system ffmpeg: the `bpy` wheel on
PyPI is built without FFmpeg, so Blender's own video writer is not there.
EEVEE renders 720p at about 30 fps of footage per minute of wall clock;
`--engine cycles` uses OptiX on the GPU for a better-looking, much slower pass.

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

## Configuration

Every option group is a dataclass, and `CityConfig` nests them so a whole build
can be written down instead of assembled from flags.

```bash
uv run city-builder config --output city.yaml   # the defaults, as a file
uv run city-builder config                      # every key, type and default
uv run city-builder config --check city.yaml    # what it changes, before a build
uv run city-builder build --input map.osm --output scene.blend --config city.yaml
```

```yaml
surfaces:
  curb_height: 0.15
viaduct:
  parapet_height: 1.1
  bridge_clearance: 2.0
  pier_spacing: 28.0
```

Unknown keys are an error rather than a shrug — a silently ignored typo in a
config file is the worst outcome available, since the run succeeds, the setting
does nothing, and the only evidence is in the geometry. Any flag passed
explicitly beats the file.

## As an MCP server

The same pipeline, exposed for an agent to drive:

```bash
uv sync --extra mcp
uv run city-builder-mcp          # stdio
```

```json
{"mcpServers": {"city-builder": {"command": "uv",
  "args": ["run", "--directory", "/path/to/city_builder", "city-builder-mcp"]}}}
```

Or from the container, which every push to `main` publishes:

```json
{"mcpServers": {"city-builder": {"command": "docker", "args": [
  "run", "-i", "--rm", "--gpus", "all", "--user", "1000:1000",
  "-v", "city-builder-models:/cache",
  "-v", "/path/to/maps:/maps:ro",
  "-v", "/path/to/out:/work",
  "ghcr.io/hakuturu583/city_builder-mcp"]}}}
```

`-i` is not optional: the client speaks MCP on the process's stdin and stdout.
Mount the maps and an output directory — scenes, exports and textures are
written where you ask for them, and nothing else leaves the container. `--user`
keeps the files yours rather than root's.

Everything works in there, including the texture tools and the photoreal
refinement: the image carries the diffusion stack and a pinned ComfyUI, which
between them are most of its 8.1 GB. There is no CUDA base image underneath —
the torch wheels bring their own runtime, so a slim Python and `--gpus all` is
the whole of it. Without `--gpus`, the geometry, export, survey and render tools
carry on regardless (the render falls back to software GL: slow but correct) and
the GPU tools say what is missing.

ComfyUI's dependencies are declared in this project's lock file rather than
installed from its `requirements.txt`, so one resolution decides the versions
and torch lands in the image once instead of twice. ComfyUI itself is a pinned
checkout, because a workflow is a set of node names and input names and both
move.

The tools are not the CLI with a different coat on. Two things change when the
caller is a language model rather than a person at a shell.

**It cannot see.** A person runs a build, opens the `.blend` and knows in a
second whether the road has holes in it. An agent gets a JSON blob. So every
tool that changes something answers with measurements, `survey_scene` exists at
all, and `render_view` hands back an actual image rather than a path to take on
trust.

**It cannot afford to rebuild.** A build is named and kept, and every later
call takes the handle. Blender is a singleton — one process, one scene — so the
handle holds the geometry (numpy and shapely) and anything wanting Blender
rebuilds into it on demand. That is why exporting twice costs twice, and why
each tool says in its own description what it costs: the agent is choosing
between them without a wall clock in front of it.

| | |
|---|---|
| `inspect_map` | what a map contains, before building it |
| `describe_options` | every build option, its type and default |
| `list_styles` | the facade characters sheets are spread across |
| `build` | map → geometry, kept under a handle; answers with the survey |
| `list_scenes` / `forget_scene` | what is held, and dropping it |
| `survey_scene` | the scene in numbers — see below |
| `make_layouts` | facade layouts and control images. Seconds, no GPU |
| `generate_facades` | paint them from your prompts. **GPU, minutes** |
| `make_tile` | a tileable ground or road texture, from a prompt |
| `export` | `.blend` / `.glb` / `.fbx` |
| `render_view` | aerial, plan or street still, returned as an image |
| `render_drive` | a drive along the roads. **Minutes** |
| `refine_render` | that drive, made photoreal by a video model. **GPU, minutes** |

`survey_scene` is the session's worth of debugging distilled into numbers, and
each of them was a bug before it was a metric:

```json
{"carriageway": {"levels": {
    "at_grade": {"surface_m2": 84253.1, "seams": 0, "openings": 12,
                 "largest_opening_m2": 2346.1},
    "elevated": {"surface_m2": 16923.2, "seams": 0, "openings": 1,
                 "largest_opening_m2": 735.7}}},
 "elevation": {"elevated": 97, "clearance_m": {"min": 1.11, "median": 5.81},
               "decks": 93, "parapets": 87, "piers": 168},
 "route": {"lanelets": 60, "length_m": 3045.4, "z_range_m": [3.53, 12.68]}}
```

The texture tools take the prompts. The control image fixes the architecture —
where the floors and windows are — so the prompt is the only thing left
deciding what a building is *made of*: one prompt gives a street built entirely
of one material, and several are spread across the sheets. Each is given a
suffix that keeps the result usable as a texture (flat elevation, overcast, no
sky, no perspective), so an agent writes the material rather than the
photograph. `list_styles` hands back the built-in set to copy or narrow with,
and both tools answer with a picture — a contact sheet of the sheets kept, the
tile itself — because a wrap that scores well can still be the wrong material.

They also take *photographs*. `reference_images` answers the same question with
a picture rather than words, and closes the loop: build a scene, render a drive,
refine it into photoreal frames, hand those frames back to `generate_facades`.
Structure stays ControlNet's job throughout, so what the reference contributes
is material. Measured against a refined street frame, floor alignment holding
at 0.80–0.92 in every case:

| `reference_strength` | |
|---|---|
| **0.4** | takes the palette and the panel material |
| 0.7 | begins copying content — one sheet came back with the reference's yellow road line painted across the facade |

### Making a render photoreal

The scene has its geometry right and its appearance approximate: procedural
facades, a tiled road, a flat sky. A video model has the opposite problem — it
knows what a street looks like and nothing about where this one's buildings
are. `refine_render` puts the render in as the sampler's **starting latent** and
returns only part of the noise, so the geometry is held by what is already
there and the model supplies the surfaces.

`denoise` is the whole dial. Measured on one 3090, 832×480, four steps, ~95 s
for five frames:

| | |
|---|---|
| **0.25** | the same street, photoreal — building masses, road width, vanishing point and crossing all stay put |
| 0.35 | more convincing, and the buildings start becoming generic |
| 0.45+ | a real Japanese street, but no longer *this* one |

Keep it low when the frames are wanted as reference for a second pass at the
textures, which is the loop this exists for: build, render, refine, re-texture.
A reference whose windows sit on different floors than the building it
describes is worse than no reference.

Image-to-video is *not* this, and trying it first is the obvious mistake. H3's
keyframe conditioning is re-injected at every sampling step and never denoised,
so a Blender frame handed to `first_frame` comes back a Blender frame — measured,
the output was indistinguishable from the input. The starting latent is the only
place the render can enter and still be changed.

One node was missing for that. An H3 latent is a nested pair, video
`[B,24,T,H/16,W/16]` and audio `[B,32,2,T40]`, and a plain `VAEEncode` makes the
video half alone; the model then reads `x[1]` for the audio stream and raises
`IndexError`. `MiniMaxH3VideoToVideo`, in `src/city_builder/comfy_nodes`,
encodes the clip and substitutes it into an H3 latent keeping that latent's
audio half. Everything downstream is stock ComfyUI.

Holes are split by **width, not area**, because width is what makes one a
defect: a forty-metre seam a handspan wide is eight square metres and a wheel
drops into it, while a roundabout's central island is three hundred and is
meant to be there. A hole that nowhere admits a one-metre circle is a *seam*
and should be zero; the rest are *openings*. Levels are judged separately,
since a viaduct and the street beneath it share a plan view, and a single union
would call the ground beside the deck a hole through it.

### One building, all the way round

`render_drive` looks at the street. `render_orbit` looks at **one building**,
because what sits downstream of it is not a texturing pass but a
reconstruction: a video model makes the procedural block photoreal, a mesh
model turns that footage back into geometry, and the footprint puts the result
back at the right size. It writes three things — `orbit.mp4`, a `mask/` PNG per
frame, and `orbit.json` with the camera geometry and the footprint.

Three decisions, none of them free.

**The frame count.** H3 counts in 17k+5 and a closed turn divides 360° by the
frame count, so only the counts that are *also* a multiple of four land a frame
exactly on each cardinal azimuth — the four views a multiview reconstruction is
conditioned on. That is 56, 124, 192 and nothing else under 200. Asking for
"about ninety" and getting 90 would put the quadrants at frame 22.5;
`snap_frames` picks 56.

**The framing distance.** The subject is the *cylinder* that contains it, which
is the right envelope precisely because the camera goes all the way round — a
cylinder looks the same from every azimuth, so one distance holds for the whole
orbit. The bounding sphere is a line of arithmetic shorter and much too far
back: the sphere round a wide, low building is as wide as the building, and
framing it left the subject **13 % of the frame**, about a hundred pixels of
building for a reconstruction to work from. The cylinder puts it at **27–32 %**
of the same frame.

**What the subject stands among.** `neighbours` takes out other *buildings*;
the carriageway and the ground stay whatever it says, because they are what
tells the video model what kind of place this is and the only thing in frame
that states how big the building is. The default takes out every other
building: each one left in the clip is a building the reconstruction has to be
told to ignore, and a mask is an instruction a model follows approximately.

| `neighbours` | buildings standing | frames of 56 that saw the subject |
|---|---|---|
| `hide` (default) | the subject alone | 56 |
| `clear` | 40 of 53 | 56 |
| `keep` | all 53 | **24** |

`keep` is what a video model would prefer — a lone building on empty ground is
not a street — and it is unusable: the camera flies at the framing distance,
which on a procedural block at 0.6 coverage is inside the next block, so more
than half the orbit saw no part of the subject at all. `clear` is the middle,
and the reason it works is exact rather than tuned: the camera looks at the
centre of a ring of radius `distance · cos(elevation)` from a point on it, so
every sightline it ever has lies inside that disc. Empty the disc and the view
*cannot* be blocked, whatever the elevation and whatever the frame.

**Dress it before shooting it.** `make_layouts` → `generate_facades` for the
sheets, `make_tile` for the carriageway and the ground, and all three go to
`render_orbit`. This is the render a video model has to work from, and at the
low denoise that keeps the geometry it will not invent what is not there: an
undressed grey box says nothing about where the storeys are or which side is
the front, and what comes back is a grey box that is merely photoreal. A sheet
scoring 0.74 for floor alignment puts three storeys of windows and a shopfront
band on a three-storey building, in the same place in all four quadrant views,
which is what the reconstruction is being asked to agree with.

One thing had to be fixed for that. `road_texture` was only ever reached
through the marking material, so a map with no paint in it — or one built with
markings off — took the argument, reported the road as dressed, and rendered it
flat. The carriageway now gets its tile either way.

The mask is **rendered, not projected**. Projecting the footprint gets the
silhouette wrong wherever anything stands in front of the building, and
something always does. A second EEVEE pass over the same camera keys, subject
white and everything else black, is right by construction, occlusion included —
emission materials, a black world and the Standard view transform, so white
comes back at 1.0 and nobody downstream has to guess a threshold.

That matters because of where the mask is used. An H3 latent is
`[B,24,T,H/16,W/16]`: a mask for it has to be reduced 16-fold in space and
folded in time by the same non-uniform grouping `_pixel_frames` uses — not by a
stride. That reduction belongs to the sampler; what belongs here is a mask that
is exact at pixel scale and frame-aligned with the clip it describes.

### Where the model weights live

The weights are **downloaded, not shipped** — about 3 GB for the SD1.5 stack,
21 GB more if you use SDXL. Nothing in the image or the repository holds them,
so where they land is a setting, and getting it wrong means paying for the
download again on every run.

One environment variable decides: `HF_HOME`. The container sets it to `/cache`
and declares that a volume, so persisting the weights is a mount — the
`-v city-builder-models:/cache` in the invocation above, after a one-off:

```bash
docker volume create city-builder-models
```

A named volume is the simple case. Mounting a host cache you already have works
just as well and shares it with everything else on the machine — point it at
whatever `HF_HOME` is set to outside:

```bash
-v "${HF_HOME:-$HOME/.cache/huggingface}:/cache"
```

Two things worth knowing. Without either mount the weights go into the
container's writable layer and are thrown away with the container, so the next
run downloads them again. And even with a full cache the hub is still asked
whether each file is current, which needs the network; `-e HF_HUB_OFFLINE=1`
stops that and makes a cached run genuinely offline.

`city-builder models` reports what is in the cache without loading anything or
touching a GPU, and `--download` fills it — worth running once, before wiring
the server into anything, so the first real call is not a 3 GB wait.

`/cache` is the whole of it, for anything the Hugging Face hub serves — the
SD1.5 and SDXL stacks, MiniMax H3, and whatever gets added next. One volume,
shared by all of them. The container links whatever it finds there into the
layout ComfyUI wants each time it starts; `/opt/ComfyUI/link_models.sh` is that
step, and running it by hand prints what is missing.

The refinement wants about 35 GB of it:

| | |
|---|---|
| `Abiray/MiniMax-H3-Pruned-GGUF` | `MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf`, 14.1 GB — the packager's recommendation for a 24 GB card |
| `Abiray/MiniMax-H3-GGUF` | `text_encoders/qwen3vl_32b_minimax_h3-Q4_K_M.gguf`, 14.6 GB |
| `Abiray/MiniMax-H3-Turbo-Lora-Pruned-ComfyUI` | `minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors`, 0.6 GB |
| `Comfy-Org/MiniMax-H3` | `vae/minimax_h3_video_vae_fp16.safetensors` and the audio VAE, 5.8 GB |

Two of those choices are not free. The Turbo LoRA has to be the **pruned** one:
a LoRA cut for the full transformer does not key-match a pruned model. And the
text encoder has to be the **ComfyUI-format** GGUF, not the llama.cpp GGUF of
the same Qwen3-VL — the latter loads down a Mistral tokenizer path and dies on
`json.loads(None)`. `nvfp4` is smaller still and useless here: fp4 is a
Blackwell instruction, and this is Ampere.

If you want none of that, the diffusion stack comes out with a build argument
and takes the image from 8.1 GB to 1.8:

```bash
docker build --build-arg EXTRAS=mcp -t city-builder-mcp:slim .
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
