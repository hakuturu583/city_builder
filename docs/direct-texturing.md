# Two ways to stop inventing the shape twice

An investigation, with what it proved and what it did not. Nothing here is in
the pipeline yet; it lives on `feat/direct-trellis` behind
`reconstruct.texture_mesh`.

## The idea

The pipeline currently throws away the one thing it is certain about. It
photographs the procedural massing, has a diffusion model re-imagine the
picture, asks TRELLIS.2 for a **new mesh** from that picture, and then fits the
new mesh back onto the footprint it started from. The fit works — 184 of 189
buildings on the Kashiwanoha map, mean IoU 0.917 — but the other five are
rejected, every one of the 184 is a few per cent off the plot it stands on, and
the whole apparatus of yaw solving, anisotropic stretch, seating and IoU exists
only to undo the damage.

TRELLIS.2 also ships `Trellis2TexturingPipeline`, which takes **a mesh and an
image** and paints the mesh. Used that way the division of labour changes:

|                | shape comes from | surface comes from | footprint |
|---|---|---|---|
| image → 3D (now) | the model | the model | fitted, 0.92 mean IoU |
| mesh → texture   | **the map** | the model | **exact** |

No yaw to solve, no scale, no stretch, no seating, no IoU, nothing to reject.

## What was proved

**It runs, on the weights already downloaded.** `texturing_pipeline.json` and
the four checkpoints it names are in the 16 GB that was pulled for the
image-to-3D path. Both gated-model workarounds this package already carries —
the background remover it never runs, and the flash-attn substitution — apply
unchanged, and are now shared through `_prepare_trellis`.

**The geometry survives.** `preprocess_mesh` normalises into a unit cube and
swaps axes; it does not re-mesh. On a 46-face massing the output came back with
42 faces (welding) and the plan aspect exactly preserved: 9.32 × 19.70 m in,
0.47 × 1.00 out.

**The mesh has to go in Y-up.** `preprocess_mesh` applies `(x, y, z) →
(x, -z, y)`, which is a Y-up convention. Handed this package's Z-up mesh it lays
the building on its side, and the picture — which shows it standing — then
paints the facade onto the roof. Fed a red brick house that produced a uniformly
red slab, which is what the first run looked like.

**The resolution is not the image-to-3D one.** At `resolution=512`,
`texture_size=1024` the roof came back washed out — a pale field with a faint
chequer in it. At `1024`/`2048` it reads as tiles and the walls as boarding.

| | 512 / 1024 | 1024 / 2048 | image → 3D |
|---|---|---|---|
| time | 7 s | 26 s | 17 s (median) |
| roof | washed out | reads as tiles | crisp |
| footprint | exact | exact | fitted |

**The inverse transform is exact.** Composing the Y-up conversion with
`preprocess_mesh`'s normalisation and inverting both reproduces the input
vertices to 0.0.

## What was not

**The GLB does not carry the placement.** Read back with `read_glb`, the
textured mesh lands at the wrong height and scores an IoU of 0.31 against the
plot it was made from — despite the transform above being exact in memory. The
exporter puts the placement somewhere the reader does not look; `read_glb` takes
accessors and ignores node transforms, which is right for the files this package
writes itself and wrong for this one. Not yet run down. Until it is,
`texture_mesh` returns the scene-space vertices as `placed` and a caller should
use those rather than re-reading the file.

**It does not yet look better.** Side by side on the same plot, the directly
textured massing at 1024/2048 is closer to the input picture than the 512 one
but still less crisp than the mesh the image-to-3D path invents. That is the
honest state: the *footprint* argument is settled and the *appearance* argument
is not.

## What to try next, in the order I would try it

1. **A photograph instead of a render.** This is the point of the exercise and
   it has not been tested. Everything measured above was fed the massing wearing
   SDXL facade sheets — the model's idea of a wall, at one remove. The texturing
   pipeline only wants a picture of the *material*, and a photograph of a real
   Japanese house is a much better one. The package already has reference-image
   support in `FacadeOptions` for exactly this reason.
2. **Subdivide the massing.** The texture is decoded into a voxel field and
   projected onto the mesh; a 42-triangle box gives it very little to project
   onto. Subdividing the walls and roof before texturing costs nothing and may
   be most of the crispness gap.
3. **More than one view.** `get_cond` takes a list of images and is called with
   one. Two or three views of the same massing would give it the sides the
   single three-quarter view cannot see.
4. **Then decide.** If the appearance closes the gap, this replaces the
   image-to-3D path and the whole fit — `fit_to_footprint`, the stretch, the
   seating, `keep_below` — becomes dead code. If it does not, it is still worth
   keeping for the buildings the fit rejects, where the alternative is a
   procedural box.


---

# Give it the envelope and let it invent inside

The texturing route above keeps the footprint but can only ever return a
textured box, because the box is what it is handed. The better question is
whether the *generation* can be constrained instead — a prompt and a footprint
in, a building out. It can, and the mechanism is already exposed.

## How TRELLIS.2 actually generates

`run()` is three stages, and the first one is the whole of the plan:

```
coords     = sample_sparse_structure(cond, 32 or 64)   # occupied voxels
shape_slat = sample_shape_slat(cond, model, coords)    # geometry in them
tex_slat   = sample_tex_slat(cond, model, shape_slat)  # PBR on it
mesh       = decode_latent(shape_slat, tex_slat, res)
```

`coords` is an ordinary tensor of occupied cells in a cube, and
`sample_shape_slat` takes it as an *argument*. So it can be replaced: voxelise
the plot's own prism — the footprint extruded to the building's height — and the
plan and the height stop being something the model guesses and something the fit
has to repair. The model is then left to invent what we actually want from it,
inside that envelope.

## Measured, on the largest plot of the Kashiwanoha map

| grid | eave room | time | footprint IoU, **uniform fit only** |
|---|---|---|---|
| 32 | 0 | 10 s | 0.822 |
| 64 | 0 | 47 s | 0.743 |
| 32 | 0.6 m | 9 s | **0.882** |

Against the current path's 0.917 — which needs the anisotropic stretch, the yaw
sweep and a 3 % rejection rate to get there. Two things in that table were not
obvious:

**A finer grid is worse, and five times slower.** At 64 the envelope is a
tighter cast of the box and the shape model has less room to depart from it;
what comes back is smaller relative to the plot, not truer to it.

**Leaving room for the eaves is worth more than resolution.** Growing the prism
by 0.6 m in plan took the IoU from 0.822 to 0.882 at the same 9 s. A roof
overhangs, and an envelope drawn exactly on the walls has nowhere to put it.

**The voxel axes are the identity.** `(i, j, k)` in this package's own order,
handed straight over. Not the Y-up convention `preprocess_mesh` uses — that
belongs to the texturing pipeline's *mesh* input, and applying it here stands
the building on end. Checked by comparing the output's sorted extents against
the envelope's: identity gives ratios 0.225 / 0.501 / 1.0 against the envelope's
0.254 / 0.473 / 1.0, and puts the height on the height axis. Every other mapping
tried put a plan dimension there.

## Where this leaves the three routes

| | shape | surface | footprint | time |
|---|---|---|---|---|
| image → 3D (in the pipeline) | invented | invented | fitted, 0.917 mean, 3 % rejected | 17 s |
| mesh → texture | **the map's, exactly** | invented | exact | 26 s |
| **envelope → 3D** | **invented inside the map's** | invented | 0.882 uniform, no stretch | **9 s** |

The third is the one worth pursuing. It is the fastest of the three, it is the
only one where the model is still free to produce something that is not a box,
and its footprint error is already close to a path that needs three corrective
mechanisms to beat it — none of which it needs.

---

# The picture, once it no longer has to carry the shape

That was the untested half, and it is now tested: with the envelope holding the
plan, the conditioning picture is free to be a photograph rather than a render
of this particular block. It is in the package as
`reconstruct.to_mesh_in_envelope` and reached from the pipeline with
`--envelope`.

## What decides whether it works, and it is not the prompt

**One whole building, on nothing.** TRELLIS takes the entire frame as the
subject. Asked for a *photograph of an ordinary Japanese suburban house*, SDXL
returns the house in its street — sky, garden, fence, the neighbours — and
`cut_out` has no plain field to key out, so all of it is fed in. What came back
was a smear with the sky and the grass baked into the walls. This is the whole
difficulty, and the prompt has to be about the *frame*, not the building.

Measured over three phrasings, two seeds each, by the share of the frame
`cut_out` can call backdrop:

| prompt | backdrop | what came back |
|---|---|---|
| "…house, plain white background" | 8–15 % | a close-up crop; the mesh is a smear |
| "studio product photograph of a **scale model** of …" | 39–45 % | a building |
| "isolated object on a flat neutral grey backdrop, product shot" | 48–59 % | a building |

"A model of" is what the frame responds to, and it costs nothing: the material
and the weathering survive it, which is all the picture is being asked for. The
number is now `reconstruct.backdrop_share`, and `reconstruct.photographs`
redraws on another seed below 0.25 — the failure is far cheaper to catch in the
picture than in the mesh.

Three pictures, three meshes, on the same plot: footprint IoU 0.878, 0.884,
0.884. The envelope owns the plan, so the picture only decides whether the
result is a *building* or a slab.

## The model does not fill its envelope, and that is a height bug

The clearest defect the first whole-map run turned up, and it is only visible
at district scale: every building was short. Over 185 reconstructions the
height came to **0.81 of the block height they were given**, 174 of them more
than a tenth short, against **1.25** for the image-to-3D path — which invents
its own massing and puts a roof on top of it.

Two things are going on and only one is a surprise. A roof rises, so a building
*should* stand above its block; `eave_room` had already given the roof somewhere
to overhang in plan and nothing gave it anywhere to rise. But the shape model
also does not reach the top of what it is handed: on a 6 m block with no
headroom it returned 4.8 m.

The allowance is a **fraction** of the block, not metres — a metre of ridge on
a shed is a different building and on an office block is nothing. One plot per
storey count:

| block | headroom | result | vs block | IoU |
|---|---|---|---|---|
| 3 m (1 storey) | 0 | 2.8 m | 0.92 | 0.888 |
| 3 m | 40 % | 3.7 m | **1.23** | 0.874 |
| 6 m (2 storeys) | 0 | 4.8 m | 0.81 | 0.920 |
| 6 m | 40 % | 7.5 m | **1.26** | 0.926 |
| 9 m (3 storeys) | 0 | 7.7 m | 0.86 | 0.929 |

0.4 lands on the other path's 1.25 at every storey count and the footprint is
unmoved. It buys a hip line as well, though only a slight one: the envelope is
a box and what is generated inside it stays boxy, which is the trade this route
makes and not a defect in it.

## A finer grid is worse, and it was not a mismatch

The earlier table showed grid 64 losing to grid 32, and the obvious explanation
was that a 64 cube was being handed to the 512 flow model, which samples at 32.
It is not that. Run as the matched pair — `pipeline_type='1024'` with grid 64,
which is what `run()` itself does — the footprint is unchanged (0.885 against
0.888) and the *shape* is worse: the 512/32 mesh has modelled ridge tiles, an
eave and a cream wall with window openings, and the 1024/64 one is a featureless
beige box. A tighter cast of the envelope leaves the shape model nothing to
invent. 32 is not a resolution compromise, it is the setting.

## What it cost, and the bug it turned up

The first six-building run lost one to `OutOfMemoryError`, three tries running,
on a card with 25 GB free — "6.00 GiB allowed".
`torch.cuda.set_per_process_memory_fraction` is process-wide and sticky, and the
tile stage sets it to 6 GB so as not to grow into a neighbour on a shared card.
Every model that ran after it in that process inherited the cap. It is now
`texture.vram_budget`, a scope rather than a setting, and it was a live defect
on the shipped path too.

## The four routes

| | shape | surface | footprint | time |
|---|---|---|---|---|
| image → 3D (was in the pipeline) | invented | invented | fitted, 0.917 mean, 3 % rejected | 17 s |
| mesh → texture | the map's, exactly | invented | exact | 26 s |
| envelope → 3D, from a render | invented inside the map's | invented | 0.882 | 9 s |
| **envelope → 3D, from a photograph** | invented inside the map's | **photographed** | 0.85–0.97 | 9 s |

## Both routes over the whole map, 189 plots

| | used | mean IoU | median | **max stretch** | median time |
|---|---|---|---|---|---|
| image → 3D | 184/189 | 0.912 | 0.916 | **1.15** | 16 s |
| envelope → 3D | 185/189 | 0.904 | 0.908 | **1.00** | 31 s |

The IoU is a wash and the stretch column is the point: the image-to-3D path
reaches 0.912 by squeezing meshes up to 15 % along the plot's long axis, and
the envelope reaches 0.904 with no distortion at all, because there is nothing
to correct. It is slower per building here rather than faster — the 9 s figure
was one plot at 512 with a warm pipeline, and a real run pays for the mesher
and the GLB as well.

Side by side at district scale the trade is plain, and it is not the one the
IoU measures. The envelope street has actual materials — dark timber lattice,
corrugated metal, kawara, cream stucco, weathering — where the image-to-3D
street is a uniform grey-blue with windows painted on. The image-to-3D street
has crisper geometry: real hips and gables, where a box envelope filled in
gives a shallow hip at best.

## The one that the numbers got wrong

`sample_sparse_structure` returns a **surface**, so a solid prism is not the
kind of object the shape model is used to: its own occupancy measured 4905
cells of 32768 on this map's conditioning photograph, each column filling 0.62
of the levels between its own top and bottom, while solid prisms for these
plots ran to 11 000 at the median and 29 600 at the worst. That is a real
observation, and it is what the memory failures were made of — the nineteen
plots that would not generate at all, six attempts each across two processes,
were exactly the densest, not the largest.

Hollowing the envelope to a one-voxel shell fixed all of it, on paper:

| | image → 3D | envelope, solid | envelope, shell |
|---|---|---|---|
| standing | 184/189 | 167/189 | **186/189** |
| mean IoU | 0.912 | 0.902 | 0.901 |
| height / block | 1.25 | 1.17 | **1.25** |
| memory failures | 0 | 19 | **0** |
| per building | 16 s | 31 s | **13 s** |

And it produced a district of cages. Walls with daylight through them, roofs
over open frames. **The footprint IoU cannot see it** — a cage has the same
plan outline as a house — and it was slightly *better* on the cages, which is
how a whole map got generated before anybody looked at it. Nor does the
obvious mesh statistic help: open edges ran 0.25 on the shells against 0.38 on
the solids, because a marching-cubes surface is ragged either way. Nine meshes
rendered on their own looked fine; the fault was only visible in the street.

So the envelope stays solid, and the memory ceiling is handled where it
belongs. 22 272 cells generate and 28 672 do not, so a plot over
`VOXEL_BUDGET` is peeled from the inside, innermost layer first — the middle
of a building being the least informative thing you can tell a model about
where the building is. On this map 159 of 189 stay whole, 30 are peeled, and
the worst lands at 19 994.

The lesson is not about voxels. Every number moved the right way and the thing
got worse, so the numbers were not measuring the thing. Look at the render.

## Still open

- **Memory.** The mesher wants a large contiguous allocation and does not
  always get one: 7 buildings in 61 were lost before `district._release_the_card`
  made the retry give the allocator its blocks back, and 1 in 128 after.
- **`texture_mesh`'s GLB placement**, from the first half of this document.
- **Subdividing the envelope.** Nothing here has tried giving the prism a
  pitched top rather than a flat one, which is the obvious way to get a roof
  form as well as a roof material out of it.
