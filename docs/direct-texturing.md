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

What is untested and would decide it: a real photograph as the conditioning
image rather than the massing wearing SDXL sheets, and whether the remaining
0.12 of IoU closes with a slightly larger eave allowance or is the shape model
declining to fill its envelope.
