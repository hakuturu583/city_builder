# Ground: land cover, and a surface the road belongs to

A design note. What is here now, what the literature says, what this map
actually measures, and the algorithm those three imply. Nothing in this
document is implemented yet; each stage names the measurement that will decide
whether it stays.

## What exists today

`city_builder.ground` interpolates a surface from the only elevation a
Lanelet2 map carries — the carriageway:

1. elevated structure is found geometrically and dropped, because these maps
   have no `bridge` or `layer` tag;
2. what is left is binned into a 10 m grid as a **lower envelope**;
3. the gaps — most of a city block — are filled by Jacobi relaxation with the
   measured cells held fixed, Gaussian-smoothed, and re-clamped under the
   envelope;
4. the result is clipped against the dissolved road outline, and seam vertices
   are snapped to the nearest road boundary point.

`HeightMap.support` records which cells are measured and which are invented.
One tiled texture is applied over the whole thing at `tile_metres = 12`.

Two things are wrong with it, and they are the two the survey below names.

## What the survey found

A literature and tooling survey ran over four areas: constrained terrain
reconstruction, land-cover assignment, heterogeneous terrain texturing, and
existing Lanelet2/OpenDRIVE scene generators. The full findings are long; these
are the load-bearing ones.

**The gap fill is the wrong operator.** Step 3 is a discrete *harmonic*
(Laplace) solve with Dirichlet constraints. Two consequences follow from that
and neither is a tuning artefact:

- A harmonic solve is only C⁰ at its constraints. There is a **crease along
  every kerb** by construction. The biharmonic (thin-plate) solve is C¹ there
  and rolls off the shoulder the way an embankment does — this is the
  `igl::harmonic(..., k=1)` versus `k=2` distinction, and it is one line of
  algebra away.
- Harmonic functions obey the **maximum principle**: an interior value can
  never exceed its boundary. A hill inside a city block is not unlikely under
  the current fill, it is *impossible*. Every block interior is a saddle
  between the streets around it.

**The seam is approximated where it could be exact.** The kerb is what GIS
calls a **breakline** — Esri's own worked example is "a highway with
fluctuating elevation … incorporated as a soft breakline". The standard
treatment is a **constrained Delaunay triangulation** with the lane boundaries
as forced edges, which is also what RoadRunner does: its terrain is not a
heightfield at all but a polygon whose boundary *shares nodes with the road
network*, so the seam is watertight by construction rather than by a
nearest-neighbour snap within a tolerance. `shapely.constrained_delaunay_triangles`
is available in the version already installed, so this costs no new dependency.

**Nobody else has solved this.** CARLA emits void beyond the carriageway and
fences it with walls; esmini emits a flat quad; blosm and CityEngine drape the
road onto a DEM rather than the reverse; Lanelet2 itself generates no mesh; and
OSM2World's own 2026 roadmap lists "OSM semantics informing the terrain" as
open. The one design worth copying is **OSM2World's `EleConstraintEnforcer`**,
which states the problem declaratively — `MIN`/`MAX`/`EXACT` constraints,
vertical clearance above a segment, incline bounds, smoothness. That vocabulary
subsumes the lower-envelope clamp, the clearance test and the drop, which are
currently three separate heuristics.

**A land-cover raster cannot see a road, but we already know where it is.**
ESA WorldCover has no paved class: roads, car parks and rooftops are all
`Built-up (50)`. So the usual remote-sensing pipeline runs backwards here. The
vectors are the authority over everything they cover, and the raster is only a
residual fill for what they do not claim.

**Do not invent a class vocabulary.** OSM2World's `surfaceMaterialMap` is
about twenty ground materials — asphalt, concrete, paving stone, sett,
cobblestone, gravel, pebble, scree, earth, grass, grass paver, sand, rock,
stone, snow, ice, wood, woodchips, scrub — that have already survived contact
with global OSM data, and it extends `classes.py` naturally.

## What this map measures

Three probes on the Kashiwanoha map, because the design depends on them.

**There is a real DEM under it, free and unregistered.** GSI elevation tiles at
`https://cyberjapandata.gsi.go.jp/xyz/dem5a_png/15/{x}/{y}.png`, decoded as
`h = (R·65536 + G·256 + B)·0.01` m with `(128,0,0)` as no-data, cover **100 %
of this map at 3.87 m/px** — the 5 m airborne-lidar DEM5A. `seamlessphoto` at
z18 gives 0.48 m/px orthophoto over the same footprint. Both are ordinary XYZ
tiles under the 国土数値情報 terms; `LocalFrame.to_wgs84` already converts scene
metres to the lat/lon they need. (`dem_png` at z14 is the 10 m DEM10B fallback;
z15+ is 404 for the 5 m product outside its coverage.)

**The DEM and the lanelets disagree by metres.** Over 674 road-surface samples:

| | |
|---|---|
| DEM coverage | 100 % |
| median (DEM − scene z) | **16.31 m** — a constant datum offset, solvable |
| residual after removing it | mean 1.30 m, p90 3.19 m, **max 3.51 m** |
| relief the lanelets claim | 5.97 m over 163 × 143 m |
| relief the DEM sees | 1.36 m over the same points |

**The lanelet z is locally clean and globally drifting.** Road-surface vertices
within 1 m of each other in plan differ by a median of 1 cm in height; within
10 m, 15 cm. So this is not sample noise — it is a smooth, low-frequency tilt of
roughly 4 % across the block that the national lidar does not see.

This measurement does not prove which source is right, and it should not be
read as proving it. What it establishes is that they cannot both be, that the
disagreement is metres rather than centimetres, and therefore that the
algorithm has to state which one wins at which spatial frequency instead of
letting whichever is loaded last decide.

## The algorithm

Five stages. Stages 1–3 make the surface; 4–5 make it look like something.

### 1. A class grid, vectors first

Rasterise into a per-cell class id + weights, in this precedence:

1. **The map's own surfaces** — the road, junction, crosswalk and walkway
   polygons already dissolved in `build_mesh`, burned in with `surface=*`
   semantics. Survey accuracy, and no classifier can match it.
2. **OSM areas** over the map footprint, by Overpass on the bbox
   `LocalFrame.to_wgs84` gives: `surface` > `natural`/`landcover` > `landuse`.
   This is where water, sand, grass, farmland and bare ground come from when
   somebody has mapped them.
3. **A raster for the residue.** Dynamic World (CC-BY, 10 m) in preference to
   ESA WorldCover, for one reason: it publishes **a probability per class**
   rather than an argmax label, and those probabilities *are* the splat weights
   — which removes the need to author blend masks at all.
4. **Procedural fallback** where nothing claims a cell: slope × distance-to-water
   rules, optionally seeded by a D-∞ flow-accumulation pass so wet ground lands
   somewhere a slope would actually put it. Not a Whittaker climate table — one
   city is one climate cell, and the two axes would be constant.

Vocabulary: OSM2World's material set, as an extension of `classes.py`.

*Measured by*: what fraction of cells each tier claims, on a map with good OSM
coverage and on one without.

### 2. Constraints, not a heightmap

Replace the binning with a constraint set, in OSM2World's vocabulary:

- **`EXACT`** on lane-boundary vertices — the carriageway is a survey.
- **`MAX`** under a deck found by the existing geometric overpass test, so the
  ground passes beneath rather than being dragged up to it. Keeping this
  geometric is right: every production tool excludes rather than interpolates,
  because a 2.5D field cannot fold. Adopt the USGS tie-break — **when it cannot
  be told whether a structure is a bridge or a culvert, treat it as a culvert**,
  i.e. leave the road as ground.
- **`MAX`** at the water surface of a `natural=water` cell, which is flat by
  definition and is the one class whose elevation is known without measuring it.
- **Incline bounds** where a class implies them — an embankment batter at the
  angle of repose rather than a smooth blend, which is what a real road does.

The lower-envelope clamp becomes one of these rather than a pass of its own.

### 3. One solve: gradients from the road, values from the DEM

This is where the measurement above is spent. A **screened Poisson** solve over
the constrained Delaunay triangulation:

    minimise   ‖∇²z‖²                        (thin plate — C¹ at the kerb)
             + λ ‖z − z_DEM − δ‖²            (screening — follow the lidar loosely)
    subject to z = z_lanelet on EXACT nodes  (Dirichlet — the road is the road)
               z ≤ z_max     on MAX nodes

with `δ` the datum offset **solved for, not assumed** — the probe says 16.31 m
here and it will be different on every map.

The split falls out of what each source is good at. The lanelets are exact at
1 m and drift at 100 m; the DEM is coarse at 1 m and correct at 100 m. So the
lanelets set the *values* on the carriageway and its immediate shoulder, and
the DEM sets the *low-frequency shape* of everything else through the screening
term. `λ` is the one knob: at 0 this is the pure thin-plate fill, i.e. today's
behaviour with the crease removed; large, it is the DEM with the roads burned
in. Mitášová's tension parameter is the same idea from the spline side and is
worth trying as the alternative parameterisation.

Where no DEM is available the term simply drops and the solve degrades to the
biharmonic fill — which is still strictly better than what is there now.

Sparse direct or multigrid, not 400 Jacobi iterations: the same answer to
machine precision in one pass.

*Measured by*: the existing `support` statistic, plus RMS against the DEM on
held-out cells, plus the seam gap at the kerb — which should be exactly zero
once the triangulation is constrained.

**Two known traps.** A thin-plate solve **overshoots** — unlike natural
neighbour interpolation it is not range-bounded, so it can bulge above the
highest constraint across a wide block; keep a clamp or lean on the screening
term. And **do not adopt AGREE-style fixed-radius ramps**: the hydrology
literature documents them leaving a parallel ridge or ditch flanking every
corridor, which is exactly the artefact this is trying to avoid.

### 4. The mesh reads the classes

Water cells are flattened and get a separate surface. Class boundaries that are
real discontinuities — a kerb, a retaining wall, a shoreline — become **hard
breaklines** in the triangulation rather than being smoothed across. Everything
else stays soft.

### 5. Texture, in the pipeline that already exists

This package already bakes paint into the carriageway texture through the
ribbon parameterisation, at a measured texels-per-metre policy, with a
`preserve`/`generate` mask separating authored from regenerable colour.
Extending that to a per-class ground texture is architecturally continuous, and
it is the same argument Far Cry 4's adaptive virtual texturing and Unreal's
runtime virtual texturing make: bake once, constant per-pixel cost, unlimited
classes and decals.

- **Per-class tiles** from the existing diffusion generator, indexed by class
  id — a texture array rather than four splat channels.
- **Height-blended boundaries** in Mishkinis' formulation: a per-material height
  map multiplied against the class weight, argmax with a small soft window, so
  gravel pokes through sand along its own silhouette instead of cross-fading.
- **The weight is perturbed by tiling noise *before* the height comparison**,
  not after. This is not optional here. A 10 m class grid is a very coarse
  weight map, and height blending mixes far fewer pixels than linear blending —
  without the perturbation the result is 10 m blobs with crisp edges, which is
  worse than the mush it replaced.
- **Anti-tiling**: Heitz & Neyret's histogram-preserving blending is the
  strongest candidate, because the tiles come from a diffusion model on exactly
  the stochastic natural exemplars it targets. If its preprocessing and
  compression coupling bite, Mikkelsen's hex-tiling needs no precomputation at
  all.
- Never let *every* layer be height-blended — the documented failure is all
  heights reaching zero together and the blend collapsing to black speckle,
  worst on normals. Base layer linear, upper layers height-blended.

## The order to do it in

Each of these stands alone and is measurable on its own:

1. **Biharmonic instead of harmonic**, solved sparsely. One operator change;
   removes the kerb crease and the maximum-principle flatness; replaces 400
   Jacobi iterations with one solve. No new data, no new dependency.
2. **Fetch the GSI DEM and solve for the datum offset.** Turns the problem from
   invention into fitting. This is where the biggest error is: 3.5 m.
3. **The class grid**, vectors first, raster second.
4. **Constrained Delaunay with the lane boundaries as breaklines**, which makes
   the seam exact and retires `snap_tolerance`.
5. **Per-class texturing** on the baking pipeline that already exists.

## Where the sources are

Named because each claim above should be checkable, not because the list is
exhaustive.

- Mitášová & Mitáš, *Interpolation by regularized spline with tension I*,
  Math. Geology 25(6), 1993 — tension as the stiffness knob; GRASS `v.surf.rst`.
- Kazhdan & Hoppe, *Screened Poisson Surface Reconstruction*, ACM TOG 32(3),
  2013 — adding samples back as an interpolation term without breaking
  linearity.
- Pérez, Gangnet & Blake, *Poisson image editing*, SIGGRAPH 2003 — Dirichlet
  fill from a surround, and the gradient-guided generalisation.
- Hnaidi et al., *Feature based terrain generation using diffusion equation*,
  CGF 29(7), 2010 — terrain from feature curves by multigrid diffusion; the
  closest published problem to this one.
- Bruneton & Neyret, *Real-Time Rendering and Editing of Vector-based
  Terrains*, CGF 27(2), 2008 — one vector description carrying both footprint
  and appearance, which is stages 1 and 5 as a single idea.
- Shewchuk, *Triangle*; `CGAL::Constrained_Delaunay_triangulation_2`; Esri,
  *Breaklines in surface modeling* — hard versus soft breaklines.
- USGS, *Lidar Base Specification* — ignored-ground near breaklines, bridge
  removal, and the culvert tie-break.
- Hellweger, *AGREE: DEM Surface Reconditioning System*, 1997 — the naive
  baseline, and its documented parallel-ridge artefact.
- Zhang et al., *Cloth Simulation Filtering*, Remote Sensing 8(6):501, 2016 —
  one-sided constrained relaxation, which is what the envelope clamp
  approximates.
- Brown et al., *Dynamic World*, Scientific Data 9:251, 2022 — 10 m land cover
  with per-class probabilities.
- OSM2World `EleConstraintEnforcer` and `DefaultMaterials.surfaceMaterialMap` —
  the constraint vocabulary and the material vocabulary.
- Mishkinis, *Advanced Terrain Texture Splatting*, 2013 — height blending.
- Heitz & Neyret, *High-Performance By-Example Noise using a
  Histogram-Preserving Blending Operator*, HPG 2018; Mikkelsen, *Practical
  Real-Time Hex-Tiling*, JCGT 11(3), 2022 — anti-tiling.
- Patchwork++ (IROS 2022) — ground segmentation, for the case below.

## One thing that would change all of it

An Autoware map bundle conventionally ships `pointcloud_map.pcd` beside
`lanelet2_map.osm`. **This map does not** — `maps/` holds the `.osm` alone.
If a bundle does carry one, the ground between the roads is not unknown at all:
it is measured, at lidar density, including verges, embankments and building
bases. Patchwork++ or cloth-simulation filtering would extract it, `support`
would collapse to near zero, and stages 2 and 3 above would reduce to fitting
a surface to dense samples rather than reconstructing one from a network. That
path should be taken whenever the input allows it, and the interpolation is
the fallback rather than the plan.
