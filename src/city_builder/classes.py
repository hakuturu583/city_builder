"""What each surface *is*, and whether its colour may be regenerated.

A texturing pass — a diffusion model, a bake, a hand-painted material — needs
to know more than "this is a triangle". Painted markings carry information: a
stop line is white because it is a stop line, and a model that reinvents its
colour has destroyed the thing the mesh was built to represent. Asphalt and
terrain are the opposite: their authored colour is a placeholder and inventing
something better is the whole point.

So every object is tagged with:

* ``label`` — a semantic name, in the vocabulary a segmentation dataset would
  use (``road``, ``road_marking``, ``sidewalk``, ``terrain``, ...);
* ``paint`` — :data:`PRESERVE` if the authored colour is meaningful and must
  survive, :data:`GENERATE` if it is a placeholder;
* ``mask_colour`` — a flat, well-separated colour, so a segmentation render
  can be read back reliably;
* ``pass_index`` — Blender's object index, for a Cycles ``IndexOB`` pass.

The tags reach a consumer three ways, because different pipelines read
different things: object custom properties (kept in ``.blend`` and exported to
glTF as ``extras``), a colour attribute on every face corner (survives a merge
or a re-import that flattens the object list), and a JSON manifest.
"""

from __future__ import annotations

from dataclasses import dataclass

PRESERVE = "preserve"
GENERATE = "generate"


@dataclass(frozen=True)
class SurfaceClass:
    """Semantics and texturing policy for one group of surfaces."""

    group: str
    label: str
    paint: str
    material: str
    mask_colour: tuple[float, float, float]
    pass_index: int
    note: str = ""

    @property
    def preserved(self) -> bool:
        return self.paint == PRESERVE

    def to_json(self) -> dict:
        return {
            "group": self.group,
            "label": self.label,
            "paint": self.paint,
            "material": self.material,
            "mask_colour": list(self.mask_colour),
            "pass_index": self.pass_index,
            "note": self.note,
        }


#: Every group this package emits, in pass-index order.
CLASSES: dict[str, SurfaceClass] = {
    c.group: c
    for c in (
        SurfaceClass(
            "Ground", "terrain", GENERATE, "ground", (0.35, 0.25, 0.15), 1,
            note="interpolated between the roads; the least trustworthy geometry here",
        ),
        SurfaceClass(
            "Roads", "road", GENERATE, "asphalt", (0.15, 0.15, 0.18), 2,
            note="carriageway away from junctions",
        ),
        SurfaceClass(
            "Junctions", "road", GENERATE, "asphalt", (0.45, 0.15, 0.55), 3,
            note="lanelets carrying a turn_direction",
        ),
        SurfaceClass(
            "Crosswalks", "road", GENERATE, "asphalt", (0.15, 0.45, 0.75), 4,
            note="the crossing surface; the bars themselves are CrosswalkStripes",
        ),
        SurfaceClass(
            "Walkways", "sidewalk", GENERATE, "concrete", (0.85, 0.6, 0.2), 5,
        ),
        SurfaceClass(
            "Curbs", "curb", GENERATE, "concrete", (0.2, 0.65, 0.35), 6,
            note="vertical faces; a texturing pass sees them almost edge-on from above",
        ),
        SurfaceClass(
            "LaneMarkings", "road_marking", PRESERVE, "marking", (0.95, 0.95, 0.95), 7,
            note="solid and dashed lane lines, straight from the map",
        ),
        SurfaceClass(
            "StopLines", "road_marking", PRESERVE, "marking", (0.9, 0.2, 0.2), 8,
        ),
        SurfaceClass(
            "CrosswalkStripes", "road_marking", PRESERVE, "marking", (0.95, 0.85, 0.25), 9,
            note="one bar per pedestrian_marking ring in the map",
        ),
        SurfaceClass(
            "Buildings", "building", GENERATE, "facade", (0.75, 0.35, 0.55), 10,
            note="procedural: the map says nothing about what stands here",
        ),
        SurfaceClass(
            "Roofs", "roof", GENERATE, "roof", (0.30, 0.35, 0.85), 11,
            note="separate from the facades; a texturing pass treats them differently",
        ),
        SurfaceClass(
            "ViaductDecks", "viaduct", GENERATE, "concrete", (0.25, 1.00, 0.25), 12,
            note="soffit and sides of an elevated road; the map surveys only its top",
        ),
        SurfaceClass(
            "ViaductParapets", "viaduct", GENERATE, "concrete", (0.75, 1.00, 0.00), 13,
            note="the wall along an elevated deck, distinct from a kerb at grade",
        ),
        SurfaceClass(
            "ViaductPiers", "viaduct", GENERATE, "concrete", (1.00, 0.25, 1.00), 14,
            note="columns from the reconstructed ground to the soffit",
        ),
        SurfaceClass(
            "RoadInfill", "road", GENERATE, "asphalt", (0.00, 0.55, 1.00), 15,
            note="the slivers between lanelets: a hole to the street on a deck, "
                 "and a 5 cm drop into the terrain at grade",
        ),
        SurfaceClass(
            "Water", "water", GENERATE, "water", (0.00, 0.85, 0.85), 16,
            note="the flat top of a body of standing water; the bed a few "
                 "centimetres under it is still Ground",
        ),
        SurfaceClass(
            "Poles", "pole", GENERATE, "concrete", (0.20, 0.20, 0.95), 18,
            note="lamp posts and utility poles on the pavement; a street "
                 "without them reads as a model, and a conditioned generator "
                 "cannot add one the geometry says is not there",
        ),
        SurfaceClass(
            "Trees", "tree", GENERATE, "ground", (0.10, 0.75, 0.10), 19,
            note="street trees: a trunk and a lump of canopy, which is all the "
                 "silhouette a control image needs",
        ),
        SurfaceClass(
            "Fences", "fence", GENERATE, "fence", (0.95, 0.55, 0.00), 17,
            note="pedestrian railings on the ground — round standing water and "
                 "along a drop; a guardrail on the carriageway is road furniture "
                 "and is not this",
        ),
    )
}

#: Groups whose authored colour must survive a texturing pass.
PRESERVED_GROUPS = tuple(name for name, c in CLASSES.items() if c.preserved)

#: Group → material key, kept for the scene builder.
MATERIALS = {name: c.material for name, c in CLASSES.items()}


def get(group: str) -> SurfaceClass:
    """Class for a group, falling back to an unlabelled generated surface."""
    known = CLASSES.get(group)
    if known is not None:
        return known
    return SurfaceClass(group, "unknown", GENERATE, "asphalt", (1.0, 0.0, 1.0), 0)


def manifest() -> dict:
    """The whole registry, for a consumer that wants it up front."""
    return {
        "policies": {
            PRESERVE: "authored colour is meaningful; keep it",
            GENERATE: "authored colour is a placeholder; regenerate freely",
        },
        "classes": [c.to_json() for c in CLASSES.values()],
    }
