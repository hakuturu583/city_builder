"""The surface class registry: semantics and texturing policy."""

from __future__ import annotations

from city_builder import GENERATE, PRESERVE, classes
from city_builder.surfaces import Z_BIAS


def test_every_emitted_group_has_a_class():
    """A surface with no class would reach a texturing pass unlabelled."""
    for group in Z_BIAS:
        assert group in classes.CLASSES, group
    assert "Ground" in classes.CLASSES


def test_only_painted_markings_are_preserved():
    """The policy is the point: paint carries meaning, asphalt does not."""
    assert set(classes.PRESERVED_GROUPS) == {"LaneMarkings", "StopLines", "CrosswalkStripes"}
    for group in classes.PRESERVED_GROUPS:
        assert classes.CLASSES[group].label == "road_marking"
        assert classes.CLASSES[group].preserved


def test_generated_surfaces_are_the_rest():
    generated = [name for name, c in classes.CLASSES.items() if c.paint == GENERATE]
    assert set(generated) == {
        "Ground", "Roads", "Junctions", "Crosswalks", "Walkways", "Curbs", "Buildings", "Roofs",
    }


def test_pass_indices_are_unique_and_nonzero():
    """0 is Blender's default, so a stray object cannot be mistaken for a class."""
    indices = [c.pass_index for c in classes.CLASSES.values()]
    assert len(set(indices)) == len(indices)
    assert 0 not in indices


def test_mask_colours_are_distinguishable():
    """A segmentation render has to survive compression and antialiasing."""
    colours = [c.mask_colour for c in classes.CLASSES.values()]
    assert len(set(colours)) == len(colours)
    for i, a in enumerate(colours):
        for b in colours[i + 1:]:
            assert sum(abs(x - y) for x, y in zip(a, b)) > 0.30


def test_unknown_groups_fall_back_to_a_generated_surface():
    fallback = classes.get("SomethingElse")
    assert fallback.paint == GENERATE
    assert fallback.label == "unknown"


def test_manifest_lists_every_class_with_its_policy():
    manifest = classes.manifest()
    assert set(manifest["policies"]) == {PRESERVE, GENERATE}
    assert len(manifest["classes"]) == len(classes.CLASSES)
    assert {c["group"] for c in manifest["classes"]} == set(classes.CLASSES)
