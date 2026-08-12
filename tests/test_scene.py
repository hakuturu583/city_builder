"""Scene-layer tests that do not need Blender.

``scene`` imports bpy only inside the functions that touch it, so the mesh
assembly around it stays testable.
"""

from __future__ import annotations

from city_builder import scene
from city_builder.geometry import Mesh, Polygon, Ribbon


def test_group_to_mesh_merges_mixed_shapes():
    ribbon = Ribbon(1, [(0, 1, 0), (10, 1, 0)], [(0, -1, 0), (10, -1, 0)])
    polygon = Polygon(2, [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])
    mesh, skipped = scene.group_to_mesh([ribbon, polygon])
    assert skipped == 0
    assert len(mesh.faces) == 1 + 2  # one quad, one fan of two
    assert max(max(f) for f in mesh.faces) == len(mesh.vertices) - 1


def test_group_to_mesh_passes_a_prebuilt_mesh_through():
    prebuilt = Mesh([(0, 0, 0), (1, 0, 0), (1, 1, 0)], [[0, 1, 2]])
    mesh, skipped = scene.group_to_mesh([prebuilt])
    assert skipped == 0
    assert mesh.faces == [[0, 1, 2]]


def test_group_to_mesh_reports_dropped_faces():
    # The bounds cross, so the quad between them is a bowtie.
    bowtie = Ribbon(1, [(0, 1, 0), (10, 1, 0)], [(0, -1, 0), (10, 5, 0)])
    mesh, skipped = scene.group_to_mesh([bowtie])
    assert skipped == 1
    assert not mesh.faces
