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


# ---------------------------------------------------------------------------
# What leaves in an FBX
# ---------------------------------------------------------------------------


def test_a_texture_an_engine_cannot_decode_is_rewritten_as_png(tmp_path):
    """FBX embeds the texture file byte for byte — the exporter does not
    transcode. The reconstructions arrive as GLBs written with WebP, so a
    district exported straight to FBX carries forty images no engine can read,
    and what that looks like on the other side is magenta."""
    import bpy

    from city_builder import scene as scene_module

    scene_module.clear_scene()
    image = bpy.data.images.new("packed_webp", width=8, height=8)
    image.pixels = [0.5] * (8 * 8 * 4)
    webp = tmp_path / "page.webp"
    image.file_format = "WEBP"
    image.filepath_raw = str(webp)
    image.save()
    image.pack()

    written = scene_module._textures_an_engine_reads(str(tmp_path / "textures"))
    assert written == 1
    assert image.file_format == "PNG"
    assert image.filepath_raw.endswith(".png")
    assert not image.packed_file, "the WebP was written back over the PNG"


def test_a_png_already_on_disk_is_left_where_it_is(tmp_path):
    import bpy

    from city_builder import scene as scene_module

    scene_module.clear_scene()
    image = bpy.data.images.new("plain_png", width=8, height=8)
    path = tmp_path / "tile.png"
    image.file_format = "PNG"
    image.filepath_raw = str(path)
    image.save()

    before = image.filepath_raw
    assert scene_module._textures_an_engine_reads(str(tmp_path / "textures")) == 0
    assert image.filepath_raw == before
