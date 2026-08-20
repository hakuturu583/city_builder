"""Turning a mesh into Gaussians: does the cloud still describe the surface?

The conversion has no ground truth to check against — a splat cloud is not the
mesh — so what is tested here is the set of properties that make it *the same
surface*: the points are on it, they are spread over it by area rather than by
triangle, each disc lies in the tangent plane, and the colour a sample gets is
the colour the mesh has at that point. Then that the file says all of it back.

No GPU: everything below is numpy. :func:`city_builder.splat.render` is the one
thing here that needs a card, and it is a way of looking at a cloud rather than
a property of one.
"""

from __future__ import annotations

import numpy as np
import pytest

from city_builder import splat as S


def _unit_square(z: float = 0.0):
    """Two triangles making a 1 m x 1 m plate in the XY plane, facing +Z."""
    vertices = np.array([[0.0, 0.0, z], [1.0, 0.0, z], [1.0, 1.0, z], [0.0, 1.0, z]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return vertices, faces


def _lopsided():
    """One big triangle and one tiny one, so by-area and by-triangle differ."""
    vertices = np.array([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0],
                         [20.0, 0.0, 0.0], [20.1, 0.0, 0.0], [20.0, 0.1, 0.0]])
    faces = np.array([[0, 1, 2], [3, 4, 5]])
    return vertices, faces


# ---------------------------------------------------------------------------
# Where the points land
# ---------------------------------------------------------------------------


def test_samples_lie_on_the_surface():
    vertices, faces = _unit_square(z=3.0)
    points, _index, _bary = S.sample_surface(vertices, faces, 2000, seed=0)

    assert len(points) == 2000
    assert np.allclose(points[:, 2], 3.0)
    assert points[:, :2].min() >= -1e-9
    assert points[:, :2].max() <= 1.0 + 1e-9


def test_barycentric_coordinates_reproduce_the_point():
    """The weights come back so colour and UV can be interpolated with them."""
    vertices, faces = _unit_square()
    points, index, bary = S.sample_surface(vertices, faces, 500, seed=1)

    rebuilt = np.einsum("nc,ncd->nd", bary, vertices[faces[index]])
    assert np.allclose(rebuilt, points)
    assert np.allclose(bary.sum(axis=1), 1.0)
    assert bary.min() >= 0.0


def test_a_triangle_gets_samples_in_proportion_to_its_area():
    """Not one each: a mesh of wildly uneven triangles is the normal case."""
    vertices, faces = _lopsided()
    areas = S.triangle_areas(vertices, faces)
    _points, index, _bary = S.sample_surface(vertices, faces, 20000, seed=2)

    got = np.bincount(index, minlength=2) / 20000
    assert np.allclose(got, areas / areas.sum(), atol=0.01)


def test_the_same_seed_gives_the_same_cloud():
    vertices, faces = _unit_square()
    first, _i, _b = S.sample_surface(vertices, faces, 300, seed=7)
    again, _i, _b = S.sample_surface(vertices, faces, 300, seed=7)
    other, _i, _b = S.sample_surface(vertices, faces, 300, seed=8)

    assert np.array_equal(first, again)
    assert not np.array_equal(first, other)


def test_an_empty_mesh_samples_to_nothing_rather_than_raising():
    points, index, bary = S.sample_surface(np.zeros((0, 3)), np.zeros((0, 3), int), 10)
    assert len(points) == len(index) == len(bary) == 0


def test_a_mesh_with_no_area_is_an_error_not_a_divide_by_zero():
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    with pytest.raises(ValueError, match="no area"):
        S.sample_surface(vertices, np.array([[0, 1, 2]]), 10)


# ---------------------------------------------------------------------------
# The disc each point becomes
# ---------------------------------------------------------------------------


def _rotate(quats, vector):
    """Apply (w, x, y, z) quaternions to one vector, the long way round."""
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    v = np.asarray(vector, dtype=float)
    # R = I + 2w[u]x + 2[u]x^2, written out as a matrix per quaternion.
    rot = np.empty((len(quats), 3, 3))
    rot[:, 0, 0] = 1 - 2 * (y * y + z * z)
    rot[:, 0, 1] = 2 * (x * y - w * z)
    rot[:, 0, 2] = 2 * (x * z + w * y)
    rot[:, 1, 0] = 2 * (x * y + w * z)
    rot[:, 1, 1] = 1 - 2 * (x * x + z * z)
    rot[:, 1, 2] = 2 * (y * z - w * x)
    rot[:, 2, 0] = 2 * (x * z - w * y)
    rot[:, 2, 1] = 2 * (y * z + w * x)
    rot[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return rot @ v


@pytest.mark.parametrize("normal", [
    (0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0),
    (0.577, 0.577, 0.577), (-0.6, 0.8, 0.0),
])
def test_the_thin_axis_of_the_disc_is_the_surface_normal(normal):
    """The whole point of the orientation: a wall must not be a blob."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    quats = S.normals_to_quaternions(normal[None])

    assert np.allclose(np.linalg.norm(quats, axis=1), 1.0)
    assert np.allclose(_rotate(quats, [0.0, 0.0, 1.0])[0], normal, atol=1e-6)


def test_a_wall_gets_a_disc_that_lies_in_the_wall():
    """A vertical quad: the flattened axis points out of it, not up it."""
    vertices = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [4.0, 0.0, 3.0], [0.0, 0.0, 3.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=200))

    thin = cloud["scales"].argmin(axis=1)
    assert (thin == 2).all()                       # the third axis is the thin one
    axes = _rotate(cloud["quats"], [0.0, 0.0, 1.0])
    assert np.allclose(np.abs(axes[:, 1]), 1.0, atol=1e-6)   # and it points along Y


def test_the_discs_are_flat_but_not_degenerate():
    vertices, faces = _unit_square()
    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=500))

    tangential = cloud["scales"][:, :2]
    thin = cloud["scales"][:, 2]
    assert (thin > 0).all()
    assert np.allclose(thin / tangential[:, 0], S.SplatOptions().thickness)


def test_spreading_the_same_surface_thinner_grows_each_disc():
    """Radius follows the spacing, so coverage survives a change of budget."""
    vertices, faces = _unit_square()
    dense = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=10_000))
    sparse = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=100))

    assert sparse["radius"] > dense["radius"]
    # A hundredth of the count over the same area is ten times the spacing.
    assert sparse["radius"] / dense["radius"] == pytest.approx(10.0, rel=1e-6)


def _corridor(length: float = 100.0, width: float = 40.0):
    """A flat plate a camera can drive along the middle of."""
    vertices, faces = [], []
    step = 2.0
    nx, ny = int(width / step), int(length / step)
    for j in range(ny + 1):
        for i in range(nx + 1):
            vertices.append((-width / 2 + i * step, -length / 2 + j * step, 0.0))
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            faces += [[a, a + 1, a + nx + 2], [a, a + nx + 2, a + nx + 1]]
    return np.asarray(vertices, dtype=float), np.asarray(faces, dtype=np.int64)


def test_sampling_towards_a_camera_path_puts_the_budget_where_it_is_seen():
    vertices, faces = _corridor()
    path = np.column_stack([np.zeros(20), np.linspace(-45, 45, 20), np.full(20, 1.5)])

    even = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=40_000))
    aimed = S.to_gaussians(vertices, faces,
                           options=S.SplatOptions(count=40_000, viewpoints=path))

    near = lambda cloud: (np.abs(cloud["means"][:, 0]) < 3.0).mean()
    assert near(aimed) > near(even) * 2.0


def test_a_disc_near_the_camera_is_smaller_than_one_far_from_it():
    vertices, faces = _corridor()
    path = np.column_stack([np.zeros(20), np.linspace(-45, 45, 20), np.full(20, 1.5)])
    cloud = S.to_gaussians(vertices, faces,
                           options=S.SplatOptions(count=60_000, viewpoints=path))

    distance = np.abs(cloud["means"][:, 0])
    radius = cloud["scales"][:, 0]
    assert radius[distance > 16.0].mean() > radius[distance < 4.0].mean() * 1.5


@pytest.mark.parametrize("falloff,power", [(2.0, 1.0), (1.0, 0.5)])
def test_the_falloff_decides_how_the_radius_follows_the_distance(falloff, power):
    """`falloff` of 2 gives equal size on screen; 1 spreads the budget wider.

    Radius comes out of the density as ``1/sqrt(density)`` and density goes as
    ``d**-falloff``, so radius goes as ``d**(falloff/2)``. Two is what makes a
    disc cover the same fraction of the frame wherever it is — and is not the
    default, because on a drive it hands the road half the budget.
    """
    vertices, faces = _corridor()
    path = np.column_stack([np.zeros(20), np.linspace(-45, 45, 20), np.full(20, 1.5)])
    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(
        count=60_000, viewpoints=path, falloff=falloff))

    distance = np.maximum(np.abs(cloud["means"][:, 0]), S.SplatOptions().viewpoint_floor)
    # One number per triangle, taken at its centroid, against a sample measured
    # where it actually landed: a couple of metres of triangle is a few per cent.
    ratio = cloud["scales"][:, 0] / distance ** power
    outer = ratio[distance > 6.0]
    assert outer.std() / outer.mean() < 0.10


def test_emphasis_buys_a_mesh_more_of_the_budget_than_its_geometry_would(tmp_path):
    """A wall carries windows and a carriageway carries nothing, and only a
    name can say so — the geometry of the two is equally flat."""
    trimesh = pytest.importorskip("trimesh")

    scene = trimesh.Scene({"Roads": trimesh.creation.box(extents=(4.0, 4.0, 0.1)),
                           "Buildings": trimesh.creation.box(extents=(4.0, 4.0, 0.1))})
    path = str(tmp_path / "two.glb")
    scene.export(path)

    plain = S.from_mesh_file(path, options=S.SplatOptions(count=20_000))
    lifted = S.from_mesh_file(path, options=S.SplatOptions(
        count=20_000, emphasis={"Buildings": 4.0}))

    # The two boxes sit on top of each other, so tell them apart by radius:
    # whichever got more of the budget has the smaller discs.
    assert len(np.unique(np.round(plain["scales"][:, 0], 6))) == 1
    assert lifted["scales"][:, 0].max() / lifted["scales"][:, 0].min() == pytest.approx(
        2.0, rel=0.05)


def test_uniform_sampling_is_unchanged_when_no_viewpoints_are_given():
    vertices, faces = _unit_square()
    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=400))
    assert np.allclose(cloud["scales"][:, 0], cloud["scales"][0, 0])
    assert cloud["radius"] == pytest.approx(np.sqrt(1.0 / (400 * np.pi)))


def test_density_is_per_square_metre_of_surface():
    one = S.to_gaussians(*_unit_square(), options=S.SplatOptions(density=400.0))
    four = S.to_gaussians(np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0],
                                    [2.0, 2.0, 0.0], [0.0, 2.0, 0.0]]),
                          np.array([[0, 1, 2], [0, 2, 3]]),
                          options=S.SplatOptions(density=400.0))

    assert len(one["means"]) == 400
    assert len(four["means"]) == 1600
    assert four["area"] == pytest.approx(4.0)


def test_quads_are_refused_rather_than_silently_misread():
    vertices, _ = _unit_square()
    with pytest.raises(ValueError, match="triangles"):
        S.to_gaussians(vertices, np.array([[0, 1, 2, 3]]))


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def test_colour_is_read_from_the_texture_through_the_uvs():
    """Left half red, right half blue; the split must land at x = 0.5."""
    vertices, faces = _unit_square()
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    image[:, :8] = (255, 0, 0)
    image[:, 8:] = (0, 0, 255)

    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=2000),
                           uv=uv, image=image)
    left = cloud["means"][:, 0] < 0.5
    assert np.allclose(cloud["colours"][left], (1.0, 0.0, 0.0))
    assert np.allclose(cloud["colours"][~left], (0.0, 0.0, 1.0))


def test_the_v_axis_agrees_with_the_library_the_uvs_come_from():
    """v = 1 is the top row, which is what `trimesh.visual.uv_to_color` says.

    Get this upside down and a building comes out with its roof colour on its
    plinth, which is a plausible-looking cloud and a wrong one.
    """
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[0, :] = (255, 255, 255)          # top row white
    got = S.sample_texture(image, np.array([[0.5, 1.0], [0.5, 0.0]]))
    assert np.allclose(got[0], (1.0, 1.0, 1.0))    # v = 1 -> top -> white
    assert np.allclose(got[1], (0.0, 0.0, 0.0))    # v = 0 -> bottom -> black


def test_uvs_past_one_wrap_because_a_facade_sheet_repeats():
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, 0] = (255, 0, 0)
    inside = S.sample_texture(image, np.array([[0.05, 0.5]]))
    wrapped = S.sample_texture(image, np.array([[3.05, 0.5]]))
    assert np.allclose(inside, wrapped)


def test_a_uv_of_exactly_one_is_the_far_edge_not_the_near_one():
    """Half the corner vertices in a glTF carry 1.0; wrapping them is a bug."""
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, -1] = (0, 255, 0)             # rightmost column green
    assert np.allclose(S.sample_texture(image, np.array([[1.0, 0.5]])), (0.0, 1.0, 0.0))
    assert np.allclose(S.sample_texture(image, np.array([[0.0, 0.5]])), (0.0, 0.0, 0.0))


def test_the_mip_chain_halves_and_averages_down_to_one_texel():
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[0, 0] = (255, 255, 255)          # one white texel in a black field
    levels = S.mip_chain(image)

    assert [level.shape[:2] for level in levels] == [(8, 8), (4, 4), (2, 2), (1, 1)]
    assert levels[0][0, 0][0] == pytest.approx(1.0)         # 0-1 from the door
    assert levels[1][0, 0][0] == pytest.approx(1 / 4)       # one of four
    assert levels[-1][0, 0][0] == pytest.approx(1 / 64)     # one of sixty-four


def test_texels_per_metre_is_the_texture_pinned_to_the_surface():
    """A 1 m plate under a 64 px texture is 64 texels to the metre."""
    vertices, faces = _unit_square()
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    got = S.texels_per_metre(vertices, faces, uv, (64, 64))
    assert np.allclose(got, 64.0)

    # Same texture over a plate four times the size: half the texels per metre.
    big = vertices * np.array([2.0, 2.0, 1.0])
    assert np.allclose(S.texels_per_metre(big, faces, uv, (64, 64)), 32.0)


def test_a_fine_high_contrast_texture_averages_instead_of_speckling():
    """A roof tile is near-white on near-black; point sampling makes noise.

    Sampled coarsely, every splat should land near the texture's mean rather
    than at one extreme or the other.
    """
    vertices, faces = _unit_square()
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    checker = np.indices((256, 256)).sum(axis=0) % 2
    image = (checker[:, :, None] * np.ones(3) * 255).astype(np.uint8)

    coarse = S.SplatOptions(count=64)              # radius ~ 7 cm, ~18 texels
    filtered = S.to_gaussians(vertices, faces, options=coarse, uv=uv, image=image)
    point = S.to_gaussians(vertices, faces,
                           options=S.SplatOptions(count=64, filter_texture=False),
                           uv=uv, image=image)

    assert np.allclose(filtered["colours"], 0.5, atol=0.05)
    # Without the filter every sample is one texel, so black or white.
    assert set(np.unique(np.round(point["colours"], 3))) == {0.0, 1.0}


def test_a_texture_coarser_than_the_splats_is_left_alone():
    """The filter must not blur a texture that is already under-sampled."""
    vertices, faces = _unit_square()
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    image = np.zeros((4, 4, 3), dtype=np.uint8)
    image[:, :2] = (255, 0, 0)
    image[:, 2:] = (0, 0, 255)

    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=4000),
                           uv=uv, image=image)
    left = cloud["means"][:, 0] < 0.5
    assert np.allclose(cloud["colours"][left], (1.0, 0.0, 0.0))
    assert np.allclose(cloud["colours"][~left], (0.0, 0.0, 1.0))


def test_vertex_colours_are_used_when_there_is_no_texture():
    vertices, faces = _unit_square()
    colours = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=50),
                           vertex_colours=colours)
    assert np.allclose(cloud["colours"], (1.0, 0.0, 0.0))


def test_a_mesh_with_no_colour_at_all_gets_the_fallback():
    cloud = S.to_gaussians(*_unit_square(), options=S.SplatOptions(count=20))
    assert np.allclose(cloud["colours"], S.SplatOptions().fallback_colour)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def test_the_ply_round_trips_through_its_logs_and_logits(tmp_path):
    vertices, faces = _unit_square()
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    image = np.full((8, 8, 3), 200, dtype=np.uint8)
    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=64),
                           uv=uv, image=image)

    path = S.write_ply(str(tmp_path / "cloud.ply"), cloud)
    back = S.read_ply(path)

    assert np.allclose(back["means"], cloud["means"], atol=1e-5)
    assert np.allclose(back["scales"], cloud["scales"], rtol=1e-4)
    assert np.allclose(back["opacities"], cloud["opacities"], atol=1e-4)
    assert np.allclose(back["colours"], cloud["colours"], atol=1e-4)
    assert np.allclose(np.abs(back["quats"]), np.abs(cloud["quats"]), atol=1e-5)


def test_the_ply_header_is_the_one_the_viewers_key_on(tmp_path):
    cloud = S.to_gaussians(*_unit_square(), options=S.SplatOptions(count=8))
    path = S.write_ply(str(tmp_path / "cloud.ply"), cloud)

    with open(path, "rb") as handle:
        header = handle.read(2048).split(b"end_header")[0].decode("ascii")

    assert "format binary_little_endian 1.0" in header
    assert "element vertex 8" in header
    for name in ("x", "f_dc_0", "opacity", "scale_0", "rot_0", "rot_3"):
        assert f"property float {name}\n" in header
    # Order matters as much as presence: the rows are read positionally.
    assert header.index("property float x") < header.index("property float f_dc_0")
    assert header.index("property float opacity") < header.index("property float scale_0")


def test_a_textured_glb_keeps_its_colours_through_the_round_trip(tmp_path):
    """The path a reconstruction actually takes: GLB in, cloud out.

    Written through trimesh and read back through it, so this pins the UV
    convention against the library rather than against our own arithmetic —
    which is the thing that can drift under us.
    """
    trimesh = pytest.importorskip("trimesh")
    from PIL import Image

    vertices, faces = _unit_square()
    uv = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    picture = np.zeros((16, 16, 3), dtype=np.uint8)
    picture[:8, :] = (0, 255, 0)          # top half green -> v > 0.5
    picture[8:, :] = (255, 0, 255)        # bottom half magenta

    material = trimesh.visual.material.PBRMaterial(
        baseColorTexture=Image.fromarray(picture))
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False,
                           visual=trimesh.visual.TextureVisuals(uv=uv, material=material))
    path = str(tmp_path / "plate.glb")
    mesh.export(path)

    cloud = S.from_mesh_file(path, options=S.SplatOptions(count=None, density=2000.0))

    # The plate's own Y is the texture's V, so the split lands at y = 0.5.
    top = cloud["means"][:, 1] > 0.5
    assert top.any() and (~top).any()
    assert np.allclose(cloud["colours"][top], (0.0, 1.0, 0.0))
    assert np.allclose(cloud["colours"][~top], (1.0, 0.0, 1.0))


def test_a_budget_is_for_the_file_and_is_split_across_its_meshes(tmp_path):
    """`--count` is what a viewer's limit looks like, so it must be honoured.

    Split by area, like the sampling within one mesh: a scene of one big
    building and one small one should not spend half its budget on the small
    one.
    """
    trimesh = pytest.importorskip("trimesh")

    big = trimesh.creation.box(extents=(4.0, 4.0, 4.0))       # 96 m2
    small = trimesh.creation.box(extents=(1.0, 1.0, 1.0))     # 6 m2
    small.apply_translation((10.0, 0.0, 0.0))
    scene = trimesh.Scene({"big": big, "small": small})
    path = str(tmp_path / "two.glb")
    scene.export(path)

    cloud = S.from_mesh_file(path, options=S.SplatOptions(count=10_000))

    assert cloud["meshes"] == 2
    assert len(cloud["means"]) == pytest.approx(10_000, abs=2)
    # The far box is the small one; it should hold 6/102 of the cloud.
    far = cloud["means"][:, 0] > 5.0
    assert far.sum() / len(cloud["means"]) == pytest.approx(6.0 / 102.0, rel=0.05)


def test_every_disc_in_a_scene_is_the_same_size(tmp_path):
    """Two meshes, one budget: a splat must not change size across a seam."""
    trimesh = pytest.importorskip("trimesh")

    scene = trimesh.Scene({"a": trimesh.creation.box(extents=(4.0, 4.0, 4.0)),
                           "b": trimesh.creation.box(extents=(1.0, 1.0, 1.0))})
    path = str(tmp_path / "two.glb")
    scene.export(path)

    cloud = S.from_mesh_file(path, options=S.SplatOptions(count=5000))
    radii = cloud["scales"][:, 0]
    assert radii.max() / radii.min() == pytest.approx(1.0, rel=0.02)


def test_colour_survives_the_spherical_harmonic_it_is_stored_as(tmp_path):
    """0.5 + C0 * f is the only reading of f_dc_* the viewers implement."""
    vertices, faces = _unit_square()
    colours = np.tile([0.25, 0.5, 0.75], (4, 1))
    cloud = S.to_gaussians(vertices, faces, options=S.SplatOptions(count=32),
                           vertex_colours=colours)
    path = S.write_ply(str(tmp_path / "cloud.ply"), cloud)

    with open(path, "rb") as handle:
        handle.read(handle.read(4096).index(b"end_header\n") + len("end_header\n"))
    back = S.read_ply(path)
    assert np.allclose(back["colours"], (0.25, 0.5, 0.75), atol=1e-4)


# ---------------------------------------------------------------------------
# Somewhere for the sky to be


def test_a_dome_sits_at_the_radius_it_was_asked_for():
    from city_builder.splat import sky_dome

    dome = sky_dome((10.0, -5.0, 0.0), radius=400.0, count=2000)
    reach = np.linalg.norm(dome["means"] - np.array([10.0, -5.0, 0.0]), axis=1)
    assert np.allclose(reach, 400.0)


def test_a_dome_stops_at_the_horizon():
    from city_builder.splat import sky_dome

    dome = sky_dome((0.0, 0.0, 1.5), radius=100.0, count=2000)
    # Below the centre is ground; a shell that continued would put sky there.
    assert (dome["means"][:, 2] >= 1.5 - 1e-6).all()


def test_a_dome_faces_the_middle_so_it_is_seen_from_inside():
    from city_builder.splat import sky_dome

    dome = sky_dome((0.0, 0.0, 0.0), radius=50.0, count=500)
    outward = dome["means"] / np.linalg.norm(dome["means"], axis=1, keepdims=True)
    assert np.allclose(dome["normals"], -outward, atol=1e-6)


def test_a_dome_splat_is_wide_enough_to_tile_the_shell():
    from city_builder.splat import sky_dome

    sparse = sky_dome((0.0, 0.0, 0.0), radius=400.0, count=10_000)
    dense = sky_dome((0.0, 0.0, 0.0), radius=400.0, count=250_000)
    # More of them means each covers less; a fixed size would leave gaps at one
    # count and a solid wall of overlap at the other.
    assert sparse["scales"][0, 0] > dense["scales"][0, 0]
    # And the spacing they have to cover is the shell's area over the count.
    spacing = np.sqrt(4 * np.pi * 400.0 ** 2 / 2 / 250_000)
    assert 0.5 * spacing < dense["scales"][0, 0] < 5 * spacing


def test_merging_keeps_every_splat_and_only_the_shared_fields():
    from city_builder.splat import merge, sky_dome

    a = sky_dome((0.0, 0.0, 0.0), radius=100.0, count=30)
    b = dict(sky_dome((0.0, 0.0, 0.0), radius=200.0, count=70))
    b["radius"] = 1.0          # a scalar one of them happens to carry
    joined = merge(a, b)
    assert len(joined["means"]) == 100
    assert "radius" not in joined, "a scalar cannot be concatenated"
    assert set(joined) <= set(a)


def test_keeping_a_subset_keeps_every_field_in_step():
    from city_builder.splat import keep_only, sky_dome

    cloud = sky_dome((0.0, 0.0, 0.0), radius=10.0, count=10)
    keep = np.zeros(10, bool)
    keep[[1, 4, 7]] = True
    out = keep_only(cloud, keep)
    assert len(out["means"]) == 3
    for field in ("quats", "scales", "opacities", "colours", "normals"):
        assert len(out[field]) == 3
    assert np.allclose(out["means"], np.asarray(cloud["means"])[keep])


def test_a_mask_of_the_wrong_length_is_refused_rather_than_broadcast():
    import pytest

    from city_builder.splat import keep_only, sky_dome

    cloud = sky_dome((0.0, 0.0, 0.0), radius=10.0, count=10)
    with pytest.raises(ValueError):
        keep_only(cloud, np.ones(9, bool))


def test_the_summary_describes_the_cloud_that_is_left():
    from city_builder.splat import keep_only

    cloud = {"means": np.zeros((4, 3)),
             "scales": np.array([[1.0, 1, 1], [2, 2, 2], [3, 3, 3], [9, 9, 9]]),
             "radius": 2.5, "radius_range": (1.0, 9.0)}
    out = keep_only(cloud, np.array([True, True, True, False]))
    # A median over splats that are no longer there describes nothing.
    assert out["radius"] == 2.0
    assert out["radius_range"] == (1.0, 3.0)
