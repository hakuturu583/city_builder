"""A mesh is a surface; a splat cloud is that surface, sampled.

Everything upstream of here makes triangles — the ground plate, the road
ribbons, the procedural massing, the reconstructions TRELLIS.2 hands back. A
3D Gaussian Splatting scene is not triangles, and the usual way to get one is to
photograph a place and optimise a cloud until it matches the photographs. That
is not what this does. The geometry is already known and already textured, so
the cloud can be *derived* rather than fitted: put points on the surface, and
give each one a Gaussian flat enough to be a piece of that surface.

**A splat here is a disc, not a blob.** Each sample gets two axes in the
tangent plane and a third one almost flattened away, oriented so the thin axis
is the surface normal. That is the surfel reading of a Gaussian, and it is the
one that matches a mesh: a wall is a wall from both sides, and a round blob
straddling it fogs the room behind. The thin axis is not zero, because a
perfectly flat Gaussian has no volume for the rasteriser to integrate and
disappears at grazing angles.

**How big is decided by how many.** Ask for a density and the discs are sized to
the spacing that density implies — spread them thinner and each one grows to
keep the surface covered. The alternative, a fixed radius, leaves either holes
or a smear the moment the sample count changes, and the count is the one knob
anybody actually turns.

**Colour is read, not guessed.** A sample lands inside a triangle, so its colour
is that triangle's colour at that point: the texture through the UVs where there
is one, the vertex colours interpolated where there are those, and a flat grey
only when the mesh carries neither. Stored as degree-0 spherical harmonics,
which is the constant term — a surface sampled from a mesh has no view-dependent
information to put in the higher bands, and writing zeros there is honest.

The output is the PLY every 3DGS viewer reads. :func:`render` is the check that
it says what the mesh said, and is the only thing here that wants a GPU.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

#: The degree-0 spherical harmonic. 3DGS stores colour as SH coefficients, and
#: this is the number that turns one back into a colour: ``rgb = 0.5 + C0 * f``.
SH_C0 = 0.28209479177387814

#: Property names, in the order the reference implementation writes them. Every
#: viewer worth the name keys on these, so they are not ours to rename.
_PLY_FIELDS = (
    ["x", "y", "z", "nx", "ny", "nz"]
    + [f"f_dc_{i}" for i in range(3)]
    + ["opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
)


@dataclass
class SplatOptions:
    """How densely to sample, and how flat the result is."""

    #: Gaussians per square metre of surface. 400 puts one about every 5 cm,
    #: which is where a facade's window frames survive the conversion.
    density: float = 400.0
    #: An exact budget instead, if the scene has to fit a viewer rather than a
    #: resolution. Overrides ``density`` when set.
    count: int | None = None
    seed: int = 0
    #: Disc radius as a multiple of the spacing the density implies. Below 1.0
    #: the surface shows holes; much above it and the texture goes soft.
    spread: float = 1.0
    #: The thin axis, as a fraction of the disc radius. Small enough that a wall
    #: reads as a wall, large enough that the rasteriser still has something to
    #: integrate at a grazing angle.
    thickness: float = 0.05
    #: Surfaces are opaque. Stored through a logit, so 1.0 is not representable.
    opacity: float = 0.99
    #: Points to size the cloud for — a camera path, usually. One size over a
    #: whole street cannot be right: a drive sees the road four metres in front
    #: of it and the buildings fifty metres away, and a disc that covers the
    #: far wall in a pixel covers the near tarmac in twenty-six. Sampling
    #: density is weighted by inverse square distance to the nearest of these,
    #: which is exactly the weighting that makes every disc the same size *on
    #: screen*. ``None`` samples uniformly by area.
    viewpoints: Any = None
    #: How close a surface is allowed to count as, in metres. Without a floor a
    #: splat on the road under the camera takes an unbounded share of the budget.
    viewpoint_floor: float = 2.0
    #: Average the texture over each splat's footprint instead of reading one
    #: texel. Off is faster and aliases; a roof tile is the texture that shows
    #: it, coming out as salt and pepper rather than as tiles.
    filter_texture: bool = True
    #: What a mesh with neither texture nor vertex colours is painted.
    fallback_colour: tuple[float, float, float] = (0.5, 0.5, 0.5)


# ---------------------------------------------------------------------------
# Sampling the surface
# ---------------------------------------------------------------------------


def triangle_areas(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Area of each triangle, which is what makes the sampling uniform."""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)


def face_weights(vertices: np.ndarray, faces: np.ndarray, viewpoints,
                 floor: float = 2.0) -> np.ndarray:
    """How much of the budget each triangle deserves, given where it is seen from.

    Inverse square distance, and the exponent is not a knob. A disc of radius
    ``r`` at distance ``d`` covers ``r/d`` of the frame, so equal *screen* size
    wants ``r`` proportional to ``d``. Radius comes out of the sampling density
    as ``1/sqrt(density)``, so density has to go as ``1/d^2`` — which is this.

    Measured from the triangle's centroid to the nearest viewpoint, with a floor
    so that the metre of road directly under the camera does not take the lot.
    """
    points = np.asarray(viewpoints, dtype=float).reshape(-1, 3)
    centroids = vertices[faces].mean(axis=1)

    # Chunked: a district is a million triangles and a drive a few hundred
    # cameras, and the full matrix of distances between them is not needed at
    # once.
    nearest = np.empty(len(centroids))
    for start in range(0, len(centroids), 4096):
        block = centroids[start:start + 4096]
        nearest[start:start + 4096] = np.sqrt(
            ((block[:, None, :] - points[None, :, :]) ** 2).sum(-1)).min(axis=1)
    return 1.0 / np.maximum(nearest, floor) ** 2


def sample_surface(vertices: np.ndarray, faces: np.ndarray, count: int,
                   seed: int = 0, weights: np.ndarray | None = None
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(points, face_index, barycentric)`` spread evenly over the surface.

    Even *by area*, not by triangle: a mesh from a marching-cubes reconstruction
    has triangles that differ by orders of magnitude, and one sample each would
    put the whole budget on the small ones. So a triangle is chosen with
    probability proportional to its area, and the point within it uniformly —
    the ``sqrt`` on the first coordinate is what keeps that uniform rather than
    bunched towards one corner.

    The barycentric coordinates come back because everything else a sample needs
    — its colour, its UV — is an interpolation of the corners by exactly these.
    """
    if count <= 0 or len(faces) == 0:
        empty_i = np.zeros(0, dtype=np.int64)
        return np.zeros((0, 3)), empty_i, np.zeros((0, 3))

    areas = triangle_areas(vertices, faces)
    if float(areas.sum()) <= 0.0:
        raise ValueError("the mesh has no area to sample")
    share = areas if weights is None else areas * weights
    total = float(share.sum())
    if total <= 0.0:
        raise ValueError("every triangle has been weighted to nothing")

    rng = np.random.default_rng(seed)
    cumulative = np.cumsum(share)
    picked = np.searchsorted(cumulative, rng.random(count) * total, side="right")
    picked = np.clip(picked, 0, len(faces) - 1)

    r1, r2 = rng.random(count), rng.random(count)
    root = np.sqrt(r1)
    bary = np.column_stack([1.0 - root, root * (1.0 - r2), root * r2])

    corners = vertices[faces[picked]]                      # [N, 3, 3]
    points = np.einsum("nc,ncd->nd", bary, corners)
    return points, picked, bary


def face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Unit normal per triangle; a degenerate one is given +Z rather than NaN."""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    normals = np.cross(b - a, c - a)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    flat = length[:, 0] <= 1e-12
    normals = np.where(length > 1e-12, normals / np.maximum(length, 1e-12), 0.0)
    normals[flat] = (0.0, 0.0, 1.0)
    return normals


def normals_to_quaternions(normals: np.ndarray) -> np.ndarray:
    """The shortest rotation taking +Z onto each normal, as ``(w, x, y, z)``.

    A disc is symmetric about its own axis, so only where the normal points is
    decided here; how the disc is spun within its plane does not matter and is
    left to whatever this formula happens to give. That symmetry is why this can
    be four lines instead of an orthonormal frame and a matrix-to-quaternion
    conversion with its four sign branches.

    The one case the shortest rotation cannot express is a normal pointing at
    -Z, where every rotation is equally short; those get a half turn about X.
    """
    n = np.asarray(normals, dtype=float)
    n = n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)

    # q = (1 + z·n, z × n) with z = (0, 0, 1), before normalising.
    quats = np.column_stack([1.0 + n[:, 2], -n[:, 1], n[:, 0], np.zeros(len(n))])
    opposite = n[:, 2] < -1.0 + 1e-8
    quats[opposite] = (0.0, 1.0, 0.0, 0.0)
    return quats / np.maximum(np.linalg.norm(quats, axis=1, keepdims=True), 1e-12)


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------


def as_unit_image(image: np.ndarray) -> np.ndarray:
    """A texture as floats in 0-1, whatever it arrived as.

    Done once at the door rather than at each read, because a mip level is a
    mean and therefore a float: leave the conversion to the reader and level 0
    gets divided by 255 while every level above it does not, which comes out as
    a texture that is white everywhere.
    """
    array = np.asarray(image)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(float) / 255.0
    return array.astype(float, copy=False)


def _wrap(coordinate: np.ndarray) -> np.ndarray:
    """A UV into ``[0, 1]``, with a whole number landing on the far edge.

    Plain ``mod 1`` is the repeat a GPU does, and it sends ``v = 1`` — which is
    the top edge of an atlas, and the value half the corner vertices in a glTF
    actually carry — round to the bottom row. So an exact integer above zero is
    kept at 1 instead of wrapped to 0: tiling still works, and an edge stays an
    edge.
    """
    fraction = coordinate - np.floor(coordinate)
    return np.where((fraction == 0.0) & (coordinate > 0.0), 1.0, fraction)


def sample_texture(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Nearest-texel lookup, with the V axis and tiling UVs both handled.

    Nearest rather than bilinear on purpose: at these densities a splat is
    smaller than a texel is wide on screen, so interpolating costs a blur and
    buys nothing. ``uv`` is wrapped rather than clamped, because a facade sheet
    is applied by repeating it and its UVs run past 1.

    ``v = 0`` is the *bottom* row. That is trimesh's convention, which is where
    these UVs come from — glTF itself puts the origin at the top left and
    trimesh flips it on the way in, so flipping back here is what agrees with
    ``trimesh.visual.color.uv_to_color`` and therefore with every other reader
    of the same file.
    """
    image = as_unit_image(image)
    height, width = image.shape[:2]
    u, v = _wrap(uv[:, 0]), _wrap(uv[:, 1])
    x = np.clip((u * width).astype(np.int64), 0, width - 1)
    y = np.clip(((1.0 - v) * height).astype(np.int64), 0, height - 1)
    return image[y, x][:, :3]


def mip_chain(image: np.ndarray) -> list[np.ndarray]:
    """The texture, then it halved, then that halved, down to a single texel.

    Box-averaged, which is what a mip level is. Odd rows and columns are dropped
    rather than weighted in: the error is a texel at the far edge of a level
    nothing is sampling closely, and the alternative is a resampling filter to
    carry around.
    """
    levels = [as_unit_image(image)]
    while min(levels[-1].shape[:2]) > 1:
        current = levels[-1]
        height, width = (current.shape[0] // 2) * 2, (current.shape[1] // 2) * 2
        cropped = current[:height, :width]
        levels.append(cropped.reshape(height // 2, 2, width // 2, 2, -1).mean(axis=(1, 3)))
    return levels


def texels_per_metre(vertices: np.ndarray, faces: np.ndarray, uv: np.ndarray,
                     shape: tuple[int, int]) -> np.ndarray:
    """How finely the texture is pinned to the surface, per triangle.

    The ratio of a triangle's area in texels to its area in metres, square
    rooted back into a length. A face with no area either way gets zero, which
    later reads as "the coarsest thing available is fine".
    """
    world = triangle_areas(vertices, faces)
    corners = uv[faces]                                     # [F, 3, 2]
    edge1, edge2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
    in_uv = 0.5 * np.abs(edge1[:, 0] * edge2[:, 1] - edge1[:, 1] * edge2[:, 0])
    in_texels = in_uv * shape[0] * shape[1]
    return np.where(world > 1e-12, np.sqrt(in_texels / np.maximum(world, 1e-12)), 0.0)


def _filtered_texture(image: np.ndarray, uv: np.ndarray, faces: np.ndarray,
                      vertices: np.ndarray, index: np.ndarray, points_uv: np.ndarray,
                      radius: float) -> np.ndarray:
    """Sample the texture at the level where one texel is one splat.

    Point-sampling a texture finer than the splats is aliasing, and on a roof
    tile — which is a near-white tile against a near-black gap — it comes out as
    salt and pepper rather than as a roof. So each sample reads the mip level
    whose texels are about the size of the disc it is colouring, which is the
    same trick and the same reason a GPU has mipmaps.
    """
    levels = mip_chain(image)
    density = texels_per_metre(vertices, faces, uv, image.shape[:2])
    footprint = np.maximum(density[index] * radius, 1.0)
    level = np.clip(np.floor(np.log2(footprint)).astype(int), 0, len(levels) - 1)

    out = np.empty((len(index), 3))
    for step in np.unique(level):
        here = level == step
        out[here] = sample_texture(levels[step], points_uv[here])
    return out


def _colours_for(samples: dict, faces: np.ndarray, options: SplatOptions,
                 vertex_colours: np.ndarray | None,
                 uv: np.ndarray | None, image: np.ndarray | None,
                 vertices: np.ndarray | None = None,
                 radius: float | None = None) -> np.ndarray:
    """A colour per sample, from whichever of the three sources the mesh has."""
    index, bary = samples["face_index"], samples["barycentric"]

    if uv is not None and image is not None:
        corners = uv[faces[index]]                          # [N, 3, 2]
        points_uv = np.einsum("nc,ncd->nd", bary, corners)
        if radius is not None and vertices is not None and options.filter_texture:
            return _filtered_texture(np.asarray(image), uv, faces, vertices,
                                     index, points_uv, radius)
        return sample_texture(image, points_uv)

    if vertex_colours is not None:
        corners = np.asarray(vertex_colours, dtype=float)[faces[index]][:, :, :3]
        return np.einsum("nc,ncd->nd", bary, corners)

    return np.tile(np.asarray(options.fallback_colour, dtype=float), (len(index), 1))


# ---------------------------------------------------------------------------
# Mesh in, Gaussians out
# ---------------------------------------------------------------------------


def to_gaussians(vertices, faces, *, options: SplatOptions | None = None,
                 vertex_colours=None, uv=None, image=None) -> dict[str, Any]:
    """Sample a triangle mesh into 3D Gaussians.

    Returns the arrays a rasteriser wants, in its units rather than the file's:
    ``means`` in scene metres, ``scales`` as standard deviations, ``opacities``
    in 0-1, ``colours`` in 0-1. :func:`write_ply` is what converts those to the
    logs and logits the format stores.

    The disc radius comes from the spacing the sample count implies over the
    area being covered: ``N`` discs sharing ``A`` square metres get ``A / N``
    each, so a radius of ``sqrt(A / (N * pi))`` is the one that tiles the
    surface without overlap. ``spread`` is the multiple of that to actually use.
    """
    options = options or SplatOptions()
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("to_gaussians wants triangles; fan-triangulate first")

    area = float(triangle_areas(vertices, faces).sum())
    count = options.count if options.count is not None else round(area * options.density)
    count = max(count, 0)

    weights = None
    if options.viewpoints is not None:
        weights = face_weights(vertices, faces, options.viewpoints,
                               options.viewpoint_floor)

    points, index, bary = sample_surface(vertices, faces, count, options.seed, weights)
    samples = {"face_index": index, "barycentric": bary}
    normals = face_normals(vertices, faces)[index]

    # Radius follows the density the sampling actually put down, which is the
    # whole point: weight a triangle up and it gets more discs, each smaller,
    # and the surface stays covered. Uniformly that reduces to the global
    # spacing, which is what it was before there were weights.
    if weights is None:
        radius = np.full(len(points), options.spread * np.sqrt(area / (max(count, 1) * np.pi)))
    else:
        share = float((triangle_areas(vertices, faces) * weights).sum())
        per_metre = max(count, 1) * weights[index] / max(share, 1e-12)
        radius = options.spread / np.sqrt(np.maximum(per_metre, 1e-12) * np.pi)

    scales = np.column_stack([radius, radius, radius * options.thickness])

    colours = _colours_for(samples, faces, options, vertex_colours, uv, image,
                           vertices=vertices, radius=float(np.median(radius))
                           if len(radius) else None)
    return {
        "means": points,
        "normals": normals,
        "quats": normals_to_quaternions(normals),
        "scales": scales,
        "opacities": np.full(len(points), options.opacity),
        "colours": np.clip(colours, 0.0, 1.0),
        "area": area,
        "radius": float(np.median(radius)) if len(radius) else 0.0,
        "radius_range": ((float(radius.min()), float(radius.max()))
                         if len(radius) else (0.0, 0.0)),
    }


def from_mesh_file(path: str, *, options: SplatOptions | None = None) -> dict[str, Any]:
    """Sample a GLB, OBJ or anything else trimesh reads.

    A glTF scene is a graph of meshes with their own transforms and their own
    materials, so it is flattened here rather than in the caller: each mesh is
    sampled with its own texture, and the clouds are concatenated. Sampling the
    concatenation instead would need one atlas for the lot, which is a texture
    problem this does not have to solve.
    """
    import trimesh

    options = options or SplatOptions()
    loaded = trimesh.load(path, process=False)
    if isinstance(loaded, trimesh.Scene):
        # Bake each mesh's placement in the scene graph into its vertices, or a
        # district comes out as every building standing on the same spot.
        meshes = []
        for name, geometry in loaded.geometry.items():
            placed = geometry.copy()
            if name in loaded.graph.nodes_geometry:
                placed.apply_transform(loaded.graph.get(name)[0])
            meshes.append(placed)
    else:
        meshes = [loaded]

    # A budget is for the file, not for each mesh in it, so it is split by area
    # — which is the same rule the sampling inside one mesh already follows, and
    # it keeps every disc in the scene the same size.
    # Weighted area, not area, when there are viewpoints: the split has to use
    # the same measure the sampling inside each mesh does, or a mesh that is
    # mostly far away takes a near mesh's share and its discs come out the wrong
    # size on screen at the seam between them.
    shares = []
    for mesh in meshes:
        points = np.asarray(mesh.vertices, dtype=float)
        triangles = np.asarray(mesh.faces, dtype=np.int64)
        area = triangle_areas(points, triangles)
        if options.viewpoints is None:
            shares.append(float(area.sum()))
        else:
            shares.append(float((area * face_weights(
                points, triangles, options.viewpoints, options.viewpoint_floor)).sum()))

    total = sum(shares)
    budgets: list[int | None] = [None] * len(meshes)
    if options.count is not None and total > 0:
        budgets = [round(options.count * share / total) for share in shares]

    clouds = [_sample_one(mesh, options, seed_offset=i, count=budget)
              for i, (mesh, budget) in enumerate(zip(meshes, budgets, strict=True))]
    clouds = [cloud for cloud in clouds if len(cloud["means"])]
    if not clouds:
        raise ValueError(f"nothing sampleable in {path}")

    joined = {key: np.concatenate([cloud[key] for cloud in clouds])
              for key in ("means", "normals", "quats", "scales", "opacities", "colours")}
    joined["area"] = float(sum(cloud["area"] for cloud in clouds))
    joined["radius"] = float(np.median(joined["scales"][:, 0]))
    joined["radius_range"] = (float(joined["scales"][:, 0].min()),
                              float(joined["scales"][:, 0].max()))
    joined["meshes"] = len(clouds)
    return joined


def _sample_one(mesh, options: SplatOptions, *, seed_offset: int,
                count: int | None = None) -> dict[str, Any]:
    """One trimesh, with whatever colour it carries dug out of its visuals."""
    uv = image = vertex_colours = None
    visual = getattr(mesh, "visual", None)

    if visual is not None and getattr(visual, "uv", None) is not None:
        material = getattr(visual, "material", None)
        picture = (getattr(material, "baseColorTexture", None)
                   or getattr(material, "image", None))
        if picture is not None:
            uv = np.asarray(visual.uv, dtype=float)
            image = np.asarray(picture.convert("RGB"))
    if image is None and visual is not None:
        painted = getattr(visual, "vertex_colors", None)
        if painted is not None and len(painted) == len(mesh.vertices):
            vertex_colours = np.asarray(painted, dtype=float) / 255.0

    # Each mesh gets its own stream, or every mesh in a scene samples alike.
    per_mesh = SplatOptions(**{**options.__dict__,
                               "seed": options.seed + seed_offset,
                               "count": count})
    return to_gaussians(mesh.vertices, mesh.faces, options=per_mesh,
                        vertex_colours=vertex_colours, uv=uv, image=image)


# ---------------------------------------------------------------------------
# The file
# ---------------------------------------------------------------------------


def write_ply(path: str, cloud: dict[str, Any]) -> str:
    """The binary PLY the 3DGS reference implementation reads and writes.

    Three of the fields are stored through a function rather than directly, and
    all three are the training parameterisation showing through: scales are
    logged so gradient descent cannot drive one negative, opacity is a logit for
    the same reason, and colour is a spherical harmonic coefficient. Nothing
    here is trained, but a file that stores them any other way is a file the
    viewers read as garbage.
    """
    means = np.asarray(cloud["means"], dtype=np.float32)
    normals = np.asarray(cloud["normals"], dtype=np.float32)
    scales = np.asarray(cloud["scales"], dtype=float)
    opacities = np.asarray(cloud["opacities"], dtype=float)
    colours = np.asarray(cloud["colours"], dtype=float)
    quats = np.asarray(cloud["quats"], dtype=np.float32)

    f_dc = (colours - 0.5) / SH_C0
    log_scales = np.log(np.maximum(scales, 1e-12))
    logit = np.log(np.clip(opacities, 1e-6, 1.0 - 1e-6)
                   / (1.0 - np.clip(opacities, 1e-6, 1.0 - 1e-6)))

    rows = np.column_stack([means, normals, f_dc, logit[:, None],
                            log_scales, quats]).astype(np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(rows)}"]
    header += [f"property float {name}" for name in _PLY_FIELDS]
    header += ["end_header", ""]
    with open(path, "wb") as handle:
        handle.write("\n".join(header).encode("ascii"))
        handle.write(rows.tobytes())
    return path


def read_ply(path: str) -> dict[str, np.ndarray]:
    """Read one back, undoing the log and the logit. Mostly for checking."""
    with open(path, "rb") as handle:
        names: list[str] = []
        count = 0
        while True:
            line = handle.readline().decode("ascii").strip()
            if not line or line == "end_header":
                break
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property float"):
                names.append(line.split()[-1])
        rows = np.frombuffer(handle.read(count * len(names) * 4),
                             dtype="<f4").reshape(count, len(names)).astype(float)

    at = {name: rows[:, i] for i, name in enumerate(names)}
    f_dc = np.column_stack([at[f"f_dc_{i}"] for i in range(3)])
    return {
        "means": np.column_stack([at["x"], at["y"], at["z"]]),
        "normals": np.column_stack([at["nx"], at["ny"], at["nz"]]),
        "colours": 0.5 + SH_C0 * f_dc,
        "opacities": 1.0 / (1.0 + np.exp(-at["opacity"])),
        "scales": np.exp(np.column_stack([at[f"scale_{i}"] for i in range(3)])),
        "quats": np.column_stack([at[f"rot_{i}"] for i in range(4)]),
    }


# ---------------------------------------------------------------------------
# Looking at it
# ---------------------------------------------------------------------------


def render(cloud: dict[str, Any], path: str, *, width: int = 1280, height: int = 720,
           elevation: float = 35.0, azimuth: float = 45.0, up: str = "z",
           distance: float | None = None, fov: float = 50.0,
           background: tuple[float, float, float] = (0.62, 0.66, 0.70)) -> str:
    """Rasterise the cloud with gsplat, so the conversion can be looked at.

    The only GPU in this module, and it is here for a reason that is not
    decoration: a splat cloud that is subtly wrong — normals flipped, discs an
    order of magnitude too big, colour read through the wrong UV axis — is a
    perfectly valid PLY, and the file will not say so. A picture will.

    The camera orbits the cloud's own bounding sphere, so this needs no scene
    and no framing from the caller. ``up`` is which axis that orbit treats as
    vertical: this package's meshes are Z-up, and a glTF is Y-up by
    specification, so a cloud read out of a ``.glb`` wants ``up="y"`` or the
    district comes out standing on its edge.
    """
    import torch
    from gsplat import rasterization
    from PIL import Image

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise RuntimeError("gsplat rasterises on the GPU; there is no card here")

    means = torch.as_tensor(np.asarray(cloud["means"]), dtype=torch.float32, device=device)
    centre = means.mean(dim=0)
    radius = float(torch.linalg.norm(means - centre, dim=1).max())
    distance = distance if distance is not None else radius * 2.6

    try:
        vertical = {"x": 0, "y": 1, "z": 2}[up.lower()]
    except KeyError:
        raise ValueError(f"up must be one of x, y, z; got {up!r}") from None
    skyward = np.eye(3)[vertical]
    plane = [np.eye(3)[i] for i in range(3) if i != vertical]

    lat, lon = np.radians(elevation), np.radians(azimuth)
    offset = (plane[0] * np.cos(lat) * np.cos(lon)
              + plane[1] * np.cos(lat) * np.sin(lon)
              + skyward * np.sin(lat))
    eye = centre.cpu().numpy() + offset * distance

    forward = centre.cpu().numpy() - eye
    forward = forward / np.linalg.norm(forward)
    right = np.cross(forward, skyward)
    right = right / np.linalg.norm(right)
    down = np.cross(forward, right)

    # World-to-camera, in the +Z-forward convention gsplat expects.
    rotation = np.stack([right, down, forward])
    view = np.eye(4)
    view[:3, :3] = rotation
    view[:3, 3] = -rotation @ eye

    focal = 0.5 * height / np.tan(0.5 * np.radians(fov))
    intrinsics = np.array([[focal, 0.0, width / 2.0],
                           [0.0, focal, height / 2.0],
                           [0.0, 0.0, 1.0]])

    def send(key, dtype=torch.float32):
        return torch.as_tensor(np.asarray(cloud[key]), dtype=dtype, device=device)

    image, _alpha, _meta = rasterization(
        means, send("quats"), send("scales"), send("opacities"), send("colours"),
        torch.as_tensor(view[None], dtype=torch.float32, device=device),
        torch.as_tensor(intrinsics[None], dtype=torch.float32, device=device),
        width, height,
        # One colour, not one per camera: gsplat wants this flat for a single
        # un-batched view, and rejects the [1, 3] that reads more naturally.
        backgrounds=torch.tensor(background, dtype=torch.float32, device=device),
    )
    picture = (image[0].clamp(0.0, 1.0).cpu().numpy() * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    Image.fromarray(picture).save(path)
    return path
