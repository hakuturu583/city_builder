"""One building, generated and then put where the map says it is.

The map gives a *plot*: a ring on the ground, an area, a storey count. It says
nothing about what stands there. So the building is generated — a prompt
describing this plot, an image of that building, and a 3D model from the image —
and the plot is what puts the result back at the right size and heading.

A generative model answers the question "what shape is this" and refuses the
question "how big". It hands back a mesh normalised into roughly a unit cube,
Y-up, with no memory of any camera — so the mesh is a shape, and everything
about *this* building's size and heading has to come from the plot.

That is what the footprint is for. :mod:`city_builder.buildings` keeps the ring
each plot was extruded from, and here it is the one measurement the fit is
solved against: find the yaw, the uniform scale and the translation that lay the
mesh's plan outline over that ring, and report how well they agree.

The plot also goes into the *prompt*, which is the cheap half of the same idea.
A model asked for "a building" returns a building of any proportion, and no
amount of scaling makes a tower fit a shop's plot without stretching it. Told
that this one is three storeys on a plan half as deep as it is wide, it comes
back roughly that shape, and the fit is then a small correction rather than a
rescue.

Three deliberate choices.

**Uniform scale, not per-axis.** Squeezing the mesh onto the ring in x and y
independently would fit the footprint exactly and put the windows out of square.
The footprint decides one number; the aspect the model produced is kept.

**Yaw is solved, not assumed.** The views are handed to the model in a known
order, so in principle the heading is known — but "front" is the model's
convention, not ours, and a reconstruction that is right except for being turned
a quarter turn is the easiest possible thing to ship by accident. Solving it
over the ring costs milliseconds and cannot be wrong in that way.

**The hull, not the outline.** Both shapes are compared through their convex
hulls, because a marching-cubes mesh of a building has a ragged skirt where the
ground was and an L-shaped plot would otherwise pull the fit into its notch. The
IoU that comes back afterwards is against the *real* ring, so it still tells you
when the reconstruction is not that building.

Nothing here imports bpy or torch.
"""

from __future__ import annotations

import json
import math
import os
import struct
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# What to ask for
# ---------------------------------------------------------------------------

DEFAULT_STYLE = (
    "a Japanese urban commercial building, precast concrete and tile facade, "
    "aluminium window frames, shopfronts at street level"
)

# What the image has to be for a reconstruction to work from it, as opposed to
# what the building looks like. Kept apart from the style because a caller
# writing a style should not have to remember them, and getting one wrong —
# a second building in shot, a cropped roof — costs a whole generation.
_FRAMING = (
    "single free-standing building, whole building visible including the roof, "
    "three-quarter view from slightly above, plain white background, "
    "no other buildings, no people, no cars, no text, even overcast lighting"
)


def plan_dimensions(footprint: Sequence[Sequence[float]]) -> tuple[float, float]:
    """``(long, short)`` side of a plot, in metres.

    From the minimum rotated rectangle rather than the axis-aligned bounds: a
    plot on a street that does not run north gets a bounding box much larger
    than the building, and the number is going into a prompt as "twice as wide
    as it is deep".
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    plot = ShapelyPolygon([(float(x), float(y)) for x, y in footprint])
    if not plot.is_valid:
        plot = plot.buffer(0)
    corners = list(plot.minimum_rotated_rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        bounds = plot.bounds
        return (max(bounds[2] - bounds[0], bounds[3] - bounds[1]),
                min(bounds[2] - bounds[0], bounds[3] - bounds[1]))
    sides = [math.dist(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    return max(sides[0], sides[1]), min(sides[0], sides[1])


def describe(plot: dict[str, Any], style: str = DEFAULT_STYLE) -> str:
    """The prompt for one plot: what kind of building, and roughly what shape.

    The metres are in there, but the proportions are what actually survive into
    the image — an image model has no scale, and "3 storeys" and "twice as wide
    as it is deep" are constraints it can act on where "35 metres" is not. Both
    are written because the metres cost nothing and the model is a language
    model too.
    """
    long_side, short_side = plan_dimensions(plot["footprint"])
    floors = int(plot.get("floors") or 1)
    height = float(plot.get("height") or 0.0)

    ratio = long_side / max(short_side, 1e-6)
    shape = ("almost square in plan" if ratio < 1.25 else
             f"about {ratio:.1f} times as wide as it is deep")
    storeys = "single-storey" if floors == 1 else f"{floors} storeys"
    return (
        f"photograph of {style}, {storeys} and {height:.0f} metres tall, "
        f"footprint about {long_side:.0f} by {short_side:.0f} metres, {shape}, "
        f"{_FRAMING}"
    )


# ---------------------------------------------------------------------------
# The image
# ---------------------------------------------------------------------------


@dataclass
class ImageOptions:
    """How to draw the building the prompt describes."""

    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    size: int = 1024
    steps: int = 30
    guidance: float = 6.5
    seed: int = 0
    negative: str = ("multiple buildings, street scene, aerial view, cropped, cut off, "
                     "people, cars, text, watermark, blurry, dark")
    vram_budget_gb: float = 0.0  # 0 = no cap; this runs alone


def cut_out(image, *, tolerance: int = 42) -> Any:
    """The backdrop of a plain-background picture, turned into transparency.

    Two things this does not do, both learned the hard way.

    It does not key on *white*. Asked for a plain background, the image model
    returns a mid-grey studio sweep about as often as a white one — measured,
    (154, 153, 159) — and a white key leaves the whole frame opaque, which
    reads downstream as "the building fills the shot" rather than as a failure.
    The backdrop colour is taken from the border instead.

    It does not threshold, it floods. A pale concrete wall is within any
    tolerance that catches the backdrop, so a threshold punches holes through
    the middle of the building. What counts as backdrop is the region *of that
    colour and connected to the border*, which a wall in the middle is not.

    All of this exists so the reconstruction never reaches for a
    background-removal model: TRELLIS.2 skips its own — which is gated — when
    the image it is handed already has an alpha channel.
    """
    import numpy as np
    from scipy import ndimage

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate([rgb[:4].reshape(-1, 3), rgb[-4:].reshape(-1, 3),
                             rgb[:, :4].reshape(-1, 3), rgb[:, -4:].reshape(-1, 3)])
    backdrop = np.median(border, axis=0)

    plain = np.abs(rgb - backdrop).max(axis=2) <= tolerance
    labels, count = ndimage.label(plain)
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    if count:
        edge = set(labels[0, :]) | set(labels[-1, :]) | set(labels[:, 0]) | set(labels[:, -1])
        edge.discard(0)
        if edge:
            alpha = np.where(np.isin(labels, list(edge)), 0, 255).astype(np.uint8)

    from PIL import Image as PILImage

    return PILImage.fromarray(np.dstack([rgb.astype(np.uint8), alpha]), mode="RGBA")


def elevation(prompt: str, path: str, options: ImageOptions | None = None) -> str:
    """One picture of one building, cut out of its background. Written to ``path``.

    Torch is imported inside, so the geometry half of this package stays usable
    on a machine with none of the model stack installed.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    options = options or ImageOptions()
    if options.vram_budget_gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(
            min(1.0, options.vram_budget_gb / total), 0)

    pipeline = AutoPipelineForText2Image.from_pretrained(
        options.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_model_cpu_offload()

    image = pipeline(
        prompt=prompt, negative_prompt=options.negative,
        width=options.size, height=options.size,
        num_inference_steps=options.steps, guidance_scale=options.guidance,
        generator=torch.Generator(device="cpu").manual_seed(options.seed),
    ).images[0]

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cut_out(image).save(path)
    return path


# ---------------------------------------------------------------------------
# The mesh
# ---------------------------------------------------------------------------


@dataclass
class MeshOptions:
    """TRELLIS.2: one image in, a PBR-textured mesh out."""

    weights: str = "microsoft/TRELLIS.2-4B"
    root: str = field(default_factory=lambda: os.environ.get("TRELLIS2_PATH", "/opt/TRELLIS.2"))
    # 512 is three seconds and 1536 is a minute. 1024 is the one worth having
    # for a building: the storey lines survive it and the file is still small
    # enough to put a few hundred of them in a scene. The model's own names:
    # '512', '1024', '1024_cascade', '1536_cascade'.
    pipeline_type: str = "1024"
    seed: int = 0
    texture_size: int = 2048
    decimation_target: int = 200_000
    # flash-attn is the only dependency in the stack that has to be compiled
    # against a matching torch, and torch's own scaled_dot_product_attention
    # runs the same maths. Nothing here is long enough for the difference to
    # show up as anything but seconds.
    attn_backend: str = "sdpa"


_PIPELINE: Any = None


def _pipeline(options: MeshOptions):
    """The loaded model, kept between calls — it is sixteen gigabytes."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    os.environ.setdefault("ATTN_BACKEND", options.attn_backend)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if options.attn_backend == "sdpa":
        _install_varlen_sdpa()
    if options.root not in sys.path:
        if not os.path.isdir(os.path.join(options.root, "trellis2")):
            raise RuntimeError(
                f"no TRELLIS.2 checkout at {options.root}. Clone "
                "https://github.com/microsoft/TRELLIS.2 and point TRELLIS2_PATH at it; "
                "its CUDA extensions (o-voxel, CuMesh, FlexGEMM, nvdiffrast, nvdiffrec) "
                "have to be built as well.")
        sys.path.insert(0, options.root)

    from trellis2.pipelines import Trellis2ImageTo3DPipeline
    from trellis2.pipelines import trellis2_image_to_3d as module

    # The background remover is built eagerly and never used: `preprocess_image`
    # only reaches for it when the image it is given has no alpha, and
    # :func:`elevation` always writes alpha. It is also a *gated* repository
    # whose access is granted by request rather than by accepting terms, so
    # building it would make this pipeline wait on somebody's approval for a
    # model it does not run. Everything downstream already handles None.
    original = module.rembg
    module.rembg = _NoBackgroundRemover()
    try:
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained(options.weights)
    except OSError as error:
        if "gated repo" not in str(error):
            raise
        raise RuntimeError(
            f"TRELLIS.2 wants a gated Hugging Face repository. {error}\n"
            "Accept its terms on an account, then `hf auth login` or set HF_TOKEN. "
            "The conditioning encoder is DINOv3, at "
            "https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m .") from error
    finally:
        module.rembg = original

    _teach_it_where_the_layers_are(pipeline.image_cond_model)
    pipeline.cuda()
    _PIPELINE = pipeline
    return pipeline


class _NoBackgroundRemover:
    """Stands in for the ``rembg`` module so its model is never constructed."""

    def __getattr__(self, _name: str):
        return lambda **_kwargs: None


def _install_varlen_sdpa() -> None:
    """Answer TRELLIS.2's ``import flash_attn`` with torch's own attention.

    Its *sparse* attention accepts three backends and sdpa is not among them —
    ``ATTN_BACKEND=sdpa`` is honoured by the dense path and silently ignored
    here, leaving flash_attn. On this card neither of the real options is
    available: FlashAttention has no wheel for a CUDA 13 torch, and the
    xformers wheel that matches this torch has no CUDA in it and its cutlass
    kernels stop at compute capability 9.0 anyway, against 12.0 on a 5090.

    So rather than patch their dispatch, the three functions it calls are
    provided. Variable-length attention over packed sequences is a loop of
    ordinary attention over the slices — flash-attn's contribution is doing it
    without materialising the block-diagonal mask, not the maths — and the
    sequence count here is the batch size, which is one building.

    Installed in ``sys.modules`` rather than on disk: it is a stand-in for one
    process, and anything that genuinely wants FlashAttention should not find
    this instead.
    """
    import importlib.machinery
    import types

    import torch
    import torch.nn.functional as F

    if "flash_attn" in sys.modules:
        return

    def _attend(q, k, v, cu_q, cu_kv):
        out = torch.empty(q.shape[0], q.shape[1], v.shape[2], dtype=q.dtype, device=q.device)
        for i in range(len(cu_q) - 1):
            qs, qe = int(cu_q[i]), int(cu_q[i + 1])
            ks, ke = int(cu_kv[i]), int(cu_kv[i + 1])
            if qe <= qs:
                continue
            # [L, H, C] -> [1, H, L, C], which is the layout sdpa wants.
            piece = F.scaled_dot_product_attention(
                q[qs:qe].transpose(0, 1).unsqueeze(0),
                k[ks:ke].transpose(0, 1).unsqueeze(0),
                v[ks:ke].transpose(0, 1).unsqueeze(0))
            out[qs:qe] = piece.squeeze(0).transpose(0, 1)
        return out

    module = types.ModuleType("flash_attn")
    module.__doc__ = "city_builder: torch attention behind flash-attn's varlen names"
    # A module put into sys.modules by hand has no spec, and importlib raises
    # rather than shrug when something later asks a real question about it.
    module.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None)

    def flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens, max_seqlen, *_args, **_kwargs):
        q, k, v = qkv.unbind(dim=1)
        return _attend(q, k, v, cu_seqlens, cu_seqlens)

    def flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv,
                                        max_seqlen_q, max_seqlen_kv, *_args, **_kwargs):
        k, v = kv.unbind(dim=1)
        return _attend(q, k, v, cu_seqlens_q, cu_seqlens_kv)

    def flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv,
                               max_seqlen_q, max_seqlen_kv, *_args, **_kwargs):
        return _attend(q, k, v, cu_seqlens_q, cu_seqlens_kv)

    module.flash_attn_varlen_qkvpacked_func = flash_attn_varlen_qkvpacked_func
    module.flash_attn_varlen_kvpacked_func = flash_attn_varlen_kvpacked_func
    module.flash_attn_varlen_func = flash_attn_varlen_func
    sys.modules["flash_attn"] = module


def _teach_it_where_the_layers_are(extractor) -> None:
    """Reach DINOv3's transformer blocks wherever this transformers puts them.

    TRELLIS.2 runs the encoder by hand rather than calling it, so it depends on
    where the blocks live: ``DINOv3ViTModel.layer`` in transformers 4, and
    ``DINOv3ViTModel.model.layer`` in 5, where an encoder object was put in
    between. Pinning transformers back would fix it and break diffusers, so the
    lookup is done here instead — and done by *searching* rather than by
    version, because the next move of that attribute should not need a release.
    """
    import torch
    import torch.nn.functional as F

    model = extractor.model
    blocks = getattr(model, "layer", None)
    if blocks is None:
        for child in model.modules():
            candidate = getattr(child, "layer", None)
            if isinstance(candidate, torch.nn.ModuleList) and len(candidate):
                blocks = candidate
                break
    if blocks is None:
        raise RuntimeError(
            "cannot find DINOv3's transformer blocks in "
            f"{type(model).__name__}; transformers {_transformers_version()} has moved them "
            "again, and TRELLIS.2 runs the encoder block by block rather than calling it")

    def extract_features(image):
        image = image.to(model.embeddings.patch_embeddings.weight.dtype)
        hidden = model.embeddings(image, bool_masked_pos=None)
        position = model.rope_embeddings(image)
        for block in blocks:
            hidden = block(hidden, position_embeddings=position)
        return F.layer_norm(hidden, hidden.shape[-1:])

    extractor.extract_features = extract_features


def _transformers_version() -> str:
    import transformers

    return transformers.__version__


def to_mesh(image_path: str, out_path: str, options: MeshOptions | None = None) -> dict[str, Any]:
    """A textured GLB from one picture. **GPU, tens of seconds.**"""
    import time

    import torch
    from PIL import Image as PILImage

    options = options or MeshOptions()
    pipeline = _pipeline(options)

    started = time.time()
    with torch.no_grad():
        # RGBA, not RGB: the alpha is what keeps the pipeline's own background
        # remover — a gated model this never downloads — out of the run.
        mesh = pipeline.run(PILImage.open(image_path).convert("RGBA"),
                            seed=options.seed, pipeline_type=options.pipeline_type)[0]
    mesh.simplify(16777216)  # the nvdiffrast limit, not a quality choice

    # Between the sampler and the mesher, because they are the two things here
    # that want the whole card. Without it the decimation inside to_glb runs out
    # of memory on a 32 GB card at the 1024 resolution.
    torch.cuda.empty_cache()

    import o_voxel

    glb = o_voxel.postprocess.to_glb(
        vertices=mesh.vertices, faces=mesh.faces, attr_volume=mesh.attrs,
        coords=mesh.coords, attr_layout=mesh.layout, voxel_size=mesh.voxel_size,
        aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
        decimation_target=options.decimation_target, texture_size=options.texture_size,
        remesh=True, remesh_band=1, remesh_project=0, verbose=False)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    glb.export(out_path, extension_webp=True)

    # The pipeline is kept but its working set is not. Reconstructing a street
    # is one call per building in one process, and the voxel volumes are large
    # enough that the second building runs out of memory inside CuMesh without
    # this.
    del mesh, glb
    torch.cuda.empty_cache()
    return {"glb": out_path, "took_seconds": round(time.time() - started, 1),
            "bytes": os.path.getsize(out_path)}


# ---------------------------------------------------------------------------
# glTF, the little of it that carries a shape
# ---------------------------------------------------------------------------

_COMPONENT = {5120: "b", 5121: "B", 5122: "h", 5123: "H", 5125: "I", 5126: "f"}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}


def read_glb(path: str) -> tuple[np.ndarray, np.ndarray]:
    """``(vertices, faces)`` from a binary glTF. Positions and indices only.

    Written out rather than pulled in with a mesh library: the file this reads
    is one this pipeline just wrote, with one mesh and one primitive in it, and
    a dependency that can load every glTF in the world is a lot to carry for
    two accessors.
    """
    with open(path, "rb") as handle:
        magic, _version, _length = struct.unpack("<III", handle.read(12))
        if magic != 0x46546C67:
            raise ValueError(f"{path} is not a GLB")
        json_length, kind = struct.unpack("<II", handle.read(8))
        if kind != 0x4E4F534A:
            raise ValueError("the first GLB chunk is not JSON")
        meta = json.loads(handle.read(json_length).decode("utf-8"))
        buffer = b""
        header = handle.read(8)
        if len(header) == 8:
            bin_length, kind = struct.unpack("<II", header)
            if kind == 0x004E4942:
                buffer = handle.read(bin_length)

    def accessor(index: int) -> np.ndarray:
        acc = meta["accessors"][index]
        view = meta["bufferViews"][acc["bufferView"]]
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        per = _COUNT[acc["type"]]
        dtype = np.dtype("<" + _COMPONENT[acc["componentType"]])
        stride = view.get("byteStride") or per * dtype.itemsize

        if stride == per * dtype.itemsize:  # tightly packed: one read
            return np.frombuffer(buffer, dtype=dtype, count=acc["count"] * per,
                                 offset=start).reshape(acc["count"], per)
        # Interleaved: take the whole span as bytes and slice the columns out.
        # A per-element loop is correct and, on a mesh with half a million
        # vertices, takes longer than everything else in this module together.
        raw = np.frombuffer(buffer, dtype=np.uint8, count=acc["count"] * stride, offset=start)
        raw = raw.reshape(acc["count"], stride)[:, : per * dtype.itemsize]
        return np.ascontiguousarray(raw).view(dtype).reshape(acc["count"], per)

    vertices, faces, base = [], [], 0
    for mesh in meta.get("meshes", []):
        for primitive in mesh.get("primitives", []):
            position = primitive.get("attributes", {}).get("POSITION")
            if position is None:
                continue
            points = accessor(position).astype(float)
            vertices.append(points)
            if "indices" in primitive:
                index = accessor(primitive["indices"]).reshape(-1).astype(np.int64)
                faces.append(index.reshape(-1, 3) + base)
            base += len(points)
    if not vertices:
        raise ValueError(f"{path} has no geometry")
    return (np.concatenate(vertices),
            np.concatenate(faces) if faces else np.zeros((0, 3), dtype=np.int64))


def write_obj(path: str, vertices: np.ndarray, faces: np.ndarray) -> str:
    """The placed mesh, in the one format that needs no library to write.

    Wavefront OBJ because what is being handed on is a mesh in *scene metres* at
    a known anchor, and the point of writing it is that anything can read it.
    """
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("# city_builder: reconstructed building, scene coordinates, metres\n")
        handle.writelines(f"v {x:.4f} {y:.4f} {z:.4f}\n" for x, y, z in vertices)
        handle.writelines(f"f {a + 1} {b + 1} {c + 1}\n" for a, b, c in faces)
    return path


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------


def to_scene_axes(vertices: np.ndarray) -> np.ndarray:
    """glTF's Y-up into this package's Z-up, without mirroring anything.

    glTF is Y-up, right-handed, looking down -Z. The scene is Z-up. Taking
    ``(x, y, z) -> (x, -z, y)`` is the rotation between them; the tempting
    ``(x, z, y)`` is a *reflection*, and a reflected building is one whose
    signage reads backwards and whose fit still scores well.
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    return np.stack([x, -z, y], axis=1)


# ---------------------------------------------------------------------------
# The fit
# ---------------------------------------------------------------------------


def _hull(points_xy: np.ndarray):
    from shapely.geometry import MultiPoint

    return MultiPoint([tuple(p) for p in points_xy]).convex_hull


def _rotate(points_xy: np.ndarray, radians: float) -> np.ndarray:
    cos, sin = math.cos(radians), math.sin(radians)
    return points_xy @ np.array([[cos, sin], [-sin, cos]])


def fit_to_footprint(vertices: np.ndarray, footprint: Sequence[Sequence[float]],
                     base_z: float, *, yaw_steps: int = 360,
                     refine_steps: int = 40) -> dict[str, Any]:
    """Yaw, uniform scale and translation that lay a mesh onto a plot.

    ``vertices`` are in scene axes (Z up) and any units. Returns the transform
    and, more usefully, the overlap it achieved: a reconstruction that came back
    as a different building fits badly, and the number says so.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    plot = ShapelyPolygon([(float(x), float(y)) for x, y in footprint])
    if not plot.is_valid:
        plot = plot.buffer(0)
    if plot.is_empty or plot.area <= 0:
        raise ValueError("this plot has no area to fit to")

    plan = vertices[:, :2] - vertices[:, :2].mean(axis=0)
    target = np.array([plot.centroid.x, plot.centroid.y])

    # The hull once, not once per angle. Rotating a point set and taking its
    # hull gives the same polygon as rotating the hull, and the sweep below asks
    # for it several hundred times over a mesh with half a million vertices.
    outline = np.asarray(_hull(plan).exterior.coords)[:-1]

    def score(radians: float) -> tuple[float, float]:
        from shapely.geometry import Polygon as ShapelyPolygon

        turned = _rotate(outline, radians)
        hull = ShapelyPolygon(turned)
        if hull.area <= 0:
            return 0.0, 1.0
        scale = math.sqrt(plot.area / hull.area)
        placed = ShapelyPolygon(turned * scale)
        placed = _translate(placed, target[0] - placed.centroid.x,
                            target[1] - placed.centroid.y)
        union = placed.union(plot).area
        return (placed.intersection(plot).area / union if union else 0.0), scale

    # A coarse sweep and then a local refinement: the objective has one broad
    # maximum per symmetry of the plan, so a sweep finds the right basin and
    # bisection is enough to land in it.
    coarse = [(score(2 * math.pi * i / yaw_steps)[0], 2 * math.pi * i / yaw_steps)
              for i in range(yaw_steps)]
    best_iou, best_yaw = max(coarse)
    window = 2 * math.pi / yaw_steps
    for _ in range(refine_steps):
        window /= 2.0
        for candidate in (best_yaw - window, best_yaw + window):
            iou, _scale = score(candidate)
            if iou > best_iou:
                best_iou, best_yaw = iou, candidate

    _iou, scale = score(best_yaw)
    placed = place(vertices, yaw=best_yaw, scale=scale,
                   centre=(target[0], target[1]), base_z=base_z)
    height = float(placed[:, 2].max() - placed[:, 2].min())
    return {
        "yaw_deg": round(math.degrees(best_yaw) % 360.0, 3),
        "scale": round(float(scale), 6),
        "centre": [round(float(target[0]), 3), round(float(target[1]), 3)],
        "base_z": round(float(base_z), 3),
        "footprint_iou": round(float(best_iou), 4),
        "height_m": round(height, 3),
    }


def _translate(geometry, dx: float, dy: float):
    from shapely.affinity import translate

    return translate(geometry, dx, dy)


def place(vertices: np.ndarray, *, yaw: float, scale: float,
          centre: tuple[float, float], base_z: float) -> np.ndarray:
    """Apply a fit: rotate about Z, scale uniformly, stand it on ``base_z``."""
    plan = _rotate(vertices[:, :2] - vertices[:, :2].mean(axis=0), yaw) * scale
    z = (vertices[:, 2] - vertices[:, 2].min()) * scale + base_z
    return np.stack([plan[:, 0] + centre[0], plan[:, 1] + centre[1], z], axis=1)


def reconstruct(plot: dict[str, Any], out_dir: str, *, image: str | None = None,
                style: str = DEFAULT_STYLE, image_options: ImageOptions | None = None,
                mesh_options: MeshOptions | None = None,
                name: str = "building") -> dict[str, Any]:
    """One plot to one placed building: a picture, a mesh, a fit.

    ``image`` is the picture to model, and
    :func:`city_builder.portrait.render_portrait` is where it should come from —
    a render of this plot's own massing, dressed in its generated facade. Left
    out, the picture is drawn from the prompt instead, which is measurably worse
    at the thing this is for: over seven promptings the plan aspect came back
    1.00 against a wanted 1.63, and the fitted footprint stalled at an IoU of
    0.68 where the render reaches 0.87.

    Writes ``<name>.glb`` (the model as it came out, in its own unit cube) and
    ``<name>.obj`` (the same mesh in scene metres, standing on the plot). The
    GLB is kept because it carries the PBR textures the OBJ cannot; the OBJ is
    what says where the building is.
    """
    os.makedirs(out_dir, exist_ok=True)
    report: dict[str, Any] = {}
    if image is None:
        report["prompt"] = describe(plot, style)
        image = elevation(report["prompt"], os.path.join(out_dir, f"{name}.png"),
                          image_options)

    made = to_mesh(image, os.path.join(out_dir, f"{name}.glb"), mesh_options)
    fit = fit_glb(made["glb"], plot, out_path=os.path.join(out_dir, f"{name}.obj"))
    return {**report, "image": image, **made, **fit}


def fit_glb(glb_path: str, plot: dict[str, Any], *, out_path: str | None = None,
            yaw_steps: int = 360) -> dict[str, Any]:
    """Read a reconstruction, fit it to the plot it came from, and write it out."""
    vertices, faces = read_glb(glb_path)
    vertices = to_scene_axes(vertices)
    fit = fit_to_footprint(vertices, plot["footprint"], float(plot["base_z"]),
                           yaw_steps=yaw_steps)
    placed = place(vertices, yaw=math.radians(fit["yaw_deg"]), scale=fit["scale"],
                   centre=(fit["centre"][0], fit["centre"][1]), base_z=fit["base_z"])

    report = {
        "source": glb_path,
        "vertices": len(vertices),
        "triangles": len(faces),
        **fit,
        # What the procedural block claimed, for comparison. The reconstruction
        # is not asked to match it — the height was invented — but a mesh twice
        # as tall as the building it was shot from is a fit that went wrong.
        "procedural_height_m": plot.get("height"),
    }
    if out_path:
        report["mesh"] = write_obj(out_path, placed, faces)
    return report
