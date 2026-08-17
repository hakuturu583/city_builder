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
solved against: find the yaw, the plan scale and the translation that lay the
mesh's plan outline over that ring, and report how well they agree.

The plot also goes into the *prompt*, which is the cheap half of the same idea.
A model asked for "a building" returns a building of any proportion, and no
amount of scaling makes a tower fit a shop's plot without stretching it. Told
that this one is three storeys on a plan half as deep as it is wide, it comes
back roughly that shape, and the fit is then a small correction rather than a
rescue.

Three deliberate choices.

**Nearly uniform scale.** Squeezing the mesh onto the ring in x and y
independently would fit the footprint exactly and put every window out of
square, so the footprint essentially decides one number and the aspect the
model produced is kept. Essentially, not exactly: the model has a prior about
the proportions of a building and acts on it — over 200 real plots it returned
a plan 1.4 to 1.5 times as long as it was deep whatever it was shown — so up to
15 % of stretch is allowed along the plot's own long axis, which is about as
much as a facade survives. On those 200 that alone took the models that fitted
well enough to stand from 148 to 172.

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
import time
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


def cut_out(image, *, tolerance: int = 42, protect=None) -> Any:
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

    ``protect`` is a mask of pixels that may not be called backdrop whatever
    their colour. A building re-imagined from a massing render is grey where
    the sky is grey, and without this the flood walks in through the roof and
    hollows it out — measured, the subject came back as 5 % of the frame
    against the 30 % it should be.

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
    if protect is not None:
        plain &= ~np.asarray(protect, dtype=bool)
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

    from .texture import vram_budget

    options = options or ImageOptions()
    with vram_budget(options.vram_budget_gb):
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
# A picture with one building in it and nothing else
# ---------------------------------------------------------------------------

#: What the buildings on a Japanese street are, as whole buildings rather than
#: as walls. :data:`city_builder.texture.FACADE_STYLES` describes a *surface*,
#: because a facade sheet is a surface; an envelope wants a subject.
BUILDING_SUBJECTS: tuple[tuple[str, str], ...] = (
    ("mortar", ("a small Japanese suburban house, cream mortar rendered walls, "
                "grey kawara tiled hipped roof with deep eaves, aluminium sliding "
                "windows, a downpipe, an air-conditioning unit")),
    ("siding", ("a Japanese suburban house, beige ceramic siding boards with white "
                "trim, dark grey tiled roof, a carport canopy, a meter box")),
    ("clapboard", ("a Japanese house with dark stained timber clapboard walls and "
                   "vertical battens, small square windows, a shallow metal roof")),
    ("machiya", ("a traditional Japanese machiya townhouse, dark timber lattice "
                 "front, white plaster panels, black kawara roof tiles, a noren")),
    ("shopfront", ("a small Japanese shophouse, glazed shopfront at street level, "
                   "tiled wall above, a vertical sign, an awning")),
    ("corrugated", ("a Japanese workshop building in painted corrugated metal, "
                    "rusted seams, a louvred vent, a roller shutter")),
    ("concrete", ("a small Japanese concrete apartment block, weathered precast "
                  "panels, punched square windows, exposed stair, roof water tank")),
    ("tiled", ("a Japanese mid-rise building faced in beige ceramic tile, narrow "
               "aluminium window frames, a parapet, rooftop plant")),
    ("brick", ("a red brown brick apartment building with pale stone lintels and "
               "railed balconies")),
    ("dark metal", ("a dark charcoal metal and glass building, bronze tinted "
                    "glazing, a flat parapet roof")),
)

#: The residential subset, for a street of houses rather than of offices.
HOUSE_SUBJECTS: tuple[str, ...] = (
    "mortar", "siding", "clapboard", "machiya", "shopfront", "corrugated",
)

# Everything in here is about the *frame*, not the building. TRELLIS treats
# whatever is in the picture as the subject, so a photograph of a house in its
# street is a photograph of the street: measured, that returned a mesh with the
# sky and the neighbour's garden baked into the walls. Asking for a photograph
# of an isolated house does not work either — SDXL keeps the setting and the
# backdrop covered 8-15 % of the frame. Asking for a *photograph of a model of*
# one took the backdrop to 39-59 %, and it is the phrasing the frame responds
# to: the material and the weathering survive it, and the mesh comes back a
# building rather than a smear.
ISOLATED_FRAME = ("studio product photograph of an architectural model of "
                  "{storeys}{subject}, isolated on a seamless plain background, "
                  "floating, no ground, no sky, soft even lighting, the whole "
                  "building visible including the roof, three-quarter view from "
                  "slightly above")

ISOLATED_NEGATIVE = ("street scene, sky, clouds, grass, garden, trees, fence, road, "
                     "adjacent buildings, neighbours, power lines, people, cars, "
                     "ground, horizon, close-up, cropped, cut off, interior, "
                     "floor plan, text, watermark")

# The envelope sets the height and the picture sets everything else, so a
# bungalow photographed for a three-storey plot comes back as a bungalow nine
# metres tall — one storey of windows stretched over three. The storey count is
# the one thing about the *shape* the picture still has to agree with.
STOREYS: dict[int, str] = {
    1: "a single-storey ",
    2: "a two-storey ",
    3: "a three-storey ",
    4: "a four-storey ",
    5: "a five-storey ",
}


def storeys_said(floors: int | None) -> str:
    """How to say a floor count to an image model, or nothing if it is unknown."""
    if not floors or floors < 1:
        return ""
    return STOREYS.get(int(floors), f"a {int(floors)}-storey ")


def isolated_prompt(subject: str, floors: int | None = None) -> str:
    """One building on nothing, which is the only kind of picture an envelope wants.

    ``floors`` is not decoration: see :data:`STOREYS`. Left out, the subject
    carries its own article and the height is the model's guess.
    """
    said = storeys_said(floors)
    if said:
        # The subjects introduce themselves ("a small Japanese house"), and
        # "a two-storey a small Japanese house" is not a prompt.
        subject = _bare(subject)
    return ISOLATED_FRAME.format(storeys=said, subject=subject)


def _bare(subject: str) -> str:
    """The subject without the article it introduces itself with."""
    head, _, rest = subject.partition(" ")
    return rest if head.lower() in {"a", "an", "the"} and rest else subject


def backdrop_share(image) -> float:
    """How much of the frame :func:`cut_out` was able to call background.

    The one number that predicts whether a picture will reconstruct. Below about
    a quarter it is a street or a close-up crop, and what comes back is a slab
    or a smear; the three pictures measured on the largest Kashiwanoha plot ran
    0.08, 0.45 and 0.59, and only the last two produced a building.
    """
    return float((np.asarray(cut_out(image))[:, :, 3] < 8).mean())


def photographs(subjects: Sequence[str], out_dir: str, *,
                floors: Sequence[int] = (0,),
                options: ImageOptions | None = None, prefix: str = "subject",
                min_backdrop: float = 0.25, attempts: int = 3) -> list[dict[str, Any]]:
    """Isolated building photographs, one per subject and floor count, drawn once.

    :func:`elevation` loads and drops SDXL per call, which is right for one
    building and wrong for a district; this keeps it for the batch. Each picture
    is scored by :func:`backdrop_share` and redrawn on another seed when the
    model has answered with a street instead of a building — the failure that
    otherwise reaches the mesh, where it is much more expensive to notice.

    ``floors`` is a family per storey count, the same shape as the facade
    sheets and for the same reason: the envelope sets the height and the
    picture sets everything else, so a bungalow photographed for a three-storey
    plot comes back as a bungalow nine metres tall.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    options = options or ImageOptions()
    os.makedirs(out_dir, exist_ok=True)
    pipeline = AutoPipelineForText2Image.from_pretrained(
        options.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True)
    pipeline.set_progress_bar_config(disable=True)
    pipeline.enable_model_cpu_offload()

    wanted = [(storeys, subject) for storeys in floors for subject in subjects]
    drawn: list[dict[str, Any]] = []
    try:
        for index, (storeys, subject) in enumerate(wanted):
            best, best_share, tries = None, -1.0, 0
            for attempt in range(max(1, attempts)):
                tries = attempt + 1
                seed = options.seed + index * 1013 + attempt * 7919
                image = pipeline(
                    prompt=isolated_prompt(subject, storeys),
                    # Its own negative is about the *building*; this one is
                    # about the frame, and the frame is the whole difficulty.
                    negative_prompt=", ".join(filter(None, (ISOLATED_NEGATIVE,
                                                            options.negative))),
                    width=options.size, height=options.size,
                    num_inference_steps=options.steps, guidance_scale=options.guidance,
                    generator=torch.Generator(device="cpu").manual_seed(seed),
                ).images[0]
                share = backdrop_share(image)
                if share > best_share:
                    best, best_share = image, share
                if share >= min_backdrop:
                    break
            stamp = f"{storeys}f_" if storeys else ""
            path = os.path.join(out_dir, f"{prefix}_{stamp}{index:02d}.png")
            cut_out(best).save(path)
            drawn.append({"path": path, "subject": subject, "floors": storeys or None,
                          "backdrop": round(best_share, 3), "tries": tries,
                          "isolated": best_share >= min_backdrop})
    finally:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return drawn


VARIED_STYLE = (
    "colour photograph of a Japanese urban building, taken with a long lens on "
    "an overcast afternoon: a walled forecourt and an inner courtyard, a "
    "recessed entrance bay, one wing set back from the street and lower than "
    "the rest, a parapet and a railed roof terrace, an outdoor stair, "
    "air-conditioning units and pipework on the roof, weathered concrete, "
    "ceramic tile, painted steel, real materials with dirt and staining"
)


@dataclass
class RestyleOptions:
    """How far to let the picture leave the massing it started from."""

    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    # The one dial. The massing is the starting latent, so what survives is
    # what has not been denoised away — the silhouette and the plan first,
    # because they are the largest structures in the image.
    strength: float = 0.55
    steps: int = 30
    guidance: float = 7.0
    seed: int = 0
    # The first four are the ones that matter. Handed a flat-shaded render an
    # image model reads it as a *drawing* and returns a better drawing, which
    # is a plausible picture and a useless one — a reconstruction of a line
    # drawing is a shell with no thickness.
    negative: str = ("line drawing, sketch, blueprint, architectural drawing, "
                     "render, cgi, clay model, white outline, "
                     "plain box, featureless slab, blank wall, "
                     "street scene, adjacent buildings, cropped, cut off, people, cars, "
                     "text, watermark")
    # What the massing is composited onto before it goes in. Mid-grey rather
    # than white: the model reads a white field as sky or as overexposure and
    # paints the building lighter to match.
    backdrop: tuple[int, int, int] = (150, 150, 152)


def restyle(image_path: str, path: str, prompt: str = VARIED_STYLE,
            options: RestyleOptions | None = None) -> str:
    """Re-imagine a massing render as a building, keeping where it stands.

    The massing that comes out of :mod:`city_builder.portrait` is a box, and a
    reconstruction of a box is a box — which is faithful and useless, because
    real streets are not made of them. This puts that render in as the starting
    latent of an image model and returns part of the noise, so the plan and the
    height are held by what is already there while the model supplies a
    courtyard, a set-back wing, a parapet, and the surfaces.

    ``strength`` is the whole trade-off and it has two ends that both fail: too
    low and the picture is the box it started from, too high and the building
    stops fitting its plot. The footprint IoU from the fit downstream is what
    measures the second, so sweep it rather than guess.
    """
    import numpy as np
    import torch
    from diffusers import AutoPipelineForImage2Image
    from PIL import Image as PILImage

    options = options or RestyleOptions()

    # The render is RGBA on nothing; an image model wants a picture. The alpha
    # is re-derived afterwards, from this same backdrop.
    source = PILImage.open(image_path).convert("RGBA")
    flat = np.asarray(source, dtype=np.float32)
    alpha = flat[..., 3:4] / 255.0
    composited = flat[..., :3] * alpha + np.array(options.backdrop, dtype=np.float32) * (1 - alpha)
    start = PILImage.fromarray(composited.astype(np.uint8))

    # Kept between calls. Loading it is ten seconds and a street is two hundred
    # buildings, so reloading per building spends half an hour doing nothing.
    # Keyed on the weights, so asking for a different model still gets one.
    global _RESTYLE
    if _RESTYLE is None or _RESTYLE[0] != options.model:
        pipeline = AutoPipelineForImage2Image.from_pretrained(
            options.model, torch_dtype=torch.float16, variant="fp16",
            use_safetensors=True)
        pipeline.set_progress_bar_config(disable=True)
        pipeline.enable_model_cpu_offload()
        _RESTYLE = (options.model, pipeline)
    pipeline = _RESTYLE[1]

    image = pipeline(
        prompt=prompt, negative_prompt=options.negative, image=start,
        strength=options.strength, num_inference_steps=options.steps,
        guidance_scale=options.guidance,
        generator=torch.Generator(device="cpu").manual_seed(options.seed),
    ).images[0]

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Where the massing stood is where the building is, give or take what the
    # model added around it. Dilating that and protecting it from the flood is
    # what stops a grey roof under a grey sky from being keyed out.
    from scipy import ndimage

    protect = ndimage.binary_dilation(alpha[..., 0] > 0.5,
                                      iterations=max(1, source.width // 64))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cut_out(image, protect=protect).save(path)
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

    # What the samplers are allowed to do. TRELLIS.2 ships shape at guidance
    # 7.5 and texture at 1.0 — which is no classifier-free guidance at all on
    # the texture — over twelve steps. Raising a guidance pulls *towards* the
    # conditioning image and lowering it drifts towards the model's own prior,
    # so neither is a "creativity" dial in the way a denoise strength is: this
    # model always generates from noise, and the only thing it is told about
    # the building is the picture. Left at None each keeps the model's default.
    steps: int | None = None
    shape_guidance: float | None = None
    tex_guidance: float | None = None


_PIPELINE: Any = None
_RESTYLE: tuple[str, Any] | None = None


def _prepare_trellis(root: str, attn_backend: str = "sdpa") -> None:
    """The environment TRELLIS.2 needs, and the checkout on the path.

    Shared by both pipelines: the flash-attn substitution and the import path
    are properties of the checkout, not of which of its two models is wanted.
    """
    os.environ.setdefault("ATTN_BACKEND", attn_backend)
    os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    if attn_backend == "sdpa":
        _install_varlen_sdpa()
    if root not in sys.path:
        if not os.path.isdir(os.path.join(root, "trellis2")):
            raise RuntimeError(
                f"no TRELLIS.2 checkout at {root}. Clone "
                "https://github.com/microsoft/TRELLIS.2 and point TRELLIS2_PATH at it; "
                "its CUDA extensions (o-voxel, CuMesh, FlexGEMM, nvdiffrast, nvdiffrec) "
                "have to be built as well.")
        sys.path.insert(0, root)


def _pipeline(options: MeshOptions):
    """The loaded model, kept between calls — it is sixteen gigabytes."""
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    _prepare_trellis(options.root, options.attn_backend)

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

    def tuned(defaults: dict, guidance: float | None) -> dict:
        params = dict(defaults or {})
        if options.steps is not None:
            params["steps"] = options.steps
        if guidance is not None:
            params["guidance_strength"] = guidance
        return params

    started = time.time()
    with torch.no_grad():
        # RGBA, not RGB: the alpha is what keeps the pipeline's own background
        # remover — a gated model this never downloads — out of the run.
        mesh = pipeline.run(
            PILImage.open(image_path).convert("RGBA"),
            seed=options.seed, pipeline_type=options.pipeline_type,
            shape_slat_sampler_params=tuned(pipeline.shape_slat_sampler_params,
                                            options.shape_guidance),
            tex_slat_sampler_params=tuned(pipeline.tex_slat_sampler_params,
                                          options.tex_guidance),
        )[0]
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
# The plot as the envelope, and the picture as the material
# ---------------------------------------------------------------------------

# `run` samples the occupied cells at 32 for the 512 models and 64 for the 1024
# ones, and the flow model that reads them is trained on that grid. Handing the
# 512 model a 64 cube is a mismatch, not a finer envelope — it measured 0.743
# against 0.822 for the same plot, which is what a mismatch looks like.
_ENVELOPE_GRID = {"512": 32, "1024": 64, "1024_cascade": 32, "1536_cascade": 32}

# A roof overhangs. An envelope drawn on the walls has nowhere to put the eaves,
# and the model answers by pulling the whole building in: on the largest plot of
# the Kashiwanoha map, growing the prism by 0.6 m in plan took the footprint IoU
# from 0.822 to 0.882 at the same nine seconds. Worth more than resolution.
EAVE_ROOM = 0.6

# And a roof rises: the same allowance in section, so the building stands above
# its block the way the path that invents its own massing does — 1.25 of the
# block height, measured over 184 of them.
#
# A fraction rather than metres, because a metre of ridge on a shed is a
# different building and on an office block is nothing. 0.4 against a solid
# envelope, where the model reaches 0.81 of what it is given, lands on 1.23-1.26
# of the block at one, two and three storeys alike.
ROOF_ROOM = 0.4

# How many cells the sampler will take. Not a tidy number: 22 272 generated and
# 28 672 did not, on a 32 GB card with the model resident, so the ceiling is
# somewhere between and this is inside it. Above it the run does not degrade,
# it throws — nineteen buildings on this map, three attempts each, twice, every
# one out of memory — so something has to give, and it is better that it be the
# envelope than the building.
VOXEL_BUDGET = 20_000


def envelope_coords(footprint: Sequence[Sequence[float]], height: float, *,
                    grid: int = 32, eave_room: float = EAVE_ROOM,
                    roof_room: float = ROOF_ROOM,
                    budget: int = VOXEL_BUDGET) -> np.ndarray:
    """The plot's own prism, voxelised into the cube TRELLIS samples in.

    TRELLIS.2 generates in three stages, and the first one is only a *choice of
    cells*::

        coords     = sample_sparse_structure(cond, 32 or 64)   # occupied voxels
        shape_slat = sample_shape_slat(cond, model, coords)    # geometry in them
        tex_slat   = sample_tex_slat(cond, model, shape_slat)  # PBR on it

    ``coords`` is an ordinary tensor and ``sample_shape_slat`` takes it as an
    argument, so it can be supplied rather than sampled. Supply the footprint
    extruded to the building's height and the plan and the storey height stop
    being things the model guesses and the fit has to repair; the model is left
    to invent what we actually want from it, inside that envelope.

    The axes are the identity — ``(i, j, k)`` in this package's own order, not
    the Y-up convention :func:`texture_mesh` needs for its *mesh* input. Checked
    by extents: identity puts the height on the height axis and every other
    mapping tried put a plan dimension there.

    ``height`` is the block height from the map, and the prism is drawn taller
    than it by ``roof_room`` — see :data:`ROOF_ROOM` — because a roof stands
    above the walls.

    **The prism is solid, unless it will not fit.** What
    ``sample_sparse_structure`` returns is a *surface* — measured on this map's
    own conditioning photograph, 4905 cells of 32768, each column filling 0.62
    of the levels between its own top and bottom — and a solid prism is
    therefore not the kind of object the shape model is used to. That is a real
    observation and it is not a reason to hollow the envelope out: the whole
    map generated from surface envelopes and came back a district of cages,
    walls you could see daylight through, while the solid prisms had produced
    buildings. The footprint IoU is blind to it, and was slightly *better* on
    the cages, which is the whole reason that run reached 189 buildings before
    anybody looked at it.

    So the envelope stays solid and only the plots that would not otherwise
    generate at all are peeled, one layer at a time, until they are inside
    :data:`VOXEL_BUDGET`. On this map that is nineteen of a hundred and
    eighty-nine.
    """
    from shapely.geometry import Point, Polygon

    ring = Polygon(footprint)
    top = float(height) * (1.0 + roof_room)
    lo = np.array([*ring.bounds[:2], 0.0])
    hi = np.array([*ring.bounds[2:], top])
    span = float((hi - lo).max())
    centre = (lo + hi) / 2.0
    step = span / grid
    grown = ring.buffer(eave_room)

    half = grid / 2.0
    columns = [(i, j) for i in range(grid) for j in range(grid)
               if grown.contains(Point(centre[0] + (i + 0.5 - half) * step,
                                       centre[1] + (j + 0.5 - half) * step))]
    levels = [k for k in range(grid)
              if 0.0 <= centre[2] + (k + 0.5 - half) * step <= top]
    if not columns or not levels:
        raise ValueError(f"the plot is too small to voxelise at {grid}: "
                         f"{len(columns)} columns, {len(levels)} levels")

    solid = {(i, j, k) for i, j in columns for k in levels}
    return np.array(sorted(_afford(solid, budget)), dtype=np.int32)


def _afford(solid: set, budget: int) -> set:
    """``solid``, peeled from the inside until it is within ``budget`` cells.

    Peeled a layer at a time rather than by a distance transform, because
    ``solid`` is at most 32768 cells and the loop runs a handful of times. The
    innermost layer goes first because it is the one the model can least see:
    what it is being told is where the building is, and the middle of a
    building is the least informative part of that.
    """
    if budget <= 0 or len(solid) <= budget:
        return solid
    faces = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    layers, inner = [], solid
    while inner:
        surface = {cell for cell in inner
                   if any((cell[0] + di, cell[1] + dj, cell[2] + dk) not in inner
                          for di, dj, dk in faces)}
        layers.append(surface)
        inner = inner - surface
    kept: set = set()
    for layer in layers:
        if kept and len(kept) + len(layer) > budget:
            break
        kept |= layer
    return kept


def to_mesh_in_envelope(image_path: str, out_path: str, *,
                        footprint: Sequence[Sequence[float]], height: float,
                        options: MeshOptions | None = None,
                        eave_room: float = EAVE_ROOM,
                        roof_room: float = ROOF_ROOM,
                        budget: int = VOXEL_BUDGET) -> dict[str, Any]:
    """A textured GLB whose *plan is the plot's* and whose surface is the picture's.

    The same model and the same call sequence as :func:`to_mesh`, with the one
    stage that guesses the massing replaced by :func:`envelope_coords`. That
    changes what the picture is for. In :func:`to_mesh` it has to carry the shape
    as well as the material, which is why that path photographs the procedural
    massing first and why the result then needs a yaw sweep, an anisotropic
    stretch and a seating pass to be put back on its own plot. Here the shape is
    already settled, so the picture is free to be what a diffusion model is
    actually good at: a photograph of what the building is made of.

    Two things about that picture, both learned by getting them wrong:

    **One whole building, on nothing.** TRELLIS treats the entire frame as the
    subject. Asked for a photograph of a Japanese house, SDXL returns a street —
    sky, garden, fence, the neighbours — and :func:`cut_out` has no plain field
    to key out, so all of it is fed in and the mesh comes back a smear of sky
    and grass. See :func:`city_builder.texture.building_photos`, which is the
    prompt that does not do that.

    **Alpha, and a real one.** As everywhere else here, alpha is what keeps the
    pipeline's own gated background remover out of the run.
    """
    import torch
    from PIL import Image as PILImage

    options = options or MeshOptions()
    if options.pipeline_type not in ("512", "1024"):
        # The cascades sample the structure twice and refine between; there is
        # no single set of coords to substitute.
        raise ValueError("an envelope needs a single-resolution pipeline_type "
                         f"('512' or '1024'), not {options.pipeline_type!r}")
    grid = _ENVELOPE_GRID[options.pipeline_type]
    resolution = int(options.pipeline_type)

    pipeline = _pipeline(options)
    coords = envelope_coords(footprint, height, grid=grid, eave_room=eave_room,
                             roof_room=roof_room, budget=budget)

    def tuned(defaults: dict, guidance: float | None) -> dict:
        params = dict(defaults or {})
        if options.steps is not None:
            params["steps"] = options.steps
        if guidance is not None:
            params["guidance_strength"] = guidance
        return params

    started = time.time()
    torch.manual_seed(options.seed)
    packed = torch.tensor(np.concatenate(
        [np.zeros((len(coords), 1), dtype=np.int32), coords], axis=1),
        dtype=torch.int32, device="cuda")
    with torch.no_grad():
        image = pipeline.preprocess_image(PILImage.open(image_path).convert("RGBA"))
        cond = pipeline.get_cond([image], resolution)
        shape_slat = pipeline.sample_shape_slat(
            cond, pipeline.models[f"shape_slat_flow_model_{resolution}"], packed,
            tuned(pipeline.shape_slat_sampler_params, options.shape_guidance))
        torch.cuda.empty_cache()
        tex_slat = pipeline.sample_tex_slat(
            cond, pipeline.models[f"tex_slat_flow_model_{resolution}"], shape_slat,
            tuned(pipeline.tex_slat_sampler_params, options.tex_guidance))
        torch.cuda.empty_cache()
        mesh = pipeline.decode_latent(shape_slat, tex_slat, resolution)[0]
    mesh.simplify(16777216)
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

    del mesh, glb, shape_slat, tex_slat
    torch.cuda.empty_cache()
    return {"glb": out_path, "took_seconds": round(time.time() - started, 1),
            "bytes": os.path.getsize(out_path), "voxels": len(coords),
            "grid": grid, "eave_room": eave_room, "roof_room": roof_room}


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


def _long_axis(plot) -> float:
    """Which way a plot lies, in radians: the angle of its longest edge.

    From the minimum rotated rectangle, so it is the plot's own heading and not
    an artefact of which way the street it stands on happens to run.
    """
    corners = list(plot.minimum_rotated_rectangle.exterior.coords)[:4]
    if len(corners) < 4:
        return 0.0
    edges = [(corners[i], corners[(i + 1) % 4]) for i in range(4)]
    (x0, y0), (x1, y1) = max(edges, key=lambda e: math.dist(e[0], e[1]))
    return math.atan2(y1 - y0, x1 - x0)


def _rotate(points_xy: np.ndarray, radians: float) -> np.ndarray:
    cos, sin = math.cos(radians), math.sin(radians)
    return points_xy @ np.array([[cos, sin], [-sin, cos]])


DEFAULT_STRETCH = 0.15


def fit_to_footprint(vertices: np.ndarray, footprint: Sequence[Sequence[float]],
                     base_z: float, *, yaw_steps: int = 360,
                     refine_steps: int = 40, stretch: float = DEFAULT_STRETCH,
                     stretch_steps: int = 13) -> dict[str, Any]:
    """Yaw, plan scale and translation that lay a mesh onto a plot.

    ``vertices`` are in scene axes (Z up) and any units. Returns the transform
    and, more usefully, the overlap it achieved: a reconstruction that came back
    as a different building fits badly, and the number says so.

    ``stretch`` is how far the two plan axes may differ, as a fraction. It is
    not zero because the model will not keep a plan aspect it was asked for —
    over 200 real plots it returned a plan 1.4 to 1.5 times as long as it was
    deep whatever it was shown, and plots at 2:1 or worse were dropped for it.
    It is small because the alternative failure is worse: squeezing a mesh onto
    a ring in x and y independently fits the footprint exactly and puts every
    window out of square. Fifteen per cent is about the most that is invisible
    on a facade. The height takes the geometric mean, so a stretched building
    does not also get taller.
    """
    from shapely.geometry import Polygon as ShapelyPolygon

    plot = ShapelyPolygon([(float(x), float(y)) for x, y in footprint])
    if not plot.is_valid:
        plot = plot.buffer(0)
    if plot.is_empty or plot.area <= 0:
        raise ValueError("this plot has no area to fit to")

    plan = vertices[:, :2] - vertices[:, :2].mean(axis=0)
    target = np.array([plot.centroid.x, plot.centroid.y])

    # Which way the plot itself lies. The stretch has to be along *its* axes:
    # scaling a plot-shaped outline along the world's x instead turns a
    # rectangle into a parallelogram, which is a worse thing to do to a facade
    # than the aspect error it was meant to fix.
    phi = _long_axis(plot)

    # The hull once, not once per angle. Rotating a point set and taking its
    # hull gives the same polygon as rotating the hull, and the sweep below asks
    # for it several hundred times over a mesh with half a million vertices.
    outline = np.asarray(_hull(plan).exterior.coords)[:-1]

    def score(radians: float, ratio: float = 1.0) -> tuple[float, np.ndarray]:
        from shapely.geometry import Polygon as ShapelyPolygon

        turned = _rotate(outline, radians - phi)  # into the plot's own frame
        hull = ShapelyPolygon(turned)
        if hull.area <= 0:
            return 0.0, np.array([1.0, 1.0])
        # The area is matched whatever the ratio, so the search below trades
        # one plan axis against the other rather than growing the building.
        axes = math.sqrt(plot.area / hull.area) * np.array([math.sqrt(ratio),
                                                            1 / math.sqrt(ratio)])
        placed = ShapelyPolygon(_rotate(turned * axes, phi))
        placed = _translate(placed, target[0] - placed.centroid.x,
                            target[1] - placed.centroid.y)
        union = placed.union(plot).area
        return (placed.intersection(plot).area / union if union else 0.0), axes

    def refine(iou: float, yaw: float, ratio: float,
               span: float = 1.0) -> tuple[float, float]:
        window = span * 2 * math.pi / yaw_steps
        for _ in range(refine_steps):
            window /= 2.0
            for candidate in (yaw - window, yaw + window):
                got, _axes = score(candidate, ratio)
                if got > iou:
                    iou, yaw = got, candidate
        return iou, yaw

    def best_stretch(iou: float, yaw: float, ratio: float) -> tuple[float, float]:
        if stretch <= 0 or stretch_steps < 2:
            return iou, ratio
        edge = math.log(1.0 + stretch)
        for candidate in np.exp(np.linspace(-edge, edge, stretch_steps)):
            got, _axes = score(yaw, float(candidate))
            if got > iou:
                iou, ratio = got, float(candidate)
        return iou, ratio

    # A coarse sweep and then a local refinement: the objective has one broad
    # maximum per symmetry of the plan, so a sweep finds the right basin and
    # bisection is enough to land in it.
    coarse = [(score(2 * math.pi * i / yaw_steps)[0], 2 * math.pi * i / yaw_steps)
              for i in range(yaw_steps)]
    best_iou, best_yaw = max(coarse)
    best_ratio = 1.0

    # Heading and stretch, alternating. The heading is settled uniformly first,
    # because searching the ratio at the coarse heading buys a stretch that is
    # really paying for a heading a degree out. Then each is refined under the
    # other — and the first refinement after a stretch is given a wide window,
    # since under a uniform scale the optimum heading of a mesh of the wrong
    # proportions sits a degree or two off the one the stretch wants and a
    # bisection that starts inside that gap cannot cross it.
    best_iou, best_yaw = refine(best_iou, best_yaw, 1.0)
    for span in (4.0, 1.0):
        was = best_ratio
        best_iou, best_ratio = best_stretch(best_iou, best_yaw, best_ratio)
        if best_ratio == was:
            break
        best_iou, best_yaw = refine(best_iou, best_yaw, best_ratio, span=span)

    _iou, axes = score(best_yaw, best_ratio)
    placed = place(vertices, yaw=best_yaw, scale=axes, stretch_deg=math.degrees(phi),
                   centre=(target[0], target[1]), base_z=base_z)
    return {
        "yaw_deg": round(math.degrees(best_yaw) % 360.0, 3),
        # Which way ``scale_xy`` points: the long axis of the plot.
        "stretch_deg": round(math.degrees(phi) % 180.0, 3),
        # The geometric mean, which is what the height is scaled by and what a
        # ledger written before the stretch existed meant by "scale".
        "scale": round(float(math.sqrt(axes[0] * axes[1])), 6),
        "scale_xy": [round(float(axes[0]), 6), round(float(axes[1]), 6)],
        "stretch": round(float(max(axes) / min(axes)), 4),
        "centre": [round(float(target[0]), 3), round(float(target[1]), 3)],
        "base_z": round(float(base_z), 3),
        "footprint_iou": round(float(best_iou), 4),
        # Above the ground, which is what there is to compare with a storey
        # count. What is below it is the underside the model invented.
        "height_m": round(float(placed[:, 2].max() - base_z), 3),
        "sunk_m": round(float(base_z - placed[:, 2].min()), 3),
    }


def _translate(geometry, dx: float, dy: float):
    from shapely.affinity import translate

    return translate(geometry, dx, dy)


def seat_z(vertices: np.ndarray, *, coverage: float = 0.30,
           grid: float = 0.02, limit: float = 0.15) -> float:
    """The height in a mesh that belongs on the ground, not its lowest vertex.

    A generative model is shown a building from above and never sees what it
    stands on, so it closes the underside with a taper: the mesh comes to a
    chamfer, a foot, or a rounded cap somewhere under the walls. Standing it on
    its lowest vertex hangs the walls over that point of contact — over a
    rebuilt street of 148 buildings, one in six met the ground with under a
    quarter of its own plan, and that is what reads as a building floating.

    A real building meets the ground with all of its plan, so the ground goes
    where the mesh is first that wide and the taper is buried. It is the same
    thing the procedural pass does with its skirt, arrived at from the other
    side: there the walls are extended down into the hill, here the underside
    is sunk into it.

    ``coverage`` is how much of the plan has to have started before the ground
    goes in. It is low on purpose. A taper is a small tail of columns that
    begin below the rest, and clearing that tail is all this has to do; asking
    for three quarters of the plan instead buries a median of 0.62 m of real
    building to chase the few columns under a recess or an entrance. Swept over
    the same 148: at 0.75 fifteen buildings still met the ground with under half
    their plan and the median burial was 0.62 m; at 0.30, twelve did and the
    median burial was 0.05 m.
    ``grid`` and ``limit`` are fractions of the mesh's own height, so this
    works on a mesh in any units — which the ones out of the pipeline are,
    being normalised into a unit cube. ``limit`` is the most that will ever be
    buried: a building that genuinely narrows towards its base is unusual but
    not wrong, and burying a fifth of it to make it sit flat would be.
    """
    z = vertices[:, 2]
    low, high = float(z.min()), float(z.max())
    span = high - low
    if span <= 0:
        return low

    # Where the mesh *starts*, column by column, rather than how wide it is in
    # a slab. Slabs do not work here: these meshes are decimated, so a flat
    # wall is a few large triangles with no vertices in the middle of it, and a
    # slab thin enough to see a chamfer is usually empty.
    cell = max(span * grid, 1e-9)
    plan = np.floor(vertices[:, :2] / cell).astype(np.int64)
    order = np.lexsort((plan[:, 1], plan[:, 0]))
    keys, starts = np.unique(plan[order], axis=0, return_index=True)
    del keys
    floors = np.minimum.reduceat(z[order], np.sort(starts))

    # Only the columns that begin in the lower half of the building. A roof
    # overhang is a column that begins near the top, and counting it here would
    # ask the ground to rise to the eaves.
    floors = floors[floors < low + span * 0.5]
    if len(floors) < 4:
        return low
    return min(float(np.quantile(floors, coverage)), low + span * limit)


def place(vertices: np.ndarray, *, yaw: float, scale: float | Sequence[float],
          centre: tuple[float, float], base_z: float,
          stretch_deg: float = 0.0) -> np.ndarray:
    """Apply a fit: rotate about Z, scale, stand it on ``base_z``.

    ``scale`` is one number or the two plan axes, and ``stretch_deg`` is which
    way the two point — the long axis of the plot, so that a stretched building
    stays rectangular instead of becoming a parallelogram. It is ignored when
    the scale is one number, a uniform scale being the same in every frame.

    The height always takes the geometric mean of the axes: the stretch is a
    plan correction and a building that got 8 % longer did not get 8 % taller.

    ``base_z`` takes the height the building is *full width* at, so whatever
    the model closed its underside with ends up below the ground rather than
    holding the walls off it. See :func:`seat_z`.
    """
    axes = np.array([scale, scale], dtype=float) if np.isscalar(scale) \
        else np.asarray(scale, dtype=float)
    phi = math.radians(stretch_deg)
    turned = _rotate(vertices[:, :2] - vertices[:, :2].mean(axis=0), yaw - phi)
    plan = _rotate(turned * axes, phi)
    z = (vertices[:, 2] - seat_z(vertices)) * math.sqrt(axes[0] * axes[1]) + base_z
    return np.stack([plan[:, 0] + centre[0], plan[:, 1] + centre[1], z], axis=1)


def reconstruct(plot: dict[str, Any], out_dir: str, *, image: str | None = None,
                style: str = DEFAULT_STYLE, image_options: ImageOptions | None = None,
                mesh_options: MeshOptions | None = None,
                restyle_options: RestyleOptions | None = None,
                restyle_prompt: str = VARIED_STYLE,
                name: str = "building") -> dict[str, Any]:
    """One plot to one placed building: a picture, a mesh, a fit.

    ``image`` is the picture to model, and
    :func:`city_builder.portrait.render_portrait` is where it should come from —
    a render of this plot's own massing, dressed in its generated facade. Left
    out, the picture is drawn from the prompt instead, which is measurably worse
    at the thing this is for: over seven promptings the plan aspect came back
    1.00 against a wanted 1.63, and the fitted footprint stalled at an IoU of
    0.68 where the render reaches 0.87.

    ``restyle_options`` is what makes this a *brush-up* rather than a copy. The
    reconstruction returns the shapes and surfaces it is shown, so a render of
    a procedural massing comes back as a procedural massing: measured over five
    settings of the mesh sampler — texture guidance 1 to 6, shape guidance 3,
    twelve steps to twenty-five — every one of them was indistinguishable from
    the default. Nothing in that model is a photorealism dial. Putting the
    render through an image model first is: same building, same footprint fit
    to within a thousandth, and a roof with ridge tiles and staining on it.

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
    if restyle_options is not None:
        image = restyle(image, os.path.join(out_dir, f"{name}_styled.png"),
                        restyle_prompt, restyle_options)
        report["styled"] = image

    made = to_mesh(image, os.path.join(out_dir, f"{name}.glb"), mesh_options)
    fit = fit_glb(made["glb"], plot, out_path=os.path.join(out_dir, f"{name}.obj"))
    return {**report, "image": image, **made, **fit}


def reconstruct_in_envelope(plot: dict[str, Any], out_dir: str, *, image: str,
                            mesh_options: MeshOptions | None = None,
                            eave_room: float = EAVE_ROOM,
                            roof_room: float = ROOF_ROOM,
                            budget: int = VOXEL_BUDGET,
                            name: str = "building") -> dict[str, Any]:
    """One plot to one placed building, with the plot holding the massing.

    :func:`reconstruct` and this differ in where the shape comes from, and
    everything else follows from that. There the picture carries the shape, so
    it has to be a render of this plot's own massing, brushed up by an image
    model; the mesh then arrives at some size and heading of its own and the fit
    has to recover them, with a stretch to make up the rest. Here
    :func:`envelope_coords` holds the plan and the height, so the picture is
    only asked what the building is made of — the same photograph can serve
    several plots — and the fit is a uniform placement rather than a rescue.

    ``image`` is required, and is not optional in the way it is for
    :func:`reconstruct`: without an envelope a prompt-drawn picture returns a
    building of the model's own proportions, which is why that path renders the
    massing at all. With one, a prompt-drawn picture is the point.
    """
    os.makedirs(out_dir, exist_ok=True)
    made = to_mesh_in_envelope(
        image, os.path.join(out_dir, f"{name}.glb"),
        footprint=plot["footprint"], height=float(plot["height"]),
        options=mesh_options, eave_room=eave_room, roof_room=roof_room,
        budget=budget)
    # No stretch: the plan came from this plot, so a mesh that does not fit it
    # is a mesh that departed from its envelope, and squeezing it would hide
    # exactly the thing the IoU is there to report.
    fit = fit_glb(made["glb"], plot, out_path=os.path.join(out_dir, f"{name}.obj"),
                  stretch=0.0)
    return {"image": image, **made, **fit}


def fit_glb(glb_path: str, plot: dict[str, Any], *, out_path: str | None = None,
            yaw_steps: int = 360, stretch: float | None = None) -> dict[str, Any]:
    """Read a reconstruction, fit it to the plot it came from, and write it out."""
    vertices, faces = read_glb(glb_path)
    vertices = to_scene_axes(vertices)
    fit = fit_to_footprint(vertices, plot["footprint"], float(plot["base_z"]),
                           yaw_steps=yaw_steps,
                           **({} if stretch is None else {"stretch": stretch}))
    placed = place(vertices, yaw=math.radians(fit["yaw_deg"]), scale=fit["scale_xy"],
                   stretch_deg=fit["stretch_deg"],
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


# ---------------------------------------------------------------------------
# The other way round: keep the shape, ask only for the surface
# ---------------------------------------------------------------------------


@dataclass
class TextureMeshOptions:
    """Painting a mesh we already have, rather than asking for a new one."""

    weights: str = "microsoft/TRELLIS.2-4B"
    config_file: str = "texturing_pipeline.json"
    root: str = field(default_factory=lambda: os.environ.get("TRELLIS2_PATH", "/opt/TRELLIS.2"))
    resolution: int = 1024        # 512 or 1024
    texture_size: int = 2048
    seed: int = 0


_TEXTURER: Any = None


def texture_pipeline(options: TextureMeshOptions):
    """TRELLIS.2's texturing pipeline, kept between buildings."""
    global _TEXTURER
    if _TEXTURER is not None:
        return _TEXTURER

    _prepare_trellis(options.root)
    import trellis2.pipelines.trellis2_texturing as module
    from trellis2.pipelines import Trellis2TexturingPipeline

    original = module.rembg
    module.rembg = _NoBackgroundRemover()
    try:
        pipeline = Trellis2TexturingPipeline.from_pretrained(
            options.weights, config_file=options.config_file)
    finally:
        module.rembg = original
    _teach_it_where_the_layers_are(pipeline.image_cond_model)
    pipeline.cuda()
    _TEXTURER = pipeline
    return pipeline


def texture_mesh(vertices: np.ndarray, faces: Sequence[Sequence[int]], image: str,
                 out_path: str, *, options: TextureMeshOptions | None = None) -> dict[str, Any]:
    """Paint *our* mesh from a picture, and hand it back in scene coordinates.

    The other half of TRELLIS.2, and a different division of labour from
    :func:`to_mesh`. That one is shown a picture and invents a shape, which then
    has to be fitted back onto the plot it came from — a fit that succeeds 97 %
    of the time and is a lie the other 3 %. This is shown a *mesh* and invents
    only the surface, so the footprint is exact by construction: no yaw to
    solve, no scale, no stretch, no IoU, nothing to reject.

    Two things have to be right or the result is a smear, and both were found
    the hard way.

    **The mesh goes in Y-up.** ``preprocess_mesh`` applies ``(x, y, z) ->
    (x, -z, y)``, which is a Y-up convention; handed this package's Z-up mesh it
    lays the building on its side, and the picture — which shows it standing —
    then paints the facade onto the roof. Fed a red brick house that produced a
    uniformly red slab.

    **The resolution is not the image-to-3D default.** At ``512``/``1024`` the
    roof came back washed out, a pale field with a faint chequer in it; at
    ``1024``/``2048`` it reads as tiles and the walls as boarding. 7 s against
    26 s, which is still under the 17 s median of asking for a new shape.

    **Unfinished, and the reason this is on a branch.** The transform back to
    scene coordinates is exact — fed the forward normalisation, the inverse
    reproduces the input vertices to 0.0 — but the *file* does not carry it:
    read back with :func:`read_glb`, the mesh lands at the wrong height with a
    footprint IoU of 0.31 against the plot it was made from. The exporter puts
    the placement somewhere the reader does not look, and that has not been run
    down yet. Until it is, use the ``placed`` vertices this returns rather than
    re-reading the GLB.
    """
    import trimesh
    from PIL import Image as PILImage

    options = options or TextureMeshOptions()
    started = time.time()
    verts = np.asarray(vertices, dtype=float)

    # Fan-triangulate: the pipeline wants triangles, and this package's walls
    # and roofs are quads and n-gons.
    triangles = [[face[0], face[k], face[k + 1]]
                 for face in faces for k in range(1, len(face) - 1)]
    up = np.column_stack([verts[:, 0], verts[:, 2], -verts[:, 1]])
    mesh = trimesh.Trimesh(vertices=up, faces=np.asarray(triangles), process=False)

    # The normalisation `preprocess_mesh` will apply, so it can be undone.
    low, high = up.min(axis=0), up.max(axis=0)
    centre = (low + high) / 2.0
    scale = 0.99999 / (high - low).max()

    import torch

    pipeline = texture_pipeline(options)
    with torch.no_grad():
        out = pipeline.run(mesh, PILImage.open(image).convert("RGBA"),
                           seed=options.seed, resolution=options.resolution,
                           texture_size=options.texture_size)

    # Back out of the unit cube, then back out of Y-up.
    got = np.asarray(out.vertices, dtype=float)
    got = np.column_stack([got[:, 0], got[:, 2], -got[:, 1]])   # undo preprocess
    got = got / scale + centre
    out.vertices = np.column_stack([got[:, 0], -got[:, 2], got[:, 1]])  # undo Y-up

    placed = np.asarray(out.vertices, dtype=float)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
    out.export(out_path, extension_webp=True)
    return {"glb": out_path, "took_seconds": round(time.time() - started, 1),
            "vertices": len(out.vertices), "triangles": len(out.faces),
            "bytes": os.path.getsize(out_path),
            # The scene-space vertices, because the GLB does not yet carry them
            # where this package's reader looks — see the caveat above.
            "placed": placed}


