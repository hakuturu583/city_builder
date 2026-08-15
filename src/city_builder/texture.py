"""Tileable materials for the surfaces that have no authored colour.

The ground is 1.2 km² of mesh. Painting it the way a texturing model paints an
*object* — render a few views, diffuse, project back — gives about half a metre
per texel, which is worse than no texture at all. A large, statistically
uniform surface wants a small **tileable** image repeated at a metric scale
instead, and that is what this module makes.

Two ways to get the tile:

* a diffusion model, with every convolution switched to circular padding so
  the result wraps (the standard trick, and it is checked rather than assumed —
  see :func:`seam_error`);
* a procedural fall-back built from a filtered noise field, which is periodic
  by construction and needs no GPU at all, so the pipeline stays runnable and
  testable on a machine with no model and no card.

Nothing here touches a surface whose class says ``preserve``: the road markings
come from the map and are left exactly as they are.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"


@dataclass
class TextureOptions:
    """How to make a tile, and how big it is on the ground."""

    size: int = 1024  # pixels, square
    tile_metres: float = 12.0  # how far one tile spans in the scene
    steps: int = 24
    guidance: float = 6.0
    seed: int = 0

    model: str = DEFAULT_MODEL
    device: str = "cuda"
    vram_budget_gb: float = 6.0  # this card is shared; do not grow into a neighbour
    diffusion: bool = True  # False = procedural, no GPU


# ---------------------------------------------------------------------------
# Procedural fall-back
# ---------------------------------------------------------------------------


def procedural_tile(size: int, seed: int = 0, *, base=(0.28, 0.26, 0.24), roughness: float = 0.55):
    """A tileable ground texture from a 1/f-filtered noise field.

    Filtering white noise in the frequency domain makes the result periodic by
    construction — the wrap is exact, not approximated — which is what a tile
    has to be.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(size, size))

    fy = np.fft.fftfreq(size)[:, None]
    fx = np.fft.fftfreq(size)[None, :]
    radius = np.sqrt(fx**2 + fy**2)
    radius[0, 0] = 1.0
    spectrum = np.fft.fft2(noise) / radius ** (1.0 + roughness)
    spectrum[0, 0] = 0.0

    field = np.real(np.fft.ifft2(spectrum))
    field = (field - field.mean()) / (field.std() + 1e-9)
    field = np.clip(field * 0.18 + 1.0, 0.55, 1.45)

    rgb = np.stack([field * channel for channel in base], axis=-1)
    return (np.clip(rgb, 0.0, 1.0) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Diffusion
# ---------------------------------------------------------------------------


def _make_seamless(module) -> None:
    """Switch every convolution to circular padding, so the output wraps."""
    import torch

    for layer in module.modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.padding_mode = "circular"


def diffusion_tile(prompt: str, options: TextureOptions, *, negative_prompt: str = ""):
    """Generate one tileable texture with a diffusion model.

    Imports torch lazily: the geometry half of this package must stay usable —
    and its tests fast — on a machine with no GPU stack installed.
    """
    import torch
    from diffusers import AutoPipelineForText2Image

    if options.vram_budget_gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(min(1.0, options.vram_budget_gb / total), 0)

    pipeline = AutoPipelineForText2Image.from_pretrained(
        options.model, torch_dtype=torch.float16, variant="fp16", use_safetensors=True
    )
    pipeline.set_progress_bar_config(disable=True)
    # Keep the peak bounded on a shared card: one module on the GPU at a time.
    pipeline.enable_model_cpu_offload()
    pipeline.enable_vae_tiling()
    pipeline.enable_attention_slicing()

    _make_seamless(pipeline.unet)
    _make_seamless(pipeline.vae)

    generator = torch.Generator(device="cpu").manual_seed(options.seed)
    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt or "people, cars, text, watermark, seam, border, vignette",
        width=options.size,
        height=options.size,
        num_inference_steps=options.steps,
        guidance_scale=options.guidance,
        generator=generator,
    ).images[0]

    del pipeline
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.asarray(image.convert("RGB"))


# ---------------------------------------------------------------------------
# Quality checks
# ---------------------------------------------------------------------------


def seam_error(tile: np.ndarray) -> float:
    """How visible the wrap is, relative to the texture's own variation.

    A tile is only tileable if the step across the wrap looks like the steps
    inside it. Returns the ratio of the two; ~1 is seamless, and a hard seam
    pushes it well above 1.
    """
    image = tile.astype(np.float32)
    wrap = np.abs(image[0] - image[-1]).mean() + np.abs(image[:, 0] - image[:, -1]).mean()
    inner = np.abs(np.diff(image, axis=0)).mean() + np.abs(np.diff(image, axis=1)).mean()
    return float(wrap / (inner + 1e-9))


def save_tile(tile: np.ndarray, path: str) -> str:
    from PIL import Image

    directory = os.path.dirname(os.path.abspath(path))
    if directory:
        os.makedirs(directory, exist_ok=True)
    Image.fromarray(tile).save(path)
    return path


def make_tile(prompt: str, options: TextureOptions | None = None, *, path: str | None = None,
              negative_prompt: str = ""):
    """A tile from the model if one is available, procedurally otherwise."""
    options = options or TextureOptions()
    if options.diffusion:
        tile = diffusion_tile(prompt, options, negative_prompt=negative_prompt)
    else:
        tile = procedural_tile(options.size, options.seed)
    if path:
        save_tile(tile, path)
    return tile


# ---------------------------------------------------------------------------
# Facade sheets
# ---------------------------------------------------------------------------


# One prompt makes one material. Ranking sheets by floor alignment alone picked
# the most literal, least coloured output there was — measured, saturation 0.06,
# a street of identical grey concrete — because the score cannot see colour. The
# character of a facade has to be asked for, and asked for differently each time.
FACADE_STYLES: tuple[tuple[str, str], ...] = (
    ("concrete", ("photograph of a grey concrete office facade, punched square windows, "
                  "weathered precast panels")),
    ("blue glass", ("photograph of a blue-green glass curtain wall office tower, "
                    "reflective glazing, dark mullions")),
    ("tiled", ("photograph of a beige ceramic tiled Japanese mid-rise facade, "
               "narrow aluminium window frames")),
    ("brick", ("photograph of a red brown brick apartment building facade, "
               "pale stone lintels")),
    ("white panel", ("photograph of a white metal panel facade, crisp shadow gaps, "
                     "anodised window frames")),
    ("dark metal", ("photograph of a dark charcoal metal and glass facade, "
                    "bronze tinted glazing")),
    ("sandstone", "photograph of a warm sandstone clad office facade, deep reveals"),
    ("green glass", ("photograph of a pale green tinted glass and steel facade, "
                     "slender vertical fins")),
    # The residential half. Everything above is a commercial mid-rise, which is
    # what a street of 900 m2 plots wants; drop the lot size to house scale and
    # the same set puts an office block on every one of them. These are what
    # the low buildings on a Japanese street are actually faced with.
    ("mortar", ("photograph of a cream mortar rendered Japanese house wall, "
                "aluminium sliding windows, thin steel awning")),
    ("clapboard", ("photograph of a dark stained timber clapboard Japanese house wall, "
                   "vertical battens, small square windows")),
    ("siding", ("photograph of a beige ceramic siding board Japanese suburban house wall, "
                "grooved panels, white trim")),
    ("machiya", ("photograph of a traditional Japanese machiya townhouse front, "
                 "dark timber lattice, plaster panels, sliding paper screens")),
    ("shopfront", ("photograph of a small Japanese shophouse front, glazed shopfront "
                   "below, tiled wall above, vertical signage")),
    ("corrugated", ("photograph of a painted corrugated metal Japanese workshop wall, "
                    "rusted seams, louvred vent")),
)

#: The styles that suit a street of houses rather than of offices.
RESIDENTIAL_STYLES: tuple[str, ...] = (
    "mortar", "clapboard", "siding", "machiya", "shopfront", "corrugated",
)

COMMON_PROMPT = "flat elevation, uniform overcast daylight, no sky, no perspective"


def styles_named(names: Sequence[str]):
    """The named subset of :data:`FACADE_STYLES`, in the order asked for.

    An unknown name is an error rather than a shrug: a typo that silently
    narrows a street to one material is only visible in the render.
    """
    known = dict(FACADE_STYLES)
    missing = [name for name in names if name not in known]
    if missing:
        raise ValueError(f"unknown facade style(s): {', '.join(missing)}; "
                         f"choose from {', '.join(known)}")
    return tuple((name, known[name]) for name in names)


def styled_prompts(count: int, *, seed: int = 0, styles=FACADE_STYLES) -> list[str]:
    """``count`` prompts spread across the styles, deterministically.

    Cycled rather than sampled so a small run still covers the range instead of
    drawing the same character three times out of four.
    """
    import random

    order = list(styles)
    random.Random(seed).shuffle(order)
    return [f"{order[i % len(order)][1]}, {COMMON_PROMPT}" for i in range(count)]


@dataclass
class FacadeOptions:
    """A family of facade textures: alike, but not the same, and on the storeys.

    Three knobs that matter. The prompt fixes what kind of building this is.
    ``variation`` says how far each sheet may drift from the family's shared
    starting point — sampling every sheet from independent noise gives a street
    with no character, sampling them all from one latent gives a street of
    clones. ``controlnet`` is what puts the windows on the floors, and it is not
    optional in practice: without it the model returns a wall with windows
    somewhere, which is what the first attempt at this produced.

    The sheet's size comes from the control image, not from here, because the
    control image is drawn at a fixed number of texels per floor.
    """

    family: str = "sd15"  # sd15 | sdxl
    controlnet: str = "canny"  # canny | mlsd | "" for none
    control_scale: float = 0.9
    lcm: bool = True

    count: int = 4
    # A photograph of a real building, to take the material from. The control
    # image still fixes where the floors and windows are, so the reference is
    # asked for what it is *made of* — which is the half a prompt is worst at.
    #
    # Measured against a refined street frame, floor counts held throughout
    # (alignment 0.92 with no reference, 0.80 at 0.4, 0.82 at 0.7): 0.4 takes
    # the palette and the panel material, and 0.7 begins copying *content* —
    # one sheet came back with the reference's yellow road line painted across
    # the facade. Structure is ControlNet's job either way; what rises with
    # this number is how literally the photograph is quoted.
    reference: bool = False
    reference_strength: float = 0.4
    steps: int = 6  # LCM territory; ~25 without it
    guidance: float = 1.5  # LCM wants this low
    seed: int = 0
    variation: float = 0.45  # 0 = identical siblings, 1 = unrelated strangers

    # What to sample at when the layout's own texel size is smaller than the
    # model can work with. SD1.5 is trained at 512 and SDXL at 1024; below that
    # they return mush whatever the prompt says.
    min_sample_side: int = 512
    max_sample_side: int = 1024

    batch: int = 1
    vram_budget_gb: float = 10.0
    offload: bool = False  # one module on the GPU at a time; only worth it when tight


def _tile_horizontally(module) -> None:
    """Circular padding across the width only.

    A wall wraps around a building, so the texture has to tile left-to-right.
    It must *not* tile top-to-bottom: the bottom of a facade is a shopfront and
    the top is a roofline, and joining them is exactly the artefact to avoid.
    ``padding_mode='circular'`` applies to both axes, so the padding is done by
    hand instead.

    Vertically the padding stays **zeros**, which is what these convolutions
    were trained with. Replicate seems the more sensible choice and is not:
    measured, it put a band of colour noise along the top and bottom edge of
    every sheet, because the decoder has never seen an edge that behaves that
    way. The horizontal axis gets away with circular precisely because a
    tileable image has no edge there to get wrong.
    """
    import torch
    import torch.nn.functional as F

    for layer in module.modules():
        if not isinstance(layer, torch.nn.Conv2d) or layer.padding == (0, 0):
            continue
        pad_h, pad_w = layer.padding if isinstance(layer.padding, tuple) else (layer.padding,) * 2
        layer.padding = (0, 0)

        def forward(self, x, _pad_h=pad_h, _pad_w=pad_w):
            x = F.pad(x, (_pad_w, _pad_w, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, _pad_h, _pad_h), mode="constant", value=0.0)
            return self._conv_forward(x, self.weight, self.bias)

        layer.forward = forward.__get__(layer, torch.nn.Conv2d)


def _as_image(reference):
    """A reference as PIL, from a path, an array or a PIL image."""
    from PIL import Image

    if isinstance(reference, str):
        return Image.open(reference).convert("RGB")
    if isinstance(reference, np.ndarray):
        return Image.fromarray(reference).convert("RGB")
    return reference.convert("RGB")


def _family_latents(shape, count: int, variation: float, seed: int, device, dtype):
    """``count`` latents drawn around one shared starting point.

    Spherical interpolation between the family latent and fresh noise, so every
    sheet keeps the norm the sampler expects while sitting a controlled
    distance from its siblings.
    """
    import torch

    generator = torch.Generator(device="cpu").manual_seed(seed)
    base = torch.randn(shape, generator=generator)
    angle = float(variation) * (torch.pi / 2)

    out = []
    for _ in range(count):
        noise = torch.randn(shape, generator=generator)
        latent = base * torch.cos(torch.tensor(angle)) + noise * torch.sin(torch.tensor(angle))
        out.append(latent.to(device=device, dtype=dtype))
    return out


def load_facade_pipeline(options: FacadeOptions):
    """Build the sampler described by ``options``, weights taken from the cache.

    The repo ids come from :mod:`city_builder.weights` rather than being written
    here, so ``city-builder models`` cannot claim a machine is ready for
    something a run then fails to load. The precision comes from there too: a
    repo cached at full precision must not be asked for ``variant="fp16"``.
    """
    import torch
    from diffusers import (
        AutoPipelineForText2Image,
        ControlNetModel,
        LCMScheduler,
        StableDiffusionControlNetPipeline,
        StableDiffusionXLControlNetPipeline,
    )

    from . import weights as W

    base = W.find(options.family, "base")
    common = {"torch_dtype": torch.float16, "use_safetensors": True}

    if options.controlnet:
        control = W.find(options.family, "controlnet", options.controlnet)
        net = ControlNetModel.from_pretrained(control.repo, variant=W.variant(control), **common)
        pipeline_class = (StableDiffusionXLControlNetPipeline if options.family == "sdxl"
                          else StableDiffusionControlNetPipeline)
        pipeline = pipeline_class.from_pretrained(
            base.repo, controlnet=net, variant=W.variant(base),
            safety_checker=None, requires_safety_checker=False, **common,
        )
    else:
        pipeline = AutoPipelineForText2Image.from_pretrained(
            base.repo, variant=W.variant(base), safety_checker=None, **common
        )

    if options.reference:
        adapter = W.find(options.family, "adapter")
        pipeline.load_ip_adapter(adapter.repo, subfolder="models",
                                 weight_name="ip-adapter_sd15.safetensors")
        pipeline.set_ip_adapter_scale(options.reference_strength)

    if options.lcm:
        lora = W.find(options.family, "lcm-lora")
        pipeline.scheduler = LCMScheduler.from_config(pipeline.scheduler.config)
        pipeline.load_lora_weights(lora.repo)
        pipeline.fuse_lora()

    pipeline.set_progress_bar_config(disable=True)
    if options.offload:
        pipeline.enable_model_cpu_offload()
        pipeline.enable_attention_slicing()
        # Only when memory is tight: tiling splits the decode into windows and
        # blends them, which severs the circular padding across the full width
        # and hands back a sheet that does not wrap.
        pipeline.vae.enable_tiling()
    else:
        pipeline.to("cuda")

    # A wall goes round the building, so the sheet has to meet itself. The
    # ControlNet convolves the line drawing, so it needs the same treatment or
    # the structure stops wrapping even where the pixels do.
    _tile_horizontally(pipeline.unet)
    _tile_horizontally(pipeline.vae)
    if options.controlnet:
        _tile_horizontally(pipeline.controlnet)
    return pipeline


def _sampling_size(wanted: tuple[int, int], options: FacadeOptions) -> tuple[int, int]:
    """The size to *sample* at, which is not the size the sheet is wanted at.

    A layout draws a fixed number of texels per floor, so a two-storey shop
    front comes out 384x344 — and a diffusion model asked for a picture below
    the resolution it was trained at returns mush. Measured over the same
    prompts and the same control images: sheets drawn for two to eight floors
    of a 12 m commercial bay scored 0.74 for floor alignment at 512 wide, while
    one- and two-storey houses on a 7 m bay scored 0.36 at 384x344. Nothing was
    wrong with the prompt.

    So the control image is scaled up to the model's own resolution, sampled
    there, and the result brought back down to the texel size the UV expects.
    Downsampling a sharp sheet is free; sharpening a soft one is not.
    """
    width, height = wanted
    floor = options.min_sample_side
    if min(width, height) >= floor:
        return wanted
    scale = floor / min(width, height)
    # The cap on the long side may pull that back — but never below the size
    # the sheet is wanted at, or raising the resolution would lower it.
    limit = options.max_sample_side / max(width * scale, height * scale)
    scale = max(1.0, scale * min(1.0, limit))
    # Both sides to a multiple of eight, which is what the VAE strides by.
    return (max(8, round(width * scale / 8) * 8), max(8, round(height * scale / 8) * 8))


def facade_sheets(prompt, control=None, options: FacadeOptions | None = None, *,
                  negative_prompt: str = "", pipeline=None, reference=None):
    """A family of facade sheets, conditioned on a control image.

    LCM turns 25 sampling steps into ~6, which is what makes "a sheet per
    building" affordable at all. It does not help the multiview texturing
    models — their UNets are custom and nobody has distilled them — which is
    why this generates 2-D sheets for UVs we control rather than painting the
    meshes. What LCM costs is structure, and ``control`` is what buys it back:
    the line drawing from :mod:`city_builder.facade_layout`, which knows where
    this building's floors are.

    ``prompt`` may be one string or one per sheet. Per sheet is the useful
    case: the control image fixes the architecture, so the prompt is the only
    thing left that decides what the building is *made of*, and a single prompt
    gives a street built entirely of the same material.

    ``reference`` is a photograph — the frames :mod:`city_builder.refine` gets
    back from a video model are the intended source — and it answers the other
    half of the question. The control image says where the floors are, the
    prompt names a material, and the reference shows one. It needs
    ``options.reference`` set, because the adapter costs a gigabyte to load and
    a run that never passes an image should not pay for it.

    Pass ``pipeline`` to reuse a loaded one across several control images —
    loading is most of the wall clock once the sampler is down to six steps.
    """
    import torch
    from PIL import Image

    options = options or FacadeOptions()
    # Explicit padding allocates a copy of every convolution input, so the
    # horizontal-wrap trick costs memory the allocator has to find somewhere.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    if options.vram_budget_gb > 0 and torch.cuda.is_available():
        total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        torch.cuda.set_per_process_memory_fraction(min(1.0, options.vram_budget_gb / total), 0)

    prompts = [prompt] * options.count if isinstance(prompt, str) else list(prompt)
    if len(prompts) != options.count:
        raise ValueError(f"{len(prompts)} prompt(s) for {options.count} sheet(s)")

    # Before the pipeline is built, not after: loading it is ten seconds and
    # three gigabytes, and a reference the adapter was never loaded for would
    # otherwise be dropped in silence at the end of all that.
    reference_image = None
    if reference is not None:
        if not options.reference:
            raise ValueError(
                "a reference image was given but options.reference is off, so the "
                "adapter was never loaded")
        reference_image = _as_image(reference)

    owned = pipeline is None
    pipeline = pipeline or load_facade_pipeline(options)

    if control is None:
        raise ValueError("a facade needs a control image; see facade_layout.control_image")
    wanted = control.shape[1], control.shape[0]  # the layout's own texel size
    width, height = _sampling_size(wanted, options)
    control_image = Image.fromarray(control).convert("RGB")
    if (width, height) != wanted:
        control_image = control_image.resize((width, height), Image.LANCZOS)

    factor = pipeline.vae_scale_factor
    shape = (1, pipeline.unet.config.in_channels, height // factor, width // factor)
    latents = _family_latents(shape, options.count, options.variation, options.seed,
                              pipeline._execution_device, torch.float16)

    sheets = []
    for start in range(0, options.count, options.batch):
        chunk = latents[start:start + options.batch]
        arguments = {
            "prompt": prompts[start:start + len(chunk)],
            "negative_prompt": [negative_prompt or _NEGATIVE] * len(chunk),
            "width": width,
            "height": height,
            "num_inference_steps": options.steps,
            "guidance_scale": options.guidance,
            "latents": torch.cat(chunk, dim=0),
        }
        if options.controlnet:
            arguments["image"] = [control_image] * len(chunk)
            arguments["controlnet_conditioning_scale"] = options.control_scale
        if reference is not None:
            arguments["ip_adapter_image"] = [reference_image] * len(chunk)
        for image in pipeline(**arguments).images:
            if image.size != wanted:
                image = image.resize(wanted, Image.LANCZOS)
            sheets.append(np.asarray(image.convert("RGB")))

    if owned:
        del pipeline
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return sheets


_NEGATIVE = "people, cars, text, watermark, blurry, tilted, perspective, sky, ground"


def seam_error_axis(tile: np.ndarray, axis: int) -> float:
    """:func:`seam_error` for one axis: 0 = the vertical wrap, 1 = horizontal."""
    image = tile.astype(np.float32)
    if axis == 0:
        wrap = np.abs(image[0] - image[-1]).mean()
    else:
        wrap = np.abs(image[:, 0] - image[:, -1]).mean()
    inner = np.abs(np.diff(image, axis=axis)).mean()
    return float(wrap / (inner + 1e-9))
