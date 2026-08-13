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


def make_tile(prompt: str, options: TextureOptions | None = None, *, path: str | None = None):
    """A tile from the model if one is available, procedurally otherwise."""
    options = options or TextureOptions()
    if options.diffusion:
        tile = diffusion_tile(prompt, options)
    else:
        tile = procedural_tile(options.size, options.seed)
    if path:
        save_tile(tile, path)
    return tile
