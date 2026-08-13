"""The model weights the texturing path needs, and whether they are here yet.

Fetching a few gigabytes is not something to discover halfway through a run on
a shared machine, so the set is declared rather than implied by whatever
``from_pretrained`` happens to ask for. Nothing in this module imports torch or
touches a GPU: it reads the Hugging Face cache and, when asked, downloads.

Two stacks, because the choice between them is not settled:

``sd15``
    Stable Diffusion 1.5 with an LCM LoRA and a structural ControlNet. Small,
    and the combination that is actually well-trodden — Pro-DG conditions
    facades this way, on a canny ControlNet fine-tuned on 23k facade images.
    Lower resolution, but a facade sheet is 12 m of wall, not a poster.

``sdxl``
    Higher resolution, and already on this machine. The doubt is the
    combination rather than the parts: LCM-LoRA and ControlNet on SDXL are
    both fine alone and reportedly weaken each other's control together.

Which one wins is a measurement — :func:`city_builder.facade_layout.floor_alignment`
scores it — so both are declared and the comparison is affordable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# fp16 safetensors only: the repos below all ship a .bin, a full-precision
# .safetensors and a single-file .ckpt of the same weights, and pulling the lot
# costs several times what is needed to run them.
DIFFUSERS_FP16 = ("*.json", "*.txt", "*.fp16.safetensors")


@dataclass(frozen=True)
class Weight:
    """One repository, and the files whose presence means we have it.

    ``probe`` is ordered by preference — half precision first — and any one of
    them counts. A repo that somebody already pulled at full precision is
    perfectly usable; it just has to be loaded without ``variant="fp16"``, which
    is why :func:`variant` exists and the report prints which one is here.
    """

    repo: str
    role: str  # base | lcm-lora | controlnet | adapter
    family: str  # sd15 | sdxl
    probe: tuple[str, ...]
    patterns: tuple[str, ...] = DIFFUSERS_FP16
    note: str = ""

    @property
    def key(self) -> str:
        return f"{self.family}:{self.role}:{self.repo.rsplit('/', 1)[-1]}"


UNET = ("unet/diffusion_pytorch_model.fp16.safetensors",
        "unet/diffusion_pytorch_model.safetensors")
CONTROLNET = ("diffusion_pytorch_model.fp16.safetensors",
              "diffusion_pytorch_model.safetensors")
LORA = ("pytorch_lora_weights.safetensors",)

STACKS: dict[str, tuple[Weight, ...]] = {
    "sd15": (
        Weight(
            "stable-diffusion-v1-5/stable-diffusion-v1-5", "base", "sd15", UNET,
            note="the runwayml repo was withdrawn; this is the maintained reupload",
        ),
        Weight(
            "latent-consistency/lcm-lora-sdv1-5", "lcm-lora", "sd15", LORA,
            patterns=("*.json", "*.safetensors"),  # no fp16 variant, and it is 135 MB
            note="25 sampling steps down to about 4",
        ),
        Weight(
            "lllyasviel/control_v11p_sd15_canny", "controlnet", "sd15", CONTROLNET,
            note="edges: takes the layout's line drawing directly",
        ),
        Weight(
            "lllyasviel/control_v11p_sd15_mlsd", "controlnet", "sd15", CONTROLNET,
            note="straight line segments: trained for architecture, worth comparing",
        ),
    ),
    "sdxl": (
        Weight("stabilityai/stable-diffusion-xl-base-1.0", "base", "sdxl", UNET),
        Weight("latent-consistency/lcm-lora-sdxl", "lcm-lora", "sdxl", LORA,
               patterns=("*.json", "*.safetensors")),
        Weight("diffusers/controlnet-canny-sdxl-1.0", "controlnet", "sdxl", CONTROLNET),
    ),
}


def stack(family: str | None = None) -> list[Weight]:
    """The declared weights, for one family or all of them."""
    if family in (None, "all"):
        return [weight for weights in STACKS.values() for weight in weights]
    if family not in STACKS:
        raise KeyError(f"unknown family {family!r}; have {', '.join(STACKS)}")
    return list(STACKS[family])


def cache_root() -> str:
    """Where the weights land. ``HF_HOME`` decides, and here it is not the default."""
    home = os.environ.get("HF_HOME")
    if home:
        return os.path.join(home, "hub")
    return os.path.expanduser(
        os.environ.get("HUGGINGFACE_HUB_CACHE", "~/.cache/huggingface/hub")
    )


def _found(weight: Weight) -> tuple[str, str] | None:
    """``(probe, path)`` for the first probe that is in the cache."""
    from huggingface_hub import try_to_load_from_cache

    for probe in weight.probe:
        path = try_to_load_from_cache(weight.repo, probe)
        if isinstance(path, str):
            return probe, path
    return None


def present(weight: Weight) -> str | None:
    """The cached path of this weight, or None if it is not here.

    Offline: asks the cache, never the network, so the report is safe to run on
    a machine with no connection and costs nothing.
    """
    found = _found(weight)
    return found[1] if found else None


def variant(weight: Weight) -> str | None:
    """``"fp16"`` if that is what is cached, None if it is full precision.

    Passing ``variant="fp16"`` to ``from_pretrained`` when only the full
    precision files are there fails outright, so the caller has to know.
    """
    found = _found(weight)
    if not found:
        return None
    return "fp16" if ".fp16." in found[0] else None


def snapshot_dir(weight: Weight) -> str | None:
    """The cached snapshot directory for this weight, or None if it is not here."""
    found = _found(weight)
    if not found:
        return None
    # try_to_load_from_cache returns <snapshot>/<probe>, so trimming the probe
    # off the tail is what is left.
    probe, path = found
    return os.path.normpath(path[: len(path) - len(probe)])


def size_on_disk(weight: Weight) -> int:
    """Bytes this repo occupies in the cache, or 0 if it is not there."""
    directory = snapshot_dir(weight)
    if not directory:
        return 0

    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:  # the snapshot entries are symlinks into blobs/
                total += os.stat(os.path.join(root, name)).st_size
            except OSError:
                continue
    return total


def download(weight: Weight) -> str:
    """Fetch one weight into the cache. Idempotent — present files are kept."""
    from huggingface_hub import snapshot_download

    return snapshot_download(weight.repo, allow_patterns=list(weight.patterns))


def report(family: str | None = None) -> list[tuple[Weight, str | None]]:
    """Every declared weight, paired with its cached path or None."""
    return [(weight, present(weight)) for weight in stack(family)]


def missing(family: str | None = None) -> list[Weight]:
    return [weight for weight, path in report(family) if path is None]
