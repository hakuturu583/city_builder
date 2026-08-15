"""Photoreal refinement of a render, through ComfyUI and MiniMax H3.

The scene this package builds has its geometry right and its appearance
approximate: procedural facades, a tiled road, a flat sky. A video model knows
what a street looks like and nothing about where this one's buildings are. So
the render goes in as the *starting latent* and only part of the noise is put
back — the geometry is held by what is already in the latent, and the model
supplies the surfaces.

`denoise` is the whole dial, and the measurement that matters (832x480, 4 steps,
one 3090):

===========  ==================================================================
0.25         the same street, photoreal. Building masses, road width, vanishing
             point and crossing all stay put.
0.35         more convincing, and the buildings start becoming generic
0.45+        a real Japanese street, but no longer *this* one
===========  ==================================================================

That ordering is why the default is low. These frames are wanted as reference
for regenerating textures, and a reference whose windows sit on different floors
than the building it is meant to describe is worse than none.

Image-to-video is not this. H3's keyframe conditioning is re-injected at every
step and never denoised, so a Blender frame handed to `first_frame` comes back
a Blender frame — measured, the output was indistinguishable from the input.
The starting latent is the only place the render can enter and still be changed.

ComfyUI runs as a subprocess speaking HTTP on localhost, because that is the
only interface it has; the workflow is built here as an API graph rather than
by editing an exported one, which carries node positions and link ids that a
caller should not have to keep in step with.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any

DEFAULT_PROMPT = (
    "photorealistic dashcam still of a Japanese city street, overcast daylight, "
    "worn asphalt, painted lane markings, concrete and tiled mid-rise buildings, "
    "shopfronts at street level, realistic materials and lighting"
)


@dataclass
class RefineOptions:
    """What to run, and how far to let it go."""

    # The 17k+5 grid the model counts in: 5, 22, 39, 56… Five frames is one
    # sampling pass at its cheapest, and reference stills is what this is for.
    length: int = 5
    width: int = 832
    height: int = 480
    steps: int = 4  # the Turbo LoRA is distilled to four
    denoise: float = 0.25
    seed: int = 0
    lora_strength: float = 1.0
    sigma_shift: float = 12.0
    sampler: str = "res_multistep"
    scheduler: str = "simple"

    # Masked refinement: how far outside the subject the model may work. One
    # latent cell is 16 pixels, and the mask has already been area-averaged down
    # to cells, so 1 keeps the subject's own silhouette edge inside the editable
    # region rather than on its boundary.
    mask_grow: int = 1
    mask_threshold: float = 0.0  # 0 keeps the soft edge; above 0 makes it hard

    unet: str = "MiniMax-H3-FL2VA-Pruned-Q5_K_M.gguf"
    lora: str = "minimax_h3_turbo_4step_ckpt600_ema_V4.safetensors"
    clip: str = "qwen3vl_32b_minimax_h3-Q4_K_M.gguf"
    video_vae: str = "minimax_h3_video_vae_fp16.safetensors"

    def __post_init__(self) -> None:
        if not 0.0 < self.denoise <= 1.0:
            raise ValueError("refine.denoise must be in (0, 1]")
        if (self.length - 5) % 17:
            raise ValueError(
                f"refine.length must be 5, 22, 39, … (17k+5); {self.length} is not")
        for name in ("width", "height"):
            if getattr(self, name) % 32:
                raise ValueError(f"refine.{name} must be a multiple of 32")


def graph(video_path: str, prompt: str, options: RefineOptions,
          mask_path: str | None = None) -> dict[str, Any]:
    """The API-format workflow, as a dict of nodes.

    With ``mask_path``, sampling is confined to where that clip is white — the
    building — and everything else comes back exactly as it was rendered. That
    is worth more here than it sounds: the road, the ground and the horizon are
    the only things in frame that state the scale and the place, and they are
    the things a video model is most inclined to rewrite.
    """
    nodes = {
        "unet":  {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": options.unet}},
        "lora":  {"class_type": "LoraLoaderModelOnly",
                  "inputs": {"model": ["unet", 0], "lora_name": options.lora,
                             "strength_model": options.lora_strength}},
        "shift": {"class_type": "MiniMaxH3SigmaShift",
                  "inputs": {"model": ["lora", 0], "shift_video": options.sigma_shift,
                             "shift_audio": 6.0}},
        "clip":  {"class_type": "CLIPLoaderGGUF",
                  "inputs": {"clip_name": options.clip, "type": "minimax"}},
        "vae":   {"class_type": "VAELoader", "inputs": {"vae_name": options.video_vae}},
        "video": {"class_type": "LoadVideo", "inputs": {"file": video_path}},
        "parts": {"class_type": "GetVideoComponents", "inputs": {"video": ["video", 0]}},
        "fit":   {"class_type": "ImageScale",
                  "inputs": {"image": ["parts", 0], "upscale_method": "lanczos",
                             "width": options.width, "height": options.height,
                             "crop": "center"}},
        # Conditioning only — its latent is taken for its shape and audio half.
        "h3":    {"class_type": "MiniMaxH3ImageToVideo",
                  "inputs": {"clip": ["clip", 0], "vae": ["vae", 0], "prompt": prompt,
                             "width": options.width, "height": options.height,
                             "length": options.length}},
        "start": {"class_type": "MiniMaxH3VideoToVideo",
                  "inputs": {"vae": ["vae", 0], "latent": ["h3", 1], "image": ["fit", 0]}},
        "guide": {"class_type": "BasicGuider",
                  "inputs": {"model": ["shift", 0], "conditioning": ["h3", 0]}},
        "noise": {"class_type": "RandomNoise", "inputs": {"noise_seed": options.seed}},
        "samp":  {"class_type": "KSamplerSelect", "inputs": {"sampler_name": options.sampler}},
        "sched": {"class_type": "BasicScheduler",
                  "inputs": {"model": ["shift", 0], "scheduler": options.scheduler,
                             "steps": options.steps, "denoise": options.denoise}},
        "run":   {"class_type": "SamplerCustomAdvanced",
                  "inputs": {"noise": ["noise", 0], "guider": ["guide", 0],
                             "sampler": ["samp", 0], "sigmas": ["sched", 0],
                             "latent_image": ["start", 0]}},
        "dec":   {"class_type": "VAEDecode", "inputs": {"samples": ["run", 0], "vae": ["vae", 0]}},
        "save":  {"class_type": "SaveImage",
                  "inputs": {"images": ["dec", 0], "filename_prefix": "refined"}},
    }
    if mask_path is None:
        return nodes

    nodes.update({
        "mvid":  {"class_type": "LoadVideo", "inputs": {"file": mask_path}},
        "mpart": {"class_type": "GetVideoComponents", "inputs": {"video": ["mvid", 0]}},
        # To the pixel size the clip was fitted to, so the mask and the frames it
        # describes are the same picture before anything is reduced.
        "mfit":  {"class_type": "ImageScale",
                  "inputs": {"image": ["mpart", 0], "upscale_method": "bilinear",
                             "width": options.width, "height": options.height,
                             "crop": "center"}},
        "mconv": {"class_type": "ImageToMask", "inputs": {"image": ["mfit", 0], "channel": "red"}},
        "mask":  {"class_type": "MiniMaxH3LatentMask",
                  "inputs": {"latent": ["start", 0], "mask": ["mconv", 0],
                             "grow": options.mask_grow,
                             "threshold": options.mask_threshold}},
    })
    nodes["run"]["inputs"]["latent_image"] = ["mask", 0]
    return nodes


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


@dataclass
class Comfy:
    """A ComfyUI on localhost: started if it is not already there.

    Loading the model is ten seconds and the weights are fifteen gigabytes, so
    the server is worth keeping between calls. It is started lazily rather than
    with the MCP process, because a session that never refines anything should
    not pay for it.
    """

    root: str = field(default_factory=lambda: os.environ.get("COMFYUI_PATH", "/opt/ComfyUI"))
    port: int = 8188
    host: str = "127.0.0.1"
    startup_timeout: float = 300.0
    # Where ComfyUI keeps what it writes. Its checkout is not the right place:
    # in the container that is a root-owned image layer, and a server run as
    # anyone else cannot open its own database there. A directory of ours also
    # means nothing accumulates inside the image.
    workdir: str | None = None
    _process: subprocess.Popen | None = field(default=None, repr=False)
    _log: str | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.workdir is None:
            import tempfile
            self.workdir = os.path.join(tempfile.gettempdir(), "city-builder-comfy")
        for name in ("input", "output", "temp", "user"):
            os.makedirs(os.path.join(self.workdir, name), exist_ok=True)

    @property
    def input_dir(self) -> str:
        return os.path.join(self.workdir, "input")

    @property
    def output_dir(self) -> str:
        return os.path.join(self.workdir, "output")

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def alive(self) -> bool:
        try:
            urllib.request.urlopen(f"{self.url}/system_stats", timeout=3).read()
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError):
            return False

    def start(self) -> None:
        if self.alive():
            return
        main = os.path.join(self.root, "main.py")
        if not os.path.exists(main):
            raise RuntimeError(
                f"no ComfyUI at {self.root}. Set COMFYUI_PATH, or use the container, "
                "which ships one.")
        self._log = os.path.join(self.workdir, "comfyui.log")
        with open(self._log, "w") as log:
            self._process = subprocess.Popen(
                # --disable-smart-memory so the text encoder is unloaded before
                # sampling: both models are around fifteen gigabytes and only one
                # of them can be resident on a 24 GB card.
                [sys.executable, main, "--listen", self.host, "--port", str(self.port),
                 "--disable-auto-launch", "--disable-smart-memory", "--cache-none",
                 "--reserve-vram", "0.6",
                 "--input-directory", self.input_dir,
                 "--output-directory", self.output_dir,
                 "--temp-directory", os.path.join(self.workdir, "temp"),
                 "--user-directory", os.path.join(self.workdir, "user")],
                cwd=self.root, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + self.startup_timeout
        while time.time() < deadline:
            if self.alive():
                return
            if self._process.poll() is not None:
                raise RuntimeError(
                    f"ComfyUI exited with {self._process.returncode}:\n{self.log_tail()}")
            time.sleep(2)
        raise TimeoutError(
            f"ComfyUI did not answer within {self.startup_timeout:.0f}s:\n{self.log_tail()}")

    def log_tail(self, lines: int = 20) -> str:
        """The end of ComfyUI's own output — the only place its reason appears."""
        if not self._log or not os.path.exists(self._log):
            return "(no log)"
        with open(self._log, errors="replace") as handle:
            return "".join(handle.readlines()[-lines:])

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=30)
        self._process = None

    # -- talking to it ------------------------------------------------------

    def _get(self, path: str) -> Any:
        with urllib.request.urlopen(self.url + path, timeout=30) as response:
            return json.load(response)

    def nodes(self) -> set[str]:
        return set(self._get("/object_info"))

    def submit(self, workflow: dict[str, Any]) -> str:
        request = urllib.request.Request(
            self.url + "/prompt", data=json.dumps({"prompt": workflow}).encode(),
            headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)["prompt_id"]
        except urllib.error.HTTPError as error:
            raise RuntimeError(f"ComfyUI refused the workflow: "
                               f"{error.read().decode()[:600]}") from error

    def wait(self, prompt_id: str, timeout: float = 1800.0) -> list[str]:
        """Block until the job finishes; return the image files it wrote."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            history = self._get(f"/history/{prompt_id}").get(prompt_id, {})
            status = history.get("status", {})
            if status.get("status_str") == "error":
                for kind, payload in status.get("messages", []):
                    if kind == "execution_error":
                        raise RuntimeError(
                            f"{payload.get('node_type')}: {payload.get('exception_message')}")
                raise RuntimeError("ComfyUI reported an error with no detail")
            if status.get("completed"):
                return [image["filename"]
                        for output in history.get("outputs", {}).values()
                        for image in output.get("images", [])]
            time.sleep(2)
        raise TimeoutError(f"the refinement did not finish within {timeout:.0f}s")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def mask_clip(source: str, target: str) -> str:
    """A directory of mask PNGs as one lossless clip, for ComfyUI's loader.

    ComfyUI reads videos, and the mask is written as PNGs because a mask that
    has been through H.264's chroma subsampling has grey where it had an edge.
    ``-crf 0`` keeps luma exact, which is all a greyscale mask has.
    """
    import subprocess

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is not on PATH; cannot pack the mask frames into a clip")

    frames = sorted(f for f in os.listdir(source) if f.endswith(".png"))
    if not frames:
        raise ValueError(f"no mask PNGs in {source}")
    stem = os.path.commonprefix(frames)
    pattern = os.path.join(source, f"{stem}%0{len(frames[0]) - len(stem) - 4}d.png")

    subprocess.run(
        [ffmpeg, "-y", "-loglevel", "error", "-framerate", "24", "-i", pattern,
         "-c:v", "libx264", "-crf", "0", "-preset", "veryfast", "-pix_fmt", "yuv420p", target],
        check=True)
    return target


def refine(video_path: str, out_dir: str, *, prompt: str = DEFAULT_PROMPT,
           mask: str | None = None, options: RefineOptions | None = None,
           comfy: Comfy | None = None) -> dict[str, Any]:
    """Refine a rendered clip and copy the frames out.

    ``video_path`` is any video ComfyUI can read; it is copied into ComfyUI's
    input directory, which is the only place its loader looks.

    ``mask`` is optional and is either a clip or a directory of per-frame PNGs —
    what :func:`city_builder.orbit.render_orbit` writes. Where it is black the
    render comes back untouched.
    """
    options = options or RefineOptions()
    comfy = comfy or Comfy()
    comfy.start()

    wanted = {"MiniMaxH3VideoToVideo", "UnetLoaderGGUF", "MiniMaxH3ImageToVideo"}
    if mask:
        wanted.add("MiniMaxH3LatentMask")
    missing = wanted - comfy.nodes()
    if missing:
        raise RuntimeError(
            f"this ComfyUI is missing {', '.join(sorted(missing))}. The GGUF loader comes "
            "from the ComfyUI-GGUF custom node; MiniMaxH3VideoToVideo is this package's "
            "own, in src/city_builder/comfy_nodes.")

    name = f"refine_{os.path.basename(video_path)}"
    shutil.copyfile(video_path, os.path.join(comfy.input_dir, name))

    mask_name = None
    if mask:
        mask_name = f"mask_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
        target = os.path.join(comfy.input_dir, mask_name)
        if os.path.isdir(mask):
            mask_clip(mask, target)
        else:
            shutil.copyfile(mask, target)

    started = time.time()
    files = comfy.wait(comfy.submit(graph(name, prompt, options, mask_name)))

    os.makedirs(out_dir, exist_ok=True)
    written = []
    for filename in files:
        source = os.path.join(comfy.output_dir, filename)
        target = os.path.join(out_dir, filename)
        if os.path.exists(source):
            shutil.copyfile(source, target)
            written.append(target)
    return {
        "frames": written,
        "took_seconds": round(time.time() - started, 1),
        "denoise": options.denoise,
        "masked": bool(mask),
        "size": [options.width, options.height],
    }
