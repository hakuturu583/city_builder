"""A ComfyUI node pack: start MiniMax H3 sampling from a rendered clip.

H3 ships image-to-video and reference-to-video, and neither is what refining a
render needs. Both *anchor* frames — the conditioning re-injects them at every
step, never denoised — so a Blender frame handed to `first_frame` comes back a
Blender frame, and the model simply continues the shot in that style. Measured:
the output was indistinguishable from the input.

Refinement is the other operation. The render already has the geometry right;
what it lacks is appearance. So the render becomes the *starting latent* and
only part of the noise is put back, which is `denoise` in any ComfyUI sampler.
The one thing in the way is the shape of an H3 latent: it is a nested pair of
video ``[B,24,T,H/16,W/16]`` and audio ``[B,32,2,T40]`` tensors, and a plain
VAEEncode produces the video half alone — the model then reads ``x[1]`` for the
audio stream and raises IndexError.

Hence this node. It encodes the clip with the video VAE and substitutes it into
an H3 latent, keeping that latent's audio half, so everything downstream stays
stock ComfyUI.
"""

from __future__ import annotations

import comfy.nested_tensor
import comfy.utils
import torch
from comfy.ldm.minimax.model import FRAME_PER_TOKEN
from comfy_api.latest import ComfyExtension, io


def _pixel_frames(latent_t: int) -> int:
    """How many pixel frames a video latent of ``latent_t`` tokens covers."""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def _resize(image, width: int, height: int, crop: str = "center"):
    samples = image[..., :3].movedim(-1, 1)
    samples = comfy.utils.common_upscale(samples, width, height, "lanczos", crop)
    return samples.movedim(1, -1)


class MiniMaxH3VideoToVideo(io.ComfyNode):
    """Start sampling from a clip instead of from noise."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3VideoToVideo",
            display_name="MiniMax H3 Video to Video",
            category="model/latent/minimax",
            description=(
                "Encode a clip into a MiniMax H3 AV latent so a sampler can start from it. "
                "Feed the result to SamplerCustomAdvanced as latent_image and lower the "
                "scheduler's denoise: 1.0 ignores the clip, ~0.5 keeps its layout and "
                "re-imagines its surfaces, ~0.2 barely touches it. The latent input is "
                "there for its shape and its audio half — take it from the node that "
                "made the conditioning."
            ),
            inputs=[
                io.Vae.Input("vae", tooltip="The video VAE, the same one the conditioning used."),
                io.Latent.Input("latent", tooltip="An H3 AV latent, for its canvas and audio track."),
                io.Image.Input("image", tooltip="The clip to start from. Short batches repeat "
                                                "their last frame; long ones are cut."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, vae, latent, image) -> io.NodeOutput:
        samples = latent["samples"]
        if not samples.is_nested or len(samples.tensors) != 2:
            raise ValueError("MiniMaxH3VideoToVideo expects a MiniMax H3 AV latent")
        video, audio = samples.tensors
        if video.ndim != 5 or video.shape[1] != 24:
            raise ValueError("MiniMaxH3VideoToVideo expects a MiniMax H3 AV latent")

        height, width = video.shape[3] * 16, video.shape[4] * 16
        wanted = _pixel_frames(video.shape[2])

        frames = image[:wanted]
        if frames.shape[0] < wanted:
            # A clip a frame or two short is the common case — the caller cut it
            # on seconds and the model counts in 17k+5. Hold the last frame
            # rather than refuse, and rather than loop back to the start, which
            # would ask the model to denoise a jump cut.
            tail = frames[-1:].repeat(wanted - frames.shape[0], 1, 1, 1)
            frames = torch.cat([frames, tail], dim=0)

        encoded = vae.encode(_resize(frames, width, height))
        if encoded.shape != video.shape:
            raise ValueError(
                f"encoded clip is {tuple(encoded.shape)}, but this latent wants "
                f"{tuple(video.shape)} — the width, height or length do not match")
        return io.NodeOutput({"samples": comfy.nested_tensor.NestedTensor((encoded, audio))})


class CityBuilderExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3VideoToVideo]


async def comfy_entrypoint() -> CityBuilderExtension:
    return CityBuilderExtension()
