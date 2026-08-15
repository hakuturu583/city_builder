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
    """How many pixel frames a video latent of ``latent_t`` tokens covers.

    ``FRAME_PER_TOKEN`` is ``(1, 4, 4, 4, 4)``, so five tokens carry seventeen
    frames and the run of accepted lengths is 17k+5. It also means the packing
    is *not* a stride: 56 frames are 17 tokens, and the first of every five
    covers one frame where its neighbours cover four. Anything reduced from
    pixel time to latent time has to follow that rhythm.
    """
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


class MiniMaxH3LatentMask(io.ComfyNode):
    """Fold a per-frame mask down onto an H3 latent, on the model's own grid."""

    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3LatentMask",
            display_name="MiniMax H3 Latent Mask",
            category="model/latent/minimax",
            description=(
                "Attach a noise mask to a MiniMax H3 latent, so sampling changes only part "
                "of the frame. The mask is one image per pixel frame; this reduces it to the "
                "latent's own resolution — 16x in space, and in time by the model's "
                "FRAME_PER_TOKEN grouping rather than by a uniform stride, which is what a "
                "plain SetLatentNoiseMask would get wrong."
            ),
            inputs=[
                io.Latent.Input("latent", tooltip="The H3 AV latent sampling will start from."),
                io.Mask.Input("mask", tooltip="White where the frame may change. One per "
                                              "pixel frame; a short batch holds its last."),
                io.Int.Input("grow", default=1, min=0, max=16,
                             tooltip="Dilate by this many latent cells. One cell is 16 pixels; "
                                     "1 keeps the subject's own edge inside the editable area."),
                io.Float.Input("threshold", default=0.0, min=0.0, max=1.0, step=0.01,
                               tooltip="0 keeps the soft edge area-averaging gives. Above 0 "
                                       "makes the mask hard at that coverage."),
            ],
            outputs=[io.Latent.Output()],
        )

    @classmethod
    def execute(cls, latent, mask, grow=1, threshold=0.0) -> io.NodeOutput:
        samples = latent["samples"]
        if not getattr(samples, "is_nested", False) or len(samples.tensors) != 2:
            raise ValueError("MiniMaxH3LatentMask expects a MiniMax H3 AV latent")
        video = samples.tensors[0]
        if video.ndim != 5:
            raise ValueError("MiniMaxH3LatentMask expects a MiniMax H3 AV latent")
        _batch, _channels, tokens, height, width = video.shape

        frames = mask
        if frames.ndim == 4:  # an IMAGE handed in as a mask: take one channel
            frames = frames[..., 0]
        elif frames.ndim == 2:
            frames = frames.unsqueeze(0)
        frames = frames.float()

        wanted = _pixel_frames(tokens)
        frames = frames[:wanted]
        if frames.shape[0] < wanted:
            # Same reason MiniMaxH3VideoToVideo holds its last frame: the caller
            # cut the clip on seconds and the model counts in 17k+5.
            frames = torch.cat(
                [frames, frames[-1:].repeat(wanted - frames.shape[0], 1, 1)], dim=0)

        # A latent cell is 16x16 pixels, so area-averaging says what share of the
        # cell the subject covers — a soft edge, which is what a boundary between
        # "hold this" and "change this" should be.
        cells = torch.nn.functional.interpolate(
            frames.unsqueeze(1), size=(height, width), mode="area")

        # A token covers 1 or 4 pixel frames, in the rhythm (1,4,4,4,4). Any frame
        # in which the subject appears has to leave its token free to change: a
        # token is one slice of the latent and cannot be half-denoised in time.
        groups, start = [], 0
        for k in range(tokens):
            span = FRAME_PER_TOKEN[k % 5]
            groups.append(cells[start:start + span].amax(dim=0))
            start += span
        packed = torch.stack(groups, dim=1)  # [1, tokens, height, width]

        if grow:
            packed = torch.nn.functional.max_pool2d(
                packed, kernel_size=2 * grow + 1, stride=1, padding=grow)
        if threshold > 0.0:
            packed = (packed > threshold).float()

        out = latent.copy()
        out["noise_mask"] = packed
        return io.NodeOutput(out)


class CityBuilderExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3VideoToVideo, MiniMaxH3LatentMask]


async def comfy_entrypoint() -> CityBuilderExtension:
    return CityBuilderExtension()
