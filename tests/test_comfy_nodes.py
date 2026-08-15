"""The ComfyUI node pack, when there is a ComfyUI to import it from.

Skipped without one — the nodes import comfy at module scope, which is right
for something that only ever runs inside ComfyUI. What is worth testing is the
one thing that is easy to get wrong and invisible when it is: reducing a
per-frame mask to latent time. H3 packs frames in the rhythm (1, 4, 4, 4, 4),
so a uniform resample lands the mask on the wrong frames — a mistake that shows
up as "the mask nearly works", which is worse than a crash.
"""

from __future__ import annotations

import os
import sys

import pytest

comfy_root = os.environ.get("COMFYUI_PATH", "/opt/ComfyUI")
if os.path.isdir(comfy_root) and comfy_root not in sys.path:
    sys.path.insert(0, comfy_root)

torch = pytest.importorskip("torch")
pytest.importorskip("comfy.nested_tensor")

from city_builder.comfy_nodes import (
    FRAME_PER_TOKEN,
    MiniMaxH3LatentMask,
    _pixel_frames,
)


def _latent(tokens=17, height=30, width=52):
    import comfy.nested_tensor

    video = torch.zeros((1, 24, tokens, height, width))
    audio = torch.zeros((1, 32, 2, 40))
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


def _apply(mask, tokens=17, **kwargs):
    out = MiniMaxH3LatentMask.execute(_latent(tokens=tokens), mask, **kwargs)
    result = out.result[0] if hasattr(out, "result") else out
    return result["noise_mask"]


# ---------------------------------------------------------------------------
# The grid
# ---------------------------------------------------------------------------


def test_the_frame_grid_is_the_one_the_lengths_come_from():
    assert FRAME_PER_TOKEN == (1, 4, 4, 4, 4)
    # Five tokens carry seventeen frames, which is where 17k+5 comes from.
    assert [_pixel_frames(t) for t in (1, 2, 7, 12, 17)] == [1, 5, 22, 39, 56]


# ---------------------------------------------------------------------------
# The reduction
# ---------------------------------------------------------------------------


def test_the_mask_comes_back_shaped_for_the_video_half():
    mask = torch.zeros((56, 480, 832))
    packed = _apply(mask)
    assert tuple(packed.shape) == (1, 17, 30, 52)


def test_a_frame_lands_on_the_token_that_carries_it():
    """The whole reason this node exists.

    Token 0 carries frame 0 alone; tokens 1-4 carry four frames each. A uniform
    resample of 56 frames onto 17 tokens would put frame 0 on token 0 and frame
    20 near token 6, and both would be wrong for the tokens in between.
    """
    for frame, token in ((0, 0), (1, 1), (4, 1), (5, 2), (16, 4), (17, 5), (18, 6)):
        mask = torch.zeros((56, 480, 832))
        mask[frame] = 1.0
        packed = _apply(mask, grow=0)
        lit = [t for t in range(17) if packed[0, t].max() > 0.5]
        assert lit == [token], f"frame {frame} lit tokens {lit}, wanted {token}"


def test_a_token_is_free_where_any_of_its_frames_is():
    """A token is one slice of the latent; it cannot be half-denoised in time."""
    mask = torch.zeros((56, 480, 832))
    mask[6] = 1.0  # one of the four frames token 2 carries
    packed = _apply(mask, grow=0)
    assert packed[0, 2].max() > 0.5
    assert packed[0, 1].max() == 0.0 and packed[0, 3].max() == 0.0


def test_space_is_reduced_by_the_sixteen_the_latent_is():
    mask = torch.zeros((56, 480, 832))
    mask[:, :240, :416] = 1.0  # the top-left quarter, in pixels
    packed = _apply(mask, grow=0)
    assert packed[0, :, :15, :26].min() == pytest.approx(1.0)
    assert packed[0, :, 15:, 26:].max() == pytest.approx(0.0)


def test_growing_takes_in_the_cells_the_silhouette_only_touches():
    mask = torch.zeros((56, 480, 832))
    mask[:, 240:256, 416:432] = 1.0  # one latent cell
    tight = _apply(mask, grow=0)
    grown = _apply(mask, grow=1)
    assert (tight[0, 0] > 0).sum() == 1
    assert (grown[0, 0] > 0).sum() == 9  # that cell and its ring


def test_a_threshold_makes_the_edge_hard():
    mask = torch.zeros((56, 480, 832))
    mask[:, 240:248, 416:432] = 1.0  # half of one cell, vertically
    soft = _apply(mask, grow=0)
    hard = _apply(mask, grow=0, threshold=0.75)
    assert 0.0 < soft[0, 0].max() < 1.0
    assert hard[0, 0].max() == 0.0


def test_a_clip_a_few_frames_short_holds_its_last_frame():
    mask = torch.zeros((50, 480, 832))
    mask[-1] = 1.0
    packed = _apply(mask, grow=0)
    # Frames 49-55 are all the held frame, so every token carrying one is lit.
    assert packed[0, -1].max() > 0.5


def test_a_latent_that_is_not_h3_is_refused():
    with pytest.raises(ValueError, match="H3"):
        MiniMaxH3LatentMask.execute({"samples": torch.zeros((1, 4, 64, 64))},
                                    torch.zeros((5, 64, 64)))
