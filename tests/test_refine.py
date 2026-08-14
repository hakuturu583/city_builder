"""The refinement workflow, without a GPU or a ComfyUI.

What is worth testing here is the graph and the options: a workflow that names
a node wrongly or wires an output to the wrong input fails a minute and fifteen
gigabytes into a run, and the failure reads as a model problem.
"""

from __future__ import annotations

import pytest

from city_builder.refine import DEFAULT_PROMPT, Comfy, RefineOptions, graph

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------


def test_length_must_sit_on_the_models_frame_grid():
    # The model counts in 17k+5. A length off the grid is only discovered as a
    # shape mismatch deep in the sampler, after the weights have loaded.
    for good in (5, 22, 39, 124):
        RefineOptions(length=good)
    for bad in (1, 6, 24, 100):
        with pytest.raises(ValueError, match="17k"):
            RefineOptions(length=bad)


def test_denoise_has_to_leave_something_of_the_render():
    RefineOptions(denoise=0.25)
    RefineOptions(denoise=1.0)
    for bad in (0.0, -0.1, 1.5):
        with pytest.raises(ValueError, match="denoise"):
            RefineOptions(denoise=bad)


def test_the_canvas_is_a_multiple_of_32():
    with pytest.raises(ValueError, match="width"):
        RefineOptions(width=830)
    with pytest.raises(ValueError, match="height"):
        RefineOptions(height=481)


def test_the_default_keeps_this_street():
    # Measured: 0.25 holds the building masses and the vanishing point, 0.45
    # gives a real street that is not ours. The default has to be the first.
    assert RefineOptions().denoise <= 0.3


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def test_sampling_starts_from_the_render_not_from_noise():
    # The whole point. Wired to the H3 node's own latent instead, the render
    # would have no bearing on the result at all.
    workflow = graph("drive.mp4", DEFAULT_PROMPT, RefineOptions())
    assert workflow["run"]["inputs"]["latent_image"] == ["start", 0]
    assert workflow["start"]["class_type"] == "MiniMaxH3VideoToVideo"
    assert workflow["start"]["inputs"]["image"] == ["fit", 0]


def test_the_h3_latent_is_used_for_its_shape_and_audio_half():
    workflow = graph("drive.mp4", DEFAULT_PROMPT, RefineOptions())
    assert workflow["start"]["inputs"]["latent"] == ["h3", 1]
    assert workflow["guide"]["inputs"]["conditioning"] == ["h3", 0]


def test_the_lora_is_between_the_model_and_everything_that_samples():
    # A sampler reading the un-distilled model at four steps returns mush.
    workflow = graph("drive.mp4", DEFAULT_PROMPT, RefineOptions())
    assert workflow["lora"]["inputs"]["model"] == ["unet", 0]
    assert workflow["shift"]["inputs"]["model"] == ["lora", 0]
    for node in ("guide", "sched"):
        assert workflow[node]["inputs"]["model"] == ["shift", 0]


def test_the_render_is_fitted_to_the_canvas_being_sampled():
    options = RefineOptions(width=640, height=384)
    workflow = graph("drive.mp4", DEFAULT_PROMPT, options)
    assert workflow["fit"]["inputs"]["width"] == 640
    assert workflow["h3"]["inputs"]["width"] == 640
    assert workflow["fit"]["inputs"]["height"] == workflow["h3"]["inputs"]["height"] == 384


def test_every_reference_points_at_a_node_that_exists():
    workflow = graph("drive.mp4", DEFAULT_PROMPT, RefineOptions())
    for name, node in workflow.items():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in workflow, f"{name}.{key} points at missing {value[0]!r}"


def test_the_options_reach_the_nodes_that_use_them():
    options = RefineOptions(steps=6, denoise=0.4, seed=99, length=22)
    workflow = graph("clip.mp4", "a prompt", options)
    assert workflow["sched"]["inputs"]["steps"] == 6
    assert workflow["sched"]["inputs"]["denoise"] == pytest.approx(0.4)
    assert workflow["noise"]["inputs"]["noise_seed"] == 99
    assert workflow["h3"]["inputs"]["length"] == 22
    assert workflow["h3"]["inputs"]["prompt"] == "a prompt"
    assert workflow["video"]["inputs"]["file"] == "clip.mp4"


# ---------------------------------------------------------------------------
# The server handle
# ---------------------------------------------------------------------------


def test_a_missing_comfyui_says_where_it_looked():
    # A port nothing is on, or `start` finds whatever is already listening and
    # returns happily — which is the behaviour that keeps a warm server.
    comfy = Comfy(root="/nowhere/at/all", port=59187)
    with pytest.raises(RuntimeError, match="/nowhere/at/all"):
        comfy.start()


def test_the_url_is_built_from_the_host_and_port():
    assert Comfy(root="/x", host="127.0.0.1", port=9999).url == "http://127.0.0.1:9999"
