"""The weight declaration. No network, no cache, no model — just the table."""

from __future__ import annotations

import pytest

from city_builder import weights as W


def test_every_weight_has_a_distinct_key():
    keys = [weight.key for weight in W.stack("all")]
    assert len(keys) == len(set(keys))


def test_both_stacks_are_complete():
    """A stack that cannot make a conditioned sheet is not a stack."""
    for family in ("sd15", "sdxl"):
        roles = {weight.role for weight in W.stack(family)}
        assert {"base", "lcm-lora", "controlnet"} <= roles


def test_probes_prefer_half_precision():
    """The report says which one is cached, so the order is what decides."""
    for weight in W.stack("all"):
        halves = [i for i, probe in enumerate(weight.probe) if ".fp16." in probe]
        assert halves == list(range(len(halves))), f"{weight.key}: fp16 must come first"


def test_configs_are_always_fetched():
    """Weights without their config.json are so much dead disk."""
    for weight in W.stack("all"):
        assert "*.json" in weight.patterns


def test_no_pattern_pulls_a_duplicate_of_the_same_weights():
    """These repos ship the same tensors three or four ways over.

    A bare ``*.safetensors`` costs several times what is needed to run the
    model, and on the SD1.5 base it also drags in two single-file checkpoints.
    """
    for weight in W.stack("all"):
        if any(probe.endswith(".fp16.safetensors") for probe in weight.probe):
            assert "*.safetensors" not in weight.patterns, weight.key
        assert not any(pattern.endswith((".bin", ".ckpt", ".pth")) for pattern in weight.patterns)


def test_a_family_can_be_asked_for_by_name():
    assert W.stack("sd15") == list(W.STACKS["sd15"])
    assert len(W.stack("all")) == sum(len(s) for s in W.STACKS.values())
    assert W.stack(None) == W.stack("all")


def test_an_unknown_family_is_refused():
    with pytest.raises(KeyError):
        W.stack("sd21")


def test_the_cache_follows_hf_home(monkeypatch):
    """This machine keeps its cache on another disk; the report has to say where."""
    monkeypatch.setenv("HF_HOME", "/mnt/elsewhere/hf")
    assert W.cache_root() == "/mnt/elsewhere/hf/hub"


def test_the_cache_falls_back_to_the_default(monkeypatch):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE", raising=False)
    assert W.cache_root().endswith("huggingface/hub")


def test_variant_reports_what_is_actually_cached(monkeypatch):
    """``variant="fp16"`` on a full-precision cache fails outright, so it must be right."""
    weight = W.stack("sd15")[0]

    monkeypatch.setattr(W, "_found", lambda _w: (weight.probe[0], "/cache/" + weight.probe[0]))
    assert W.variant(weight) == "fp16"

    monkeypatch.setattr(W, "_found", lambda _w: (weight.probe[1], "/cache/" + weight.probe[1]))
    assert W.variant(weight) is None

    monkeypatch.setattr(W, "_found", lambda _w: None)
    assert W.variant(weight) is None
    assert W.present(weight) is None


def test_the_snapshot_directory_is_the_path_minus_the_probe(monkeypatch):
    weight = W.stack("sd15")[0]
    probe = weight.probe[0]
    monkeypatch.setattr(W, "_found", lambda _w: (probe, f"/cache/models--x/snapshots/abc/{probe}"))
    assert W.snapshot_dir(weight) == "/cache/models--x/snapshots/abc"
    assert W.size_on_disk(weight) == 0  # nothing there to walk


def test_missing_is_the_absent_half_of_the_report(monkeypatch):
    monkeypatch.setattr(W, "_found", lambda _w: None)
    assert W.missing("sd15") == W.stack("sd15")
    assert all(path is None for _, path in W.report("sd15"))
