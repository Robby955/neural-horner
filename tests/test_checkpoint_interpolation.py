from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build_interpolated_candidate.py"
SPEC = importlib.util.spec_from_file_location("checkpoint_interpolation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def checkpoint(value: float, *, width: int = 32) -> dict:
    return {
        "L": width,
        "config": {"dmodel": 2, "hidden": 3},
        "state_dict": {
            "weight": torch.tensor([value, value + 2], dtype=torch.float32),
            "bias": torch.tensor([value - 1], dtype=torch.float32),
        },
    }


def test_interpolation_endpoints_and_midpoint() -> None:
    base = checkpoint(0.0)
    endpoint = checkpoint(4.0)

    at_base = MODULE.interpolate_checkpoint(base, endpoint, 0.0)
    midpoint = MODULE.interpolate_checkpoint(base, endpoint, 0.5)
    at_endpoint = MODULE.interpolate_checkpoint(base, endpoint, 1.0)

    assert torch.equal(at_base["state_dict"]["weight"], torch.tensor([0.0, 2.0]))
    assert torch.equal(midpoint["state_dict"]["weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(at_endpoint["state_dict"]["weight"], torch.tensor([4.0, 6.0]))
    assert set(midpoint) == {"L", "config", "state_dict"}


def test_interpolation_rejects_incompatible_checkpoints() -> None:
    with pytest.raises(ValueError, match="L values differ"):
        MODULE.interpolate_checkpoint(checkpoint(0.0), checkpoint(1.0, width=64), 0.5)

    endpoint = checkpoint(1.0)
    endpoint["state_dict"]["weight"] = torch.ones(3)
    with pytest.raises(ValueError, match="shape mismatch"):
        MODULE.interpolate_checkpoint(checkpoint(0.0), endpoint, 0.5)


def test_interpolation_manifest_replaces_weight_provenance() -> None:
    manifest = MODULE.interpolation_manifest(
        {"model_name": "candidate", "training_description": "base weights"},
        alpha=0.9375,
        base_label="v8",
        base_sha256="a" * 64,
        endpoint_label="function-space-step1500",
        endpoint_sha256="b" * 64,
    )

    description = manifest["training_description"]
    assert "alpha=0.9375" in description
    assert "v8" in description
    assert "function-space-step1500" in description
    assert "candidate-specific evaluation receipts" in description


def test_interpolation_manifest_can_select_original_schedule() -> None:
    manifest = MODULE.interpolation_manifest(
        {"model_description": "stale", "training_description": "base"},
        alpha=0.875,
        base_label="v8",
        base_sha256="a" * 64,
        endpoint_label="l2sp",
        endpoint_sha256="b" * 64,
        inference_description="original three-pass bit-serial Horner schedule",
        model_description="current L=2048 control",
    )

    assert manifest["model_description"] == "current L=2048 control"
    assert "original three-pass" in manifest["training_description"]
