#!/usr/bin/env python3
"""Build a provenance-bound checkpoint interpolation artifact.

The output is a self-contained copy of a submission template whose weights are
``(1 - alpha) * base + alpha * endpoint``.  Checkpoint structure, tensor names,
shapes, and dtypes must match exactly.  No training is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_compatible(base: dict, endpoint: dict) -> None:
    if int(base.get("L", 32)) != int(endpoint.get("L", 32)):
        raise ValueError("checkpoint L values differ")
    if base.get("config", {}) != endpoint.get("config", {}):
        raise ValueError("checkpoint model configs differ")

    base_state = base["state_dict"]
    endpoint_state = endpoint["state_dict"]
    if list(base_state) != list(endpoint_state):
        raise ValueError("state_dict tensor names or order differ")
    for name in base_state:
        left = base_state[name]
        right = endpoint_state[name]
        if left.shape != right.shape:
            raise ValueError(f"shape mismatch for {name}: {left.shape} != {right.shape}")
        if left.dtype != right.dtype:
            raise ValueError(f"dtype mismatch for {name}: {left.dtype} != {right.dtype}")
        if not (left.is_floating_point() and right.is_floating_point()):
            raise ValueError(f"non-floating parameter cannot be interpolated: {name}")


def interpolate_checkpoint(base: dict, endpoint: dict, alpha: float) -> dict:
    validate_compatible(base, endpoint)
    return {
        "L": int(base.get("L", 32)),
        "config": dict(base.get("config", {})),
        "state_dict": {
            name: base["state_dict"][name].lerp(
                endpoint["state_dict"][name], alpha
            )
            for name in base["state_dict"]
        },
    }


def interpolation_manifest(
    template: dict,
    *,
    alpha: float,
    base_label: str,
    base_sha256: str,
    endpoint_label: str,
    endpoint_sha256: str,
    inference_description: str = "fixed direct two-pass bit-serial Horner schedule",
    model_description: str | None = None,
) -> dict:
    output = dict(template)
    if model_description is not None:
        output["model_description"] = model_description
    output["training_description"] = (
        "Weight-space interpolation with alpha="
        f"{alpha:g} between {base_label} (sha256:{base_sha256}) and "
        f"{endpoint_label} (sha256:{endpoint_sha256}). The inference program "
        f"is the {inference_description}. This artifact "
        "requires its own candidate-specific evaluation receipts."
    )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--base-label")
    parser.add_argument("--endpoint", type=Path, required=True)
    parser.add_argument("--endpoint-label")
    parser.add_argument("--alpha", type=float, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--inference-description",
        default="fixed direct two-pass bit-serial Horner schedule",
    )
    parser.add_argument("--model-description")
    args = parser.parse_args()

    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    base_path = args.base.resolve()
    endpoint_path = args.endpoint.resolve()
    template = args.template.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    for required in ("model.py", "manifest.json"):
        if not (template / required).is_file():
            raise FileNotFoundError(f"template is missing {required}")

    base = torch.load(base_path, map_location="cpu", weights_only=True)
    endpoint = torch.load(endpoint_path, map_location="cpu", weights_only=True)
    interpolated = interpolate_checkpoint(base, endpoint, args.alpha)
    base_sha256 = sha256_file(base_path)
    endpoint_sha256 = sha256_file(endpoint_path)
    base_label = args.base_label or base_path.name
    endpoint_label = args.endpoint_label or endpoint_path.name

    output.mkdir(parents=True)
    shutil.copy2(template / "model.py", output / "model.py")
    template_manifest = json.loads((template / "manifest.json").read_text())
    manifest = interpolation_manifest(
        template_manifest,
        alpha=args.alpha,
        base_label=base_label,
        base_sha256=base_sha256,
        endpoint_label=endpoint_label,
        endpoint_sha256=endpoint_sha256,
        inference_description=args.inference_description,
        model_description=args.model_description,
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    torch.save(interpolated, output / "weights.pt")
    provenance = {
        "method": "linear_checkpoint_interpolation",
        "alpha": args.alpha,
        "base": base_label,
        "base_sha256": base_sha256,
        "endpoint": endpoint_label,
        "endpoint_sha256": endpoint_sha256,
        "base_checkpoint_metadata": {
            key: value for key, value in base.items() if key != "state_dict"
        },
        "endpoint_checkpoint_metadata": {
            key: value for key, value in endpoint.items() if key != "state_dict"
        },
        "template": template.name,
        "inference_description": args.inference_description,
        "output_weights_sha256": sha256_file(output / "weights.pt"),
        "L": int(interpolated.get("L", 32)),
        "config": interpolated.get("config", {}),
        "parameter_tensors": len(interpolated["state_dict"]),
        "parameters": sum(
            tensor.numel() for tensor in interpolated["state_dict"].values()
        ),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
