#!/usr/bin/env python3
"""Replay the two decisive transitions in the canonical v8 F11 x 1 failure.

The historical full-prefix receipt establishes that reducing F11 is exact for
2,048 transitions and reaches state 2^2047 before the final input bit for the
checkpoint named in that receipt. This script independently evaluates the final
reduction transition and one-bit second scan with a selected candidate. Prefix
evidence is reusable only when its checkpoint hash matches the candidate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from held_out_battery import generate_battery
from submission_utils import artifact_identity, load_submission, sha256_file


def bits_to_int(row: torch.Tensor) -> int:
    value = 0
    for bit in row.long().tolist():
        value = 2 * value + int(bit)
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--prefix-receipt", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--require-counterexample", action="store_true")
    parser.add_argument("--require-two-step-exact", action="store_true")
    args = parser.parse_args()

    submission, manifest, module, model = load_submission(args.submission)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    model.device = device
    model.model.to(device)
    model.model.eval()
    if model.L != 2048:
        raise ValueError(f"expected L=2048, found {model.L}")

    p, _, _, _ = generate_battery(model.L, 1)
    f11 = (1 << (1 << 11)) + 1
    scan_bits, reduce_bits = module._canonical_scan_reduce(
        module._bits_of(f11), module._bits_of(1)
    )
    routing_exact = scan_bits == [1] and reduce_bits == module._bits_of(f11)

    width = 2048
    p_bits = module.to_bits_limbs([p], device, width).float()
    one_bits = module.to_bits_limbs([1], device, width).float()
    state_before_final = module.to_bits_limbs([1 << 2047], device, width).float()
    digit_one = torch.ones(1, dtype=torch.long, device=device)

    with torch.no_grad():
        predicted_residue_bits = model._step(
            state_before_final, one_bits, p_bits, digit_one
        )
        predicted_residue = bits_to_int(predicted_residue_bits[0])
        expected_residue = f11 % p
        reduction_exact = predicted_residue == expected_residue

        zero_state = torch.zeros((1, width), device=device)
        final_bits = model._step(
            zero_state, predicted_residue_bits, p_bits, digit_one
        )
        final_value = bits_to_int(final_bits[0])
        expected_from_wrong_residue = predicted_residue % p
        second_scan_exact_relative_to_input = (
            final_value == expected_from_wrong_residue
        )
        final_exact = final_value == expected_residue

    prefix_data = json.loads(args.prefix_receipt.resolve().read_text())
    prefix_checkpoint_sha256 = (
        prefix_data.get("provenance", {})
        .get("frozen_artifacts", {})
        .get("checkpoint_sha256")
    )
    identity = artifact_identity(submission)
    prefix_compatible = (
        prefix_checkpoint_sha256 is not None
        and prefix_checkpoint_sha256 == identity.get("weights.pt")
    )
    counterexample_reproduced = (
        prefix_compatible
        and routing_exact
        and not reduction_exact
        and second_scan_exact_relative_to_input
        and not final_exact
    )
    two_step_exact = reduction_exact and second_scan_exact_relative_to_input and final_exact
    if counterexample_reproduced:
        status = "counterexample_reproduced"
    elif two_step_exact:
        status = (
            "full_prefix_plus_two_step_exact"
            if prefix_compatible
            else "injected_two_step_exact"
        )
    else:
        status = "inconclusive"
    receipt = {
        "status": status,
        "submission": str(submission),
        "entry_class": manifest["entry_class"],
        "artifact_sha256": identity,
        "runner_sha256": sha256_file(Path(__file__)),
        "prefix_receipt": str(args.prefix_receipt.resolve()),
        "prefix_receipt_sha256": sha256_file(args.prefix_receipt.resolve()),
        "prefix_checkpoint_sha256": prefix_checkpoint_sha256,
        "prefix_compatible_with_candidate_weights": prefix_compatible,
        "candidate_prefix_executed": False,
        "incoming_state_source": "exact_state_injected_before_final_f11_bit",
        "device": str(device),
        "torch": torch.__version__,
        "p_bits": p.bit_length(),
        "routing_reduces_f11": routing_exact,
        "final_reduction_transition_exact": reduction_exact,
        "incorrect_reduction_bits": sum(
            left != right
            for left, right in zip(
                predicted_residue_bits[0].long().tolist(),
                module.to_bits_limbs([expected_residue], device, width)[0].tolist(),
            )
        ),
        "second_scan_exact_relative_to_input_residue": (
            second_scan_exact_relative_to_input
        ),
        "final_product_exact": final_exact,
        "two_step_exact": two_step_exact,
        "counterexample_reproduced": counterexample_reproduced,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if args.require_counterexample and not counterexample_reproduced:
        return 1
    if args.require_two_step_exact and not two_step_exact:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
