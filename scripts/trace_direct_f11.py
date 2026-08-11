#!/usr/bin/env python3
"""Trace every learned transition on legacy and fresh F11 companion cases.

Both original and operand-swapped orientations are evaluated. The canonical
direct schedule must choose the identical trajectory for both orientations and
route every pair identically after an operand swap. Some fresh companions are
shorter than F11 and intentionally force F11 into the reduction phase.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from pathlib import Path

import torch

from held_out_battery import generate_battery
from submission_utils import artifact_identity, load_submission, sha256_file


def bits_to_int(row) -> int:
    value = 0
    for bit in row.tolist():
        value = 2 * value + int(bit > 0.5)
    return value


def case_id(a: int, b: int, p: int) -> str:
    return hashlib.sha256(f"{a}:{b}:{p}".encode()).hexdigest()[:20]


def fresh_f11_companions(seed: int = 20260801) -> list[tuple[int, str]]:
    rng = random.Random(seed)
    companions = []
    for width in (1, 32, 256, 1024, 2048, 2049, 3072, 4096):
        value = 1 if width == 1 else (1 << (width - 1)) | rng.getrandbits(width - 1)
        companions.append((value, f"fresh-width-{width}"))
    f11 = (1 << (1 << 11)) + 1
    companions.extend(
        [
            (f11 - 1, "fresh-equal-length-below-f11"),
            (f11, "fresh-equal-length-equal-f11"),
            (f11 + 2, "fresh-equal-length-above-f11"),
        ]
    )
    return companions


def make_tracer(base_type):
    class TracingDirect(base_type):
        def reset_trace(self, row_count: int) -> None:
            self._trace_scan_index = 0
            self.trace_ok = [0] * row_count
            self.trace_bad = [0] * row_count
            self.first_divergence = [None] * row_count
            self.trace_hashes = [hashlib.sha256() for _ in range(row_count)]
            self.phase_counts = [
                {"reduce_operand": 0, "scan_operand": 0}
                for _ in range(row_count)
            ]

        def _scan_bits(self, bit_lists, x_bits, p_bits, effective_width):
            phase = (
                "reduce_operand"
                if self._trace_scan_index == 0
                else "scan_operand"
            )
            self._trace_scan_index += 1
            row_count = len(bit_lists)
            scan_width = max(len(bits) for bits in bit_lists)
            padded = torch.zeros(
                (row_count, scan_width),
                dtype=torch.long,
                device=self.device,
            )
            starts = torch.empty(row_count, dtype=torch.long, device=self.device)
            for row_index, bits in enumerate(bit_lists):
                start = scan_width - len(bits)
                starts[row_index] = start
                padded[row_index, start:] = torch.tensor(
                    bits, dtype=torch.long, device=self.device
                )

            state = torch.zeros(
                (row_count, effective_width), device=self.device
            )
            active_steps = [0] * row_count
            for position in range(scan_width):
                digits = padded[:, position]
                next_state = super()._step(state, x_bits, p_bits, digits)
                active = position >= starts
                for row_index in range(row_count):
                    if not bool(active[row_index].item()):
                        continue
                    s = bits_to_int(state[row_index])
                    x = bits_to_int(x_bits[row_index])
                    p = bits_to_int(p_bits[row_index])
                    d = int(digits[row_index].item())
                    neural = bits_to_int(next_state[row_index])
                    expected = (2 * s + d * x) % p
                    self.trace_hashes[row_index].update(
                        (
                            f"{phase}:{active_steps[row_index]}:{s}:{x}:"
                            f"{p}:{d}:{neural}:{expected}\n"
                        ).encode()
                    )
                    self.phase_counts[row_index][phase] += 1
                    if neural == expected:
                        self.trace_ok[row_index] += 1
                    else:
                        self.trace_bad[row_index] += 1
                        if self.first_divergence[row_index] is None:
                            self.first_divergence[row_index] = {
                                "phase": phase,
                                "step": active_steps[row_index],
                                "s": str(s),
                                "x": str(x),
                                "d": d,
                                "neural": str(neural),
                                "expected": str(expected),
                            }
                    active_steps[row_index] += 1
                state = torch.where(active[:, None], next_state, state)
            return state

    return TracingDirect


def decode(bits) -> int:
    value = 0
    for bit in bits:
        value = 2 * value + int(bit)
    return value


def orientation_output_count_exact(expected: int, outputs) -> bool:
    """Reject missing or extra model outputs before interpreting a trace prefix."""
    return len(outputs) == expected


def paired_result_counts_exact(expected: int, original, swapped) -> bool:
    """Require a complete one-to-one swap comparison, not a truncated zip."""
    return len(original) == expected and len(swapped) == expected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument(
        "--suite",
        choices=("legacy", "fresh", "both"),
        default="both",
    )
    parser.add_argument(
        "--orientation",
        choices=("original", "swapped", "both"),
        default="both",
    )
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--require-exact", action="store_true")
    args = parser.parse_args()

    submission, manifest, module, loaded = load_submission(args.submission)
    if not hasattr(type(loaded), "_scan_bits"):
        raise TypeError("entry class is not a direct two-pass scanner")

    tracer_type = make_tracer(type(loaded))
    model = tracer_type()
    model.load(str(submission))
    p, _, _, categories = generate_battery(model.L, 128)
    f11 = (1 << (1 << 11)) + 1
    legacy = [
        (a, b) for a, b in categories["fermat numbers"] if a == f11
    ]
    if len(legacy) != 9:
        raise RuntimeError(f"expected nine legacy F11 cases, found {len(legacy)}")
    selected: list[tuple[int, int, str]] = []
    if args.suite in ("legacy", "both"):
        selected.extend((a, b, f"legacy-{index}") for index, (a, b) in enumerate(legacy))
    if args.suite in ("fresh", "both"):
        selected.extend((f11, b, label) for b, label in fresh_f11_companions())

    orientations = (
        ("original", "swapped") if args.orientation == "both" else (args.orientation,)
    )
    receipt = {
        "status": "running",
        "submission": str(submission),
        "entry_class": manifest["entry_class"],
        "artifact_sha256": artifact_identity(submission),
        "battery_seed": 20260627,
        "fresh_seed": 20260801,
        "suite": args.suite,
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "device": str(model.device),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "expected_cases_per_orientation": len(selected),
        "expected_orientations": len(orientations),
        "expected_total_cases": len(selected) * len(orientations),
        "orientations": {},
        "orientation_output_counts": {},
    }
    frozen_digest = hashlib.sha256()
    frozen_digest.update(str(p).encode())
    for a, b, label in selected:
        frozen_digest.update(f"{label}:{a}:{b}\n".encode())
    receipt["frozen_cases_sha256"] = frozen_digest.hexdigest()
    all_exact = True
    for orientation in orientations:
        cases = (
            selected
            if orientation == "original"
            else [(b, a, label) for a, b, label in selected]
        )
        encoded = [
            (
                model.preprocess_a(a),
                model.preprocess_b(b),
                model.preprocess_p(p),
            )
            for a, b, _ in cases
        ]
        model.reset_trace(len(cases))
        digits = model.predict_digits_batch(encoded)
        output_count_exact = orientation_output_count_exact(len(cases), digits)
        receipt["orientation_output_counts"][orientation] = {
            "expected": len(cases),
            "observed": len(digits),
            "exact": output_count_exact,
        }
        all_exact &= output_count_exact
        results = []
        for index, ((a, b, suite_case), output) in enumerate(zip(cases, digits)):
            final_exact = decode(output) == (a * b) % p
            transitions_exact = model.trace_bad[index] == 0
            expected_steps = a.bit_length() + b.bit_length()
            observed_steps = sum(model.phase_counts[index].values())
            step_count_exact = expected_steps == observed_steps
            expected_phase_counts = {
                "reduce_operand": max(a.bit_length(), b.bit_length()),
                "scan_operand": min(a.bit_length(), b.bit_length()),
            }
            phase_count_exact = (
                model.phase_counts[index] == expected_phase_counts
            )
            exact = (
                final_exact
                and transitions_exact
                and step_count_exact
                and phase_count_exact
            )
            all_exact &= exact
            result = {
                "case_index": index,
                "suite_case": suite_case,
                "case_id": case_id(a, b, p),
                "a_bits": a.bit_length(),
                "b_bits": b.bit_length(),
                "p_bits": p.bit_length(),
                "final_exact": final_exact,
                "transitions_exact": transitions_exact,
                "step_count_exact": step_count_exact,
                "phase_count_exact": phase_count_exact,
                "expected_steps": expected_steps,
                "observed_steps": observed_steps,
                "phase_counts": model.phase_counts[index],
                "expected_phase_counts": expected_phase_counts,
                "trajectory_sha256": model.trace_hashes[index].hexdigest(),
                "output_sha256": hashlib.sha256(
                    bytes(int(bit) for bit in output)
                ).hexdigest(),
                "verified_transitions": model.trace_ok[index],
                "failed_transitions": model.trace_bad[index],
                "first_divergence": model.first_divergence[index],
            }
            results.append(result)
            print(
                f"{orientation} case={index} final={final_exact} "
                f"transitions={model.trace_ok[index]}/{observed_steps} "
                f"step_count={step_count_exact}"
            )
        receipt["orientations"][orientation] = results

    swap_trajectory_exact = True
    swap_comparison_count_exact = True
    if args.orientation == "both":
        comparisons = []
        originals = receipt["orientations"]["original"]
        swaps = receipt["orientations"]["swapped"]
        swap_comparison_count_exact = paired_result_counts_exact(
            len(selected), originals, swaps
        )
        all_exact &= swap_comparison_count_exact
        for index, (original, swapped) in enumerate(zip(originals, swaps)):
            trajectory_equal = (
                original["trajectory_sha256"] == swapped["trajectory_sha256"]
            )
            output_equal = original["output_sha256"] == swapped["output_sha256"]
            phase_counts_equal = original["phase_counts"] == swapped["phase_counts"]
            comparison_exact = trajectory_equal and output_equal and phase_counts_equal
            swap_trajectory_exact &= comparison_exact
            comparisons.append({
                "case_index": index,
                "trajectory_equal": trajectory_equal,
                "output_equal": output_equal,
                "phase_counts_equal": phase_counts_equal,
                "exact": comparison_exact,
            })
        receipt["swap_comparisons"] = comparisons
        all_exact &= swap_trajectory_exact

    completed_orientations = len(receipt["orientations"])
    completed_cases = sum(
        len(results) for results in receipt["orientations"].values()
    )
    orientation_count_exact = completed_orientations == len(orientations)
    total_case_count_exact = completed_cases == len(selected) * len(orientations)
    all_exact &= orientation_count_exact and total_case_count_exact
    receipt.update({
        "status": "completed" if all_exact else "failed",
        "all_exact": all_exact,
        "swap_trajectory_exact": swap_trajectory_exact,
        "swap_comparison_count_exact": swap_comparison_count_exact,
        "completed_orientations": completed_orientations,
        "completed_cases": completed_cases,
        "orientation_count_exact": orientation_count_exact,
        "total_case_count_exact": total_case_count_exact,
    })

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(receipt, indent=2) + "\n")
        print(f"receipt={args.json_out}")
    print(f"SUMMARY all_exact={all_exact}")
    return 1 if args.require_exact and not all_exact else 0


if __name__ == "__main__":
    sys.exit(main())
