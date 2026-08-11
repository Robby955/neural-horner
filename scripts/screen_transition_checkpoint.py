#!/usr/bin/env python3
"""Evaluate a candidate cell on frozen exact transition rows.

This is a one-step diagnostic, not a rollout or competition evaluation.  It is
kept separate from the submission package and binds its receipt to the exact
artifact, runner, input file, selected rows, backend, and expected counts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

from submission_utils import artifact_identity, load_submission, sha256_file


def bits(value: int, width: int) -> list[int]:
    if value < 0 or value >= (1 << width):
        raise ValueError(f"value does not fit in {width} bits")
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def canonical_row_digest(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        payload = json.dumps(
            row,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("cases", type=Path)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--exclude-split", action="append", default=[])
    parser.add_argument("--expected-selected", type=int)
    parser.add_argument("--expected-excluded", type=int)
    parser.add_argument("--require-exact", action="store_true")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if not args.exclude_split:
        parser.error("at least one --exclude-split is required for a sealed screen")
    if args.require_exact and (
        args.expected_selected is None or args.expected_excluded is None
    ):
        parser.error(
            "--require-exact requires --expected-selected and --expected-excluded"
        )

    cases_path = args.cases.resolve()
    all_rows = [
        json.loads(line)
        for line in cases_path.read_text().splitlines()
        if line.strip()
    ]
    excluded_splits = set(args.exclude_split)
    selected = [row for row in all_rows if row.get("split") not in excluded_splits]
    excluded_count = len(all_rows) - len(selected)
    count_errors = []
    if args.expected_selected is not None and len(selected) != args.expected_selected:
        count_errors.append(
            f"selected {len(selected)} rows, expected {args.expected_selected}"
        )
    if args.expected_excluded is not None and excluded_count != args.expected_excluded:
        count_errors.append(
            f"excluded {excluded_count} rows, expected {args.expected_excluded}"
        )
    if not selected:
        count_errors.append("selection is empty")

    submission, manifest, _, model = load_submission(args.submission)
    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable")
    model.device = device
    model.model.to(device)
    model.model.eval()

    correct = 0
    minimum_margin = float("inf")
    failures: list[dict] = []
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in selected:
        groups[int(row["width_bits"])].append(row)

    with torch.inference_mode():
        for width, width_rows in sorted(groups.items()):
            for start in range(0, len(width_rows), args.batch_size):
                chunk = width_rows[start : start + args.batch_size]
                features = torch.tensor(
                    [
                        list(
                            zip(
                                bits(int(row["s_decimal"]), width),
                                bits(int(row["x_decimal"]), width),
                                bits(int(row["p_decimal"]), width),
                            )
                        )
                        for row in chunk
                    ],
                    dtype=torch.float32,
                    device=device,
                )
                digits = torch.tensor(
                    [int(row["d"]) for row in chunk],
                    dtype=torch.long,
                    device=device,
                )
                targets = torch.tensor(
                    [
                        bits(
                            int(row["mathematical_metadata"]["expected_next_decimal"]),
                            width,
                        )
                        for row in chunk
                    ],
                    dtype=torch.long,
                    device=device,
                )
                logits = model.model(features, digits).float()
                predictions = (logits > 0).long()
                row_exact = predictions.eq(targets).all(dim=1)
                correct += int(row_exact.sum().item())
                signed = torch.where(targets.bool(), logits, -logits)
                minimum_margin = min(minimum_margin, float(signed.min().item()))
                for row, is_exact, predicted, target in zip(
                    chunk,
                    row_exact.tolist(),
                    predictions,
                    targets,
                ):
                    if not is_exact and len(failures) < 20:
                        failures.append(
                            {
                                "case_id": row["case_id"],
                                "incorrect_bits": int(predicted.ne(target).sum().item()),
                            }
                        )

    unique_transitions = {
        (
            int(row["width_bits"]),
            row["s_decimal"],
            row["x_decimal"],
            row["p_decimal"],
            int(row["d"]),
            row["mathematical_metadata"]["expected_next_decimal"],
        )
        for row in selected
    }
    all_exact = not count_errors and correct == len(selected)
    receipt = {
        "status": "completed" if all_exact else "failed",
        "scope": "one_step_transition_screen_not_rollout",
        "submission": str(submission),
        "entry_class": manifest["entry_class"],
        "artifact_sha256": artifact_identity(submission),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "cases_path": str(cases_path),
        "cases_sha256": sha256_file(cases_path),
        "selected_rows_sha256": canonical_row_digest(selected),
        "selected_case_ids": [row["case_id"] for row in selected],
        "excluded_splits": sorted(excluded_splits),
        "selected_count": len(selected),
        "excluded_count": excluded_count,
        "expected_selected": args.expected_selected,
        "expected_excluded": args.expected_excluded,
        "count_errors": count_errors,
        "selected_split_counts": dict(
            sorted(Counter(row.get("split", "<missing>") for row in selected).items())
        ),
        "width_counts": dict(
            sorted(Counter(int(row["width_bits"]) for row in selected).items())
        ),
        "unique_transition_tuples": len(unique_transitions),
        "correct": correct,
        "total": len(selected),
        "all_exact": all_exact,
        "minimum_signed_bit_logit_margin": (
            minimum_margin if selected else None
        ),
        "failures": failures,
        "device": str(device),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 1 if args.require_exact and not all_exact else 0


if __name__ == "__main__":
    sys.exit(main())
