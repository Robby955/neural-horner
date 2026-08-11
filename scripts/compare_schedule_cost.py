#!/usr/bin/env python3
"""Compare baseline and direct-Horner inference work on a JSONL benchmark.

This is an architecture-level work proxy, not a wall-clock benchmark.  One
learned transition runs a bidirectional GRU over ``effective_width`` state bits,
so ``batch_rows * serial_steps * effective_width`` tracks the dominant recurrent
work while respecting the padding and grouping performed by each implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def bit_length(value: int) -> int:
    return max(1, value.bit_length())


def effective_width(prime: int, maximum_width: int) -> int:
    return min(
        maximum_width,
        max(32, ((prime.bit_length() + 31) // 32) * 32),
    )


def benchmark_sha256(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        relative = path.name.encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def batch_cost(rows: list[dict], maximum_width: int) -> dict[str, int]:
    encoded = []
    for row in rows:
        a = int(row["a"])
        b = int(row["b"])
        p = int(row["p"])
        a_bits = bit_length(a)
        b_bits = bit_length(b)
        if (
            p < 2
            or p >= (1 << maximum_width)
            or a_bits > 2 * maximum_width
            or b_bits > 2 * maximum_width
        ):
            continue
        encoded.append((a_bits, b_bits, effective_width(p, maximum_width)))
    if not encoded:
        return {
            "rows": len(rows),
            "active_rows": 0,
            "abstained_rows": len(rows),
            "effective_width_groups": 0,
            "baseline_serial_step_invocations": 0,
            "direct_serial_step_invocations": 0,
            "baseline_recurrent_bit_work": 0,
            "direct_recurrent_bit_work": 0,
        }
    baseline_width = max(row[2] for row in encoded)
    baseline_steps = (
        max(row[0] for row in encoded)
        + max(row[1] for row in encoded)
        + baseline_width
    )
    baseline_work = len(encoded) * baseline_steps * baseline_width

    groups: dict[int, list[tuple[int, int]]] = {}
    for a_bits, b_bits, width in encoded:
        groups.setdefault(width, []).append(
            (min(a_bits, b_bits), max(a_bits, b_bits))
        )
    direct_steps = 0
    direct_work = 0
    for width, group in groups.items():
        group_steps = max(row[0] for row in group) + max(row[1] for row in group)
        direct_steps += group_steps
        direct_work += len(group) * group_steps * width

    return {
        "rows": len(rows),
        "active_rows": len(encoded),
        "abstained_rows": len(rows) - len(encoded),
        "effective_width_groups": len(groups),
        "baseline_serial_step_invocations": baseline_steps,
        "direct_serial_step_invocations": direct_steps,
        "baseline_recurrent_bit_work": baseline_work,
        "direct_recurrent_bit_work": direct_work,
    }


def aggregate(parts: list[dict[str, int]]) -> dict[str, int | float]:
    baseline_steps = sum(part["baseline_serial_step_invocations"] for part in parts)
    direct_steps = sum(part["direct_serial_step_invocations"] for part in parts)
    baseline_work = sum(part["baseline_recurrent_bit_work"] for part in parts)
    direct_work = sum(part["direct_recurrent_bit_work"] for part in parts)
    return {
        "baseline_serial_step_invocations": baseline_steps,
        "direct_serial_step_invocations": direct_steps,
        "serial_step_reduction": (
            1.0 - direct_steps / baseline_steps if baseline_steps else 0.0
        ),
        "baseline_recurrent_bit_work": baseline_work,
        "direct_recurrent_bit_work": direct_work,
        "recurrent_bit_work_reduction": (
            1.0 - direct_work / baseline_work if baseline_work else 0.0
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("--maximum-width", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--benchmark-git-sha", required=True)
    parser.add_argument("--baseline-source", type=Path, required=True)
    parser.add_argument("--candidate-source", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    baseline_source = args.baseline_source.resolve()
    candidate_source = args.candidate_source.resolve()
    for label, path in (
        ("baseline", baseline_source),
        ("candidate", candidate_source),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label} source not found: {path}")

    paths = sorted(
        args.benchmark.glob("tier_*.jsonl"),
        key=lambda path: int(path.stem.split("_")[-1]),
    )
    if not paths:
        raise FileNotFoundError(f"no tier_*.jsonl files under {args.benchmark}")

    tier_results = []
    scored_parts = []
    all_parts = []
    for path in paths:
        tier_id = int(path.stem.split("_")[-1])
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        parts = [
            batch_cost(rows[start : start + args.batch_size], args.maximum_width)
            for start in range(0, len(rows), args.batch_size)
        ]
        result = {
            "tier_id": tier_id,
            "cases": len(rows),
            "batches": len(parts),
            **aggregate(parts),
            "effective_width_groups": sum(
                part["effective_width_groups"] for part in parts
            ),
            "active_cases": sum(part["active_rows"] for part in parts),
            "abstained_cases": sum(part["abstained_rows"] for part in parts),
        }
        tier_results.append(result)
        all_parts.extend(parts)
        if tier_id > 0:
            scored_parts.extend(parts)
        print(
            f"tier={tier_id:2d} cases={len(rows):3d} "
            f"steps={result['baseline_serial_step_invocations']}->"
            f"{result['direct_serial_step_invocations']} "
            f"work_reduction={result['recurrent_bit_work_reduction']:.3%}"
        )

    receipt = {
        "status": "completed",
        "method": "architecture_work_proxy_not_wall_clock",
        "runner_sha256": file_sha256(Path(__file__).resolve()),
        "baseline_source": str(baseline_source),
        "baseline_source_sha256": file_sha256(baseline_source),
        "candidate_source": str(candidate_source),
        "candidate_source_sha256": file_sha256(candidate_source),
        "benchmark_git_sha": args.benchmark_git_sha,
        "benchmark_sha256": benchmark_sha256(paths),
        "maximum_width": args.maximum_width,
        "batch_size": args.batch_size,
        "tiers": tier_results,
        "scored_tiers": aggregate(scored_parts),
        "all_tiers": aggregate(all_parts),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2) + "\n")
    print(
        "SUMMARY scored_work_reduction="
        f"{receipt['scored_tiers']['recurrent_bit_work_reduction']:.3%} "
        "all_work_reduction="
        f"{receipt['all_tiers']['recurrent_bit_work_reduction']:.3%}"
    )
    print(f"receipt={args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
