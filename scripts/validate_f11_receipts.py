#!/usr/bin/env python3
"""Fail-closed union validation for NeuralHorner F11 qualification receipts."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPECTED_GROUPS = ("decisive", "companions", "ties", "legacy")


class ReceiptValidationError(ValueError):
    """Raised when a receipt union cannot satisfy the qualification contract."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ReceiptValidationError(message)


def require_equal(values: set[Any], expected: Any, label: str) -> None:
    require(values == {expected}, f"{label} mismatch: {sorted(map(str, values))}")


def validate_scorer_identity(identity: dict[str, Any], label: str) -> None:
    scorer = identity.get("scorer")
    require(isinstance(scorer, dict), f"{label} lacks scorer identity")
    require(scorer.get("status") == "verified", f"{label} scorer is not verified")
    require(
        scorer.get("commit_matches_contract") is True,
        f"{label} scorer commit does not match the contract",
    )
    require(
        scorer.get("sources_match_contract") is True,
        f"{label} scorer sources do not match the contract",
    )
    require(
        scorer.get("tracked_files_clean") is True,
        f"{label} scorer checkout is not clean",
    )


def validate_current_identity(receipt: dict[str, Any]) -> dict[str, Any]:
    """Recheck the local sources, artifact, scorer checkout, and import origin."""
    source_identity = receipt["qualification_source_identity"]
    local_sources = source_identity["local_source_sha256"]
    observed_local: dict[str, str] = {}
    for relative_name, expected_digest in local_sources.items():
        relative_path = Path(relative_name)
        require(
            not relative_path.is_absolute() and ".." not in relative_path.parts,
            f"invalid local source path: {relative_name}",
        )
        source_path = PROJECT_ROOT / relative_path
        require(source_path.is_file(), f"missing local source: {relative_name}")
        observed_digest = sha256_file(source_path)
        require(
            observed_digest == expected_digest,
            f"current local source mismatch: {relative_name}",
        )
        observed_local[relative_name] = observed_digest

    scorer = source_identity["scorer"]
    repository = Path(scorer["declared_repository"]).resolve()
    require(repository.is_dir(), f"missing scorer checkout: {repository}")
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(head == scorer["declared_commit"], "current scorer commit mismatch")
    status = subprocess.run(
        ["git", "-C", str(repository), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    require(not status, "current scorer checkout is not clean")
    observed_scorer_sources: dict[str, str] = {}
    for relative_name, expected_digest in scorer["declared_source_sha256"].items():
        source_path = repository / relative_name
        require(source_path.is_file(), f"missing scorer source: {relative_name}")
        observed_digest = sha256_file(source_path)
        require(
            observed_digest == expected_digest,
            f"current scorer source mismatch: {relative_name}",
        )
        observed_scorer_sources[relative_name] = observed_digest

    module_spec = importlib.util.find_spec("modchallenge")
    require(module_spec is not None, "modchallenge import is not discoverable")
    require(module_spec.origin is not None, "modchallenge import lacks an origin")
    import_origin = Path(module_spec.origin).resolve()
    expected_package = (repository / "src" / "modchallenge").resolve()
    require(
        import_origin == expected_package / "__init__.py"
        or expected_package in import_origin.parents,
        "imported modchallenge package is not from the verified scorer checkout",
    )

    submission = Path(receipt["submission"]).resolve()
    require(submission.is_dir(), f"missing submission artifact: {submission}")
    observed_artifact: dict[str, str] = {}
    for relative_name, expected_digest in receipt["artifact_sha256"].items():
        artifact_path = submission / relative_name
        require(artifact_path.is_file(), f"missing artifact file: {relative_name}")
        observed_digest = sha256_file(artifact_path)
        require(
            observed_digest == expected_digest,
            f"current artifact mismatch: {relative_name}",
        )
        observed_artifact[relative_name] = observed_digest

    return {
        "local_source_sha256": observed_local,
        "scorer_commit": head,
        "scorer_source_sha256": observed_scorer_sources,
        "modchallenge_import_origin": str(import_origin),
        "artifact_sha256": observed_artifact,
    }


def validate_receipt_union(
    receipts: list[dict[str, Any]],
    *,
    expected_groups: set[str],
    expected_cases: int,
    expected_results: int,
    expected_transitions: int,
    expected_runner_sha256: str,
    expected_source_set_sha256: str,
    expected_artifact_set_sha256: str,
    expected_full_case_set_sha256: str,
    expected_logit_dtype: str = "torch.float32",
    verify_current_files: bool = True,
) -> dict[str, Any]:
    require(bool(receipts), "at least one receipt is required")

    runner_hashes: set[str] = set()
    source_hashes: set[str] = set()
    artifact_hashes: set[str] = set()
    full_case_hashes: set[str] = set()
    submissions: set[str] = set()
    groups_seen: set[str] = set()
    case_hashes: list[str] = []
    oriented_hashes: list[str] = []
    total_results = 0
    total_transitions = 0
    union_worst_margin: dict[str, Any] | None = None

    for receipt_index, receipt in enumerate(receipts):
        label = f"receipt[{receipt_index}]"
        require(receipt.get("status") == "completed_exact", f"{label} not complete")
        require(receipt.get("all_exact") is True, f"{label} is not exact")
        require(
            receipt.get("stopped_early_on_nonpassing_result") is False,
            f"{label} stopped early",
        )
        require(
            receipt.get("first_nonpassing_result") is None,
            f"{label} contains a nonpassing result",
        )
        for field in (
            "artifact_unchanged_during_load",
            "artifact_unchanged_during_run",
            "artifact_unchanged_across_load_and_run",
            "qualification_sources_unchanged_during_load",
            "qualification_sources_unchanged_during_run",
            "result_count_exact",
            "transition_cardinality_exact",
            "orientation_cardinality_exact",
        ):
            require(receipt.get(field) is True, f"{label} failed {field}")

        runner_hashes.add(receipt["runner_sha256"])
        source_hashes.add(receipt["qualification_source_set_sha256"])
        artifact_hashes.add(receipt["artifact_set_sha256"])
        full_case_hashes.add(receipt["full_case_set_sha256"])
        submissions.add(str(Path(receipt["submission"]).resolve()))

        for phase, identity_key, digest_key in (
            (
                "before_load",
                "qualification_source_identity",
                "qualification_source_set_sha256",
            ),
            (
                "after_load",
                "qualification_source_identity_after_load",
                "qualification_source_set_sha256_after_load",
            ),
            (
                "after_run",
                "qualification_source_identity_after_run",
                "qualification_source_set_sha256_after_run",
            ),
        ):
            identity = receipt.get(identity_key)
            require(isinstance(identity, dict), f"{label} lacks {identity_key}")
            validate_scorer_identity(identity, f"{label}.{phase}")
            require(
                canonical_json_sha256(identity) == receipt.get(digest_key),
                f"{label} has an invalid {phase} source-set digest",
            )

        artifact_before = receipt["artifact_sha256_before_load"]
        require(
            artifact_before == receipt["artifact_sha256_after_load"],
            f"{label} artifact changed during load",
        )
        require(
            artifact_before == receipt["artifact_sha256_after_run"],
            f"{label} artifact changed during execution",
        )
        require(
            canonical_json_sha256(artifact_before) == receipt["artifact_set_sha256"],
            f"{label} artifact-set digest is invalid",
        )

        selected_groups = receipt.get("selected_groups")
        require(
            isinstance(selected_groups, list) and bool(selected_groups),
            f"{label} has no selected groups",
        )
        selected_group_set = set(selected_groups)
        require(
            groups_seen.isdisjoint(selected_group_set),
            f"{label} overlaps an earlier group",
        )
        groups_seen.update(selected_group_set)

        selected_cases = receipt.get("selected_cases")
        results = receipt.get("results")
        require(isinstance(selected_cases, list), f"{label} lacks selected cases")
        require(isinstance(results, list), f"{label} lacks results")
        require(
            receipt["completed_results"] == receipt["expected_results"] == len(results),
            f"{label} result cardinality mismatch",
        )
        require(
            receipt["completed_transitions"] == receipt["expected_transitions"],
            f"{label} transition cardinality mismatch",
        )
        require(
            len(results) == 2 * len(selected_cases),
            f"{label} must contain both orientations for every case",
        )
        require(
            set(receipt["orientation_counts"]) == {"original", "swapped"}
            and set(receipt["orientation_counts"].values()) == {len(selected_cases)},
            f"{label} orientation counts are invalid",
        )

        selected_case_hashes = [item["case_sha256"] for item in selected_cases]
        require(
            len(selected_case_hashes) == len(set(selected_case_hashes)),
            f"{label} duplicates a selected case",
        )
        require(
            all(item.get("group") in selected_group_set for item in selected_cases),
            f"{label} selected case lies outside its declared groups",
        )
        case_hashes.extend(selected_case_hashes)
        expected_case_orientation_pairs = {
            (case_sha256, orientation)
            for case_sha256 in selected_case_hashes
            for orientation in ("original", "swapped")
        }
        observed_case_orientation_pairs: list[tuple[str, str]] = []
        observed_transition_sum = 0
        expected_transition_sum = 0
        for result in results:
            for field in (
                "all_exact",
                "final_exact",
                "candidate_full_rollout_certified",
                "transition_count_exact",
                "phase_counts_exact",
                "phase_order_exact",
                "transitions_exact",
                "strictly_positive_margin",
                "margin_gate_passed",
                "output_count_exact",
                "output_digit_count_exact",
                "output_alphabet_exact",
            ):
                require(result.get(field) is True, f"{label} result failed {field}")
            require(
                result["manifest_route_validation"]["route_exact"] is True,
                f"{label} result route is not exact",
            )
            require(
                result["program_output_validation"]["exact"] is True,
                f"{label} result program output is not exact",
            )
            margin = result["worst_signed_target_logit_margin"]
            require(
                margin["minimum_signed_target_logit_margin"] > 0,
                f"{label} result margin is not positive",
            )
            if (
                union_worst_margin is None
                or margin["minimum_signed_target_logit_margin"]
                < union_worst_margin["minimum_signed_target_logit_margin"]
            ):
                union_worst_margin = {
                    "receipt_index": receipt_index,
                    "group": result["group"],
                    "label": result["label"],
                    "orientation": result["orientation"],
                    "case_sha256": result["case_sha256"],
                    **margin,
                }
            require(
                result["captured_logit_dtypes"] == [expected_logit_dtype],
                f"{label} result logit dtype mismatch",
            )
            require(
                result["observed_transitions"] == result["expected_transitions"],
                f"{label} result transition count mismatch",
            )
            require(
                result["case_sha256"] in selected_case_hashes,
                f"{label} result does not belong to a selected case",
            )
            require(
                result.get("group") in selected_group_set,
                f"{label} result lies outside its declared groups",
            )
            require(
                result.get("orientation") in {"original", "swapped"},
                f"{label} result has an invalid orientation",
            )
            observed_case_orientation_pairs.append(
                (result["case_sha256"], result["orientation"])
            )
            observed_transition_sum += result["observed_transitions"]
            expected_transition_sum += result["expected_transitions"]
            oriented_hashes.append(result["oriented_case_sha256"])

        require(
            len(observed_case_orientation_pairs)
            == len(set(observed_case_orientation_pairs)),
            f"{label} duplicates a case-orientation result",
        )
        require(
            set(observed_case_orientation_pairs) == expected_case_orientation_pairs,
            f"{label} does not cover the complete case-orientation matrix",
        )
        recomputed_orientation_counts = {
            orientation: sum(
                observed_orientation == orientation
                for _case_sha256, observed_orientation in observed_case_orientation_pairs
            )
            for orientation in ("original", "swapped")
        }
        require(
            receipt["orientation_counts"] == recomputed_orientation_counts,
            f"{label} top-level orientation counts are inconsistent",
        )
        require(
            receipt.get("expected_orientations") == 2
            and receipt.get("expected_cases_per_orientation") == len(selected_cases),
            f"{label} expected orientation geometry is inconsistent",
        )
        require(
            observed_transition_sum == receipt["completed_transitions"],
            f"{label} completed transition sum is inconsistent",
        )
        require(
            expected_transition_sum == receipt["expected_transitions"],
            f"{label} expected transition sum is inconsistent",
        )

        require(
            receipt.get("captured_logit_dtypes") == [expected_logit_dtype],
            f"{label} receipt logit dtype mismatch",
        )
        total_results += len(results)
        total_transitions += receipt["completed_transitions"]

    require_equal(runner_hashes, expected_runner_sha256, "runner SHA-256")
    require_equal(source_hashes, expected_source_set_sha256, "source-set SHA-256")
    require_equal(
        artifact_hashes,
        expected_artifact_set_sha256,
        "artifact-set SHA-256",
    )
    require_equal(
        full_case_hashes,
        expected_full_case_set_sha256,
        "full-case-set SHA-256",
    )
    require(len(submissions) == 1, "receipt submissions differ")
    require(groups_seen == expected_groups, f"group coverage mismatch: {groups_seen}")
    require(len(case_hashes) == expected_cases, "selected case count mismatch")
    require(len(set(case_hashes)) == expected_cases, "selected cases are not unique")
    require(total_results == expected_results, "oriented result count mismatch")
    require(
        len(set(oriented_hashes)) == expected_results,
        "oriented cases are not unique",
    )
    require(total_transitions == expected_transitions, "transition total mismatch")
    require(union_worst_margin is not None, "union has no signed-margin record")

    current_identity = validate_current_identity(receipts[0]) if verify_current_files else None
    return {
        "schema": "neural-horner-f11-receipt-union-v1",
        "status": "validated_exact",
        "receipt_count": len(receipts),
        "groups": sorted(groups_seen),
        "unique_cases": len(set(case_hashes)),
        "unique_oriented_cases": len(set(oriented_hashes)),
        "transitions": total_transitions,
        "minimum_signed_target_logit_margin": union_worst_margin[
            "minimum_signed_target_logit_margin"
        ],
        "worst_margin": union_worst_margin,
        "runner_sha256": expected_runner_sha256,
        "qualification_source_set_sha256": expected_source_set_sha256,
        "artifact_set_sha256": expected_artifact_set_sha256,
        "full_case_set_sha256": expected_full_case_set_sha256,
        "submission": next(iter(submissions)),
        "expected_logit_dtype": expected_logit_dtype,
        "current_identity": current_identity,
    }


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", action="append", type=Path, required=True)
    parser.add_argument(
        "--expected-group",
        action="append",
        choices=EXPECTED_GROUPS,
        default=[],
    )
    parser.add_argument("--expected-cases", type=int, required=True)
    parser.add_argument("--expected-results", type=int, required=True)
    parser.add_argument("--expected-transitions", type=int, required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-source-set-sha256", required=True)
    parser.add_argument("--expected-artifact-set-sha256", required=True)
    parser.add_argument("--expected-full-case-set-sha256", required=True)
    parser.add_argument("--expected-logit-dtype", default="torch.float32")
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    receipt_paths = [path.resolve() for path in args.receipt]
    receipts = [json.loads(path.read_text()) for path in receipt_paths]
    report = validate_receipt_union(
        receipts,
        expected_groups=set(args.expected_group or EXPECTED_GROUPS),
        expected_cases=args.expected_cases,
        expected_results=args.expected_results,
        expected_transitions=args.expected_transitions,
        expected_runner_sha256=args.expected_runner_sha256,
        expected_source_set_sha256=args.expected_source_set_sha256,
        expected_artifact_set_sha256=args.expected_artifact_set_sha256,
        expected_full_case_set_sha256=args.expected_full_case_set_sha256,
        expected_logit_dtype=args.expected_logit_dtype,
        verify_current_files=True,
    )
    report["validator_sha256"] = sha256_file(Path(__file__).resolve())
    report["receipt_sha256"] = {
        str(path): sha256_file(path) for path in receipt_paths
    }
    write_json(args.json_out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except ReceiptValidationError as error:
        print(f"receipt validation failed: {error}", file=sys.stderr)
        sys.exit(2)
