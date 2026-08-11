#!/usr/bin/env python3
"""Validate the fixed-Hamming four-way weight/schedule causal ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import trace_f11_trajectories as trace
from submission_utils import artifact_identity


CONFIG_ORDER = (
    "v8_original",
    "v8_direct",
    "repaired_original",
    "repaired_direct",
)
CASE_ORDER = (
    "fixed-hamming-source-003",
    "fixed-hamming-source-035",
    "fixed-hamming-source-110",
)
EXPECTED_CONFIGS = {
    "v8_original": {
        "schedule": "original_three_pass",
        "manifest.json": "79a620ae0f0d65bb6b99f44be787491dce09e44d21d8410bb187fa51f2428bfd",
        "model.py": "d7683f7d3a079452ba04296c5eeac16a8665ef1eff921f82eb0fff4c03e7700e",
        "weights.pt": "294fbf809ad42bb0ac66dd51303d21ac5dddeca9a081b5790f1af39d36b21609",
    },
    "v8_direct": {
        "schedule": "direct_two_pass",
        "manifest.json": "84ac7c4dc5a0625169fd39a656322d24cd6836c9acb0cfd6ddda2641e606a969",
        "model.py": "5edb53dae33d6284dbc6c8ccbd64f57c70999c4470b0f3139967de008ab527ff",
        "weights.pt": "294fbf809ad42bb0ac66dd51303d21ac5dddeca9a081b5790f1af39d36b21609",
    },
    "repaired_original": {
        "schedule": "original_three_pass",
        "manifest.json": "279c2bcecce93fceced47cb4eadedd43f59eb6e1d2d9d278b6d0a853d99edce7",
        "model.py": "d7683f7d3a079452ba04296c5eeac16a8665ef1eff921f82eb0fff4c03e7700e",
        "weights.pt": "6d1c8a09e555778fb7961778678a1473e681707ada0d20f1465712313b3e8f01",
        "provenance.json": "71fd4bac01b7ad1e37593a1b21624513123ba136610f650b2876c0d150f2506f",
    },
    "repaired_direct": {
        "schedule": "direct_two_pass",
        "manifest.json": "470aa6427967af96fe18d7ed76b0544a82eb5d84ed7960799ecdb6caecce5ed0",
        "model.py": "5edb53dae33d6284dbc6c8ccbd64f57c70999c4470b0f3139967de008ab527ff",
        "weights.pt": "6d1c8a09e555778fb7961778678a1473e681707ada0d20f1465712313b3e8f01",
        "provenance.json": "9d0dec371ef8c896320fe997e23cd3e71910062736fae1b78d5ceb91f5cb51fe",
    },
}
EXPECTED_TRUTH_TABLE = {
    "fixed-hamming-source-003": (True, True, True, False),
    "fixed-hamming-source-035": (True, False, True, False),
    "fixed-hamming-source-110": (True, True, False, False),
}
TEACHER_DIVERGENCE_FIELDS = (
    "phase",
    "phase_step",
    "global_step",
    "input_bit",
    "state",
    "x",
    "p",
    "expected_value",
    "pre_mod_value",
    "modulus_subtract_count",
    "boundary_distance_to_nearest_modulus_multiple",
    "double_carry_out",
    "add_after_truncated_double_carry_out",
    "full_pre_mod_word_overflow_count",
)


class ValidationError(RuntimeError):
    """Raised when any causal-ablation identity or result gate fails."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    return trace.canonical_json_sha256(value)


def require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def classify_truth_table(values: tuple[bool, bool, bool, bool]) -> str:
    labels = {
        (True, True, True, False): "weight_schedule_interaction_required",
        (True, False, True, False): "direct_schedule_sufficient",
        (True, True, False, False): "repaired_weights_sufficient",
    }
    if values not in labels:
        raise ValidationError(f"unclassified four-way truth table: {values}")
    return labels[values]


def teacher_divergence_signature(divergence: dict[str, Any]) -> str:
    return canonical_json_sha256(
        {field: divergence.get(field) for field in TEACHER_DIVERGENCE_FIELDS}
    )


def summarize_divergence(
    divergence: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if divergence is None:
        return None
    boundary = str(divergence["boundary_distance_to_nearest_modulus_multiple"])
    return {
        "phase": divergence["phase"],
        "phase_step": divergence["phase_step"],
        "global_step": divergence["global_step"],
        "wrong_output_bit_count": divergence["wrong_output_bit_count"],
        "minimum_signed_target_logit_margin": divergence[
            "minimum_signed_target_logit_margin"
        ],
        "minimum_margin_bit_index": divergence["minimum_margin_bit_index"],
        "modulus_subtract_count": divergence["modulus_subtract_count"],
        "boundary_distance_bit_length": int(boundary).bit_length(),
        "boundary_distance_sha256": hashlib.sha256(boundary.encode()).hexdigest(),
        "double_carry_out": divergence["double_carry_out"],
        "add_after_truncated_double_carry_out": divergence[
            "add_after_truncated_double_carry_out"
        ],
        "full_pre_mod_word_overflow_count": divergence[
            "full_pre_mod_word_overflow_count"
        ],
        "teacher_transition_sha256": teacher_divergence_signature(divergence),
    }


def validate(
    receipt_paths: dict[str, Path],
    fixture_path: Path,
) -> dict[str, Any]:
    errors: list[str] = []
    cases, fixture_identity = trace.load_external_case_fixture(fixture_path)
    require(
        tuple(case.label for case in cases) == CASE_ORDER,
        "fixture case order mismatch",
        errors,
    )
    compact_cases = [trace.compact_case_identity(case) for case in cases]
    current_source_identity = trace.qualification_source_identity()

    receipts: dict[str, dict[str, Any]] = {}
    receipt_identity: dict[str, dict[str, str]] = {}
    current_artifacts: dict[str, dict[str, str]] = {}
    for name in CONFIG_ORDER:
        path = receipt_paths[name].resolve()
        if path.is_symlink() or not path.is_file():
            errors.append(f"{name} receipt is missing or symlinked")
            continue
        receipt = json.loads(path.read_text())
        receipts[name] = receipt
        receipt_identity[name] = {"path": str(path), "sha256": sha256_file(path)}
        expected = EXPECTED_CONFIGS[name]
        submission = Path(receipt.get("submission", "")).resolve()
        try:
            observed_artifacts = artifact_identity(submission)
        except Exception as error:
            errors.append(f"{name} artifact identity failed: {error}")
            continue
        current_artifacts[name] = observed_artifacts
        expected_artifacts = {
            key: value for key, value in expected.items() if key != "schedule"
        }

        require(
            observed_artifacts == expected_artifacts,
            f"{name} current artifact identity mismatch",
            errors,
        )
        for field in (
            "artifact_sha256",
            "artifact_sha256_before_load",
            "artifact_sha256_after_load",
            "artifact_sha256_after",
            "artifact_sha256_after_run",
        ):
            require(
                receipt.get(field) == observed_artifacts,
                f"{name} {field} mismatch",
                errors,
            )
        require(
            receipt.get("artifact_unchanged_across_load_and_run") is True,
            f"{name} artifacts changed during execution",
            errors,
        )
        require(
            receipt.get("schedule") == expected["schedule"],
            f"{name} schedule mismatch",
            errors,
        )
        require(
            receipt.get("execution_mode") == "vectorized_exact_prefix_induction",
            f"{name} execution mode mismatch",
            errors,
        )
        require(
            receipt.get("transition_batch_size") == 256,
            f"{name} transition batch size mismatch",
            errors,
        )
        require(
            receipt.get("orientation_request") == "original",
            f"{name} orientation mismatch",
            errors,
        )
        require(
            receipt.get("continue_after_failure") is True,
            f"{name} did not continue after failure",
            errors,
        )
        require(
            receipt.get("stopped_early_on_nonpassing_result") is False,
            f"{name} stopped early",
            errors,
        )
        require(
            receipt.get("case_fixture_identity") == fixture_identity,
            f"{name} fixture identity mismatch",
            errors,
        )
        require(
            receipt.get("selected_cases") == compact_cases,
            f"{name} selected cases mismatch",
            errors,
        )
        require(
            receipt.get("expected_results") == len(CASE_ORDER)
            and receipt.get("completed_results") == len(CASE_ORDER),
            f"{name} result cardinality mismatch",
            errors,
        )
        require(
            receipt.get("result_count_exact") is True
            and receipt.get("transition_cardinality_exact") is True
            and receipt.get("orientation_cardinality_exact") is True,
            f"{name} receipt cardinality gate failed",
            errors,
        )
        require(
            receipt.get("completed_transitions") == receipt.get("expected_transitions"),
            f"{name} transition total mismatch",
            errors,
        )
        require(
            receipt.get("qualification_source_identity") == current_source_identity
            == receipt.get("qualification_source_identity_after_load")
            == receipt.get("qualification_source_identity_after_run"),
            f"{name} qualification source identity mismatch",
            errors,
        )
        require(
            receipt.get("qualification_sources_unchanged_during_run") is True,
            f"{name} qualification sources changed",
            errors,
        )
        require(
            receipt.get("environment", {}).get("device") == "mps"
            and receipt.get("environment", {}).get("default_dtype")
            == "torch.float32",
            f"{name} device/dtype mismatch",
            errors,
        )
        results = receipt.get("results", [])
        require(
            [result.get("label") for result in results] == list(CASE_ORDER),
            f"{name} result order mismatch",
            errors,
        )
        for result in results:
            exact = result.get("all_exact") is True
            require(
                result.get("orientation") == "original",
                f"{name}/{result.get('label')} orientation mismatch",
                errors,
            )
            require(
                result.get("transition_count_exact") is True
                and result.get("phase_counts_exact") is True,
                f"{name}/{result.get('label')} transition structure mismatch",
                errors,
            )
            require(
                result.get("final_exact") is exact,
                f"{name}/{result.get('label')} final/trajectory exactness mismatch",
                errors,
            )
            require(
                (result.get("first_divergence") is None) is exact,
                f"{name}/{result.get('label')} divergence presence mismatch",
                errors,
            )
            margin = result.get("worst_signed_target_logit_margin", {}).get(
                "minimum_signed_target_logit_margin"
            )
            require(
                isinstance(margin, (int, float))
                and ((margin > 0) if exact else (margin < 0)),
                f"{name}/{result.get('label')} margin sign mismatch",
                errors,
            )
            if not exact:
                require(
                    result.get("failed_transitions") == 1,
                    f"{name}/{result.get('label')} expected one failed transition",
                    errors,
                )

    if errors:
        raise ValidationError("; ".join(errors))

    truth_tables: dict[str, tuple[bool, bool, bool, bool]] = {}
    classifications: dict[str, str] = {}
    cases_summary: dict[str, dict[str, Any]] = {}
    for case_label in CASE_ORDER:
        values = tuple(
            next(
                result["all_exact"]
                for result in receipts[name]["results"]
                if result["label"] == case_label
            )
            for name in CONFIG_ORDER
        )
        truth_tables[case_label] = values
        require(
            values == EXPECTED_TRUTH_TABLE[case_label],
            f"{case_label} truth table mismatch",
            errors,
        )
        classifications[case_label] = classify_truth_table(values)
        cases_summary[case_label] = {
            "truth_table": {
                name: value for name, value in zip(CONFIG_ORDER, values)
            },
            "classification": classifications[case_label],
            "configurations": {
                name: {
                    "all_exact": result["all_exact"],
                    "final_exact": result["final_exact"],
                    "failed_transitions": result["failed_transitions"],
                    "verified_transitions": result["verified_transitions"],
                    "observed_transitions": result["observed_transitions"],
                    "worst_signed_target_logit_margin": result[
                        "worst_signed_target_logit_margin"
                    ]["minimum_signed_target_logit_margin"],
                    "first_divergence": summarize_divergence(
                        result["first_divergence"]
                    ),
                }
                for name in CONFIG_ORDER
                for result in receipts[name]["results"]
                if result["label"] == case_label
            },
        }

    direct_case_35 = {
        name: next(
            result["first_divergence"]
            for result in receipts[name]["results"]
            if result["label"] == "fixed-hamming-source-035"
        )
        for name in ("v8_direct", "repaired_direct")
    }
    direct_case_35_signatures = {
        name: teacher_divergence_signature(divergence)
        for name, divergence in direct_case_35.items()
    }
    require(
        len(set(direct_case_35_signatures.values())) == 1,
        "case 35 direct schedules do not fail on the same teacher transition",
        errors,
    )
    require(
        current_artifacts["v8_original"]["weights.pt"]
        == current_artifacts["v8_direct"]["weights.pt"],
        "v8 control weights are not byte-identical",
        errors,
    )
    require(
        current_artifacts["repaired_original"]["weights.pt"]
        == current_artifacts["repaired_direct"]["weights.pt"],
        "repaired control weights are not byte-identical",
        errors,
    )
    if errors:
        raise ValidationError("; ".join(errors))

    return {
        "schema": "neural-horner-fixed-hamming-four-way-validation-v1",
        "status": "validated_causal_ablation",
        "fixture_identity": fixture_identity,
        "configuration_order": list(CONFIG_ORDER),
        "receipt_identity": receipt_identity,
        "artifact_identity": current_artifacts,
        "qualification_source_identity": current_source_identity,
        "case_results": cases_summary,
        "case_35_shared_direct_teacher_transition": {
            "validated": True,
            "teacher_transition_sha256": next(
                iter(direct_case_35_signatures.values())
            ),
            "configuration_signatures": direct_case_35_signatures,
        },
        "scope_nonclaims": [
            "three_frozen_development_counterexamples_only",
            "vectorized_exact_prefix_induction_not_literal_post_divergence_replay",
            "not_a_sealed_generalization_result",
            "not_a_promotion_result",
            "no_training_performed",
        ],
    }


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    resolved = path.resolve()
    if os.path.lexists(resolved):
        raise ValidationError(f"refusing to overwrite validation output: {resolved}")
    resolved.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved.with_name(f".{resolved.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=Path, required=True)
    for name in CONFIG_ORDER:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    receipt_paths = {
        name: getattr(args, name)
        for name in CONFIG_ORDER
    }
    report = validate(receipt_paths, args.fixture)
    report.update(
        {
            "validator": str(Path(__file__).resolve()),
            "validator_sha256": sha256_file(Path(__file__).resolve()),
        }
    )
    write_json_once(args.json_out, report)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValidationError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"four-way validation failed: {error}", file=sys.stderr)
        sys.exit(1)
