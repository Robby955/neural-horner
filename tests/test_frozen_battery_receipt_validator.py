from __future__ import annotations

import importlib.util
from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_frozen_battery_receipt.py"
SPEC = importlib.util.spec_from_file_location("frozen_battery_validator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


EXPECTED = {
    "artifact_set_sha256": "artifact-set",
    "runner_sha256": "runner-hash",
    "helper_sha256": "helper-hash",
    "decoder_sha256": "decoder-hash",
    "scorer_sha": "8" * 40,
    "scorer_python_tree_sha256": "scorer-tree-hash",
    "base_fixture_sha256": "base-fixture",
    "original_fixture_sha256": "original-fixture",
    "swapped_fixture_sha256": "swapped-fixture",
    "oriented_fixture_sha256": "both-fixture",
    "grand_total": 1536,
    "prelaunch_manifest_sha256": "prelaunch-hash",
}


def fixture_statistics() -> dict:
    return {
        "base_row_count": 768,
        "base_unique_numerical_case_count": 693,
        "base_duplicate_numerical_row_count": 75,
        "oriented_row_count": 1536,
        "oriented_unique_effective_numerical_case_count": 1382,
        "oriented_effective_duplicate_row_count": 154,
        "oriented_unique_labelled_numerical_case_count": 1386,
        "oriented_labelled_duplicate_row_count": 150,
        "base_fixture_sha256": EXPECTED["base_fixture_sha256"],
        "oriented_fixture_sha256": {
            "original": EXPECTED["original_fixture_sha256"],
            "swapped": EXPECTED["swapped_fixture_sha256"],
        },
        "combined_oriented_fixture_sha256": EXPECTED["oriented_fixture_sha256"],
    }


def exact_cases() -> list[dict]:
    return [
        {
            "case_index": index,
            "correct": True,
            "output_present": True,
            "output_type_exact": True,
            "output_alphabet_exact": True,
            "malformed_output": None,
        }
        for index in range(128)
    ]


def exact_families() -> list[dict]:
    return [
        {
            "family": family,
            "correct": 128,
            "total": 128,
            "attempted": 128,
            "completed": True,
            "stopped_early": False,
            "failures": [],
            "cases": exact_cases(),
            "outer_output_sized": True,
            "batch_contract_exact": True,
            "output_count": 128,
            "expected_output_count": 128,
            "output_count_exact": True,
        }
        for family in VALIDATOR.EXPECTED_FAMILIES
    ]


def endpoint(
    *,
    scorer_caches: bool = True,
    runner_caches: bool = True,
) -> dict:
    submission = Path("/private/tmp/test-submission")
    runner = Path("/private/tmp/test-sources/held_out_battery.py")
    helper = Path("/private/tmp/test-sources/submission_utils.py")
    scorer = Path("/private/tmp/test-scorer")
    prelaunch = Path("/private/tmp/test-control/prelaunch.json")
    pycache = Path("/private/tmp/test-control/fresh-pycache")
    python = Path("/private/tmp/test-python/bin/python3")
    artifact_files = {
        "manifest.json": "manifest-hash",
        "model.py": "model-hash",
        "weights.pt": "weights-hash",
        "provenance.json": "provenance-hash",
    }
    EXPECTED["artifact_set_sha256"] = VALIDATOR.canonical_json_sha256(artifact_files)
    module_paths = {
        "modchallenge": str(scorer / "src/modchallenge/__init__.py"),
        "modchallenge.evaluation.decoder": str(
            scorer / "src/modchallenge/evaluation/decoder.py"
        ),
        "modchallenge.evaluation.pipeline": str(
            scorer / "src/modchallenge/evaluation/pipeline.py"
        ),
        "modchallenge.interface.base_model": str(
            scorer / "src/modchallenge/interface/base_model.py"
        ),
    }
    result = {
        "submission": str(submission),
        "artifact_files": artifact_files,
        "artifact_tree": {
            "files": artifact_files,
            "symlinks": [],
            "cache_entries": [],
        },
        "runner": {"path": str(runner), "sha256": EXPECTED["runner_sha256"]},
        "helper": {"path": str(helper), "sha256": EXPECTED["helper_sha256"]},
        "runner_tree": {
            "cache_entries": (
                ["__pycache__/held_out_battery.cpython-313.pyc"]
                if runner_caches
                else []
            ),
        },
        "scorer_git": {
            "repository": str(scorer),
            "head": EXPECTED["scorer_sha"],
            "tracked_clean": True,
            "tracked_status": [],
            "detached": True,
        },
        "scorer_python_tree_sha256": EXPECTED["scorer_python_tree_sha256"],
        "scorer_tree": {
            "cache_entries": (
                ["src/modchallenge/__pycache__"] if scorer_caches else []
            ),
        },
        "runtime_probe": {
            "runner_import_path": str(runner),
            "helper_import_path": str(helper),
            "module_paths": module_paths,
            "operand_width": 4096,
            "families": list(VALIDATOR.EXPECTED_FAMILIES),
            "rows_per_family": {family: 128 for family in VALIDATOR.EXPECTED_FAMILIES},
            "fixture_statistics": fixture_statistics(),
        },
        "official_module_sha256": {
            "modchallenge.evaluation.decoder": EXPECTED["decoder_sha256"],
            "modchallenge.evaluation.pipeline": "pipeline-hash",
            "modchallenge.interface.base_model": "base-model-hash",
        },
        "python": {
            "executable": str(python),
            "executable_sha256": "python-hash",
        },
    }
    payload = {
        "schema": VALIDATOR.PRELAUNCH_SCHEMA,
        "run_id": "test-run",
        "created_at_utc": "2026-08-02T00:00:00Z",
        "observation": {
            "paths": {
                "submission": str(submission),
                "runner_root": str(runner.parent),
                "scorer_repo": str(scorer),
                "external_pycache_prefix": str(pycache),
            },
            "python": result["python"],
            "environment": {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(pycache),
                "sys_dont_write_bytecode": True,
                "sys_pycache_prefix": str(pycache),
            },
            "cache_inventory": {
                "submission": [],
                "runner": [],
                "scorer": [],
                "external_pycache_prefix": [],
            },
        },
    }
    result["prelaunch_manifest"] = {
        "path": str(prelaunch),
        "sha256": EXPECTED["prelaunch_manifest_sha256"],
        "read_only": True,
        "payload": payload,
        "external_pycache_prefix_tree": {
            "root": str(pycache),
            "files": {},
            "symlinks": [],
            "cache_entries": [],
        },
    }
    return result


def bound_cache_provenance(current: dict) -> dict:
    manifest = current["prelaunch_manifest"]
    payload = manifest["payload"]
    empty_post = {
        "prelaunch_manifest_sha256": manifest["sha256"],
        "prelaunch_manifest_read_only": True,
        "submission_cache_entries": [],
        "runner_cache_entries": [],
        "scorer_cache_entries": [],
        "external_pycache_prefix_entries": [],
        "bytecode_cache_writes_disabled": True,
    }
    return {
        "schema": "neural-horner-frozen-battery-cache-provenance-v2",
        "verified_before_model_load": True,
        "pre_run_inventory_recorded": True,
        "prelaunch_manifest_path": manifest["path"],
        "prelaunch_manifest_sha256": manifest["sha256"],
        "prelaunch_manifest_expected_sha256": manifest["sha256"],
        "prelaunch_manifest_read_only": True,
        "prelaunch_manifest_payload": payload,
        "submission_cache_entries": [],
        "runner_cache_entries": [],
        "scorer_cache_entries": [],
        "external_pycache_prefix_entries": [],
        "external_pycache_prefix": payload["observation"]["paths"][
            "external_pycache_prefix"
        ],
        "fresh_external_pycache_prefix": True,
        "bytecode_cache_writes_disabled": True,
        "post_run": empty_post,
        "stable_during_run": True,
    }


def receipt(current: dict) -> dict:
    source_identity = {
        "battery_runner": {
            "path": current["runner"]["path"],
            "sha256": current["runner"]["sha256"],
        },
        "submission_utils": {
            "path": current["helper"]["path"],
            "sha256": current["helper"]["sha256"],
        },
    }
    for module_name, digest in current["official_module_sha256"].items():
        source_identity[module_name] = {
            "path": current["runtime_probe"]["module_paths"][module_name],
            "sha256": digest,
        }
    identities = [
        {"name": name, "actual": True, "expected": True, "passed": True}
        for name in sorted(VALIDATOR.REQUIRED_IDENTITY_CHECKS)
    ]
    artifact_files = current["artifact_files"]
    artifact_set = VALIDATOR.canonical_json_sha256(artifact_files)
    orientations = {
        orientation: {
            "status": "completed_exact",
            "correct": 768,
            "total": 768,
            "row_count": 768,
            "families": exact_families(),
        }
        for orientation in VALIDATOR.EXPECTED_ORIENTATIONS
    }
    return {
        "status": "completed_exact",
        "all_exact": True,
        "counts_exact": True,
        "grand_correct": 1536,
        "grand_total": 1536,
        "expected_grand_total": 1536,
        "expected_orientations": 2,
        "expected_cases_per_orientation": 768,
        "identity_mode": "qualification_locked",
        "seed": 20260627,
        "n_per_family": 128,
        "model_L": 2048,
        "manifest_unchanged_during_load": True,
        "artifact_unchanged_during_run": True,
        "selection": {
            "orientation_request": "both",
            "resolved_orientations": list(VALIDATOR.EXPECTED_ORIENTATIONS),
            "family_filters": [],
            "f11_only": False,
        },
        "qualification_identity_checks": identities,
        "orientations": orientations,
        "actual_batching": {
            "never_exceeded_effective_scorer_batch_size": True,
            "maximum_actual_batch_size": 128,
        },
        "effective_scorer_batch_size": 256,
        "submission": current["submission"],
        "artifact_sha256": artifact_files,
        "artifact_sha256_after": artifact_files,
        "artifact_set_sha256": artifact_set,
        "artifact_set_sha256_after": artifact_set,
        "runner_sha256": EXPECTED["runner_sha256"],
        "helper_runner_sha256": EXPECTED["helper_sha256"],
        "source_identity": source_identity,
        "source_identity_sha256": VALIDATOR.canonical_json_sha256(source_identity),
        "official_scorer_git_commit": EXPECTED["scorer_sha"],
        "official_scorer_git_state": {
            "commit": EXPECTED["scorer_sha"],
            "clean": True,
            "detached_head": True,
        },
        "official_decoder_sha256": EXPECTED["decoder_sha256"],
        "base_battery_sha256": EXPECTED["base_fixture_sha256"],
        "selected_battery_sha256": EXPECTED["base_fixture_sha256"],
        "selected_oriented_battery_sha256": EXPECTED["oriented_fixture_sha256"],
        "fixture_statistics": fixture_statistics(),
    }


def test_exact_result_remains_valid_when_cache_provenance_is_not_established() -> None:
    current = endpoint(scorer_caches=True)
    report = VALIDATOR.validate_completed_receipt(receipt(current), current, EXPECTED)

    assert report["status"] == "validated_development_exact"
    assert report["result_validation"] == "validated_exact"
    assert report["qualification_validation"] == ("not_established_cache_provenance")
    assert not report["promotion_eligible"]
    assert report["exactness_errors"] == []
    assert report["identity_errors"] == []
    assert report["cache_provenance_blockers"] == [
        "receipt_does_not_bind_verified_prelaunch_cache_manifest",
        "runner_source_tree_contains_bytecode_cache_at_validation",
        "scorer_checkout_contains_bytecode_cache_at_validation",
    ]


def test_cache_free_receipt_with_pre_run_inventory_can_qualify() -> None:
    current = endpoint(scorer_caches=False, runner_caches=False)
    data = receipt(current)
    data["cache_provenance"] = bound_cache_provenance(current)

    report = VALIDATOR.validate_completed_receipt(data, current, EXPECTED)

    assert report["result_validation"] == "validated_exact"
    assert report["qualification_validation"] == "validated"
    assert report["promotion_eligible"]
    assert report["cache_provenance_blockers"] == []


def test_unbound_cache_booleans_cannot_qualify() -> None:
    current = endpoint(scorer_caches=False, runner_caches=False)
    data = receipt(current)
    data["cache_provenance"] = {
        "pre_run_inventory_recorded": True,
        "submission_cache_entries": [],
        "runner_cache_entries": [],
        "scorer_cache_entries": [],
        "fresh_external_pycache_prefix": True,
        "bytecode_cache_writes_disabled": True,
    }

    report = VALIDATOR.validate_completed_receipt(data, current, EXPECTED)

    assert report["result_validation"] == "validated_exact"
    assert report["qualification_validation"] == ("not_established_cache_provenance")
    assert not report["promotion_eligible"]
    assert report["cache_provenance_blockers"] == [
        "receipt_does_not_bind_verified_prelaunch_cache_manifest"
    ]


def test_changed_prelaunch_manifest_hash_cannot_qualify() -> None:
    current = endpoint(scorer_caches=False, runner_caches=False)
    data = receipt(current)
    data["cache_provenance"] = bound_cache_provenance(current)
    current = deepcopy(current)
    current["prelaunch_manifest"]["sha256"] = "changed-prelaunch"

    report = VALIDATOR.validate_completed_receipt(data, current, EXPECTED)

    assert report["result_validation"] == "validated_exact"
    assert not report["promotion_eligible"]
    assert report["cache_provenance_blockers"] == [
        "receipt_does_not_bind_verified_prelaunch_cache_manifest"
    ]


def test_running_or_incomplete_receipt_is_not_accepted() -> None:
    current = endpoint(scorer_caches=False)
    data = receipt(current)
    data["status"] = "running"
    data["all_exact"] = False

    report = VALIDATOR.validate_completed_receipt(data, current, EXPECTED)

    assert report["status"] == "failed"
    assert report["result_validation"] == "failed"
    assert "receipt is not terminal completed_exact" in report["exactness_errors"]
    assert "receipt all_exact is not true" in report["exactness_errors"]


def test_current_source_mismatch_invalidates_result_identity() -> None:
    current = endpoint(scorer_caches=False)
    data = receipt(current)
    current = deepcopy(current)
    current["runner"]["sha256"] = "changed-runner"

    report = VALIDATOR.validate_completed_receipt(data, current, EXPECTED)

    assert report["result_validation"] == "failed"
    assert "current runner SHA-256 mismatch" in report["identity_errors"]
    assert "receipt battery_runner hash mismatch" in report["identity_errors"]


def test_one_inexact_case_row_invalidates_exactness() -> None:
    current = endpoint(scorer_caches=False)
    data = receipt(current)
    data["orientations"]["swapped"]["families"][0]["cases"][0]["correct"] = False

    report = VALIDATOR.validate_completed_receipt(data, current, EXPECTED)

    assert report["result_validation"] == "failed"
    assert any(
        "contains a non-exact or malformed case row" in error
        for error in report["exactness_errors"]
    )
