from __future__ import annotations

import importlib.util
import json
import sys
import uuid
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_official_eval.py"


def load_runner():
    spec = importlib.util.spec_from_file_location("official_eval_runner_tests", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


RUNNER = load_runner()


def make_submission(root: Path) -> Path:
    submission = root / "submission"
    submission.mkdir()
    (submission / "manifest.json").write_text(
        json.dumps({"entry_class": "model.Submission", "output_base": 2}) + "\n"
    )
    (submission / "model.py").write_text("class Submission:\n    pass\n")
    (submission / "weights.pt").write_bytes(b"fixed-weights")
    (submission / "helper.py").write_text("VALUE = 7\n")
    return submission


def clean_pinned_scorer_identity(root: Path) -> dict[str, Any]:
    return {
        "scorer_package_version": "test-source-tree",
        "scorer_package_dir": str(root / "src" / "modchallenge"),
        "scorer_package_sha256": RUNNER.PINNED_SCORER_PYTHON_TREE_SHA256,
        "scorer_repo": str(root),
        "scorer_sha": RUNNER.PINNED_SCORER_SHA,
        "scorer_git_clean": True,
        "scorer_git_status": [],
        "scorer_git_detached": True,
        "scorer_source_sha256": RUNNER.PINNED_SCORER_PYTHON_TREE_SHA256,
        "scorer_source_matches_import": True,
        "scorer_repo_prohibited_entries": [],
        "scorer_package_prohibited_entries": [],
    }


def public_case_identity() -> dict[str, Any]:
    return {
        "schema": "modchallenge-generated-case-identity-v1",
        "sha256": RUNNER.EXPECTED_PUBLIC_CASES_SHA256,
        "seed_hex": RUNNER.PUBLIC_SEED_HEX,
        "total_cases": 1100,
        "tier_order": list(range(11)),
        "tier_counts": [
            {"tier_id": tier_id, "count": 100} for tier_id in range(11)
        ],
        "case_tier_matches_container": True,
        "decimal_fields_canonical": True,
        "expected_answers_exact": True,
        "scorer_tier_geometry": list(RUNNER.EXPECTED_TIER_GEOMETRY),
        "scorer_multiplication_sub_tiers": [
            list(bounds) for bounds in RUNNER.EXPECTED_MULT_SUB_TIERS
        ],
    }


def perfect_summary() -> dict[str, Any]:
    return {
        "deterministic": True,
        "tiers": [
            {
                "tier_id": tier_id,
                "total": 100,
                "correct": 100,
                "completed": True,
            }
            for tier_id in range(11)
        ],
    }


class FakeResult:
    def __init__(self, summary: dict[str, Any]) -> None:
        self._summary = summary

    def summary(self) -> dict[str, Any]:
        return self._summary


class BrokenSummaryResult:
    @staticmethod
    def summary() -> dict[str, Any]:
        raise RuntimeError("summary finalization failed")


def qualification_argv(submission: Path, receipt: Path, scorer: Path) -> list[str]:
    return [
        str(SCRIPT),
        str(submission),
        "--scorer-repo",
        str(scorer),
        "--expected-scorer-sha",
        RUNNER.PINNED_SCORER_SHA,
        "--json-out",
        str(receipt),
        "--require-scored-perfect",
    ]


def patch_cpu_qualification(
    monkeypatch: pytest.MonkeyPatch,
    scorer_identity: dict[str, Any],
) -> None:
    monkeypatch.setattr(RUNNER, "scorer_identity", lambda _path: scorer_identity)
    monkeypatch.setattr(
        RUNNER,
        "generated_case_identity",
        lambda seed, config: public_case_identity(),
    )
    monkeypatch.setattr(RUNNER.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(RUNNER.torch.backends.mps, "is_available", lambda: False)


def test_artifact_tree_hash_detects_content_change_and_hashes_python_cache(
    tmp_path: Path,
) -> None:
    submission = make_submission(tmp_path)
    before = RUNNER.artifact_tree_identity(submission)

    cache = submission / "__pycache__"
    cache.mkdir()
    (cache / "model.cpython-313.pyc").write_bytes(b"runtime-cache")
    after_cache = RUNNER.artifact_tree_identity(submission)
    assert after_cache["sha256"] != before["sha256"]
    assert after_cache["cache_exclusions"] == []
    (cache / "model.cpython-313.pyc").unlink()
    cache.rmdir()

    (submission / "legacy_helper.pyc").write_bytes(b"submitted-bytecode")
    with_submitted_bytecode = RUNNER.artifact_tree_identity(submission)
    assert with_submitted_bytecode["sha256"] != before["sha256"]
    (submission / "legacy_helper.pyc").unlink()

    (submission / "helper.py").write_text("VALUE = 8\n")
    after_change = RUNNER.artifact_tree_identity(submission)
    assert after_change["sha256"] != before["sha256"]
    assert RUNNER.artifact_tree_difference(before, after_change) == {
        "added": [],
        "removed": [],
        "changed": ["helper.py"],
    }


def test_prohibited_tree_scan_finds_caches_bytecode_and_symlinks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "module.pyc").write_bytes(b"cache")
    (root / "legacy.pyc").write_bytes(b"bytecode")
    (root / "optimized.pyo").write_bytes(b"optimized")
    (root / "target.py").write_text("VALUE = 1\n")
    (root / "alias.py").symlink_to(root / "target.py")

    prohibited = RUNNER.prohibited_tree_entries(root)
    assert {item["kind"] for item in prohibited} == {
        "bytecode_cache",
        "bytecode_file",
        "symlink",
    }
    assert {item["path"] for item in prohibited} == {
        "__pycache__",
        "alias.py",
        "legacy.pyc",
        "optimized.pyo",
    }


def test_wrapper_source_set_binds_runner_and_submission_utils(tmp_path: Path) -> None:
    root = tmp_path / "scripts"
    root.mkdir()
    runner = root / "run_official_eval.py"
    helper = root / "submission_utils.py"
    runner.write_text("RUNNER = 1\n")
    helper.write_text("HELPER = 1\n")

    before = RUNNER.source_set_identity([runner, helper], root=root)
    assert [entry["path"] for entry in before["entries"]] == [
        "run_official_eval.py",
        "submission_utils.py",
    ]
    helper.write_text("HELPER = 2\n")
    after = RUNNER.source_set_identity([runner, helper], root=root)
    assert after["sha256"] != before["sha256"]


def test_generated_case_identity_binds_order_values_and_expected_answers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cases = [
        SimpleNamespace(a="2", b="3", p="5", expected="1", tier_id=0),
        SimpleNamespace(a="4", b="7", p="11", expected="6", tier_id=0),
    ]
    test_set = SimpleNamespace(
        tiers=[SimpleNamespace(tier_id=0, cases=cases)],
    )
    monkeypatch.setattr(
        RUNNER,
        "generate_private_test_set",
        lambda master_seed, config: test_set,
    )
    identity = RUNNER.generated_case_identity(
        b"fixed-seed", RUNNER.EvalConfig(total_problems=11)
    )
    assert identity["total_cases"] == 2
    assert identity["case_tier_matches_container"]
    assert identity["decimal_fields_canonical"]
    assert identity["expected_answers_exact"]

    first = RUNNER.canonical_generated_case_bytes(test_set, b"fixed-seed")
    reversed_set = SimpleNamespace(
        tiers=[SimpleNamespace(tier_id=0, cases=list(reversed(cases)))],
    )
    second = RUNNER.canonical_generated_case_bytes(reversed_set, b"fixed-seed")
    assert first != second


def test_public_case_gate_rejects_hash_and_geometry_drift() -> None:
    identity = public_case_identity()
    assert RUNNER.public_case_identity_errors(identity) == []

    identity["sha256"] = "0" * 64
    identity["tier_counts"][10]["count"] = 99
    errors = RUNNER.public_case_identity_errors(identity)
    assert "generated public case hash differs from the pinned case-set hash" in errors
    assert "generated tier counts are not exactly 100 per tier" in errors


def test_pinned_public_case_hash_and_geometry_regression() -> None:
    identity = RUNNER.generated_case_identity(
        RUNNER.PUBLIC_SEED,
        RUNNER.EvalConfig(total_problems=1100, timeout_seconds=300),
    )
    assert identity["sha256"] == RUNNER.EXPECTED_PUBLIC_CASES_SHA256
    assert RUNNER.public_case_identity_errors(identity) == []


def test_environment_reports_cpu_availability_without_claiming_model_device_or_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(RUNNER.torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(RUNNER.torch.backends.mps, "is_available", lambda: False)
    environment = RUNNER.execution_environment()

    assert environment["cuda_available"] is False
    assert environment["mps_available"] is False
    assert environment["actual_inference_device"] is None
    assert environment["actual_inference_dtype"] is None
    assert environment["model_parameter_dtypes"] is None
    assert "unobserved" in environment["actual_inference_device_status"]
    assert "unobserved" in environment["actual_inference_dtype_status"]
    assert environment["torch_default_dtype"].startswith("torch.")
    expected_hash_input = {
        key: value for key, value in environment.items() if key != "environment_sha256"
    }
    assert environment["environment_sha256"] == RUNNER.canonical_json_sha256(
        expected_hash_input
    )


def test_main_passes_exact_identity_bound_inputs_to_mocked_official_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / "receipt.json"
    scorer_identity = clean_pinned_scorer_identity(scorer)
    patch_cpu_qualification(monkeypatch, scorer_identity)
    calls: list[tuple[Path, bytes, Any]] = []

    def fake_evaluate_local(model_dir: Path, *, master_seed: bytes, config: Any):
        assert RUNNER.sys.dont_write_bytecode is True
        assert RUNNER.os.environ["PYTHONDONTWRITEBYTECODE"] == "1"
        calls.append((model_dir, master_seed, config))
        return FakeResult(perfect_summary())

    monkeypatch.setattr(RUNNER, "evaluate_local", fake_evaluate_local)
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 0
    assert len(calls) == 1
    assert calls[0][0] == submission.resolve()
    assert calls[0][1] == RUNNER.PUBLIC_SEED
    assert calls[0][2].total_problems == 1100
    assert calls[0][2].timeout_seconds == 300
    assert calls[0][2].primes_per_tier == 5
    assert calls[0][2].edge_cases_per_tier == 4
    assert calls[0][2].skip_static_check is False

    data = json.loads(receipt.read_text())
    assert data["status"] == "completed"
    uuid.UUID(data["run_id"])
    assert datetime.fromisoformat(data["started_at_utc"].replace("Z", "+00:00"))
    assert datetime.fromisoformat(data["ended_at_utc"].replace("Z", "+00:00"))
    assert data["artifact_unchanged"]
    assert data["runner_unchanged"]
    assert data["wrapper_source_set_unchanged"]
    assert data["scorer_unchanged"]
    assert data["promotion_gate"]["passed"]
    assert data["promotion_gate"]["generated_public_cases_sha256_exact"]


def test_main_rejects_receipt_inside_submission_without_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission = make_submission(tmp_path)
    before = RUNNER.artifact_tree_identity(submission)
    scorer = tmp_path / "scorer"
    receipt = submission / "receipts" / "must-not-exist.json"
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    assert not receipt.exists()
    assert not receipt.parent.exists()
    assert RUNNER.artifact_tree_identity(submission) == before
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "invalid_receipt_path"
    assert error["receipt_persisted"] is True
    error_path = Path(error["error_receipt_path"])
    assert error_path.is_file()
    assert not RUNNER._path_is_within(error_path, submission)


def test_main_refuses_to_overwrite_existing_external_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    receipt = tmp_path / "existing.json"
    receipt.write_text("preserve-me\n")
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    assert receipt.read_text() == "preserve-me\n"
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "receipt_path_exists"
    assert Path(error["error_receipt_path"]).is_file()


def test_main_refuses_existing_broken_receipt_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    missing_target = tmp_path / "missing-target.json"
    receipt_link = tmp_path / "receipt-link.json"
    receipt_link.symlink_to(missing_target)
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt_link, scorer)
    )

    assert RUNNER.main() == 2
    assert receipt_link.is_symlink()
    assert not missing_target.exists()
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "receipt_path_exists"
    assert Path(error["error_receipt_path"]).is_file()


def test_main_rejects_receipt_inside_scorer_and_persists_error_outside(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = scorer / "receipt.json"
    patch_cpu_qualification(monkeypatch, clean_pinned_scorer_identity(scorer))
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    assert not receipt.exists()
    error = json.loads(capsys.readouterr().out)
    assert error["status"] == "invalid_receipt_path"
    error_path = Path(error["error_receipt_path"])
    assert error_path.is_file()
    assert not RUNNER._path_is_within(error_path, scorer)
    assert not RUNNER._path_is_within(error_path, submission)


@pytest.mark.parametrize("prohibited_kind", ["cache", "pyc", "pyo", "symlink"])
def test_qualification_rejects_preexisting_submission_executable_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    prohibited_kind: str,
) -> None:
    submission = make_submission(tmp_path)
    if prohibited_kind == "cache":
        (submission / "__pycache__").mkdir()
    elif prohibited_kind in {"pyc", "pyo"}:
        (submission / f"legacy.{prohibited_kind}").write_bytes(b"bytecode")
    else:
        (submission / "model_alias.py").symlink_to(submission / "model.py")
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / f"reject-{prohibited_kind}.json"
    patch_cpu_qualification(monkeypatch, clean_pinned_scorer_identity(scorer))
    called = False

    def should_not_run(*args: Any, **kwargs: Any):
        nonlocal called
        called = True
        raise AssertionError("qualification must stop before evaluate_local")

    monkeypatch.setattr(RUNNER, "evaluate_local", should_not_run)
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    assert called is False
    data = json.loads(receipt.read_text())
    assert data["status"] == "qualification_identity_error"
    assert data["submission_prohibited_entries"]
    assert "submission contains a bytecode cache" in data["errors"][0]


def test_qualification_rejects_symlinked_submission_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = make_submission(tmp_path)
    submission_link = tmp_path / "submission-link"
    submission_link.symlink_to(submission, target_is_directory=True)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / "root-symlink-error.json"
    patch_cpu_qualification(monkeypatch, clean_pinned_scorer_identity(scorer))
    monkeypatch.setattr(
        RUNNER,
        "evaluate_local",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must stop before evaluate_local")
        ),
    )
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission_link, receipt, scorer)
    )

    assert RUNNER.main() == 2
    data = json.loads(receipt.read_text())
    assert data["submission_prohibited_entries"][0] == {
        "path": ".",
        "kind": "symlink",
    }


def test_qualification_rejects_scorer_cache_and_pinned_tree_digest_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / "scorer-tree-error.json"
    scorer_identity = clean_pinned_scorer_identity(scorer)
    scorer_identity["scorer_source_sha256"] = "0" * 64
    scorer_identity["scorer_repo_prohibited_entries"] = [
        {"path": "src/modchallenge/__pycache__", "kind": "bytecode_cache"}
    ]
    patch_cpu_qualification(monkeypatch, scorer_identity)
    monkeypatch.setattr(
        RUNNER,
        "evaluate_local",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("must stop before evaluate_local")
        ),
    )
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    data = json.loads(receipt.read_text())
    assert data["status"] == "qualification_identity_error"
    assert "scorer checkout Python tree differs from the pinned digest" in data["errors"]
    assert any("bytecode cache" in error for error in data["errors"])


def test_main_fails_closed_when_mocked_run_mutates_weights(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / "mutation-receipt.json"
    patch_cpu_qualification(monkeypatch, clean_pinned_scorer_identity(scorer))

    def mutating_evaluate_local(model_dir: Path, *, master_seed: bytes, config: Any):
        del master_seed, config
        (model_dir / "weights.pt").write_bytes(b"mutated-weights")
        return FakeResult(perfect_summary())

    monkeypatch.setattr(RUNNER, "evaluate_local", mutating_evaluate_local)
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    data = json.loads(receipt.read_text())
    assert data["status"] == "post_run_identity_error"
    assert not data["artifact_unchanged"]
    assert data["artifact_difference"]["changed"] == ["weights.pt"]
    assert "submission artifact tree changed during evaluation" in data["errors"]
    assert "promotion_gate" not in data


def test_main_persists_external_error_when_run_deletes_critical_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / "deleted-file-receipt.json"
    patch_cpu_qualification(monkeypatch, clean_pinned_scorer_identity(scorer))

    def deleting_evaluate_local(model_dir: Path, *, master_seed: bytes, config: Any):
        del master_seed, config
        (model_dir / "weights.pt").unlink()
        return FakeResult(perfect_summary())

    monkeypatch.setattr(RUNNER, "evaluate_local", deleting_evaluate_local)
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    assert receipt.is_file()
    data = json.loads(receipt.read_text())
    assert data["status"] == "post_run_identity_error"
    assert data["artifact_difference"]["removed"] == ["weights.pt"]
    assert any("critical artifact files" in error for error in data["errors"])
    assert data["post_run_identity_capture"]["artifact_files_error"].startswith(
        "FileNotFoundError:"
    )
    uuid.UUID(data["run_id"])
    assert data["ended_at_utc"].endswith("Z")


def test_main_fails_closed_when_submission_utils_source_identity_changes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / "wrapper-source-change.json"
    patch_cpu_qualification(monkeypatch, clean_pinned_scorer_identity(scorer))
    before_source = RUNNER.wrapper_source_identity()
    after_source = json.loads(json.dumps(before_source))
    helper_entry = next(
        entry
        for entry in after_source["entries"]
        if entry["path"] == "submission_utils.py"
    )
    helper_entry["sha256"] = "0" * 64
    after_source["sha256"] = RUNNER.canonical_json_sha256(
        {key: value for key, value in after_source.items() if key != "sha256"}
    )
    source_identities = iter([before_source, after_source])
    monkeypatch.setattr(
        RUNNER, "wrapper_source_identity", lambda: next(source_identities)
    )
    monkeypatch.setattr(
        RUNNER,
        "evaluate_local",
        lambda *args, **kwargs: FakeResult(perfect_summary()),
    )
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    data = json.loads(receipt.read_text())
    assert data["status"] == "post_run_identity_error"
    assert data["wrapper_source_set_unchanged"] is False
    assert "wrapper source set changed during evaluation" in data["errors"]


def test_main_persists_result_summary_finalization_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / "summary-error.json"
    patch_cpu_qualification(monkeypatch, clean_pinned_scorer_identity(scorer))
    monkeypatch.setattr(
        RUNNER,
        "evaluate_local",
        lambda *args, **kwargs: BrokenSummaryResult(),
    )
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    data = json.loads(receipt.read_text())
    assert data["status"] == "result_summary_error"
    assert data["error_type"] == "RuntimeError"
    assert data["ended_at_utc"].endswith("Z")


def test_main_rejects_dirty_scorer_before_mocked_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    submission = make_submission(tmp_path)
    scorer = tmp_path / "scorer"
    scorer.mkdir()
    receipt = tmp_path / "dirty-receipt.json"
    scorer_identity = clean_pinned_scorer_identity(scorer)
    scorer_identity["scorer_git_clean"] = False
    scorer_identity["scorer_git_status"] = [" M src/modchallenge/config.py"]
    patch_cpu_qualification(monkeypatch, scorer_identity)

    def should_not_run(*args: Any, **kwargs: Any):
        raise AssertionError("official pipeline must not run with a dirty scorer")

    monkeypatch.setattr(RUNNER, "evaluate_local", should_not_run)
    monkeypatch.setattr(
        sys, "argv", qualification_argv(submission, receipt, scorer)
    )

    assert RUNNER.main() == 2
    data = json.loads(receipt.read_text())
    assert data["status"] == "qualification_identity_error"
    assert "scorer checkout is dirty" in data["errors"]
