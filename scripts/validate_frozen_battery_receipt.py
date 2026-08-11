#!/usr/bin/env python3
"""Validate a completed frozen-battery receipt without rerunning inference.

Invoke this once, only after the battery process has exited.  The validator
rehashes the terminal receipt, immutable candidate files, frozen runner/helper,
scorer source tree and import origins, and deterministically regenerated fixture.

Result exactness and execution-cache provenance are intentionally separate.  A
terminal exact receipt can remain useful development evidence even when the
original runner did not prove that it could not read pre-existing bytecode.  In
that case ``result_validation`` is ``validated_exact`` while
``qualification_validation`` is ``not_established_cache_provenance``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any


EXPECTED_FAMILIES = (
    "fibonacci values",
    "fermat numbers",
    "alternating bits",
    "fixed Hamming weight W/2",
    "product straddles k*p",
    "prime=3mod4 (random ops)",
)
EXPECTED_ORIENTATIONS = ("original", "swapped")
REQUIRED_IDENTITY_CHECKS = frozenset(
    {
        "official_scorer_git_commit",
        "official_scorer_git_checkout_available",
        "official_scorer_git_checkout_clean",
        "official_scorer_git_checkout_detached",
        "qualification_n_per_family",
        "qualification_orientation_request",
        "qualification_family_filters",
        "qualification_f11_only",
        "qualification_model_L",
        "base_battery_sha256",
        "selected_oriented_battery_sha256",
        "original_oriented_fixture_sha256",
        "swapped_oriented_fixture_sha256",
        "base_row_count",
        "base_unique_numerical_case_count",
        "oriented_row_count",
        "oriented_unique_effective_numerical_case_count",
        "oriented_unique_labelled_numerical_case_count",
        "prelaunch_manifest_read_only",
        "prelaunch_manifest_sha256",
        "prelaunch_manifest_schema",
        "prelaunch_manifest_metadata",
        "prelaunch_manifest_path_external",
        "prelaunch_manifest_bound_observation",
        "prelaunch_cache_inventory_empty",
        "prelaunch_bytecode_controls",
    }
)
CRITICAL_ARTIFACT_FILES = ("manifest.json", "model.py", "weights.pt")
PRELAUNCH_SCHEMA = "neural-horner-frozen-battery-prelaunch-v1"


class ReceiptValidationError(RuntimeError):
    """Raised for malformed arguments or an unusable validation boundary."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def python_tree_sha256(root: Path) -> str:
    root = root.resolve()
    sources = sorted(root.rglob("*.py"))
    if not sources:
        raise ReceiptValidationError(f"no Python sources found under {root}")
    digest = hashlib.sha256()
    for source in sources:
        if source.is_symlink() or not source.is_file():
            raise ReceiptValidationError(f"invalid Python source: {source}")
        relative = source.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(source.read_bytes())
    return digest.hexdigest()


def artifact_identity(submission: Path) -> dict[str, str]:
    submission = submission.resolve()
    names = list(CRITICAL_ARTIFACT_FILES)
    if (submission / "provenance.json").is_file():
        names.append("provenance.json")
    identity: dict[str, str] = {}
    for name in names:
        path = submission / name
        if path.is_symlink() or not path.is_file():
            raise ReceiptValidationError(f"missing or symlinked artifact file: {path}")
        identity[name] = sha256_file(path)
    return identity


def tree_inventory(
    root: Path,
    *,
    ignored_top_level: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise ReceiptValidationError(f"tree is missing or symlinked: {root}")
    files: dict[str, str] = {}
    symlinks: list[str] = []
    cache_entries: list[str] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        if current_path == root:
            directory_names[:] = [
                name for name in directory_names if name not in ignored_top_level
            ]
        retained = []
        for name in sorted(directory_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                symlinks.append(relative)
            else:
                retained.append(name)
                if name == "__pycache__":
                    cache_entries.append(relative)
        directory_names[:] = retained
        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                symlinks.append(relative)
                continue
            if not path.is_file():
                continue
            files[relative] = sha256_file(path)
            if path.suffix.lower() in {".pyc", ".pyo"}:
                cache_entries.append(relative)
    return {
        "root": str(root),
        "files": files,
        "file_set_sha256": canonical_json_sha256(files),
        "symlinks": sorted(symlinks),
        "cache_entries": sorted(set(cache_entries)),
    }


def prelaunch_manifest_identity(path: Path) -> dict[str, Any]:
    requested = path if path.is_absolute() else Path.cwd() / path
    if requested.is_symlink() or not requested.is_file():
        raise ReceiptValidationError(
            f"prelaunch manifest is missing or symlinked: {requested}"
        )
    resolved = requested.resolve()
    try:
        payload = json.loads(resolved.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReceiptValidationError(
            f"could not parse prelaunch manifest: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ReceiptValidationError("prelaunch manifest must contain one JSON object")
    prefix_value = (
        payload.get("observation", {}).get("paths", {}).get("external_pycache_prefix")
    )
    if not isinstance(prefix_value, str) or not prefix_value:
        raise ReceiptValidationError(
            "prelaunch manifest external pycache prefix is missing"
        )
    prefix = Path(prefix_value)
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "read_only": resolved.stat().st_mode & 0o222 == 0,
        "payload": payload,
        "external_pycache_prefix_tree": tree_inventory(prefix),
    }


def _git(
    repository: Path, *arguments: str, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=check,
        capture_output=True,
        text=True,
    )


def scorer_git_identity(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    head = _git(repository, "rev-parse", "HEAD").stdout.strip()
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=no",
    ).stdout.splitlines()
    symbolic = _git(repository, "symbolic-ref", "-q", "HEAD", check=False)
    return {
        "repository": str(repository),
        "head": head,
        "tracked_clean": not status,
        "tracked_status": status,
        "detached": symbolic.returncode != 0,
    }


def runtime_probe(
    *,
    runner: Path,
    scorer_repo: Path,
    python_executable: Path,
) -> dict[str, Any]:
    """Import fresh source paths and regenerate the fixture in a child process."""
    runner = runner.resolve()
    scorer_repo = scorer_repo.resolve()
    scorer_src = scorer_repo / "src"
    code = r"""
import importlib
import json
from pathlib import Path

import held_out_battery as battery
import submission_utils as helper

modules = {}
for name in (
    "modchallenge",
    "modchallenge.evaluation.decoder",
    "modchallenge.evaluation.pipeline",
    "modchallenge.interface.base_model",
):
    module = importlib.import_module(name)
    modules[name] = str(Path(module.__file__).resolve())

p, p2, operand_width, categories = battery.generate_battery(
    2048, 128, seed=20260627
)
selected = [
    (name, p2 if name.startswith("prime=3mod4") else p, cases)
    for name, cases in categories.items()
]
statistics = battery.fixture_statistics(selected, ("original", "swapped"))
print(json.dumps({
    "runner_import_path": str(Path(battery.__file__).resolve()),
    "helper_import_path": str(Path(helper.__file__).resolve()),
    "module_paths": modules,
    "operand_width": operand_width,
    "families": list(categories),
    "rows_per_family": {name: len(cases) for name, cases in categories.items()},
    "fixture_statistics": statistics,
}, sort_keys=True))
"""
    with tempfile.TemporaryDirectory(
        prefix="neural-horner-battery-validator-"
    ) as temporary:
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPYCACHEPREFIX": str(Path(temporary) / "fresh-pycache"),
                "PYTHONPATH": os.pathsep.join((str(scorer_src), str(runner.parent))),
            }
        )
        completed = subprocess.run(
            [str(python_executable), "-B", "-c", code],
            check=True,
            capture_output=True,
            text=True,
            cwd=temporary,
            env=environment,
        )
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise ReceiptValidationError(
            "runtime fixture/import probe did not emit one JSON object: "
            + completed.stdout[-1000:]
        ) from error


def collect_endpoint_identity(
    *,
    submission: Path,
    runner: Path,
    helper: Path,
    scorer_repo: Path,
    python_executable: Path,
    prelaunch_manifest: Path,
) -> dict[str, Any]:
    submission = submission.resolve()
    runner = runner.resolve()
    helper = helper.resolve()
    scorer_repo = scorer_repo.resolve()
    for source in (runner, helper):
        if source.is_symlink() or not source.is_file():
            raise ReceiptValidationError(f"missing or symlinked source: {source}")

    scorer_package = scorer_repo / "src" / "modchallenge"
    probe = runtime_probe(
        runner=runner,
        scorer_repo=scorer_repo,
        python_executable=python_executable,
    )
    module_hashes = {
        name: sha256_file(Path(path))
        for name, path in probe["module_paths"].items()
        if name != "modchallenge"
    }
    return {
        "submission": str(submission),
        "artifact_files": artifact_identity(submission),
        "artifact_tree": tree_inventory(submission),
        "runner": {"path": str(runner), "sha256": sha256_file(runner)},
        "helper": {"path": str(helper), "sha256": sha256_file(helper)},
        "runner_tree": tree_inventory(runner.parent),
        "scorer_git": scorer_git_identity(scorer_repo),
        "scorer_python_tree_sha256": python_tree_sha256(scorer_package),
        "scorer_tree": tree_inventory(
            scorer_repo,
            ignored_top_level=frozenset({".git"}),
        ),
        "runtime_probe": probe,
        "official_module_sha256": module_hashes,
        "prelaunch_manifest": prelaunch_manifest_identity(prelaunch_manifest),
        "python": {
            "executable": str(python_executable.resolve()),
            "executable_sha256": sha256_file(python_executable.resolve()),
        },
    }


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def _path_equal(left: object, right: Path) -> bool:
    try:
        return Path(str(left)).resolve() == right.resolve()
    except (OSError, TypeError, ValueError):
        return False


def validate_completed_receipt(
    receipt: dict[str, Any],
    endpoint: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    exactness_errors: list[str] = []
    identity_errors: list[str] = []

    expected_total = int(expected["grand_total"])
    per_orientation = expected_total // len(EXPECTED_ORIENTATIONS)
    per_family = per_orientation // len(EXPECTED_FAMILIES)

    _require(
        receipt.get("status") == "completed_exact",
        "receipt is not terminal completed_exact",
        exactness_errors,
    )
    _require(
        receipt.get("all_exact") is True,
        "receipt all_exact is not true",
        exactness_errors,
    )
    _require(
        receipt.get("counts_exact") is True,
        "receipt counts_exact is not true",
        exactness_errors,
    )
    _require(
        receipt.get("grand_correct") == expected_total,
        "grand_correct mismatch",
        exactness_errors,
    )
    _require(
        receipt.get("grand_total") == expected_total,
        "grand_total mismatch",
        exactness_errors,
    )
    _require(
        receipt.get("expected_grand_total") == expected_total,
        "expected_grand_total mismatch",
        exactness_errors,
    )
    _require(
        receipt.get("expected_orientations") == 2,
        "expected orientation count mismatch",
        exactness_errors,
    )
    _require(
        receipt.get("expected_cases_per_orientation") == per_orientation,
        "expected cases per orientation mismatch",
        exactness_errors,
    )
    _require(
        receipt.get("identity_mode") == "qualification_locked",
        "receipt is not qualification_locked",
        identity_errors,
    )
    _require(receipt.get("seed") == 20260627, "battery seed mismatch", identity_errors)
    _require(
        receipt.get("n_per_family") == per_family,
        "n_per_family mismatch",
        identity_errors,
    )
    _require(receipt.get("model_L") == 2048, "model L mismatch", identity_errors)
    _require(
        receipt.get("manifest_unchanged_during_load") is True,
        "manifest changed during load",
        identity_errors,
    )
    _require(
        receipt.get("artifact_unchanged_during_run") is True,
        "artifact changed during run",
        identity_errors,
    )

    selection = receipt.get("selection", {})
    _require(
        selection.get("orientation_request") == "both",
        "orientation request is not both",
        identity_errors,
    )
    _require(
        selection.get("resolved_orientations") == list(EXPECTED_ORIENTATIONS),
        "resolved orientations mismatch",
        identity_errors,
    )
    _require(
        selection.get("family_filters") == [],
        "family filters were applied",
        identity_errors,
    )
    _require(
        selection.get("f11_only") is False,
        "F11-only filter was applied",
        identity_errors,
    )

    checks = receipt.get("qualification_identity_checks")
    if not isinstance(checks, list):
        identity_errors.append("qualification identity checks are missing")
        check_names: set[str] = set()
    else:
        check_names = {
            str(check.get("name")) for check in checks if isinstance(check, dict)
        }
        _require(
            all(
                isinstance(check, dict) and check.get("passed") is True
                for check in checks
            ),
            "one or more qualification identity checks failed",
            identity_errors,
        )
    missing_checks = sorted(REQUIRED_IDENTITY_CHECKS - check_names)
    _require(
        not missing_checks,
        f"required identity checks missing: {missing_checks}",
        identity_errors,
    )

    orientations = receipt.get("orientations")
    if not isinstance(orientations, dict):
        exactness_errors.append("orientation results are missing")
        orientations = {}
    _require(
        set(orientations) == set(EXPECTED_ORIENTATIONS),
        "orientation result set mismatch",
        exactness_errors,
    )
    for orientation in EXPECTED_ORIENTATIONS:
        result = orientations.get(orientation, {})
        _require(
            result.get("status") == "completed_exact",
            f"{orientation} did not complete exactly",
            exactness_errors,
        )
        _require(
            result.get("correct") == per_orientation,
            f"{orientation} correct count mismatch",
            exactness_errors,
        )
        _require(
            result.get("total") == per_orientation,
            f"{orientation} total mismatch",
            exactness_errors,
        )
        _require(
            result.get("row_count") == per_orientation,
            f"{orientation} row count mismatch",
            exactness_errors,
        )
        families = result.get("families")
        if not isinstance(families, list):
            exactness_errors.append(f"{orientation} family results are missing")
            continue
        _require(
            [family.get("family") for family in families if isinstance(family, dict)]
            == list(EXPECTED_FAMILIES),
            f"{orientation} family order/set mismatch",
            exactness_errors,
        )
        for family in families:
            if not isinstance(family, dict):
                exactness_errors.append(
                    f"{orientation} contains a malformed family result"
                )
                continue
            label = f"{orientation}/{family.get('family', '<missing>')}"
            _require(
                family.get("correct") == per_family,
                f"{label} correct count mismatch",
                exactness_errors,
            )
            _require(
                family.get("total") == per_family,
                f"{label} total mismatch",
                exactness_errors,
            )
            _require(
                family.get("attempted") == per_family,
                f"{label} attempted count mismatch",
                exactness_errors,
            )
            _require(
                family.get("completed") is True,
                f"{label} is incomplete",
                exactness_errors,
            )
            _require(
                family.get("stopped_early") is False,
                f"{label} stopped early",
                exactness_errors,
            )
            _require(
                family.get("failures") == [],
                f"{label} records failures",
                exactness_errors,
            )
            _require(
                family.get("output_count") == per_family,
                f"{label} output count mismatch",
                exactness_errors,
            )
            _require(
                family.get("expected_output_count") == per_family,
                f"{label} expected output count mismatch",
                exactness_errors,
            )
            _require(
                family.get("output_count_exact") is True,
                f"{label} output count is not exact",
                exactness_errors,
            )
            _require(
                family.get("batch_contract_exact") is True,
                f"{label} batch contract failed",
                exactness_errors,
            )
            _require(
                family.get("outer_output_sized") is True,
                f"{label} outer output was not sized",
                exactness_errors,
            )
            cases = family.get("cases")
            if not isinstance(cases, list):
                exactness_errors.append(f"{label} case rows are missing")
                continue
            _require(
                len(cases) == per_family,
                f"{label} case-row count mismatch",
                exactness_errors,
            )
            _require(
                [case.get("case_index") for case in cases if isinstance(case, dict)]
                == list(range(per_family)),
                f"{label} case indexes are incomplete or out of order",
                exactness_errors,
            )
            _require(
                all(
                    isinstance(case, dict)
                    and case.get("correct") is True
                    and case.get("output_present") is True
                    and case.get("output_type_exact") is True
                    and case.get("output_alphabet_exact") is True
                    and case.get("malformed_output") is None
                    for case in cases
                ),
                f"{label} contains a non-exact or malformed case row",
                exactness_errors,
            )

    batching = receipt.get("actual_batching", {})
    _require(
        batching.get("never_exceeded_effective_scorer_batch_size") is True,
        "batch size exceeded scorer contract",
        exactness_errors,
    )
    effective_batch_size = receipt.get("effective_scorer_batch_size")
    maximum_batch_size = batching.get("maximum_actual_batch_size")
    _require(
        isinstance(effective_batch_size, int)
        and isinstance(maximum_batch_size, int)
        and maximum_batch_size <= effective_batch_size,
        "observed maximum batch size is invalid",
        exactness_errors,
    )

    current_artifact = endpoint["artifact_files"]
    current_artifact_set = canonical_json_sha256(current_artifact)
    _require(
        current_artifact_set == expected["artifact_set_sha256"],
        "current artifact-set SHA-256 mismatch",
        identity_errors,
    )
    _require(
        receipt.get("artifact_sha256") == current_artifact,
        "receipt pre-load artifact identity mismatch",
        identity_errors,
    )
    _require(
        receipt.get("artifact_sha256_after") == current_artifact,
        "receipt post-run artifact identity mismatch",
        identity_errors,
    )
    _require(
        receipt.get("artifact_set_sha256") == current_artifact_set,
        "receipt artifact-set digest mismatch",
        identity_errors,
    )
    _require(
        receipt.get("artifact_set_sha256_after") == current_artifact_set,
        "receipt post-run artifact-set digest mismatch",
        identity_errors,
    )
    _require(
        _path_equal(receipt.get("submission"), Path(endpoint["submission"])),
        "receipt submission path mismatch",
        identity_errors,
    )
    artifact_tree = endpoint["artifact_tree"]
    _require(
        not artifact_tree["symlinks"], "submission contains symlinks", identity_errors
    )
    _require(
        set(artifact_tree["files"]) == set(current_artifact),
        "submission tree contains unexpected files",
        identity_errors,
    )

    _require(
        endpoint["runner"]["sha256"] == expected["runner_sha256"],
        "current runner SHA-256 mismatch",
        identity_errors,
    )
    _require(
        endpoint["helper"]["sha256"] == expected["helper_sha256"],
        "current helper SHA-256 mismatch",
        identity_errors,
    )
    _require(
        receipt.get("runner_sha256") == expected["runner_sha256"],
        "receipt runner SHA-256 mismatch",
        identity_errors,
    )
    _require(
        receipt.get("helper_runner_sha256") == expected["helper_sha256"],
        "receipt helper SHA-256 mismatch",
        identity_errors,
    )
    source_identity = receipt.get("source_identity")
    if not isinstance(source_identity, dict):
        identity_errors.append("receipt source identity is missing")
        source_identity = {}
    _require(
        receipt.get("source_identity_sha256") == canonical_json_sha256(source_identity),
        "receipt source-identity digest mismatch",
        identity_errors,
    )
    for key, endpoint_key in (
        ("battery_runner", "runner"),
        ("submission_utils", "helper"),
    ):
        recorded = source_identity.get(key, {})
        _require(
            recorded.get("sha256") == endpoint[endpoint_key]["sha256"],
            f"receipt {key} hash mismatch",
            identity_errors,
        )
        _require(
            _path_equal(recorded.get("path"), Path(endpoint[endpoint_key]["path"])),
            f"receipt {key} path mismatch",
            identity_errors,
        )

    scorer_git = endpoint["scorer_git"]
    _require(
        scorer_git["head"] == expected["scorer_sha"],
        "scorer HEAD mismatch",
        identity_errors,
    )
    _require(
        scorer_git["tracked_clean"] is True,
        "scorer tracked tree is dirty",
        identity_errors,
    )
    _require(
        scorer_git["detached"] is True,
        "scorer checkout is not detached",
        identity_errors,
    )
    _require(
        endpoint["scorer_python_tree_sha256"] == expected["scorer_python_tree_sha256"],
        "scorer Python-tree digest mismatch",
        identity_errors,
    )
    recorded_git = receipt.get("official_scorer_git_state", {})
    _require(
        receipt.get("official_scorer_git_commit") == expected["scorer_sha"],
        "receipt scorer commit mismatch",
        identity_errors,
    )
    _require(
        recorded_git.get("commit") == expected["scorer_sha"],
        "receipt scorer git-state commit mismatch",
        identity_errors,
    )
    _require(
        recorded_git.get("clean") is True,
        "receipt scorer checkout was not clean",
        identity_errors,
    )
    _require(
        recorded_git.get("detached_head") is True,
        "receipt scorer checkout was not detached",
        identity_errors,
    )

    probe = endpoint["runtime_probe"]
    _require(
        _path_equal(probe.get("runner_import_path"), Path(endpoint["runner"]["path"])),
        "fixture probe imported a different runner",
        identity_errors,
    )
    _require(
        _path_equal(probe.get("helper_import_path"), Path(endpoint["helper"]["path"])),
        "fixture probe imported a different helper",
        identity_errors,
    )
    _require(
        probe.get("operand_width") == 4096,
        "regenerated operand width mismatch",
        identity_errors,
    )
    scorer_root = Path(scorer_git["repository"]) / "src" / "modchallenge"
    for module_name, module_path in probe.get("module_paths", {}).items():
        try:
            Path(module_path).resolve().relative_to(scorer_root.resolve())
        except (ValueError, OSError):
            identity_errors.append(
                f"{module_name} imported outside the pinned scorer tree"
            )
    module_hashes = endpoint["official_module_sha256"]
    _require(
        module_hashes.get("modchallenge.evaluation.decoder")
        == expected["decoder_sha256"],
        "current decoder SHA-256 mismatch",
        identity_errors,
    )
    _require(
        receipt.get("official_decoder_sha256") == expected["decoder_sha256"],
        "receipt decoder SHA-256 mismatch",
        identity_errors,
    )
    for module_name in (
        "modchallenge.evaluation.decoder",
        "modchallenge.evaluation.pipeline",
        "modchallenge.interface.base_model",
    ):
        recorded = source_identity.get(module_name, {})
        _require(
            recorded.get("sha256") == module_hashes.get(module_name),
            f"receipt {module_name} hash mismatch",
            identity_errors,
        )
        _require(
            _path_equal(
                recorded.get("path"), Path(probe["module_paths"].get(module_name, ""))
            ),
            f"receipt {module_name} path mismatch",
            identity_errors,
        )

    fixture = probe["fixture_statistics"]
    expected_fixture = {
        "base_fixture_sha256": expected["base_fixture_sha256"],
        "combined_oriented_fixture_sha256": expected["oriented_fixture_sha256"],
    }
    _require(
        probe.get("families") == list(EXPECTED_FAMILIES),
        "regenerated family set/order mismatch",
        identity_errors,
    )
    _require(
        set(probe.get("rows_per_family", {}).values()) == {per_family},
        "regenerated family row counts mismatch",
        identity_errors,
    )
    _require(
        fixture.get("base_fixture_sha256") == expected_fixture["base_fixture_sha256"],
        "regenerated base fixture hash mismatch",
        identity_errors,
    )
    _require(
        fixture.get("oriented_fixture_sha256", {}).get("original")
        == expected["original_fixture_sha256"],
        "regenerated original fixture hash mismatch",
        identity_errors,
    )
    _require(
        fixture.get("oriented_fixture_sha256", {}).get("swapped")
        == expected["swapped_fixture_sha256"],
        "regenerated swapped fixture hash mismatch",
        identity_errors,
    )
    _require(
        fixture.get("combined_oriented_fixture_sha256")
        == expected_fixture["combined_oriented_fixture_sha256"],
        "regenerated combined fixture hash mismatch",
        identity_errors,
    )
    for key, expected_value in (
        ("base_row_count", per_orientation),
        ("base_unique_numerical_case_count", 693),
        ("oriented_row_count", expected_total),
        ("oriented_unique_effective_numerical_case_count", 1382),
        ("oriented_unique_labelled_numerical_case_count", 1386),
    ):
        _require(
            fixture.get(key) == expected_value,
            f"regenerated {key} mismatch",
            identity_errors,
        )
    _require(
        receipt.get("base_battery_sha256") == expected["base_fixture_sha256"],
        "receipt base fixture hash mismatch",
        identity_errors,
    )
    _require(
        receipt.get("selected_battery_sha256") == expected["base_fixture_sha256"],
        "receipt selected base fixture hash mismatch",
        identity_errors,
    )
    _require(
        receipt.get("selected_oriented_battery_sha256")
        == expected["oriented_fixture_sha256"],
        "receipt selected oriented fixture hash mismatch",
        identity_errors,
    )
    receipt_fixture = receipt.get("fixture_statistics", {})
    _require(
        receipt_fixture == fixture,
        "receipt fixture statistics differ from regeneration",
        identity_errors,
    )

    result_valid = not exactness_errors and not identity_errors
    cache_blockers: list[str] = []
    cache_provenance = receipt.get("cache_provenance")
    endpoint_manifest = endpoint.get("prelaunch_manifest", {})
    endpoint_manifest_payload = endpoint_manifest.get("payload")
    manifest_observation = (
        endpoint_manifest_payload.get("observation", {})
        if isinstance(endpoint_manifest_payload, dict)
        else {}
    )
    manifest_cache_inventory = manifest_observation.get("cache_inventory", {})
    manifest_paths = manifest_observation.get("paths", {})
    manifest_python = manifest_observation.get("python", {})
    manifest_environment = manifest_observation.get("environment", {})
    prefix_tree = endpoint_manifest.get("external_pycache_prefix_tree", {})
    post_run = (
        cache_provenance.get("post_run", {})
        if isinstance(cache_provenance, dict)
        else {}
    )
    cache_provenance_bound = isinstance(cache_provenance, dict) and all(
        (
            cache_provenance.get("schema")
            == "neural-horner-frozen-battery-cache-provenance-v2",
            cache_provenance.get("verified_before_model_load") is True,
            cache_provenance.get("pre_run_inventory_recorded") is True,
            cache_provenance.get("prelaunch_manifest_path")
            == endpoint_manifest.get("path"),
            cache_provenance.get("prelaunch_manifest_sha256")
            == endpoint_manifest.get("sha256")
            == expected["prelaunch_manifest_sha256"],
            cache_provenance.get("prelaunch_manifest_expected_sha256")
            == expected["prelaunch_manifest_sha256"],
            cache_provenance.get("prelaunch_manifest_read_only") is True,
            endpoint_manifest.get("read_only") is True,
            cache_provenance.get("prelaunch_manifest_payload")
            == endpoint_manifest_payload,
            isinstance(endpoint_manifest_payload, dict)
            and endpoint_manifest_payload.get("schema") == PRELAUNCH_SCHEMA,
            isinstance(endpoint_manifest_payload, dict)
            and all(
                isinstance(endpoint_manifest_payload.get(name), str)
                and bool(endpoint_manifest_payload[name].strip())
                for name in ("run_id", "created_at_utc")
            ),
            manifest_paths.get("submission") == endpoint.get("submission"),
            manifest_paths.get("runner_root")
            == str(Path(endpoint["runner"]["path"]).parent),
            manifest_paths.get("scorer_repo")
            == endpoint.get("scorer_git", {}).get("repository"),
            manifest_paths.get("external_pycache_prefix") == prefix_tree.get("root"),
            manifest_python == endpoint.get("python"),
            manifest_environment.get("PYTHONDONTWRITEBYTECODE") == "1",
            manifest_environment.get("PYTHONPYCACHEPREFIX")
            == manifest_paths.get("external_pycache_prefix"),
            manifest_environment.get("sys_dont_write_bytecode") is True,
            manifest_environment.get("sys_pycache_prefix")
            == manifest_paths.get("external_pycache_prefix"),
            isinstance(manifest_cache_inventory, dict)
            and set(manifest_cache_inventory)
            == {
                "submission",
                "runner",
                "scorer",
                "external_pycache_prefix",
            },
            isinstance(manifest_cache_inventory, dict)
            and all(value == [] for value in manifest_cache_inventory.values()),
            cache_provenance.get("submission_cache_entries") == [],
            cache_provenance.get("runner_cache_entries") == [],
            cache_provenance.get("scorer_cache_entries") == [],
            cache_provenance.get("external_pycache_prefix_entries") == [],
            cache_provenance.get("fresh_external_pycache_prefix") is True,
            cache_provenance.get("bytecode_cache_writes_disabled") is True,
            cache_provenance.get("stable_during_run") is True,
            post_run.get("submission_cache_entries") == [],
            post_run.get("runner_cache_entries") == [],
            post_run.get("scorer_cache_entries") == [],
            post_run.get("external_pycache_prefix_entries") == [],
            post_run.get("bytecode_cache_writes_disabled") is True,
        )
    )
    if not cache_provenance_bound:
        cache_blockers.append("receipt_does_not_bind_verified_prelaunch_cache_manifest")
    if endpoint["artifact_tree"]["cache_entries"]:
        cache_blockers.append("submission_contains_bytecode_cache_at_validation")
    if endpoint["runner_tree"]["cache_entries"]:
        cache_blockers.append(
            "runner_source_tree_contains_bytecode_cache_at_validation"
        )
    if endpoint["scorer_tree"]["cache_entries"]:
        cache_blockers.append("scorer_checkout_contains_bytecode_cache_at_validation")
    if (
        prefix_tree.get("files")
        or prefix_tree.get("symlinks")
        or prefix_tree.get("cache_entries")
    ):
        cache_blockers.append("external_pycache_prefix_not_empty_at_validation")

    if result_valid and not cache_blockers:
        qualification_validation = "validated"
    elif result_valid:
        qualification_validation = "not_established_cache_provenance"
    else:
        qualification_validation = "failed_result_or_identity"

    return {
        "schema": "neural-horner-frozen-battery-endpoint-validation-v1",
        "status": "validated_development_exact" if result_valid else "failed",
        "result_validation": "validated_exact" if result_valid else "failed",
        "qualification_validation": qualification_validation,
        "promotion_eligible": result_valid and not cache_blockers,
        "exactness_errors": exactness_errors,
        "identity_errors": identity_errors,
        "cache_provenance_blockers": cache_blockers,
        "scope_nonclaims": [
            "not_a_sealed_generalization_result",
            "not_the_official_scorer_pipeline",
            "not_official_runtime_evidence",
            "endpoint_rehash_cannot_retroactively_prove_executed_bytecode_identity",
        ],
        "validated_counts": {
            "grand_total": expected_total,
            "orientations": len(EXPECTED_ORIENTATIONS),
            "cases_per_orientation": per_orientation,
            "families_per_orientation": len(EXPECTED_FAMILIES),
            "cases_per_family": per_family,
        },
    }


def write_json_once(path: Path, value: dict[str, Any]) -> None:
    path = path.resolve()
    if os.path.lexists(path):
        raise ReceiptValidationError(f"refusing to overwrite validation output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--helper", type=Path, required=True)
    parser.add_argument("--scorer-repo", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--prelaunch-manifest", type=Path, required=True)
    parser.add_argument("--expected-prelaunch-manifest-sha256", required=True)
    parser.add_argument("--expected-artifact-set-sha256", required=True)
    parser.add_argument("--expected-runner-sha256", required=True)
    parser.add_argument("--expected-helper-sha256", required=True)
    parser.add_argument("--expected-decoder-sha256", required=True)
    parser.add_argument("--expected-scorer-sha", required=True)
    parser.add_argument("--expected-scorer-python-tree-sha256", required=True)
    parser.add_argument("--expected-base-fixture-sha256", required=True)
    parser.add_argument("--expected-original-fixture-sha256", required=True)
    parser.add_argument("--expected-swapped-fixture-sha256", required=True)
    parser.add_argument("--expected-oriented-fixture-sha256", required=True)
    parser.add_argument("--expected-grand-total", type=int, default=1536)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    receipt_path = args.receipt.resolve()
    if not receipt_path.is_file() or receipt_path.is_symlink():
        raise ReceiptValidationError(f"receipt is missing or symlinked: {receipt_path}")
    receipt = json.loads(receipt_path.read_text())
    endpoint = collect_endpoint_identity(
        submission=args.submission,
        runner=args.runner,
        helper=args.helper,
        scorer_repo=args.scorer_repo,
        python_executable=args.python,
        prelaunch_manifest=args.prelaunch_manifest,
    )
    expected = {
        "artifact_set_sha256": args.expected_artifact_set_sha256,
        "runner_sha256": args.expected_runner_sha256,
        "helper_sha256": args.expected_helper_sha256,
        "decoder_sha256": args.expected_decoder_sha256,
        "scorer_sha": args.expected_scorer_sha,
        "scorer_python_tree_sha256": args.expected_scorer_python_tree_sha256,
        "base_fixture_sha256": args.expected_base_fixture_sha256,
        "original_fixture_sha256": args.expected_original_fixture_sha256,
        "swapped_fixture_sha256": args.expected_swapped_fixture_sha256,
        "oriented_fixture_sha256": args.expected_oriented_fixture_sha256,
        "grand_total": args.expected_grand_total,
        "prelaunch_manifest_sha256": (args.expected_prelaunch_manifest_sha256),
    }
    report = validate_completed_receipt(receipt, endpoint, expected)
    report.update(
        {
            "receipt": str(receipt_path),
            "receipt_sha256": sha256_file(receipt_path),
            "validator": str(Path(__file__).resolve()),
            "validator_sha256": sha256_file(Path(__file__).resolve()),
            "expected": expected,
            "endpoint_identity": endpoint,
        }
    )
    write_json_once(args.json_out, report)
    print(json.dumps(report, indent=2))
    return 0 if report["result_validation"] == "validated_exact" else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ReceiptValidationError, OSError, ValueError, json.JSONDecodeError) as error:
        print(f"battery receipt validation failed: {error}", file=sys.stderr)
        sys.exit(2)
