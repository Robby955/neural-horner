#!/usr/bin/env python3
"""Run the official scorer with a qualification-grade local receipt.

The wrapper calls :func:`modchallenge.evaluation.pipeline.evaluate_local`
unchanged.  It binds the submission, generated cases, wrapper, and scorer
checkout before that call and verifies that the executable identities did not
change afterward.

The official API does not return the loaded model, its parameter dtypes, or the
devices used by its operations.  The receipt therefore reports those facts as
unobserved.  Accelerator availability and PyTorch process defaults are useful
environment facts, but they are not evidence of the candidate's actual device
or inference dtype and are never presented as such.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

# Disable bytecode cache writes before importing the scorer or candidate code.
# Qualification also rejects caches that existed before this process started.
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import modchallenge  # noqa: E402
import torch  # noqa: E402
from modchallenge.config import EvalConfig, MULT_SUB_TIERS, TIERS  # noqa: E402
from modchallenge.evaluation.pipeline import evaluate_local  # noqa: E402
from modchallenge.testgen.generator import generate_private_test_set  # noqa: E402

from submission_utils import artifact_identity, sha256_file  # noqa: E402


PUBLIC_SEED = b"modchallenge-public-benchmark-v1"
PUBLIC_SEED_HEX = PUBLIC_SEED.hex()
PINNED_SCORER_SHA = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
PINNED_SCORER_PYTHON_TREE_SHA256 = (
    "c9039003eeb32a1bee99cb8c0e2a9b22dde60f9d913558bc70576754fbbd1478"
)

# This hash is over ``canonical_generated_case_bytes`` below for the pinned
# scorer, PUBLIC_SEED, and EvalConfig(total_problems=1100,
# timeout_seconds=300).  Keeping it independent of the imported generator
# turns the case-set receipt into a comparison, not just a self-description.
EXPECTED_PUBLIC_CASES_SHA256 = (
    "a6e6582c28874d1f37a1787c914091593c7fbd4e375d03ea121425c452ec7b91"
)

EXPECTED_TIER_GEOMETRY: tuple[dict[str, Any], ...] = (
    {
        "tier_id": 0,
        "min_bits": 1,
        "max_bits": 4096,
        "operand_bits": 0,
        "fixed_primes": [],
        "is_multiplication_only": True,
    },
    {
        "tier_id": 1,
        "min_bits": 1,
        "max_bits": 3,
        "operand_bits": 32,
        "fixed_primes": [2, 3, 5, 7],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 2,
        "min_bits": 4,
        "max_bits": 8,
        "operand_bits": 48,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 3,
        "min_bits": 9,
        "max_bits": 16,
        "operand_bits": 64,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 4,
        "min_bits": 17,
        "max_bits": 32,
        "operand_bits": 96,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 5,
        "min_bits": 33,
        "max_bits": 64,
        "operand_bits": 128,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 6,
        "min_bits": 65,
        "max_bits": 128,
        "operand_bits": 256,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 7,
        "min_bits": 129,
        "max_bits": 256,
        "operand_bits": 512,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 8,
        "min_bits": 257,
        "max_bits": 512,
        "operand_bits": 1024,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 9,
        "min_bits": 513,
        "max_bits": 1024,
        "operand_bits": 2048,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
    {
        "tier_id": 10,
        "min_bits": 1025,
        "max_bits": 2048,
        "operand_bits": 4096,
        "fixed_primes": [],
        "is_multiplication_only": False,
    },
)

EXPECTED_MULT_SUB_TIERS: tuple[tuple[int, int], ...] = (
    (1, 4),
    (5, 16),
    (17, 32),
    (33, 64),
    (65, 128),
    (129, 256),
    (257, 512),
    (513, 1024),
    (1025, 2048),
    (2049, 4096),
)

_PROHIBITED_BYTECODE_SUFFIXES = {".pyc", ".pyo"}


def canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def new_run_context() -> dict[str, str]:
    return {
        "run_id": str(uuid.uuid4()),
        "started_at_utc": utc_timestamp(),
    }


def prohibited_tree_entries(
    root: Path,
    *,
    ignored_top_level: frozenset[str] = frozenset(),
) -> list[dict[str, str]]:
    """Find bytecode caches and symlinks without following either."""
    root = root.resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"tree is not a directory: {root}")
    prohibited: list[dict[str, str]] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        if current_path == root:
            directory_names[:] = [
                name for name in directory_names if name not in ignored_top_level
            ]

        retained_directories = []
        for name in sorted(directory_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                prohibited.append({"path": relative, "kind": "symlink"})
            elif name == "__pycache__":
                prohibited.append({"path": relative, "kind": "bytecode_cache"})
            else:
                retained_directories.append(name)
        directory_names[:] = retained_directories

        for name in sorted(file_names):
            path = current_path / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                prohibited.append({"path": relative, "kind": "symlink"})
            elif path.suffix.lower() in _PROHIBITED_BYTECODE_SUFFIXES:
                prohibited.append({"path": relative, "kind": "bytecode_file"})
    return sorted(prohibited, key=lambda item: (item["path"], item["kind"]))


def source_set_identity(paths: list[Path], *, root: Path) -> dict[str, Any]:
    root = root.resolve()
    entries = []
    absolute_paths = [
        path if path.is_absolute() else Path.cwd() / path for path in paths
    ]
    for unresolved_path in sorted(absolute_paths):
        if unresolved_path.is_symlink():
            raise FileNotFoundError(
                f"wrapper source must not be a symlink: {unresolved_path}"
            )
        path = unresolved_path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"wrapper source is missing or not regular: {path}")
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError as error:
            raise ValueError(f"wrapper source lies outside source root: {path}") from error
        entries.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    identity = {
        "schema": "neural-horner-wrapper-source-set-v1",
        "root": str(root),
        "entries": entries,
    }
    identity["sha256"] = canonical_json_sha256(identity)
    return identity


def wrapper_source_identity() -> dict[str, Any]:
    scripts_root = Path(__file__).resolve().parent
    return source_set_identity(
        [Path(__file__).resolve(), scripts_root / "submission_utils.py"],
        root=scripts_root,
    )


def git_head(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def git_status(repo: Path) -> list[str]:
    completed = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain=v1", "--untracked-files=all"],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.splitlines()


def git_is_detached(repo: Path) -> bool:
    completed = subprocess.run(
        ["git", "-C", str(repo), "symbolic-ref", "-q", "HEAD"],
        capture_output=True,
        text=True,
    )
    return completed.returncode != 0


def python_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    paths = sorted(root.rglob("*.py"))
    if not paths:
        raise FileNotFoundError(f"no Python sources found under {root}")
    for path in paths:
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.read_bytes())
    return digest.hexdigest()


def installed_source_hint() -> Path | None:
    try:
        distribution = importlib.metadata.distribution("modchallenge")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None
    parsed = urlparse(json.loads(direct_url)["url"])
    if parsed.scheme != "file":
        return None
    return Path(unquote(parsed.path)).resolve()


def scorer_package_version() -> str:
    """Report a version without rejecting an immutable source-tree import."""
    try:
        return importlib.metadata.version("modchallenge")
    except importlib.metadata.PackageNotFoundError:
        return getattr(modchallenge, "__version__", "source-tree-uninstalled")


def git_toplevel(path: Path) -> Path:
    completed = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(completed.stdout.strip()).resolve()


def scorer_identity(explicit_repo: Path | None) -> dict[str, Any]:
    package_root = Path(modchallenge.__file__).resolve().parent
    source_hint = explicit_repo.resolve() if explicit_repo else installed_source_hint()
    if source_hint is None:
        raise RuntimeError(
            "cannot bind the installed scorer to a source checkout; pass --scorer-repo"
        )
    scorer_repo = git_toplevel(source_hint)
    source_package = scorer_repo / "src" / "modchallenge"
    installed_digest = python_tree_sha256(package_root)
    source_digest = python_tree_sha256(source_package)
    status = git_status(scorer_repo)
    repo_prohibited = prohibited_tree_entries(
        scorer_repo, ignored_top_level=frozenset({".git"})
    )
    package_prohibited = prohibited_tree_entries(package_root)
    identity = {
        "scorer_package_version": scorer_package_version(),
        "scorer_package_dir": str(package_root),
        "scorer_package_sha256": installed_digest,
        "scorer_repo": str(scorer_repo),
        "scorer_sha": git_head(scorer_repo),
        "scorer_git_clean": not status,
        "scorer_git_status": status,
        "scorer_git_detached": git_is_detached(scorer_repo),
        "scorer_source_sha256": source_digest,
        "scorer_source_matches_import": installed_digest == source_digest,
        "scorer_repo_prohibited_entries": repo_prohibited,
        "scorer_package_prohibited_entries": package_prohibited,
    }
    identity["scorer_identity_sha256"] = canonical_json_sha256(identity)
    return identity


def scorer_tier_geometry() -> list[dict[str, Any]]:
    return [
        {
            "tier_id": tier.tier_id,
            "min_bits": tier.min_bits,
            "max_bits": tier.max_bits,
            "operand_bits": tier.operand_bits,
            "fixed_primes": list(tier.fixed_primes),
            "is_multiplication_only": tier.is_multiplication_only,
        }
        for tier in TIERS
    ]


def _artifact_entry(path: Path, relative: str) -> dict[str, Any]:
    if path.is_symlink():
        return {
            "path": relative,
            "kind": "symlink",
            "target": os.readlink(path),
        }
    if not path.is_file():
        raise RuntimeError(f"unsupported non-file artifact entry: {path}")
    return {
        "path": relative,
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_tree_identity(submission: Path) -> dict[str, Any]:
    """Hash every regular file and symlink target in the submission tree."""
    submission = submission.resolve()
    if not submission.is_dir():
        raise NotADirectoryError(f"submission is not a directory: {submission}")

    entries: list[dict[str, Any]] = []
    for path in sorted(submission.rglob("*")):
        relative_path = path.relative_to(submission)
        if path.is_dir() and not path.is_symlink():
            continue
        entries.append(_artifact_entry(path, relative_path.as_posix()))

    canonical = json.dumps(
        entries,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "schema": "neural-horner-artifact-tree-v1",
        "cache_exclusions": [],
        "entry_count": len(entries),
        "regular_file_bytes": sum(
            entry.get("bytes", 0) for entry in entries if entry["kind"] == "file"
        ),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "entries": entries,
    }


def artifact_tree_difference(
    before: dict[str, Any], after: dict[str, Any]
) -> dict[str, list[str]]:
    before_by_path = {entry["path"]: entry for entry in before["entries"]}
    after_by_path = {entry["path"]: entry for entry in after["entries"]}
    before_paths = set(before_by_path)
    after_paths = set(after_by_path)
    return {
        "added": sorted(after_paths - before_paths),
        "removed": sorted(before_paths - after_paths),
        "changed": sorted(
            path
            for path in before_paths & after_paths
            if before_by_path[path] != after_by_path[path]
        ),
    }


def canonical_generated_case_bytes(test_set: Any, seed: bytes) -> bytes:
    payload = {
        "schema": "modchallenge-generated-full-cases-v1",
        "seed_hex": seed.hex(),
        "tiers": [
            {
                "tier_id": tier.tier_id,
                "cases": [
                    {
                        "a": case.a,
                        "b": case.b,
                        "p": case.p,
                        "expected": case.expected,
                        "tier_id": case.tier_id,
                    }
                    for case in tier.cases
                ],
            }
            for tier in test_set.tiers
        ],
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _is_canonical_nonnegative_decimal(value: Any) -> bool:
    return (
        isinstance(value, str)
        and value.isascii()
        and value.isdigit()
        and str(int(value)) == value
    )


def generated_case_identity(seed: bytes, config: EvalConfig) -> dict[str, Any]:
    """Generate and bind exactly the cases the official pipeline will regenerate."""
    test_set = generate_private_test_set(master_seed=seed, config=config)
    canonical = canonical_generated_case_bytes(test_set, seed)
    tier_order = [tier.tier_id for tier in test_set.tiers]
    tier_counts = [
        {"tier_id": tier.tier_id, "count": len(tier.cases)}
        for tier in test_set.tiers
    ]
    cases = [case for tier in test_set.tiers for case in tier.cases]

    tier_container_matches = all(
        case.tier_id == tier.tier_id
        for tier in test_set.tiers
        for case in tier.cases
    )
    decimal_fields_canonical = all(
        _is_canonical_nonnegative_decimal(value)
        for case in cases
        for value in (case.a, case.b, case.p, case.expected)
    )
    expected_answers_exact = all(
        int(case.expected) == (int(case.a) * int(case.b)) % int(case.p)
        for case in cases
    )

    observed_tiers = []
    for tier in test_set.tiers:
        prime_values = [int(case.p) for case in tier.cases]
        a_values = [int(case.a) for case in tier.cases]
        b_values = [int(case.b) for case in tier.cases]
        observed_tiers.append(
            {
                "tier_id": tier.tier_id,
                "count": len(tier.cases),
                "distinct_prime_count": len(set(prime_values)),
                "prime_bit_length_min": min(value.bit_length() for value in prime_values),
                "prime_bit_length_max": max(value.bit_length() for value in prime_values),
                "operand_bit_length_max": max(
                    max(value.bit_length() for value in a_values),
                    max(value.bit_length() for value in b_values),
                ),
            }
        )

    return {
        "schema": "modchallenge-generated-case-identity-v1",
        "canonical_serialization": (
            "UTF-8 compact JSON with sorted object keys; ordered tiers and cases; "
            "fields a,b,p,expected,tier_id; schema and seed included"
        ),
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "canonical_bytes": len(canonical),
        "seed_hex": seed.hex(),
        "total_cases": len(cases),
        "tier_order": tier_order,
        "tier_counts": tier_counts,
        "case_tier_matches_container": tier_container_matches,
        "decimal_fields_canonical": decimal_fields_canonical,
        "expected_answers_exact": expected_answers_exact,
        "observed_tiers": observed_tiers,
        "scorer_tier_geometry": scorer_tier_geometry(),
        "scorer_multiplication_sub_tiers": [list(bounds) for bounds in MULT_SUB_TIERS],
        "generator_module": str(
            Path(sys.modules[generate_private_test_set.__module__].__file__).resolve()
        ),
    }


def public_case_identity_errors(case_identity: dict[str, Any]) -> list[str]:
    expected_counts = [
        {"tier_id": tier_id, "count": 100} for tier_id in range(11)
    ]
    errors = []
    if case_identity["seed_hex"] != PUBLIC_SEED_HEX:
        errors.append("generated-case seed is not the public benchmark seed")
    if case_identity["total_cases"] != 1100:
        errors.append("generated case count is not exactly 1100")
    if case_identity["tier_order"] != list(range(11)):
        errors.append("generated tier order is not exactly 0 through 10")
    if case_identity["tier_counts"] != expected_counts:
        errors.append("generated tier counts are not exactly 100 per tier")
    if not case_identity["case_tier_matches_container"]:
        errors.append("one or more generated cases has the wrong tier_id")
    if not case_identity["decimal_fields_canonical"]:
        errors.append("one or more generated case fields is not canonical decimal")
    if not case_identity["expected_answers_exact"]:
        errors.append("one or more generated expected answers is arithmetically wrong")
    if case_identity["scorer_tier_geometry"] != list(EXPECTED_TIER_GEOMETRY):
        errors.append("imported scorer tier geometry differs from the pinned contract")
    if case_identity["scorer_multiplication_sub_tiers"] != [
        list(bounds) for bounds in EXPECTED_MULT_SUB_TIERS
    ]:
        errors.append(
            "imported scorer multiplication sub-tier geometry differs from the pinned contract"
        )
    if case_identity["sha256"] != EXPECTED_PUBLIC_CASES_SHA256:
        errors.append("generated public case hash differs from the pinned case-set hash")
    return errors


def execution_environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    mps_available = torch.backends.mps.is_available()
    try:
        process_default_device = str(torch.get_default_device())
    except (AttributeError, RuntimeError):
        process_default_device = "unavailable_from_this_torch_build"
    environment = {
        "runner_path": str(Path(__file__).resolve()),
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "python": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": platform.processor() or "unreported",
        "cpu_count": os.cpu_count(),
        "torch": str(torch.__version__),
        "torch_cuda_version": torch.version.cuda,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "torch_default_device": process_default_device,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "sys_dont_write_bytecode": sys.dont_write_bytecode,
        "python_dont_write_bytecode_env": os.environ.get("PYTHONDONTWRITEBYTECODE"),
        "bytecode_cache_writes_disabled": (
            sys.dont_write_bytecode
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        ),
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_devices": (
            [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
            if cuda_available
            else []
        ),
        "mps_available": mps_available,
        "actual_inference_device": None,
        "actual_inference_device_status": "unobserved_by_official_evaluate_local_api",
        "actual_inference_dtype": None,
        "actual_inference_dtype_status": "unobserved_by_official_evaluate_local_api",
        "model_parameter_dtypes": None,
        "model_parameter_dtypes_status": "unobserved_by_official_evaluate_local_api",
    }
    environment["environment_sha256"] = canonical_json_sha256(environment)
    return environment


def runtime_measurements() -> dict[str, Any]:
    """Record process-level peaks without pretending to observe model internals."""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_bytes = peak_rss if sys.platform == "darwin" else peak_rss * 1024
    measurements: dict[str, Any] = {
        "process_peak_rss_bytes": peak_rss_bytes,
        "process_peak_rss_scope": "evaluation_wrapper_process",
        "actual_inference_device": None,
        "actual_inference_device_status": "unobserved_by_official_evaluate_local_api",
        "actual_inference_dtype": None,
        "actual_inference_dtype_status": "unobserved_by_official_evaluate_local_api",
        "model_scoped_peak_memory": None,
        "model_scoped_peak_memory_status": "unobserved_by_official_evaluate_local_api",
        "final_deployment_gate_sufficient": False,
        "missing_for_final_deployment_gate": [
            "organizer-confirmed image and hardware",
            "observed model inference device",
            "observed model inference dtype",
            "model-scoped peak memory",
        ],
    }
    if torch.cuda.is_available():
        measurements.update(
            {
                "cuda_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "cuda_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "cuda_peak_scope": "current_process_all_torch_cuda_allocations",
                "cuda_peak_is_model_scoped": False,
            }
        )
    return measurements


def _path_is_within(path: Path, directory: Path) -> bool:
    try:
        path.resolve().relative_to(directory.resolve())
    except ValueError:
        return False
    return True


def finalized_receipt(
    run_context: dict[str, str], payload: dict[str, Any]
) -> dict[str, Any]:
    return {
        **run_context,
        **payload,
        "ended_at_utc": utc_timestamp(),
    }


def write_receipt(path: Path, receipt: dict[str, Any]) -> None:
    """Atomically create a new receipt; never replace an existing path."""
    path = path if path.is_absolute() else Path.cwd() / path
    path.parent.mkdir(parents=True, exist_ok=True)
    if os.path.lexists(path):
        raise FileExistsError(f"refusing to overwrite existing receipt: {path}")
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    encoded = (json.dumps(receipt, indent=2) + "\n").encode("utf-8")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes the fully written inode and
        # fails atomically if another process won the destination name.
        os.link(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    print(json.dumps(receipt, indent=2))


def persist_receipt(
    path: Path,
    run_context: dict[str, str],
    payload: dict[str, Any],
) -> bool:
    receipt = finalized_receipt(run_context, payload)
    try:
        write_receipt(path, receipt)
    except BaseException as error:
        fallback = finalized_receipt(
            run_context,
            {
                "status": "receipt_persistence_error",
                "intended_receipt_path": str(path.resolve()),
                "error_type": type(error).__name__,
                "error": str(error),
                "original_status": payload.get("status"),
            },
        )
        fallback_paths = [
            path.resolve().with_name(
                f"{path.name}.{run_context['run_id']}.persistence-error.json"
            ),
            Path(tempfile.gettempdir())
            / "neural-horner-official-eval-errors"
            / f"{run_context['run_id']}.json",
        ]
        persisted = False
        for fallback_path in fallback_paths:
            try:
                write_receipt(fallback_path, fallback)
                persisted = True
                break
            except BaseException:
                continue
        if not persisted:
            print(json.dumps(fallback, indent=2), file=sys.stderr)
        return False
    return True


def external_preflight_error_path(
    requested: Path,
    submission: Path,
    run_context: dict[str, str],
    *,
    additional_forbidden: tuple[Path, ...] = (),
) -> Path:
    forbidden = (submission, *additional_forbidden)
    candidates = [
        requested.resolve().with_name(
            f"{requested.name}.{run_context['run_id']}.error.json"
        ),
        submission.resolve().parent
        / f"{submission.name}.official-eval-{run_context['run_id']}.error.json",
        Path(tempfile.gettempdir())
        / "neural-horner-official-eval-errors"
        / f"{run_context['run_id']}.json",
    ]
    for candidate in candidates:
        if not any(_path_is_within(candidate, tree) for tree in forbidden):
            return candidate
    raise RuntimeError("could not derive an external error-receipt path")


def persist_external_preflight_error(
    requested: Path,
    submission: Path,
    run_context: dict[str, str],
    payload: dict[str, Any],
    *,
    additional_forbidden: tuple[Path, ...] = (),
) -> None:
    fallback = external_preflight_error_path(
        requested,
        submission,
        run_context,
        additional_forbidden=additional_forbidden,
    )
    persist_receipt(
        fallback,
        run_context,
        {
            **payload,
            "requested_receipt_path": str(requested.resolve()),
            "error_receipt_path": str(fallback.resolve()),
            "receipt_persisted": True,
        },
    )


def _safe_post_identity(
    submission: Path,
    scorer_repo: Path | None,
) -> tuple[dict[str, Any], list[str]]:
    post: dict[str, Any] = {}
    errors: list[str] = []
    try:
        post["artifact_tree"] = artifact_tree_identity(submission)
    except BaseException as error:
        post["artifact_tree_error"] = f"{type(error).__name__}: {error}"
        errors.append("could not rehash the artifact tree after evaluation")
    try:
        post["artifact_files"] = artifact_identity(submission)
    except BaseException as error:
        post["artifact_files_error"] = f"{type(error).__name__}: {error}"
        errors.append("could not rehash critical artifact files after evaluation")
    try:
        post["wrapper_source_set"] = wrapper_source_identity()
        post["runner_sha256"] = next(
            entry["sha256"]
            for entry in post["wrapper_source_set"]["entries"]
            if entry["path"] == Path(__file__).name
        )
    except BaseException as error:
        post["wrapper_source_error"] = f"{type(error).__name__}: {error}"
        errors.append("could not rehash the wrapper source set after evaluation")
    try:
        post["scorer"] = scorer_identity(scorer_repo)
    except BaseException as error:
        post["scorer_error"] = f"{type(error).__name__}: {error}"
        errors.append("could not re-establish scorer identity after evaluation")
    return post, errors


def post_identity_errors(
    before: dict[str, Any], post: dict[str, Any], capture_errors: list[str]
) -> tuple[list[str], dict[str, list[str]] | None]:
    errors = list(capture_errors)
    artifact_difference = None
    if "artifact_tree" in post:
        artifact_difference = artifact_tree_difference(
            before["artifact_tree"], post["artifact_tree"]
        )
        if before["artifact_tree"]["sha256"] != post["artifact_tree"]["sha256"]:
            errors.append("submission artifact tree changed during evaluation")
    if "artifact_files" in post:
        if before["artifact_files"] != post["artifact_files"]:
            errors.append("one or more critical artifact files changed during evaluation")
    if "wrapper_source_set" in post and (
        before["wrapper_source_set"] != post["wrapper_source_set"]
    ):
        errors.append("wrapper source set changed during evaluation")
    if "scorer" in post and before["scorer"] != post["scorer"]:
        errors.append("scorer source or checkout identity changed during evaluation")
    return errors, artifact_difference


def exact_result_geometry(summary: dict[str, Any]) -> bool:
    tiers = summary.get("tiers", [])
    return (
        len(tiers) == 11
        and {tier.get("tier_id") for tier in tiers} == set(range(11))
        and all(tier.get("total") == 100 for tier in tiers)
        and all(tier.get("completed") is True for tier in tiers)
    )


def persist_and_return(
    path: Path,
    run_context: dict[str, str],
    payload: dict[str, Any],
    intended_code: int,
) -> int:
    return intended_code if persist_receipt(path, run_context, payload) else 2


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission", type=Path)
    parser.add_argument("--total", type=int, default=1100)
    parser.add_argument("--seed", default=PUBLIC_SEED_HEX)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--scorer-repo", type=Path)
    parser.add_argument("--expected-scorer-sha")
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--require-scored-perfect", action="store_true")
    args = parser.parse_args()

    run_context = new_run_context()
    requested_submission_path = (
        args.submission
        if args.submission.is_absolute()
        else Path.cwd() / args.submission
    )
    submission_root_is_symlink = requested_submission_path.is_symlink()
    submission = requested_submission_path.resolve()
    requested_receipt_path = (
        args.json_out
        if args.json_out.is_absolute()
        else Path.cwd() / args.json_out
    )
    receipt_path = requested_receipt_path
    resolved_receipt_path = requested_receipt_path.resolve()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    started = time.perf_counter()
    declared_scorer_forbidden = (
        (args.scorer_repo.resolve(),) if args.scorer_repo is not None else ()
    )

    # This check must precede directory creation or receipt persistence.  With
    # an invalid destination, persist a unique terminal record outside the tree.
    if _path_is_within(resolved_receipt_path, submission):
        persist_external_preflight_error(
            requested_receipt_path,
            submission,
            run_context,
            {
                "status": "invalid_receipt_path",
                "submission": str(submission),
                "error": "--json-out must be outside the submission artifact",
            },
            additional_forbidden=declared_scorer_forbidden,
        )
        return 2
    if os.path.lexists(requested_receipt_path) or os.path.lexists(
        resolved_receipt_path
    ):
        persist_external_preflight_error(
            requested_receipt_path,
            submission,
            run_context,
            {
                "status": "receipt_path_exists",
                "submission": str(submission),
                "error": "refusing to overwrite an existing receipt path",
            },
            additional_forbidden=declared_scorer_forbidden,
        )
        return 2

    identity: dict[str, Any] = {
        "submission": str(submission),
        "requested_submission_path": str(requested_submission_path),
        "receipt_path": str(resolved_receipt_path),
        "pinned_scorer_sha": PINNED_SCORER_SHA,
        "pinned_scorer_python_tree_sha256": PINNED_SCORER_PYTHON_TREE_SHA256,
        "seed_hex": args.seed,
        "total_problems": args.total,
        "timeout_seconds": args.timeout,
    }
    try:
        seed = bytes.fromhex(args.seed)
        config = EvalConfig(
            total_problems=args.total,
            timeout_seconds=args.timeout,
        )
        scorer = scorer_identity(args.scorer_repo)
        identity.update(scorer)

        # A receipt in the scorer checkout would invalidate the source identity
        # while recording it.  Reject it without writing there.
        if _path_is_within(receipt_path, Path(scorer["scorer_repo"])):
            persist_external_preflight_error(
                requested_receipt_path,
                submission,
                run_context,
                {
                    **identity,
                    "status": "invalid_receipt_path",
                    "error": "--json-out must be outside the scorer checkout",
                },
                additional_forbidden=(Path(scorer["scorer_repo"]),),
            )
            return 2

        submission_prohibited = prohibited_tree_entries(submission)
        if submission_root_is_symlink:
            submission_prohibited.insert(0, {"path": ".", "kind": "symlink"})
        wrapper_sources = wrapper_source_identity()
        environment = execution_environment()
        before = {
            "artifact_tree": artifact_tree_identity(submission),
            "artifact_files": artifact_identity(submission),
            "execution_environment": environment,
            "wrapper_source_set": wrapper_sources,
            "scorer": scorer,
        }
        identity.update(
            {
                # Preserve the earlier field while adding the complete tree.
                "artifact_sha256": before["artifact_files"],
                "artifact_tree_before": before["artifact_tree"],
                "submission_prohibited_entries": submission_prohibited,
                "wrapper_source_set_before": wrapper_sources,
                "execution_environment": environment,
            }
        )

        identity_errors = []
        if not scorer["scorer_source_matches_import"]:
            identity_errors.append(
                "imported scorer package does not match source checkout"
            )
        if not scorer["scorer_git_clean"]:
            identity_errors.append("scorer checkout is dirty")
        if args.expected_scorer_sha and scorer["scorer_sha"] != args.expected_scorer_sha:
            identity_errors.append("scorer HEAD does not match --expected-scorer-sha")
        if args.require_scored_perfect:
            if args.total != 1100:
                identity_errors.append(
                    "promotion gate requires exactly 1100 generated problems"
                )
            if args.seed != PUBLIC_SEED_HEX:
                identity_errors.append(
                    "promotion gate requires the canonical public benchmark seed"
                )
            if args.timeout != 300:
                identity_errors.append(
                    "promotion gate requires the official 300-second timeout"
                )
            if args.expected_scorer_sha != PINNED_SCORER_SHA:
                identity_errors.append(
                    "promotion gate requires --expected-scorer-sha at the pinned commit"
                )
            if scorer["scorer_sha"] != PINNED_SCORER_SHA:
                identity_errors.append(
                    "promotion gate scorer HEAD is not the pinned commit"
                )
            if not scorer["scorer_git_detached"]:
                identity_errors.append(
                    "promotion gate requires a detached scorer checkout"
                )
            if scorer["scorer_source_sha256"] != PINNED_SCORER_PYTHON_TREE_SHA256:
                identity_errors.append(
                    "scorer checkout Python tree differs from the pinned digest"
                )
            if scorer["scorer_package_sha256"] != PINNED_SCORER_PYTHON_TREE_SHA256:
                identity_errors.append(
                    "imported scorer Python tree differs from the pinned digest"
                )
            if submission_prohibited:
                identity_errors.append(
                    "submission contains a bytecode cache, bytecode file, or symlink"
                )
            if scorer["scorer_repo_prohibited_entries"]:
                identity_errors.append(
                    "scorer checkout contains a bytecode cache, bytecode file, or symlink"
                )
            if scorer["scorer_package_prohibited_entries"]:
                identity_errors.append(
                    "imported scorer package contains a bytecode cache, bytecode file, or symlink"
                )
            if not environment["bytecode_cache_writes_disabled"]:
                identity_errors.append(
                    "qualification process does not have bytecode cache writes disabled"
                )
        if identity_errors:
            return persist_and_return(
                receipt_path,
                run_context,
                {
                    **identity,
                    "status": "qualification_identity_error",
                    "errors": identity_errors,
                },
                2,
            )

        try:
            case_identity = generated_case_identity(seed, config)
        except BaseException as error:
            return persist_and_return(
                receipt_path,
                run_context,
                {
                    **identity,
                    "status": "generated_case_identity_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
                2,
            )
        identity["generated_cases"] = case_identity
        if args.require_scored_perfect:
            case_errors = public_case_identity_errors(case_identity)
            if case_errors:
                return persist_and_return(
                    receipt_path,
                    run_context,
                    {
                        **identity,
                        "status": "generated_case_identity_error",
                        "errors": case_errors,
                    },
                    2,
                )

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        result = None
        evaluation_error: BaseException | None = None
        evaluation_call_started = time.perf_counter()
        try:
            # Call the pinned pipeline without replacing its loader, batching,
            # determinism, timeout, decoding, or scoring logic.
            result = evaluate_local(
                submission,
                master_seed=seed,
                config=config,
            )
        except BaseException as error:
            evaluation_error = error
        evaluation_call_wall_seconds = time.perf_counter() - evaluation_call_started

        cache_disabled_after_call = (
            sys.dont_write_bytecode
            and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        )
        # Restore the qualification invariant before any post-run imports or
        # identity work, while retaining evidence if candidate code changed it.
        sys.dont_write_bytecode = True
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

        post, capture_errors = _safe_post_identity(submission, args.scorer_repo)
        stability_errors, artifact_difference = post_identity_errors(
            before, post, capture_errors
        )
        if not cache_disabled_after_call:
            stability_errors.append(
                "bytecode cache disabling was changed during evaluate_local"
            )
        try:
            measurements = runtime_measurements()
        except BaseException as error:
            measurements = {
                "status": "runtime_measurement_error",
                "error_type": type(error).__name__,
                "error": str(error),
                "final_deployment_gate_sufficient": False,
            }
            stability_errors.append("could not finalize runtime measurements")

        wrapper_wall_seconds = time.perf_counter() - started
        receipt = {
            **identity,
            "artifact_tree_after": post.get("artifact_tree"),
            "artifact_sha256_after": post.get("artifact_files"),
            "wrapper_source_set_after": post.get("wrapper_source_set"),
            "post_run_runner_sha256": post.get("runner_sha256"),
            "post_run_scorer_identity": post.get("scorer"),
            "artifact_difference": artifact_difference,
            "post_run_identity_capture": post,
            "identity_stability_errors": stability_errors,
            "artifact_unchanged": not any(
                "artifact" in error for error in stability_errors
            ),
            "wrapper_source_set_unchanged": not any(
                "wrapper" in error or "runner" in error
                for error in stability_errors
            ),
            "runner_unchanged": not any(
                "wrapper" in error or "runner" in error
                for error in stability_errors
            ),
            "scorer_unchanged": not any(
                "scorer" in error for error in stability_errors
            ),
            "bytecode_cache_writes_disabled_after_evaluate_local": (
                cache_disabled_after_call
            ),
            "evaluation_wall_seconds": evaluation_call_wall_seconds,
            "evaluation_wall_seconds_scope": (
                "entire pinned evaluate_local call, including validation, static check, "
                "case generation, load, isolation, determinism, inference, and scoring"
            ),
            "qualification_wrapper_wall_seconds": wrapper_wall_seconds,
            "qualification_wrapper_wall_seconds_scope": (
                "preflight identity and duplicate case hashing, evaluate_local, and "
                "post-run identity checks"
            ),
            "runtime_measurements": measurements,
        }

        if stability_errors:
            receipt.update(
                {
                    "status": "post_run_identity_error",
                    "evaluation_status_before_identity_gate": (
                        "error" if evaluation_error is not None else "completed"
                    ),
                    "errors": stability_errors,
                }
            )
            if evaluation_error is not None:
                receipt.update(
                    {
                        "evaluation_error_type": type(evaluation_error).__name__,
                        "evaluation_error": str(evaluation_error),
                    }
                )
            return persist_and_return(receipt_path, run_context, receipt, 2)

        if evaluation_error is not None:
            receipt.update(
                {
                    "status": (
                        "interrupted"
                        if isinstance(evaluation_error, KeyboardInterrupt)
                        else "error"
                    ),
                    "error_type": type(evaluation_error).__name__,
                    "error": str(evaluation_error),
                }
            )
            intended_code = 130 if isinstance(evaluation_error, KeyboardInterrupt) else 1
            return persist_and_return(
                receipt_path, run_context, receipt, intended_code
            )

        if result is None:
            raise RuntimeError("evaluate_local returned no result and no exception")
        try:
            summary = result.summary()
        except BaseException as error:
            receipt.update(
                {
                    "status": "result_summary_error",
                    "error_type": type(error).__name__,
                    "error": str(error),
                }
            )
            return persist_and_return(receipt_path, run_context, receipt, 2)
        receipt.update({"status": "completed", "result": summary})

        if not args.require_scored_perfect:
            return persist_and_return(receipt_path, run_context, receipt, 0)

        tiers = summary.get("tiers", [])
        scored = [tier for tier in tiers if tier.get("tier_id", 0) > 0]
        exact_geometry = exact_result_geometry(summary)
        conservative_wall_time_ok = evaluation_call_wall_seconds <= args.timeout
        scored_tiers_perfect = (
            len(scored) == 10
            and all(tier.get("correct") == 100 for tier in scored)
        )
        deterministic = summary.get("deterministic") is True
        perfect = (
            exact_geometry
            and scored_tiers_perfect
            and deterministic
            and conservative_wall_time_ok
        )
        receipt["promotion_gate"] = {
            "scope": (
                "local artifact/case/scorer identity, accuracy, determinism, "
                "geometry, and entire pinned evaluate_local-call wall time"
            ),
            "artifact_unchanged": True,
            "wrapper_source_set_unchanged": True,
            "runner_unchanged": True,
            "scorer_unchanged": True,
            "generated_public_cases_sha256_exact": (
                case_identity["sha256"] == EXPECTED_PUBLIC_CASES_SHA256
            ),
            "exact_1100_geometry": exact_geometry,
            "scored_tiers_perfect": scored_tiers_perfect,
            "deterministic": deterministic,
            "conservative_entire_evaluate_local_call_within_300_seconds": (
                conservative_wall_time_ok
            ),
            "passed": perfect,
            "final_deployment_gate_sufficient": False,
        }
        return persist_and_return(
            receipt_path, run_context, receipt, 0 if perfect else 1
        )
    except BaseException as error:
        # Any unexpected preflight or finalization failure after a valid
        # external receipt path becomes an explicit terminal receipt.
        return persist_and_return(
            receipt_path,
            run_context,
            {
                **identity,
                "status": (
                    "interrupted"
                    if isinstance(error, KeyboardInterrupt)
                    else "wrapper_internal_error"
                ),
                "error_type": type(error).__name__,
                "error": str(error),
                "qualification_wrapper_wall_seconds": time.perf_counter() - started,
            },
            130 if isinstance(error, KeyboardInterrupt) else 2,
        )


if __name__ == "__main__":
    sys.exit(main())
