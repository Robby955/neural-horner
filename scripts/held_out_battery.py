"""Frozen structured adversarial/development battery.

The earlier adversarial_stress.py families (powers of two, sparse, near-multiple,
all-ones, symmetric, multiply-control) are exactly the families used to refine the
DAgger training data, so passing them is in-distribution fit, not robustness. The
families below were disjoint from the submitted v8 checkpoint's original
training and supplied a valid first-contact diagnostic. They were later used to
select multiple repair attempts and routing decisions, so they are now frozen
development data, not a sealed generalization set: Fibonacci-valued operands, Fermat
numbers, alternating bit patterns, fixed-Hamming-weight operands, operands whose
legacy family name says their product straddles a multiple of p (the actual frozen
construction is quotient-correlated; receipts diagnose its boundary distances), and
a structurally-chosen prime (p = 3 mod 4). Reports exact-match per family vs an
independent Python ground truth.

Usage: python held_out_battery.py <submission_dir> [--n 128]
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Sized
from pathlib import Path
from statistics import fmean
from typing import Any

# Capture the caller's bytecode controls before enforcing the runner invariant.
# Qualification requires the caller to have supplied both controls before Python
# started; setting them only after imports would not establish cache provenance.
_CALLER_PYTHONDONTWRITEBYTECODE = os.environ.get("PYTHONDONTWRITEBYTECODE")
_CALLER_PYTHONPYCACHEPREFIX = os.environ.get("PYTHONPYCACHEPREFIX")
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
sys.dont_write_bytecode = True

import torch  # noqa: E402
from modchallenge.evaluation.decoder import (  # noqa: E402
    MalformedOutput,
    decode_answer,
    resolve_base,
)

from submission_utils import (  # noqa: E402
    artifact_identity,
    load_submission,
    sha256_file,
)

_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
_SEED = 20260627
_PRODUCT_STRADDLES_FAMILY = "product straddles k*p"
_PINNED_SCORER_COMMIT = "82510bba00a1126649bd76dd1a451f14d0b3eb60"
_FROZEN_L2048_N128_BASE_SHA256 = (
    "1be39e006b2b75e19648a9845a391f79831dc7d8ea1c1c3868e991350e159c4f"
)
_FROZEN_L2048_N128_ORIGINAL_SHA256 = (
    "1be39e006b2b75e19648a9845a391f79831dc7d8ea1c1c3868e991350e159c4f"
)
_FROZEN_L2048_N128_SWAPPED_SHA256 = (
    "a8ba93e03fd0a7e5ba498a6b50ae7fefa8de14452137193621d17282cc59ed77"
)
_FROZEN_L2048_N128_BOTH_SHA256 = (
    "c1c345ddd2eb2438848898907c0481d4be2b8855eac2e3c5663ce7b31fb3112e"
)
_OFFICIAL_MODULES = (
    "modchallenge.evaluation.decoder",
    "modchallenge.evaluation.pipeline",
    "modchallenge.interface.base_model",
)
_PRELAUNCH_SCHEMA = "neural-horner-frozen-battery-prelaunch-v1"
_BYTECODE_SUFFIXES = frozenset({".pyc", ".pyo"})


class QualificationIdentityError(RuntimeError):
    """Raised before inference when a required immutable identity mismatches."""


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def write_receipt(path: Path, receipt: dict) -> None:
    """Atomically preserve partial progress for long battery evaluations."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)


def _path_is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def bytecode_cache_entries(
    root: Path,
    *,
    ignored_top_level: frozenset[str] = frozenset(),
) -> list[str]:
    """List bytecode caches without following symlinks."""
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise QualificationIdentityError(f"cache inventory root is invalid: {root}")
    entries: list[str] = []
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
                continue
            if name == "__pycache__":
                entries.append(relative)
            else:
                retained.append(name)
        directory_names[:] = retained
        for name in sorted(file_names):
            path = current_path / name
            if path.is_symlink():
                continue
            if path.suffix.lower() in _BYTECODE_SUFFIXES:
                entries.append(path.relative_to(root).as_posix())
    return sorted(set(entries))


def tree_entries(root: Path) -> list[str]:
    """List every entry in a directory without following symlinks."""
    root = root.resolve()
    if root.is_symlink() or not root.is_dir():
        raise QualificationIdentityError(f"tree inventory root is invalid: {root}")
    entries: list[str] = []
    for current, directory_names, file_names in os.walk(root, followlinks=False):
        current_path = Path(current)
        for name in sorted(directory_names):
            path = current_path / name
            entries.append(path.relative_to(root).as_posix())
        for name in sorted(file_names):
            path = current_path / name
            entries.append(path.relative_to(root).as_posix())
        directory_names[:] = [
            name for name in directory_names if not (current_path / name).is_symlink()
        ]
    return sorted(set(entries))


def prelaunch_observation(
    *,
    submission: Path,
    runner_root: Path,
    scorer_repo: Path,
    external_pycache_prefix: Path,
) -> dict[str, Any]:
    """Recompute the exact cache boundary described by a prelaunch manifest."""
    python_executable = Path(sys.executable).resolve()
    sys_pycache_prefix = (
        str(Path(sys.pycache_prefix).resolve())
        if sys.pycache_prefix is not None
        else None
    )
    return {
        "paths": {
            "submission": str(submission.resolve()),
            "runner_root": str(runner_root.resolve()),
            "scorer_repo": str(scorer_repo.resolve()),
            "external_pycache_prefix": str(external_pycache_prefix.resolve()),
        },
        "python": {
            "executable": str(python_executable),
            "executable_sha256": sha256_file(python_executable),
        },
        "environment": {
            "PYTHONDONTWRITEBYTECODE": _CALLER_PYTHONDONTWRITEBYTECODE,
            "PYTHONPYCACHEPREFIX": _CALLER_PYTHONPYCACHEPREFIX,
            "sys_dont_write_bytecode": sys.dont_write_bytecode,
            "sys_pycache_prefix": sys_pycache_prefix,
        },
        "cache_inventory": {
            "submission": bytecode_cache_entries(submission),
            "runner": bytecode_cache_entries(runner_root),
            "scorer": bytecode_cache_entries(
                scorer_repo,
                ignored_top_level=frozenset({".git"}),
            ),
            "external_pycache_prefix": tree_entries(external_pycache_prefix),
        },
    }


def verify_prelaunch_manifest(
    receipt: dict[str, Any],
    *,
    manifest_path: Path,
    expected_sha256: str,
    submission: Path,
    runner_root: Path,
    scorer_repo: Path,
) -> dict[str, Any]:
    """Verify and bind caller-recorded cache state before model loading."""
    requested_path = (
        manifest_path if manifest_path.is_absolute() else Path.cwd() / manifest_path
    )
    if requested_path.is_symlink() or not requested_path.is_file():
        raise QualificationIdentityError(
            f"prelaunch manifest is missing or symlinked: {requested_path}"
        )
    resolved_path = requested_path.resolve()
    read_only = resolved_path.stat().st_mode & 0o222 == 0
    require_identity(
        receipt,
        name="prelaunch_manifest_read_only",
        actual=read_only,
        expected=True,
    )
    actual_sha256 = sha256_file(resolved_path)
    require_identity(
        receipt,
        name="prelaunch_manifest_sha256",
        actual=actual_sha256,
        expected=expected_sha256,
    )
    try:
        payload = json.loads(resolved_path.read_text())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise QualificationIdentityError(
            f"could not parse prelaunch manifest: {type(error).__name__}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise QualificationIdentityError(
            "prelaunch manifest must contain one JSON object"
        )
    require_identity(
        receipt,
        name="prelaunch_manifest_schema",
        actual=payload.get("schema"),
        expected=_PRELAUNCH_SCHEMA,
    )
    metadata_present = all(
        isinstance(payload.get(name), str) and bool(payload[name].strip())
        for name in ("run_id", "created_at_utc")
    )
    require_identity(
        receipt,
        name="prelaunch_manifest_metadata",
        actual=metadata_present,
        expected=True,
    )
    external = all(
        not _path_is_within(resolved_path, root)
        for root in (submission, runner_root, scorer_repo)
    )
    require_identity(
        receipt,
        name="prelaunch_manifest_path_external",
        actual=external,
        expected=True,
    )

    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise QualificationIdentityError("prelaunch manifest observation is missing")
    prefix_value = observation.get("paths", {}).get("external_pycache_prefix")
    if not isinstance(prefix_value, str) or not prefix_value:
        raise QualificationIdentityError(
            "prelaunch manifest external pycache prefix is missing"
        )
    external_pycache_prefix = Path(prefix_value)
    if external_pycache_prefix.is_symlink() or not external_pycache_prefix.is_dir():
        raise QualificationIdentityError(
            "prelaunch external pycache prefix is missing or symlinked: "
            f"{external_pycache_prefix}"
        )
    current = prelaunch_observation(
        submission=submission,
        runner_root=runner_root,
        scorer_repo=scorer_repo,
        external_pycache_prefix=external_pycache_prefix,
    )
    require_identity(
        receipt,
        name="prelaunch_manifest_bound_observation",
        actual=current,
        expected=observation,
    )
    cache_inventory = current["cache_inventory"]
    cache_free = all(entries == [] for entries in cache_inventory.values())
    caller_controls = current["environment"]
    controls_exact = (
        caller_controls["PYTHONDONTWRITEBYTECODE"] == "1"
        and caller_controls["PYTHONPYCACHEPREFIX"]
        == current["paths"]["external_pycache_prefix"]
        and caller_controls["sys_dont_write_bytecode"] is True
        and caller_controls["sys_pycache_prefix"]
        == current["paths"]["external_pycache_prefix"]
    )
    require_identity(
        receipt,
        name="prelaunch_cache_inventory_empty",
        actual=cache_free,
        expected=True,
    )
    require_identity(
        receipt,
        name="prelaunch_bytecode_controls",
        actual=controls_exact,
        expected=True,
    )
    provenance = {
        "schema": "neural-horner-frozen-battery-cache-provenance-v2",
        "verified_before_model_load": True,
        "pre_run_inventory_recorded": True,
        "prelaunch_manifest_path": str(resolved_path),
        "prelaunch_manifest_sha256": actual_sha256,
        "prelaunch_manifest_expected_sha256": expected_sha256,
        "prelaunch_manifest_read_only": read_only,
        "prelaunch_manifest_payload": payload,
        "submission_cache_entries": cache_inventory["submission"],
        "runner_cache_entries": cache_inventory["runner"],
        "scorer_cache_entries": cache_inventory["scorer"],
        "external_pycache_prefix_entries": cache_inventory["external_pycache_prefix"],
        "external_pycache_prefix": current["paths"]["external_pycache_prefix"],
        "fresh_external_pycache_prefix": (
            cache_inventory["external_pycache_prefix"] == []
        ),
        "bytecode_cache_writes_disabled": controls_exact,
    }
    receipt["cache_provenance"] = provenance
    return provenance


def finalize_cache_provenance(receipt: dict[str, Any]) -> bool:
    """Recheck bound cache state without hiding a post-run failure."""
    provenance = receipt.get("cache_provenance")
    if provenance is None:
        return True
    if not isinstance(provenance, dict) or not provenance.get(
        "verified_before_model_load"
    ):
        return False
    try:
        payload = provenance["prelaunch_manifest_payload"]
        paths = payload["observation"]["paths"]
        manifest_path = Path(provenance["prelaunch_manifest_path"])
        post = {
            "prelaunch_manifest_sha256": sha256_file(manifest_path),
            "prelaunch_manifest_read_only": (manifest_path.stat().st_mode & 0o222 == 0),
            "submission_cache_entries": bytecode_cache_entries(
                Path(paths["submission"])
            ),
            "runner_cache_entries": bytecode_cache_entries(Path(paths["runner_root"])),
            "scorer_cache_entries": bytecode_cache_entries(
                Path(paths["scorer_repo"]),
                ignored_top_level=frozenset({".git"}),
            ),
            "external_pycache_prefix_entries": tree_entries(
                Path(paths["external_pycache_prefix"])
            ),
            "bytecode_cache_writes_disabled": (
                sys.dont_write_bytecode
                and os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
            ),
        }
    except BaseException as error:
        provenance["post_run_error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        provenance["stable_during_run"] = False
        return False
    stable = (
        post["prelaunch_manifest_sha256"] == provenance["prelaunch_manifest_sha256"]
        and post["prelaunch_manifest_read_only"] is True
        and post["submission_cache_entries"] == []
        and post["runner_cache_entries"] == []
        and post["scorer_cache_entries"] == []
        and post["external_pycache_prefix_entries"] == []
        and post["bytecode_cache_writes_disabled"] is True
    )
    provenance["post_run"] = post
    provenance["stable_during_run"] = stable
    return stable


def json_scalar(value: object) -> int | float | str | bool | None:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return repr(value)


def source_identity() -> dict[str, dict[str, str]]:
    """Bind this runner, its local helper, and the official scorer sources."""
    sources = {
        "battery_runner": Path(__file__).resolve(),
        "submission_utils": Path(__file__).with_name("submission_utils.py").resolve(),
    }
    for module_name in _OFFICIAL_MODULES:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise RuntimeError(f"official scorer module unavailable: {module_name}")
        sources[module_name] = Path(spec.origin).resolve()
    return {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sources.items()
    }


def scorer_git_state(source_path: Path) -> dict[str, Any]:
    """Return commit, cleanliness, and detached state for the scorer checkout."""
    unavailable = {
        "available": False,
        "commit": None,
        "clean": None,
        "detached_head": None,
        "branch": None,
        "repository": None,
        "dirty_paths": [],
    }
    try:
        commit_result = subprocess.run(
            ["git", "-C", str(source_path.parent), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return unavailable
    commit = commit_result.stdout.strip()
    if commit_result.returncode != 0 or len(commit) != 40:
        return unavailable
    root_result = subprocess.run(
        [
            "git",
            "-C",
            str(source_path.parent),
            "rev-parse",
            "--show-toplevel",
        ],
        capture_output=True,
        check=False,
        text=True,
    )
    repository = (
        str(Path(root_result.stdout.strip()).resolve())
        if root_result.returncode == 0 and root_result.stdout.strip()
        else None
    )
    try:
        branch_result = subprocess.run(
            [
                "git",
                "-C",
                str(source_path.parent),
                "symbolic-ref",
                "--quiet",
                "--short",
                "HEAD",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
        status_result = subprocess.run(
            [
                "git",
                "-C",
                str(source_path.parent),
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return {**unavailable, "commit": commit}
    if status_result.returncode != 0:
        return {**unavailable, "commit": commit}
    dirty_paths = status_result.stdout.splitlines()
    branch = branch_result.stdout.strip() if branch_result.returncode == 0 else None
    return {
        "available": True,
        "commit": commit,
        "clean": not dirty_paths,
        "detached_head": branch is None,
        "branch": branch,
        "repository": repository,
        "dirty_paths": dirty_paths,
    }


def scorer_git_commit(source_path: Path) -> str | None:
    """Compatibility accessor for the scorer checkout commit."""
    return scorer_git_state(source_path)["commit"]


def require_identity(
    receipt: dict[str, Any],
    *,
    name: str,
    actual: object,
    expected: object,
) -> None:
    passed = actual == expected
    receipt.setdefault("qualification_identity_checks", []).append(
        {
            "name": name,
            "actual": actual,
            "expected": expected,
            "passed": passed,
        }
    )
    if not passed:
        raise QualificationIdentityError(
            f"{name} mismatch: expected {expected!r}, got {actual!r}"
        )


def model_environment(model: object) -> dict[str, Any]:
    """Record the execution facts that can affect a candidate receipt."""
    parameters = []
    parameter_owner = model
    if not callable(getattr(parameter_owner, "parameters", None)):
        nested_model = getattr(model, "model", None)
        if nested_model is not None:
            parameter_owner = nested_model
    parameter_method = getattr(parameter_owner, "parameters", None)
    if callable(parameter_method):
        try:
            parameters = list(parameter_method())
        except (RuntimeError, TypeError):
            parameters = []
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    parameter_devices = sorted({str(parameter.device) for parameter in parameters})
    cuda_available = torch.cuda.is_available()
    return {
        "model_device": str(getattr(model, "device", "unreported")),
        "torch_version": torch.__version__,
        "torch_default_dtype": str(torch.get_default_dtype()),
        "parameter_owner_type": type(parameter_owner).__name__,
        "parameter_count_observed": sum(parameter.numel() for parameter in parameters),
        "parameter_dtypes": parameter_dtypes,
        "parameter_devices": parameter_devices,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_count": torch.cuda.device_count() if cuda_available else 0,
        "cuda_device_names": (
            [
                torch.cuda.get_device_name(device_index)
                for device_index in range(torch.cuda.device_count())
            ]
            if cuda_available
            else []
        ),
        "mps_built": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_built()
        ),
        "mps_available": bool(
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "logical_cpu_count": os.cpu_count(),
    }


def is_prime(n, rng):
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    for a in list(_SMALL) + [rng.randrange(2, n - 1) for _ in range(16)]:
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def prime_3mod4(lo, hi, rng):
    for _ in range(100000):
        c = rng.randint(lo, hi) | 3  # ensures odd and == 3 mod 4
        if c % 4 == 3 and is_prime(c, rng):
            return c
    return None


def generate_battery(L: int, n: int, seed: int = 20260627):
    """Materialize the original six-family battery without changing RNG order."""
    rng = random.Random(seed)
    lo, hi = 1 << (L - 2), (1 << L) - 1
    p = next(c for c in iter(lambda: rng.randint(lo, hi), None) if is_prime(c, rng))
    operand_width = 2 * L

    categories = {}
    fib = [1, 2]
    while fib[-1] < (1 << operand_width):
        fib.append(fib[-1] + fib[-2])
    categories["fibonacci values"] = [
        (rng.choice(fib), rng.choice(fib)) for _ in range(n)
    ]

    fermat = [
        (1 << (1 << k)) + 1
        for k in range(0, 12)
        if (1 << (1 << k)) + 1 < (1 << operand_width)
    ]
    categories["fermat numbers"] = [
        (rng.choice(fermat), rng.randrange(0, 1 << operand_width)) for _ in range(n)
    ]

    alternating_5 = int("01" * (operand_width // 2), 2)
    alternating_a = int("10" * (operand_width // 2), 2)
    categories["alternating bits"] = [
        (
            rng.choice([alternating_5, alternating_a]),
            rng.choice(
                [
                    alternating_5,
                    alternating_a,
                    rng.randrange(0, 1 << operand_width),
                ]
            ),
        )
        for _ in range(n)
    ]

    def fixed_weight(half):
        positions = rng.sample(range(operand_width), half)
        value = 0
        for position in positions:
            value |= 1 << position
        return value

    categories["fixed Hamming weight W/2"] = [
        (fixed_weight(operand_width // 2), fixed_weight(operand_width // 2))
        for _ in range(n)
    ]

    straddle = []
    for _ in range(n):
        a = rng.randrange(
            1 << (operand_width // 2 - 1),
            1 << (operand_width // 2),
        )
        multiple = rng.randrange(1, 1 << (operand_width // 2))
        target = multiple * p + rng.randrange(-3, 4)
        b = max(1, target // a)
        straddle.append((a, b))
    categories["product straddles k*p"] = straddle

    p_3mod4 = prime_3mod4(lo, hi, rng) or p
    categories["prime=3mod4 (random ops)"] = [
        (
            rng.randrange(0, 1 << operand_width),
            rng.randrange(0, 1 << operand_width),
        )
        for _ in range(n)
    ]
    return p, p_3mod4, operand_width, categories


def fixture_rows(
    selected: list[tuple[str, int, list[tuple[int, int]]]],
    *,
    orientation: str | None = None,
) -> list[dict[str, str]]:
    rows = []
    for family, p, cases in selected:
        for a, b in cases:
            if orientation == "swapped":
                a, b = b, a
            rows.append(
                {
                    "family": family,
                    "a": str(a),
                    "b": str(b),
                    "p": str(p),
                }
            )
    return rows


def fixture_statistics(
    selected: list[tuple[str, int, list[tuple[int, int]]]],
    orientations: tuple[str, ...],
) -> dict[str, Any]:
    """Count rows and numerical duplicates before and after orientation."""
    base_rows = fixture_rows(selected)
    base_numerical = {(row["a"], row["b"], row["p"]) for row in base_rows}
    oriented_rows = {
        orientation: fixture_rows(selected, orientation=orientation)
        for orientation in orientations
    }
    effective_numerical = {
        (row["a"], row["b"], row["p"])
        for rows in oriented_rows.values()
        for row in rows
    }
    labelled_oriented_numerical = {
        (orientation, row["a"], row["b"], row["p"])
        for orientation, rows in oriented_rows.items()
        for row in rows
    }
    oriented_row_count = sum(len(rows) for rows in oriented_rows.values())
    return {
        "base_row_count": len(base_rows),
        "base_unique_numerical_case_count": len(base_numerical),
        "base_duplicate_numerical_row_count": len(base_rows) - len(base_numerical),
        "oriented_row_count": oriented_row_count,
        "oriented_unique_effective_numerical_case_count": len(effective_numerical),
        "oriented_effective_duplicate_row_count": (
            oriented_row_count - len(effective_numerical)
        ),
        "oriented_unique_labelled_numerical_case_count": len(
            labelled_oriented_numerical
        ),
        "oriented_labelled_duplicate_row_count": (
            oriented_row_count - len(labelled_oriented_numerical)
        ),
        "base_fixture_sha256": canonical_json_sha256(base_rows),
        "oriented_fixture_sha256": {
            orientation: canonical_json_sha256(rows)
            for orientation, rows in oriented_rows.items()
        },
        "combined_oriented_fixture_sha256": canonical_json_sha256(oriented_rows),
    }


def family_inventory(
    p: int,
    p2: int,
    categories: dict[str, list[tuple[int, int]]],
) -> list[dict[str, Any]]:
    inventory = []
    for family, cases in categories.items():
        modulus = p2 if family.startswith("prime=3mod4") else p
        unique = {(a, b, modulus) for a, b in cases}
        inventory.append(
            {
                "family": family,
                "rows": len(cases),
                "unique_numerical_cases": len(unique),
                "duplicate_numerical_rows": len(cases) - len(unique),
                "modulus_bits": modulus.bit_length(),
                "modulus_sha256": hashlib.sha256(str(modulus).encode()).hexdigest(),
            }
        )
    return inventory


def bit_length_statistics(values: list[int]) -> dict[str, float | int | None]:
    lengths = [value.bit_length() for value in values]
    if not lengths:
        return {"minimum": None, "maximum": None, "mean": None}
    return {
        "minimum": min(lengths),
        "maximum": max(lengths),
        "mean": fmean(lengths),
    }


def quotient_correlated_diagnostics(
    p: int,
    cases: list[tuple[int, int]],
) -> dict[str, Any]:
    """Describe what the legacy product-straddles generator actually emits."""
    products = [a * b for a, b in cases]
    floor_quotients = [product // p for product in products]
    nearest_distances = [min(product % p, (-product) % p) for product in products]
    near_count = sum(distance <= 3 for distance in nearest_distances)
    return {
        "family": _PRODUCT_STRADDLES_FAMILY,
        "legacy_name_preserved": True,
        "semantics": (
            "The frozen generator first samples a quotient-like multiple k, "
            "forms k*p plus an offset in [-3,3], then sets b=floor(target/a). "
            "Flooring can move a*b far from k*p; this is quotient-correlated "
            "boundary construction, not a guarantee that a*b is within 3 of "
            "a multiple of p."
        ),
        "row_count": len(cases),
        "distance_to_nearest_multiple_at_most_3_count": near_count,
        "distance_to_nearest_multiple_at_most_3_denominator": len(cases),
        "distance_to_nearest_multiple_at_most_3_fraction": (
            near_count / len(cases) if cases else None
        ),
        "exact_multiple_count": sum(distance == 0 for distance in nearest_distances),
        "a_bit_length": bit_length_statistics([a for a, _ in cases]),
        "b_bit_length": bit_length_statistics([b for _, b in cases]),
        "product_bit_length": bit_length_statistics(products),
        "floor_quotient_bit_length": bit_length_statistics(floor_quotients),
        "nearest_boundary_distance_bit_length": bit_length_statistics(
            nearest_distances
        ),
    }


def scorer_batch_size(model: object) -> tuple[object, object]:
    """Use the pinned scorer's exact max(1, model.max_batch_size()) rule."""
    declared = model.max_batch_size()
    return declared, max(1, declared)


def run_cases(
    model,
    p: int,
    cases: list[tuple[int, int]],
    *,
    output_base: int | str = 2,
    batch_size: int | None = None,
    stop_on_failure: bool = False,
):
    if batch_size is None:
        _, batch_size = scorer_batch_size(model)

    ok, fails, case_results = 0, [], []
    output_count = 0
    actual_batch_sizes: list[int] = []
    outer_output_sized = True
    batch_contract_exact = True
    stopped_early = False

    for batch_start in range(0, len(cases), batch_size):
        batch_cases = cases[batch_start : batch_start + batch_size]
        actual_batch_sizes.append(len(batch_cases))
        inputs = [
            (
                model.preprocess_a(str(a)),
                model.preprocess_b(str(b)),
                model.preprocess_p(str(p)),
            )
            for a, b in batch_cases
        ]
        raw_outputs = model.predict_digits_batch(inputs)
        if not isinstance(raw_outputs, Sized):
            outer_output_sized = False
            batch_contract_exact = False
            outputs: list[object] = []
        else:
            batch_output_count = len(raw_outputs)
            output_count += batch_output_count
            batch_contract_exact &= batch_output_count == len(batch_cases)
            outputs = list(raw_outputs)

        batch_failed = not batch_contract_exact
        for offset, (a, b) in enumerate(batch_cases):
            case_index = batch_start + offset
            output_present = offset < len(outputs)
            raw_digits = outputs[offset] if output_present else None
            malformed_message = None
            predicted = None
            output_alphabet_exact = False
            if output_present:
                try:
                    actual_base = resolve_base(output_base, p)
                    output_alphabet_exact = isinstance(raw_digits, list) and all(
                        isinstance(digit, int)
                        and not isinstance(digit, bool)
                        and 0 <= digit < actual_base
                        for digit in raw_digits
                    )
                    predicted = decode_answer(
                        raw_digits,
                        base=output_base,
                        prime=p,
                        is_tier_zero=False,
                    )
                except MalformedOutput as error:
                    malformed_message = str(error)
            truth = (a * b) % p
            correct = predicted == truth and malformed_message is None
            batch_failed |= not correct
            if correct:
                ok += 1
            elif len(fails) < 3:
                fails.append(
                    {
                        "case_index": case_index,
                        "a_bits": a.bit_length(),
                        "b_bits": b.bit_length(),
                        "output_present": output_present,
                        "output_type_exact": output_present
                        and isinstance(raw_digits, list),
                        "output_alphabet_exact": output_alphabet_exact,
                        "malformed_output": malformed_message,
                    }
                )
            case_results.append(
                {
                    "case_index": case_index,
                    "case_id": hashlib.sha256(f"{a}:{b}:{p}".encode()).hexdigest()[:20],
                    "a_bits": a.bit_length(),
                    "b_bits": b.bit_length(),
                    "p_bits": p.bit_length(),
                    "output_present": output_present,
                    "output_type_exact": output_present
                    and isinstance(raw_digits, list),
                    "output_digit_count": (
                        len(raw_digits) if isinstance(raw_digits, list) else None
                    ),
                    "output_alphabet_exact": output_alphabet_exact,
                    "malformed_output": malformed_message,
                    "correct": correct,
                    "predicted": str(predicted) if not correct else None,
                    "expected": str(truth) if not correct else None,
                }
            )

        if stop_on_failure and batch_failed:
            stopped_early = batch_start + len(batch_cases) < len(cases)
            break

    completed = len(case_results) == len(cases)
    output_count_exact = completed and batch_contract_exact
    return {
        "correct": ok,
        "total": len(cases),
        "attempted": len(case_results),
        "completed": completed,
        "stopped_early": stopped_early,
        "failures": fails,
        "cases": case_results,
        "outer_output_sized": outer_output_sized,
        "batch_contract_exact": batch_contract_exact,
        "output_count": output_count,
        "expected_output_count": len(cases),
        "output_count_exact": output_count_exact,
        "extra_output_count": max(0, output_count - len(case_results)),
        "actual_batch_sizes": actual_batch_sizes,
        "actual_batch_count": len(actual_batch_sizes),
    }


def finalize_artifact_identity(
    receipt: dict[str, Any],
    submission: Path,
    before_identity: dict[str, str],
) -> bool:
    """Post-hash without allowing a rehash error to erase the run receipt."""
    try:
        after_identity = artifact_identity(submission)
    except Exception as error:
        receipt.update(
            {
                "artifact_sha256_after": None,
                "artifact_unchanged_during_run": False,
                "artifact_rehash_error": {
                    "type": type(error).__name__,
                    "message": str(error),
                },
            }
        )
        return False
    unchanged = before_identity == after_identity
    receipt.update(
        {
            "artifact_sha256_after": after_identity,
            "artifact_set_sha256_after": canonical_json_sha256(after_identity),
            "artifact_unchanged_during_run": unchanged,
        }
    )
    return unchanged


def model_state_width(model: Any) -> int:
    """Resolve the state-width declaration used to generate the battery."""
    width = getattr(model, "L", None)
    if width is None:
        width = getattr(model, "width", 32)
    if type(width) is not int or width <= 0:
        raise ValueError("model state width must be a positive integer")
    return width


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument(
        "--orientation",
        choices=("original", "swapped", "both"),
        default="original",
    )
    ap.add_argument(
        "--family",
        action="append",
        default=[],
        help="case-insensitive family-name substring; may be repeated",
    )
    ap.add_argument("--f11-only", action="store_true")
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--require-exact", action="store_true")
    ap.add_argument(
        "--prelaunch-manifest",
        type=Path,
        help="read-only cache-inventory manifest created before Python launch",
    )
    ap.add_argument(
        "--require-prelaunch-manifest-sha256",
        help="fail before model loading unless the prelaunch manifest has this hash",
    )
    ap.add_argument(
        "--require-scorer-commit",
        help="fail before inference unless the scorer checkout has this commit",
    )
    ap.add_argument(
        "--require-clean-detached-scorer",
        action="store_true",
        help="fail before inference unless the scorer Git checkout is clean/detached",
    )
    ap.add_argument(
        "--require-base-fixture-sha256",
        help="fail before inference unless the generated base fixture has this hash",
    )
    ap.add_argument(
        "--require-oriented-fixture-sha256",
        help="fail before inference unless the selected/oriented fixture has this hash",
    )
    ap.add_argument(
        "--qualification-l2048-n128",
        action="store_true",
        help=(
            "lock scorer and fixture identities for the frozen L=2048, n=128, "
            "all-family, both-orientation qualification battery"
        ),
    )
    args = ap.parse_args()
    manifest_arguments = (
        args.prelaunch_manifest is not None,
        args.require_prelaunch_manifest_sha256 is not None,
    )
    if manifest_arguments[0] != manifest_arguments[1]:
        ap.error(
            "--prelaunch-manifest and --require-prelaunch-manifest-sha256 "
            "must be supplied together"
        )
    if args.require_prelaunch_manifest_sha256 is not None and (
        len(args.require_prelaunch_manifest_sha256) != 64
        or any(
            character not in "0123456789abcdef"
            for character in args.require_prelaunch_manifest_sha256
        )
    ):
        ap.error("--require-prelaunch-manifest-sha256 must be 64 lowercase hex digits")
    if args.qualification_l2048_n128:
        qualification_errors = []
        if not args.require_exact:
            qualification_errors.append("--require-exact")
        if args.json_out is None:
            qualification_errors.append("--json-out PATH")
        if args.prelaunch_manifest is None:
            qualification_errors.append("--prelaunch-manifest PATH")
        if args.require_prelaunch_manifest_sha256 is None:
            qualification_errors.append("--require-prelaunch-manifest-sha256 SHA256")
        if args.n != 128:
            qualification_errors.append("--n 128")
        if args.orientation != "both":
            qualification_errors.append("--orientation both")
        if args.family:
            qualification_errors.append("no --family filters")
        if args.f11_only:
            qualification_errors.append("no --f11-only filter")
        if args.require_scorer_commit not in (None, _PINNED_SCORER_COMMIT):
            qualification_errors.append(
                f"--require-scorer-commit {_PINNED_SCORER_COMMIT}"
            )
        if args.require_base_fixture_sha256 not in (
            None,
            _FROZEN_L2048_N128_BASE_SHA256,
        ):
            qualification_errors.append(
                f"--require-base-fixture-sha256 {_FROZEN_L2048_N128_BASE_SHA256}"
            )
        if args.require_oriented_fixture_sha256 not in (
            None,
            _FROZEN_L2048_N128_BOTH_SHA256,
        ):
            qualification_errors.append(
                "--require-oriented-fixture-sha256 " + _FROZEN_L2048_N128_BOTH_SHA256
            )
        if qualification_errors:
            ap.error(
                "--qualification-l2048-n128 requires: "
                + ", ".join(qualification_errors)
            )
    N = args.n
    if N <= 0:
        ap.error("--n must be positive")
    sub = Path(args.submission).resolve()
    orientations = (
        ("original", "swapped") if args.orientation == "both" else (args.orientation,)
    )
    manifest_path = sub / "manifest.json"
    try:
        preloaded_manifest = json.loads(manifest_path.read_text())
        before_identity = artifact_identity(sub)
        sources = source_identity()
    except Exception as error:
        receipt = {
            "status": "preload_error",
            "submission": str(sub),
            "all_exact": False,
            "error": {"type": type(error).__name__, "message": str(error)},
        }
        if args.json_out:
            write_receipt(args.json_out, receipt)
        print(f"ERROR: pre-load provenance failed: {type(error).__name__}: {error}")
        return 2

    scorer_source = Path(sources["modchallenge.evaluation.decoder"]["path"])
    scorer_state = scorer_git_state(scorer_source)
    required_scorer_commit = (
        _PINNED_SCORER_COMMIT
        if args.qualification_l2048_n128
        else args.require_scorer_commit
    )
    required_base_fixture = (
        _FROZEN_L2048_N128_BASE_SHA256
        if args.qualification_l2048_n128
        else args.require_base_fixture_sha256
    )
    required_oriented_fixture = (
        _FROZEN_L2048_N128_BOTH_SHA256
        if args.qualification_l2048_n128
        else args.require_oriented_fixture_sha256
    )
    require_clean_detached = (
        args.qualification_l2048_n128 or args.require_clean_detached_scorer
    )
    identity_gates_requested = any(
        (
            required_scorer_commit,
            required_base_fixture,
            required_oriented_fixture,
            require_clean_detached,
            args.prelaunch_manifest,
        )
    )
    receipt = {
        "status": "running",
        "submission": str(sub),
        "entry_class": preloaded_manifest.get("entry_class"),
        "output_base": preloaded_manifest.get("output_base"),
        "artifact_sha256": before_identity,
        "artifact_set_sha256": canonical_json_sha256(before_identity),
        "artifact_hashed_before_model_load": True,
        "source_identity": sources,
        "source_identity_sha256": canonical_json_sha256(sources),
        "runner_sha256": sources["battery_runner"]["sha256"],
        "helper_runner_sha256": sources["submission_utils"]["sha256"],
        "official_decoder_sha256": sources["modchallenge.evaluation.decoder"]["sha256"],
        "official_scorer_git_commit": scorer_state["commit"],
        "official_scorer_git_state": scorer_state,
        "evaluation_scope": (
            "Frozen development qualification gate using the manifest-selected "
            "local loader."
            if args.qualification_l2048_n128
            else "Frozen development diagnostic using the manifest-selected local "
            "loader."
        ),
        "scope_nonclaims": (
            "This is not the official scorer pipeline, a sealed battery, or an "
            "official sandbox/runtime receipt. The official pipeline remains a "
            "separate qualification gate."
        ),
        "identity_mode": (
            "qualification_locked"
            if args.qualification_l2048_n128
            else "identity_locked_diagnostic"
            if identity_gates_requested
            else "diagnostic_unlocked"
        ),
        "qualification_identity_requirements": {
            "preset": (
                "l2048_n128_all_families_both_orientations"
                if args.qualification_l2048_n128
                else None
            ),
            "scorer_commit": required_scorer_commit,
            "clean_detached_scorer": require_clean_detached,
            "base_fixture_sha256": required_base_fixture,
            "oriented_fixture_sha256": required_oriented_fixture,
            "require_exact": args.require_exact,
            "receipt_path": str(args.json_out) if args.json_out else None,
            "prelaunch_manifest_path": (
                str(args.prelaunch_manifest.resolve())
                if args.prelaunch_manifest is not None
                else None
            ),
            "prelaunch_manifest_sha256": (args.require_prelaunch_manifest_sha256),
        },
        "qualification_identity_checks": [],
        "seed": _SEED,
        "n_per_family": N,
        "selection": {
            "orientation_request": args.orientation,
            "resolved_orientations": list(orientations),
            "family_filters": list(args.family),
            "f11_only": args.f11_only,
            "f11": {
                "exponent": 2048,
                "bit_length": 2049,
                "decimal_sha256": hashlib.sha256(
                    str((1 << (1 << 11)) + 1).encode()
                ).hexdigest(),
            },
        },
        "expected_orientations": len(orientations),
        "orientations": {},
    }
    if args.json_out:
        write_receipt(args.json_out, receipt)

    all_exact = False
    counts_exact = False
    artifact_unchanged = False
    cache_provenance_stable = False
    grand_ok = grand_total = 0
    expected_per_orientation = 0
    selected: list[tuple[str, int, list[tuple[int, int]]]] = []
    observed_batch_sizes: list[int] = []
    execution_error: BaseException | None = None
    try:
        if required_scorer_commit is not None:
            require_identity(
                receipt,
                name="official_scorer_git_commit",
                actual=scorer_state["commit"],
                expected=required_scorer_commit,
            )
        if require_clean_detached:
            require_identity(
                receipt,
                name="official_scorer_git_checkout_available",
                actual=scorer_state["available"],
                expected=True,
            )
            require_identity(
                receipt,
                name="official_scorer_git_checkout_clean",
                actual=scorer_state["clean"],
                expected=True,
            )
            require_identity(
                receipt,
                name="official_scorer_git_checkout_detached",
                actual=scorer_state["detached_head"],
                expected=True,
            )
        if args.qualification_l2048_n128:
            require_identity(
                receipt,
                name="qualification_n_per_family",
                actual=N,
                expected=128,
            )
            require_identity(
                receipt,
                name="qualification_orientation_request",
                actual=args.orientation,
                expected="both",
            )
            require_identity(
                receipt,
                name="qualification_family_filters",
                actual=list(args.family),
                expected=[],
            )
            require_identity(
                receipt,
                name="qualification_f11_only",
                actual=args.f11_only,
                expected=False,
            )
        if args.prelaunch_manifest is not None:
            scorer_repository = scorer_state.get("repository")
            if not isinstance(scorer_repository, str) or not scorer_repository:
                raise QualificationIdentityError(
                    "could not resolve scorer repository for prelaunch manifest"
                )
            verify_prelaunch_manifest(
                receipt,
                manifest_path=args.prelaunch_manifest,
                expected_sha256=args.require_prelaunch_manifest_sha256,
                submission=sub,
                runner_root=Path(__file__).resolve().parent,
                scorer_repo=Path(scorer_repository),
            )
            if args.json_out:
                write_receipt(args.json_out, receipt)
        loaded_sub, manifest, _, model = load_submission(sub)
        if loaded_sub != sub:
            raise RuntimeError(f"loader changed submission path: {loaded_sub} != {sub}")
        receipt["manifest_unchanged_during_load"] = manifest == preloaded_manifest
        receipt["loaded_manifest_sha256"] = canonical_json_sha256(manifest)
        output_base = manifest["output_base"]
        L = model_state_width(model)
        declared_batch_size, batch_size = scorer_batch_size(model)
        environment = model_environment(model)
        receipt.update(
            {
                "entry_class": manifest["entry_class"],
                "output_base": output_base,
                "model_L": L,
                "declared_max_batch_size": json_scalar(declared_batch_size),
                "declared_max_batch_size_type": type(declared_batch_size).__name__,
                "effective_scorer_batch_size": json_scalar(batch_size),
                "effective_scorer_batch_size_type": type(batch_size).__name__,
                "environment": environment,
                "environment_sha256": canonical_json_sha256(environment),
            }
        )
        if args.qualification_l2048_n128:
            require_identity(
                receipt,
                name="qualification_model_L",
                actual=L,
                expected=2048,
            )

        p, p2, operand_width, categories = generate_battery(L, N, seed=_SEED)
        print(
            f"submission={sub.name} entry={manifest['entry_class']} L={L} "
            f"prime={p.bit_length()}b operand_width={operand_width}b n={N}  "
            "[FROZEN DEVELOPMENT families]"
        )
        all_generated = [
            (
                name,
                p2 if name.startswith("prime=3mod4") else p,
                cases,
            )
            for name, cases in categories.items()
        ]
        receipt["family_inventory"] = family_inventory(p, p2, categories)
        receipt["base_battery_sha256"] = canonical_json_sha256(
            fixture_rows(all_generated)
        )
        if required_base_fixture is not None:
            require_identity(
                receipt,
                name="base_battery_sha256",
                actual=receipt["base_battery_sha256"],
                expected=required_base_fixture,
            )
        receipt["family_semantics_diagnostics"] = {
            _PRODUCT_STRADDLES_FAMILY: quotient_correlated_diagnostics(
                p, categories[_PRODUCT_STRADDLES_FAMILY]
            )
        }

        f11 = (1 << (1 << 11)) + 1
        selection_inventory = []
        for name, cases in categories.items():
            family_match = not args.family or any(
                token.lower() in name.lower() for token in args.family
            )
            filtered_cases = cases
            if args.f11_only:
                filtered_cases = [
                    (a, b) for a, b in filtered_cases if a == f11 or b == f11
                ]
            selected_rows = len(filtered_cases) if family_match else 0
            selected_unique_rows = len(set(filtered_cases)) if family_match else 0
            selection_inventory.append(
                {
                    "family": name,
                    "generated_rows": len(cases),
                    "family_filter_match": family_match,
                    "f11_rows": sum(a == f11 or b == f11 for a, b in cases),
                    "selected_rows": selected_rows,
                    "selected_unique_numerical_cases": selected_unique_rows,
                    "selected_duplicate_numerical_rows": (
                        selected_rows - selected_unique_rows
                    ),
                }
            )
            if family_match and filtered_cases:
                pp = p2 if name.startswith("prime=3mod4") else p
                selected.append((name, pp, filtered_cases))
        receipt["selection"]["family_inventory"] = selection_inventory

        if not selected:
            receipt.update(
                {
                    "status": "empty_selection",
                    "all_exact": False,
                    "grand_correct": 0,
                    "grand_total": 0,
                }
            )
            print("ERROR: filters selected no battery cases")
        else:
            fixture_info = fixture_statistics(selected, orientations)
            receipt["fixture_statistics"] = fixture_info
            receipt["selected_battery_sha256"] = fixture_info["base_fixture_sha256"]
            receipt["selected_oriented_battery_sha256"] = fixture_info[
                "combined_oriented_fixture_sha256"
            ]
            if required_oriented_fixture is not None:
                require_identity(
                    receipt,
                    name="selected_oriented_battery_sha256",
                    actual=receipt["selected_oriented_battery_sha256"],
                    expected=required_oriented_fixture,
                )
            if args.qualification_l2048_n128:
                require_identity(
                    receipt,
                    name="original_oriented_fixture_sha256",
                    actual=fixture_info["oriented_fixture_sha256"].get("original"),
                    expected=_FROZEN_L2048_N128_ORIGINAL_SHA256,
                )
                require_identity(
                    receipt,
                    name="swapped_oriented_fixture_sha256",
                    actual=fixture_info["oriented_fixture_sha256"].get("swapped"),
                    expected=_FROZEN_L2048_N128_SWAPPED_SHA256,
                )
                for count_name, expected_count in (
                    ("base_row_count", 768),
                    ("base_unique_numerical_case_count", 693),
                    ("oriented_row_count", 1536),
                    ("oriented_unique_effective_numerical_case_count", 1382),
                    ("oriented_unique_labelled_numerical_case_count", 1386),
                ):
                    require_identity(
                        receipt,
                        name=count_name,
                        actual=fixture_info[count_name],
                        expected=expected_count,
                    )
            expected_per_orientation = fixture_info["base_row_count"]
            receipt["expected_cases_per_orientation"] = expected_per_orientation
            receipt["expected_grand_total"] = expected_per_orientation * len(
                orientations
            )
            if args.json_out:
                write_receipt(args.json_out, receipt)

        stop_all = not selected
        for orientation in orientations:
            if stop_all:
                break
            orientation_ok = orientation_total = 0
            family_results = []
            oriented_rows = fixture_rows(selected, orientation=orientation)
            receipt["orientations"][orientation] = {
                "status": "running",
                "correct": 0,
                "total": 0,
                "fixture_sha256": canonical_json_sha256(oriented_rows),
                "row_count": len(oriented_rows),
                "unique_numerical_case_count": len(
                    {(row["a"], row["b"], row["p"]) for row in oriented_rows}
                ),
                "duplicate_numerical_row_count": len(oriented_rows)
                - len({(row["a"], row["b"], row["p"]) for row in oriented_rows}),
                "families": family_results,
            }
            print(f"orientation={orientation}")
            for name, pp, cases in selected:
                evaluated = (
                    cases if orientation == "original" else [(b, a) for a, b in cases]
                )
                result = run_cases(
                    model,
                    pp,
                    evaluated,
                    output_base=output_base,
                    batch_size=batch_size,
                    stop_on_failure=args.require_exact,
                )
                observed_batch_sizes.extend(result["actual_batch_sizes"])
                orientation_ok += result["correct"]
                orientation_total += result["total"]
                family_exact = (
                    result["correct"] == result["total"]
                    and result["output_count_exact"]
                )
                flag = "OK" if family_exact else "FAIL"
                suffix = "" if family_exact else f"  failed cases={result['failures']}"
                print(
                    f"  [{flag}] {name}: {result['correct']}/{result['total']}{suffix}"
                )
                family_results.append({"family": name, **result})
                receipt["orientations"][orientation].update(
                    {
                        "correct": orientation_ok,
                        "total": orientation_total,
                    }
                )
                if args.json_out:
                    write_receipt(args.json_out, receipt)
                if args.require_exact and not family_exact:
                    receipt["orientations"][orientation]["status"] = (
                        "stopped_on_failure"
                    )
                    stop_all = True
                    break
            rate = orientation_ok / orientation_total if orientation_total else 0.0
            print(
                f"TOTAL {orientation} exact-match: "
                f"{orientation_ok}/{orientation_total} = {rate:.4f}"
            )
            if not stop_all:
                receipt["orientations"][orientation]["status"] = (
                    "completed_exact"
                    if len(family_results) == len(selected)
                    and all(
                        family["correct"] == family["total"]
                        and family["output_count_exact"]
                        for family in family_results
                    )
                    else "failed"
                )
            elif receipt["orientations"][orientation]["status"] == "running":
                receipt["orientations"][orientation]["status"] = "failed"
            grand_ok += orientation_ok
            grand_total += orientation_total
            if args.json_out:
                write_receipt(args.json_out, receipt)

        counts_exact = bool(selected) and (
            len(receipt["orientations"]) == len(orientations)
            and all(
                result["total"] == expected_per_orientation
                and len(result["families"]) == len(selected)
                and all(
                    family["output_count_exact"]
                    and family["attempted"] == family["total"]
                    for family in result["families"]
                )
                for result in receipt["orientations"].values()
            )
        )
    except BaseException as error:
        execution_error = error
        receipt["status"] = (
            "qualification_identity_failed"
            if isinstance(error, QualificationIdentityError)
            else "interrupted"
            if isinstance(error, KeyboardInterrupt)
            else "error"
        )
        receipt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
    finally:
        cache_provenance_stable = finalize_cache_provenance(receipt)
        artifact_unchanged = finalize_artifact_identity(receipt, sub, before_identity)
        all_exact = (
            execution_error is None
            and grand_total > 0
            and counts_exact
            and grand_ok == grand_total
            and receipt.get("manifest_unchanged_during_load") is True
            and artifact_unchanged
            and cache_provenance_stable
        )
        receipt.update(
            {
                "status": (
                    "interrupted"
                    if isinstance(execution_error, KeyboardInterrupt)
                    else "completed_exact"
                    if all_exact
                    else "qualification_identity_failed"
                    if isinstance(execution_error, QualificationIdentityError)
                    else "error"
                    if execution_error is not None
                    else receipt.get("status", "failed")
                    if receipt.get("status") == "empty_selection"
                    else "failed"
                ),
                "all_exact": all_exact,
                "counts_exact": counts_exact,
                "cache_provenance_stable": cache_provenance_stable,
                "grand_correct": grand_ok,
                "grand_total": grand_total,
                "actual_batching": {
                    "batch_count": len(observed_batch_sizes),
                    "observed_batch_sizes": sorted(set(observed_batch_sizes)),
                    "maximum_actual_batch_size": (
                        max(observed_batch_sizes) if observed_batch_sizes else None
                    ),
                    "never_exceeded_effective_scorer_batch_size": (
                        not observed_batch_sizes
                        or (
                            isinstance(receipt.get("effective_scorer_batch_size"), int)
                            and max(observed_batch_sizes)
                            <= receipt["effective_scorer_batch_size"]
                        )
                    ),
                },
            }
        )
        if args.json_out:
            write_receipt(args.json_out, receipt)
            print(f"receipt={args.json_out}")

    if isinstance(execution_error, KeyboardInterrupt):
        raise execution_error
    if execution_error is not None:
        print(
            f"ERROR: {type(execution_error).__name__}: {execution_error}",
            file=sys.stderr,
        )
        return 2
    if receipt["status"] == "empty_selection":
        return 2
    return 1 if args.require_exact and not all_exact else 0


if __name__ == "__main__":
    sys.exit(main())
