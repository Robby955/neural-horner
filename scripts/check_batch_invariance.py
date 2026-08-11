#!/usr/bin/env python3
"""Qualification gate for sampled batching and order equality.

The gate uses only the pinned public scorer and its fixed public benchmark.  It
compares the same sampled scored-tier cases as singletons, in declared batches,
in reversed and deterministically permuted order, with operands swapped, on a
repeat call, and after a clean reload.  Both the emitted digit lists and the
pinned scorer's decoded values must agree by default.

This is a finite nonsealed equality check.  A passing receipt makes no claim
about untested inputs, private cases, or runtime compliance.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable, NamedTuple

import torch

from submission_utils import artifact_identity, load_submission, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORER_CONTRACT = (
    PROJECT_ROOT / "research" / "receipts" / "scorer_runtime_contract_82510.json"
)
PINNED_SCORER_FILES = (
    "src/modchallenge/config.py",
    "src/modchallenge/evaluation/decoder.py",
    "src/modchallenge/evaluation/pipeline.py",
    "src/modchallenge/interface/base_model.py",
    "src/modchallenge/interface/submission_schema.py",
    "src/modchallenge/testgen/generator.py",
    "src/modchallenge/testgen/primes.py",
)
SCORED_TIER_OPERAND_BITS = {
    1: 32,
    2: 48,
    3: 64,
    4: 96,
    5: 128,
    6: 256,
    7: 512,
    8: 1024,
    9: 2048,
    10: 4096,
}


class GateCase(NamedTuple):
    case_id: str
    source: str
    tier_id: int | None
    source_index: int | None
    a: int
    b: int
    p: int
    expected: int


def canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _git(repository: Path, *arguments: str) -> str:
    git_environment = os.environ.copy()
    git_environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
        env=git_environment,
    )
    return completed.stdout.strip()


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    git_environment = os.environ.copy()
    git_environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        env=git_environment,
    )
    return completed.stdout


def _verify_committed_scorer_tree(
    repository: Path,
    commit: str,
) -> dict[str, str]:
    """Require the scorer's entire inserted import root to match committed bytes."""
    import_root = repository / "src"
    tracked_paths = tuple(
        name
        for name in _git(
            repository,
            "ls-tree",
            "-r",
            "--name-only",
            commit,
            "--",
            "src",
        ).splitlines()
        if name
    )
    if not tracked_paths:
        raise RuntimeError("pinned scorer commit has no src import tree")
    tracked_set = set(tracked_paths)
    missing_qualification = sorted(set(PINNED_SCORER_FILES) - tracked_set)
    if missing_qualification:
        raise RuntimeError(
            "pinned scorer commit is missing qualification sources: "
            + ", ".join(missing_qualification)
        )

    if not import_root.is_dir() or import_root.is_symlink():
        raise RuntimeError(f"invalid scorer import root: {import_root}")
    unexpected_files: list[str] = []
    for path in import_root.rglob("*"):
        if path.is_symlink():
            raise RuntimeError(f"scorer module tree contains a symlink: {path}")
        if path.is_file():
            relative_name = str(path.relative_to(repository))
            if relative_name not in tracked_set:
                unexpected_files.append(relative_name)
    if unexpected_files:
        raise RuntimeError(
            "scorer import tree contains uncommitted or cached files: "
            + ", ".join(sorted(unexpected_files))
        )

    source_hashes: dict[str, str] = {}
    for relative_name in tracked_paths:
        source = repository / relative_name
        if not source.is_file() or source.is_symlink():
            raise RuntimeError(f"missing committed scorer source: {relative_name}")
        observed_digest = sha256_file(source)
        committed_digest = hashlib.sha256(
            _git_bytes(repository, "show", f"{commit}:{relative_name}")
        ).hexdigest()
        if observed_digest != committed_digest:
            raise RuntimeError(
                f"scorer working-tree/commit mismatch for {relative_name}: "
                f"committed {committed_digest}, observed {observed_digest}"
            )
        source_hashes[relative_name] = observed_digest
    return source_hashes


def load_pinned_scorer(
    contract_path: Path = DEFAULT_SCORER_CONTRACT,
    scorer_repository: Path | None = None,
) -> tuple[dict[str, object], Callable[..., int], Callable[[], object]]:
    """Verify and import the decoder/generator from the declared scorer commit.

    Historical contracts retain the checkout path used for their receipts.
    ``scorer_repository`` may point to a reconstructed checkout, but it never
    overrides the declared commit or source hashes.
    """
    contract_path = contract_path.resolve()
    if not contract_path.is_file():
        raise FileNotFoundError(f"scorer contract not found: {contract_path}")
    contract = json.loads(contract_path.read_text())
    declared_repository = Path(contract["scorer_repository"]).resolve()
    repository = (
        scorer_repository.resolve()
        if scorer_repository is not None
        else declared_repository
    )
    expected_commit = str(contract["scorer_commit"])
    if not repository.is_dir():
        raise FileNotFoundError(f"pinned scorer checkout not found: {repository}")

    observed_root = Path(_git(repository, "rev-parse", "--show-toplevel")).resolve()
    if repository != observed_root:
        raise RuntimeError(
            "scorer repository must be the exact Git worktree root: "
            f"requested={repository} observed_root={observed_root}"
        )
    observed_commit = _git(repository, "rev-parse", "HEAD")
    tracked_status = _git(
        repository,
        "status",
        "--porcelain",
        "--untracked-files=all",
    )
    if observed_commit != expected_commit or tracked_status:
        raise RuntimeError(
            "scorer checkout identity mismatch: "
            f"expected_commit={expected_commit} observed_commit={observed_commit} "
            f"worktree_clean={not bool(tracked_status)}"
        )

    committed_source_hashes = _verify_committed_scorer_tree(
        repository,
        expected_commit,
    )

    declared_hashes = {
        str(name): str(digest)
        for name, digest in contract["source_sha256"].items()
    }
    observed_declared: dict[str, str] = {}
    for relative_name, expected_digest in declared_hashes.items():
        source = repository / relative_name
        observed_digest = sha256_file(source)
        observed_declared[relative_name] = observed_digest
        if observed_digest != expected_digest:
            raise RuntimeError(
                f"scorer source hash mismatch for {relative_name}: "
                f"expected {expected_digest}, observed {observed_digest}"
            )

    qualification_hashes = {
        relative_name: committed_source_hashes[relative_name]
        for relative_name in PINNED_SCORER_FILES
    }
    scorer_src = repository / "src"
    scorer_src_text = str(scorer_src)
    sys.path[:] = [entry for entry in sys.path if entry != scorer_src_text]
    sys.path.insert(0, scorer_src_text)
    for module_name in tuple(sys.modules):
        if module_name == "modchallenge" or module_name.startswith("modchallenge."):
            del sys.modules[module_name]
    importlib.invalidate_caches()
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        decoder_module = importlib.import_module("modchallenge.evaluation.decoder")
        generator_module = importlib.import_module("modchallenge.testgen.generator")
    finally:
        sys.dont_write_bytecode = previous_dont_write_bytecode

    expected_import_paths = {
        "modchallenge.evaluation.decoder": (
            repository / "src/modchallenge/evaluation/decoder.py"
        ).resolve(),
        "modchallenge.testgen.generator": (
            repository / "src/modchallenge/testgen/generator.py"
        ).resolve(),
    }
    for imported_module in (decoder_module, generator_module):
        imported_path = Path(imported_module.__file__).resolve()
        expected_path = expected_import_paths[imported_module.__name__]
        if imported_path != expected_path:
            raise RuntimeError(
                "resolved scorer module at the wrong source path: "
                f"{imported_module.__name__} -> {imported_path}; "
                f"expected {expected_path}"
            )

    loaded_module_paths: dict[str, str] = {}
    for module_name, module in tuple(sys.modules.items()):
        if module_name != "modchallenge" and not module_name.startswith(
            "modchallenge."
        ):
            continue
        module_file = getattr(module, "__file__", None)
        if module_file is None:
            raise RuntimeError(f"scorer module has no source origin: {module_name}")
        module_path = Path(module_file).resolve()
        try:
            relative_name = str(module_path.relative_to(repository))
        except ValueError as error:
            raise RuntimeError(
                f"resolved scorer dependency outside pinned checkout: "
                f"{module_name} -> {module_path}"
            ) from error
        if relative_name not in committed_source_hashes:
            raise RuntimeError(
                f"resolved scorer dependency is not committed at the pin: "
                f"{module_name} -> {relative_name}"
            )
        module_base = Path("src", *module_name.split("."))
        expected_candidates = {
            str(module_base.with_suffix(".py")),
            str(module_base / "__init__.py"),
        }
        committed_candidates = expected_candidates.intersection(
            committed_source_hashes
        )
        if committed_candidates != {relative_name}:
            raise RuntimeError(
                "resolved scorer dependency at the wrong module-name path: "
                f"{module_name} -> {relative_name}; expected one of "
                f"{sorted(committed_candidates)}"
            )
        loaded_module_paths[module_name] = relative_name

    identity: dict[str, object] = {
        "status": "verified",
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "declared_repository": str(declared_repository),
        "repository": str(repository),
        "repository_override_used": repository != declared_repository,
        "repository_root_exact": True,
        "commit": observed_commit,
        "tracked_files_clean": True,
        "worktree_clean_including_untracked": True,
        "declared_source_sha256": observed_declared,
        "committed_module_source_sha256": committed_source_hashes,
        "qualification_source_sha256": qualification_hashes,
        "qualification_source_set_sha256": canonical_json_sha256(
            qualification_hashes
        ),
        "decoder_path": str(Path(decoder_module.__file__).resolve()),
        "generator_path": str(Path(generator_module.__file__).resolve()),
        "loaded_module_paths": loaded_module_paths,
    }
    return identity, decoder_module.decode_answer, generator_module.generate_public_test_set


def local_source_identity(
    runner_path: Path,
    *,
    additional_paths: Iterable[Path] = (),
) -> dict[str, object]:
    paths = [
        runner_path.resolve(),
        Path(__file__).resolve().with_name("submission_utils.py"),
        *(path.resolve() for path in additional_paths),
    ]
    hashes = {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in paths
    }
    return {
        "source_sha256": hashes,
        "source_set_sha256": canonical_json_sha256(hashes),
    }


def guarded_load_submission(submission: str | Path, loader=None):
    submission_path = Path(submission).resolve()
    before = artifact_identity(submission_path)
    active_loader = load_submission if loader is None else loader
    loaded_path, manifest, module, model = active_loader(submission_path)
    loaded_path = Path(loaded_path).resolve()
    if loaded_path != submission_path:
        raise RuntimeError(
            f"loader resolved {loaded_path}; expected {submission_path}"
        )
    after = artifact_identity(submission_path)
    if before != after:
        raise RuntimeError("submission artifact changed during load")
    return loaded_path, manifest, module, model, before, after


def artifact_tree_identity(
    submission: str | Path,
    *,
    excluded_paths: Iterable[Path] = (),
) -> dict[str, object]:
    """Hash every regular artifact file except explicit receipt/cache outputs."""
    submission_path = Path(submission).resolve()
    excluded = {path.resolve() for path in excluded_paths}
    files: dict[str, str] = {}
    for path in sorted(submission_path.rglob("*")):
        relative = path.relative_to(submission_path)
        if "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"artifact tree contains a symlink: {relative}")
        if path.resolve() in excluded:
            continue
        if not path.is_file() or path.suffix == ".pyc":
            continue
        files[str(relative)] = sha256_file(path)
    if not files:
        raise RuntimeError("artifact tree contains no regular files")
    return {
        "file_count": len(files),
        "file_sha256": files,
        "file_set_sha256": canonical_json_sha256(files),
        "excluded_cache_directories": ["__pycache__"],
        "explicit_excluded_paths": sorted(str(path) for path in excluded),
    }


def receipt_output_exclusions(
    submission: str | Path,
    json_out: Path | None,
) -> tuple[Path, ...]:
    if json_out is None:
        return ()
    submission_path = Path(submission).resolve()
    output_path = json_out.resolve()
    try:
        output_path.relative_to(submission_path)
    except ValueError:
        return ()
    return (output_path,)


def force_device(model, requested: str) -> torch.device:
    if requested == "auto":
        device = torch.device(getattr(model, "device", "cpu"))
    else:
        if requested == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        if requested == "mps" and not (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        ):
            raise RuntimeError("MPS was requested but is unavailable")
        device = torch.device(requested)
        module = getattr(model, "model", None)
        if not isinstance(module, torch.nn.Module):
            raise RuntimeError("loaded entry does not expose model: torch.nn.Module")
        module.to(device)
        module.eval()
        model.device = device
    return device


def model_tensor_identity(model) -> dict[str, object]:
    module = getattr(model, "model", None)
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError("loaded entry does not expose model: torch.nn.Module")

    def summarize(items) -> dict[str, object]:
        rows = []
        for name, tensor in items:
            rows.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "elements": tensor.numel(),
                }
            )
        return {
            "tensor_count": len(rows),
            "element_count": sum(int(row["elements"]) for row in rows),
            "rows": rows,
        }

    identity = {
        "parameters": summarize(module.named_parameters(recurse=True)),
        "buffers": summarize(module.named_buffers(recurse=True)),
    }
    identity["sha256"] = canonical_json_sha256(identity)
    return identity


def state_dict_sha256(model) -> str:
    module = getattr(model, "model", None)
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError("loaded entry does not expose model: torch.nn.Module")
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        cpu = tensor.detach().cpu().contiguous()
        metadata = {
            "name": name,
            "shape": list(cpu.shape),
            "dtype": str(cpu.dtype),
        }
        digest.update(json.dumps(metadata, sort_keys=True).encode())
        digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def environment_identity(device: torch.device) -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    mps_available = bool(
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    )
    identity: dict[str, object] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "os_cpu_count": os.cpu_count(),
        "device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if cuda_available else None,
        "mps_available": mps_available,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
        "deterministic_algorithms_enabled": (
            torch.are_deterministic_algorithms_enabled()
        ),
        "default_dtype": str(torch.get_default_dtype()),
    }
    if device.type == "cuda":
        identity["cuda_device_name"] = torch.cuda.get_device_name(device)
        identity["cuda_device_capability"] = list(
            torch.cuda.get_device_capability(device)
        )
    return identity


def _public_case(tier_id: int, index: int, case, role: str) -> GateCase:
    a, b, p = int(case.a), int(case.b), int(case.p)
    expected = int(case.expected)
    if expected != (a * b) % p:
        raise RuntimeError(
            f"public fixture arithmetic mismatch at tier {tier_id}, row {index}"
        )
    return GateCase(
        case_id=f"public-v1-tier{tier_id}-{role}-row{index}",
        source="pinned_public_benchmark_v1",
        tier_id=tier_id,
        source_index=index,
        a=a,
        b=b,
        p=p,
        expected=expected,
    )


def select_public_boundary_cases(public_test_set) -> list[GateCase]:
    """Select one public edge row and the widest public row from each scored tier."""
    by_tier = {int(tier.tier_id): tier for tier in public_test_set.tiers}
    selected: list[GateCase] = []
    for tier_id, operand_bits in SCORED_TIER_OPERAND_BITS.items():
        if tier_id not in by_tier:
            raise RuntimeError(f"public benchmark is missing scored tier {tier_id}")
        rows = list(by_tier[tier_id].cases)
        if len(rows) < 5:
            raise RuntimeError(f"public tier {tier_id} has too few rows")
        edge_index = min(
            range(min(4, len(rows))),
            key=lambda index: (
                min(int(rows[index].a).bit_length(), int(rows[index].b).bit_length()),
                index,
            ),
        )
        boundary_index = max(
            range(len(rows)),
            key=lambda index: (
                max(int(rows[index].a).bit_length(), int(rows[index].b).bit_length()),
                int(rows[index].a).bit_length() + int(rows[index].b).bit_length(),
                -index,
            ),
        )
        boundary = rows[boundary_index]
        observed_width = max(
            int(boundary.a).bit_length(), int(boundary.b).bit_length()
        )
        if observed_width != operand_bits:
            raise RuntimeError(
                f"public tier {tier_id} did not expose its {operand_bits}-bit "
                f"operand boundary; widest selected row was {observed_width} bits"
            )
        selected.append(_public_case(tier_id, edge_index, rows[edge_index], "edge"))
        selected.append(
            _public_case(tier_id, boundary_index, boundary, "operand-boundary")
        )

    if len(selected) != 2 * len(SCORED_TIER_OPERAND_BITS):
        raise RuntimeError("public boundary selection cardinality mismatch")
    if max(max(case.a.bit_length(), case.b.bit_length()) for case in selected) != 4096:
        raise RuntimeError("public boundary suite does not reach 4096-bit operands")
    if len({case.case_id for case in selected}) != len(selected):
        raise RuntimeError("public boundary case identifiers are not unique")
    return selected


def _probable_prime(value: int, rng: random.Random) -> bool:
    if value < 2:
        return False
    for divisor in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
        if value % divisor == 0:
            return value == divisor
    d = value - 1
    shifts = 0
    while d % 2 == 0:
        d //= 2
        shifts += 1
    bases = [2, 3, 5, 7, 11, 13, 17] + [
        rng.randrange(2, value - 1) for _ in range(8)
    ]
    for base in bases:
        x = pow(base, d, value)
        if x in (1, value - 1):
            continue
        for _ in range(shifts - 1):
            x = (x * x) % value
            if x == value - 1:
                break
        else:
            return False
    return True


def previous_probable_prime(width: int, rng: random.Random) -> int:
    if width == 2:
        return 3
    candidate = (1 << width) - 1
    if candidate % 2 == 0:
        candidate -= 1
    lower = 1 << (width - 1)
    while candidate >= lower:
        if _probable_prime(candidate, rng):
            return candidate
        candidate -= 2
    raise RuntimeError(f"no probable prime found at width {width}")


def make_cases(widths: list[int], seed: int) -> list[tuple[int, int, int, str]]:
    """Retain the legacy custom-width fixture as an optional diagnostic add-on."""
    rng = random.Random(seed)
    cases = []
    for width in widths:
        p = previous_probable_prime(width, rng)
        cases.append((1, 1, p, f"custom-p{width}-short"))
        operand_width = min(2 * width + 1, 4096)
        a = rng.randrange(0, 1 << max(1, operand_width - 1))
        b = rng.randrange(0, 1 << operand_width)
        cases.append((a, b, p, f"custom-p{width}-long"))
    return cases


def custom_gate_cases(widths: list[int], seed: int) -> list[GateCase]:
    return [
        GateCase(
            case_id=case_id,
            source="legacy_custom_width_diagnostic",
            tier_id=None,
            source_index=index,
            a=a,
            b=b,
            p=p,
            expected=(a * b) % p,
        )
        for index, (a, b, p, case_id) in enumerate(make_cases(widths, seed))
    ]


def case_payload(case: GateCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "source": case.source,
        "tier_id": case.tier_id,
        "source_index": case.source_index,
        "a": str(case.a),
        "b": str(case.b),
        "p": str(case.p),
        "expected": str(case.expected),
    }


def compact_case_identity(case: GateCase) -> dict[str, object]:
    return {
        "case_id": case.case_id,
        "source": case.source,
        "tier_id": case.tier_id,
        "source_index": case.source_index,
        "case_sha256": canonical_json_sha256(case_payload(case)),
        "a_bits": case.a.bit_length(),
        "b_bits": case.b.bit_length(),
        "p_bits": case.p.bit_length(),
    }


def scorer_batch_size(model) -> tuple[int, int]:
    declared = model.max_batch_size()
    if isinstance(declared, bool) or not isinstance(declared, int):
        raise TypeError("max_batch_size() must return int")
    return declared, max(1, declared)


def deterministic_orders(case_count: int, seed: int, permutations: int) -> dict[str, list[int]]:
    orders = {
        "declared_batch": list(range(case_count)),
        "reverse": list(reversed(range(case_count))),
    }
    rng = random.Random(seed ^ 0xB47C4)
    for permutation_index in range(permutations):
        order = list(range(case_count))
        rng.shuffle(order)
        orders[f"permutation_{permutation_index}"] = order
    return orders


def run_layout(
    model,
    cases: list[GateCase],
    *,
    order: list[int],
    batch_size: int,
    label: str,
    swap_operands: bool = False,
) -> dict[str, object]:
    """Run one official-style decimal-preprocessed layout, failing closed."""
    if sorted(order) != list(range(len(cases))):
        raise ValueError(f"{label} order is not a permutation of all cases")
    restored: list[object | None] = [None] * len(cases)
    batch_checks: list[dict[str, object]] = []
    completed = 0
    for start in range(0, len(order), batch_size):
        indices = order[start : start + batch_size]
        encoded = []
        for case_index in indices:
            case = cases[case_index]
            a, b = (case.b, case.a) if swap_operands else (case.a, case.b)
            encoded.append(
                (
                    model.preprocess_a(str(a)),
                    model.preprocess_b(str(b)),
                    model.preprocess_p(str(case.p)),
                )
            )
        try:
            outputs = model.predict_digits_batch(encoded)
        except Exception as error:
            batch_checks.append(
                {
                    "batch_start": start,
                    "expected": len(indices),
                    "observed": None,
                    "outer_type": None,
                    "outer_list_exact": False,
                    "cardinality_exact": False,
                    "error": {"type": type(error).__name__, "message": str(error)},
                }
            )
            break
        outer_list_exact = isinstance(outputs, list)
        observed = len(outputs) if hasattr(outputs, "__len__") else None
        cardinality_exact = outer_list_exact and observed == len(indices)
        batch_checks.append(
            {
                "batch_start": start,
                "expected": len(indices),
                "observed": observed,
                "outer_type": type(outputs).__name__,
                "outer_list_exact": outer_list_exact,
                "cardinality_exact": cardinality_exact,
                "error": None,
            }
        )
        if not cardinality_exact:
            break
        for case_index, output in zip(indices, outputs):
            restored[case_index] = output
            completed += 1

    cardinality_exact = completed == len(cases) and all(
        bool(check["cardinality_exact"]) for check in batch_checks
    )
    return {
        "label": label,
        "order": order,
        "order_sha256": canonical_json_sha256(order),
        "swap_operands": swap_operands,
        "batch_size": batch_size,
        "expected_outputs": len(cases),
        "completed_outputs": completed,
        "cardinality_exact": cardinality_exact,
        "batch_checks": batch_checks,
        "outputs": restored,
    }


def decode_output(
    decoder: Callable[..., int],
    output: object,
    case: GateCase,
) -> tuple[bool, int | None, dict[str, str] | None]:
    if output is None:
        return False, None, {"type": "MissingOutput", "message": "output missing"}
    try:
        value = int(
            decoder(output, base=2, prime=case.p, is_tier_zero=False)
        )
    except Exception as error:
        return False, None, {"type": type(error).__name__, "message": str(error)}
    return True, value, None


def evaluate_equality_layouts(
    model,
    reload_model_factory: Callable[[], object],
    cases: list[GateCase],
    *,
    decoder: Callable[..., int],
    output_base: int | str,
    seed: int,
    permutations: int,
    require_exact: bool,
    require_raw_equality: bool,
) -> dict[str, object]:
    if output_base != 2:
        # decode_output receives base below through a local closure so all
        # manifest-supported bases still use the pinned decoder.
        def manifest_decoder(output, *, base, prime, is_tier_zero):
            return decoder(
                output,
                base=output_base,
                prime=prime,
                is_tier_zero=is_tier_zero,
            )
    else:
        manifest_decoder = decoder

    declared, effective = scorer_batch_size(model)
    reload_declared = None
    reload_effective = None
    layouts: dict[str, dict[str, object]] = {}
    singleton_outputs: list[object | None] = [None] * len(cases)
    singleton_checks = []
    for case_index in range(len(cases)):
        result = run_layout(
            model,
            [cases[case_index]],
            order=[0],
            batch_size=1,
            label=f"singleton_{case_index}",
        )
        singleton_checks.append(
            {
                "case_index": case_index,
                "cardinality_exact": result["cardinality_exact"],
                "batch_checks": result["batch_checks"],
            }
        )
        if not result["cardinality_exact"]:
            break
        singleton_outputs[case_index] = result["outputs"][0]
    singleton_exact = all(output is not None for output in singleton_outputs)
    layouts["singletons"] = {
        "label": "singletons",
        "expected_outputs": len(cases),
        "completed_outputs": sum(output is not None for output in singleton_outputs),
        "cardinality_exact": singleton_exact,
        "batch_checks": singleton_checks,
        "outputs": singleton_outputs,
    }

    if singleton_exact:
        for label, order in deterministic_orders(len(cases), seed, permutations).items():
            layouts[label] = run_layout(
                model,
                cases,
                order=order,
                batch_size=effective,
                label=label,
            )
            if not layouts[label]["cardinality_exact"]:
                break
    if all(layout["cardinality_exact"] for layout in layouts.values()):
        layouts["operand_swap"] = run_layout(
            model,
            cases,
            order=list(range(len(cases))),
            batch_size=effective,
            label="operand_swap",
            swap_operands=True,
        )
    if all(layout["cardinality_exact"] for layout in layouts.values()):
        layouts["repeat"] = run_layout(
            model,
            cases,
            order=list(range(len(cases))),
            batch_size=effective,
            label="repeat",
        )
    if all(layout["cardinality_exact"] for layout in layouts.values()):
        reloaded_model = reload_model_factory()
        reload_declared, reload_effective = scorer_batch_size(reloaded_model)
        layouts["reload"] = run_layout(
            reloaded_model,
            cases,
            order=list(range(len(cases))),
            batch_size=reload_effective,
            label="reload",
        )

    expected_layouts = {
        "singletons",
        "declared_batch",
        "reverse",
        *(f"permutation_{index}" for index in range(permutations)),
        "operand_swap",
        "repeat",
        "reload",
    }
    layout_set_exact = set(layouts) == expected_layouts
    all_cardinalities_exact = layout_set_exact and all(
        bool(layout["cardinality_exact"]) for layout in layouts.values()
    )

    case_results = []
    raw_equality = True
    decoded_equality = True
    outputs_valid = True
    exact = True
    if all_cardinalities_exact:
        baseline = layouts["singletons"]["outputs"]
        for case_index, case in enumerate(cases):
            baseline_valid, baseline_value, baseline_error = decode_output(
                manifest_decoder,
                baseline[case_index],
                case,
            )
            outputs_valid &= baseline_valid
            case_exact = baseline_valid and baseline_value == case.expected
            exact &= case_exact
            comparisons = {}
            for label in sorted(expected_layouts - {"singletons"}):
                output = layouts[label]["outputs"][case_index]
                valid, value, error = decode_output(manifest_decoder, output, case)
                raw_equal = output == baseline[case_index]
                decoded_equal = (
                    baseline_valid and valid and value == baseline_value
                )
                raw_equality &= raw_equal
                decoded_equality &= decoded_equal
                outputs_valid &= valid
                comparisons[label] = {
                    "raw_equal": raw_equal,
                    "decoded_equal": decoded_equal,
                    "output_valid": valid,
                    "decode_error": error,
                }
            case_results.append(
                {
                    **compact_case_identity(case),
                    "baseline_output_valid": baseline_valid,
                    "baseline_decode_error": baseline_error,
                    "exact": case_exact,
                    "comparisons": comparisons,
                }
            )
    else:
        raw_equality = decoded_equality = outputs_valid = exact = False

    qualification_passed = all(
        (
            all_cardinalities_exact,
            outputs_valid,
            raw_equality,
            decoded_equality,
            exact,
            declared == reload_declared,
            effective == reload_effective,
        )
    )
    requested_passed = all(
        (
            all_cardinalities_exact,
            outputs_valid,
            decoded_equality,
            declared == reload_declared,
            effective == reload_effective,
            exact or not require_exact,
            raw_equality or not require_raw_equality,
        )
    )
    layout_receipts = {}
    for label, layout in layouts.items():
        outputs = layout.pop("outputs")
        output_digest = (
            canonical_json_sha256(outputs)
            if layout["cardinality_exact"]
            else None
        )
        layout_receipts[label] = {**layout, "output_sha256": output_digest}
        layout["outputs"] = outputs
    return {
        "declared_batch_size": declared,
        "effective_batch_size": effective,
        "reload_declared_batch_size": reload_declared,
        "reload_effective_batch_size": reload_effective,
        "reload_batch_contract_equal": (
            declared == reload_declared and effective == reload_effective
        ),
        "expected_layout_count": len(expected_layouts),
        "observed_layout_count": len(layouts),
        "layout_set_exact": layout_set_exact,
        "output_cardinality_exact": all_cardinalities_exact,
        "outputs_decoder_valid": outputs_valid,
        "raw_equality": raw_equality,
        "decoded_equality": decoded_equality,
        "exact_on_sampled_cases": exact,
        "passed_requested_gate": requested_passed,
        "passed_qualification_gate": qualification_passed,
        "layouts": layout_receipts,
        "cases": case_results,
    }


def write_receipt(path: Path | None, receipt: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument(
        "--widths",
        type=int,
        nargs="+",
        default=None,
        help="optional legacy custom-width cases, added to the public suite",
    )
    parser.add_argument("--seed", type=int, default=20260801)
    parser.add_argument("--permutations", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--scorer-contract", type=Path, default=DEFAULT_SCORER_CONTRACT)
    parser.add_argument(
        "--scorer-repo",
        type=Path,
        help=(
            "clean checkout of the contract's exact scorer commit; permits "
            "reconstructing an expired historical checkout path without "
            "weakening commit or source-hash checks"
        ),
    )
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.set_defaults(require_exact=True, require_raw_equality=True)
    parser.add_argument("--require-exact", dest="require_exact", action="store_true")
    parser.add_argument(
        "--allow-inexact",
        dest="require_exact",
        action="store_false",
        help="diagnostic only; cannot pass the qualification profile",
    )
    parser.add_argument(
        "--require-raw-equality",
        dest="require_raw_equality",
        action="store_true",
    )
    parser.add_argument(
        "--allow-raw-variation",
        dest="require_raw_equality",
        action="store_false",
        help="diagnostic only; cannot pass the qualification profile",
    )
    args = parser.parse_args()

    if args.widths and any(width < 2 for width in args.widths):
        parser.error("--widths must contain integers >= 2")
    if args.permutations < 1:
        parser.error("--permutations must be positive")

    runner_path = Path(__file__).resolve()
    scorer_identity, decoder, public_generator = load_pinned_scorer(
        args.scorer_contract,
        args.scorer_repo,
    )
    source_before = local_source_identity(runner_path)
    cases = select_public_boundary_cases(public_generator())
    if args.widths:
        cases.extend(custom_gate_cases(args.widths, args.seed))
    case_payloads = [case_payload(case) for case in cases]
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("case identifier collision")

    artifact_tree_exclusions = receipt_output_exclusions(
        args.submission,
        args.json_out,
    )
    artifact_tree_before = artifact_tree_identity(
        args.submission,
        excluded_paths=artifact_tree_exclusions,
    )

    (
        submission,
        manifest,
        _module,
        model,
        artifact_before,
        _artifact_after_first_load,
    ) = guarded_load_submission(args.submission)
    artifact_tree_after_first_load = artifact_tree_identity(
        submission,
        excluded_paths=artifact_tree_exclusions,
    )
    if artifact_tree_before != artifact_tree_after_first_load:
        raise RuntimeError("artifact tree changed during initial load")
    device = force_device(model, args.device)
    tensor_identity = model_tensor_identity(model)
    trained_state_before = state_dict_sha256(model)
    environment = environment_identity(device)
    reload_holder: dict[str, object] = {}

    def load_reload_after_repeat():
        (
            reload_submission,
            reload_manifest,
            _reload_module,
            reloaded_model,
            reload_artifact_before,
            _reload_artifact_after,
        ) = guarded_load_submission(args.submission)
        reload_device = force_device(reloaded_model, args.device)
        reload_tensor_identity = model_tensor_identity(reloaded_model)
        reload_state_before = state_dict_sha256(reloaded_model)
        if submission != reload_submission or manifest != reload_manifest:
            raise RuntimeError("reload resolved a different submission identity")
        if artifact_before != reload_artifact_before:
            raise RuntimeError("artifact identity changed between loads")
        reload_tree_before = artifact_tree_identity(
            submission,
            excluded_paths=artifact_tree_exclusions,
        )
        if reload_tree_before != artifact_tree_before:
            raise RuntimeError("artifact tree changed before clean reload")
        if trained_state_before != reload_state_before:
            raise RuntimeError("clean reload state differs before inference")
        reload_holder.update(
            {
                "model": reloaded_model,
                "environment": environment_identity(reload_device),
                "tensor_identity": reload_tensor_identity,
                "state_before": reload_state_before,
            }
        )
        return reloaded_model

    receipt: dict[str, object] = {
        "schema_version": 2,
        "status": "running",
        "claim_scope": "sampled_public_and_optional_generated_nonsealed_cases_only",
        "submission": str(submission),
        "entry_class": manifest["entry_class"],
        "output_base": manifest["output_base"],
        "artifact_sha256_before": artifact_before,
        "artifact_set_sha256": canonical_json_sha256(artifact_before),
        "artifact_tree_identity_before": artifact_tree_before,
        "artifact_tree_identity_sha256": canonical_json_sha256(
            artifact_tree_before
        ),
        "runner_sha256": source_before["source_sha256"][
            str(runner_path.relative_to(PROJECT_ROOT))
        ],
        "local_source_identity_before": source_before,
        "scorer_identity": scorer_identity,
        "scorer_identity_sha256": canonical_json_sha256(scorer_identity),
        "case_set_sha256": canonical_json_sha256(case_payloads),
        "expected_cases": len(cases),
        "public_boundary_cases": 2 * len(SCORED_TIER_OPERAND_BITS),
        "custom_cases": len(cases) - 2 * len(SCORED_TIER_OPERAND_BITS),
        "selected_cases": [compact_case_identity(case) for case in cases],
        "seed": args.seed,
        "permutations": args.permutations,
        "require_exact": args.require_exact,
        "require_raw_equality": args.require_raw_equality,
        "environment": environment,
        "environment_sha256": canonical_json_sha256(environment),
        "reload_environment": None,
        "reload_environment_sha256": None,
        "model_tensor_identity": tensor_identity,
        "reload_model_tensor_identity": None,
        "trained_state_sha256_before": trained_state_before,
        "reload_state_sha256_before": None,
        "passed_requested_gate": False,
        "passed_qualification_gate": False,
    }
    write_receipt(args.json_out, receipt)

    try:
        result = evaluate_equality_layouts(
            model,
            load_reload_after_repeat,
            cases,
            decoder=decoder,
            output_base=manifest["output_base"],
            seed=args.seed,
            permutations=args.permutations,
            require_exact=args.require_exact,
            require_raw_equality=args.require_raw_equality,
        )
        receipt["result"] = result
    except BaseException as error:
        receipt["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        )
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        write_receipt(args.json_out, receipt)
        raise

    try:
        trained_state_after = state_dict_sha256(model)
        reloaded_model = reload_holder.get("model")
        reload_environment = reload_holder.get("environment")
        reload_tensor_identity = reload_holder.get("tensor_identity")
        reload_state_before = reload_holder.get("state_before")
        reload_state_after = (
            state_dict_sha256(reloaded_model)
            if reloaded_model is not None
            else None
        )
        artifact_after = artifact_identity(submission)
        artifact_tree_after = artifact_tree_identity(
            submission,
            excluded_paths=artifact_tree_exclusions,
        )
        source_after = local_source_identity(runner_path)
        scorer_identity_after, _decoder_after, _generator_after = (
            load_pinned_scorer(args.scorer_contract, args.scorer_repo)
        )
    except BaseException as error:
        receipt["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        )
        receipt["error"] = {
            "phase": "post_run_identity_finalization",
            "type": type(error).__name__,
            "message": str(error),
        }
        write_receipt(args.json_out, receipt)
        raise
    artifacts_unchanged = artifact_before == artifact_after
    artifact_tree_unchanged = artifact_tree_before == artifact_tree_after
    sources_unchanged = source_before == source_after
    scorer_unchanged = scorer_identity == scorer_identity_after
    states_unchanged = (
        trained_state_before == trained_state_after
        and reload_state_before is not None
        and reload_state_before == reload_state_after
    )
    qualification_passed = all(
        (
            result["passed_qualification_gate"],
            artifacts_unchanged,
            artifact_tree_unchanged,
            sources_unchanged,
            scorer_unchanged,
            states_unchanged,
            reload_environment is not None and environment == reload_environment,
            reload_tensor_identity is not None
            and tensor_identity == reload_tensor_identity,
        )
    )
    requested_passed = all(
        (
            result["passed_requested_gate"],
            artifacts_unchanged,
            artifact_tree_unchanged,
            sources_unchanged,
            scorer_unchanged,
            states_unchanged,
        )
    )
    receipt.update(
        {
            "status": (
                "completed_exact"
                if qualification_passed
                else "completed_diagnostic"
                if requested_passed
                else "failed"
            ),
            "artifact_sha256_after": artifact_after,
            "artifact_unchanged_during_gate": artifacts_unchanged,
            "artifact_tree_identity_after": artifact_tree_after,
            "artifact_tree_unchanged_during_gate": artifact_tree_unchanged,
            "local_source_identity_after": source_after,
            "sources_unchanged_during_gate": sources_unchanged,
            "scorer_identity_after": scorer_identity_after,
            "scorer_unchanged_during_gate": scorer_unchanged,
            "trained_state_sha256_after": trained_state_after,
            "reload_environment": reload_environment,
            "reload_environment_sha256": (
                canonical_json_sha256(reload_environment)
                if reload_environment is not None
                else None
            ),
            "reload_model_tensor_identity": reload_tensor_identity,
            "reload_state_sha256_before": reload_state_before,
            "reload_state_sha256_after": reload_state_after,
            "model_states_unchanged_during_gate": states_unchanged,
            "passed_requested_gate": requested_passed,
            "passed_qualification_gate": qualification_passed,
        }
    )
    write_receipt(args.json_out, receipt)
    if args.json_out:
        print(f"receipt={args.json_out}")
    print(
        "SUMMARY "
        f"cases={len(cases)} raw_equal={result['raw_equality']} "
        f"decoded_equal={result['decoded_equality']} "
        f"exact={result['exact_on_sampled_cases']} "
        f"cardinality={result['output_cardinality_exact']} "
        f"qualification_passed={qualification_passed}"
    )
    return 0 if requested_passed else 1


if __name__ == "__main__":
    sys.exit(main())
