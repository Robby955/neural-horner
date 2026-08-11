#!/usr/bin/env python3
"""Trace candidate-owned F11 trajectories for both NeuralHorner schedules.

Each selected case is evaluated as a singleton from the all-zero state.  This
keeps phase cardinalities independent of batch padding and ensures that every
reported prefix was produced by the checkpoint named in the same receipt.

The tracer subclasses the exact manifest entry class and reproduces its `_step`
implementation while retaining the pre-threshold logits.  It therefore records
the first learned transition that disagrees with

    (2 * state + input_bit * x) mod p

without injecting a state, borrowing a prefix, or changing inference routing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import types
from pathlib import Path
from typing import NamedTuple

import torch

from held_out_battery import generate_battery
from submission_utils import artifact_identity, load_submission, sha256_file


F11 = (1 << (1 << 11)) + 1
BATTERY_SEED = 20260627
COMPANION_SEED = 20260801
COMPANION_WIDTHS = (1, 32, 256, 1024, 2048, 2049, 3072, 4096)
EXPECTED_LEGACY_CASES = 9
EXPECTED_ALL_CASES = EXPECTED_LEGACY_CASES + len(COMPANION_WIDTHS) + 3
EXTERNAL_CASE_FIXTURE_SCHEMA = "neural-horner-trajectory-case-fixture-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORER_CONTRACT_PATH = (
    PROJECT_ROOT / "research" / "receipts" / "scorer_runtime_contract_82510.json"
)


class TraceCase(NamedTuple):
    label: str
    group: str
    source: str
    a: int
    b: int
    p: int


class TeacherTransition(NamedTuple):
    phase: str
    phase_step: int
    global_step: int
    state: int
    x: int
    p: int
    digit: int
    expected: int
    width: int


def canonical_json_sha256(value) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def guarded_load_submission(submission: str | Path, loader=None):
    """Load a submission only if loading leaves every artifact byte unchanged."""
    submission_path = Path(submission).resolve()
    before_load = artifact_identity(submission_path)
    active_loader = load_submission if loader is None else loader
    loaded_submission, manifest, module, loaded = active_loader(submission_path)
    loaded_submission = Path(loaded_submission).resolve()
    if loaded_submission != submission_path:
        raise RuntimeError(
            "submission loader resolved a different artifact directory: "
            f"expected {submission_path}, found {loaded_submission}"
        )
    after_load = artifact_identity(loaded_submission)
    if before_load != after_load:
        changed = sorted(
            name
            for name in set(before_load) | set(after_load)
            if before_load.get(name) != after_load.get(name)
        )
        raise RuntimeError(
            "submission artifacts changed during load: " + ", ".join(changed)
        )
    return (
        loaded_submission,
        manifest,
        module,
        loaded,
        before_load,
        after_load,
    )


def _git_repository_identity(repository: Path) -> dict[str, object]:
    def run_git(*arguments: str) -> str:
        completed = subprocess.run(
            ["git", "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()

    return {
        "head": run_git("rev-parse", "HEAD"),
        "tracked_files_clean": not bool(
            run_git("status", "--porcelain", "--untracked-files=no")
        ),
    }


def scorer_source_identity(
    contract_path: Path = DEFAULT_SCORER_CONTRACT_PATH,
) -> dict[str, object]:
    """Bind the pinned scorer declaration and verify its checkout when present."""
    contract_path = contract_path.resolve()
    if not contract_path.is_file():
        return {
            "status": "contract_not_discovered",
            "contract_path": str(contract_path),
        }

    contract = json.loads(contract_path.read_text())
    declared_repository = Path(contract["scorer_repository"]).resolve()
    declared_commit = str(contract["scorer_commit"])
    declared_sources = {
        str(name): str(digest)
        for name, digest in contract["source_sha256"].items()
    }
    identity: dict[str, object] = {
        "status": "checkout_not_discovered",
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "declared_repository": str(declared_repository),
        "declared_commit": declared_commit,
        "declared_source_sha256": declared_sources,
        "declared_source_set_sha256": canonical_json_sha256(declared_sources),
    }
    if not declared_repository.is_dir():
        return identity

    repository_identity = _git_repository_identity(declared_repository)
    observed_sources: dict[str, str | None] = {}
    for relative_name in declared_sources:
        relative_path = Path(relative_name)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise RuntimeError(
                f"invalid scorer source path in contract: {relative_name}"
            )
        source_path = declared_repository / relative_path
        observed_sources[relative_name] = (
            sha256_file(source_path) if source_path.is_file() else None
        )

    commit_matches = repository_identity["head"] == declared_commit
    sources_match = observed_sources == declared_sources
    clean_matches = bool(repository_identity["tracked_files_clean"])
    identity.update(
        {
            "status": "verified" if all(
                (commit_matches, sources_match, clean_matches)
            ) else "mismatch",
            "observed_commit": repository_identity["head"],
            "tracked_files_clean": clean_matches,
            "commit_matches_contract": commit_matches,
            "observed_source_sha256": observed_sources,
            "observed_source_set_sha256": canonical_json_sha256(observed_sources),
            "sources_match_contract": sources_match,
        }
    )
    if identity["status"] != "verified":
        raise RuntimeError(
            "scorer source identity mismatch: "
            f"commit_matches={commit_matches}, "
            f"sources_match={sources_match}, tracked_files_clean={clean_matches}"
        )
    return identity


def qualification_source_identity(
    scorer_contract_path: Path = DEFAULT_SCORER_CONTRACT_PATH,
) -> dict[str, object]:
    """Hash the runner, imported local helpers, and pinned scorer sources."""
    local_sources = {
        "scripts/trace_f11_trajectories.py": sha256_file(Path(__file__).resolve()),
        "scripts/held_out_battery.py": sha256_file(
            Path(__file__).resolve().with_name("held_out_battery.py")
        ),
        "scripts/submission_utils.py": sha256_file(
            Path(__file__).resolve().with_name("submission_utils.py")
        ),
    }
    return {
        "local_source_sha256": local_sources,
        "local_source_set_sha256": canonical_json_sha256(local_sources),
        "scorer": scorer_source_identity(scorer_contract_path),
    }


def model_tensor_dtype_identity(model) -> dict[str, object]:
    """Describe effective dtypes/devices after the loaded cell is placed."""
    module = getattr(model, "model", None)
    if not isinstance(module, torch.nn.Module):
        raise RuntimeError("loaded entry class does not expose a torch module")

    def summarize(named_tensors) -> dict[str, object]:
        dtype_counts: dict[str, int] = {}
        device_counts: dict[str, int] = {}
        tensor_count = 0
        element_count = 0
        for _name, tensor in named_tensors:
            tensor_count += 1
            element_count += tensor.numel()
            dtype_name = str(tensor.dtype)
            device_name = str(tensor.device)
            dtype_counts[dtype_name] = dtype_counts.get(dtype_name, 0) + 1
            device_counts[device_name] = device_counts.get(device_name, 0) + 1
        return {
            "tensor_count": tensor_count,
            "element_count": element_count,
            "dtype_counts": dict(sorted(dtype_counts.items())),
            "device_counts": dict(sorted(device_counts.items())),
        }

    return {
        "parameters": summarize(module.named_parameters(recurse=True)),
        "buffers": summarize(module.named_buffers(recurse=True)),
    }


def bits_to_int(row: torch.Tensor) -> int:
    value = 0
    for bit in row.detach().long().cpu().tolist():
        value = 2 * value + int(bit)
    return value


def int_to_bit_tensor(value: int, width: int, device: torch.device) -> torch.Tensor:
    if value < 0 or value >= (1 << width):
        raise ValueError(f"value does not fit in {width} bits")
    return torch.tensor(
        [(value >> shift) & 1 for shift in range(width - 1, -1, -1)],
        dtype=torch.float32,
        device=device,
    )


def decode_digits(digits: list[int]) -> int:
    value = 0
    for digit in digits:
        value = 2 * value + int(digit)
    return value


def transition_arithmetic_metadata(
    state: int,
    x: int,
    p: int,
    digit: int,
    width: int,
) -> dict[str, int | str | bool]:
    """Describe the exact carry and modular-subtraction geometry of one step."""
    if p < 2:
        raise ValueError("p must be at least 2")
    if digit not in (0, 1):
        raise ValueError("digit must be binary")
    word_limit = 1 << width
    doubled = 2 * state
    truncated_double = doubled % word_limit
    addend = digit * x
    pre_mod = doubled + addend
    expected = pre_mod % p
    subtract_count = pre_mod // p
    distance_lower = expected
    distance_upper = p - expected
    return {
        "state_width_bits": width,
        "pre_mod_value": str(pre_mod),
        "expected_value": str(expected),
        "modulus_subtract_count": str(subtract_count),
        "modulus_subtract_required": subtract_count > 0,
        "distance_to_lower_modulus_multiple": str(distance_lower),
        "distance_to_upper_modulus_multiple": str(distance_upper),
        "boundary_distance_to_nearest_modulus_multiple": str(
            min(distance_lower, distance_upper)
        ),
        "state_less_than_modulus": state < p,
        "x_less_than_modulus": x < p,
        "double_carry_out": doubled >= word_limit,
        "add_after_truncated_double_carry_out": (
            truncated_double + addend >= word_limit
        ),
        "full_pre_mod_word_overflow_count": str(pre_mod // word_limit),
    }


def effective_width(L: int, p: int) -> int:
    return min(L, max(32, ((p.bit_length() + 31) // 32) * 32))


def detect_schedule(entry_type: type) -> str:
    direct = hasattr(entry_type, "_scan_bits")
    original = hasattr(entry_type, "_reduce") and hasattr(entry_type, "_mul")
    if direct == original:
        raise TypeError(
            "manifest entry class must expose exactly one supported schedule: "
            "_scan_bits or (_reduce and _mul)"
        )
    return "direct_two_pass" if direct else "original_three_pass"


def expected_canonical_scan_reduce(
    a_bits: list[int],
    b_bits: list[int],
) -> tuple[list[int], list[int]]:
    """Independent direct-schedule specification for scan/reduce ordering."""
    if (len(a_bits), a_bits) <= (len(b_bits), b_bits):
        return a_bits, b_bits
    return b_bits, a_bits


def expected_phase_counts(
    schedule: str,
    a_bits: list[int],
    b_bits: list[int],
    p: int,
    L: int,
    module,
) -> dict[str, int]:
    if schedule == "original_three_pass":
        return {
            "reduce_a": len(a_bits),
            "reduce_b": len(b_bits),
            "multiply": effective_width(L, p),
        }
    if schedule == "direct_two_pass":
        scan_bits, reduce_bits = expected_canonical_scan_reduce(a_bits, b_bits)
        return {
            "reduce_operand": len(reduce_bits),
            "scan_operand": len(scan_bits),
        }
    raise ValueError(f"unsupported schedule: {schedule}")


def append_teacher_phase(
    transitions: list[TeacherTransition],
    phase: str,
    digits: list[int],
    x: int,
    p: int,
    width: int,
) -> int:
    """Append an exact-state phase and return its final exact residue."""
    state = 0
    for phase_step, digit in enumerate(digits):
        expected = (2 * state + int(digit) * x) % p
        transitions.append(
            TeacherTransition(
                phase=phase,
                phase_step=phase_step,
                global_step=len(transitions),
                state=state,
                x=x,
                p=p,
                digit=int(digit),
                expected=expected,
                width=width,
            )
        )
        state = expected
    return state


def fixed_width_bits(value: int, width: int) -> list[int]:
    if value < 0 or value >= (1 << width):
        raise ValueError(f"value does not fit in {width} bits")
    return [(value >> shift) & 1 for shift in range(width - 1, -1, -1)]


def build_teacher_transitions(
    schedule: str,
    a_bits: list[int],
    b_bits: list[int],
    p: int,
    L: int,
    module,
) -> tuple[list[TeacherTransition], int]:
    """Materialize the exact prefixes implied by one candidate schedule."""
    width = effective_width(L, p)
    transitions: list[TeacherTransition] = []
    if schedule == "original_three_pass":
        ra = append_teacher_phase(
            transitions,
            "reduce_a",
            a_bits,
            1,
            p,
            width,
        )
        rb = append_teacher_phase(
            transitions,
            "reduce_b",
            b_bits,
            1,
            p,
            width,
        )
        product = append_teacher_phase(
            transitions,
            "multiply",
            fixed_width_bits(rb, width),
            ra,
            p,
            width,
        )
        return transitions, product
    if schedule == "direct_two_pass":
        scan_bits, reduce_bits = expected_canonical_scan_reduce(a_bits, b_bits)
        reduced = append_teacher_phase(
            transitions,
            "reduce_operand",
            reduce_bits,
            1,
            p,
            width,
        )
        product = append_teacher_phase(
            transitions,
            "scan_operand",
            scan_bits,
            reduced,
            p,
            width,
        )
        return transitions, product
    raise ValueError(f"unsupported schedule: {schedule}")


def build_cases(L: int) -> list[TraceCase]:
    if L != 2048:
        raise ValueError(f"the frozen F11 suite requires L=2048, found L={L}")

    p, _, _, categories = generate_battery(L, 128, seed=BATTERY_SEED)
    cases = [
        TraceCase(
            label="decisive-f11-x-1",
            group="decisive",
            source="explicit_regression",
            a=F11,
            b=1,
            p=p,
        )
    ]

    legacy_rows = [
        (source_index, a, b)
        for source_index, (a, b) in enumerate(categories["fermat numbers"])
        if a == F11
    ]
    if len(legacy_rows) != EXPECTED_LEGACY_CASES:
        raise RuntimeError(
            f"expected {EXPECTED_LEGACY_CASES} frozen legacy F11 cases, "
            f"found {len(legacy_rows)}"
        )
    cases.extend(
        TraceCase(
            label=f"legacy-fermat-source-{source_index:03d}",
            group="legacy",
            source="frozen_structured_battery_seed_20260627_n_128",
            a=a,
            b=b,
            p=p,
        )
        for source_index, a, b in legacy_rows
    )

    rng = random.Random(COMPANION_SEED)
    for width in COMPANION_WIDTHS:
        # Width 1 is the decisive F11 x 1 regression already inserted first.
        if width == 1:
            continue
        companion = (1 << (width - 1)) | rng.getrandbits(width - 1)
        if companion.bit_length() != width:
            raise RuntimeError(
                f"companion width mismatch: expected {width}, "
                f"found {companion.bit_length()}"
            )
        cases.append(
            TraceCase(
                label=f"companion-width-{width}",
                group="companions",
                source="deterministic_companion_seed_20260801",
                a=F11,
                b=companion,
                p=p,
            )
        )

    cases.extend(
        [
            TraceCase(
                label="equal-length-below-f11",
                group="ties",
                source="explicit_equal_length_tie",
                a=F11,
                b=F11 - 1,
                p=p,
            ),
            TraceCase(
                label="equal-length-equal-f11",
                group="ties",
                source="explicit_equal_length_tie",
                a=F11,
                b=F11,
                p=p,
            ),
            TraceCase(
                label="equal-length-above-f11",
                group="ties",
                source="explicit_equal_length_tie",
                a=F11,
                b=F11 + 2,
                p=p,
            ),
        ]
    )
    if len(cases) != EXPECTED_ALL_CASES:
        raise RuntimeError(
            f"expected {EXPECTED_ALL_CASES} total F11 cases, found {len(cases)}"
        )
    if len({case.label for case in cases}) != len(cases):
        raise RuntimeError("F11 case labels are not unique")
    if len({(case.a, case.b, case.p) for case in cases}) != len(cases):
        raise RuntimeError("F11 numerical cases are not unique")
    return cases


def select_cases(cases: list[TraceCase], groups: list[str]) -> list[TraceCase]:
    requested = groups or ["all"]
    if "all" in requested and len(requested) != 1:
        raise ValueError("--case-group all cannot be combined with another group")
    selected = (
        cases
        if requested == ["all"]
        else [case for case in cases if case.group in requested]
    )
    if not selected:
        raise ValueError("case-group selection is empty")
    expected = (
        len(cases)
        if requested == ["all"]
        else sum(1 for case in cases if case.group in set(requested))
    )
    if len(selected) != expected:
        raise RuntimeError(
            f"case selection cardinality mismatch: expected {expected}, "
            f"found {len(selected)}"
        )
    return selected


def case_payload(case: TraceCase) -> dict[str, str]:
    return {
        "label": case.label,
        "group": case.group,
        "source": case.source,
        "a": str(case.a),
        "b": str(case.b),
        "p": str(case.p),
    }


def load_external_case_fixture(
    fixture_path: Path,
) -> tuple[list[TraceCase], dict[str, object]]:
    """Load a hash-bound decimal case fixture without regenerating its inputs."""
    requested_path = (
        fixture_path if fixture_path.is_absolute() else Path.cwd() / fixture_path
    )
    if requested_path.is_symlink() or not requested_path.is_file():
        raise ValueError(f"case fixture is missing or symlinked: {requested_path}")
    resolved_path = requested_path.resolve()
    payload = json.loads(resolved_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("case fixture must contain one JSON object")
    if payload.get("schema") != EXTERNAL_CASE_FIXTURE_SCHEMA:
        raise ValueError(
            "case fixture schema mismatch: "
            f"expected {EXTERNAL_CASE_FIXTURE_SCHEMA!r}, "
            f"found {payload.get('schema')!r}"
        )
    rows = payload.get("cases")
    if not isinstance(rows, list) or not rows:
        raise ValueError("case fixture cases must be a non-empty list")
    expected_count = payload.get("expected_case_count")
    if expected_count != len(rows):
        raise ValueError(
            "case fixture cardinality mismatch: "
            f"expected {expected_count!r}, found {len(rows)}"
        )

    cases: list[TraceCase] = []
    observed_case_sha256: list[str] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"case fixture row {index} is not an object")
        text_fields: dict[str, str] = {}
        for field in ("label", "group", "source"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(
                    f"case fixture row {index} has invalid {field!r}"
                )
            text_fields[field] = value

        integers: dict[str, int] = {}
        for field in ("a", "b", "p"):
            value = row.get(field)
            if (
                not isinstance(value, str)
                or not value.isdecimal()
                or (len(value) > 1 and value.startswith("0"))
            ):
                raise ValueError(
                    f"case fixture row {index} has non-canonical {field!r}"
                )
            integers[field] = int(value)
        if integers["p"] < 2:
            raise ValueError(f"case fixture row {index} has p < 2")

        case = TraceCase(
            label=text_fields["label"],
            group=text_fields["group"],
            source=text_fields["source"],
            a=integers["a"],
            b=integers["b"],
            p=integers["p"],
        )
        digest = canonical_json_sha256(case_payload(case))
        if row.get("case_sha256") != digest:
            raise ValueError(
                f"case fixture row {index} SHA-256 mismatch: "
                f"expected {digest}, found {row.get('case_sha256')!r}"
            )
        cases.append(case)
        observed_case_sha256.append(digest)

    if len({case.label for case in cases}) != len(cases):
        raise ValueError("case fixture labels are not unique")
    if len({(case.a, case.b, case.p) for case in cases}) != len(cases):
        raise ValueError("case fixture numerical cases are not unique")

    identity = {
        "status": "verified",
        "path": str(resolved_path),
        "sha256": sha256_file(resolved_path),
        "schema": payload["schema"],
        "run_id": payload.get("run_id"),
        "expected_case_count": expected_count,
        "case_sha256": observed_case_sha256,
        "case_set_sha256": canonical_json_sha256(
            [case_payload(case) for case in cases]
        ),
    }
    return cases, identity


def compact_case_identity(case: TraceCase) -> dict[str, str | int]:
    return {
        "label": case.label,
        "group": case.group,
        "source": case.source,
        "case_sha256": canonical_json_sha256(case_payload(case)),
        "a_bits": case.a.bit_length(),
        "b_bits": case.b.bit_length(),
        "p_bits": case.p.bit_length(),
    }


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


def force_device(model, requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not (
        hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS was requested but is unavailable")
    device = torch.device(requested)
    if getattr(model, "model", None) is None:
        raise RuntimeError("loaded entry class does not expose a loaded model")
    model.device = device
    model.model.to(device)
    model.model.eval()
    return device


def invoke_candidate_step_with_logits(
    model,
    step,
    s_bits: torch.Tensor,
    x_bits: torch.Tensor,
    p_bits: torch.Tensor,
    digits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Invoke the entry's actual ``_step`` and capture its cell logits.

    A forward hook keeps the tracer downstream of the manifest implementation:
    routing, autocast, thresholding, and any future entry-level step logic are
    exercised by the candidate itself rather than reproduced by this runner.
    """
    if getattr(model, "model", None) is None:
        raise RuntimeError("model is not loaded")
    captured: list[torch.Tensor] = []

    def capture_logits(_module, _inputs, output) -> None:
        if not isinstance(output, torch.Tensor):
            raise RuntimeError("candidate cell forward did not return a tensor")
        captured.append(output)

    handle = model.model.register_forward_hook(capture_logits)
    try:
        predictions = step(s_bits, x_bits, p_bits, digits)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            "candidate _step must invoke the cell exactly once; "
            f"observed {len(captured)} forwards"
        )
    if not isinstance(predictions, torch.Tensor):
        raise RuntimeError("candidate _step did not return a tensor")
    logits = captured[0]
    if predictions.shape != s_bits.shape:
        raise RuntimeError(
            "candidate _step shape mismatch: "
            f"expected {tuple(s_bits.shape)}, found {tuple(predictions.shape)}"
        )
    if logits.shape != s_bits.shape:
        raise RuntimeError(
            "cell logits shape mismatch: "
            f"expected {tuple(s_bits.shape)}, found {tuple(logits.shape)}"
        )
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeError("cell produced a non-finite logit")
    binary_predictions = (predictions == 0) | (predictions == 1)
    if not bool(binary_predictions.all().item()):
        raise RuntimeError("candidate _step returned a non-binary state")
    return predictions.float(), logits


def make_tracer(base_type: type, schedule: str):
    """Build a tracer around the exact manifest entry implementation."""
    expected_order = (
        ["reduce_a", "reduce_b", "multiply"]
        if schedule == "original_three_pass"
        else ["reduce_operand", "scan_operand"]
    )

    class TrajectoryTracer(base_type):
        def begin_trace(self) -> None:
            self._trace_phase: str | None = None
            self._trace_phase_step = 0
            self._trace_global_step = 0
            self._trace_reduce_calls = 0
            self._trace_scan_calls = 0
            self._trace_phase_order: list[str] = []
            self._trace_phase_counts = {phase: 0 for phase in expected_order}
            self._trace_verified = 0
            self._trace_failed = 0
            self._trace_first_divergence: dict[str, object] | None = None
            self._trace_worst_margin: dict[str, object] | None = None
            self._trace_logit_dtypes: set[str] = set()
            self._trace_digest = hashlib.sha256()

        def _run_phase(self, phase: str, function, *args):
            if self._trace_phase is not None:
                raise RuntimeError(
                    f"nested trace phases: {self._trace_phase} then {phase}"
                )
            if phase not in self._trace_phase_counts:
                raise RuntimeError(f"unexpected trace phase: {phase}")
            self._trace_phase = phase
            self._trace_phase_step = 0
            self._trace_phase_order.append(phase)
            try:
                return function(*args)
            finally:
                self._trace_phase = None

        def _reduce(self, bit_lists, p_bits, dev):
            phase_index = self._trace_reduce_calls
            self._trace_reduce_calls += 1
            if phase_index >= 2:
                raise RuntimeError("original schedule called _reduce more than twice")
            phase = "reduce_a" if phase_index == 0 else "reduce_b"
            return self._run_phase(
                phase,
                super()._reduce,
                bit_lists,
                p_bits,
                dev,
            )

        def _mul(self, ra_bits, rb_bits, p_bits):
            return self._run_phase(
                "multiply",
                super()._mul,
                ra_bits,
                rb_bits,
                p_bits,
            )

        def _scan_bits(self, bit_lists, x_bits, p_bits, scan_width):
            phase_index = self._trace_scan_calls
            self._trace_scan_calls += 1
            if phase_index >= 2:
                raise RuntimeError("direct schedule called _scan_bits more than twice")
            phase = "reduce_operand" if phase_index == 0 else "scan_operand"
            return self._run_phase(
                phase,
                super()._scan_bits,
                bit_lists,
                x_bits,
                p_bits,
                scan_width,
            )

        def _step(self, s_bits, x_bits, p_bits, d):
            if self._trace_phase is None:
                raise RuntimeError("learned transition executed outside a trace phase")
            if s_bits.ndim != 2 or s_bits.shape[0] != 1:
                raise RuntimeError(
                    "F11 qualification traces require singleton transition batches"
                )
            if x_bits.shape != s_bits.shape or p_bits.shape != s_bits.shape:
                raise RuntimeError(
                    "state, x, and modulus tensors must have equal shape"
                )
            if d.ndim != 1 or d.shape[0] != 1:
                raise RuntimeError("transition digit tensor must contain one row")
            next_state, decision_logits = invoke_candidate_step_with_logits(
                self,
                super()._step,
                s_bits,
                x_bits,
                p_bits,
                d,
            )
            self._record_transition(
                s_bits,
                x_bits,
                p_bits,
                d,
                decision_logits,
                next_state,
            )
            return next_state

        def _record_transition(
            self,
            s_bits,
            x_bits,
            p_bits,
            d,
            logits,
            next_state,
        ) -> None:
            phase = self._trace_phase
            if phase is None:
                raise RuntimeError("missing phase while recording transition")
            self._trace_logit_dtypes.add(str(logits.dtype))
            width = int(s_bits.shape[1])
            state = bits_to_int(s_bits[0])
            x = bits_to_int(x_bits[0])
            p = bits_to_int(p_bits[0])
            digit = int(d[0].item())
            metadata = transition_arithmetic_metadata(state, x, p, digit, width)
            expected = int(metadata["expected_value"])
            expected_bits = int_to_bit_tensor(expected, width, logits.device)
            predicted_bits = next_state[0]
            wrong_mask = predicted_bits != expected_bits
            wrong_indices = torch.nonzero(wrong_mask, as_tuple=False).flatten()
            wrong_count = int(wrong_indices.numel())

            target_signs = expected_bits.mul(2).sub(1)
            signed_margins = target_signs * logits[0].float()
            minimum_margin, minimum_index = torch.min(signed_margins, dim=0)
            margin_value = float(minimum_margin.item())
            margin_index = int(minimum_index.item())

            location = {
                "phase": phase,
                "phase_step": self._trace_phase_step,
                "input_bit_index": self._trace_phase_step,
                "global_step": self._trace_global_step,
                "input_bit": digit,
                "state_width_bits": width,
            }
            margin_record = {
                **location,
                "minimum_signed_target_logit_margin": margin_value,
                "minimum_margin_bit_index": margin_index,
            }
            if (
                self._trace_worst_margin is None
                or margin_value
                < self._trace_worst_margin["minimum_signed_target_logit_margin"]
            ):
                self._trace_worst_margin = margin_record

            self._trace_digest.update(
                (
                    f"{phase}:{self._trace_phase_step}:"
                    f"{self._trace_global_step}:{digit}:{width}\n"
                ).encode()
            )
            for tensor in (
                s_bits[0],
                x_bits[0],
                p_bits[0],
                expected_bits,
                predicted_bits,
            ):
                self._trace_digest.update(bytes(tensor.detach().long().cpu().tolist()))

            self._trace_phase_counts[phase] += 1
            if wrong_count == 0:
                self._trace_verified += 1
            else:
                self._trace_failed += 1
                if self._trace_first_divergence is None:
                    first_wrong = int(wrong_indices[0].item())
                    last_wrong = int(wrong_indices[-1].item())
                    self._trace_first_divergence = {
                        **location,
                        "state": str(state),
                        "x": str(x),
                        "p": str(p),
                        "predicted_value": str(bits_to_int(predicted_bits)),
                        "expected_value": str(expected),
                        "wrong_output_bit_count": wrong_count,
                        "first_wrong_output_bit_index": first_wrong,
                        "last_wrong_output_bit_index": last_wrong,
                        "wrong_output_mask_sha256": hashlib.sha256(
                            bytes(wrong_mask.detach().long().cpu().tolist())
                        ).hexdigest(),
                        "minimum_signed_target_logit_margin": margin_value,
                        "minimum_margin_bit_index": margin_index,
                        "margin_source": "pre_threshold_logits_used_by_candidate_step",
                        **metadata,
                    }

            self._trace_phase_step += 1
            self._trace_global_step += 1

        def trace_summary(self) -> dict[str, object]:
            return {
                "phase_order": list(self._trace_phase_order),
                "phase_counts": dict(self._trace_phase_counts),
                "observed_transitions": self._trace_global_step,
                "verified_transitions": self._trace_verified,
                "failed_transitions": self._trace_failed,
                "first_divergence": self._trace_first_divergence,
                "worst_signed_target_logit_margin": self._trace_worst_margin,
                "margin_status": "available_from_exact_candidate_logits_hook",
                "captured_logit_dtypes": sorted(self._trace_logit_dtypes),
                "trajectory_sha256": self._trace_digest.hexdigest(),
            }

    TrajectoryTracer.__name__ = f"Tracing{base_type.__name__}"
    return TrajectoryTracer


def teacher_route_entry(transition: TeacherTransition) -> dict[str, object]:
    """Return the routing fields that the manifest replay must reproduce."""
    return {
        "phase": transition.phase,
        "phase_step": transition.phase_step,
        "global_step": transition.global_step,
        "state": str(transition.state),
        "x": str(transition.x),
        "p": str(transition.p),
        "d": transition.digit,
        "width": transition.width,
    }


def compare_routes(
    expected: list[dict[str, object]],
    observed: list[dict[str, object]],
) -> dict[str, object]:
    """Hash and compare two complete ordered manifest transition routes."""
    first_mismatch: dict[str, object] | None = None
    for index in range(max(len(expected), len(observed))):
        expected_row = expected[index] if index < len(expected) else None
        observed_row = observed[index] if index < len(observed) else None
        if expected_row == observed_row:
            continue
        fields = sorted(
            {
                *(expected_row.keys() if expected_row is not None else ()),
                *(observed_row.keys() if observed_row is not None else ()),
            }
        )
        differing_fields = [
            field
            for field in fields
            if (expected_row.get(field) if expected_row is not None else None)
            != (observed_row.get(field) if observed_row is not None else None)
        ]
        first_mismatch = {
            "transition_index": index,
            "differing_fields": differing_fields,
            "expected": expected_row,
            "observed": observed_row,
        }
        break
    count_exact = len(expected) == len(observed)
    route_exact = count_exact and first_mismatch is None
    return {
        "comparison_fields": [
            "phase",
            "phase_step",
            "global_step",
            "state",
            "x",
            "p",
            "d",
            "width",
        ],
        "expected_transition_count": len(expected),
        "observed_transition_count": len(observed),
        "transition_count_exact": count_exact,
        "expected_route_sha256": canonical_json_sha256(expected),
        "observed_route_sha256": canonical_json_sha256(observed),
        "first_route_mismatch": first_mismatch,
        "route_exact": route_exact,
    }


def validate_program_output_with_exact_step(
    model,
    module,
    schedule: str,
    a_enc: list[int],
    b_enc: list[int],
    p_enc: int,
    expected_output: int,
    teacher_transitions: list[TeacherTransition],
) -> dict[str, object]:
    """Replay manifest routing cheaply with an exact recurrence step.

    Candidate logits are certified separately.  This replay checks that the
    entry class emits exactly one binary, effective-width output when each
    learned transition is replaced by its specified recurrence.
    """

    observed_route: list[dict[str, object]] = []
    current_phase: str | None = None
    phase_step = 0
    global_step = 0
    reduce_calls = 0
    scan_calls = 0

    def exact_step(instance, s_bits, x_bits, p_bits, digits):
        nonlocal phase_step, global_step
        if s_bits.ndim != 2:
            raise RuntimeError("exact-step replay requires rank-two state tensors")
        if s_bits.shape[0] != 1:
            raise RuntimeError(
                "exact-step route replay requires a singleton oriented case"
            )
        if current_phase is None:
            raise RuntimeError("exact-step replay transition lacks a route phase")
        state = bits_to_int(s_bits[0])
        x = bits_to_int(x_bits[0])
        p = bits_to_int(p_bits[0])
        digit = int(digits[0].item())
        observed_route.append(
            {
                "phase": current_phase,
                "phase_step": phase_step,
                "global_step": global_step,
                "state": str(state),
                "x": str(x),
                "p": str(p),
                "d": digit,
                "width": int(s_bits.shape[1]),
            }
        )
        phase_step += 1
        global_step += 1
        values = [(2 * state + digit * x) % p]
        return module.to_bits_limbs(
            values,
            instance.device,
            s_bits.shape[1],
        ).float()

    def run_phase(phase: str, function, *args):
        nonlocal current_phase, phase_step
        if current_phase is not None:
            raise RuntimeError(f"nested exact-step replay phases at {phase}")
        current_phase = phase
        phase_step = 0
        try:
            return function(*args)
        finally:
            current_phase = None

    replaced_names = ["_step"]
    prior_instance_values = {"_step": model.__dict__.get("_step")}
    had_instance_values = {"_step": "_step" in model.__dict__}
    model._step = types.MethodType(exact_step, model)
    if schedule == "original_three_pass":
        original_reduce = model._reduce
        original_mul = model._mul

        def traced_reduce(instance, bit_lists, p_bits, dev):
            nonlocal reduce_calls
            if reduce_calls >= 2:
                raise RuntimeError("manifest replay called _reduce more than twice")
            phase = "reduce_a" if reduce_calls == 0 else "reduce_b"
            reduce_calls += 1
            return run_phase(phase, original_reduce, bit_lists, p_bits, dev)

        def traced_mul(instance, ra_bits, rb_bits, p_bits):
            return run_phase("multiply", original_mul, ra_bits, rb_bits, p_bits)

        for name, method in (("_reduce", traced_reduce), ("_mul", traced_mul)):
            replaced_names.append(name)
            prior_instance_values[name] = model.__dict__.get(name)
            had_instance_values[name] = name in model.__dict__
            setattr(model, name, types.MethodType(method, model))
    elif schedule == "direct_two_pass":
        original_scan = model._scan_bits

        def traced_scan(instance, bit_lists, x_bits, p_bits, scan_width):
            nonlocal scan_calls
            if scan_calls >= 2:
                raise RuntimeError("manifest replay called _scan_bits more than twice")
            phase = "reduce_operand" if scan_calls == 0 else "scan_operand"
            scan_calls += 1
            return run_phase(
                phase,
                original_scan,
                bit_lists,
                x_bits,
                p_bits,
                scan_width,
            )

        replaced_names.append("_scan_bits")
        prior_instance_values["_scan_bits"] = model.__dict__.get("_scan_bits")
        had_instance_values["_scan_bits"] = "_scan_bits" in model.__dict__
        model._scan_bits = types.MethodType(traced_scan, model)
    else:
        raise ValueError(f"unsupported schedule: {schedule}")
    try:
        # The manifest program owns routing; only its learned recurrence is
        # replaced for this independent route replay.
        outputs = model.predict_digits_batch([(a_enc, b_enc, p_enc)])
    finally:
        for name in reversed(replaced_names):
            if had_instance_values[name]:
                setattr(model, name, prior_instance_values[name])
            else:
                delattr(model, name)

    output_count_exact = len(outputs) == 1
    output = outputs[0] if output_count_exact else []
    expected_digit_count = effective_width(model.L, int(p_enc))
    digit_count_exact = len(output) == expected_digit_count
    alphabet_exact = all(
        isinstance(digit, int) and not isinstance(digit, bool) and digit in (0, 1)
        for digit in output
    )
    decoded = decode_digits(output) if alphabet_exact else None
    route_validation = compare_routes(
        [teacher_route_entry(transition) for transition in teacher_transitions],
        observed_route,
    )
    exact = all(
        (
            output_count_exact,
            digit_count_exact,
            alphabet_exact,
            decoded == expected_output,
            route_validation["route_exact"],
        )
    )
    return {
        "mode": "manifest_program_replay_with_exact_recurrence_step",
        "output_count": {"expected": 1, "observed": len(outputs)},
        "output_count_exact": output_count_exact,
        "output_digit_count": {
            "expected": expected_digit_count,
            "observed": len(output),
        },
        "output_digit_count_exact": digit_count_exact,
        "output_alphabet_exact": alphabet_exact,
        "decoded_output": str(decoded) if decoded is not None else None,
        "expected_output": str(expected_output),
        "output_sha256": (
            hashlib.sha256(bytes(output)).hexdigest() if alphabet_exact else None
        ),
        "route_validation": route_validation,
        "exact": exact,
    }


@torch.no_grad()
def run_inductive_case(
    model,
    module,
    schedule: str,
    case: TraceCase,
    orientation: str,
    transition_batch_size: int,
) -> dict[str, object]:
    """Certify a rollout by batched evaluation of every exact teacher prefix."""
    if orientation not in ("original", "swapped"):
        raise ValueError(f"unsupported orientation: {orientation}")
    if transition_batch_size <= 0:
        raise ValueError("transition_batch_size must be positive")
    a, b = (case.a, case.b) if orientation == "original" else (case.b, case.a)
    a_enc = list(model.preprocess_a(a))
    b_enc = list(model.preprocess_b(b))
    p_enc = model.preprocess_p(case.p)
    phases_expected = expected_phase_counts(
        schedule,
        a_enc,
        b_enc,
        case.p,
        model.L,
        module,
    )
    transitions, teacher_final = build_teacher_transitions(
        schedule,
        a_enc,
        b_enc,
        case.p,
        model.L,
        module,
    )
    transitions_expected = sum(phases_expected.values())
    if len(transitions) != transitions_expected:
        raise RuntimeError(
            "teacher transition cardinality mismatch: "
            f"expected {transitions_expected}, found {len(transitions)}"
        )
    expected_output = (a * b) % case.p
    if teacher_final != expected_output:
        raise RuntimeError(
            "exact teacher schedule does not compute the modular product"
        )

    phase_counts = {phase: 0 for phase in phases_expected}
    verified = 0
    failed = 0
    first_divergence: dict[str, object] | None = None
    worst_margin: dict[str, object] | None = None
    captured_logit_dtypes: set[str] = set()
    trace_digest = hashlib.sha256()
    width = effective_width(model.L, case.p)

    for start in range(0, len(transitions), transition_batch_size):
        chunk = transitions[start : start + transition_batch_size]
        states = module.to_bits_limbs(
            [transition.state for transition in chunk],
            model.device,
            width,
        ).float()
        xs = module.to_bits_limbs(
            [transition.x for transition in chunk],
            model.device,
            width,
        ).float()
        ps = module.to_bits_limbs(
            [transition.p for transition in chunk],
            model.device,
            width,
        ).float()
        digits = torch.tensor(
            [transition.digit for transition in chunk],
            dtype=torch.long,
            device=model.device,
        )
        targets = module.to_bits_limbs(
            [transition.expected for transition in chunk],
            model.device,
            width,
        ).float()
        predictions, logits = invoke_candidate_step_with_logits(
            model,
            model._step,
            states,
            xs,
            ps,
            digits,
        )
        captured_logit_dtypes.add(str(logits.dtype))
        wrong_masks = predictions != targets
        wrong_counts = wrong_masks.sum(dim=1).detach().cpu().tolist()
        signed_margins = targets.mul(2).sub(1) * logits.float()
        margin_values, margin_indices = torch.min(signed_margins, dim=1)
        margin_values_cpu = margin_values.detach().cpu().tolist()
        margin_indices_cpu = margin_indices.detach().cpu().tolist()

        for row_index, transition in enumerate(chunk):
            phase_counts[transition.phase] += 1
            wrong_count = int(wrong_counts[row_index])
            margin_value = float(margin_values_cpu[row_index])
            margin_index = int(margin_indices_cpu[row_index])
            location = {
                "phase": transition.phase,
                "phase_step": transition.phase_step,
                "input_bit_index": transition.phase_step,
                "global_step": transition.global_step,
                "input_bit": transition.digit,
                "state_width_bits": transition.width,
            }
            margin_record = {
                **location,
                "minimum_signed_target_logit_margin": margin_value,
                "minimum_margin_bit_index": margin_index,
            }
            if (
                worst_margin is None
                or margin_value < worst_margin["minimum_signed_target_logit_margin"]
            ):
                worst_margin = margin_record

            predicted_bits = predictions[row_index]
            predicted_value = bits_to_int(predicted_bits)
            trace_digest.update(
                (
                    f"{transition.phase}:{transition.phase_step}:"
                    f"{transition.global_step}:{transition.state}:"
                    f"{transition.x}:{transition.p}:{transition.digit}:"
                    f"{transition.expected}:{predicted_value}\n"
                ).encode()
            )
            if wrong_count == 0:
                verified += 1
                continue

            failed += 1
            if first_divergence is None:
                wrong_indices = torch.nonzero(
                    wrong_masks[row_index],
                    as_tuple=False,
                ).flatten()
                metadata = transition_arithmetic_metadata(
                    transition.state,
                    transition.x,
                    transition.p,
                    transition.digit,
                    transition.width,
                )
                first_divergence = {
                    **location,
                    "state": str(transition.state),
                    "x": str(transition.x),
                    "p": str(transition.p),
                    "predicted_value": str(predicted_value),
                    "expected_value": str(transition.expected),
                    "wrong_output_bit_count": wrong_count,
                    "first_wrong_output_bit_index": int(wrong_indices[0].item()),
                    "last_wrong_output_bit_index": int(wrong_indices[-1].item()),
                    "wrong_output_mask_sha256": hashlib.sha256(
                        bytes(wrong_masks[row_index].detach().long().cpu().tolist())
                    ).hexdigest(),
                    "minimum_signed_target_logit_margin": margin_value,
                    "minimum_margin_bit_index": margin_index,
                    "margin_source": ("pre_threshold_logits_used_by_candidate_step"),
                    "free_running_divergence_valid_by_exact_prefix_induction": True,
                    **metadata,
                }

    phase_order = list(phases_expected)
    phase_counts_exact = phase_counts == phases_expected
    transition_count_exact = len(transitions) == transitions_expected
    transitions_exact = failed == 0
    strictly_positive_margin = bool(
        worst_margin is not None
        and worst_margin["minimum_signed_target_logit_margin"] > 0
    )
    certified_prefix = (
        transitions_expected
        if first_divergence is None
        else int(first_divergence["global_step"])
    )
    output_validation = validate_program_output_with_exact_step(
        model,
        module,
        schedule,
        a_enc,
        b_enc,
        p_enc,
        expected_output,
        transitions,
    )
    candidate_full_rollout_certified = all(
        (
            transitions_exact,
            phase_counts_exact,
            transition_count_exact,
            strictly_positive_margin,
            output_validation["exact"],
        )
    )

    oriented_payload = {
        **case_payload(case),
        "orientation": orientation,
        "oriented_a": str(a),
        "oriented_b": str(b),
    }
    return {
        **compact_case_identity(case),
        "orientation": orientation,
        "oriented_case_sha256": canonical_json_sha256(oriented_payload),
        "oriented_a_bits": a.bit_length(),
        "oriented_b_bits": b.bit_length(),
        "execution_mode": "vectorized_exact_prefix_induction",
        "certification_basis": (
            "all candidate logits are evaluated on exact teacher states; the "
            "first failure is free-running-valid because every prior transition "
            "was verified; a zero-failure result certifies the full rollout by "
            "induction"
        ),
        "transition_batch_size": transition_batch_size,
        "expected_phase_order": phase_order,
        "phase_order": phase_order,
        "phase_order_exact": True,
        "expected_phase_counts": phases_expected,
        "phase_counts": phase_counts,
        "phase_counts_exact": phase_counts_exact,
        "expected_transitions": transitions_expected,
        "observed_transitions": len(transitions),
        "teacher_transitions_evaluated": len(transitions),
        "transition_count_exact": transition_count_exact,
        "verified_transitions": verified,
        "failed_transitions": failed,
        "transitions_exact": transitions_exact,
        "free_running_prefix_transitions_certified": certified_prefix,
        "post_divergence_teacher_transitions_evaluated": (
            0
            if first_divergence is None
            else transitions_expected - certified_prefix - 1
        ),
        "first_divergence": first_divergence,
        "worst_signed_target_logit_margin": worst_margin,
        "margin_status": "available_from_exact_candidate_logits_hook",
        "captured_logit_dtypes": sorted(captured_logit_dtypes),
        "strictly_positive_margin": strictly_positive_margin,
        "margin_gate_passed": strictly_positive_margin,
        "candidate_prediction_path": (
            "entry_instance_actual__step_with_cell_forward_hook"
        ),
        "trajectory_sha256": trace_digest.hexdigest(),
        "trajectory_hash_semantics": (
            "ordered exact-teacher inputs, candidate predictions, and targets"
        ),
        "program_output_validation": output_validation,
        "manifest_route_validation": output_validation["route_validation"],
        "output_count": output_validation["output_count"],
        "output_count_exact": output_validation["output_count_exact"],
        "output_digit_count": output_validation["output_digit_count"],
        "output_digit_count_exact": output_validation["output_digit_count_exact"],
        "output_alphabet_exact": output_validation["output_alphabet_exact"],
        "output_sha256": output_validation["output_sha256"],
        "decoded_output": (
            output_validation["decoded_output"]
            if candidate_full_rollout_certified
            else None
        ),
        "expected_output": str(expected_output),
        "final_exact": candidate_full_rollout_certified,
        "candidate_full_rollout_certified": candidate_full_rollout_certified,
        "all_exact": candidate_full_rollout_certified,
        "candidate_prefix_executed": True,
        "incoming_state_source": "candidate_owned_exact_prefix_induction",
    }


def run_oriented_case(
    model,
    module,
    schedule: str,
    case: TraceCase,
    orientation: str,
) -> dict[str, object]:
    if orientation not in ("original", "swapped"):
        raise ValueError(f"unsupported orientation: {orientation}")
    a, b = (case.a, case.b) if orientation == "original" else (case.b, case.a)
    a_enc = model.preprocess_a(a)
    b_enc = model.preprocess_b(b)
    p_enc = model.preprocess_p(case.p)
    phases_expected = expected_phase_counts(
        schedule,
        list(a_enc),
        list(b_enc),
        case.p,
        model.L,
        module,
    )
    transitions_expected = sum(phases_expected.values())

    model.begin_trace()
    outputs = model.predict_digits_batch([(a_enc, b_enc, p_enc)])
    trace = model.trace_summary()

    output_count_exact = len(outputs) == 1
    output = outputs[0] if output_count_exact else []
    expected_digit_count = effective_width(model.L, case.p)
    output_digit_count_exact = len(output) == expected_digit_count
    output_alphabet_exact = all(
        isinstance(digit, int) and not isinstance(digit, bool) and digit in (0, 1)
        for digit in output
    )
    decoded = decode_digits(output) if output_alphabet_exact else None
    expected_output = (a * b) % case.p
    final_exact = decoded == expected_output

    expected_order = list(phases_expected)
    phase_order_exact = trace["phase_order"] == expected_order
    phase_counts_exact = trace["phase_counts"] == phases_expected
    transition_count_exact = trace["observed_transitions"] == transitions_expected
    transitions_exact = trace["failed_transitions"] == 0
    worst_margin = trace["worst_signed_target_logit_margin"]
    strictly_positive_margin = bool(
        worst_margin is not None
        and worst_margin["minimum_signed_target_logit_margin"] > 0
    )
    all_exact = all(
        (
            output_count_exact,
            output_digit_count_exact,
            output_alphabet_exact,
            final_exact,
            phase_order_exact,
            phase_counts_exact,
            transition_count_exact,
            transitions_exact,
            strictly_positive_margin,
        )
    )

    oriented_payload = {
        **case_payload(case),
        "orientation": orientation,
        "oriented_a": str(a),
        "oriented_b": str(b),
    }
    result: dict[str, object] = {
        **compact_case_identity(case),
        "orientation": orientation,
        "oriented_case_sha256": canonical_json_sha256(oriented_payload),
        "oriented_a_bits": a.bit_length(),
        "oriented_b_bits": b.bit_length(),
        "execution_mode": "literal_sequential_replay",
        "certification_basis": (
            "candidate entry class executed every learned transition in literal "
            "free-running schedule order"
        ),
        "expected_phase_order": expected_order,
        "expected_phase_counts": phases_expected,
        "expected_transitions": transitions_expected,
        **trace,
        "output_count": {"expected": 1, "observed": len(outputs)},
        "output_count_exact": output_count_exact,
        "output_digit_count": {
            "expected": expected_digit_count,
            "observed": len(output),
        },
        "output_digit_count_exact": output_digit_count_exact,
        "output_alphabet_exact": output_alphabet_exact,
        "output_sha256": (
            hashlib.sha256(bytes(output)).hexdigest() if output_alphabet_exact else None
        ),
        "decoded_output": str(decoded) if decoded is not None else None,
        "expected_output": str(expected_output),
        "final_exact": final_exact,
        "phase_order_exact": phase_order_exact,
        "phase_counts_exact": phase_counts_exact,
        "transition_count_exact": transition_count_exact,
        "transitions_exact": transitions_exact,
        "strictly_positive_margin": strictly_positive_margin,
        "margin_gate_passed": strictly_positive_margin,
        "candidate_prediction_path": (
            "entry_instance_actual__step_with_cell_forward_hook"
        ),
        "all_exact": all_exact,
        "candidate_prefix_executed": transition_count_exact,
        "incoming_state_source": "candidate_owned_full_trajectory_from_zero",
    }
    return result


def write_receipt(path: Path, receipt: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(receipt, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument(
        "--case-group",
        action="append",
        choices=("all", "decisive", "legacy", "companions", "ties"),
        default=[],
        help="repeat to combine groups; default is the complete suite",
    )
    parser.add_argument(
        "--case-fixture",
        type=Path,
        help="external hash-bound case fixture; cannot be combined with --case-group",
    )
    parser.add_argument(
        "--orientation",
        choices=("original", "swapped", "both"),
        default="both",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "mps", "cuda"),
        default="cpu",
    )
    parser.add_argument(
        "--mode",
        choices=("inductive", "sequential"),
        default="inductive",
        help="vectorized exact-prefix induction (default) or literal replay",
    )
    parser.add_argument(
        "--chunk-size",
        "--transition-batch-size",
        dest="transition_batch_size",
        type=int,
        default=256,
        help="teacher transitions per candidate-cell forward in inductive mode",
    )
    parser.add_argument(
        "--continue-after-failure",
        action="store_true",
        help="trace every selected case after recording the first nonpassing result",
    )
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()
    if args.transition_batch_size <= 0:
        parser.error("--chunk-size must be positive")
    if args.case_fixture is not None and args.case_group:
        parser.error("--case-fixture cannot be combined with --case-group")

    source_identity_before_load = qualification_source_identity()
    source_set_sha256 = canonical_json_sha256(source_identity_before_load)
    (
        submission,
        manifest,
        module,
        loaded,
        artifact_identity_before_load,
        artifact_identity_after_guarded_load,
    ) = guarded_load_submission(args.submission)
    entry_type = type(loaded)
    schedule = detect_schedule(entry_type)
    artifact_set_sha256 = canonical_json_sha256(artifact_identity_before_load)

    if args.mode == "sequential":
        tracer_type = make_tracer(entry_type, schedule)
        model = tracer_type()
        model.load(str(submission))
    else:
        model = loaded
    artifact_identity_after_load = artifact_identity(submission)
    if artifact_identity_before_load != artifact_identity_after_load:
        changed = sorted(
            name
            for name in set(artifact_identity_before_load)
            | set(artifact_identity_after_load)
            if artifact_identity_before_load.get(name)
            != artifact_identity_after_load.get(name)
        )
        raise RuntimeError(
            "submission artifacts changed during active model load: "
            + ", ".join(changed)
        )
    if artifact_identity_after_guarded_load != artifact_identity_after_load:
        raise RuntimeError("submission artifacts changed between model loads")
    source_identity_after_load = qualification_source_identity()
    if source_identity_before_load != source_identity_after_load:
        raise RuntimeError("qualification sources changed during submission load")
    device = force_device(model, args.device)
    model_tensor_identity = model_tensor_dtype_identity(model)
    if model.L != 2048:
        raise ValueError(f"F11 qualification requires L=2048, found {model.L}")

    case_fixture_identity: dict[str, object] | None = None
    if args.case_fixture is None:
        all_cases = build_cases(model.L)
        selected = select_cases(all_cases, args.case_group)
    else:
        all_cases, case_fixture_identity = load_external_case_fixture(
            args.case_fixture
        )
        selected = all_cases
    orientations = (
        ("original", "swapped") if args.orientation == "both" else (args.orientation,)
    )
    expected_results = len(selected) * len(orientations)
    expected_transitions = 0
    for case in selected:
        for orientation in orientations:
            a, b = (case.a, case.b) if orientation == "original" else (case.b, case.a)
            expected_transitions += sum(
                expected_phase_counts(
                    schedule,
                    model.preprocess_a(a),
                    model.preprocess_b(b),
                    case.p,
                    model.L,
                    module,
                ).values()
            )

    environment = environment_identity(device)
    selected_payloads = [case_payload(case) for case in selected]
    full_payloads = [case_payload(case) for case in all_cases]
    receipt: dict[str, object] = {
        "status": "running",
        "all_exact": False,
        "submission": str(submission),
        "entry_class": manifest["entry_class"],
        "loaded_entry_type": f"{entry_type.__module__}.{entry_type.__name__}",
        "schedule": schedule,
        "execution_mode": (
            "vectorized_exact_prefix_induction"
            if args.mode == "inductive"
            else "literal_sequential_replay"
        ),
        "transition_batch_size": (
            args.transition_batch_size if args.mode == "inductive" else None
        ),
        "artifact_sha256": artifact_identity_before_load,
        "artifact_sha256_before_load": artifact_identity_before_load,
        "artifact_sha256_after_load": artifact_identity_after_load,
        "artifact_unchanged_during_load": (
            artifact_identity_before_load == artifact_identity_after_load
        ),
        "artifact_set_sha256": artifact_set_sha256,
        "runner_sha256": source_identity_before_load["local_source_sha256"][
            "scripts/trace_f11_trajectories.py"
        ],
        "qualification_source_identity": source_identity_before_load,
        "qualification_source_set_sha256": source_set_sha256,
        "qualification_source_identity_after_load": source_identity_after_load,
        "qualification_source_set_sha256_after_load": canonical_json_sha256(
            source_identity_after_load
        ),
        "qualification_sources_unchanged_during_load": True,
        "full_case_set_sha256": canonical_json_sha256(full_payloads),
        "selected_case_set_sha256": canonical_json_sha256(selected_payloads),
        "battery_seed": BATTERY_SEED if case_fixture_identity is None else None,
        "companion_seed": COMPANION_SEED if case_fixture_identity is None else None,
        "case_fixture_identity": case_fixture_identity,
        "selected_groups": (
            args.case_group or ["all"]
            if case_fixture_identity is None
            else ["external_fixture"]
        ),
        "selected_cases": [compact_case_identity(case) for case in selected],
        "orientation_request": args.orientation,
        "expected_orientations": len(orientations),
        "expected_cases_per_orientation": len(selected),
        "expected_results": expected_results,
        "expected_transitions": expected_transitions,
        "continue_after_failure": args.continue_after_failure,
        "environment": environment,
        "environment_sha256": canonical_json_sha256(environment),
        "model_tensor_dtype_identity": model_tensor_identity,
        "model_tensor_dtype_identity_sha256": canonical_json_sha256(
            model_tensor_identity
        ),
        "margin_status": "available_from_exact_candidate_logits_hook",
        "prefix_policy": (
            "candidate_owned_exact_teacher_prefixes_evaluated_in_order"
            if args.mode == "inductive"
            else "candidate_owned_literal_full_trajectory_from_zero_only"
        ),
        "results": [],
        "completed_results": 0,
        "completed_transitions": 0,
    }
    write_receipt(args.json_out, receipt)

    try:
        # Case-major order makes the decisive F11 x 1 result first and keeps its
        # two orientations adjacent in a partial long-running receipt.
        first_nonpassing_result: dict[str, object] | None = None
        stop_requested = False
        for case in selected:
            for orientation in orientations:
                if args.mode == "inductive":
                    result = run_inductive_case(
                        model,
                        module,
                        schedule,
                        case,
                        orientation,
                        args.transition_batch_size,
                    )
                else:
                    result = run_oriented_case(
                        model,
                        module,
                        schedule,
                        case,
                        orientation,
                    )
                receipt["results"].append(result)
                receipt["completed_results"] = len(receipt["results"])
                receipt["completed_transitions"] = sum(
                    item["observed_transitions"] for item in receipt["results"]
                )
                write_receipt(args.json_out, receipt)
                print(
                    f"case={case.label} orientation={orientation} "
                    f"transitions={result['verified_transitions']}/"
                    f"{result['observed_transitions']} "
                    f"final_exact={result['final_exact']}"
                )
                if not result["all_exact"]:
                    if first_nonpassing_result is None:
                        first_nonpassing_result = {
                            "label": case.label,
                            "orientation": orientation,
                            "oriented_case_sha256": result[
                                "oriented_case_sha256"
                            ],
                            "completed_result_index": len(receipt["results"]) - 1,
                        }
                    if not args.continue_after_failure:
                        stop_requested = True
                        break
            if stop_requested:
                break
    except BaseException as error:
        receipt["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        )
        receipt["error"] = {
            "type": type(error).__name__,
            "message": str(error),
        }
        write_receipt(args.json_out, receipt)
        raise

    artifact_identity_after_run = artifact_identity(submission)
    artifact_unchanged_during_run = (
        artifact_identity_after_load == artifact_identity_after_run
    )
    artifact_unchanged_across_load_and_run = (
        artifact_identity_before_load
        == artifact_identity_after_load
        == artifact_identity_after_run
    )
    source_identity_after_run = qualification_source_identity()
    qualification_sources_unchanged_during_run = (
        source_identity_after_load == source_identity_after_run
    )
    result_count_exact = len(receipt["results"]) == expected_results
    transition_cardinality_exact = (
        receipt["completed_transitions"] == expected_transitions
    )
    orientation_counts = {
        orientation: sum(
            result["orientation"] == orientation for result in receipt["results"]
        )
        for orientation in orientations
    }
    orientation_cardinality_exact = all(
        count == len(selected) for count in orientation_counts.values()
    )
    all_exact = all(result["all_exact"] for result in receipt["results"]) and all(
        (
            artifact_unchanged_across_load_and_run,
            qualification_sources_unchanged_during_run,
            result_count_exact,
            transition_cardinality_exact,
            orientation_cardinality_exact,
        )
    )
    receipt.update(
        {
            "status": "completed_exact" if all_exact else "failed",
            "all_exact": all_exact,
            "stopped_early_on_nonpassing_result": (
                first_nonpassing_result is not None
                and len(receipt["results"]) < expected_results
            ),
            "first_nonpassing_result": first_nonpassing_result,
            "artifact_sha256_after": artifact_identity_after_run,
            "artifact_sha256_after_run": artifact_identity_after_run,
            "artifact_unchanged_during_run": artifact_unchanged_during_run,
            "artifact_unchanged_across_load_and_run": (
                artifact_unchanged_across_load_and_run
            ),
            "qualification_source_identity_after_run": (
                source_identity_after_run
            ),
            "qualification_source_set_sha256_after_run": canonical_json_sha256(
                source_identity_after_run
            ),
            "qualification_sources_unchanged_during_run": (
                qualification_sources_unchanged_during_run
            ),
            "captured_logit_dtypes": sorted(
                {
                    dtype_name
                    for result in receipt["results"]
                    for dtype_name in result["captured_logit_dtypes"]
                }
            ),
            "result_count_exact": result_count_exact,
            "transition_cardinality_exact": transition_cardinality_exact,
            "orientation_counts": orientation_counts,
            "orientation_cardinality_exact": orientation_cardinality_exact,
        }
    )
    write_receipt(args.json_out, receipt)
    print(
        f"SUMMARY status={receipt['status']} "
        f"results={len(receipt['results'])}/{expected_results} "
        f"receipt={args.json_out}"
    )
    return 0 if all_exact else 1


if __name__ == "__main__":
    sys.exit(main())
