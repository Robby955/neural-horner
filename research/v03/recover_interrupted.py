#!/usr/bin/env python3
"""Fail-closed recovery for an interrupted NeuralHorner v0.3 horizon run.

This runner preserves the original experiment receipt, model, optimizer,
scheduler, and random streams.  It accepts only a fully recorded evaluation
boundary and continues the same absolute training horizon.  A provider exit is
recorded as an operational interruption, not as a scientific gate result.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import torch
from torch import nn


ROOT = pathlib.Path(__file__).resolve().parents[2]
TRAINER_PATH = ROOT / "research/v02/train_scale.py"
RECOVERY_SCHEMA = "neuralhorner-v03-interrupted-recovery-v1"
INTERRUPTED_STATUSES = {"running", "interrupted_operational"}


def _load_trainer() -> Any:
    spec = importlib.util.spec_from_file_location(
        "neuralhorner_v03_original_trainer", TRAINER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load the original v0.3 trainer")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


TRAINER = _load_trainer()


def _git(*arguments: str, binary: bool = False) -> str | bytes:
    output = subprocess.check_output(
        ["git", *arguments],
        cwd=ROOT,
        text=not binary,
    )
    return output if binary else output.strip()


def recovery_source_identity(config_path: pathlib.Path) -> dict[str, Any]:
    runner_path = pathlib.Path(__file__).resolve()
    status = str(_git("status", "--porcelain=v1", "--untracked-files=all")).splitlines()
    identity = {
        "runner_path": TRAINER.receipt_source_path(runner_path),
        "runner_sha256": TRAINER.sha256_file(runner_path),
        "trainer_path": TRAINER.receipt_source_path(TRAINER_PATH),
        "trainer_sha256": TRAINER.sha256_file(TRAINER_PATH),
        "config_path": TRAINER.receipt_source_path(config_path),
        "config_sha256": TRAINER.sha256_file(config_path),
        "git": {
            "head": _git("rev-parse", "HEAD"),
            "branch": _git("branch", "--show-current"),
            "status": status,
        },
    }
    if status:
        raise RuntimeError("interrupted recovery requires a clean Git worktree")
    for path_key, sha_key in (
        ("runner_path", "runner_sha256"),
        ("trainer_path", "trainer_sha256"),
        ("config_path", "config_sha256"),
    ):
        path = identity[path_key]
        if pathlib.Path(path).is_absolute():
            raise RuntimeError(
                f"recovery source {path_key} must be inside the repository"
            )
        try:
            head_bytes = _git("show", f"HEAD:{path}", binary=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"recovery source {path} is not present at HEAD"
            ) from error
        if TRAINER.sha256_bytes(head_bytes) != identity[sha_key]:
            raise RuntimeError(f"recovery source {path} differs from its HEAD bytes")
    return identity


def _original_source_is_available(source: dict[str, Any]) -> None:
    git = source.get("git")
    if not isinstance(git, dict) or not TRAINER.is_git_sha(git.get("head")):
        raise ValueError("interrupted receipt has invalid original Git identity")
    expected = (
        ("trainer_path", "trainer_sha256"),
        ("config_path", "config_sha256"),
        ("source_provenance_path", "source_provenance_sha256"),
    )
    for path_key, sha_key in expected:
        relative = source.get(path_key)
        digest = source.get(sha_key)
        if (
            not isinstance(relative, str)
            or pathlib.Path(relative).is_absolute()
            or not TRAINER.is_sha256(digest)
        ):
            raise ValueError(f"interrupted receipt has invalid source field {path_key}")
        local_path = ROOT / relative
        if not local_path.is_file() or TRAINER.sha256_file(local_path) != digest:
            raise ValueError(f"original source bytes are unavailable: {relative}")
        try:
            original_bytes = _git("show", f"{git['head']}:{relative}", binary=True)
        except subprocess.CalledProcessError as error:
            raise ValueError(
                f"original source commit does not contain {relative}"
            ) from error
        if TRAINER.sha256_bytes(original_bytes) != digest:
            raise ValueError(f"original Git bytes do not match receipt: {relative}")


def _ordered_steps(rows: Any, field: str) -> list[int]:
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"interrupted receipt has no {field}")
    steps: list[int] = []
    for row in rows:
        if not isinstance(row, dict) or type(row.get("step")) is not int:
            raise ValueError(f"interrupted receipt has invalid {field} step")
        steps.append(row["step"])
    if steps != sorted(steps) or len(steps) != len(set(steps)):
        raise ValueError(f"interrupted receipt {field} are not strictly ordered")
    return steps


def _safe_checkpoint_path(output: pathlib.Path, name: Any) -> pathlib.Path:
    if not isinstance(name, str) or pathlib.Path(name).name != name:
        raise ValueError("interrupted receipt checkpoint path is not a basename")
    expected_prefix = "weights_step"
    if not name.startswith(expected_prefix) or not name.endswith(".pt"):
        raise ValueError("interrupted receipt checkpoint path is not canonical")
    return output / name


def find_uncommitted_artifacts(
    output: pathlib.Path,
    recorded_checkpoint_names: set[str],
    latest_step: int,
) -> list[dict[str, Any]]:
    artifacts: list[dict[str, Any]] = []
    for path in sorted(output.iterdir()):
        name = path.name
        if name in recorded_checkpoint_names or not path.is_file():
            continue
        step: int | None = None
        kind: str | None = None
        if name.startswith("weights_step") and name.endswith(".pt"):
            raw_step = name[len("weights_step") : -len(".pt")]
            kind = "unreferenced_checkpoint"
        elif name.startswith(".weights_step") and name.endswith(".pt.tmp"):
            raw_step = name[len(".weights_step") : -len(".pt.tmp")]
            kind = "checkpoint_temporary"
        elif name == "receipt.json.tmp":
            raw_step = ""
            kind = "receipt_temporary"
        else:
            continue
        if raw_step:
            if not raw_step.isdigit():
                raise ValueError(f"noncanonical uncommitted checkpoint name: {name}")
            step = int(raw_step)
            if step <= latest_step:
                raise ValueError(
                    f"unreferenced checkpoint is not after the receipt boundary: {name}"
                )
        artifacts.append(
            {
                "path": name,
                "sha256": TRAINER.sha256_file(path),
                "size": path.stat().st_size,
                "step": step,
                "kind": kind,
            }
        )
    return artifacts


def quarantine_uncommitted_artifacts(
    output: pathlib.Path,
    artifacts: list[dict[str, Any]],
    recovery_id: str,
) -> list[dict[str, Any]]:
    if not artifacts:
        return []
    quarantine = output / "interrupted_uncommitted" / recovery_id
    quarantine.mkdir(parents=True, exist_ok=False)
    moved: list[dict[str, Any]] = []
    for artifact in artifacts:
        source = output / artifact["path"]
        if TRAINER.sha256_file(source) != artifact["sha256"]:
            raise RuntimeError(f"uncommitted artifact changed: {source.name}")
        destination = quarantine / source.name
        os.replace(source, destination)
        if TRAINER.sha256_file(destination) != artifact["sha256"]:
            raise RuntimeError(
                f"quarantined artifact verification failed: {source.name}"
            )
        moved.append(
            {
                **artifact,
                "quarantine_path": str(destination.relative_to(output)),
            }
        )
    return moved


def validate_interrupted_receipt(
    output: pathlib.Path,
    config: dict[str, Any],
    arm_name: str,
    environment: dict[str, Any],
    expected_receipt_sha256: str,
) -> tuple[dict[str, Any], pathlib.Path, dict[str, Any], str]:
    receipt_path = output / "receipt.json"
    if not TRAINER.is_sha256(expected_receipt_sha256):
        raise ValueError("--expected-receipt-sha256 must be a lowercase SHA-256")
    receipt_sha = TRAINER.sha256_file(receipt_path)
    if receipt_sha != expected_receipt_sha256:
        raise ValueError("interrupted receipt SHA-256 does not match the operator pin")
    receipt = json.loads(receipt_path.read_text())
    arm = TRAINER.validate_config(config, arm_name)
    history = receipt.get("history")
    checkpoints = receipt.get("checkpoints")
    history_steps = _ordered_steps(history, "history")
    checkpoint_steps = _ordered_steps(checkpoints, "checkpoints")
    parent_end = config["schedule_extension"]["parent_end_step"]
    eval_every = config["eval_every"]
    latest = history_steps[-1]
    expected_steps = list(range(parent_end + eval_every, latest + 1, eval_every))
    source = receipt.get("source_identity")
    checks = {
        "schema": receipt.get("schema") == TRAINER.HORIZON_RECEIPT_SCHEMA,
        "status": receipt.get("status") in INTERRUPTED_STATUSES,
        "experiment": receipt.get("experiment") == config["name"],
        "role": receipt.get("role") == config["role"],
        "arm": receipt.get("arm") == arm_name,
        "architecture": receipt.get("architecture") == arm["architecture"],
        "parameters": receipt.get("parameters") == arm["expected_parameters"],
        "config": receipt.get("config") == config,
        "source": isinstance(source, dict),
        "environment": receipt.get("environment") == environment,
        "runtime_policy": receipt.get("runtime_policy") == config["runtime_policy"],
        "history_checkpoint_alignment": history_steps == checkpoint_steps,
        "contiguous_evaluation_steps": history_steps == expected_steps,
        "latest_step_range": 63_000 <= latest <= 117_000,
        "latest_step_alignment": latest % 3_000 == 0,
        "selected_checkpoint": receipt.get("selected_checkpoint") is None,
        "final_gate_unset": receipt.get("final_gate") is None,
        "no_finished_at": "finished_at" not in receipt,
        "no_steps_completed": "steps_completed" not in receipt,
        "original_started_at": isinstance(receipt.get("started_at"), str)
        and bool(receipt["started_at"]),
        "latest_elapsed": isinstance(history[-1].get("elapsed_s"), (int, float))
        and not isinstance(history[-1].get("elapsed_s"), bool)
        and math.isfinite(float(history[-1]["elapsed_s"])),
        "no_stop_marker": not any(
            (output / marker).exists()
            for marker in ("DONE", "FAILED_GATE", "SELECTED.json")
        ),
        "initial_parent_step": receipt.get("resume", {})
        .get("parent_checkpoint", {})
        .get("step")
        == parent_end,
        "initial_boundary_exact": receipt.get("resume", {})
        .get("restoration", {})
        .get("boundary_screen_exact")
        is True,
        "prior_recoveries": "interrupted_recoveries" not in receipt
        or isinstance(receipt.get("interrupted_recoveries"), list),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("invalid interrupted receipt: " + ", ".join(failed))
    assert isinstance(history, list)
    assert isinstance(checkpoints, list)
    assert isinstance(source, dict)
    _original_source_is_available(source)

    for entry in checkpoints:
        checkpoint_path = _safe_checkpoint_path(output, entry.get("path"))
        expected_name = f"weights_step{entry['step']}.pt"
        if checkpoint_path.name != expected_name:
            raise ValueError("interrupted checkpoint step/path mismatch")
        if not checkpoint_path.is_file():
            raise ValueError(
                f"interrupted checkpoint is missing: {checkpoint_path.name}"
            )
        if not TRAINER.is_sha256(entry.get("sha256")):
            raise ValueError("interrupted checkpoint has invalid recorded SHA-256")
        if TRAINER.sha256_file(checkpoint_path) != entry["sha256"]:
            raise ValueError(
                f"interrupted checkpoint SHA-256 mismatch: {checkpoint_path.name}"
            )

    latest_entry = checkpoints[-1]
    latest_path = output / latest_entry["path"]
    payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    validate_interrupted_checkpoint(
        payload,
        latest_entry,
        receipt,
        config,
        arm_name,
        arm,
        environment,
    )
    current_sha = TRAINER.sha256_file(receipt_path)
    if current_sha != receipt_sha:
        raise RuntimeError("interrupted receipt changed during validation")
    receipt["_validated_uncommitted_artifacts"] = find_uncommitted_artifacts(
        output,
        {entry["path"] for entry in checkpoints},
        latest,
    )
    return receipt, latest_path, payload, receipt_sha


def validate_interrupted_checkpoint(
    payload: dict[str, Any],
    entry: dict[str, Any],
    receipt: dict[str, Any],
    config: dict[str, Any],
    arm_name: str,
    arm: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    step = entry["step"]
    state_dict = payload.get("state_dict")
    expected_model = TRAINER.Cell(**arm["architecture"])
    actual_signature = (
        TRAINER.state_dict_signature(state_dict)
        if isinstance(state_dict, dict)
        and all(isinstance(value, torch.Tensor) for value in state_dict.values())
        else None
    )
    expected_signature = TRAINER.state_dict_signature(expected_model.state_dict())
    optimizer_state = payload.get("optimizer_state_dict")
    scheduler_state = payload.get("scheduler_state_dict")
    saved_rng = payload.get("rng_state")
    checks = {
        "schema": payload.get("schema") == TRAINER.CHECKPOINT_SCHEMA,
        "step": type(payload.get("step")) is int and payload["step"] == step,
        "arm": payload.get("arm") == arm_name,
        "width": payload.get("L") == config["width"],
        "architecture": payload.get("config") == arm["architecture"],
        "experiment_config": payload.get("experiment_config") == config,
        "source_identity": payload.get("source_identity") == receipt["source_identity"],
        "state_signature": actual_signature == expected_signature,
        "state_finite": isinstance(state_dict, dict)
        and all(torch.isfinite(value).all().item() for value in state_dict.values()),
        "optimizer_state": isinstance(optimizer_state, dict),
        "scheduler_state": isinstance(scheduler_state, dict),
        "rng_state": isinstance(saved_rng, dict),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("invalid interrupted checkpoint: " + ", ".join(failed))
    assert isinstance(optimizer_state, dict)
    assert isinstance(scheduler_state, dict)
    assert isinstance(saved_rng, dict)

    parameters = list(expected_model.parameters())
    groups = optimizer_state.get("param_groups")
    slots = optimizer_state.get("state")
    if not isinstance(groups, list) or len(groups) != 1 or not isinstance(slots, dict):
        raise ValueError("invalid interrupted optimizer structure")
    group = groups[0]
    expected_ids = list(range(len(parameters)))
    optimizer_checks = {
        "parameter_ids": group.get("params") == expected_ids,
        "slot_ids": set(slots) == set(expected_ids),
        "learning_rate": math.isclose(
            group.get("lr", math.nan),
            config["schedule_extension"]["floor_learning_rate"],
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "initial_learning_rate": group.get("initial_lr") == config["peak_lr"],
        "weight_decay": group.get("weight_decay") == config["weight_decay"],
        "betas": tuple(group.get("betas", ())) == (0.9, 0.999),
        "eps": group.get("eps") == 1e-8,
        "amsgrad": group.get("amsgrad") is False,
        "maximize": group.get("maximize") is False,
    }
    for parameter_id, parameter in enumerate(parameters):
        slot = slots.get(parameter_id)
        valid = isinstance(slot, dict) and set(slot) == {
            "step",
            "exp_avg",
            "exp_avg_sq",
        }
        valid = bool(valid and TRAINER._scalar_step(slot["step"]) == step)
        for name in ("exp_avg", "exp_avg_sq"):
            moment = slot.get(name) if isinstance(slot, dict) else None
            valid = bool(
                valid
                and isinstance(moment, torch.Tensor)
                and moment.shape == parameter.shape
                and moment.dtype == parameter.dtype
                and torch.isfinite(moment).all().item()
            )
        optimizer_checks[f"slot_{parameter_id}"] = valid
    failed = [name for name, passed in optimizer_checks.items() if not passed]
    if failed:
        raise ValueError("invalid interrupted optimizer: " + ", ".join(failed))

    floor = config["schedule_extension"]["floor_learning_rate"]
    scheduler_checks = {
        "base_lrs": scheduler_state.get("base_lrs") == [config["peak_lr"]],
        "last_epoch": scheduler_state.get("last_epoch") == step,
        "step_count": scheduler_state.get("_step_count") == step + 1,
        "last_lr": isinstance(scheduler_state.get("_last_lr"), list)
        and len(scheduler_state["_last_lr"]) == 1
        and math.isclose(
            scheduler_state["_last_lr"][0],
            floor,
            rel_tol=0.0,
            abs_tol=1e-15,
        ),
        "lambda_slots": scheduler_state.get("lr_lambdas") == [None],
    }
    failed = [name for name, passed in scheduler_checks.items() if not passed]
    if failed:
        raise ValueError("invalid interrupted scheduler: " + ", ".join(failed))

    probe = random.Random()
    try:
        probe.setstate(saved_rng.get("python_data"))
    except (TypeError, ValueError) as error:
        raise ValueError("invalid interrupted RNG: python_data") from error
    cpu_rng = saved_rng.get("torch_cpu")
    if not isinstance(cpu_rng, torch.Tensor) or cpu_rng.dtype != torch.uint8:
        raise ValueError("invalid interrupted RNG: torch_cpu")
    try:
        torch.Generator(device="cpu").set_state(cpu_rng)
    except RuntimeError as error:
        raise ValueError("invalid interrupted RNG: torch_cpu state") from error
    if environment["device"] == "cuda":
        cuda_rng = saved_rng.get("torch_cuda_all")
        if (
            not isinstance(cuda_rng, list)
            or len(cuda_rng) != environment["cuda_device_count"]
            or not all(
                isinstance(state, torch.Tensor) and state.dtype == torch.uint8
                for state in cuda_rng
            )
        ):
            raise ValueError("invalid interrupted RNG: torch_cuda_all")
        for index, state in enumerate(cuda_rng):
            try:
                torch.Generator(device=f"cuda:{index}").set_state(state)
            except RuntimeError as error:
                raise ValueError(
                    f"invalid interrupted RNG: torch_cuda_all[{index}] state"
                ) from error
    elif "torch_cuda_all" in saved_rng:
        raise ValueError("invalid interrupted RNG: unexpected CUDA state")


def parent_schedule_lambda(config: dict[str, Any]) -> Any:
    parent_config = dict(config)
    parent_config["steps"] = config["schedule_extension"]["parent_end_step"]
    return TRAINER._parent_cosine_lambda(parent_config)


def save_checkpoint_atomic(
    path: pathlib.Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_rng: random.Random,
    device: torch.device,
    config: dict[str, Any],
    arm_name: str,
    architecture: dict[str, Any],
    step: int,
    source_identity: dict[str, Any],
) -> str:
    temporary = path.with_name(f".{path.name}.tmp")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint: {path.name}")
    if temporary.exists():
        raise FileExistsError(f"stale checkpoint temporary exists: {temporary.name}")
    payload = {
        "schema": TRAINER.CHECKPOINT_SCHEMA,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": TRAINER.rng_state(data_rng, device),
        "config": architecture,
        "experiment_config": config,
        "arm": arm_name,
        "L": config["width"],
        "step": step,
        "source_identity": source_identity,
    }
    try:
        torch.save(payload, temporary)
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise
    return TRAINER.sha256_file(path)


def _bank_receipt(
    output: pathlib.Path,
    receipt_sha: str,
    step: int,
) -> pathlib.Path:
    source = output / "receipt.json"
    destination = output / f"receipt.pre-recovery-step{step}.{receipt_sha}.json"
    if destination.exists():
        if TRAINER.sha256_file(destination) != receipt_sha:
            raise FileExistsError(
                "pre-recovery receipt backup path has different bytes"
            )
    else:
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        shutil.copyfile(source, temporary)
        os.replace(temporary, destination)
    if TRAINER.sha256_file(destination) != receipt_sha:
        raise RuntimeError("pre-recovery receipt backup verification failed")
    return destination


def write_text_atomic(path: pathlib.Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale text temporary exists: {temporary.name}")
    temporary.write_text(value)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    output = args.out.resolve()
    config = json.loads(config_path.read_text())
    arm = TRAINER.validate_config(config, args.arm)
    if config.get("schema") != TRAINER.HORIZON_CONFIG_SCHEMA:
        raise ValueError("interrupted recovery requires a v0.3 horizon config")
    TRAINER.configure_canonical_runtime(config, args.device)
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    device = TRAINER.resolve_device(args.device)
    environment = TRAINER.environment_identity(device)
    executor = recovery_source_identity(config_path)
    receipt, checkpoint_path, payload, input_receipt_sha = validate_interrupted_receipt(
        output,
        config,
        args.arm,
        environment,
        args.expected_receipt_sha256,
    )

    seeds = receipt["seeds"]
    model = TRAINER.Cell(**arm["architecture"])
    model.load_state_dict(payload["state_dict"], strict=True)
    model = model.to(device)
    if TRAINER.parameter_count(model) != arm["expected_parameters"]:
        raise RuntimeError("restored interrupted model parameter count changed")
    if not TRAINER.all_parameters_finite(model):
        raise ValueError("restored interrupted model is nonfinite")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["peak_lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, parent_schedule_lambda(config)
    )
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    scheduler.load_state_dict(payload["scheduler_state_dict"])
    if any(
        isinstance(optimizer.state[parameter].get(moment_name), torch.Tensor)
        and optimizer.state[parameter][moment_name].device != parameter.device
        for parameter in model.parameters()
        for moment_name in ("exp_avg", "exp_avg_sq")
    ):
        raise ValueError("restored interrupted optimizer state is on the wrong device")
    data_rng = random.Random()
    TRAINER.restore_training_rng(payload["rng_state"], data_rng, device)
    floor = config["schedule_extension"]["floor_learning_rate"]
    if not math.isclose(
        parent_schedule_lambda(config)(payload["step"]),
        config["schedule_extension"]["floor_fraction"],
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError("reconstructed interrupted schedule is not at its floor")
    if any(
        not math.isclose(group["lr"], floor, rel_tol=0.0, abs_tol=1e-15)
        for group in optimizer.param_groups
    ):
        raise ValueError("restored interrupted optimizer is not at the floor LR")

    selection = TRAINER.selection_spec(config)
    expected_boundary = receipt["history"][-1]["tiers"]
    model.eval()
    observed_boundary = TRAINER.evaluate_rollouts(
        model,
        config,
        seeds,
        device,
        selection["screen_n"],
    )
    if observed_boundary != expected_boundary:
        raise RuntimeError(
            "interrupted recovery boundary differs from latest receipt row"
        )
    boundary_sha = TRAINER.sha256_bytes(
        TRAINER.canonical_json(observed_boundary).encode()
    )
    TRAINER.restore_training_rng(payload["rng_state"], data_rng, device)
    if TRAINER.sha256_file(output / "receipt.json") != input_receipt_sha:
        raise RuntimeError("interrupted receipt changed during boundary replay")

    latest_step = payload["step"]
    backup_path = _bank_receipt(output, input_receipt_sha, latest_step)
    recovered_at = datetime.now(timezone.utc).isoformat()
    recovery_id = recovered_at.replace(":", "-").replace("+", "_")
    uncommitted_artifacts = receipt.pop("_validated_uncommitted_artifacts")
    quarantined_artifacts = quarantine_uncommitted_artifacts(
        output,
        uncommitted_artifacts,
        recovery_id,
    )
    recovery_event = {
        "schema": RECOVERY_SCHEMA,
        "classification": "provider_forced_exit_operational_interruption",
        "status": "running",
        "from_step": latest_step,
        "target_step": config["steps"],
        "original_started_at": receipt["started_at"],
        "recovered_at": recovered_at,
        "input_receipt": {
            "path": "receipt.json",
            "sha256": input_receipt_sha,
            "backup_path": backup_path.name,
        },
        "checkpoint": {
            "path": checkpoint_path.name,
            "sha256": TRAINER.sha256_file(checkpoint_path),
            "step": latest_step,
        },
        "boundary_screen": {
            "count": selection["screen_n"],
            "tiers_sha256": boundary_sha,
            "matches_latest_receipt_row": True,
        },
        "uncommitted_artifacts": quarantined_artifacts,
        "restoration": {
            "model_state": True,
            "optimizer_state": True,
            "scheduler_state": True,
            "python_data_rng": True,
            "torch_cpu_rng": True,
            "torch_cuda_rng": device.type == "cuda",
            "environment_exact": True,
            "floor_learning_rate": floor,
        },
        "executor": executor,
        "environment_sha256": TRAINER.environment_sha256(environment),
    }
    events = receipt.setdefault("interrupted_recoveries", [])
    if not isinstance(events, list):
        raise ValueError("interrupted_recoveries must be a list")
    events.append(recovery_event)
    receipt["status"] = "running"
    TRAINER.write_json(output / "receipt.json", receipt)

    started = time.perf_counter()
    elapsed_offset = float(receipt["history"][-1]["elapsed_s"])
    loss_function = nn.BCEWithLogitsLoss()
    history = receipt["history"]
    checkpoints = receipt["checkpoints"]
    existing_steps = {row["step"] for row in history}
    checkpoint_steps = {row["step"] for row in checkpoints}
    screen_thresholds = TRAINER.selection_minimum_correct_by_tier(config, "screen")
    selected_checkpoint = None
    last_loss = math.nan
    for step in range(latest_step + 1, config["steps"] + 1):
        model.train()
        features, digits, targets = TRAINER.train_batch(
            config["batch_size"], data_rng, config["width"], device
        )
        loss = loss_function(model(features, digits), targets)
        if not torch.isfinite(loss).item():
            raise FloatingPointError(f"nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
        optimizer.step()
        scheduler.step()
        last_loss = float(loss.detach().cpu())
        if not math.isclose(
            optimizer.param_groups[0]["lr"], floor, rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(f"learning rate left its floor at step {step}")
        if step % config["eval_every"] != 0 and step != config["steps"]:
            continue
        if step in existing_steps or step in checkpoint_steps:
            raise RuntimeError(f"refusing duplicate receipt step {step}")

        TRAINER.synchronize(device)
        model.eval()
        parameters_finite = TRAINER.all_parameters_finite(model)
        tiers = TRAINER.evaluate_rollouts(
            model, config, seeds, device, selection["screen_n"]
        )
        screen_gate = TRAINER.rollout_gate(
            tiers,
            config["tiers"],
            config["evaluation_width_modes"],
            selection["screen_n"],
            screen_thresholds,
        )
        screen_gate["parameters_finite"] = parameters_finite
        screen_gate["passed"] = screen_gate["passed"] and parameters_finite
        confirmation = None
        small_prime = None
        if screen_gate["passed"]:
            confirmation_tiers = TRAINER.evaluate_rollouts(
                model, config, seeds, device, selection["confirmation_n"]
            )
            small_prime = TRAINER.evaluate_small_primes(model, config, device)
            confirmation_gate = TRAINER.confirmed_gate(
                confirmation_tiers,
                small_prime,
                config,
                parameters_finite,
                selection["confirmation_n"],
            )
            confirmation = {
                "tiers": confirmation_tiers,
                "small_prime_exhaustive": small_prime,
                "gate": confirmation_gate,
            }
        TRAINER.synchronize(device)
        row = {
            "step": step,
            "loss": last_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(elapsed_offset + time.perf_counter() - started, 3),
            "tiers": tiers,
            "screen_gate": screen_gate,
            "confirmation": confirmation,
            "small_prime_exhaustive": small_prime,
            "parameters_finite": parameters_finite,
        }
        checkpoint = output / f"weights_step{step}.pt"
        checkpoint_sha = save_checkpoint_atomic(
            checkpoint,
            model,
            optimizer,
            scheduler,
            data_rng,
            device,
            config,
            args.arm,
            arm["architecture"],
            step,
            receipt["source_identity"],
        )
        checkpoint_entry = {
            "step": step,
            "path": checkpoint.name,
            "sha256": checkpoint_sha,
        }
        history.append(row)
        checkpoints.append(checkpoint_entry)
        existing_steps.add(step)
        checkpoint_steps.add(step)
        selected_checkpoint = TRAINER.freeze_first_confirmed_checkpoint(
            selected_checkpoint, checkpoint_entry, confirmation
        )
        receipt["selected_checkpoint"] = selected_checkpoint
        TRAINER.write_json(output / "receipt.json", receipt)
        print(
            f"recovered_step={step} loss={last_loss:.6f} elapsed_s={row['elapsed_s']}",
            flush=True,
        )
        if selected_checkpoint is not None:
            break

    final = history[-1]
    if selected_checkpoint is not None:
        final_gate = selected_checkpoint["confirmation_gate"]
    else:
        final_gate = {
            "parameters_finite": final["parameters_finite"],
            "rollout_threshold_passed": False,
            "rollout_exact": False,
            "small_prime_exact": False,
            "passed": False,
            "reason": "no checkpoint passed the confirmation gate",
        }
    receipt["status"] = (
        "completed_pass" if final_gate["passed"] else "completed_failed_gate"
    )
    receipt["finished_at"] = datetime.now(timezone.utc).isoformat()
    receipt["elapsed_s"] = round(elapsed_offset + time.perf_counter() - started, 3)
    receipt["steps_completed"] = final["step"]
    receipt["stopped_at_first_confirmed_pass"] = selected_checkpoint is not None
    receipt["selected_checkpoint"] = selected_checkpoint
    receipt["final_gate"] = final_gate
    recovery_event["status"] = "completed"
    recovery_event["completed_at"] = receipt["finished_at"]
    recovery_event["completed_step"] = final["step"]
    recovery_event["terminal_status"] = receipt["status"]
    TRAINER.write_json(output / "receipt.json", receipt)
    if selected_checkpoint is not None:
        TRAINER.write_json(output / "SELECTED.json", selected_checkpoint)
    marker = "DONE" if final_gate["passed"] else "FAILED_GATE"
    write_text_atomic(output / marker, receipt["status"] + "\n")
    print(
        "=== recovered run completed === " + TRAINER.canonical_json(final_gate),
        flush=True,
    )
    return 0 if final_gate["passed"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--expected-receipt-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), required=True)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
