from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_PATH = ROOT / "research/v03/recover_interrupted.py"
SPEC = importlib.util.spec_from_file_location(
    "neuralhorner_v03_interrupted_recovery", RECOVERY_PATH
)
assert SPEC is not None and SPEC.loader is not None
RECOVERY = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RECOVERY
SPEC.loader.exec_module(RECOVERY)
TRAINER = RECOVERY.TRAINER


def _fixture(
    tmp_path: Path,
) -> tuple[Path, dict[str, Any], dict[str, Any], str]:
    config = json.loads(
        (ROOT / "research/v03/configs/horizon_l2048_b127_60k_120k.json").read_text()
    )
    architecture = {
        "dmodel": 3,
        "hidden": 1,
        "num_layers": 2,
        "bidirectional": True,
    }
    model = TRAINER.Cell(**architecture)
    parameters = TRAINER.parameter_count(model)
    config["arms"]["B127"]["architecture"] = architecture
    config["arms"]["B127"]["expected_parameters"] = parameters
    config["initialization"]["parent"]["architecture"] = architecture
    config["initialization"]["parent"]["parameters"] = parameters
    environment = {
        "device": "cpu",
        "torch": torch.__version__,
        "python": sys.version,
        "platform": "test-platform",
        "machine": "test-machine",
        "cuda_available": False,
        "mps_available": False,
        "deterministic_algorithms_enabled": True,
        "runtime_settings": config["runtime_policy"] | {"device": "cpu"},
    }
    source = {
        "trainer_path": "research/v02/train_scale.py",
        "trainer_sha256": "1" * 64,
        "config_path": "research/v03/configs/horizon_l2048_b127_60k_120k.json",
        "config_sha256": TRAINER.sha256_file(
            ROOT / "research/v03/configs/horizon_l2048_b127_60k_120k.json"
        ),
        "source_provenance_path": "research/v02/source_provenance.json",
        "source_provenance_sha256": "3" * 64,
        "git": {"head": "4" * 40, "branch": "test", "status": []},
    }
    step = 63_000
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["peak_lr"], weight_decay=config["weight_decay"]
    )
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(step)),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    for group in optimizer.param_groups:
        group["lr"] = config["schedule_extension"]["floor_learning_rate"]
        group["initial_lr"] = config["peak_lr"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, RECOVERY.parent_schedule_lambda(config)
    )
    for group in optimizer.param_groups:
        group["lr"] = config["schedule_extension"]["floor_learning_rate"]
    scheduler.last_epoch = step
    scheduler._step_count = step + 1
    scheduler._last_lr = [config["schedule_extension"]["floor_learning_rate"]]
    data_rng = random.Random(7182)
    torch.manual_seed(991)
    payload = {
        "schema": TRAINER.CHECKPOINT_SCHEMA,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": TRAINER.rng_state(data_rng, torch.device("cpu")),
        "config": architecture,
        "experiment_config": config,
        "arm": "B127",
        "L": config["width"],
        "step": step,
        "source_identity": source,
    }
    output = tmp_path / "run" / "B127"
    output.mkdir(parents=True)
    checkpoint = output / f"weights_step{step}.pt"
    torch.save(payload, checkpoint)
    checkpoint_sha = TRAINER.sha256_file(checkpoint)
    tiers = {
        str(tier): {
            mode: {"correct": 64, "total": 64}
            for mode in config["evaluation_width_modes"]
        }
        for tier in config["tiers"]
    }
    screen_gate = TRAINER.rollout_gate(
        tiers,
        config["tiers"],
        config["evaluation_width_modes"],
        config["selection"]["screen_n"],
        TRAINER.selection_minimum_correct_by_tier(config, "screen"),
    )
    screen_gate["parameters_finite"] = True
    screen_gate["passed"] = screen_gate["passed"] and True
    receipt = {
        "schema": TRAINER.HORIZON_RECEIPT_SCHEMA,
        "status": "running",
        "experiment": config["name"],
        "role": config["role"],
        "arm": "B127",
        "architecture": architecture,
        "parameters": parameters,
        "config": config,
        "seeds": {
            "master": 23,
            "initialization": 1,
            "training_data": 2,
            "evaluation": 3,
        },
        "source_identity": source,
        "runtime_policy": config["runtime_policy"],
        "environment": environment,
        "started_at": "2026-08-17T00:00:00+00:00",
        "resume": {
            "parent_checkpoint": {"step": 60_000},
            "restoration": {"boundary_screen_exact": True},
        },
        "history": [
            {
                "step": step,
                "loss": 0.125,
                "learning_rate": config["schedule_extension"][
                    "floor_learning_rate"
                ],
                "elapsed_s": 10.0,
                "tiers": tiers,
                "screen_gate": screen_gate,
                "confirmation": None,
                "small_prime_exhaustive": None,
                "parameters_finite": True,
            }
        ],
        "checkpoints": [
            {"step": step, "path": checkpoint.name, "sha256": checkpoint_sha}
        ],
        "selection": config["selection"],
        "selected_checkpoint": None,
        "final_gate": None,
    }
    receipt_path = output / "receipt.json"
    TRAINER.write_json(receipt_path, receipt)
    return output, config, environment, TRAINER.sha256_file(receipt_path)


def _validate(
    monkeypatch: pytest.MonkeyPatch,
    output: Path,
    config: dict[str, Any],
    environment: dict[str, Any],
    receipt_sha: str,
) -> tuple[dict[str, Any], Path, dict[str, Any], str]:
    monkeypatch.setattr(RECOVERY, "_original_source_is_available", lambda source: None)
    return RECOVERY.validate_interrupted_receipt(
        output, config, "B127", environment, receipt_sha
    )


def _add_passing_confirmation(receipt: dict[str, Any]) -> dict[str, Any]:
    config = receipt["config"]
    row = receipt["history"][-1]
    screen_n = config["selection"]["screen_n"]
    confirmation_n = config["selection"]["confirmation_n"]

    def tiers(count: int) -> dict[str, Any]:
        return {
            str(tier): {
                mode: {
                    "correct": count,
                    "total": count,
                    "case_manifest_sha256": "a" * 64,
                    "prime_sha256": "b" * 64,
                }
                for mode in config["evaluation_width_modes"]
            }
            for tier in config["tiers"]
        }

    row["tiers"] = tiers(screen_n)
    row["screen_gate"] = TRAINER.rollout_gate(
        row["tiers"],
        config["tiers"],
        config["evaluation_width_modes"],
        screen_n,
        TRAINER.selection_minimum_correct_by_tier(config, "screen"),
    )
    row["screen_gate"]["parameters_finite"] = True
    row["screen_gate"]["passed"] = row["screen_gate"]["passed"] and True
    confirmation_tiers = tiers(confirmation_n)
    expected_total = sum(
        2 * prime * prime
        for prime in range(2, config["small_prime_limit"])
        if TRAINER.is_prime(prime)
    )
    small_prime = {
        "fixed": {
            "prime_limit_exclusive": config["small_prime_limit"],
            "sequence_width": config["width"],
            "total": expected_total,
            "correct": expected_total,
        },
        "dynamic": {
            "prime_limit_exclusive": config["small_prime_limit"],
            "sequence_width": min(config["width"], 32),
            "total": expected_total,
            "correct": expected_total,
        },
    }
    gate = TRAINER.confirmed_gate(
        confirmation_tiers,
        small_prime,
        config,
        True,
        confirmation_n,
    )
    row["confirmation"] = {
        "tiers": confirmation_tiers,
        "small_prime_exhaustive": small_prime,
        "gate": gate,
    }
    row["small_prime_exhaustive"] = small_prime
    selected = {
        **receipt["checkpoints"][-1],
        "reason": "first_confirmed_pass",
        "confirmation_gate": gate,
    }
    receipt["selected_checkpoint"] = selected
    return selected


def _extend_fixture_to_horizon(output: Path, receipt: dict[str, Any]) -> None:
    original_row = receipt["history"][0]
    for step in range(66_000, 120_000, 3_000):
        row = deepcopy(original_row)
        row["step"] = step
        row["elapsed_s"] = float(step - 60_000)
        receipt["history"].append(row)
        path = output / f"weights_step{step}.pt"
        path.write_bytes(f"committed-{step}".encode())
        receipt["checkpoints"].append(
            {
                "step": step,
                "path": path.name,
                "sha256": TRAINER.sha256_file(path),
            }
        )

    latest_path = output / "weights_step63000.pt"
    payload = torch.load(latest_path, map_location="cpu", weights_only=False)
    payload["step"] = 120_000
    for slot in payload["optimizer_state_dict"]["state"].values():
        slot["step"] = torch.tensor(120_000.0)
    payload["scheduler_state_dict"]["last_epoch"] = 120_000
    payload["scheduler_state_dict"]["_step_count"] = 120_001
    terminal_path = output / "weights_step120000.pt"
    torch.save(payload, terminal_path)
    row = deepcopy(original_row)
    row["step"] = 120_000
    row["elapsed_s"] = 60_000.0
    receipt["history"].append(row)
    receipt["checkpoints"].append(
        {
            "step": 120_000,
            "path": terminal_path.name,
            "sha256": TRAINER.sha256_file(terminal_path),
        }
    )


def _patch_run_environment(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, Any],
    boundary_tiers: dict[str, Any],
) -> None:
    monkeypatch.setattr(RECOVERY.TRAINER, "configure_canonical_runtime", lambda *_: None)
    monkeypatch.setattr(
        RECOVERY.TRAINER, "resolve_device", lambda _: torch.device("cpu")
    )
    monkeypatch.setattr(
        RECOVERY.TRAINER, "environment_identity", lambda _: environment
    )
    monkeypatch.setattr(
        RECOVERY,
        "recovery_source_identity",
        lambda _: {
            "git": {"head": "f" * 40, "branch": "test", "status": []},
            "runner_sha256": "a" * 64,
        },
    )
    monkeypatch.setattr(RECOVERY, "_original_source_is_available", lambda _: None)
    monkeypatch.setattr(
        RECOVERY.TRAINER,
        "evaluate_rollouts",
        lambda *_: deepcopy(boundary_tiers),
    )
    monkeypatch.setattr(
        RECOVERY.TRAINER,
        "train_batch",
        lambda *_: pytest.fail("terminal recovery retrained the model"),
    )


def test_accepts_latest_fully_recorded_boundary_and_ignores_unlisted_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, receipt_sha = _fixture(tmp_path)
    (output / "weights_step66000.pt").write_bytes(b"provider-exit-partial")

    receipt, checkpoint, payload, validated_sha = _validate(
        monkeypatch, output, config, environment, receipt_sha
    )

    assert receipt["history"][-1]["step"] == 63_000
    assert checkpoint.name == "weights_step63000.pt"
    assert payload["step"] == 63_000
    assert validated_sha == receipt_sha
    assert receipt["_validated_uncommitted_artifacts"] == [
        {
            "kind": "unreferenced_checkpoint",
            "path": "weights_step66000.pt",
            "sha256": TRAINER.sha256_file(output / "weights_step66000.pt"),
            "size": len(b"provider-exit-partial"),
            "step": 66_000,
        }
    ]


def test_quarantines_uncommitted_artifact_without_accepting_it(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    partial = output / "weights_step66000.pt"
    partial.write_bytes(b"partial")
    artifact = {
        "kind": "unreferenced_checkpoint",
        "path": partial.name,
        "sha256": TRAINER.sha256_file(partial),
        "size": partial.stat().st_size,
        "step": 66_000,
    }

    moved = RECOVERY.quarantine_uncommitted_artifacts(
        output, [artifact], "recovery-test"
    )

    assert not partial.exists()
    destination = output / moved[0]["quarantine_path"]
    assert destination.read_bytes() == b"partial"
    assert moved[0]["sha256"] == TRAINER.sha256_file(destination)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda receipt: receipt.update(status="completed_failed_gate"),
            "completed-failure",
        ),
        (
            lambda receipt: receipt["checkpoints"][0].update(step=66_000),
            "alignment",
        ),
        (
            lambda receipt: receipt["history"].append(deepcopy(receipt["history"][0])),
            "strictly ordered",
        ),
        (
            lambda receipt: receipt.update(final_gate={"passed": False}),
            "final gate",
        ),
    ],
)
def test_rejects_non_interrupted_or_misaligned_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: Any,
    message: str,
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    mutation(receipt)
    TRAINER.write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match=message):
        _validate(
            monkeypatch,
            output,
            config,
            environment,
            TRAINER.sha256_file(receipt_path),
        )


def test_rejects_operator_sha_mismatch_without_writing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    before = (output / "receipt.json").read_bytes()

    with pytest.raises(ValueError, match="operator pin"):
        _validate(monkeypatch, output, config, environment, "f" * 64)

    assert (output / "receipt.json").read_bytes() == before
    assert not list(output.glob("receipt.pre-recovery-*"))


def test_rejects_checkpoint_hash_and_state_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, receipt_sha = _fixture(tmp_path)
    checkpoint = output / "weights_step63000.pt"
    with checkpoint.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        _validate(monkeypatch, output, config, environment, receipt_sha)


def test_rejects_scheduler_and_rng_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    checkpoint = output / "weights_step63000.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["scheduler_state_dict"]["last_epoch"] = 62_999
    payload["rng_state"]["torch_cpu"] = torch.zeros(1, dtype=torch.int64)
    torch.save(payload, checkpoint)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["checkpoints"][0]["sha256"] = TRAINER.sha256_file(checkpoint)
    TRAINER.write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="scheduler"):
        _validate(
            monkeypatch,
            output,
            config,
            environment,
            TRAINER.sha256_file(receipt_path),
        )


def test_parent_schedule_remains_at_floor_after_original_60k() -> None:
    config = json.loads(
        (ROOT / "research/v03/configs/horizon_l2048_b127_60k_120k.json").read_text()
    )
    schedule = RECOVERY.parent_schedule_lambda(config)
    assert schedule(60_000) == pytest.approx(0.03, abs=1e-15)
    assert schedule(63_000) == pytest.approx(0.03, abs=1e-15)
    assert schedule(120_000) == pytest.approx(0.03, abs=1e-15)


def test_git_binary_source_read_preserves_trailing_bytes() -> None:
    assert (
        RECOVERY._git("show", "HEAD:research/v02/train_scale.py", binary=True)
        == (ROOT / "research/v02/train_scale.py").read_bytes()
    )


def test_atomic_checkpoint_refuses_existing_target(tmp_path: Path) -> None:
    path = tmp_path / "weights_step66000.pt"
    path.write_bytes(b"preserve")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        RECOVERY.save_checkpoint_atomic(
            path,
            torch.nn.Linear(1, 1),
            torch.optim.AdamW(torch.nn.Linear(1, 1).parameters()),
            object(),
            random.Random(1),
            torch.device("cpu"),
            {"width": 1},
            "B127",
            {},
            66_000,
            {},
        )
    assert path.read_bytes() == b"preserve"


def test_accepts_pending_confirmed_checkpoint_for_terminal_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    selected = _add_passing_confirmation(receipt)
    TRAINER.write_json(receipt_path, receipt)

    validated, _, _, _ = _validate(
        monkeypatch,
        output,
        config,
        environment,
        TRAINER.sha256_file(receipt_path),
    )

    assert validated["_validated_phase"] == "pending_terminal_pass"
    assert validated["selected_checkpoint"] == selected


def test_run_finalizes_pending_confirmation_without_retraining_and_repairs_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    _add_passing_confirmation(receipt)
    TRAINER.write_json(receipt_path, receipt)
    _patch_run_environment(monkeypatch, environment, receipt["history"][-1]["tiers"])
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    args = argparse.Namespace(
        config=config_path,
        arm="B127",
        out=output,
        expected_receipt_sha256=TRAINER.sha256_file(receipt_path),
        device="cpu",
    )

    assert RECOVERY.run(args) == 0
    terminal = json.loads(receipt_path.read_text())
    assert terminal["status"] == "completed_pass"
    assert terminal["steps_completed"] == 63_000
    assert terminal["interrupted_recoveries"][-1]["status"] == "completed"
    assert (output / "SELECTED.json").is_file()
    assert (output / "DONE").read_text() == "completed_pass\n"

    (output / "DONE").unlink()
    terminal_bytes = receipt_path.read_bytes()
    args.expected_receipt_sha256 = TRAINER.sha256_file(receipt_path)
    assert RECOVERY.run(args) == 0
    assert receipt_path.read_bytes() == terminal_bytes
    assert (output / "DONE").read_text() == "completed_pass\n"


def test_rejects_confirmed_checkpoint_with_missing_or_tampered_selection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    _add_passing_confirmation(receipt)
    receipt["selected_checkpoint"] = None
    TRAINER.write_json(receipt_path, receipt)

    with pytest.raises(ValueError, match="selected checkpoint"):
        _validate(
            monkeypatch,
            output,
            config,
            environment,
            TRAINER.sha256_file(receipt_path),
        )

    selected = _add_passing_confirmation(receipt)
    selected["sha256"] = "f" * 64
    receipt["selected_checkpoint"] = selected
    TRAINER.write_json(receipt_path, receipt)
    with pytest.raises(ValueError, match="selected checkpoint"):
        _validate(
            monkeypatch,
            output,
            config,
            environment,
            TRAINER.sha256_file(receipt_path),
        )


def test_terminal_pass_artifacts_are_repairable_and_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    selected = _add_passing_confirmation(receipt)
    receipt.update(
        status="completed_pass",
        final_gate=selected["confirmation_gate"],
        finished_at="2026-08-17T01:00:00+00:00",
        steps_completed=63_000,
        stopped_at_first_confirmed_pass=True,
        elapsed_s=10.0,
    )
    TRAINER.write_json(receipt_path, receipt)

    validated, _, _, _ = _validate(
        monkeypatch,
        output,
        config,
        environment,
        TRAINER.sha256_file(receipt_path),
    )
    assert validated["_validated_phase"] == "terminal_pass"

    validated.pop("_validated_phase")
    validated.pop("_validated_uncommitted_artifacts")
    RECOVERY.ensure_terminal_artifacts(output, validated)
    first_selected = (output / "SELECTED.json").read_bytes()
    first_done = (output / "DONE").read_bytes()
    RECOVERY.ensure_terminal_artifacts(output, validated)
    assert (output / "SELECTED.json").read_bytes() == first_selected
    assert (output / "DONE").read_bytes() == first_done


def test_accepts_pending_failed_horizon_for_terminal_finalization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    _extend_fixture_to_horizon(output, receipt)
    TRAINER.write_json(receipt_path, receipt)

    validated, _, _, _ = _validate(
        monkeypatch,
        output,
        config,
        environment,
        TRAINER.sha256_file(receipt_path),
    )

    assert validated["_validated_phase"] == "pending_terminal_fail"
    validated.pop("_validated_phase")
    validated.pop("_validated_uncommitted_artifacts")
    event = {"status": "running"}
    RECOVERY.complete_receipt(validated, event, None)
    assert validated["status"] == "completed_failed_gate"
    assert validated["steps_completed"] == 120_000
    assert validated["selected_checkpoint"] is None
    assert validated["final_gate"]["passed"] is False
    assert event["terminal_status"] == "completed_failed_gate"
    RECOVERY.ensure_terminal_artifacts(output, validated)
    assert (output / "FAILED_GATE").read_text() == "completed_failed_gate\n"


def test_rejects_terminal_artifact_that_disagrees_with_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output, config, environment, _ = _fixture(tmp_path)
    receipt_path = output / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    selected = _add_passing_confirmation(receipt)
    receipt.update(
        status="completed_pass",
        final_gate=selected["confirmation_gate"],
        finished_at="2026-08-17T01:00:00+00:00",
        steps_completed=63_000,
        stopped_at_first_confirmed_pass=True,
        elapsed_s=10.0,
    )
    TRAINER.write_json(receipt_path, receipt)
    TRAINER.write_json(output / "SELECTED.json", {"step": 66_000})

    with pytest.raises(ValueError, match="SELECTED.json differs"):
        _validate(
            monkeypatch,
            output,
            config,
            environment,
            TRAINER.sha256_file(receipt_path),
        )


def test_finds_stale_terminal_temporaries_for_quarantine(tmp_path: Path) -> None:
    output = tmp_path / "run"
    output.mkdir()
    for name in ("DONE.tmp", "FAILED_GATE.tmp", "SELECTED.json.tmp"):
        (output / name).write_text("partial")

    artifacts = RECOVERY.find_uncommitted_artifacts(output, set(), 63_000)

    assert [artifact["path"] for artifact in artifacts] == [
        "DONE.tmp",
        "FAILED_GATE.tmp",
        "SELECTED.json.tmp",
    ]
    assert {artifact["kind"] for artifact in artifacts} == {
        "terminal_artifact_temporary"
    }
