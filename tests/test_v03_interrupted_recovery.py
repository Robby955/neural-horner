from __future__ import annotations

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
                "elapsed_s": 10.0,
                "tiers": tiers,
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
        (lambda receipt: receipt.update(status="completed_failed_gate"), "status"),
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
            "final_gate_unset",
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
