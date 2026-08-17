from __future__ import annotations

import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "research/v02/train_scale.py"
SPEC = importlib.util.spec_from_file_location("neuralhorner_v03_training", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINER
SPEC.loader.exec_module(TRAINER)


def _rollout_rows(tiers: list[int], count: int) -> dict[str, Any]:
    return {
        str(tier): {
            mode: {
                "correct": count,
                "total": count,
                "accuracy": 1.0,
                "sequence_width": 2048 if mode == "fixed" else 128,
                "case_manifest_sha256": f"cases-{tier}",
                "prime_sha256": f"primes-{tier}",
                "first_failures": [],
            }
            for mode in ("fixed", "dynamic")
        }
        for tier in tiers
    }


def _small_prime_rows(width: int, prime_limit: int) -> dict[str, Any]:
    total = sum(
        2 * prime * prime
        for prime in range(2, prime_limit)
        if TRAINER.is_prime(prime)
    )
    return {
        mode: {
            "prime_limit_exclusive": prime_limit,
            "sequence_width": width if mode == "fixed" else min(width, 32),
            "correct": total,
            "total": total,
            "accuracy": 1.0,
            "first_failures": [],
        }
        for mode in ("fixed", "dynamic")
    }


def _make_resume_fixture(
    tmp_path: Path,
) -> tuple[dict[str, Any], Path, Path, Path, dict[str, Any]]:
    config = json.loads(
        (
            ROOT
            / "research/v03/configs/horizon_l2048_b127_60k_120k.json"
        ).read_text()
    )
    architecture = {
        "dmodel": 3,
        "hidden": 1,
        "num_layers": 2,
        "bidirectional": True,
    }
    model = TRAINER.Cell(**architecture)
    parameters = TRAINER.parameter_count(model)
    parent_step = 2
    child_step = 3
    peak_lr = 0.0015
    minimum_fraction = 0.03
    floor_lr = peak_lr * minimum_fraction
    arm = config["arms"]["B127"]
    arm["architecture"] = architecture
    arm["expected_parameters"] = parameters
    parent = config["initialization"]["parent"]
    parent.update(
        {
            "width": config["width"],
            "checkpoint_step": parent_step,
            "parameters": parameters,
            "architecture": architecture,
        }
    )
    config["steps"] = child_step
    config["batch_size"] = 1
    config["eval_every"] = 1
    config["checkpoint_eval_n"] = 2
    config["final_eval_n"] = 4
    config["eval_batch_size"] = 1
    config["selection"]["screen_n"] = 2
    config["selection"]["confirmation_n"] = 4
    config["selection"]["screen_minimum_correct_by_tier"] = {
        str(tier): 2 for tier in config["tiers"]
    }
    config["selection"]["confirmation_minimum_correct_by_tier"] = {
        str(tier): 4 for tier in config["tiers"]
    }
    config["schedule_extension"].update(
        {
            "parent_end_step": parent_step,
            "floor_fraction": minimum_fraction,
            "floor_learning_rate": floor_lr,
        }
    )

    source = {
        "trainer_path": "research/v02/train_scale.py",
        "trainer_sha256": "1" * 64,
        "config_path": "research/v02/configs/test-parent.json",
        "config_sha256": "2" * 64,
        "source_provenance_path": "research/v02/source_provenance.json",
        "source_provenance_sha256": "3" * 64,
        "git": {"head": "4" * 40, "branch": "test-parent", "status": []},
    }
    parent["source_identity"] = {
        "git_head": source["git"]["head"],
        "trainer_sha256": source["trainer_sha256"],
        "config_sha256": source["config_sha256"],
        "source_provenance_sha256": source["source_provenance_sha256"],
    }
    environment = {
        "device": "cpu",
        "torch": torch.__version__,
        "python": sys.version,
        "platform": "test-platform",
        "machine": "test-machine",
        "cuda_available": False,
        "mps_available": False,
        "deterministic_algorithms_enabled": True,
        "runtime_settings": {},
    }
    parent["environment_sha256"] = TRAINER.environment_sha256(environment)

    parent_config = json.loads(json.dumps(config))
    parent_config.update(
        {
            "schema": TRAINER.CONFIG_SCHEMA,
            "name": parent["experiment"],
            "role": parent["role"],
            "steps": parent_step,
        }
    )
    parent_config["selection"]["evaluate_step_zero"] = True
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=peak_lr, weight_decay=config["weight_decay"]
    )
    for parameter in model.parameters():
        optimizer.state[parameter] = {
            "step": torch.tensor(float(parent_step)),
            "exp_avg": torch.zeros_like(parameter),
            "exp_avg_sq": torch.zeros_like(parameter),
        }
    for group in optimizer.param_groups:
        group["lr"] = floor_lr
        group["initial_lr"] = peak_lr
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, TRAINER._parent_cosine_lambda(parent_config)
    )
    for group in optimizer.param_groups:
        group["lr"] = floor_lr
    scheduler.last_epoch = parent_step
    scheduler._step_count = parent_step + 1
    scheduler._last_lr = [floor_lr]

    data_rng = random.Random(9917)
    data_rng.random()
    torch.manual_seed(7713)
    torch.rand(3)
    saved_rng = TRAINER.rng_state(data_rng, torch.device("cpu"))
    checkpoint = {
        "schema": TRAINER.CHECKPOINT_SCHEMA,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": saved_rng,
        "config": architecture,
        "experiment_config": parent_config,
        "arm": "B127",
        "L": config["width"],
        "step": parent_step,
        "source_identity": source,
    }
    parent["training_state"] = {
        "optimizer_state_slots": len(list(model.parameters())),
        "optimizer_step": parent_step,
        "scheduler_last_epoch": parent_step,
        "scheduler_step_count": parent_step + 1,
        "learning_rate": floor_lr,
        "cuda_rng_state_count": 0,
        "next_batch_manifest_sha256": TRAINER.next_batch_manifest_sha256(checkpoint),
    }
    checkpoint_path = tmp_path / "weights_step2.pt"
    torch.save(checkpoint, checkpoint_path)
    parent["checkpoint_sha256"] = TRAINER.sha256_file(checkpoint_path)
    parent["state_dict_signature_sha256"] = TRAINER.state_dict_signature_sha256(
        model.state_dict()
    )

    boundary = _rollout_rows(config["tiers"], config["selection"]["screen_n"])
    receipt = {
        "schema": TRAINER.RECEIPT_SCHEMA,
        "status": parent["status"],
        "experiment": parent["experiment"],
        "role": parent["role"],
        "arm": "B127",
        "architecture": architecture,
        "parameters": parameters,
        "config": parent_config,
        "seeds": {
            "master": config["master_seed"],
            "initialization": TRAINER.derived_seed(
                config["master_seed"], "initialization"
            ),
            "training_data": TRAINER.derived_seed(
                config["master_seed"], "training_data"
            ),
            "evaluation": TRAINER.derived_seed(
                config["master_seed"], "evaluation"
            ),
        },
        "source_identity": source,
        "environment": environment,
        "selected_checkpoint": None,
        "stopped_at_first_confirmed_pass": False,
        "steps_completed": parent_step,
        "final_gate": {"passed": False},
        "history": [{"step": parent_step, "tiers": boundary}],
        "checkpoints": [
            {
                "step": parent_step,
                "path": checkpoint_path.name,
                "sha256": parent["checkpoint_sha256"],
            }
        ],
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    parent["receipt_sha256"] = TRAINER.sha256_file(receipt_path)
    config_path = tmp_path / "horizon.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    return config, config_path, receipt_path, checkpoint_path, environment


def _toy_batch(
    _batch_size: int,
    data_rng: random.Random,
    _width: int,
    _device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.rand((1, 1, 3))
    digits = torch.tensor([data_rng.randrange(2)], dtype=torch.long)
    targets = torch.tensor([[float(data_rng.randrange(2))]])
    return features, digits, targets


def _assert_nested_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, list):
        assert isinstance(right, list)
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_nested_equal(first, second)
    else:
        assert left == right


def test_frozen_horizon_config_is_an_absolute_floor_clamped_extension() -> None:
    config = json.loads(
        (
            ROOT
            / "research/v03/configs/horizon_l2048_b127_60k_120k.json"
        ).read_text()
    )
    arm = TRAINER.validate_config(config, "B127")
    assert config["steps"] == 120_000
    assert config["initialization"]["parent"]["checkpoint_step"] == 60_000
    assert config["schedule_extension"] == {
        "mode": "clamp_parent_cosine_at_floor",
        "parent_end_step": 60_000,
        "floor_fraction": 0.03,
        "floor_learning_rate": 0.000045,
    }
    assert arm["expected_parameters"] == 126_603


def test_parent_cosine_schedule_stays_at_floor_after_original_horizon() -> None:
    parent = {
        "steps": 60_000,
        "warmup_fraction": 0.025,
        "minimum_lr_fraction": 0.03,
    }
    schedule = TRAINER._parent_cosine_lambda(parent)
    assert math.isclose(schedule(60_000), 0.03, abs_tol=1e-15)
    assert math.isclose(schedule(60_001), 0.03, abs_tol=1e-15)
    assert math.isclose(schedule(120_000), 0.03, abs_tol=1e-15)


def test_exact_resume_rejects_environment_drift(tmp_path: Path) -> None:
    config, _config_path, receipt_path, checkpoint_path, environment = (
        _make_resume_fixture(tmp_path)
    )
    arm = TRAINER.validate_config(config, "B127")
    drifted = {**environment, "torch": "different"}
    with pytest.raises(ValueError, match="environment_continuity"):
        TRAINER.validate_exact_resume_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
            drifted,
        )


def test_exact_resume_rejects_missing_optimizer_slot(tmp_path: Path) -> None:
    config, config_path, receipt_path, checkpoint_path, environment = (
        _make_resume_fixture(tmp_path)
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    payload["optimizer_state_dict"]["state"].pop(0)
    torch.save(payload, checkpoint_path)
    checkpoint_sha = TRAINER.sha256_file(checkpoint_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["checkpoints"][-1]["sha256"] = checkpoint_sha
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    config["initialization"]["parent"]["checkpoint_sha256"] = checkpoint_sha
    config["initialization"]["parent"]["receipt_sha256"] = TRAINER.sha256_file(
        receipt_path
    )
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(ValueError, match="invalid exact-resume optimizer"):
        TRAINER.validate_exact_resume_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
            environment,
        )


def test_boundary_mismatch_aborts_before_output_checkpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, receipt_path, checkpoint_path, environment = (
        _make_resume_fixture(tmp_path)
    )
    monkeypatch.setattr(TRAINER, "configure_canonical_runtime", lambda *_args: None)
    monkeypatch.setattr(TRAINER, "resolve_device", lambda _requested: torch.device("cpu"))
    monkeypatch.setattr(TRAINER, "environment_identity", lambda _device: environment)
    monkeypatch.setattr(
        TRAINER,
        "git_identity",
        lambda: {"head": "a" * 40, "branch": "test", "status": []},
    )
    monkeypatch.setattr(TRAINER, "enforce_source_policy", lambda *_args: None)
    observed = _rollout_rows(config["tiers"], config["selection"]["screen_n"])
    observed["7"]["dynamic"]["correct"] -= 1
    monkeypatch.setattr(
        TRAINER,
        "evaluate_rollouts",
        lambda *_args, **_kwargs: observed,
    )
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="boundary screen differs"):
        TRAINER.run(
            SimpleNamespace(
                config=config_path,
                arm="B127",
                out=output,
                device="cuda",
                warm_start=None,
                parent_receipt=None,
                predecessor_receipt=None,
                resume_receipt=receipt_path,
                resume=checkpoint_path,
            )
        )
    assert not output.exists()


def test_exact_resume_matches_one_update_continuation_and_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, config_path, receipt_path, checkpoint_path, environment = (
        _make_resume_fixture(tmp_path)
    )
    parent_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    expected_model = TRAINER.Cell(**config["arms"]["B127"]["architecture"])
    expected_model.load_state_dict(parent_payload["state_dict"], strict=True)
    expected_optimizer = torch.optim.AdamW(
        expected_model.parameters(),
        lr=config["peak_lr"],
        weight_decay=config["weight_decay"],
    )
    expected_scheduler = torch.optim.lr_scheduler.LambdaLR(
        expected_optimizer,
        TRAINER._parent_cosine_lambda(parent_payload["experiment_config"]),
    )
    expected_optimizer.load_state_dict(parent_payload["optimizer_state_dict"])
    expected_scheduler.load_state_dict(parent_payload["scheduler_state_dict"])
    expected_rng = random.Random()
    TRAINER.restore_training_rng(
        parent_payload["rng_state"], expected_rng, torch.device("cpu")
    )
    expected_features, expected_digits, expected_targets = _toy_batch(
        1, expected_rng, config["width"], torch.device("cpu")
    )
    expected_loss = torch.nn.BCEWithLogitsLoss()(
        expected_model(expected_features, expected_digits), expected_targets
    )
    expected_optimizer.zero_grad(set_to_none=True)
    expected_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        expected_model.parameters(), config["gradient_clip"]
    )
    expected_optimizer.step()
    expected_scheduler.step()
    expected_rng_state = TRAINER.rng_state(expected_rng, torch.device("cpu"))

    monkeypatch.setattr(TRAINER, "configure_canonical_runtime", lambda *_args: None)
    monkeypatch.setattr(TRAINER, "resolve_device", lambda _requested: torch.device("cpu"))
    monkeypatch.setattr(TRAINER, "environment_identity", lambda _device: environment)
    monkeypatch.setattr(
        TRAINER,
        "git_identity",
        lambda: {"head": "a" * 40, "branch": "test", "status": []},
    )
    monkeypatch.setattr(TRAINER, "enforce_source_policy", lambda *_args: None)
    monkeypatch.setattr(TRAINER, "synchronize", lambda _device: None)
    monkeypatch.setattr(TRAINER, "train_batch", _toy_batch)
    monkeypatch.setattr(
        TRAINER,
        "evaluate_rollouts",
        lambda _model, child_config, _seeds, _device, count: _rollout_rows(
            child_config["tiers"], count
        ),
    )
    monkeypatch.setattr(
        TRAINER,
        "evaluate_small_primes",
        lambda _model, child_config, _device: _small_prime_rows(
            child_config["width"], child_config["small_prime_limit"]
        ),
    )
    output = tmp_path / "output"
    result = TRAINER.run(
        SimpleNamespace(
            config=config_path,
            arm="B127",
            out=output,
            device="cuda",
            warm_start=None,
            parent_receipt=None,
            predecessor_receipt=None,
            resume_receipt=receipt_path,
            resume=checkpoint_path,
        )
    )
    assert result == 0
    receipt = json.loads((output / "receipt.json").read_text())
    assert receipt["steps_completed"] == 3
    assert receipt["status"] == "completed_pass"
    assert receipt["history"][0]["step"] == 3
    assert receipt["resume"]["restoration"] == {
        "model_state": True,
        "optimizer_state": True,
        "scheduler_state": True,
        "python_data_rng": True,
        "torch_cpu_rng": True,
        "torch_cuda_rng": False,
        "environment_exact": True,
        "boundary_screen_exact": True,
    }
    assert receipt["resume"]["boundary_screen"]["step"] == 2

    child_payload = torch.load(
        output / "weights_step3.pt", map_location="cpu", weights_only=False
    )
    _assert_nested_equal(child_payload["state_dict"], expected_model.state_dict())
    _assert_nested_equal(
        child_payload["optimizer_state_dict"], expected_optimizer.state_dict()
    )
    assert child_payload["scheduler_state_dict"] == expected_scheduler.state_dict()
    assert child_payload["rng_state"]["python_data"] == expected_rng_state["python_data"]
    assert torch.equal(
        child_payload["rng_state"]["torch_cpu"], expected_rng_state["torch_cpu"]
    )
    assert child_payload["step"] == 3
    assert math.isclose(
        child_payload["scheduler_state_dict"]["_last_lr"][0],
        0.000045,
        rel_tol=0.0,
        abs_tol=1e-15,
    )


def test_exact_resume_requires_both_parent_artifacts(tmp_path: Path) -> None:
    config, config_path, receipt_path, _checkpoint_path, _environment = (
        _make_resume_fixture(tmp_path)
    )
    assert TRAINER.validate_config(config, "B127")
    with pytest.raises(ValueError, match="requires --resume-receipt and --resume"):
        TRAINER.run(
            SimpleNamespace(
                config=config_path,
                arm="B127",
                out=tmp_path / "out",
                device="cuda",
                warm_start=None,
                parent_receipt=None,
                predecessor_receipt=None,
                resume_receipt=receipt_path,
                resume=None,
            )
        )
