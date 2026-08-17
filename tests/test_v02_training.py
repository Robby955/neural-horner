from __future__ import annotations

import hashlib
import importlib.util
import json
import random
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAINER_PATH = ROOT / "research/v02/train_scale.py"
SPEC = importlib.util.spec_from_file_location("neuralhorner_v02_training", TRAINER_PATH)
assert SPEC is not None and SPEC.loader is not None
TRAINER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = TRAINER
SPEC.loader.exec_module(TRAINER)


@pytest.mark.parametrize(
    ("hidden", "expected"),
    [
        (128, 470_849),
        (90, 249_157),
        (61, 126_603),
        (40, 63_057),
        (26, 32_453),
    ],
)
def test_two_layer_bidirectional_parameter_ladder(hidden: int, expected: int) -> None:
    model = TRAINER.Cell(dmodel=96, hidden=hidden, num_layers=2, bidirectional=True)
    assert TRAINER.parameter_count(model) == expected
    assert 24 * hidden * hidden + 602 * hidden + 577 == expected


@pytest.mark.parametrize(
    ("hidden", "expected"),
    [(128, 174_401), (103, 125_001), (88, 98_961), (64, 62_913)],
)
def test_one_layer_bidirectional_control_counts(hidden: int, expected: int) -> None:
    model = TRAINER.Cell(dmodel=96, hidden=hidden, num_layers=1, bidirectional=True)
    assert TRAINER.parameter_count(model) == expected
    assert 6 * hidden * hidden + 590 * hidden + 577 == expected


def test_default_cell_loads_frozen_v8_state_dict_exactly() -> None:
    checkpoint = torch.load(ROOT / "model/weights.pt", map_location="cpu", weights_only=True)
    model = TRAINER.Cell()
    load_result = model.load_state_dict(checkpoint["state_dict"], strict=True)
    assert not load_result.missing_keys
    assert not load_result.unexpected_keys
    assert checkpoint["L"] == 2048
    assert TRAINER.parameter_count(model) == 470_849


def test_domain_separated_seeds_are_stable_and_distinct() -> None:
    first = TRAINER.derived_seed(23, "training_data")
    assert first == TRAINER.derived_seed(23, "training_data")
    assert first != TRAINER.derived_seed(23, "initialization")
    assert first != TRAINER.derived_seed(24, "training_data")


def test_sample_stream_is_architecture_independent() -> None:
    seed = TRAINER.derived_seed(23, "training_data")
    first = TRAINER.sample_transition_values(64, random.Random(seed), 128)
    second = TRAINER.sample_transition_values(64, random.Random(seed), 128)
    assert first == second


@pytest.mark.parametrize("width", [1, 7, 32, 33, 64, 128])
def test_limb_bit_conversion_round_trips(width: int) -> None:
    maximum = (1 << width) - 1
    values = [0, 1, maximum // 2, maximum]
    bits = TRAINER.to_bits_limbs(values, torch.device("cpu"), width)
    assert TRAINER.bits_to_integers(bits) == values


def test_exact_small_prime_case_count_matches_receipt_denominator() -> None:
    count = sum(
        2 * prime * prime
        for prime in range(2, 64)
        if TRAINER.is_prime(prime)
    )
    assert count == 40_954


def test_public_provenance_redacts_machine_local_locators() -> None:
    provenance_path = ROOT / "research/v02/source_provenance.json"
    provenance_text = provenance_path.read_text()
    assert "/Users/" not in provenance_text
    assert "AGENT_CORPUS" not in provenance_text
    assert "Robby955/MAC" not in provenance_text

    provenance = json.loads(provenance_text)
    assert provenance["public_redaction"]["archived_source_provenance_sha256"] == (
        "2d84692b49a3ebdc7550ff61606cb999dd27c2fbd7bcaa06f0e1da812be1b4f0"
    )
    assert provenance["historical_v8"]["transcript"]["sha256"] == (
        "c951501d0d2930083d8f7aed9dce8014bf20344d098579e729124cb3be84ac57"
    )

    playground = json.loads(
        (
            ROOT
            / "research/v02/evidence/"
            "sair_playground_d9d611833d340c72d90a97d995a94031b798cf7c.json"
        ).read_text()
    )
    assert playground["source"]["classification"] == "UI_TRANSCRIPTION"
    assert playground["source"]["transcribed_by"] == "repository owner"


def test_l2048_terminal_evidence_summary_is_internally_consistent() -> None:
    evidence = json.loads(
        (
            ROOT
            / "research/v02/evidence/l2048_scale_repair_b310c5e_terminal.json"
        ).read_text()
    )
    assert evidence["source"]["git_head"] == (
        "b310c5e9e7e0e096f613e5cf6bbfa7b2b247281b"
    )
    assert evidence["artifact"]["parameters"] == 126_603
    assert evidence["run"]["status"] == "completed_failed_gate"
    assert evidence["run"]["steps_completed"] == 60_000
    assert evidence["run"]["selected_checkpoint"] is None
    assert evidence["gate"]["confirmation_status"] == "not_run"
    assert evidence["gate"]["small_prime_status"] == "not_run"

    history = evidence["screen_history"]
    assert len(history) == 21
    assert [row["step"] for row in history] == list(range(0, 60_001, 3_000))
    assert all(row["screen_passed"] is False for row in history)
    assert history[0]["fixed"] == [64, 64, 64, 64, 63]
    assert history[0]["dynamic"] == [64, 64, 64, 64, 63]
    assert history[1]["fixed"] == [64, 64, 64, 64, 64]
    assert history[1]["dynamic"] == [64, 61, 63, 64, 64]
    assert history[-1]["fixed"] == [64, 64, 64, 64, 63]
    assert history[-1]["dynamic"] == [43, 55, 49, 54, 63]
    history_digest = hashlib.sha256(
        json.dumps(history, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert history_digest == (
        "6452b8152f5b44c68b621f2ad809ef147443649b4951d43e1358f34b452c6276"
    )

    assert set(evidence["evaluation_manifests"]) == {"6", "7", "8", "9", "10"}
    archive = evidence["archive"]
    assert archive["full_run_directory_tracked_in_git"] is False
    assert archive["receipt_sha256"] == (
        "77ce649080558c7c4154599461da33d3d482516bbd1c2ecc838161ea29592705"
    )
    assert archive["console_log_sha256"] == (
        "370fb9641e9b933b61a846048ff71883864de2c2b6ef6e48e44cc8cab0b91528"
    )
    assert archive["failed_gate_marker_sha256"] == (
        "57c1d2976daa3c3279b6ec1133023c504cd3d8e2d583588402fd680e05cc6366"
    )


def test_aligned_unidirectional_obstruction_witnesses() -> None:
    modulus = 5
    # MSB-first: 010 and 011 share their first input bit, but the first output
    # bits of 100 and 001 differ.
    msb_outputs = [(2 * state) % modulus for state in (2, 3)]
    assert [f"{state:03b}"[0] for state in (2, 3)] == ["0", "0"]
    assert [f"{value:03b}"[0] for value in msb_outputs] == ["1", "0"]

    # LSB-first processing: 001 and 011 share their first processed bit, but
    # outputs 010 and 001 have different first processed bits.
    lsb_outputs = [(2 * state) % modulus for state in (1, 3)]
    assert [state & 1 for state in (1, 3)] == [1, 1]
    assert [value & 1 for value in lsb_outputs] == [0, 1]


class ExactTransitionOracle(torch.nn.Module):
    def forward(self, features: torch.Tensor, digits: torch.Tensor) -> torch.Tensor:
        states = TRAINER.bits_to_integers(features[:, :, 0])
        multiplicands = TRAINER.bits_to_integers(features[:, :, 1])
        moduli = TRAINER.bits_to_integers(features[:, :, 2])
        targets = [
            (2 * state + digit * multiplicand) % modulus
            for state, multiplicand, modulus, digit in zip(
                states,
                multiplicands,
                moduli,
                digits.cpu().tolist(),
                strict=True,
            )
        ]
        bits = TRAINER.to_bits_limbs(
            targets,
            features.device,
            features.shape[1],
        ).float()
        return torch.where(bits > 0, 20.0, -20.0)


@pytest.mark.parametrize("tier", [4, 5, 6])
def test_rollout_width_modes_use_identical_cases_and_accept_oracle(tier: int) -> None:
    seed = TRAINER.derived_seed(5100268371224831233, f"tier_{tier}")
    results = {
        mode: TRAINER.eval_tier(
            ExactTransitionOracle(),
            tier,
            random.Random(seed),
            128,
            torch.device("cpu"),
            count=3,
            batch_size=3,
            width_mode=mode,
        )
        for mode in ("fixed", "dynamic")
    }
    assert results["fixed"]["correct"] == results["dynamic"]["correct"] == 3
    assert results["fixed"]["prime_sha256"] == results["dynamic"]["prime_sha256"]
    assert (
        results["fixed"]["case_manifest_sha256"]
        == results["dynamic"]["case_manifest_sha256"]
    )
    assert results["fixed"]["sequence_widths"] == [128]
    assert results["dynamic"]["sequence_widths"] == [{4: 32, 5: 64, 6: 128}[tier]]


def test_frozen_configs_validate_and_bind_expected_counts() -> None:
    for name in (
        "smoke_l32.json",
        "pilot_l128.json",
        "bridge_l256_b127.json",
        "scale_l512_b127.json",
        "scale_l1024_b127.json",
        "scale_l2048_b127.json",
    ):
        config = json.loads((ROOT / "research/v02/configs" / name).read_text())
        for arm_name, arm in config["arms"].items():
            validated = TRAINER.validate_config(config, arm_name)
            assert validated == arm
            model = TRAINER.Cell(**arm["architecture"])
            assert TRAINER.parameter_count(model) == arm["expected_parameters"]


def test_config_rejects_tier_wider_than_state() -> None:
    config = json.loads((ROOT / "research/v02/configs/smoke_l32.json").read_text())
    config["tiers"] = [5]
    with pytest.raises(ValueError, match="exceeds configured width"):
        TRAINER.validate_config(config, "B471")


def test_config_rejects_mislabeled_parameter_count() -> None:
    config = json.loads((ROOT / "research/v02/configs/pilot_l128.json").read_text())
    config["arms"]["B471"]["expected_parameters"] = 249_157
    with pytest.raises(ValueError, match="parameter mismatch"):
        TRAINER.validate_config(config, "B471")


def test_compression_arm_requires_passing_predecessor_receipt(tmp_path: Path) -> None:
    config_path = ROOT / "research/v02/configs/pilot_l128.json"
    config = json.loads(config_path.read_text())
    arm = TRAINER.validate_config(config, "B249")
    source_identity = {
        "config_sha256": TRAINER.sha256_file(config_path),
        "trainer_sha256": TRAINER.sha256_file(TRAINER_PATH),
    }
    seeds = {
        "master": 23,
        "initialization": 1,
        "training_data": 2,
        "evaluation": 3,
    }
    environment = {
        "torch": "test",
        "device": "mps",
        "platform": "test",
        "python": "test",
        "machine": "arm64",
    }
    with pytest.raises(ValueError, match="requires a passing B471"):
        TRAINER.validate_predecessor_receipt(
            None,
            "B249",
            arm,
            config,
            source_identity,
            seeds,
            environment,
        )

    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "schema": TRAINER.RECEIPT_SCHEMA,
                "status": "completed_pass",
                "arm": "B471",
                "experiment": config["name"],
                "final_gate": {"passed": True},
                "source_identity": source_identity,
                "warm_start": None,
                "seeds": seeds,
                "environment": environment,
            }
        )
    )
    identity = TRAINER.validate_predecessor_receipt(
        receipt_path,
        "B249",
        arm,
        config,
        source_identity,
        seeds,
        environment,
    )
    assert identity is not None
    assert identity["arm"] == "B471"


def _exact_rollout_rows(tiers: list[int], count: int) -> dict[str, object]:
    return {
        str(tier): {
            width_mode: {
                "correct": count,
                "total": count,
                "case_manifest_sha256": f"cases-tier-{tier}",
                "prime_sha256": f"primes-tier-{tier}",
            }
            for width_mode in ("fixed", "dynamic")
        }
        for tier in tiers
    }


def _exact_small_prime_rows(width: int, prime_limit: int) -> dict[str, object]:
    total = sum(
        2 * prime * prime
        for prime in range(2, prime_limit)
        if TRAINER.is_prime(prime)
    )
    return {
        "fixed": {
            "prime_limit_exclusive": prime_limit,
            "sequence_width": width,
            "correct": total,
            "total": total,
        },
        "dynamic": {
            "prime_limit_exclusive": prime_limit,
            "sequence_width": min(width, 32),
            "correct": total,
            "total": total,
        },
    }


def _write_bound_parent_artifacts(
    tmp_path: Path,
) -> tuple[dict[str, object], Path, Path]:
    config = json.loads(
        (ROOT / "research/v02/configs/bridge_l256_b127.json").read_text()
    )
    parent = config["initialization"]["parent"]
    architecture = parent["architecture"]
    required_gate = parent["required_gate"]
    source = {
        "trainer_path": "research/v02/train_scale.py",
        "trainer_sha256": parent["source_identity"]["trainer_sha256"],
        "config_path": "research/v02/configs/pilot_l128.json",
        "config_sha256": parent["source_identity"]["config_sha256"],
        "source_provenance_path": "research/v02/source_provenance.json",
        "source_provenance_sha256": parent["source_identity"][
            "source_provenance_sha256"
        ],
        "git": {
            "head": parent["source_identity"]["git_head"],
            "branch": "codex/neuralhorner-v02-20260816",
            "status": [],
        },
    }
    model = TRAINER.Cell(**architecture)
    checkpoint = {
        "schema": TRAINER.CHECKPOINT_SCHEMA,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": {"ignored": True},
        "scheduler_state_dict": {"ignored": True},
        "rng_state": {"ignored": True},
        "config": architecture,
        "experiment_config": {
            "name": parent["experiment"],
            "width": parent["width"],
            "tiers": required_gate["tiers"],
            "evaluation_width_modes": required_gate["width_modes"],
            "final_eval_n": required_gate["rollout_n"],
            "small_prime_limit": required_gate["small_prime_limit"],
            "arms": {parent["arm"]: {"architecture": architecture}},
        },
        "arm": parent["arm"],
        "L": parent["width"],
        "step": parent["checkpoint_step"],
        "source_identity": source,
    }
    checkpoint_path = tmp_path / "weights_step24000.pt"
    torch.save(checkpoint, checkpoint_path)
    parent["checkpoint_sha256"] = TRAINER.sha256_file(checkpoint_path)
    parent["state_dict_signature_sha256"] = TRAINER.state_dict_signature_sha256(
        model.state_dict()
    )

    receipt = {
        "schema": TRAINER.LEGACY_RECEIPT_SCHEMA,
        "status": "completed_pass",
        "experiment": parent["experiment"],
        "arm": parent["arm"],
        "architecture": architecture,
        "parameters": parent["parameters"],
        "config": {
            "width": parent["width"],
            "master_seed": parent["master_seed"],
            "tiers": required_gate["tiers"],
            "evaluation_width_modes": required_gate["width_modes"],
            "final_eval_n": required_gate["rollout_n"],
            "small_prime_limit": required_gate["small_prime_limit"],
        },
        "source_identity": source,
        "environment": {
            "device": "mps",
            "torch": "test-parent",
            "deterministic_algorithms_enabled": True,
        },
        "warm_start": None,
        "final_gate": {
            "passed": True,
            "parameters_finite": True,
            "rollout_exact": True,
            "small_prime_exact": True,
        },
        "history": [
            {
                "step": parent["checkpoint_step"],
                "parameters_finite": True,
                "tiers": _exact_rollout_rows(
                    required_gate["tiers"], required_gate["rollout_n"]
                ),
                "small_prime_exhaustive": _exact_small_prime_rows(
                    parent["width"], required_gate["small_prime_limit"]
                ),
            }
        ],
        "checkpoints": [
            {
                "step": parent["checkpoint_step"],
                "path": checkpoint_path.name,
                "sha256": parent["checkpoint_sha256"],
            }
        ],
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    parent["receipt_sha256"] = TRAINER.sha256_file(receipt_path)
    return config, receipt_path, checkpoint_path


def _write_v2_parent_artifacts(
    tmp_path: Path,
    *,
    confirmation_minimum_by_tier: dict[str, int] | None = None,
    confirmation_misses: dict[tuple[int, str], int] | None = None,
) -> tuple[dict[str, object], Path, Path]:
    parent_config = json.loads(
        (ROOT / "research/v02/configs/bridge_l256_b127.json").read_text()
    )
    parent_config["final_eval_n"] = 37
    parent_config["selection"]["confirmation_n"] = 37
    parent_config["small_prime_limit"] = 32
    parent_config["evaluation_width_modes"] = ["dynamic", "fixed"]
    if confirmation_minimum_by_tier is not None:
        parent_config["selection"]["confirmation_minimum_correct_by_tier"] = (
            confirmation_minimum_by_tier
        )
    child_config = deepcopy(parent_config)
    child_config["name"] = "nh02-bridge-l512-b127"
    child_config["width"] = 512
    child_config["tiers"] = [6, 7, 8]
    child_config["selection"].pop("screen_minimum_correct_by_tier", None)
    child_config["selection"].pop("confirmation_minimum_correct_by_tier", None)

    architecture = parent_config["arms"]["B127"]["architecture"]
    source = {
        "trainer_path": "research/v02/train_scale.py",
        "trainer_sha256": "1" * 64,
        "config_path": "research/v02/configs/bridge_l256_b127.json",
        "config_sha256": "2" * 64,
        "source_provenance_path": "research/v02/source_provenance.json",
        "source_provenance_sha256": "3" * 64,
        "git": {"head": "4" * 40, "branch": "test-parent", "status": []},
    }
    step = 6_000
    model = TRAINER.Cell(**architecture)
    checkpoint = {
        "schema": TRAINER.CHECKPOINT_SCHEMA,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": {"not_loaded": True},
        "scheduler_state_dict": {"not_loaded": True},
        "rng_state": {"not_loaded": True},
        "config": architecture,
        "experiment_config": parent_config,
        "arm": "B127",
        "L": 256,
        "step": step,
        "source_identity": source,
    }
    checkpoint_path = tmp_path / "weights_step6000.pt"
    torch.save(checkpoint, checkpoint_path)
    checkpoint_sha = TRAINER.sha256_file(checkpoint_path)

    confirmation_tiers = _exact_rollout_rows(
        parent_config["tiers"], parent_config["selection"]["confirmation_n"]
    )
    for (tier, width_mode), misses in (confirmation_misses or {}).items():
        confirmation_tiers[str(tier)][width_mode]["correct"] -= misses
    small_prime = _exact_small_prime_rows(
        parent_config["width"], parent_config["small_prime_limit"]
    )
    confirmation_gate = TRAINER.confirmed_gate(
        confirmation_tiers,
        small_prime,
        parent_config,
        True,
        parent_config["selection"]["confirmation_n"],
    )
    checkpoint_entry = {
        "step": step,
        "path": checkpoint_path.name,
        "sha256": checkpoint_sha,
    }
    selected = {
        **checkpoint_entry,
        "reason": "first_confirmed_pass",
        "confirmation_gate": confirmation_gate,
    }
    receipt = {
        "schema": TRAINER.RECEIPT_SCHEMA,
        "status": "completed_pass",
        "experiment": parent_config["name"],
        "role": parent_config["role"],
        "arm": "B127",
        "architecture": architecture,
        "parameters": parent_config["arms"]["B127"]["expected_parameters"],
        "config": parent_config,
        "seeds": {"master": parent_config["master_seed"]},
        "source_identity": source,
        "environment": {
            "device": "cuda",
            "torch": "test-parent",
            "deterministic_algorithms_enabled": True,
        },
        "selection": parent_config["selection"],
        "selected_checkpoint": selected,
        "steps_completed": step,
        "stopped_at_first_confirmed_pass": True,
        "final_gate": confirmation_gate,
        "history": [
            {
                "step": step,
                "parameters_finite": True,
                "tiers": _exact_rollout_rows(
                    parent_config["tiers"],
                    parent_config["selection"]["screen_n"],
                ),
                "screen_gate": {"passed": True},
                "confirmation": {
                    "tiers": confirmation_tiers,
                    "small_prime_exhaustive": small_prime,
                    "gate": confirmation_gate,
                },
                "small_prime_exhaustive": small_prime,
            }
        ],
        "checkpoints": [checkpoint_entry],
    }
    receipt_path = tmp_path / "receipt_v2.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    selected_path = tmp_path / "SELECTED.json"
    selected_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")

    child_parent = child_config["initialization"]["parent"]
    child_parent.update(
        {
            "experiment": parent_config["name"],
            "arm": "B127",
            "width": parent_config["width"],
            "checkpoint_step": step,
            "parameters": parent_config["arms"]["B127"][
                "expected_parameters"
            ],
            "master_seed": parent_config["master_seed"],
            "architecture": architecture,
            "receipt_sha256": TRAINER.sha256_file(receipt_path),
            "selected_sha256": TRAINER.sha256_file(selected_path),
            "checkpoint_sha256": checkpoint_sha,
            "state_dict_signature_sha256": TRAINER.state_dict_signature_sha256(
                model.state_dict()
            ),
            "source_identity": {
                "git_head": source["git"]["head"],
                "trainer_sha256": source["trainer_sha256"],
                "config_sha256": source["config_sha256"],
                "source_provenance_sha256": source[
                    "source_provenance_sha256"
                ],
            },
            "required_gate": {
                "tiers": parent_config["tiers"],
                "width_modes": parent_config["evaluation_width_modes"],
                "rollout_n": parent_config["selection"]["confirmation_n"],
                "minimum_correct_by_tier": (
                    TRAINER.selection_minimum_correct_by_tier(parent_config, "confirmation")
                ),
                "small_prime_limit": parent_config["small_prime_limit"],
            },
        }
    )
    return child_config, receipt_path, checkpoint_path


def test_l256_bridge_config_is_frozen_to_declared_protocol() -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/bridge_l256_b127.json").read_text()
    )
    arm = TRAINER.validate_config(config, "B127")
    assert config["width"] == 256
    assert config["tiers"] == [4, 5, 6, 7]
    assert config["steps"] == 80_000
    assert config["batch_size"] == 256
    assert config["master_seed"] == 23
    assert config["peak_lr"] == 0.0015
    assert config["weight_decay"] == 0.01
    assert config["warmup_fraction"] == 0.025
    assert config["eval_every"] == 3_000
    assert config["selection"] == {
        "mode": "first_confirmed_pass",
        "screen_n": 64,
        "confirmation_n": 512,
        "require_small_prime_exhaustive": True,
    }
    assert config["runtime_policy"] == {
        "device": "cuda",
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    assert config["source_policy"] == {"require_clean_git": True}
    assert arm["expected_parameters"] == 126_603


def test_l512_scale_config_is_bound_to_l256_and_declares_approximate_gate() -> None:
    config = json.loads((ROOT / "research/v02/configs/scale_l512_b127.json").read_text())
    arm = TRAINER.validate_config(config, "B127")
    parent = config["initialization"]["parent"]
    assert config["role"] == "single_seed_cross_width_scale_bridge"
    assert config["width"] == 512
    assert config["tiers"] == [6, 7, 8]
    assert config["steps"] == 50_000
    assert config["batch_size"] == 192
    assert config["master_seed"] == 23
    assert config["peak_lr"] == 0.0015
    assert config["weight_decay"] == 0.01
    assert config["warmup_fraction"] == 0.025
    assert config["eval_every"] == 3_000
    assert config["selection"] == {
        "mode": "first_confirmed_pass",
        "screen_n": 64,
        "screen_minimum_correct_by_tier": {"6": 64, "7": 64, "8": 64},
        "confirmation_n": 256,
        "confirmation_minimum_correct_by_tier": {
            "6": 256,
            "7": 255,
            "8": 255,
        },
        "require_small_prime_exhaustive": True,
    }
    assert parent["experiment"] == "nh02-bridge-l256-b127"
    assert parent["width"] == 256
    assert parent["checkpoint_step"] == 21_000
    assert parent["receipt_sha256"] == (
        "c49f66783c62a05847c9e5461ffd6b7362e338a2800f28d0a50f2f3784b58158"
    )
    assert parent["checkpoint_sha256"] == (
        "ab90103e630a26e49f1021fd06e9c63ec7ae0998b411651904b2b94497f5f1c1"
    )
    assert parent["source_identity"] == {
        "git_head": "6b6e1a59e2437c7384f68087e87dd4d34ea3c9e7",
        "trainer_sha256": ("30fdfbe6cd2bb6c871134e0952a1a4264b9a38db9a544cd8d9e97959a74c457b"),
        "config_sha256": ("4dd2f792a61d1902129bd5e1f1e8fd019d175270252b131bc672a027930206bc"),
        "source_provenance_sha256": (
            "2d84692b49a3ebdc7550ff61606cb999dd27c2fbd7bcaa06f0e1da812be1b4f0"
        ),
    }
    assert parent["required_gate"]["minimum_correct_by_tier"] == {
        "4": 512,
        "5": 512,
        "6": 512,
        "7": 512,
    }
    assert arm["expected_parameters"] == 126_603


def test_l512_scale_gate_keeps_threshold_pass_distinct_from_exactness() -> None:
    config = json.loads((ROOT / "research/v02/configs/scale_l512_b127.json").read_text())
    tiers = _exact_rollout_rows(config["tiers"], 256)
    tiers["7"]["dynamic"]["correct"] = 255
    tiers["8"]["fixed"]["correct"] = 255
    small_prime = _exact_small_prime_rows(512, 64)

    gate = TRAINER.confirmed_gate(tiers, small_prime, config, True, 256)
    assert gate["passed"] is True
    assert gate["rollout_threshold_passed"] is True
    assert gate["rollout_exact"] is False
    assert gate["small_prime_exact"] is True
    assert gate["rollout"]["minimum_correct_by_tier"] == {
        "6": 256,
        "7": 255,
        "8": 255,
    }

    below_threshold = deepcopy(tiers)
    below_threshold["7"]["dynamic"]["correct"] = 254
    failed = TRAINER.confirmed_gate(below_threshold, small_prime, config, True, 256)
    assert failed["passed"] is False
    assert "tier_7_dynamic_below_threshold" in failed["rollout"]["failures"]


def test_l1024_scale_config_is_bound_to_selected_l512_parent() -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/scale_l1024_b127.json").read_text()
    )
    arm = TRAINER.validate_config(config, "B127")
    parent = config["initialization"]["parent"]

    assert config["role"] == "single_seed_cross_width_scale_bridge"
    assert config["width"] == 1024
    assert config["tiers"] == [6, 7, 8, 9]
    assert config["steps"] == 60_000
    assert config["batch_size"] == 96
    assert config["eval_batch_size"] == 32
    assert config["master_seed"] == 23
    assert config["peak_lr"] == 0.0015
    assert config["weight_decay"] == 0.01
    assert config["warmup_fraction"] == 0.025
    assert config["eval_every"] == 3_000
    assert config["selection"] == {
        "mode": "first_confirmed_pass",
        "evaluate_step_zero": True,
        "screen_n": 64,
        "screen_minimum_correct_by_tier": {
            "6": 64,
            "7": 64,
            "8": 64,
            "9": 64,
        },
        "confirmation_n": 256,
        "confirmation_minimum_correct_by_tier": {
            "6": 256,
            "7": 256,
            "8": 256,
            "9": 255,
        },
        "require_small_prime_exhaustive": True,
    }
    assert parent["experiment"] == "nh02-scale-l512-b127"
    assert parent["width"] == 512
    assert parent["checkpoint_step"] == 48_000
    assert parent["receipt_sha256"] == (
        "abc5700038342725c443e200ef9bb9efd912ab4350f390272feaaaa2d392705a"
    )
    assert parent["checkpoint_sha256"] == (
        "970165b1af9e1518b83529ee8280521d9e3baece67d79efeee020e61c31f8ec7"
    )
    assert parent["state_dict_signature_sha256"] == (
        "ee8b058770fbdef625a8e09e210f9f13ca5fc17ee5aa4d7bcf6bcebaf51e17fe"
    )
    assert parent["source_identity"] == {
        "git_head": "7afdbaaca7495ab683ab1aac2bc9c002900dc88f",
        "trainer_sha256": (
            "767e0c96194266b82c6c65c64a18e656b8046b25c07bd46e69511c8a1dce5682"
        ),
        "config_sha256": (
            "23f25349bb98a2c0863d31d472a05311f737fba589817f29ff0961695002f6b7"
        ),
        "source_provenance_sha256": (
            "2d84692b49a3ebdc7550ff61606cb999dd27c2fbd7bcaa06f0e1da812be1b4f0"
        ),
    }
    assert parent["required_gate"] == {
        "tiers": [6, 7, 8],
        "width_modes": ["fixed", "dynamic"],
        "rollout_n": 256,
        "minimum_correct_by_tier": {"6": 256, "7": 255, "8": 255},
        "small_prime_limit": 64,
    }
    assert config["runtime_policy"] == {
        "device": "cuda",
        "cublas_workspace_config": ":4096:8",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
    }
    assert config["source_policy"] == {"require_clean_git": True}
    assert arm["expected_parameters"] == 126_603


def test_l1024_scale_gate_is_exact_on_anchor_and_approximate_on_frontier() -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/scale_l1024_b127.json").read_text()
    )
    tiers = _exact_rollout_rows(config["tiers"], 256)
    tiers["9"]["dynamic"]["correct"] = 255
    small_prime = _exact_small_prime_rows(1024, 64)

    gate = TRAINER.confirmed_gate(tiers, small_prime, config, True, 256)
    assert gate["passed"] is True
    assert gate["rollout_threshold_passed"] is True
    assert gate["rollout_exact"] is False
    assert gate["small_prime_exact"] is True
    assert gate["rollout"]["minimum_correct_by_tier"] == {
        "6": 256,
        "7": 256,
        "8": 256,
        "9": 255,
    }

    retention_miss = deepcopy(tiers)
    retention_miss["7"]["fixed"]["correct"] = 255
    failed_retention = TRAINER.confirmed_gate(
        retention_miss, small_prime, config, True, 256
    )
    assert failed_retention["passed"] is False
    assert "tier_7_fixed_not_exact" in failed_retention["rollout"]["failures"]

    frontier_below_threshold = deepcopy(tiers)
    frontier_below_threshold["9"]["dynamic"]["correct"] = 254
    failed_frontier = TRAINER.confirmed_gate(
        frontier_below_threshold, small_prime, config, True, 256
    )
    assert failed_frontier["passed"] is False
    assert (
        "tier_9_dynamic_below_threshold"
        in failed_frontier["rollout"]["failures"]
    )


def test_l2048_scale_config_is_bound_to_exact_l1024_update_zero_parent() -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/scale_l2048_b127.json").read_text()
    )
    arm = TRAINER.validate_config(config, "B127")
    parent = config["initialization"]["parent"]

    assert config["role"] == "single_seed_full_width_scale_repair_bridge"
    assert config["width"] == 2048
    assert config["tiers"] == [6, 7, 8, 9, 10]
    assert config["steps"] == 60_000
    assert config["batch_size"] == 48
    assert config["eval_batch_size"] == 64
    assert config["master_seed"] == 23
    assert config["peak_lr"] == 0.0015
    assert config["weight_decay"] == 0.01
    assert config["warmup_fraction"] == 0.025
    assert config["eval_every"] == 3_000
    assert config["selection"] == {
        "mode": "first_confirmed_pass",
        "evaluate_step_zero": True,
        "screen_n": 64,
        "screen_minimum_correct_by_tier": {
            "6": 64,
            "7": 64,
            "8": 64,
            "9": 64,
            "10": 64,
        },
        "confirmation_n": 256,
        "confirmation_minimum_correct_by_tier": {
            "6": 256,
            "7": 256,
            "8": 256,
            "9": 256,
            "10": 255,
        },
        "require_small_prime_exhaustive": True,
    }
    assert parent["experiment"] == "nh02-scale-l1024-b127"
    assert parent["width"] == 1024
    assert parent["checkpoint_step"] == 0
    assert parent["receipt_sha256"] == (
        "80948748a41183a809faf282600cc0f8343691b6cdb4bebaac4f1f468df95651"
    )
    assert parent["selected_sha256"] == (
        "53addcb1a400952cf5e52f6b8c4e0fd129ede9c1471b82a026c62dbd7783d1c7"
    )
    assert parent["checkpoint_sha256"] == (
        "d296b711bb6a7faaa1dd81e05478cfa75f11071c42a8c36fbf60e758ee7eb407"
    )
    assert parent["state_dict_signature_sha256"] == (
        "ee8b058770fbdef625a8e09e210f9f13ca5fc17ee5aa4d7bcf6bcebaf51e17fe"
    )
    assert parent["source_identity"] == {
        "git_head": "1e57afc18539b3dca2e959d7d09c27fbed592601",
        "trainer_sha256": (
            "4dcf4fab0c731629ffcdfc2a20424d32e9a4d6e12604ea6c1e91a353b139108a"
        ),
        "config_sha256": (
            "5fde374dd060b19f477549e21dbd4c39ca71934091a4afd571cfcaa379b97d19"
        ),
        "source_provenance_sha256": (
            "2d84692b49a3ebdc7550ff61606cb999dd27c2fbd7bcaa06f0e1da812be1b4f0"
        ),
    }
    assert parent["required_gate"] == {
        "tiers": [6, 7, 8, 9],
        "width_modes": ["fixed", "dynamic"],
        "rollout_n": 256,
        "minimum_correct_by_tier": {
            "6": 256,
            "7": 256,
            "8": 256,
            "9": 255,
        },
        "small_prime_limit": 64,
    }
    assert arm["expected_parameters"] == 126_603


def test_l2048_gate_is_exact_on_retention_and_approximate_only_on_tier10() -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/scale_l2048_b127.json").read_text()
    )
    tiers = _exact_rollout_rows(config["tiers"], 256)
    tiers["10"]["dynamic"]["correct"] = 255
    small_prime = _exact_small_prime_rows(2048, 64)

    gate = TRAINER.confirmed_gate(tiers, small_prime, config, True, 256)
    assert gate["passed"] is True
    assert gate["rollout_threshold_passed"] is True
    assert gate["rollout_exact"] is False
    assert gate["small_prime_exact"] is True
    assert gate["rollout"]["minimum_correct_by_tier"] == {
        "6": 256,
        "7": 256,
        "8": 256,
        "9": 256,
        "10": 255,
    }

    retention_miss = deepcopy(tiers)
    retention_miss["9"]["fixed"]["correct"] = 255
    failed_retention = TRAINER.confirmed_gate(
        retention_miss, small_prime, config, True, 256
    )
    assert failed_retention["passed"] is False
    assert "tier_9_fixed_not_exact" in failed_retention["rollout"]["failures"]

    frontier_below_threshold = deepcopy(tiers)
    frontier_below_threshold["10"]["dynamic"]["correct"] = 254
    failed_frontier = TRAINER.confirmed_gate(
        frontier_below_threshold, small_prime, config, True, 256
    )
    assert failed_frontier["passed"] is False
    assert (
        "tier_10_dynamic_below_threshold"
        in failed_frontier["rollout"]["failures"]
    )

    screen_tiers = _exact_rollout_rows(config["tiers"], 64)
    screen_tiers["10"]["fixed"]["correct"] = 63
    screen = TRAINER.rollout_gate(
        screen_tiers,
        config["tiers"],
        config["evaluation_width_modes"],
        64,
        TRAINER.selection_minimum_correct_by_tier(config, "screen"),
    )
    assert screen["passed"] is False
    assert "tier_10_fixed_not_exact" in screen["failures"]


def test_step_zero_selection_requires_boolean_and_warm_start() -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/scale_l1024_b127.json").read_text()
    )

    nonboolean = deepcopy(config)
    nonboolean["selection"]["evaluate_step_zero"] = 1
    with pytest.raises(ValueError, match="evaluate_step_zero must be boolean"):
        TRAINER.validate_config(nonboolean, "B127")

    scratch = deepcopy(config)
    scratch["initialization"] = {"mode": "scratch"}
    scratch["allow_warm_start"] = False
    with pytest.raises(ValueError, match="requires a warm-start"):
        TRAINER.validate_config(scratch, "B127")


def test_l512_screen_gate_is_exact_and_threshold_maps_fail_closed() -> None:
    config = json.loads((ROOT / "research/v02/configs/scale_l512_b127.json").read_text())
    tiers = _exact_rollout_rows(config["tiers"], 64)
    tiers["8"]["fixed"]["correct"] = 63
    gate = TRAINER.rollout_gate(
        tiers,
        config["tiers"],
        config["evaluation_width_modes"],
        64,
        TRAINER.selection_minimum_correct_by_tier(config, "screen"),
    )
    assert gate["passed"] is False
    assert "tier_8_fixed_not_exact" in gate["failures"]

    missing_tier = deepcopy(config)
    del missing_tier["selection"]["confirmation_minimum_correct_by_tier"]["8"]
    with pytest.raises(ValueError, match="must name exactly"):
        TRAINER.validate_config(missing_tier, "B127")

    impossible = deepcopy(config)
    impossible["selection"]["confirmation_minimum_correct_by_tier"]["7"] = 257
    with pytest.raises(ValueError, match=r"integer in \[1, 256\]"):
        TRAINER.validate_config(impossible, "B127")


def test_resume_is_explicitly_rejected_before_run_setup(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="resume artifacts require"):
        TRAINER.run(
            SimpleNamespace(
                config=ROOT / "research/v02/configs/bridge_l256_b127.json",
                arm="B127",
                resume=tmp_path / "resume.pt",
            )
        )


def test_canonical_cuda_policy_fails_closed_and_applies_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/bridge_l256_b127.json").read_text()
    )
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    with pytest.raises(ValueError, match="explicit --device cuda"):
        TRAINER.configure_canonical_runtime(config, "auto")
    with pytest.raises(RuntimeError, match="CUBLAS_WORKSPACE_CONFIG"):
        TRAINER.configure_canonical_runtime(config, "cuda")

    old_settings = TRAINER.runtime_settings_identity()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        policy = TRAINER.configure_canonical_runtime(config, "cuda")
        assert policy == config["runtime_policy"]
        assert TRAINER.runtime_settings_identity() == {
            key: value
            for key, value in config["runtime_policy"].items()
            if key != "device"
        }
    finally:
        torch.backends.cudnn.benchmark = old_settings["cudnn_benchmark"]
        torch.backends.cudnn.deterministic = old_settings["cudnn_deterministic"]
        torch.backends.cuda.matmul.allow_tf32 = old_settings[
            "cuda_matmul_allow_tf32"
        ]
        torch.backends.cudnn.allow_tf32 = old_settings["cudnn_allow_tf32"]
        torch.set_float32_matmul_precision(
            old_settings["float32_matmul_precision"]
        )


def test_cuda_environment_receipt_records_driver_cudnn_and_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        name="Test H100",
        major=9,
        minor=0,
        total_memory=80 * 1024**3,
        multi_processor_count=132,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _device: properties)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(torch.backends.cudnn, "version", lambda: 9_100)
    monkeypatch.setattr(torch.version, "cuda", "12.4")
    monkeypatch.setattr(TRAINER, "nvidia_driver_versions", lambda: ["550.54.15"])
    identity = TRAINER.environment_identity(torch.device("cuda"))
    assert identity["cuda_runtime"] == "12.4"
    assert identity["cuda_driver_versions"] == ["550.54.15"]
    assert identity["cudnn_version"] == 9_100
    assert identity["device_name"] == "Test H100"
    assert identity["device_capability"] == [9, 0]
    assert identity["runtime_settings"] == TRAINER.runtime_settings_identity()


def test_canonical_source_policy_rejects_dirty_and_binds_head_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/bridge_l256_b127.json").read_text()
    )
    with pytest.raises(RuntimeError, match="clean Git worktree"):
        TRAINER.enforce_source_policy(
            config,
            {"git": {"head": "a" * 40, "status": [" M trainer.py"]}},
        )
    blobs = {
        "trainer.py": b"trainer\n",
        "config.json": b"config\n",
        "provenance.json": b"provenance\n",
    }
    monkeypatch.setattr(
        TRAINER.subprocess,
        "check_output",
        lambda arguments, cwd: blobs[arguments[-1].removeprefix("HEAD:")],
    )
    source = {
        "git": {"head": "a" * 40, "status": []},
        "trainer_path": "trainer.py",
        "trainer_sha256": TRAINER.sha256_bytes(blobs["trainer.py"]),
        "config_path": "config.json",
        "config_sha256": TRAINER.sha256_bytes(blobs["config.json"]),
        "source_provenance_path": "provenance.json",
        "source_provenance_sha256": TRAINER.sha256_bytes(
            blobs["provenance.json"]
        ),
    }
    TRAINER.enforce_source_policy(config, source)
    source["config_sha256"] = "f" * 64
    with pytest.raises(RuntimeError, match="differs from its HEAD bytes"):
        TRAINER.enforce_source_policy(config, source)


def test_loop_stops_at_first_confirmation_and_writes_selected_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, parent_receipt, parent_checkpoint = _write_bound_parent_artifacts(
        tmp_path
    )
    config["steps"] = 3
    config["batch_size"] = 1
    config["eval_every"] = 1
    config["eval_batch_size"] = 1
    config_path = tmp_path / "loop_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "loop_output"

    monkeypatch.setattr(
        TRAINER,
        "configure_canonical_runtime",
        lambda child_config, requested: child_config["runtime_policy"],
    )
    monkeypatch.setattr(TRAINER, "resolve_device", lambda _requested: torch.device("cuda"))
    monkeypatch.setattr(
        TRAINER,
        "environment_identity",
        lambda _device: {
            "device": "cuda",
            "torch": "test-child",
            "deterministic_algorithms_enabled": True,
        },
    )
    monkeypatch.setattr(
        TRAINER,
        "git_identity",
        lambda: {"head": "a" * 40, "branch": "test", "status": []},
    )
    monkeypatch.setattr(
        TRAINER,
        "enforce_source_policy",
        lambda _config, _source_identity: None,
    )
    monkeypatch.setattr(TRAINER.Cell, "to", lambda self, _device: self)
    monkeypatch.setattr(TRAINER, "synchronize", lambda _device: None)
    monkeypatch.setattr(TRAINER, "rng_state", lambda _rng, _device: {"test": True})
    monkeypatch.setattr(
        TRAINER,
        "train_batch",
        lambda _batch, _rng, _width, _device: (
            torch.zeros((1, 1, 3)),
            torch.zeros(1, dtype=torch.long),
            torch.zeros((1, 1)),
        ),
    )

    screen_calls = 0

    def fake_rollouts(
        _model: object,
        child_config: dict[str, object],
        _seeds: dict[str, int],
        _device: torch.device,
        count: int,
    ) -> dict[str, object]:
        nonlocal screen_calls
        rows = _exact_rollout_rows(child_config["tiers"], count)
        if count == child_config["selection"]["screen_n"]:
            screen_calls += 1
            if screen_calls == 1:
                rows["4"]["fixed"]["correct"] = count - 1
        return rows

    monkeypatch.setattr(TRAINER, "evaluate_rollouts", fake_rollouts)
    monkeypatch.setattr(
        TRAINER,
        "evaluate_small_primes",
        lambda _model, child_config, _device: _exact_small_prime_rows(
            child_config["width"], child_config["small_prime_limit"]
        ),
    )

    result = TRAINER.run(
        SimpleNamespace(
            config=config_path,
            arm="B127",
            out=output,
            device="cuda",
            warm_start=parent_checkpoint,
            parent_receipt=parent_receipt,
            predecessor_receipt=None,
            resume=None,
        )
    )
    assert result == 0
    receipt = json.loads((output / "receipt.json").read_text())
    selected = json.loads((output / "SELECTED.json").read_text())
    assert receipt["status"] == "completed_pass"
    assert receipt["steps_completed"] == 2
    assert receipt["stopped_at_first_confirmed_pass"] is True
    assert receipt["selected_checkpoint"] == selected
    assert selected["step"] == 2
    assert selected["reason"] == "first_confirmed_pass"
    assert len(receipt["history"]) == 2
    assert (output / "weights_step1.pt").is_file()
    assert (output / "weights_step2.pt").is_file()
    assert not (output / "weights_step3.pt").exists()


def test_warm_start_step_zero_can_select_without_training_and_seed_next_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, parent_receipt, parent_checkpoint = _write_v2_parent_artifacts(tmp_path)
    config["name"] = "test-step-zero-child"
    config["steps"] = 3
    config["batch_size"] = 1
    config["eval_every"] = 1
    config["checkpoint_eval_n"] = 2
    config["final_eval_n"] = 4
    config["eval_batch_size"] = 1
    config["selection"] = {
        "mode": "first_confirmed_pass",
        "evaluate_step_zero": True,
        "screen_n": 2,
        "screen_minimum_correct_by_tier": {"6": 2, "7": 2, "8": 2},
        "confirmation_n": 4,
        "confirmation_minimum_correct_by_tier": {"6": 4, "7": 4, "8": 4},
        "require_small_prime_exhaustive": True,
    }
    config_path = tmp_path / "step_zero_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "step_zero_output"

    monkeypatch.setattr(
        TRAINER,
        "configure_canonical_runtime",
        lambda child_config, requested: child_config["runtime_policy"],
    )
    monkeypatch.setattr(TRAINER, "resolve_device", lambda _requested: torch.device("cuda"))
    monkeypatch.setattr(
        TRAINER,
        "environment_identity",
        lambda _device: {
            "device": "cuda",
            "torch": "test-child",
            "deterministic_algorithms_enabled": True,
        },
    )
    monkeypatch.setattr(
        TRAINER,
        "git_identity",
        lambda: {"head": "a" * 40, "branch": "test", "status": []},
    )
    monkeypatch.setattr(
        TRAINER,
        "enforce_source_policy",
        lambda _config, _source_identity: None,
    )
    monkeypatch.setattr(TRAINER.Cell, "to", lambda self, _device: self)
    monkeypatch.setattr(TRAINER, "synchronize", lambda _device: None)
    monkeypatch.setattr(TRAINER, "rng_state", lambda _rng, _device: {"test": True})

    def reject_training(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("step-zero selection must not draw a training batch")

    monkeypatch.setattr(TRAINER, "train_batch", reject_training)
    monkeypatch.setattr(
        TRAINER,
        "evaluate_rollouts",
        lambda _model, child_config, _seeds, _device, count: _exact_rollout_rows(
            child_config["tiers"], count
        ),
    )
    monkeypatch.setattr(
        TRAINER,
        "evaluate_small_primes",
        lambda _model, child_config, _device: _exact_small_prime_rows(
            child_config["width"], child_config["small_prime_limit"]
        ),
    )

    result = TRAINER.run(
        SimpleNamespace(
            config=config_path,
            arm="B127",
            out=output,
            device="cuda",
            warm_start=parent_checkpoint,
            parent_receipt=parent_receipt,
            predecessor_receipt=None,
            resume=None,
        )
    )
    assert result == 0
    receipt = json.loads((output / "receipt.json").read_text())
    selected = json.loads((output / "SELECTED.json").read_text())
    assert receipt["status"] == "completed_pass"
    assert receipt["steps_completed"] == 0
    assert receipt["stopped_at_first_confirmed_pass"] is True
    assert receipt["history"][0]["step"] == 0
    assert receipt["history"][0]["loss"] is None
    assert selected["step"] == 0
    assert selected["path"] == "weights_step0.pt"
    assert (output / "weights_step0.pt").is_file()
    assert not (output / "weights_step1.pt").exists()

    original_payload = torch.load(
        parent_checkpoint, map_location="cpu", weights_only=True
    )
    selected_payload = torch.load(
        output / "weights_step0.pt", map_location="cpu", weights_only=True
    )
    assert selected_payload["step"] == 0
    assert selected_payload["L"] == config["width"]
    assert selected_payload["experiment_config"] == config
    assert selected_payload["state_dict"].keys() == original_payload["state_dict"].keys()
    for key in selected_payload["state_dict"]:
        assert torch.equal(
            selected_payload["state_dict"][key], original_payload["state_dict"][key]
        )

    next_config = deepcopy(config)
    next_config["name"] = "test-after-step-zero"
    next_config["width"] = 1024
    next_config["tiers"] = [8, 9]
    next_config["selection"] = {
        "mode": "first_confirmed_pass",
        "evaluate_step_zero": True,
        "screen_n": 2,
        "screen_minimum_correct_by_tier": {"8": 2, "9": 2},
        "confirmation_n": 4,
        "confirmation_minimum_correct_by_tier": {"8": 4, "9": 4},
        "require_small_prime_exhaustive": True,
    }
    child_source = receipt["source_identity"]
    next_parent = next_config["initialization"]["parent"]
    next_parent.update(
        {
            "experiment": config["name"],
            "width": config["width"],
            "checkpoint_step": 0,
            "receipt_sha256": TRAINER.sha256_file(output / "receipt.json"),
            "selected_sha256": TRAINER.sha256_file(output / "SELECTED.json"),
            "checkpoint_sha256": TRAINER.sha256_file(output / "weights_step0.pt"),
            "state_dict_signature_sha256": TRAINER.state_dict_signature_sha256(
                selected_payload["state_dict"]
            ),
            "source_identity": {
                "git_head": child_source["git"]["head"],
                "trainer_sha256": child_source["trainer_sha256"],
                "config_sha256": child_source["config_sha256"],
                "source_provenance_sha256": child_source[
                    "source_provenance_sha256"
                ],
            },
            "required_gate": {
                "tiers": config["tiers"],
                "width_modes": config["evaluation_width_modes"],
                "rollout_n": config["selection"]["confirmation_n"],
                "minimum_correct_by_tier": config["selection"][
                    "confirmation_minimum_correct_by_tier"
                ],
                "small_prime_limit": config["small_prime_limit"],
            },
        }
    )
    next_arm = TRAINER.validate_config(next_config, "B127")
    identity, state_dict = TRAINER.validate_warm_start_parent(
        output / "receipt.json",
        output / "weights_step0.pt",
        "B127",
        next_arm,
        next_config,
    )
    assert identity is not None
    assert state_dict is not None
    assert identity["parent_checkpoint"]["step"] == 0

    # The following cases isolate receipt/checkpoint step validation. The
    # selected-artifact binding has its own tamper test.
    next_parent.pop("selected_sha256")
    boolean_receipt = deepcopy(receipt)
    boolean_receipt["history"][0]["step"] = False
    boolean_receipt["checkpoints"][0]["step"] = False
    boolean_receipt["selected_checkpoint"]["step"] = False
    boolean_receipt["steps_completed"] = False
    boolean_receipt_path = tmp_path / "boolean_step_receipt.json"
    boolean_receipt_path.write_text(
        json.dumps(boolean_receipt, indent=2, sort_keys=True) + "\n"
    )
    next_parent["receipt_sha256"] = TRAINER.sha256_file(boolean_receipt_path)
    with pytest.raises(ValueError, match="invalid warm-start parent receipt"):
        TRAINER.validate_warm_start_parent(
            boolean_receipt_path,
            output / "weights_step0.pt",
            "B127",
            next_arm,
            next_config,
        )

    boolean_checkpoint_payload = deepcopy(selected_payload)
    boolean_checkpoint_payload["step"] = False
    boolean_checkpoint_path = tmp_path / "boolean_step_checkpoint.pt"
    torch.save(boolean_checkpoint_payload, boolean_checkpoint_path)
    boolean_checkpoint_sha = TRAINER.sha256_file(boolean_checkpoint_path)
    checkpoint_bound_receipt = deepcopy(receipt)
    checkpoint_bound_receipt["checkpoints"][0]["sha256"] = boolean_checkpoint_sha
    checkpoint_bound_receipt["selected_checkpoint"][
        "sha256"
    ] = boolean_checkpoint_sha
    checkpoint_bound_receipt_path = tmp_path / "checkpoint_bound_receipt.json"
    checkpoint_bound_receipt_path.write_text(
        json.dumps(checkpoint_bound_receipt, indent=2, sort_keys=True) + "\n"
    )
    next_parent["receipt_sha256"] = TRAINER.sha256_file(
        checkpoint_bound_receipt_path
    )
    next_parent["checkpoint_sha256"] = boolean_checkpoint_sha
    with pytest.raises(ValueError, match="invalid warm-start parent checkpoint: step"):
        TRAINER.validate_warm_start_parent(
            checkpoint_bound_receipt_path,
            boolean_checkpoint_path,
            "B127",
            next_arm,
            next_config,
        )


def test_l512_scale_loop_selects_first_threshold_pass_without_calling_it_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, parent_receipt, parent_checkpoint = _write_v2_parent_artifacts(tmp_path)
    config["steps"] = 3
    config["batch_size"] = 1
    config["eval_every"] = 1
    config["checkpoint_eval_n"] = 2
    config["final_eval_n"] = 4
    config["eval_batch_size"] = 1
    config["selection"] = {
        "mode": "first_confirmed_pass",
        "screen_n": 2,
        "screen_minimum_correct_by_tier": {"6": 2, "7": 2, "8": 2},
        "confirmation_n": 4,
        "confirmation_minimum_correct_by_tier": {"6": 4, "7": 3, "8": 3},
        "require_small_prime_exhaustive": True,
    }
    config_path = tmp_path / "scale_loop_config.json"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    output = tmp_path / "scale_loop_output"

    monkeypatch.setattr(
        TRAINER,
        "configure_canonical_runtime",
        lambda child_config, requested: child_config["runtime_policy"],
    )
    monkeypatch.setattr(TRAINER, "resolve_device", lambda _requested: torch.device("cuda"))
    monkeypatch.setattr(
        TRAINER,
        "environment_identity",
        lambda _device: {
            "device": "cuda",
            "torch": "test-child",
            "deterministic_algorithms_enabled": True,
        },
    )
    monkeypatch.setattr(
        TRAINER,
        "git_identity",
        lambda: {"head": "a" * 40, "branch": "test", "status": []},
    )
    monkeypatch.setattr(
        TRAINER,
        "enforce_source_policy",
        lambda _config, _source_identity: None,
    )
    monkeypatch.setattr(TRAINER.Cell, "to", lambda self, _device: self)
    monkeypatch.setattr(TRAINER, "synchronize", lambda _device: None)
    monkeypatch.setattr(TRAINER, "rng_state", lambda _rng, _device: {"test": True})
    monkeypatch.setattr(
        TRAINER,
        "train_batch",
        lambda _batch, _rng, _width, _device: (
            torch.zeros((1, 1, 3)),
            torch.zeros(1, dtype=torch.long),
            torch.zeros((1, 1)),
        ),
    )

    confirmation_calls = 0

    def fake_rollouts(
        _model: object,
        child_config: dict[str, object],
        _seeds: dict[str, int],
        _device: torch.device,
        count: int,
    ) -> dict[str, object]:
        nonlocal confirmation_calls
        rows = _exact_rollout_rows(child_config["tiers"], count)
        if count == child_config["selection"]["confirmation_n"]:
            confirmation_calls += 1
            rows["7"]["dynamic"]["correct"] = count - 2 if confirmation_calls == 1 else count - 1
            rows["8"]["fixed"]["correct"] = count - 1
        return rows

    monkeypatch.setattr(TRAINER, "evaluate_rollouts", fake_rollouts)
    monkeypatch.setattr(
        TRAINER,
        "evaluate_small_primes",
        lambda _model, child_config, _device: _exact_small_prime_rows(
            child_config["width"], child_config["small_prime_limit"]
        ),
    )

    result = TRAINER.run(
        SimpleNamespace(
            config=config_path,
            arm="B127",
            out=output,
            device="cuda",
            warm_start=parent_checkpoint,
            parent_receipt=parent_receipt,
            predecessor_receipt=None,
            resume=None,
        )
    )
    assert result == 0
    receipt = json.loads((output / "receipt.json").read_text())
    selected = json.loads((output / "SELECTED.json").read_text())
    assert receipt["steps_completed"] == 2
    assert receipt["selected_checkpoint"] == selected
    assert selected["step"] == 2
    assert selected["confirmation_gate"]["passed"] is True
    assert selected["confirmation_gate"]["rollout_threshold_passed"] is True
    assert selected["confirmation_gate"]["rollout_exact"] is False
    assert receipt["final_gate"] == selected["confirmation_gate"]
    assert len(receipt["history"]) == 2
    assert receipt["history"][0]["confirmation"]["gate"]["passed"] is False
    assert not (output / "weights_step3.pt").exists()


def test_cross_width_parent_accepts_environment_change_but_not_training_state(
    tmp_path: Path,
) -> None:
    config, receipt_path, checkpoint_path = _write_bound_parent_artifacts(tmp_path)
    arm = TRAINER.validate_config(config, "B127")
    identity, state_dict = TRAINER.validate_warm_start_parent(
        receipt_path,
        checkpoint_path,
        "B127",
        arm,
        config,
    )
    assert identity is not None
    assert state_dict is not None
    assert identity["parent_receipt"]["environment"]["device"] == "mps"
    assert identity["transfer"] == {
        "model_state_loaded": True,
        "optimizer_state_loaded": False,
        "scheduler_state_loaded": False,
        "rng_state_loaded": False,
        "parent_environment_continuity_required": False,
    }


def test_v2_parent_roundtrip_uses_confirmation_and_arbitrary_declared_tiers(
    tmp_path: Path,
) -> None:
    config, receipt_path, checkpoint_path = _write_v2_parent_artifacts(tmp_path)
    arm = TRAINER.validate_config(config, "B127")
    identity, state_dict = TRAINER.validate_warm_start_parent(
        receipt_path,
        checkpoint_path,
        "B127",
        arm,
        config,
    )
    assert identity is not None
    assert state_dict is not None
    assert identity["parent_receipt"]["schema"] == TRAINER.RECEIPT_SCHEMA
    assert identity["parent_receipt"]["selected_checkpoint"]["step"] == 6_000
    assert config["initialization"]["parent"]["required_gate"]["tiers"] == [
        4,
        5,
        6,
        7,
    ]
    assert config["initialization"]["parent"]["required_gate"]["rollout_n"] == 37
    assert config["initialization"]["parent"]["required_gate"]["width_modes"] == [
        "dynamic",
        "fixed",
    ]
    assert (
        config["initialization"]["parent"]["required_gate"]["small_prime_limit"]
        == 32
    )


def test_v2_parent_roundtrip_binds_approximate_confirmation_thresholds(
    tmp_path: Path,
) -> None:
    minimums = {"4": 37, "5": 37, "6": 37, "7": 36}
    config, receipt_path, checkpoint_path = _write_v2_parent_artifacts(
        tmp_path,
        confirmation_minimum_by_tier=minimums,
        confirmation_misses={(7, "dynamic"): 1},
    )
    arm = TRAINER.validate_config(config, "B127")
    identity, state_dict = TRAINER.validate_warm_start_parent(
        receipt_path,
        checkpoint_path,
        "B127",
        arm,
        config,
    )
    assert identity is not None
    assert state_dict is not None
    receipt = json.loads(receipt_path.read_text())
    assert receipt["final_gate"]["passed"] is True
    assert receipt["final_gate"]["rollout_exact"] is False

    config["initialization"]["parent"]["required_gate"]["minimum_correct_by_tier"]["7"] = 37
    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(
        ValueError,
        match="v2_confirmation_thresholds_declared|bound_rollout_gate",
    ):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


def test_v2_parent_rejects_mismatched_selected_artifact(tmp_path: Path) -> None:
    config, receipt_path, checkpoint_path = _write_v2_parent_artifacts(tmp_path)
    selected_path = receipt_path.parent / "SELECTED.json"
    selected = json.loads(selected_path.read_text())
    selected["step"] = 3_000
    selected_path.write_text(json.dumps(selected, indent=2, sort_keys=True) + "\n")

    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(ValueError, match="v2_selected_artifact_sha"):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("rollout_threshold_passed", False),
        ("rollout_exact", True),
        ("small_prime_exact", False),
        ("parameters_finite", False),
        ("nested_rollout_exact", True),
    ],
)
def test_v2_approximate_parent_rejects_stored_gate_semantic_tamper(
    tmp_path: Path,
    field: str,
    value: bool,
) -> None:
    config, receipt_path, checkpoint_path = _write_v2_parent_artifacts(
        tmp_path,
        confirmation_minimum_by_tier={"4": 37, "5": 37, "6": 37, "7": 36},
        confirmation_misses={(7, "dynamic"): 1},
    )
    receipt = json.loads(receipt_path.read_text())
    stored_gates = (
        receipt["selected_checkpoint"]["confirmation_gate"],
        receipt["history"][0]["confirmation"]["gate"],
        receipt["final_gate"],
    )
    for gate in stored_gates:
        if field == "nested_rollout_exact":
            gate["rollout"]["exact"] = value
        else:
            gate[field] = value
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    config["initialization"]["parent"]["receipt_sha256"] = TRAINER.sha256_file(
        receipt_path
    )

    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(
        ValueError,
        match="v2_stored_gate_semantics|v2_explicit_threshold_gate_fields",
    ):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


def test_v2_parent_recomputes_earlier_confirmation_instead_of_trusting_flag(
    tmp_path: Path,
) -> None:
    config, receipt_path, checkpoint_path = _write_v2_parent_artifacts(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    earlier = deepcopy(receipt["history"][0])
    earlier["step"] = 3_000
    earlier["confirmation"]["gate"]["passed"] = False
    receipt["history"].insert(0, earlier)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    config["initialization"]["parent"]["receipt_sha256"] = TRAINER.sha256_file(
        receipt_path
    )

    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(ValueError, match="v2_no_earlier_confirmed_checkpoint"):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


@pytest.mark.parametrize(
    ("field", "value", "failure"),
    [
        ("stopped_at_first_confirmed_pass", False, "v2_stopped"),
        ("selected_reason", "endpoint", "v2_selected_reason"),
        ("selected_sha", "f" * 64, "v2_selected_sha"),
    ],
)
def test_v2_parent_rejects_inconsistent_selected_checkpoint(
    tmp_path: Path,
    field: str,
    value: object,
    failure: str,
) -> None:
    config, receipt_path, checkpoint_path = _write_v2_parent_artifacts(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    if field == "selected_reason":
        receipt["selected_checkpoint"]["reason"] = value
    elif field == "selected_sha":
        receipt["selected_checkpoint"]["sha256"] = value
    else:
        receipt[field] = value
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    config["initialization"]["parent"]["receipt_sha256"] = TRAINER.sha256_file(
        receipt_path
    )
    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(ValueError, match=failure):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


def test_v2_parent_rejects_an_earlier_confirmed_row(tmp_path: Path) -> None:
    config, receipt_path, checkpoint_path = _write_v2_parent_artifacts(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    earlier = deepcopy(receipt["history"][0])
    earlier["step"] = 3_000
    receipt["history"].insert(0, earlier)
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    config["initialization"]["parent"]["receipt_sha256"] = TRAINER.sha256_file(
        receipt_path
    )
    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(ValueError, match="v2_no_earlier_confirmed_checkpoint"):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


def test_cross_width_parent_rejects_mismatched_checkpoint_file(tmp_path: Path) -> None:
    config, receipt_path, checkpoint_path = _write_bound_parent_artifacts(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    payload["step"] = 21_000
    torch.save(payload, checkpoint_path)
    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


def test_cross_width_parent_rejects_mismatched_receipt_content(tmp_path: Path) -> None:
    config, receipt_path, checkpoint_path = _write_bound_parent_artifacts(tmp_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["arm"] = "B063"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    config["initialization"]["parent"]["receipt_sha256"] = TRAINER.sha256_file(
        receipt_path
    )
    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(ValueError, match="invalid warm-start parent receipt:.*arm"):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


def test_cross_width_parent_rejects_state_dtype_change(tmp_path: Path) -> None:
    config, receipt_path, checkpoint_path = _write_bound_parent_artifacts(tmp_path)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    first_key = next(iter(payload["state_dict"]))
    payload["state_dict"][first_key] = payload["state_dict"][first_key].double()
    torch.save(payload, checkpoint_path)
    parent = config["initialization"]["parent"]
    parent["checkpoint_sha256"] = TRAINER.sha256_file(checkpoint_path)
    receipt = json.loads(receipt_path.read_text())
    receipt["checkpoints"][0]["sha256"] = parent["checkpoint_sha256"]
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    parent["receipt_sha256"] = TRAINER.sha256_file(receipt_path)
    arm = TRAINER.validate_config(config, "B127")
    with pytest.raises(
        ValueError,
        match="checkpoint_state_signature|state_keys_shapes_dtypes",
    ):
        TRAINER.validate_warm_start_parent(
            receipt_path,
            checkpoint_path,
            "B127",
            arm,
            config,
        )


def test_confirmation_gate_and_first_pass_freeze_are_strict() -> None:
    config = json.loads(
        (ROOT / "research/v02/configs/bridge_l256_b127.json").read_text()
    )
    tiers = _exact_rollout_rows([4, 5, 6, 7], 512)
    small_prime = _exact_small_prime_rows(256, 64)
    gate = TRAINER.confirmed_gate(tiers, small_prime, config, True, 512)
    assert gate["passed"] is True

    first = TRAINER.freeze_first_confirmed_checkpoint(
        None,
        {"step": 3_000, "path": "first.pt", "sha256": "a" * 64},
        {"gate": gate},
    )
    assert first is not None
    later = TRAINER.freeze_first_confirmed_checkpoint(
        first,
        {"step": 6_000, "path": "later.pt", "sha256": "b" * 64},
        {"gate": gate},
    )
    assert later == first

    broken = deepcopy(tiers)
    broken["7"]["dynamic"]["total"] = 511
    assert TRAINER.confirmed_gate(broken, small_prime, config, True, 512)[
        "passed"
    ] is False
