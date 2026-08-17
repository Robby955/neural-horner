#!/usr/bin/env python3
"""Receipt-bound NeuralHorner v0.2 width-compression trainer.

This is a new experiment derived from the recovered production ``scale.py``
driver. It preserves the transition generator, loss, optimizer, schedule, and
rollout evaluator while making architecture and provenance explicit. It is not
a byte-for-byte replay of the historical v8 run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pathlib
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import torch
from torch import nn


LEGACY_CONFIG_SCHEMA = "neuralhorner-v02-scale-config-v1"
CONFIG_SCHEMA = "neuralhorner-v02-scale-config-v2"
SUPPORTED_CONFIG_SCHEMAS = {LEGACY_CONFIG_SCHEMA, CONFIG_SCHEMA}
LEGACY_RECEIPT_SCHEMA = "neuralhorner-v02-scale-receipt-v1"
RECEIPT_SCHEMA = "neuralhorner-v02-scale-receipt-v2"
SUPPORTED_PARENT_RECEIPT_SCHEMAS = {LEGACY_RECEIPT_SCHEMA, RECEIPT_SCHEMA}
CHECKPOINT_SCHEMA = "neuralhorner-v02-training-checkpoint-v1"
ROOT = pathlib.Path(__file__).resolve().parents[2]
SOURCE_PROVENANCE = pathlib.Path(__file__).with_name("source_provenance.json")

# (minimum modulus bits, maximum modulus bits, operand bits)
TIERS = {
    1: (1, 3, 32),
    2: (4, 8, 48),
    3: (9, 16, 64),
    4: (17, 32, 96),
    5: (33, 64, 128),
    6: (65, 128, 256),
    7: (129, 256, 512),
    8: (257, 512, 1024),
    9: (513, 1024, 2048),
    10: (1025, 2048, 4096),
}

_MASK32 = (1 << 32) - 1
_SMALL_PRIMES = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derived_seed(master_seed: int, domain: str) -> int:
    payload = f"nh02|{domain}|{master_seed}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


def is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def is_git_sha(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 40
        and all(character in "0123456789abcdef" for character in value)
    )


class Cell(nn.Module):
    """The v8 transition-cell topology with explicit depth/direction controls."""

    def __init__(
        self,
        dmodel: int = 96,
        hidden: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()
        self.in_proj = nn.Linear(3, dmodel)
        self.d_emb = nn.Embedding(2, dmodel)
        self.gru = nn.GRU(
            dmodel,
            hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )
        directions = 2 if bidirectional else 1
        self.head = nn.Linear(directions * hidden, 1)

    def forward(self, feat: torch.Tensor, digit: torch.Tensor) -> torch.Tensor:
        embedded = self.in_proj(feat) + self.d_emb(digit)[:, None, :]
        hidden, _ = self.gru(embedded)
        return self.head(hidden).squeeze(-1)


def parameter_count(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def _to_bits_small(values: torch.Tensor, width: int) -> torch.Tensor:
    shifts = torch.arange(width - 1, -1, -1, device=values.device)
    return (values[:, None] >> shifts[None, :]) & 1


def to_bits_limbs(
    integers: list[int],
    device: torch.device,
    width: int,
) -> torch.Tensor:
    """Convert arbitrary-width Python integers to an MSB-first bit tensor."""

    limb_count = (width + 31) // 32
    columns = []
    for limb_index in range(limb_count - 1, -1, -1):
        limb = torch.tensor(
            [(value >> (32 * limb_index)) & _MASK32 for value in integers],
            dtype=torch.int64,
            device=device,
        )
        columns.append(_to_bits_small(limb, 32))
    bits = torch.cat(columns, dim=1)
    excess = limb_count * 32 - width
    return bits[:, excess:] if excess else bits


def bitmat_operands(
    integers: list[int],
    width: int,
    device: torch.device,
) -> torch.Tensor:
    matrix = torch.zeros((len(integers), width), dtype=torch.long, device=device)
    for row, original in enumerate(integers):
        value = original
        position = width - 1
        while value > 0 and position >= 0:
            matrix[row, position] = value & 1
            value >>= 1
            position -= 1
    return matrix


def is_prime(value: int, rng: random.Random | None = None) -> bool:
    if value < 2:
        return False
    for prime in _SMALL_PRIMES:
        if value % prime == 0:
            return value == prime
    odd_part = value - 1
    exponent = 0
    while odd_part % 2 == 0:
        odd_part //= 2
        exponent += 1
    bases = list(_SMALL_PRIMES)
    if value >= 3_317_044_064_679_887_385_961_981:
        generator = rng or random.Random(0)
        bases.extend(generator.randrange(2, value - 1) for _ in range(20))
    for base in bases:
        witness = pow(base, odd_part, value)
        if witness in (1, value - 1):
            continue
        for _ in range(exponent - 1):
            witness = (witness * witness) % value
            if witness == value - 1:
                break
        else:
            return False
    return True


def primes_in_bits(
    low_bits: int,
    high_bits: int,
    rng: random.Random,
    count: int,
) -> list[int]:
    low = max(2, 1 << (low_bits - 1))
    high = (1 << high_bits) - 1
    found: set[int] = set()
    attempts = 0
    while len(found) < count and attempts < 500_000:
        candidate = rng.randint(low, high)
        if is_prime(candidate, rng):
            found.add(candidate)
        attempts += 1
    return sorted(found)


def sample_modulus(rng: random.Random, width: int) -> int:
    bit_length = rng.randint(1, width)
    low = max(2, 1 << (bit_length - 1))
    high = (1 << bit_length) - 1
    return rng.randint(low, high) if high >= low else 2


def sample_transition_values(
    batch_size: int,
    rng: random.Random,
    width: int,
) -> tuple[list[int], list[int], list[int], list[int], list[int]]:
    states: list[int] = []
    multiplicands: list[int] = []
    moduli: list[int] = []
    digits: list[int] = []
    targets: list[int] = []
    for _ in range(batch_size):
        modulus = sample_modulus(rng, width)
        state = rng.randrange(0, modulus)
        mixture_draw = rng.random()
        if mixture_draw < 0.35:
            multiplicand = 1
        elif mixture_draw < 0.45:
            multiplicand = 0
        else:
            multiplicand = rng.randrange(0, modulus)
        digit = rng.randrange(0, 2)
        if rng.random() < 0.5:
            margin = max(1, modulus // 64)
            for _ in range(40):
                candidate_state = rng.randrange(0, modulus)
                candidate_digit = rng.randrange(0, 2)
                result = (2 * candidate_state + candidate_digit * multiplicand) % modulus
                if result < margin or result >= modulus - margin:
                    state = candidate_state
                    digit = candidate_digit
                    break
        target = (2 * state + digit * multiplicand) % modulus
        states.append(state)
        multiplicands.append(multiplicand)
        moduli.append(modulus)
        digits.append(digit)
        targets.append(target)
    return states, multiplicands, moduli, digits, targets


def train_batch(
    batch_size: int,
    rng: random.Random,
    width: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    states, multiplicands, moduli, digits, targets = sample_transition_values(
        batch_size,
        rng,
        width,
    )
    features = torch.stack(
        [
            to_bits_limbs(states, device, width).float(),
            to_bits_limbs(multiplicands, device, width).float(),
            to_bits_limbs(moduli, device, width).float(),
        ],
        dim=-1,
    )
    digit_tensor = torch.tensor(digits, dtype=torch.long, device=device)
    target_tensor = to_bits_limbs(targets, device, width).float()
    return features, digit_tensor, target_tensor


@torch.no_grad()
def learned_step(
    model: Cell,
    state_bits: torch.Tensor,
    multiplicand_bits: torch.Tensor,
    modulus_bits: torch.Tensor,
    digits: torch.Tensor,
) -> torch.Tensor:
    features = torch.stack([state_bits, multiplicand_bits, modulus_bits], dim=-1)
    return (model(features, digits) > 0).float()


def bits_to_integers(bits: torch.Tensor) -> list[int]:
    values = []
    for row in bits.long().cpu().tolist():
        value = 0
        for bit in row:
            value = 2 * value + bit
        values.append(value)
    return values


@torch.no_grad()
def eval_tier(
    model: Cell,
    tier: int,
    rng: random.Random,
    width: int,
    device: torch.device,
    count: int,
    batch_size: int,
    width_mode: str,
) -> dict[str, Any]:
    low_bits, high_bits, operand_bits = TIERS[tier]
    primes = primes_in_bits(low_bits, high_bits, rng, count=6)
    if not primes:
        raise RuntimeError(f"could not generate evaluation primes for tier {tier}")
    correct = 0
    total = 0
    failures: list[dict[str, int]] = []
    sequence_widths: set[int] = set()
    case_hasher = hashlib.sha256()
    edges = ((0, 5), (5, 0), (1, 7), (7, 1))
    while total < count:
        current_batch = min(batch_size, count - total)
        moduli: list[int] = []
        left_operands: list[int] = []
        right_operands: list[int] = []
        truth: list[int] = []
        for _ in range(current_batch):
            modulus = rng.choice(primes)
            if rng.random() < 0.05:
                left, right = rng.choice(edges)
            else:
                left = rng.randrange(0, 1 << operand_bits)
                right = rng.randrange(0, 1 << operand_bits)
            moduli.append(modulus)
            left_operands.append(left)
            right_operands.append(right)
            truth.append((left * right) % modulus)
            case_hasher.update(
                (
                    canonical_json(
                        {"a": left, "b": right, "p": modulus, "truth": truth[-1]}
                    )
                    + "\n"
                ).encode()
            )

        if width_mode == "fixed":
            sequence_width = width
        elif width_mode == "dynamic":
            maximum_bits = max(modulus.bit_length() for modulus in moduli)
            sequence_width = min(
                width,
                max(32, ((maximum_bits + 31) // 32) * 32),
            )
        else:
            raise ValueError(f"unsupported evaluation width mode: {width_mode}")
        sequence_widths.add(sequence_width)
        modulus_bits = to_bits_limbs(moduli, device, sequence_width).float()
        left_bits = bitmat_operands(left_operands, operand_bits, device)
        right_bits = bitmat_operands(right_operands, operand_bits, device)
        ones = to_bits_limbs([1] * current_batch, device, sequence_width).float()

        state = torch.zeros((current_batch, sequence_width), device=device)
        for position in range(operand_bits):
            state = learned_step(
                model,
                state,
                ones,
                modulus_bits,
                left_bits[:, position],
            )
        left_residue = state

        state = torch.zeros((current_batch, sequence_width), device=device)
        for position in range(operand_bits):
            state = learned_step(
                model,
                state,
                ones,
                modulus_bits,
                right_bits[:, position],
            )
        right_residue = state

        state = torch.zeros((current_batch, sequence_width), device=device)
        for position in range(sequence_width):
            state = learned_step(
                model,
                state,
                left_residue,
                modulus_bits,
                right_residue[:, position].long(),
            )
        predictions = bits_to_integers(state)
        for index, prediction in enumerate(predictions):
            if prediction == truth[index]:
                correct += 1
            elif len(failures) < 8:
                failures.append(
                    {
                        "a": left_operands[index],
                        "b": right_operands[index],
                        "p": moduli[index],
                        "expected": truth[index],
                        "predicted": prediction,
                    }
                )
        total += current_batch
    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "width_mode": width_mode,
        "sequence_widths": sorted(sequence_widths),
        "prime_sha256": sha256_bytes(canonical_json(primes).encode()),
        "case_manifest_sha256": case_hasher.hexdigest(),
        "first_failures": failures,
    }


@torch.no_grad()
def eval_small_prime_exhaustive(
    model: Cell,
    sequence_width: int,
    device: torch.device,
    prime_limit: int,
    batch_size: int,
) -> dict[str, Any]:
    cases = [
        (state, multiplicand, modulus, digit)
        for modulus in range(2, prime_limit)
        if is_prime(modulus)
        for state in range(modulus)
        for multiplicand in range(modulus)
        for digit in (0, 1)
    ]
    correct = 0
    failures: list[dict[str, int]] = []
    for start in range(0, len(cases), batch_size):
        batch = cases[start : start + batch_size]
        states = [case[0] for case in batch]
        multiplicands = [case[1] for case in batch]
        moduli = [case[2] for case in batch]
        digits = [case[3] for case in batch]
        targets = [
            (2 * state + digit * multiplicand) % modulus
            for state, multiplicand, modulus, digit in batch
        ]
        features = torch.stack(
            [
                to_bits_limbs(states, device, sequence_width).float(),
                to_bits_limbs(multiplicands, device, sequence_width).float(),
                to_bits_limbs(moduli, device, sequence_width).float(),
            ],
            dim=-1,
        )
        digit_tensor = torch.tensor(digits, dtype=torch.long, device=device)
        predictions = bits_to_integers((model(features, digit_tensor) > 0).float())
        for index, prediction in enumerate(predictions):
            if prediction == targets[index]:
                correct += 1
            elif len(failures) < 16:
                state, multiplicand, modulus, digit = batch[index]
                failures.append(
                    {
                        "s": state,
                        "x": multiplicand,
                        "p": modulus,
                        "d": digit,
                        "expected": targets[index],
                        "predicted": prediction,
                    }
                )
    total = len(cases)
    return {
        "prime_limit_exclusive": prime_limit,
        "sequence_width": sequence_width,
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
        "first_failures": failures,
    }


def initialization_spec(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") == LEGACY_CONFIG_SCHEMA:
        return {"mode": "scratch"}
    initialization = config.get("initialization")
    if not isinstance(initialization, dict):
        raise ValueError("v2 config requires an initialization object")
    return initialization


def selection_spec(config: dict[str, Any]) -> dict[str, Any]:
    if config.get("schema") == LEGACY_CONFIG_SCHEMA:
        return {
            "mode": "endpoint",
            "screen_n": config["checkpoint_eval_n"],
            "confirmation_n": config["final_eval_n"],
        }
    selection = config.get("selection")
    if not isinstance(selection, dict):
        raise ValueError("v2 config requires a selection object")
    return selection


def normalize_minimum_correct_by_tier(
    value: Any,
    required_tiers: list[int],
    expected_count: int,
    *,
    field_name: str,
) -> dict[str, int]:
    exact = {str(tier): expected_count for tier in required_tiers}
    if value is None:
        return exact
    if not isinstance(value, dict):
        raise ValueError(f"{field_name} must be an object keyed by tier")
    if set(value) != set(exact):
        raise ValueError(f"{field_name} must name exactly the declared tiers {sorted(exact)}")
    normalized: dict[str, int] = {}
    for tier in required_tiers:
        key = str(tier)
        minimum = value[key]
        if type(minimum) is not int or not 1 <= minimum <= expected_count:
            raise ValueError(f"{field_name}[{key}] must be an integer in [1, {expected_count}]")
        normalized[key] = minimum
    return normalized


def selection_minimum_correct_by_tier(
    config: dict[str, Any],
    phase: str,
) -> dict[str, int]:
    if phase not in {"screen", "confirmation"}:
        raise ValueError(f"unsupported selection phase: {phase!r}")
    selection = selection_spec(config)
    count_field = f"{phase}_n"
    threshold_field = f"{phase}_minimum_correct_by_tier"
    return normalize_minimum_correct_by_tier(
        selection.get(threshold_field),
        config["tiers"],
        selection[count_field],
        field_name=f"selection.{threshold_field}",
    )


def runtime_policy_spec(config: dict[str, Any]) -> dict[str, Any] | None:
    if config.get("schema") == LEGACY_CONFIG_SCHEMA:
        return None
    policy = config.get("runtime_policy")
    if not isinstance(policy, dict):
        raise ValueError("v2 config requires a runtime_policy object")
    return policy


def state_dict_signature(state_dict: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {
        name: {"shape": list(value.shape), "dtype": str(value.dtype)}
        for name, value in state_dict.items()
    }


def state_dict_signature_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    return sha256_bytes(canonical_json(state_dict_signature(state_dict)).encode())


def validate_config(config: dict[str, Any], arm_name: str) -> dict[str, Any]:
    if config.get("schema") not in SUPPORTED_CONFIG_SCHEMAS:
        raise ValueError(f"unsupported config schema: {config.get('schema')!r}")
    required_positive = (
        "width",
        "steps",
        "batch_size",
        "eval_every",
        "checkpoint_eval_n",
        "final_eval_n",
        "eval_batch_size",
        "small_prime_limit",
    )
    for key in required_positive:
        if not isinstance(config.get(key), int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    tiers = config.get("tiers")
    if not isinstance(tiers, list) or not tiers:
        raise ValueError("tiers must be a nonempty list")
    for tier in tiers:
        if tier not in TIERS:
            raise ValueError(f"unsupported tier: {tier}")
        if TIERS[tier][1] > config["width"]:
            raise ValueError(f"tier {tier} exceeds configured width")
    width_modes = config.get("evaluation_width_modes")
    if (
        not isinstance(width_modes, list)
        or len(width_modes) != 2
        or set(width_modes) != {"fixed", "dynamic"}
    ):
        raise ValueError("evaluation_width_modes must contain fixed and dynamic")
    arms = config.get("arms")
    if not isinstance(arms, dict) or arm_name not in arms:
        raise ValueError(f"unknown arm: {arm_name}")
    arm = arms[arm_name]
    if not isinstance(arm, dict) or not isinstance(arm.get("architecture"), dict):
        raise ValueError(f"arm {arm_name} must contain an architecture")
    architecture = arm["architecture"]
    for key in ("dmodel", "hidden", "num_layers"):
        if not isinstance(architecture.get(key), int) or architecture[key] <= 0:
            raise ValueError(f"arm {arm_name} field {key} must be positive")
    if not isinstance(architecture.get("bidirectional"), bool):
        raise ValueError(f"arm {arm_name} field bidirectional must be boolean")
    if not isinstance(arm.get("expected_parameters"), int) or arm["expected_parameters"] <= 0:
        raise ValueError(f"arm {arm_name} expected_parameters must be positive")
    constructed_parameters = parameter_count(Cell(**architecture))
    if constructed_parameters != arm["expected_parameters"]:
        raise ValueError(
            f"arm {arm_name} parameter mismatch: expected "
            f"{arm['expected_parameters']}, constructed {constructed_parameters}"
        )
    predecessor = arm.get("predecessor")
    if predecessor is not None and predecessor not in arms:
        raise ValueError(f"arm {arm_name} names unknown predecessor {predecessor}")
    if not isinstance(config.get("require_final_gate"), bool):
        raise ValueError("require_final_gate must be boolean")
    if not isinstance(config.get("allow_warm_start"), bool):
        raise ValueError("allow_warm_start must be boolean")
    if not isinstance(config.get("deterministic_algorithms"), bool):
        raise ValueError("deterministic_algorithms must be boolean")
    if config["small_prime_limit"] > 1 << config["width"]:
        raise ValueError("small-prime limit exceeds configured width")

    initialization = initialization_spec(config)
    initialization_mode = initialization.get("mode")
    if initialization_mode not in {"scratch", "warm_start_only"}:
        raise ValueError(f"unsupported initialization mode: {initialization_mode!r}")
    if initialization_mode == "scratch":
        if config["allow_warm_start"]:
            raise ValueError("scratch initialization must forbid warm starts")
        if "parent" in initialization:
            raise ValueError("scratch initialization cannot declare a parent")
    else:
        if not config["allow_warm_start"]:
            raise ValueError("warm_start_only initialization must allow warm starts")
        if predecessor is not None:
            raise ValueError(
                "warm_start_only initialization cannot also use an arm predecessor"
            )
        parent = initialization.get("parent")
        if not isinstance(parent, dict):
            raise ValueError("warm_start_only initialization requires a parent")
        required_parent_strings = (
            "experiment",
            "arm",
            "receipt_sha256",
            "checkpoint_sha256",
            "state_dict_signature_sha256",
        )
        for key in required_parent_strings:
            if not isinstance(parent.get(key), str) or not parent[key]:
                raise ValueError(f"parent field {key} must be a nonempty string")
        for key in (
            "receipt_sha256",
            "checkpoint_sha256",
            "state_dict_signature_sha256",
        ):
            if not is_sha256(parent[key]):
                raise ValueError(f"parent field {key} must be a lowercase SHA-256")
        if "selected_sha256" in parent and not is_sha256(
            parent["selected_sha256"]
        ):
            raise ValueError(
                "parent field selected_sha256 must be a lowercase SHA-256"
            )
        for key in ("width", "parameters", "master_seed"):
            if not isinstance(parent.get(key), int) or parent[key] <= 0:
                raise ValueError(f"parent field {key} must be a positive integer")
        if (
            type(parent.get("checkpoint_step")) is not int
            or parent["checkpoint_step"] < 0
        ):
            raise ValueError("parent field checkpoint_step must be a nonnegative integer")
        if parent["width"] >= config["width"]:
            raise ValueError("warm-start parent width must be smaller than child width")
        if parent["arm"] != arm_name:
            raise ValueError("warm-start parent and child arm names must match")
        if parent["parameters"] != arm["expected_parameters"]:
            raise ValueError("warm-start parent parameter count must match child arm")
        if parent.get("architecture") != architecture:
            raise ValueError("warm-start parent architecture must match child arm")
        source = parent.get("source_identity")
        if not isinstance(source, dict):
            raise ValueError("warm-start parent requires source_identity")
        if not is_git_sha(source.get("git_head")):
            raise ValueError("parent source field git_head must be a full Git SHA")
        for key in (
            "trainer_sha256",
            "config_sha256",
            "source_provenance_sha256",
        ):
            if not is_sha256(source.get(key)):
                raise ValueError(f"parent source field {key} must be a lowercase SHA-256")
        gate = parent.get("required_gate")
        if not isinstance(gate, dict):
            raise ValueError("warm-start parent requires required_gate")
        gate_tiers = gate.get("tiers")
        if (
            not isinstance(gate_tiers, list)
            or not gate_tiers
            or len(gate_tiers) != len(set(gate_tiers))
        ):
            raise ValueError("parent gate tiers must be a nonempty unique list")
        for tier in gate_tiers:
            if tier not in TIERS:
                raise ValueError(f"parent gate names unsupported tier {tier!r}")
            if TIERS[tier][1] > parent["width"]:
                raise ValueError(f"parent gate tier {tier} exceeds parent width")
        gate_modes = gate.get("width_modes")
        if (
            not isinstance(gate_modes, list)
            or not gate_modes
            or len(gate_modes) != len(set(gate_modes))
            or not set(gate_modes).issubset({"fixed", "dynamic"})
        ):
            raise ValueError("parent gate width_modes must be a unique supported list")
        for key in ("rollout_n", "small_prime_limit"):
            if not isinstance(gate.get(key), int) or gate[key] <= 0:
                raise ValueError(f"parent gate field {key} must be positive")
        normalize_minimum_correct_by_tier(
            gate.get("minimum_correct_by_tier"),
            gate_tiers,
            gate["rollout_n"],
            field_name="parent.required_gate.minimum_correct_by_tier",
        )
        if gate["small_prime_limit"] > 1 << parent["width"]:
            raise ValueError("parent gate small-prime limit exceeds parent width")

    selection = selection_spec(config)
    selection_mode = selection.get("mode")
    if selection_mode not in {"endpoint", "first_confirmed_pass"}:
        raise ValueError(f"unsupported selection mode: {selection_mode!r}")
    for key in ("screen_n", "confirmation_n"):
        if not isinstance(selection.get(key), int) or selection[key] <= 0:
            raise ValueError(f"selection field {key} must be a positive integer")
    if selection["screen_n"] != config["checkpoint_eval_n"]:
        raise ValueError("selection screen_n must equal checkpoint_eval_n")
    if selection["confirmation_n"] != config["final_eval_n"]:
        raise ValueError("selection confirmation_n must equal final_eval_n")
    selection_minimum_correct_by_tier(config, "screen")
    selection_minimum_correct_by_tier(config, "confirmation")
    if selection_mode == "first_confirmed_pass":
        if config.get("schema") != CONFIG_SCHEMA:
            raise ValueError("first_confirmed_pass requires a v2 config")
        if selection.get("require_small_prime_exhaustive") is not True:
            raise ValueError(
                "first_confirmed_pass must require the small-prime exhaustive gate"
            )
    evaluate_step_zero = selection.get("evaluate_step_zero", False)
    if not isinstance(evaluate_step_zero, bool):
        raise ValueError("selection evaluate_step_zero must be boolean")
    if evaluate_step_zero and (
        selection_mode != "first_confirmed_pass"
        or initialization_mode != "warm_start_only"
    ):
        raise ValueError(
            "selection evaluate_step_zero requires a warm-start first-confirmed-pass run"
        )

    runtime_policy = runtime_policy_spec(config)
    if runtime_policy is not None:
        expected_runtime_policy = {
            "device": "cuda",
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
            "float32_matmul_precision": "highest",
        }
        if runtime_policy != expected_runtime_policy:
            raise ValueError(
                "v2 canonical runtime_policy does not match the frozen CUDA policy"
            )
        if config.get("source_policy") != {"require_clean_git": True}:
            raise ValueError("v2 config must require a clean Git worktree")
    return arm


def validate_predecessor_receipt(
    path: pathlib.Path | None,
    arm_name: str,
    arm: dict[str, Any],
    config: dict[str, Any],
    source_identity: dict[str, Any],
    seeds: dict[str, int],
    environment: dict[str, Any],
) -> dict[str, Any] | None:
    expected = arm["predecessor"]
    if expected is None:
        if path is not None:
            raise ValueError(f"arm {arm_name} has no predecessor receipt")
        return None
    if path is None:
        raise ValueError(f"arm {arm_name} requires a passing {expected} predecessor receipt")
    resolved = path.resolve()
    receipt = json.loads(resolved.read_text())
    checks = {
        "schema": receipt.get("schema") in SUPPORTED_PARENT_RECEIPT_SCHEMAS,
        "status": receipt.get("status") == "completed_pass",
        "arm": receipt.get("arm") == expected,
        "experiment": receipt.get("experiment") == config["name"],
        "gate": receipt.get("final_gate", {}).get("passed") is True,
        "config_sha256": receipt.get("source_identity", {}).get("config_sha256")
        == source_identity["config_sha256"],
        "trainer_sha256": receipt.get("source_identity", {}).get("trainer_sha256")
        == source_identity["trainer_sha256"],
        "scratch": receipt.get("warm_start") is None,
        "seeds": receipt.get("seeds") == seeds,
        "torch": receipt.get("environment", {}).get("torch") == environment["torch"],
        "device": receipt.get("environment", {}).get("device") == environment["device"],
        "platform": receipt.get("environment", {}).get("platform")
        == environment["platform"],
        "python": receipt.get("environment", {}).get("python") == environment["python"],
        "machine": receipt.get("environment", {}).get("machine") == environment["machine"],
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(
            f"invalid predecessor receipt for arm {arm_name}: {', '.join(failed)}"
        )
    return {"path": str(resolved), "sha256": sha256_file(resolved), "arm": expected}


def rollout_gate(
    tiers: dict[str, Any],
    required_tiers: list[int],
    width_modes: list[str],
    expected_count: int,
    minimum_correct_by_tier: dict[str, int] | None = None,
) -> dict[str, Any]:
    thresholds = normalize_minimum_correct_by_tier(
        minimum_correct_by_tier,
        required_tiers,
        expected_count,
        field_name="rollout minimum_correct_by_tier",
    )
    failures: list[str] = []
    all_exact = True
    for tier in required_tiers:
        tier_result = tiers.get(str(tier))
        if not isinstance(tier_result, dict):
            failures.append(f"tier_{tier}_missing")
            all_exact = False
            continue
        manifests: list[str] = []
        prime_sets: list[str] = []
        for width_mode in width_modes:
            result = tier_result.get(width_mode)
            if not isinstance(result, dict):
                failures.append(f"tier_{tier}_{width_mode}_missing")
                all_exact = False
                continue
            if result.get("total") != expected_count:
                failures.append(f"tier_{tier}_{width_mode}_denominator")
                all_exact = False
            correct = result.get("correct")
            if type(correct) is not int or not 0 <= correct <= expected_count:
                failures.append(f"tier_{tier}_{width_mode}_invalid_correct")
                all_exact = False
            else:
                if correct < thresholds[str(tier)]:
                    failure = (
                        "not_exact"
                        if thresholds[str(tier)] == expected_count
                        else "below_threshold"
                    )
                    failures.append(f"tier_{tier}_{width_mode}_{failure}")
                if correct != expected_count:
                    all_exact = False
            if isinstance(result.get("case_manifest_sha256"), str):
                manifests.append(result["case_manifest_sha256"])
            else:
                failures.append(f"tier_{tier}_{width_mode}_case_manifest")
            if isinstance(result.get("prime_sha256"), str):
                prime_sets.append(result["prime_sha256"])
            else:
                failures.append(f"tier_{tier}_{width_mode}_prime_manifest")
        if len(manifests) != len(width_modes) or len(set(manifests)) != 1:
            failures.append(f"tier_{tier}_case_pairing")
        if len(prime_sets) != len(width_modes) or len(set(prime_sets)) != 1:
            failures.append(f"tier_{tier}_prime_pairing")
    return {
        "required_tiers": required_tiers,
        "width_modes": width_modes,
        "expected_count": expected_count,
        "minimum_correct_by_tier": thresholds,
        "failures": sorted(set(failures)),
        "exact": not failures and all_exact,
        "passed": not failures,
    }


def small_prime_gate(
    results: dict[str, Any] | None,
    width: int,
    prime_limit: int,
) -> dict[str, Any]:
    failures: list[str] = []
    expected_total = sum(
        2 * prime * prime
        for prime in range(2, prime_limit)
        if is_prime(prime)
    )
    expected_widths = {"fixed": width, "dynamic": min(width, 32)}
    if not isinstance(results, dict):
        failures.append("missing")
    else:
        for width_mode, expected_width in expected_widths.items():
            result = results.get(width_mode)
            if not isinstance(result, dict):
                failures.append(f"{width_mode}_missing")
                continue
            if result.get("prime_limit_exclusive") != prime_limit:
                failures.append(f"{width_mode}_prime_limit")
            if result.get("sequence_width") != expected_width:
                failures.append(f"{width_mode}_sequence_width")
            if result.get("total") != expected_total:
                failures.append(f"{width_mode}_denominator")
            if result.get("correct") != expected_total:
                failures.append(f"{width_mode}_not_exact")
    return {
        "prime_limit_exclusive": prime_limit,
        "expected_total": expected_total,
        "expected_sequence_widths": expected_widths,
        "failures": sorted(set(failures)),
        "passed": not failures,
    }


def confirmed_gate(
    tiers: dict[str, Any],
    small_prime_results: dict[str, Any] | None,
    config: dict[str, Any],
    parameters_finite: bool,
    expected_count: int,
) -> dict[str, Any]:
    rollout = rollout_gate(
        tiers,
        config["tiers"],
        config["evaluation_width_modes"],
        expected_count,
        selection_minimum_correct_by_tier(config, "confirmation"),
    )
    small_prime = small_prime_gate(
        small_prime_results,
        config["width"],
        config["small_prime_limit"],
    )
    return {
        "parameters_finite": parameters_finite,
        "rollout": rollout,
        "small_prime": small_prime,
        "rollout_threshold_passed": rollout["passed"],
        "rollout_exact": rollout["exact"],
        "small_prime_exact": small_prime["passed"],
        "passed": parameters_finite and rollout["passed"] and small_prime["passed"],
    }


def freeze_first_confirmed_checkpoint(
    selected: dict[str, Any] | None,
    checkpoint: dict[str, Any],
    confirmation: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if selected is not None:
        return selected
    if confirmation is None or confirmation.get("gate", {}).get("passed") is not True:
        return None
    return {
        **checkpoint,
        "reason": "first_confirmed_pass",
        "confirmation_gate": confirmation["gate"],
    }


def _parent_source_matches(
    source_identity: dict[str, Any],
    expected: dict[str, str],
) -> bool:
    return (
        source_identity.get("trainer_sha256") == expected["trainer_sha256"]
        and source_identity.get("config_sha256") == expected["config_sha256"]
        and source_identity.get("source_provenance_sha256")
        == expected["source_provenance_sha256"]
        and source_identity.get("git", {}).get("head") == expected["git_head"]
    )


def validate_warm_start_parent(
    receipt_path: pathlib.Path | None,
    checkpoint_path: pathlib.Path | None,
    arm_name: str,
    arm: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any] | None, dict[str, torch.Tensor] | None]:
    initialization = initialization_spec(config)
    if initialization["mode"] == "scratch":
        if receipt_path is not None or checkpoint_path is not None:
            raise ValueError("scratch initialization rejects parent artifacts")
        return None, None

    if receipt_path is None or checkpoint_path is None:
        raise ValueError(
            "warm_start_only initialization requires --parent-receipt and "
            "--warm-start"
        )
    parent = initialization["parent"]
    resolved_receipt = receipt_path.resolve()
    resolved_checkpoint = checkpoint_path.resolve()
    receipt_sha = sha256_file(resolved_receipt)
    checkpoint_sha = sha256_file(resolved_checkpoint)
    if receipt_sha != parent["receipt_sha256"]:
        raise ValueError("parent receipt SHA-256 does not match the frozen config")
    if checkpoint_sha != parent["checkpoint_sha256"]:
        raise ValueError("parent checkpoint SHA-256 does not match the frozen config")

    receipt = json.loads(resolved_receipt.read_text())
    receipt_schema = receipt.get("schema")
    receipt_source = receipt.get("source_identity", {})
    expected_source = parent["source_identity"]
    receipt_config = receipt.get("config", {})
    if not isinstance(receipt_config, dict):
        receipt_config = {}
    receipt_architecture = receipt.get("architecture")
    required_gate = parent["required_gate"]
    required_gate_thresholds = normalize_minimum_correct_by_tier(
        required_gate.get("minimum_correct_by_tier"),
        required_gate["tiers"],
        required_gate["rollout_n"],
        field_name="parent.required_gate.minimum_correct_by_tier",
    )

    def confirmation_evidence_passes(row: dict[str, Any]) -> bool:
        confirmation = row.get("confirmation")
        if not isinstance(confirmation, dict):
            return False
        tiers = confirmation.get("tiers")
        if not isinstance(tiers, dict):
            return False
        try:
            rollout = rollout_gate(
                tiers,
                required_gate["tiers"],
                required_gate["width_modes"],
                required_gate["rollout_n"],
                required_gate_thresholds,
            )
        except ValueError:
            return False
        small_prime = small_prime_gate(
            confirmation.get("small_prime_exhaustive"),
            parent["width"],
            required_gate["small_prime_limit"],
        )
        return (
            row.get("parameters_finite") is True
            and rollout["passed"]
            and small_prime["passed"]
        )

    history = receipt.get("history", [])
    if not isinstance(history, list):
        history = []
    receipt_checkpoints = receipt.get("checkpoints", [])
    if not isinstance(receipt_checkpoints, list):
        receipt_checkpoints = []
    matching_rows = [
        row
        for row in history
        if isinstance(row, dict)
        and type(row.get("step")) is int
        and row["step"] == parent["checkpoint_step"]
    ]
    checkpoint_entries = [
        entry
        for entry in receipt_checkpoints
        if isinstance(entry, dict)
        and type(entry.get("step")) is int
        and entry["step"] == parent["checkpoint_step"]
    ]
    receipt_checks = {
        "schema": receipt_schema in SUPPORTED_PARENT_RECEIPT_SCHEMAS,
        "step_zero_receipt_schema": (
            parent["checkpoint_step"] != 0 or receipt_schema == RECEIPT_SCHEMA
        ),
        "status": receipt.get("status") == "completed_pass",
        "experiment": receipt.get("experiment") == parent["experiment"],
        "arm": receipt.get("arm") == parent["arm"] == arm_name,
        "architecture": receipt_architecture == parent["architecture"]
        == arm["architecture"],
        "parameters": receipt.get("parameters") == parent["parameters"]
        == arm["expected_parameters"],
        "width": receipt_config.get("width") == parent["width"],
        "master_seed": receipt_config.get("master_seed") == parent["master_seed"],
        "gate_tiers_declared": receipt_config.get("tiers")
        == required_gate["tiers"],
        "gate_width_modes_declared": receipt_config.get("evaluation_width_modes")
        == required_gate["width_modes"],
        "gate_rollout_n_declared": receipt_config.get("final_eval_n")
        == required_gate["rollout_n"],
        "gate_small_prime_limit_declared": receipt_config.get("small_prime_limit")
        == required_gate["small_prime_limit"],
        "source": _parent_source_matches(receipt_source, expected_source),
        "environment_recorded": isinstance(receipt.get("environment"), dict),
        "deterministic_parent": receipt.get("environment", {}).get(
            "deterministic_algorithms_enabled"
        )
        is True,
        "final_gate": receipt.get("final_gate", {}).get("passed") is True,
        "history_row": len(matching_rows) == 1,
        "checkpoint_entry": len(checkpoint_entries) == 1,
    }
    if len(checkpoint_entries) == 1:
        receipt_checks["checkpoint_listed_sha"] = (
            checkpoint_entries[0].get("sha256") == parent["checkpoint_sha256"]
        )
    selected_checkpoint = receipt.get("selected_checkpoint")
    selected_confirmation_gate = None
    gate_tiers = None
    gate_small_prime = None
    gate_parameters_finite = False
    if len(matching_rows) == 1:
        row = matching_rows[0]
        if receipt_schema == LEGACY_RECEIPT_SCHEMA:
            gate_tiers = row.get("tiers")
            gate_small_prime = row.get("small_prime_exhaustive")
            gate_parameters_finite = row.get("parameters_finite") is True
        elif receipt_schema == RECEIPT_SCHEMA:
            confirmation = row.get("confirmation")
            receipt_checks["v2_confirmation"] = isinstance(confirmation, dict)
            if isinstance(confirmation, dict):
                gate_tiers = confirmation.get("tiers")
                gate_small_prime = confirmation.get("small_prime_exhaustive")
                selected_confirmation_gate = confirmation.get("gate")
                receipt_checks["v2_confirmation_gate"] = (
                    isinstance(selected_confirmation_gate, dict)
                    and selected_confirmation_gate.get("passed") is True
                )
            gate_parameters_finite = row.get("parameters_finite") is True
    if receipt_schema == RECEIPT_SCHEMA:
        selection = receipt.get("selection")
        history_steps = [
            row.get("step") for row in history if isinstance(row, dict)
        ]
        checkpoint_steps = [
            entry.get("step")
            for entry in receipt_checkpoints
            if isinstance(entry, dict)
        ]
        history_steps_valid = all(type(step) is int for step in history_steps)
        checkpoint_steps_valid = all(
            type(step) is int for step in checkpoint_steps
        )
        receipt_checks["v2_selection_mode"] = (
            isinstance(selection, dict)
            and selection.get("mode") == "first_confirmed_pass"
            and selection == receipt_config.get("selection")
        )
        receipt_checks["v2_step_zero_declared"] = (
            parent["checkpoint_step"] != 0
            or (
                isinstance(selection, dict)
                and selection.get("evaluate_step_zero") is True
            )
        )
        receipt_checks["v2_confirmation_count_declared"] = (
            isinstance(selection, dict)
            and selection.get("confirmation_n") == required_gate["rollout_n"]
        )
        try:
            receipt_thresholds = normalize_minimum_correct_by_tier(
                (
                    selection.get("confirmation_minimum_correct_by_tier")
                    if isinstance(selection, dict)
                    else None
                ),
                required_gate["tiers"],
                required_gate["rollout_n"],
                field_name="parent receipt confirmation thresholds",
            )
        except ValueError:
            receipt_thresholds = None
        receipt_checks["v2_confirmation_thresholds_declared"] = (
            receipt_thresholds == required_gate_thresholds
        )
        receipt_checks["v2_selected_checkpoint"] = isinstance(
            selected_checkpoint, dict
        )
        receipt_checks["v2_stopped_at_first_confirmed_pass"] = (
            receipt.get("stopped_at_first_confirmed_pass") is True
        )
        receipt_checks["v2_steps_completed"] = (
            type(receipt.get("steps_completed")) is int
            and receipt["steps_completed"] == parent["checkpoint_step"]
        )
        receipt_checks["v2_history_order"] = (
            history_steps_valid
            and history_steps == sorted(history_steps)
            and len(history_steps) == len(set(history_steps))
        )
        receipt_checks["v2_checkpoint_order"] = (
            checkpoint_steps_valid
            and checkpoint_steps == sorted(checkpoint_steps)
            and len(checkpoint_steps) == len(set(checkpoint_steps))
        )
        receipt_checks["v2_selected_is_last_history"] = (
            bool(history_steps) and history_steps[-1] == parent["checkpoint_step"]
        )
        receipt_checks["v2_selected_is_last_checkpoint"] = (
            bool(checkpoint_steps)
            and checkpoint_steps[-1] == parent["checkpoint_step"]
        )
        receipt_checks["v2_no_earlier_confirmed_checkpoint"] = not any(
            isinstance(row, dict)
            and (
                confirmation_evidence_passes(row)
                or (
                    isinstance(row.get("confirmation"), dict)
                    and isinstance(row["confirmation"].get("gate"), dict)
                    and row["confirmation"]["gate"].get("passed") is True
                )
            )
            for row in history
            if isinstance(row, dict)
            and type(row.get("step")) is int
            and row["step"] < parent["checkpoint_step"]
        )
        if isinstance(selected_checkpoint, dict):
            receipt_checks["v2_selected_step"] = (
                type(selected_checkpoint.get("step")) is int
                and selected_checkpoint["step"] == parent["checkpoint_step"]
            )
            receipt_checks["v2_selected_sha"] = (
                selected_checkpoint.get("sha256") == parent["checkpoint_sha256"]
            )
            receipt_checks["v2_selected_reason"] = (
                selected_checkpoint.get("reason") == "first_confirmed_pass"
            )
            if len(checkpoint_entries) == 1:
                receipt_checks["v2_selected_path"] = (
                    selected_checkpoint.get("path")
                    == checkpoint_entries[0].get("path")
                )
            receipt_checks["v2_selected_gate_consistency"] = (
                isinstance(selected_confirmation_gate, dict)
                and selected_checkpoint.get("confirmation_gate")
                == selected_confirmation_gate
                == receipt.get("final_gate")
            )
        expected_selected_sha = parent.get("selected_sha256")
        if expected_selected_sha is not None:
            selected_artifact_path = resolved_receipt.parent / "SELECTED.json"
            receipt_checks["v2_selected_artifact_exists"] = (
                selected_artifact_path.is_file()
            )
            if selected_artifact_path.is_file():
                receipt_checks["v2_selected_artifact_sha"] = (
                    sha256_file(selected_artifact_path) == expected_selected_sha
                )
                try:
                    selected_artifact = json.loads(selected_artifact_path.read_text())
                except (OSError, json.JSONDecodeError):
                    selected_artifact = None
                receipt_checks["v2_selected_artifact_matches_receipt"] = (
                    selected_artifact == selected_checkpoint
                )
    elif receipt_schema == LEGACY_RECEIPT_SCHEMA:
        receipt_checks["legacy_exact_gate_contract"] = all(
            minimum == required_gate["rollout_n"]
            for minimum in required_gate_thresholds.values()
        )
    if isinstance(gate_tiers, dict):
        parent_rollout = rollout_gate(
            gate_tiers,
            required_gate["tiers"],
            required_gate["width_modes"],
            required_gate["rollout_n"],
            required_gate_thresholds,
        )
        parent_small_prime = small_prime_gate(
            gate_small_prime,
            parent["width"],
            required_gate["small_prime_limit"],
        )
        receipt_checks["bound_rollout_gate"] = parent_rollout["passed"]
        receipt_checks["bound_small_prime_gate"] = parent_small_prime["passed"]
        receipt_checks["finite_parameters"] = gate_parameters_finite
        if receipt_schema == RECEIPT_SCHEMA:
            stored_gate = selected_confirmation_gate
            stored_rollout = (
                stored_gate.get("rollout", {})
                if isinstance(stored_gate, dict)
                else {}
            )
            stored_small_prime = (
                stored_gate.get("small_prime", {})
                if isinstance(stored_gate, dict)
                else {}
            )
            exact_contract = all(
                minimum == required_gate["rollout_n"]
                for minimum in required_gate_thresholds.values()
            )
            stored_threshold_passed = (
                stored_gate.get("rollout_threshold_passed")
                if isinstance(stored_gate, dict)
                else None
            )
            if stored_threshold_passed is None and exact_contract:
                stored_threshold_passed = stored_rollout.get("passed")
            recomputed_passed = (
                gate_parameters_finite
                and parent_rollout["passed"]
                and parent_small_prime["passed"]
            )
            receipt_checks["v2_stored_gate_semantics"] = (
                isinstance(stored_gate, dict)
                and stored_gate.get("parameters_finite") is gate_parameters_finite
                and stored_threshold_passed is parent_rollout["passed"]
                and stored_gate.get("rollout_exact") is parent_rollout["exact"]
                and stored_gate.get("small_prime_exact")
                is parent_small_prime["passed"]
                and stored_gate.get("passed") is recomputed_passed
                and stored_rollout.get("passed") is parent_rollout["passed"]
                and all(
                    stored_rollout.get(key) == parent_rollout[key]
                    for key in (
                        "required_tiers",
                        "width_modes",
                        "expected_count",
                        "failures",
                    )
                )
                and stored_small_prime == parent_small_prime
            )
            if not exact_contract:
                receipt_checks["v2_explicit_threshold_gate_fields"] = (
                    isinstance(stored_gate, dict)
                    and "rollout_threshold_passed" in stored_gate
                    and stored_rollout.get("minimum_correct_by_tier")
                    == required_gate_thresholds
                    and stored_rollout.get("exact") is parent_rollout["exact"]
                )
    elif len(matching_rows) == 1:
        receipt_checks["bound_gate_evidence"] = False
    failed_receipt_checks = [
        name for name, passed in receipt_checks.items() if not passed
    ]
    if failed_receipt_checks:
        raise ValueError(
            "invalid warm-start parent receipt: "
            + ", ".join(failed_receipt_checks)
        )

    payload = torch.load(resolved_checkpoint, map_location="cpu", weights_only=True)
    state_dict = payload.get("state_dict")
    if not isinstance(state_dict, dict) or not all(
        isinstance(value, torch.Tensor) for value in state_dict.values()
    ):
        raise ValueError("parent checkpoint does not contain a tensor state_dict")
    expected_model = Cell(**arm["architecture"])
    expected_signature = state_dict_signature(expected_model.state_dict())
    actual_signature = state_dict_signature(state_dict)
    expected_signature_sha = sha256_bytes(canonical_json(expected_signature).encode())
    actual_signature_sha = sha256_bytes(canonical_json(actual_signature).encode())
    checkpoint_source = payload.get("source_identity", {})
    experiment_config = payload.get("experiment_config", {})
    experiment_selection = experiment_config.get("selection", {})
    try:
        checkpoint_thresholds = normalize_minimum_correct_by_tier(
            (
                experiment_selection.get("confirmation_minimum_correct_by_tier")
                if isinstance(experiment_selection, dict)
                else None
            ),
            required_gate["tiers"],
            required_gate["rollout_n"],
            field_name="parent checkpoint confirmation thresholds",
        )
    except ValueError:
        checkpoint_thresholds = None
    checkpoint_checks = {
        "schema": payload.get("schema") == CHECKPOINT_SCHEMA,
        "arm": payload.get("arm") == parent["arm"] == arm_name,
        "width": payload.get("L") == parent["width"],
        "step": (
            type(payload.get("step")) is int
            and payload["step"] == parent["checkpoint_step"]
        ),
        "architecture": payload.get("config") == parent["architecture"],
        "experiment": experiment_config.get("name") == parent["experiment"],
        "experiment_width": experiment_config.get("width") == parent["width"],
        "experiment_gate_tiers": experiment_config.get("tiers")
        == required_gate["tiers"],
        "experiment_gate_width_modes": experiment_config.get(
            "evaluation_width_modes"
        )
        == required_gate["width_modes"],
        "experiment_gate_rollout_n": experiment_config.get("final_eval_n")
        == required_gate["rollout_n"],
        "experiment_gate_thresholds": (
            receipt_schema == LEGACY_RECEIPT_SCHEMA
            and all(
                minimum == required_gate["rollout_n"]
                for minimum in required_gate_thresholds.values()
            )
        )
        or (receipt_schema == RECEIPT_SCHEMA and checkpoint_thresholds == required_gate_thresholds),
        "experiment_gate_small_prime_limit": experiment_config.get(
            "small_prime_limit"
        )
        == required_gate["small_prime_limit"],
        "experiment_step_zero_declared": (
            parent["checkpoint_step"] != 0
            or (
                isinstance(experiment_selection, dict)
                and experiment_selection.get("evaluate_step_zero") is True
            )
        ),
        "experiment_arm": experiment_config.get("arms", {})
        .get(parent["arm"], {})
        .get("architecture")
        == parent["architecture"],
        "source": _parent_source_matches(checkpoint_source, expected_source),
        "source_receipt_pair": checkpoint_source == receipt_source,
        "expected_state_signature": expected_signature_sha
        == parent["state_dict_signature_sha256"],
        "checkpoint_state_signature": actual_signature_sha
        == parent["state_dict_signature_sha256"],
        "state_keys_shapes_dtypes": actual_signature == expected_signature,
    }
    failed_checkpoint_checks = [
        name for name, passed in checkpoint_checks.items() if not passed
    ]
    if failed_checkpoint_checks:
        raise ValueError(
            "invalid warm-start parent checkpoint: "
            + ", ".join(failed_checkpoint_checks)
        )

    identity = {
        "mode": "warm_start_only",
        "parent_receipt": {
            "path": str(resolved_receipt),
            "sha256": receipt_sha,
            "schema": receipt["schema"],
            "experiment": parent["experiment"],
            "arm": parent["arm"],
            "source_identity": receipt_source,
            "environment": receipt["environment"],
            "selected_checkpoint": selected_checkpoint,
        },
        "parent_checkpoint": {
            "path": str(resolved_checkpoint),
            "sha256": checkpoint_sha,
            "schema": payload["schema"],
            "width": payload["L"],
            "step": payload["step"],
            "state_dict_signature_sha256": actual_signature_sha,
        },
        "transfer": {
            "model_state_loaded": True,
            "optimizer_state_loaded": False,
            "scheduler_state_loaded": False,
            "rng_state_loaded": False,
            "parent_environment_continuity_required": False,
        },
    }
    return identity, state_dict


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported device: {requested}")
    return torch.device(requested)


def runtime_settings_identity() -> dict[str, Any]:
    return {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def configure_canonical_runtime(
    config: dict[str, Any],
    requested_device: str,
) -> dict[str, Any] | None:
    policy = runtime_policy_spec(config)
    if policy is None:
        return None
    if requested_device != policy["device"]:
        raise ValueError(
            f"canonical config requires explicit --device {policy['device']}"
        )
    actual_workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    if actual_workspace != policy["cublas_workspace_config"]:
        raise RuntimeError(
            "canonical CUDA run requires CUBLAS_WORKSPACE_CONFIG="
            f"{policy['cublas_workspace_config']} before Python starts"
        )
    torch.backends.cudnn.benchmark = policy["cudnn_benchmark"]
    torch.backends.cudnn.deterministic = policy["cudnn_deterministic"]
    torch.backends.cuda.matmul.allow_tf32 = policy["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = policy["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(policy["float32_matmul_precision"])
    actual_settings = runtime_settings_identity()
    expected_settings = {
        key: value for key, value in policy.items() if key != "device"
    }
    if actual_settings != expected_settings:
        raise RuntimeError("failed to apply the frozen canonical CUDA settings")
    return policy


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def git_identity() -> dict[str, Any]:
    def run(*arguments: str) -> str:
        return subprocess.check_output(
            ["git", *arguments],
            cwd=ROOT,
            text=True,
        ).strip()

    return {
        "head": run("rev-parse", "HEAD"),
        "branch": run("branch", "--show-current"),
        "status": run("status", "--porcelain=v1", "--untracked-files=all").splitlines(),
    }


def enforce_source_policy(
    config: dict[str, Any],
    source_identity: dict[str, Any],
) -> None:
    policy = config.get("source_policy")
    if policy is None:
        return
    if policy.get("require_clean_git") is True and source_identity.get("git", {}).get(
        "status"
    ):
        raise RuntimeError(
            "canonical run requires a clean Git worktree so HEAD contains the "
            "trainer and config bytes"
        )
    bound_paths = (
        ("trainer_path", "trainer_sha256"),
        ("config_path", "config_sha256"),
        ("source_provenance_path", "source_provenance_sha256"),
    )
    for path_key, sha_key in bound_paths:
        recorded_path = source_identity.get(path_key)
        if not isinstance(recorded_path, str) or pathlib.Path(recorded_path).is_absolute():
            raise RuntimeError(
                f"canonical source {path_key} must be tracked inside the repository"
            )
        try:
            head_bytes = subprocess.check_output(
                ["git", "show", f"HEAD:{recorded_path}"],
                cwd=ROOT,
            )
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"canonical source {recorded_path} is not present at HEAD"
            ) from error
        if sha256_bytes(head_bytes) != source_identity.get(sha_key):
            raise RuntimeError(
                f"canonical source {recorded_path} differs from its HEAD bytes"
            )


def receipt_source_path(path: pathlib.Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def nvidia_driver_versions() -> list[str]:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError("could not record the NVIDIA driver version") from error
    versions = sorted({line.strip() for line in output.splitlines() if line.strip()})
    if not versions:
        raise RuntimeError("nvidia-smi returned no NVIDIA driver version")
    return versions


def environment_identity(device: torch.device) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version,
        "torch": torch.__version__,
        "device": str(device),
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        "runtime_settings": runtime_settings_identity(),
    }
    if device.type == "cuda":
        cuda_runtime = torch.version.cuda
        cudnn_version = torch.backends.cudnn.version()
        if cuda_runtime is None:
            raise RuntimeError("CUDA device selected but PyTorch has no CUDA runtime")
        if cudnn_version is None:
            raise RuntimeError("CUDA device selected but cuDNN version is unavailable")
        properties = torch.cuda.get_device_properties(device)
        identity["cuda_runtime"] = cuda_runtime
        identity["cuda_driver_versions"] = nvidia_driver_versions()
        identity["cudnn_version"] = cudnn_version
        identity["cudnn_enabled"] = torch.backends.cudnn.enabled
        identity["cuda_device_count"] = torch.cuda.device_count()
        identity["cuda_current_device"] = torch.cuda.current_device()
        identity["device_name"] = properties.name
        identity["device_capability"] = [properties.major, properties.minor]
        identity["device_total_memory"] = properties.total_memory
        identity["device_multiprocessor_count"] = properties.multi_processor_count
    return identity


def rng_state(data_rng: random.Random, device: torch.device) -> dict[str, Any]:
    state: dict[str, Any] = {
        "python_data": data_rng.getstate(),
        "torch_cpu": torch.get_rng_state(),
    }
    if device.type == "cuda":
        state["torch_cuda_all"] = torch.cuda.get_rng_state_all()
    elif device.type == "mps" and hasattr(torch.mps, "get_rng_state"):
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def all_parameters_finite(model: nn.Module) -> bool:
    return all(torch.isfinite(parameter).all().item() for parameter in model.parameters())


def write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def save_checkpoint(
    path: pathlib.Path,
    model: Cell,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    data_rng: random.Random,
    device: torch.device,
    config: dict[str, Any],
    arm_name: str,
    arm: dict[str, Any],
    step: int,
    source_identity: dict[str, Any],
) -> str:
    checkpoint = {
        "schema": CHECKPOINT_SCHEMA,
        "state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "rng_state": rng_state(data_rng, device),
        "config": arm,
        "experiment_config": config,
        "arm": arm_name,
        "L": config["width"],
        "step": step,
        "source_identity": source_identity,
    }
    torch.save(checkpoint, path)
    return sha256_file(path)


def evaluate_rollouts(
    model: Cell,
    config: dict[str, Any],
    seeds: dict[str, int],
    device: torch.device,
    count: int,
) -> dict[str, Any]:
    tiers: dict[str, Any] = {}
    for tier in config["tiers"]:
        tiers[str(tier)] = {}
        tier_seed = derived_seed(seeds["evaluation"], f"tier_{tier}")
        for width_mode in config["evaluation_width_modes"]:
            tiers[str(tier)][width_mode] = eval_tier(
                model,
                tier,
                random.Random(tier_seed),
                config["width"],
                device,
                count,
                config["eval_batch_size"],
                width_mode,
            )
    return tiers


def evaluate_small_primes(
    model: Cell,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    return {
        "fixed": eval_small_prime_exhaustive(
            model,
            config["width"],
            device,
            config["small_prime_limit"],
            config["eval_batch_size"],
        ),
        "dynamic": eval_small_prime_exhaustive(
            model,
            min(config["width"], 32),
            device,
            config["small_prime_limit"],
            config["eval_batch_size"],
        ),
    }


def run(args: argparse.Namespace) -> int:
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text())
    arm = validate_config(config, args.arm)
    architecture = arm["architecture"]
    if args.resume is not None:
        raise ValueError(
            "resume is not implemented; warm_start_only loads model weights and "
            "deliberately resets optimizer, scheduler, and RNG state"
        )
    output = args.out.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing nonempty output directory: {output}")

    runtime_policy = configure_canonical_runtime(config, args.device)
    torch.use_deterministic_algorithms(config["deterministic_algorithms"])
    device = resolve_device(args.device)
    if runtime_policy is not None and device.type != runtime_policy["device"]:
        raise RuntimeError("resolved device violates the canonical runtime policy")
    source_path = pathlib.Path(__file__).resolve()
    source_identity = {
        "trainer_path": receipt_source_path(source_path),
        "trainer_sha256": sha256_file(source_path),
        "config_path": receipt_source_path(config_path),
        "config_sha256": sha256_file(config_path),
        "source_provenance_path": receipt_source_path(SOURCE_PROVENANCE),
        "source_provenance_sha256": sha256_file(SOURCE_PROVENANCE),
        "git": git_identity(),
    }
    enforce_source_policy(config, source_identity)
    seeds = {
        "master": config["master_seed"],
        "initialization": derived_seed(config["master_seed"], "initialization"),
        "training_data": derived_seed(config["master_seed"], "training_data"),
        "evaluation": derived_seed(config["master_seed"], "evaluation"),
    }
    environment = environment_identity(device)

    torch.manual_seed(seeds["initialization"])
    data_rng = random.Random(seeds["training_data"])
    predecessor_identity = validate_predecessor_receipt(
        args.predecessor_receipt,
        args.arm,
        arm,
        config,
        source_identity,
        seeds,
        environment,
    )
    parent_identity, parent_state_dict = validate_warm_start_parent(
        args.parent_receipt,
        args.warm_start,
        args.arm,
        arm,
        config,
    )
    model = Cell(**architecture)
    if parent_state_dict is not None:
        load_result = model.load_state_dict(parent_state_dict, strict=True)
        if load_result.missing_keys or load_result.unexpected_keys:
            raise ValueError("validated parent state failed strict model loading")
    model = model.to(device)
    params = parameter_count(model)
    if params != arm["expected_parameters"]:
        raise RuntimeError(
            f"arm {args.arm} parameter mismatch: expected "
            f"{arm['expected_parameters']}, constructed {params}"
        )
    warm_start_identity = (
        None if parent_identity is None else parent_identity["parent_checkpoint"]
    )

    output.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["peak_lr"],
        weight_decay=config["weight_decay"],
    )
    warmup = max(1, int(config["warmup_fraction"] * config["steps"]))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, config["steps"] - warmup)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        minimum = config["minimum_lr_fraction"]
        return minimum + (1 - minimum) * cosine

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    loss_function = nn.BCEWithLogitsLoss()
    started_at = datetime.now(timezone.utc).isoformat()
    history: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    selection = selection_spec(config)
    screen_thresholds = selection_minimum_correct_by_tier(config, "screen")
    confirmation_thresholds = selection_minimum_correct_by_tier(config, "confirmation")
    selected_checkpoint: dict[str, Any] | None = None
    meta = {
        "schema": RECEIPT_SCHEMA,
        "status": "running",
        "experiment": config["name"],
        "role": config["role"],
        "arm": args.arm,
        "architecture": architecture,
        "parameters": params,
        "config": config,
        "seeds": seeds,
        "source_identity": source_identity,
        "initialization": (
            {"mode": "scratch"} if parent_identity is None else parent_identity
        ),
        "warm_start": warm_start_identity,
        "resume": None,
        "predecessor_receipt": predecessor_identity,
        "runtime_policy": runtime_policy,
        "environment": environment,
        "started_at": started_at,
        "history": history,
        "checkpoints": checkpoints,
        "selection": selection,
        "selected_checkpoint": selected_checkpoint,
        "final_gate": None,
    }
    write_json(output / "receipt.json", meta)
    print(
        "=== NeuralHorner v0.2 === "
        + canonical_json(
            {
                "experiment": config["name"],
                "arm": args.arm,
                "device": str(device),
                "width": config["width"],
                "steps": config["steps"],
                "batch": config["batch_size"],
                "parameters": params,
            }
        ),
        flush=True,
    )

    started = time.perf_counter()
    last_loss = math.nan
    first_step = 0 if selection.get("evaluate_step_zero", False) else 1
    for step in range(first_step, config["steps"] + 1):
        if step > 0:
            model.train()
            features, digits, targets = train_batch(
                config["batch_size"],
                data_rng,
                config["width"],
                device,
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

            if step % config["eval_every"] != 0 and step != config["steps"]:
                continue

        synchronize(device)
        model.eval()
        parameters_finite = all_parameters_finite(model)
        confirmation = None
        if selection["mode"] == "first_confirmed_pass":
            tiers = evaluate_rollouts(
                model,
                config,
                seeds,
                device,
                selection["screen_n"],
            )
            screen_gate = rollout_gate(
                tiers,
                config["tiers"],
                config["evaluation_width_modes"],
                selection["screen_n"],
                screen_thresholds,
            )
            screen_gate["parameters_finite"] = parameters_finite
            screen_gate["passed"] = screen_gate["passed"] and parameters_finite
            small_prime = None
            if screen_gate["passed"]:
                confirmation_tiers = evaluate_rollouts(
                    model,
                    config,
                    seeds,
                    device,
                    selection["confirmation_n"],
                )
                small_prime = evaluate_small_primes(model, config, device)
                confirmation_gate = confirmed_gate(
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
        else:
            evaluation_count = (
                selection["confirmation_n"]
                if step == config["steps"]
                else selection["screen_n"]
            )
            tiers = evaluate_rollouts(
                model,
                config,
                seeds,
                device,
                evaluation_count,
            )
            screen_gate = rollout_gate(
                tiers,
                config["tiers"],
                config["evaluation_width_modes"],
                evaluation_count,
                (
                    confirmation_thresholds
                    if evaluation_count == selection["confirmation_n"]
                    else screen_thresholds
                ),
            )
            screen_gate["parameters_finite"] = parameters_finite
            screen_gate["passed"] = screen_gate["passed"] and parameters_finite
            small_prime = (
                evaluate_small_primes(model, config, device)
                if step == config["steps"]
                else None
            )
        synchronize(device)
        row = {
            "step": step,
            "loss": None if step == 0 else last_loss,
            "learning_rate": optimizer.param_groups[0]["lr"],
            "elapsed_s": round(time.perf_counter() - started, 3),
            "tiers": tiers,
            "screen_gate": screen_gate,
            "confirmation": confirmation,
            "small_prime_exhaustive": small_prime,
            "parameters_finite": parameters_finite,
        }
        history.append(row)
        checkpoint_path = output / f"weights_step{step}.pt"
        checkpoint_sha = save_checkpoint(
            checkpoint_path,
            model,
            optimizer,
            scheduler,
            data_rng,
            device,
            config,
            args.arm,
            architecture,
            step,
            source_identity,
        )
        checkpoint_entry = {
            "step": step,
            "path": checkpoint_path.name,
            "sha256": checkpoint_sha,
        }
        checkpoints.append(checkpoint_entry)
        selected_checkpoint = freeze_first_confirmed_checkpoint(
            selected_checkpoint,
            checkpoint_entry,
            confirmation,
        )
        meta["history"] = history
        meta["checkpoints"] = checkpoints
        meta["selected_checkpoint"] = selected_checkpoint
        write_json(output / "receipt.json", meta)
        tier_summary = " ".join(
            f"t{tier}/{mode}={tiers[str(tier)][mode]['correct']}/"
            f"{tiers[str(tier)][mode]['total']}"
            for tier in config["tiers"]
            for mode in config["evaluation_width_modes"]
        )
        loss_summary = "not_run" if row["loss"] is None else f"{row['loss']:.6f}"
        print(
            f"step={step} loss={loss_summary} {tier_summary} "
            f"elapsed_s={row['elapsed_s']}",
            flush=True,
        )
        if selected_checkpoint is not None:
            break

    final = history[-1]
    if selection["mode"] == "first_confirmed_pass":
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
    else:
        final_gate = confirmed_gate(
            final["tiers"],
            final["small_prime_exhaustive"],
            config,
            final["parameters_finite"],
            selection["confirmation_n"],
        )
    if final_gate["passed"]:
        meta["status"] = "completed_pass"
    elif config["require_final_gate"]:
        meta["status"] = "completed_failed_gate"
    else:
        meta["status"] = "completed_diagnostic"
    meta["finished_at"] = datetime.now(timezone.utc).isoformat()
    meta["elapsed_s"] = round(time.perf_counter() - started, 3)
    meta["steps_completed"] = final["step"]
    meta["stopped_at_first_confirmed_pass"] = (
        selection["mode"] == "first_confirmed_pass"
        and selected_checkpoint is not None
    )
    meta["selected_checkpoint"] = selected_checkpoint
    meta["final_gate"] = final_gate
    write_json(output / "receipt.json", meta)
    if selected_checkpoint is not None:
        write_json(output / "SELECTED.json", selected_checkpoint)
    marker = "DONE" if not config["require_final_gate"] or final_gate["passed"] else "FAILED_GATE"
    (output / marker).write_text(meta["status"] + "\n")
    print("=== completed === " + canonical_json(final_gate), flush=True)
    return 0 if not config["require_final_gate"] or final_gate["passed"] else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=pathlib.Path, required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--warm-start", type=pathlib.Path)
    parser.add_argument("--parent-receipt", type=pathlib.Path)
    parser.add_argument("--predecessor-receipt", type=pathlib.Path)
    parser.add_argument("--resume", type=pathlib.Path)
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
