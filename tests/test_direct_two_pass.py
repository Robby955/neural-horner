from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BASELINE_DIR = ROOT / "model"
CANDIDATE_DIR = ROOT / "candidates" / "direct_two_pass"


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASELINE = load_module(BASELINE_DIR / "model.py", "neural_horner_baseline")
CANDIDATE = load_module(CANDIDATE_DIR / "model.py", "neural_horner_direct")


def bits_to_int(bits) -> int:
    value = 0
    for bit in bits:
        value = 2 * value + int(bit)
    return value


class ExactStepMixin:
    """Replace the learned cell with the recurrence it is trained to model."""

    def __init__(self, width: int = 64) -> None:
        super().__init__()
        self.L = width
        self.device = torch.device("cpu")
        self.step_calls = 0

    def _step(self, s_bits, x_bits, p_bits, d):
        self.step_calls += 1
        outputs = []
        for row in range(s_bits.shape[0]):
            s = bits_to_int(s_bits[row].long().tolist())
            x = bits_to_int(x_bits[row].long().tolist())
            p = bits_to_int(p_bits[row].long().tolist())
            digit = int(d[row].item())
            outputs.append((2 * s + digit * x) % p)
        return CANDIDATE.to_bits_limbs(
            outputs, self.device, s_bits.shape[1]
        ).float()


class ExactBaseline(ExactStepMixin, BASELINE.BitSerialReducer):
    pass


class ExactDirect(ExactStepMixin, CANDIDATE.DirectBitSerialReducer):
    pass


class PaddingSensitiveDirect(CANDIDATE.DirectBitSerialReducer):
    """Every active step increments state, exposing accidental padding steps."""

    def __init__(self) -> None:
        super().__init__()
        self.L = 64
        self.device = torch.device("cpu")

    def _step(self, s_bits, x_bits, p_bits, d):
        values = [bits_to_int(row.long().tolist()) + 1 for row in s_bits]
        return CANDIDATE.to_bits_limbs(
            values, self.device, s_bits.shape[1]
        ).float()


def encode(model, a: int, b: int, p: int):
    return (
        model.preprocess_a(a),
        model.preprocess_b(b),
        model.preprocess_p(p),
    )


def predict_int(model, a: int, b: int, p: int) -> int:
    return bits_to_int(model.predict_digits(*encode(model, a, b, p)))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_direct_schedule_computes_modular_product() -> None:
    model = ExactDirect(width=128)
    cases = [
        (0, 0, 2),
        (0, 91, 17),
        (1, 1, 2),
        (123456, 789012, 65537),
        ((1 << 120) + 3, (1 << 119) + 5, (1 << 61) - 1),
        ((1 << 255) - 1, (1 << 254) + 17, (1 << 61) - 1),
    ]

    for a, b, p in cases:
        assert predict_int(model, a, b, p) == (a * b) % p


def test_direct_schedule_removes_fixed_width_third_pass() -> None:
    a, b, p = (1 << 47) + 5, (1 << 39) + 3, (1 << 31) - 1
    baseline = ExactBaseline()
    direct = ExactDirect()

    assert predict_int(baseline, a, b, p) == (a * b) % p
    assert predict_int(direct, a, b, p) == (a * b) % p
    assert baseline.step_calls == a.bit_length() + b.bit_length() + 32
    assert direct.step_calls == a.bit_length() + b.bit_length()


def test_canonical_operand_order_is_swap_invariant() -> None:
    cases = [
        (0, 1),
        (17, 17),
        ((1 << 31) + 1, (1 << 63) + 3),
        ((1 << 64) + 9, (1 << 64) + 3),
    ]
    for a, b in cases:
        a_bits = CANDIDATE._bits_of(a)
        b_bits = CANDIDATE._bits_of(b)
        ordered = CANDIDATE._canonical_scan_reduce(a_bits, b_bits)
        swapped = CANDIDATE._canonical_scan_reduce(b_bits, a_bits)

        assert ordered == swapped
        assert (len(ordered[0]), ordered[0]) <= (len(ordered[1]), ordered[1])


def test_direct_batch_is_permutation_and_singleton_invariant() -> None:
    cases = [
        (1, 1, 2),
        ((1 << 7) + 3, (1 << 5) + 1, 127),
        ((1 << 34) + 7, (1 << 9) + 5, (1 << 31) - 1),
        ((1 << 63) + 9, (1 << 65) + 11, (1 << 61) - 1),
    ]
    model = ExactDirect(width=128)
    encoded = [encode(model, *case) for case in cases]
    singletons = [bits_to_int(model.predict_digits_batch([item])[0]) for item in encoded]
    batched = [bits_to_int(bits) for bits in model.predict_digits_batch(encoded)]
    reversed_batch = [
        bits_to_int(bits)
        for bits in model.predict_digits_batch(list(reversed(encoded)))
    ][::-1]
    truth = [(a * b) % p for a, b, p in cases]

    assert singletons == truth
    assert batched == singletons
    assert reversed_batch == singletons


def test_artificial_padding_is_not_applied_to_short_rows() -> None:
    model = PaddingSensitiveDirect()
    short = encode(model, 1, 1, 17)
    long = encode(model, (1 << 20) + 1, (1 << 18) + 1, 17)

    singleton = bits_to_int(model.predict_digits_batch([short])[0])
    batched = bits_to_int(model.predict_digits_batch([short, long])[0])

    assert singleton == 1
    assert batched == singleton


def test_fail_closed_bounds_and_output_order() -> None:
    model = ExactDirect(width=8)
    exact_limit = 1 << 15
    too_wide = 1 << 16
    inputs = [
        encode(model, 3, 5, 1),
        encode(model, exact_limit, 5, 127),
        encode(model, 3, 5, 1 << 8),
        encode(model, too_wide, 5, 127),
        encode(model, 7, 9, 127),
    ]
    outputs = model.predict_digits_batch(inputs)

    assert outputs[0] == [0]
    assert bits_to_int(outputs[1]) == (exact_limit * 5) % 127
    assert outputs[2] == [0]
    assert outputs[3] == [0]
    assert bits_to_int(outputs[4]) == (7 * 9) % 127
    assert model.predict_digits_batch([]) == []


def test_candidate_manifest_and_checkpoint_identity() -> None:
    manifest = json.loads((CANDIDATE_DIR / "manifest.json").read_text())

    assert manifest["entry_class"] == "model.DirectBitSerialReducer"
    assert "L=2048" in manifest["model_description"]
    assert "unchanged NeuralHorner v8 weights" in manifest["training_description"]
    assert sha256(CANDIDATE_DIR / "weights.pt") == sha256(
        BASELINE_DIR / "weights.pt"
    )
    assert sha256(CANDIDATE_DIR / "weights.pt") == (
        "294fbf809ad42bb0ac66dd51303d21ac5dddeca9a081b5790f1af39d36b21609"
    )
