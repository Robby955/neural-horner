from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_fixed_hamming_four_way.py"
SPEC = importlib.util.spec_from_file_location("fixed_hamming_validator", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
VALIDATOR = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(ROOT / "scripts"))
try:
    SPEC.loader.exec_module(VALIDATOR)
finally:
    sys.path.pop(0)


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (
            (True, True, True, False),
            "weight_schedule_interaction_required",
        ),
        ((True, False, True, False), "direct_schedule_sufficient"),
        ((True, True, False, False), "repaired_weights_sufficient"),
    ],
)
def test_classify_truth_table(values, expected) -> None:
    assert VALIDATOR.classify_truth_table(values) == expected


def test_classify_truth_table_rejects_unplanned_pattern() -> None:
    with pytest.raises(VALIDATOR.ValidationError, match="unclassified"):
        VALIDATOR.classify_truth_table((False, False, False, False))


def divergence() -> dict:
    return {
        "phase": "scan_operand",
        "phase_step": 149,
        "global_step": 4245,
        "input_bit": 1,
        "state": "10",
        "x": "20",
        "p": "23",
        "expected_value": "7",
        "pre_mod_value": "40",
        "wrong_output_bit_count": 1,
        "minimum_signed_target_logit_margin": -2.0,
        "minimum_margin_bit_index": 7,
        "modulus_subtract_count": "1",
        "boundary_distance_to_nearest_modulus_multiple": "7",
        "double_carry_out": False,
        "add_after_truncated_double_carry_out": True,
        "full_pre_mod_word_overflow_count": "1",
    }


def test_teacher_signature_ignores_model_specific_failure_severity() -> None:
    first = divergence()
    second = {**first, "wrong_output_bit_count": 7}
    second["minimum_signed_target_logit_margin"] = -13.0

    assert VALIDATOR.teacher_divergence_signature(first) == (
        VALIDATOR.teacher_divergence_signature(second)
    )


def test_divergence_summary_records_carry_and_hides_large_boundary() -> None:
    summary = VALIDATOR.summarize_divergence(divergence())

    assert summary is not None
    assert summary["phase"] == "scan_operand"
    assert summary["boundary_distance_bit_length"] == 3
    assert "boundary_distance" not in summary
    assert summary["add_after_truncated_double_carry_out"] is True
    assert len(summary["teacher_transition_sha256"]) == 64
