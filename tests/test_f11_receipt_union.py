from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "validate_f11_receipts.py"
RECEIPT_DIR = ROOT / "candidates" / "direct_two_pass_l2sp_a0875" / "receipts"
CURRENT_RUNNER = "87e0b15e8a234ea03c43f5621d11fc1b83da5a1480d20efbc985c29ad22268f1"
CURRENT_SOURCES = "0ca78bb1fdfc797ff559aaca0f623351022f92fb0a6208d892050e8832dc6837"
CURRENT_ARTIFACT = "9b0ba4f1c6ff5ed8ccf7b64b5b173baf10b8881c407ce430ec3338adcc0e06fc"
CURRENT_CASE_SET = "54f9d2342fcdbfa6fc21a5d6a6560d13364059be3c48174a6c66a407834a86ee"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "receipt_union_validator_under_test",
        SCRIPT_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


VALIDATOR = load_module()


def current_full_receipts() -> list[dict]:
    names = (
        "f11_decisive_both_mps_fp32_routebound_v3.json",
        "f11_companions_both_mps_fp32_routebound_v3.json",
        "f11_ties_both_mps_fp32_routebound_v3.json",
        "f11_legacy_both_mps_fp32_routebound_v3.json",
    )
    return [json.loads((RECEIPT_DIR / name).read_text()) for name in names]


def validate(receipts: list[dict]):
    return VALIDATOR.validate_receipt_union(
        receipts,
        expected_groups={"decisive", "companions", "ties", "legacy"},
        expected_cases=20,
        expected_results=40,
        expected_transitions=193_116,
        expected_runner_sha256=CURRENT_RUNNER,
        expected_source_set_sha256=CURRENT_SOURCES,
        expected_artifact_set_sha256=CURRENT_ARTIFACT,
        expected_full_case_set_sha256=CURRENT_CASE_SET,
        verify_current_files=False,
    )


def test_current_v3_full_union_is_exact_and_unique() -> None:
    report = validate(current_full_receipts())

    assert report["status"] == "validated_exact"
    assert report["unique_cases"] == 20
    assert report["unique_oriented_cases"] == 40
    assert report["transitions"] == 193_116
    assert report["minimum_signed_target_logit_margin"] == 5.016554832458496
    assert report["worst_margin"]["label"] == "decisive-f11-x-1"


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (
            lambda receipt: receipt["qualification_source_identity"]["scorer"].update(
                status="checkout_not_discovered"
            ),
            "scorer is not verified",
        ),
        (
            lambda receipt: receipt.update(runner_sha256="0" * 64),
            "runner SHA-256 mismatch",
        ),
        (
            lambda receipt: receipt.update(artifact_set_sha256="1" * 64),
            "artifact-set digest is invalid",
        ),
        (
            lambda receipt: receipt["results"][0]["manifest_route_validation"].update(
                route_exact=False
            ),
            "result route is not exact",
        ),
        (
            lambda receipt: receipt["results"][0].update(margin_gate_passed=False),
            "result failed margin_gate_passed",
        ),
    ],
)
def test_union_rejects_identity_route_and_margin_failures(mutation, message) -> None:
    receipts = copy.deepcopy(current_full_receipts())
    mutation(receipts[0])

    with pytest.raises(VALIDATOR.ReceiptValidationError, match=message):
        validate(receipts)


def test_union_rejects_duplicate_selected_and_oriented_cases() -> None:
    receipts = copy.deepcopy(current_full_receipts())
    receipts[1]["selected_cases"][1]["case_sha256"] = receipts[1][
        "selected_cases"
    ][0]["case_sha256"]

    with pytest.raises(
        VALIDATOR.ReceiptValidationError,
        match="duplicates a selected case",
    ):
        validate(receipts)

    receipts = copy.deepcopy(current_full_receipts())
    receipts[1]["results"][1]["oriented_case_sha256"] = receipts[1]["results"][0][
        "oriented_case_sha256"
    ]
    with pytest.raises(
        VALIDATOR.ReceiptValidationError,
        match="oriented cases are not unique",
    ):
        validate(receipts)


def test_union_rejects_invalid_or_incomplete_orientation_matrix() -> None:
    receipts = copy.deepcopy(current_full_receipts())
    receipts[0]["results"][0]["orientation"] = "forged"

    with pytest.raises(
        VALIDATOR.ReceiptValidationError,
        match="invalid orientation",
    ):
        validate(receipts)

    receipts = copy.deepcopy(current_full_receipts())
    receipts[1]["results"][2]["case_sha256"] = receipts[1]["results"][0][
        "case_sha256"
    ]
    with pytest.raises(
        VALIDATOR.ReceiptValidationError,
        match="duplicates a case-orientation result|complete case-orientation matrix",
    ):
        validate(receipts)


def test_union_rejects_inconsistent_per_result_transition_sum() -> None:
    receipts = copy.deepcopy(current_full_receipts())
    receipts[0]["results"][0]["observed_transitions"] += 1

    with pytest.raises(
        VALIDATOR.ReceiptValidationError,
        match="result transition count mismatch|completed transition sum",
    ):
        validate(receipts)
