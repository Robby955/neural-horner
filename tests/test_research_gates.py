from __future__ import annotations

import importlib.util
import json
import random
import sys
from pathlib import Path

import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


TRACE = load_script("trace_direct_f11")
NO_SHORTCUT = load_script("verify_no_shortcut")
OFFICIAL = load_script("run_official_eval")
HELD_OUT = load_script("held_out_battery")


def test_trace_count_gates_reject_truncation_and_asymmetry() -> None:
    assert TRACE.orientation_output_count_exact(2, [[0], [1]])
    assert not TRACE.orientation_output_count_exact(2, [[0]])
    assert not TRACE.orientation_output_count_exact(2, [[0], [1], [0]])

    assert TRACE.paired_result_counts_exact(2, [{}, {}], [{}, {}])
    assert not TRACE.paired_result_counts_exact(2, [{}], [{}])
    assert not TRACE.paired_result_counts_exact(2, [{}, {}], [{}])


def test_trained_no_shortcut_gate_requires_every_check() -> None:
    checks = {
        "not_a_mod_p": True,
        "not_b_mod_p": True,
        "not_unreduced_residue_product": True,
        "exact_modular_product": True,
    }
    assert NO_SHORTCUT.passes_requested_gate(
        randomized=False,
        correct=4,
        total=4,
        max_randomized_correct=0,
        shortcut_checks=checks,
    )
    checks["not_a_mod_p"] = False
    assert not NO_SHORTCUT.passes_requested_gate(
        randomized=False,
        correct=4,
        total=4,
        max_randomized_correct=0,
        shortcut_checks=checks,
    )


def test_randomized_collapse_does_not_require_shortcut_checks() -> None:
    checks = {
        "not_a_mod_p": False,
        "not_b_mod_p": False,
        "not_unreduced_residue_product": False,
        "exact_modular_product": False,
    }
    assert NO_SHORTCUT.passes_requested_gate(
        randomized=True,
        correct=0,
        total=4,
        max_randomized_correct=0,
        shortcut_checks=checks,
    )
    assert not NO_SHORTCUT.passes_requested_gate(
        randomized=True,
        correct=1,
        total=4,
        max_randomized_correct=0,
        shortcut_checks=checks,
    )


def test_official_gate_accepts_uninstalled_source_tree_metadata(monkeypatch) -> None:
    def package_missing(_name: str):
        raise OFFICIAL.importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(OFFICIAL.importlib.metadata, "distribution", package_missing)
    monkeypatch.setattr(OFFICIAL.importlib.metadata, "version", package_missing)
    monkeypatch.delattr(OFFICIAL.modchallenge, "__version__", raising=False)

    assert OFFICIAL.installed_source_hint() is None
    assert OFFICIAL.scorer_package_version() == "source-tree-uninstalled"


def test_shortcut_case_is_distinguishing() -> None:
    p = 2_147_483_647
    a, b = NO_SHORTCUT.draw_distinguishing_case(random.Random(2026), p, 64)
    truth = (a * b) % p

    assert truth != a % p
    assert truth != b % p
    assert truth != (a % p) * (b % p)


class MockBatteryModel:
    def __init__(self, outputs) -> None:
        self.outputs = outputs

    @staticmethod
    def preprocess_a(value):
        return value

    @staticmethod
    def preprocess_b(value):
        return value

    @staticmethod
    def preprocess_p(value):
        return value

    def predict_digits_batch(self, _inputs):
        return self.outputs

    def max_batch_size(self):
        return 1024


def binary_digits(value: int) -> list[int]:
    return [int(digit) for digit in f"{value:b}"]


def decimal_digits(value: int) -> list[int]:
    return [int(digit) for digit in str(value)]


def test_frozen_battery_rejects_extra_missing_and_nonbinary_outputs() -> None:
    cases = [(2, 3), (4, 5)]
    p = 17
    exact = [binary_digits((a * b) % p) for a, b in cases]

    valid = HELD_OUT.run_cases(MockBatteryModel(exact), p, cases)
    assert valid["correct"] == valid["total"] == 2
    assert valid["output_count_exact"]

    extra = HELD_OUT.run_cases(
        MockBatteryModel([*exact, binary_digits(0)]),
        p,
        cases,
    )
    assert extra["correct"] == extra["total"] == 2
    assert not extra["output_count_exact"]
    assert extra["extra_output_count"] == 1

    missing = HELD_OUT.run_cases(MockBatteryModel(exact[:1]), p, cases)
    assert missing["correct"] == 1
    assert not missing["output_count_exact"]
    assert not missing["cases"][1]["output_present"]

    nonbinary = HELD_OUT.run_cases(
        MockBatteryModel([[2], exact[1]]),
        p,
        cases,
    )
    assert nonbinary["correct"] == 1
    assert not nonbinary["cases"][0]["output_alphabet_exact"]

    wrong_type = HELD_OUT.run_cases(
        MockBatteryModel([tuple(exact[0]), exact[1]]),
        p,
        cases,
    )
    assert wrong_type["correct"] == 1
    assert not wrong_type["cases"][0]["output_type_exact"]


class ContractBatteryModel:
    def __init__(self, *, batch_size: int, output_base: int | str = 2) -> None:
        self.batch_size = batch_size
        self.output_base = output_base
        self.preprocessed: list[tuple[str, str]] = []
        self.actual_batch_sizes: list[int] = []

    def preprocess_a(self, value):
        assert isinstance(value, str)
        self.preprocessed.append(("a", value))
        return value

    def preprocess_b(self, value):
        assert isinstance(value, str)
        self.preprocessed.append(("b", value))
        return value

    def preprocess_p(self, value):
        assert isinstance(value, str)
        self.preprocessed.append(("p", value))
        return value

    def max_batch_size(self):
        return self.batch_size

    def predict_digits_batch(self, inputs):
        self.actual_batch_sizes.append(len(inputs))
        outputs = []
        for a, b, p in inputs:
            answer = (int(a) * int(b)) % int(p)
            if self.output_base == 2:
                outputs.append(binary_digits(answer))
            elif self.output_base == 10:
                outputs.append(decimal_digits(answer))
            elif self.output_base == "p":
                outputs.append([answer])
            else:
                raise AssertionError("unsupported test base")
        return outputs


def test_frozen_battery_uses_decimal_preprocessing_and_scorer_batching() -> None:
    cases = [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)]
    model = ContractBatteryModel(batch_size=2)

    result = HELD_OUT.run_cases(model, 17, cases)

    assert result["correct"] == result["total"] == 5
    assert result["actual_batch_sizes"] == [2, 2, 1]
    assert model.actual_batch_sizes == [2, 2, 1]
    assert all(isinstance(value, str) for _, value in model.preprocessed)
    assert model.preprocessed[:3] == [("a", "2"), ("b", "3"), ("p", "17")]

    zero_declared = ContractBatteryModel(batch_size=0)
    declared, effective = HELD_OUT.scorer_batch_size(zero_declared)
    assert declared == 0
    assert effective == 1


def test_frozen_battery_uses_manifest_decimal_and_prime_bases() -> None:
    cases = [(123, 456), (987, 654)]
    p = 997
    for output_base in (10, "p"):
        model = ContractBatteryModel(batch_size=8, output_base=output_base)
        result = HELD_OUT.run_cases(
            model,
            p,
            cases,
            output_base=output_base,
        )
        assert result["correct"] == result["total"] == len(cases)
        assert all(case["malformed_output"] is None for case in result["cases"])


class GeneratorOuterModel(ContractBatteryModel):
    def predict_digits_batch(self, inputs):
        outputs = super().predict_digits_batch(inputs)
        return (output for output in outputs)


def test_frozen_battery_rejects_generator_outer_output() -> None:
    model = GeneratorOuterModel(batch_size=2)
    result = HELD_OUT.run_cases(model, 17, [(2, 3), (4, 5)])

    assert not result["outer_output_sized"]
    assert not result["batch_contract_exact"]
    assert not result["output_count_exact"]
    assert result["correct"] == 0


class WrongBatteryModel(ContractBatteryModel):
    def predict_digits_batch(self, inputs):
        self.actual_batch_sizes.append(len(inputs))
        return [[0] for _ in inputs]


def test_frozen_battery_require_exact_stops_after_first_failed_batch() -> None:
    model = WrongBatteryModel(batch_size=2)
    result = HELD_OUT.run_cases(
        model,
        17,
        [(2, 3), (4, 5), (6, 7), (8, 9), (10, 11)],
        stop_on_failure=True,
    )

    assert result["actual_batch_sizes"] == [2]
    assert result["attempted"] == 2
    assert result["stopped_early"]
    assert not result["output_count_exact"]


def test_oriented_fixture_hashes_and_duplicate_denominators() -> None:
    selected = [
        ("family-a", 17, [(2, 3), (2, 3)]),
        ("family-b", 17, [(3, 2)]),
    ]

    statistics = HELD_OUT.fixture_statistics(selected, ("original", "swapped"))

    assert statistics["base_row_count"] == 3
    assert statistics["base_unique_numerical_case_count"] == 2
    assert statistics["base_duplicate_numerical_row_count"] == 1
    assert statistics["oriented_row_count"] == 6
    assert statistics["oriented_unique_effective_numerical_case_count"] == 2
    assert statistics["oriented_effective_duplicate_row_count"] == 4
    assert statistics["oriented_unique_labelled_numerical_case_count"] == 4
    assert statistics["oriented_labelled_duplicate_row_count"] == 2
    assert set(statistics["oriented_fixture_sha256"]) == {"original", "swapped"}
    assert (
        statistics["base_fixture_sha256"]
        != statistics["combined_oriented_fixture_sha256"]
    )


def test_frozen_battery_family_keys_and_rng_fixture_are_stable() -> None:
    p, p2, operand_width, categories = HELD_OUT.generate_battery(8, 3)
    selected = [
        (name, p2 if name.startswith("prime=3mod4") else p, cases)
        for name, cases in categories.items()
    ]

    assert p == 89
    assert p2 == 251
    assert operand_width == 16
    assert list(categories) == [
        "fibonacci values",
        "fermat numbers",
        "alternating bits",
        "fixed Hamming weight W/2",
        "product straddles k*p",
        "prime=3mod4 (random ops)",
    ]
    assert HELD_OUT.canonical_json_sha256(HELD_OUT.fixture_rows(selected)) == (
        "b7ea84c432c1a9baed53dd5a7c0ca21f354196c9630510e5ed1d2afdfa14b361"
    )


def test_frozen_battery_main_detects_load_time_artifact_mutation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    manifest = {"entry_class": "model.Test", "output_base": 10}
    (submission / "manifest.json").write_text(json.dumps(manifest))
    (submission / "model.py").write_text("# test model\n")
    (submission / "weights.pt").write_bytes(b"before-load")
    receipt_path = tmp_path / "receipt.json"
    model = ContractBatteryModel(batch_size=2, output_base=10)
    model.L = 3
    model.device = torch.device("cpu")

    def mutating_load(path):
        assert Path(path) == submission
        (submission / "weights.pt").write_bytes(b"mutated-during-load")
        return submission, manifest, object(), model

    monkeypatch.setattr(HELD_OUT, "load_submission", mutating_load)
    monkeypatch.setattr(
        HELD_OUT,
        "generate_battery",
        lambda _L, _n, seed: (
            5,
            3,
            6,
            {HELD_OUT._PRODUCT_STRADDLES_FAMILY: [(2, 2)]},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "held_out_battery.py",
            str(submission),
            "--n",
            "1",
            "--json-out",
            str(receipt_path),
            "--require-exact",
        ],
    )

    assert HELD_OUT.main() == 1
    receipt = json.loads(receipt_path.read_text())
    assert receipt["artifact_hashed_before_model_load"]
    assert receipt["helper_runner_sha256"]
    assert receipt["official_decoder_sha256"]
    assert receipt["source_identity"]["modchallenge.evaluation.pipeline"]["sha256"]
    assert not receipt["artifact_unchanged_during_run"]
    assert not receipt["all_exact"]
    assert receipt["status"] == "failed"


def test_required_fixture_mismatch_stops_before_prediction(
    monkeypatch,
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission"
    submission.mkdir()
    manifest = {"entry_class": "model.Test", "output_base": 2}
    (submission / "manifest.json").write_text(json.dumps(manifest))
    (submission / "model.py").write_text("# test model\n")
    (submission / "weights.pt").write_bytes(b"stable")
    receipt_path = tmp_path / "receipt.json"
    model = ContractBatteryModel(batch_size=2)
    model.L = 3
    model.device = torch.device("cpu")

    monkeypatch.setattr(
        HELD_OUT,
        "load_submission",
        lambda path: (Path(path), manifest, object(), model),
    )
    monkeypatch.setattr(
        HELD_OUT,
        "generate_battery",
        lambda _L, _n, seed: (
            5,
            3,
            6,
            {HELD_OUT._PRODUCT_STRADDLES_FAMILY: [(2, 2)]},
        ),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "held_out_battery.py",
            str(submission),
            "--n",
            "1",
            "--json-out",
            str(receipt_path),
            "--require-base-fixture-sha256",
            "0" * 64,
        ],
    )

    assert HELD_OUT.main() == 2
    assert model.actual_batch_sizes == []
    receipt = json.loads(receipt_path.read_text())
    assert receipt["status"] == "qualification_identity_failed"
    assert receipt["artifact_unchanged_during_run"]
    assert receipt["qualification_identity_checks"][-1]["passed"] is False


def test_qualification_preset_requires_exact_receipt_and_both_orientations(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "held_out_battery.py",
            "unused-submission",
            "--qualification-l2048-n128",
        ],
    )

    with pytest.raises(SystemExit) as error:
        HELD_OUT.main()

    assert error.value.code == 2
    stderr = capsys.readouterr().err
    assert "--require-exact" in stderr
    assert "--json-out PATH" in stderr
    assert "--orientation both" in stderr
    assert "--prelaunch-manifest PATH" in stderr
    assert "--require-prelaunch-manifest-sha256 SHA256" in stderr


def test_prelaunch_manifest_is_bound_before_model_load(
    monkeypatch,
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission"
    runner_root = tmp_path / "source"
    scorer = tmp_path / "scorer"
    pycache = tmp_path / "fresh-pycache"
    control = tmp_path / "control"
    for directory in (submission, runner_root, scorer, pycache, control):
        directory.mkdir()
    (runner_root / "held_out_battery.py").write_text("# runner\n")
    (runner_root / "submission_utils.py").write_text("# helper\n")
    (scorer / ".git").mkdir()

    monkeypatch.setattr(HELD_OUT, "_CALLER_PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(
        HELD_OUT,
        "_CALLER_PYTHONPYCACHEPREFIX",
        str(pycache),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(pycache))
    observation = HELD_OUT.prelaunch_observation(
        submission=submission,
        runner_root=runner_root,
        scorer_repo=scorer,
        external_pycache_prefix=pycache,
    )
    payload = {
        "schema": HELD_OUT._PRELAUNCH_SCHEMA,
        "run_id": "test-run",
        "created_at_utc": "2026-08-02T00:00:00Z",
        "observation": observation,
    }
    manifest = control / "prelaunch.json"
    manifest.write_text(json.dumps(payload, sort_keys=True) + "\n")
    manifest.chmod(0o444)
    expected_sha256 = HELD_OUT.sha256_file(manifest)
    receipt: dict = {}

    provenance = HELD_OUT.verify_prelaunch_manifest(
        receipt,
        manifest_path=manifest,
        expected_sha256=expected_sha256,
        submission=submission,
        runner_root=runner_root,
        scorer_repo=scorer,
    )

    assert provenance["verified_before_model_load"]
    assert provenance["prelaunch_manifest_sha256"] == expected_sha256
    assert provenance["prelaunch_manifest_payload"] == payload
    assert provenance["submission_cache_entries"] == []
    assert provenance["runner_cache_entries"] == []
    assert provenance["scorer_cache_entries"] == []
    assert provenance["external_pycache_prefix_entries"] == []
    assert all(check["passed"] for check in receipt["qualification_identity_checks"])
    assert HELD_OUT.finalize_cache_provenance(receipt)


def test_prelaunch_manifest_rejects_post_manifest_cache(
    monkeypatch,
    tmp_path: Path,
) -> None:
    submission = tmp_path / "submission"
    runner_root = tmp_path / "source"
    scorer = tmp_path / "scorer"
    pycache = tmp_path / "fresh-pycache"
    control = tmp_path / "control"
    for directory in (submission, runner_root, scorer, pycache, control):
        directory.mkdir()
    monkeypatch.setattr(HELD_OUT, "_CALLER_PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setattr(
        HELD_OUT,
        "_CALLER_PYTHONPYCACHEPREFIX",
        str(pycache),
    )
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(sys, "pycache_prefix", str(pycache))
    payload = {
        "schema": HELD_OUT._PRELAUNCH_SCHEMA,
        "run_id": "test-run",
        "created_at_utc": "2026-08-02T00:00:00Z",
        "observation": HELD_OUT.prelaunch_observation(
            submission=submission,
            runner_root=runner_root,
            scorer_repo=scorer,
            external_pycache_prefix=pycache,
        ),
    }
    manifest = control / "prelaunch.json"
    manifest.write_text(json.dumps(payload) + "\n")
    manifest.chmod(0o444)
    expected_sha256 = HELD_OUT.sha256_file(manifest)
    (runner_root / "late.pyc").write_bytes(b"stale")

    with pytest.raises(HELD_OUT.QualificationIdentityError):
        HELD_OUT.verify_prelaunch_manifest(
            {},
            manifest_path=manifest,
            expected_sha256=expected_sha256,
            submission=submission,
            runner_root=runner_root,
            scorer_repo=scorer,
        )


def test_prelaunch_manifest_must_be_read_only(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest = tmp_path / "prelaunch.json"
    manifest.write_text("{}\n")
    manifest.chmod(0o644)

    with pytest.raises(HELD_OUT.QualificationIdentityError):
        HELD_OUT.verify_prelaunch_manifest(
            {},
            manifest_path=manifest,
            expected_sha256=HELD_OUT.sha256_file(manifest),
            submission=tmp_path / "submission",
            runner_root=tmp_path / "source",
            scorer_repo=tmp_path / "scorer",
        )


def test_final_artifact_rehash_error_is_recorded(monkeypatch, tmp_path: Path) -> None:
    receipt = {}

    def failed_identity(_submission):
        raise OSError("rehash unavailable")

    monkeypatch.setattr(HELD_OUT, "artifact_identity", failed_identity)
    unchanged = HELD_OUT.finalize_artifact_identity(
        receipt,
        tmp_path,
        {"weights.pt": "before"},
    )

    assert not unchanged
    assert receipt["artifact_sha256_after"] is None
    assert receipt["artifact_rehash_error"]["type"] == "OSError"
