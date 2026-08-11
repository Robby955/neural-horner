from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_module(path: Path, name: str, extra_path: Path | None = None):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    if extra_path is not None:
        sys.path.insert(0, str(extra_path))
    try:
        spec.loader.exec_module(module)
    finally:
        if extra_path is not None:
            sys.path.pop(0)
    return module


TRACE = load_module(
    SCRIPTS / "trace_f11_trajectories.py",
    "test_trace_f11_trajectories",
    SCRIPTS,
)
DIRECT = load_module(
    ROOT / "candidates" / "direct_two_pass_l2sp_a0875" / "model.py",
    "test_f11_direct_model",
)
ORIGINAL = load_module(
    ROOT / "candidates" / "original_three_pass_l2sp_a0875" / "model.py",
    "test_f11_original_model",
)


def bits_to_int(bits: torch.Tensor) -> int:
    value = 0
    for bit in bits.long().tolist():
        value = 2 * value + int(bit)
    return value


class ExactLogitCell(nn.Module):
    """Return saturated logits for the exact learned recurrence target."""

    def __init__(
        self,
        fail_call: int | None = None,
        zero_margin_call: int | None = None,
        logit_dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.calls = 0
        self.fail_call = fail_call
        self.zero_margin_call = zero_margin_call
        self.logit_dtype = logit_dtype

    def forward(self, features, digits):
        rows, width, _ = features.shape
        outputs = []
        for row in range(rows):
            state = bits_to_int(features[row, :, 0])
            x = bits_to_int(features[row, :, 1])
            p = bits_to_int(features[row, :, 2])
            digit = int(digits[row].item())
            expected = (2 * state + digit * x) % p
            target = torch.tensor(
                [(expected >> shift) & 1 for shift in range(width - 1, -1, -1)],
                dtype=torch.float32,
                device=features.device,
            )
            logits = target.mul(40).sub(20)
            if self.fail_call == self.calls:
                target[-1] = 1 - target[-1]
                logits = target.mul(40).sub(20)
            if self.zero_margin_call == self.calls:
                zero_indices = torch.nonzero(target == 0, as_tuple=False)
                assert zero_indices.numel() > 0
                logits[int(zero_indices[0].item())] = 0
            outputs.append(logits.to(self.logit_dtype))
        self.calls += 1
        return torch.stack(outputs)


def make_exact_model(
    base_type,
    fail_call: int | None = None,
    zero_margin_call: int | None = None,
    logit_dtype: torch.dtype = torch.float32,
):
    model = base_type()
    model.L = 64
    model.device = torch.device("cpu")
    model.model = ExactLogitCell(
        fail_call=fail_call,
        zero_margin_call=zero_margin_call,
        logit_dtype=logit_dtype,
    )
    model.model.eval()
    return model


def make_exact_tracer(
    base_type,
    schedule: str,
    fail_call: int | None = None,
    zero_margin_call: int | None = None,
    logit_dtype: torch.dtype = torch.float32,
):
    tracer_type = TRACE.make_tracer(base_type, schedule)
    return make_exact_model(
        tracer_type,
        fail_call=fail_call,
        zero_margin_call=zero_margin_call,
        logit_dtype=logit_dtype,
    )


def case(label: str, a: int, b: int, p: int):
    return TRACE.TraceCase(label, "unit", "mock", a, b, p)


def test_direct_trace_is_exact_and_uses_two_candidate_owned_phases() -> None:
    model = make_exact_tracer(
        DIRECT.DirectBitSerialReducer,
        "direct_two_pass",
    )
    item = case("direct", (1 << 12) + 5, (1 << 7) + 3, 65_537)

    result = TRACE.run_oriented_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
    )

    assert result["all_exact"]
    assert result["expected_phase_counts"] == {
        "reduce_operand": item.a.bit_length(),
        "scan_operand": item.b.bit_length(),
    }
    assert result["phase_counts"] == result["expected_phase_counts"]
    assert result["candidate_prefix_executed"]
    assert result["incoming_state_source"] == (
        "candidate_owned_full_trajectory_from_zero"
    )
    assert result["first_divergence"] is None
    assert result["captured_logit_dtypes"] == ["torch.float32"]
    assert (
        result["worst_signed_target_logit_margin"]["minimum_signed_target_logit_margin"]
        == 20.0
    )


def test_original_trace_counts_both_reductions_and_dynamic_multiply() -> None:
    model = make_exact_tracer(
        ORIGINAL.BitSerialReducer,
        "original_three_pass",
    )
    item = case("original", (1 << 12) + 5, (1 << 7) + 3, 65_537)

    result = TRACE.run_oriented_case(
        model,
        ORIGINAL,
        "original_three_pass",
        item,
        "swapped",
    )

    assert result["all_exact"]
    assert result["expected_phase_counts"] == {
        "reduce_a": item.b.bit_length(),
        "reduce_b": item.a.bit_length(),
        "multiply": 32,
    }
    assert result["phase_order"] == ["reduce_a", "reduce_b", "multiply"]
    assert result["observed_transitions"] == sum(
        result["expected_phase_counts"].values()
    )


def test_first_divergence_has_margin_boundary_carry_and_subtract_metadata() -> None:
    model = make_exact_tracer(
        DIRECT.DirectBitSerialReducer,
        "direct_two_pass",
        fail_call=2,
    )
    item = case("fault", 13, 7, 17)

    result = TRACE.run_oriented_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
    )
    divergence = result["first_divergence"]

    assert not result["all_exact"]
    assert result["failed_transitions"] >= 1
    assert divergence["global_step"] == 2
    assert divergence["phase"] == "reduce_operand"
    assert divergence["minimum_signed_target_logit_margin"] == -20.0
    assert divergence["wrong_output_bit_count"] == 1
    assert "boundary_distance_to_nearest_modulus_multiple" in divergence
    assert "double_carry_out" in divergence
    assert "add_after_truncated_double_carry_out" in divergence
    assert "modulus_subtract_count" in divergence


def test_vectorized_induction_certifies_exact_direct_rollout() -> None:
    model = make_exact_model(DIRECT.DirectBitSerialReducer)
    candidate_step = model._step
    step_calls = 0

    def counted_step(instance, s_bits, x_bits, p_bits, digits):
        nonlocal step_calls
        step_calls += 1
        return candidate_step(s_bits, x_bits, p_bits, digits)

    model._step = types.MethodType(counted_step, model)
    item = case("inductive", (1 << 12) + 5, (1 << 7) + 3, 65_537)

    result = TRACE.run_inductive_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
        transition_batch_size=4,
    )

    assert result["all_exact"]
    assert result["candidate_full_rollout_certified"]
    assert result["execution_mode"] == "vectorized_exact_prefix_induction"
    assert result["teacher_transitions_evaluated"] == result["expected_transitions"]
    assert (
        result["free_running_prefix_transitions_certified"]
        == result["expected_transitions"]
    )
    assert result["program_output_validation"]["exact"]
    route = result["program_output_validation"]["route_validation"]
    assert route["route_exact"]
    assert route["expected_route_sha256"] == route["observed_route_sha256"]
    assert result["strictly_positive_margin"]
    assert result["captured_logit_dtypes"] == ["torch.float32"]
    assert result["candidate_prediction_path"] == (
        "entry_instance_actual__step_with_cell_forward_hook"
    )
    expected_step_calls = (
        result["expected_transitions"] + result["transition_batch_size"] - 1
    ) // result["transition_batch_size"]
    assert step_calls == expected_step_calls


def test_vectorized_induction_certifies_exact_original_schedule() -> None:
    model = make_exact_model(ORIGINAL.BitSerialReducer)
    item = case("original-inductive", (1 << 12) + 5, (1 << 7) + 3, 65_537)

    result = TRACE.run_inductive_case(
        model,
        ORIGINAL,
        "original_three_pass",
        item,
        "swapped",
        transition_batch_size=5,
    )

    assert result["all_exact"]
    assert result["phase_order"] == ["reduce_a", "reduce_b", "multiply"]
    assert result["program_output_validation"]["route_validation"]["route_exact"]
    assert result["strictly_positive_margin"]


def test_vectorized_first_failure_is_free_running_valid_by_induction() -> None:
    model = make_exact_model(DIRECT.DirectBitSerialReducer, fail_call=1)
    item = case("inductive-failure", 13, 7, 17)

    result = TRACE.run_inductive_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
        transition_batch_size=2,
    )
    divergence = result["first_divergence"]

    # ExactLogitCell fails a whole chunk call, so the first bad transition is
    # the first row in the second two-transition chunk.
    assert not result["all_exact"]
    assert divergence["global_step"] == 2
    assert divergence["free_running_divergence_valid_by_exact_prefix_induction"]
    assert result["free_running_prefix_transitions_certified"] == 2
    assert result["post_divergence_teacher_transitions_evaluated"] > 0
    assert divergence["state_width_bits"] == 32


def test_zero_signed_margin_fails_inductive_qualification_gate() -> None:
    model = make_exact_model(
        DIRECT.DirectBitSerialReducer,
        zero_margin_call=0,
    )
    item = case("zero-margin", 13, 7, 17)

    result = TRACE.run_inductive_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
        transition_batch_size=64,
    )

    assert result["failed_transitions"] == 0
    assert (
        result["worst_signed_target_logit_margin"]["minimum_signed_target_logit_margin"]
        == 0
    )
    assert not result["strictly_positive_margin"]
    assert not result["candidate_full_rollout_certified"]
    assert not result["all_exact"]


def test_zero_signed_margin_fails_sequential_qualification_gate() -> None:
    model = make_exact_tracer(
        DIRECT.DirectBitSerialReducer,
        "direct_two_pass",
        zero_margin_call=0,
    )
    item = case("sequential-zero-margin", 13, 7, 17)

    result = TRACE.run_oriented_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
    )

    assert result["failed_transitions"] == 0
    assert result["final_exact"]
    assert not result["strictly_positive_margin"]
    assert not result["all_exact"]


def test_exact_step_manifest_replay_detects_route_mismatch() -> None:
    model = make_exact_model(DIRECT.DirectBitSerialReducer)
    original_scan = model._scan_bits

    def rerouted_scan(instance, bit_lists, x_bits, p_bits, effective_width):
        changed = [list(bits) for bits in bit_lists]
        changed[0][0] = 1 - changed[0][0]
        return original_scan(changed, x_bits, p_bits, effective_width)

    model._scan_bits = types.MethodType(rerouted_scan, model)
    item = case("route-mismatch", 13, 7, 17)

    result = TRACE.run_inductive_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
        transition_batch_size=64,
    )
    route = result["program_output_validation"]["route_validation"]

    assert not route["route_exact"]
    assert route["expected_transition_count"] == route["observed_transition_count"]
    assert route["expected_route_sha256"] != route["observed_route_sha256"]
    assert route["first_route_mismatch"]["transition_index"] == 0
    assert "d" in route["first_route_mismatch"]["differing_fields"]
    assert not result["candidate_full_rollout_certified"]
    assert not result["all_exact"]


def test_route_spec_is_independent_of_candidate_ordering_helper(monkeypatch) -> None:
    model = make_exact_model(DIRECT.DirectBitSerialReducer)
    item = case("ordering-mismatch", 13, 2, 17)

    # The expected route belongs to the qualification runner. Deliberately
    # corrupting the candidate module's helper must change only the observed
    # manifest route and must not also rewrite the expected route.
    monkeypatch.setattr(
        DIRECT,
        "_canonical_scan_reduce",
        lambda a_bits, b_bits: (a_bits, b_bits),
    )
    result = TRACE.run_inductive_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
        transition_batch_size=64,
    )
    route = result["manifest_route_validation"]

    assert not route["route_exact"]
    assert route["expected_route_sha256"] != route["observed_route_sha256"]
    assert not result["all_exact"]


def test_transition_metadata_distinguishes_word_carry_and_modular_subtract() -> None:
    metadata = TRACE.transition_arithmetic_metadata(
        state=200,
        x=100,
        p=251,
        digit=1,
        width=8,
    )

    assert metadata["double_carry_out"]
    assert metadata["add_after_truncated_double_carry_out"] is False
    assert metadata["modulus_subtract_required"]
    assert metadata["modulus_subtract_count"] == "1"
    assert metadata["expected_value"] == "249"
    assert metadata["boundary_distance_to_nearest_modulus_multiple"] == "2"


def test_case_group_selection_is_cardinality_checked() -> None:
    cases = [
        case("d", 1, 1, 17)._replace(group="decisive"),
        case("c1", 1, 1, 17)._replace(group="companions"),
        case("c2", 1, 1, 17)._replace(group="companions"),
    ]

    assert [item.label for item in TRACE.select_cases(cases, ["decisive"])] == ["d"]
    assert [
        item.label for item in TRACE.select_cases(cases, ["decisive", "companions"])
    ] == ["d", "c1", "c2"]
    try:
        TRACE.select_cases(cases, ["all", "decisive"])
    except ValueError as error:
        assert "cannot be combined" in str(error)
    else:
        raise AssertionError("mixed all/group selection must fail closed")


def test_full_f11_suite_has_twenty_unique_numerical_cases() -> None:
    cases = TRACE.build_cases(2048)

    assert len(cases) == 20
    assert len({item.label for item in cases}) == 20
    assert len({(item.a, item.b, item.p) for item in cases}) == 20
    assert cases[0].label == "decisive-f11-x-1"
    assert cases[0].b == 1


def write_case_fixture(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": TRACE.EXTERNAL_CASE_FIXTURE_SCHEMA,
                "run_id": "unit-fixture",
                "expected_case_count": len(rows),
                "cases": rows,
            }
        )
        + "\n"
    )


def fixture_row(item) -> dict[str, object]:
    payload = TRACE.case_payload(item)
    return {
        **payload,
        "source_case_index": 3,
        "case_sha256": TRACE.canonical_json_sha256(payload),
    }


def test_external_case_fixture_is_hash_bound_and_ordered(tmp_path: Path) -> None:
    items = [
        case("first", 13, 7, 17),
        case("second", 19, 11, 23),
    ]
    fixture = tmp_path / "cases.json"
    write_case_fixture(fixture, [fixture_row(item) for item in items])

    loaded, identity = TRACE.load_external_case_fixture(fixture)

    assert loaded == items
    assert identity["status"] == "verified"
    assert identity["sha256"] == TRACE.sha256_file(fixture)
    assert identity["expected_case_count"] == 2
    assert identity["case_set_sha256"] == TRACE.canonical_json_sha256(
        [TRACE.case_payload(item) for item in items]
    )


def test_external_case_fixture_rejects_changed_case_digest(tmp_path: Path) -> None:
    item = case("changed", 13, 7, 17)
    row = fixture_row(item)
    row["case_sha256"] = "0" * 64
    fixture = tmp_path / "cases.json"
    write_case_fixture(fixture, [row])

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        TRACE.load_external_case_fixture(fixture)


def test_external_case_fixture_rejects_noncanonical_integer(tmp_path: Path) -> None:
    item = case("leading-zero", 13, 7, 17)
    row = fixture_row(item)
    row["a"] = "013"
    fixture = tmp_path / "cases.json"
    write_case_fixture(fixture, [row])

    with pytest.raises(ValueError, match="non-canonical 'a'"):
        TRACE.load_external_case_fixture(fixture)


def write_minimal_submission(path: Path) -> None:
    path.mkdir()
    (path / "manifest.json").write_text('{"entry_class":"model.Mock"}\n')
    (path / "model.py").write_text("class Mock:\n    pass\n")
    (path / "weights.pt").write_bytes(b"stable-weights")


def test_guarded_load_hashes_artifacts_before_and_after_load(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    write_minimal_submission(submission)

    def stable_loader(path: Path):
        return path, {"entry_class": "model.Mock"}, object(), object()

    loaded = TRACE.guarded_load_submission(submission, loader=stable_loader)

    assert loaded[0] == submission.resolve()
    assert loaded[4] == loaded[5]
    assert loaded[4]["weights.pt"] == TRACE.sha256_file(
        submission / "weights.pt"
    )


def test_guarded_load_fails_if_artifact_changes_during_load(tmp_path: Path) -> None:
    submission = tmp_path / "submission"
    write_minimal_submission(submission)

    def mutating_loader(path: Path):
        (path / "weights.pt").write_bytes(b"mutated-during-load")
        return path, {"entry_class": "model.Mock"}, object(), object()

    with pytest.raises(
        RuntimeError,
        match=r"submission artifacts changed during load: weights\.pt",
    ):
        TRACE.guarded_load_submission(submission, loader=mutating_loader)


def test_source_identity_binds_runner_and_imported_helpers(tmp_path: Path) -> None:
    identity = TRACE.qualification_source_identity(
        tmp_path / "missing-scorer-contract.json"
    )
    local_sources = identity["local_source_sha256"]

    assert set(local_sources) == {
        "scripts/trace_f11_trajectories.py",
        "scripts/held_out_battery.py",
        "scripts/submission_utils.py",
    }
    assert identity["local_source_set_sha256"] == TRACE.canonical_json_sha256(
        local_sources
    )
    assert identity["scorer"]["status"] == "contract_not_discovered"


def test_discovered_scorer_sources_are_verified_and_mismatch_fails_closed(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scorer = tmp_path / "scorer"
    source = scorer / "rules" / "overview.md"
    source.parent.mkdir(parents=True)
    source.write_text("pinned scorer source\n")
    declared_commit = "8" * 40
    contract = tmp_path / "scorer-contract.json"
    contract.write_text(
        json.dumps(
            {
                "scorer_repository": str(scorer),
                "scorer_commit": declared_commit,
                "source_sha256": {
                    "rules/overview.md": TRACE.sha256_file(source),
                },
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        TRACE,
        "_git_repository_identity",
        lambda _repository: {
            "head": declared_commit,
            "tracked_files_clean": True,
        },
    )

    identity = TRACE.scorer_source_identity(contract)

    assert identity["status"] == "verified"
    assert identity["commit_matches_contract"]
    assert identity["sources_match_contract"]
    assert identity["observed_source_sha256"] == identity[
        "declared_source_sha256"
    ]

    source.write_text("changed scorer source\n")
    with pytest.raises(RuntimeError, match="scorer source identity mismatch"):
        TRACE.scorer_source_identity(contract)


def test_effective_model_tensor_dtype_identity_records_parameters_and_buffers() -> None:
    cell = nn.Linear(3, 2, dtype=torch.float64)
    cell.register_buffer("integer_buffer", torch.ones(2, dtype=torch.int32))
    wrapper = types.SimpleNamespace(model=cell)

    identity = TRACE.model_tensor_dtype_identity(wrapper)

    assert identity["parameters"] == {
        "tensor_count": 2,
        "element_count": 8,
        "dtype_counts": {"torch.float64": 2},
        "device_counts": {"cpu": 2},
    }
    assert identity["buffers"] == {
        "tensor_count": 1,
        "element_count": 2,
        "dtype_counts": {"torch.int32": 1},
        "device_counts": {"cpu": 1},
    }


def test_inductive_result_records_raw_captured_logit_dtype() -> None:
    model = make_exact_model(
        DIRECT.DirectBitSerialReducer,
        logit_dtype=torch.float16,
    )
    item = case("half-logits", 13, 7, 17)

    result = TRACE.run_inductive_case(
        model,
        DIRECT,
        "direct_two_pass",
        item,
        "original",
        transition_batch_size=64,
    )

    assert result["all_exact"]
    assert result["captured_logit_dtypes"] == ["torch.float16"]
