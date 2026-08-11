#!/usr/bin/env python3
"""Paired sampled check for trained exactness and randomized-weight collapse.

The default gate evaluates trained and deterministically randomized copies of
the same artifact on the exact same nonsealed cases.  The fixture includes the
pinned public scored-tier operand boundaries through 4096 bits plus a small set
of generated shortcut-distinguishing rows.  All inputs reach preprocessing as
official decimal strings and all outputs are interpreted by the pinned scorer
decoder.

A pass is evidence only for these sampled cases.  It does not prove that every
answer is encoded in learned weights, exclude every possible shortcut, establish
private-set accuracy, or establish official runtime compliance.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch

from check_batch_invariance import (
    DEFAULT_SCORER_CONTRACT,
    PROJECT_ROOT,
    GateCase,
    artifact_tree_identity,
    canonical_json_sha256,
    case_payload,
    compact_case_identity,
    decode_output,
    environment_identity,
    force_device,
    guarded_load_submission,
    load_pinned_scorer,
    local_source_identity,
    model_tensor_identity,
    receipt_output_exclusions,
    run_layout,
    scorer_batch_size,
    select_public_boundary_cases,
    state_dict_sha256,
)
from submission_utils import artifact_identity


_SMALL = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
DEFAULT_CASE_SEED = 2026
DEFAULT_RANDOMIZATION_SEED = 12345


def is_prime(n: int, rng: random.Random) -> bool:
    if n < 2:
        return False
    for p in _SMALL:
        if n % p == 0:
            return n == p
    d, r = n - 1, 0
    while d % 2 == 0:
        d //= 2
        r += 1
    bases = list(_SMALL) + [rng.randrange(2, n - 1) for _ in range(16)]
    for a in bases:
        x = pow(a, d, n)
        if x == 1 or x == n - 1:
            continue
        for _ in range(r - 1):
            x = (x * x) % n
            if x == n - 1:
                break
        else:
            return False
    return True


def draw_distinguishing_case(
    rng: random.Random,
    p: int,
    op_bits: int,
    *,
    max_attempts: int = 10_000,
) -> tuple[int, int]:
    """Draw operands whose exact answer differs from each tested shortcut."""
    for _ in range(max_attempts):
        a = rng.randrange(0, 1 << op_bits)
        b = rng.randrange(0, 1 << op_bits)
        a_residue = a % p
        b_residue = b % p
        truth = (a * b) % p
        if truth not in (a_residue, b_residue, a_residue * b_residue):
            return a, b
    raise RuntimeError(
        f"could not draw a shortcut-distinguishing case in {max_attempts} attempts"
    )


def passes_requested_gate(
    *,
    randomized: bool,
    correct: int,
    total: int,
    max_randomized_correct: int,
    shortcut_checks: dict[str, bool],
) -> bool:
    """Compatibility helper retained for focused trained/randomized diagnostics."""
    if randomized:
        return correct <= max_randomized_correct
    return correct == total and all(shortcut_checks.values())


def _sample_prime_for_width(
    public_cases: list[GateCase],
    width: int,
    rng: random.Random,
) -> int:
    if width == 2:
        return 3
    exact_width = [case.p for case in public_cases if case.p.bit_length() == width]
    if exact_width:
        return exact_width[0]
    lo, hi = 1 << (width - 1), (1 << width) - 1
    for _ in range(100_000):
        candidate = rng.randint(lo, hi) | 1
        if is_prime(candidate, rng):
            return candidate
    raise RuntimeError(f"could not draw a probable {width}-bit prime")


def build_paired_cases(
    public_cases: list[GateCase],
    *,
    n: int,
    prime_bits: int,
    max_model_width: int,
    seed: int,
) -> list[GateCase]:
    if n <= 0:
        raise ValueError("n must be positive")
    if not 2 <= prime_bits <= max_model_width:
        raise ValueError(f"prime bits must be in [2, {max_model_width}]")
    rng = random.Random(seed)
    p = _sample_prime_for_width(public_cases, prime_bits, rng)
    operand_bits = min(2 * prime_bits, 4096)
    generated: list[GateCase] = []
    for index in range(n):
        a, b = draw_distinguishing_case(rng, p, operand_bits)
        generated.append(
            GateCase(
                case_id=f"generated-distinguishing-{prime_bits}-{index}",
                source="deterministic_nonsealed_shortcut_screen",
                tier_id=None,
                source_index=index,
                a=a,
                b=b,
                p=p,
                expected=(a * b) % p,
            )
        )
    cases = [*public_cases, *generated]
    if len({case.case_id for case in cases}) != len(cases):
        raise RuntimeError("paired fixture case identifier collision")
    return cases


def randomize_loaded_model(model, module, submission: Path, seed: int) -> str:
    """Replace the loaded cell with a deterministic fresh initialization."""
    checkpoint = torch.load(
        submission / "weights.pt",
        map_location="cpu",
        weights_only=True,
    )
    cell_type = getattr(module, "Cell", None)
    if cell_type is None:
        raise RuntimeError("submission module does not expose Cell for randomization")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        fresh = cell_type(**checkpoint.get("config", {}))
    target = getattr(model, "model", None)
    if not isinstance(target, torch.nn.Module):
        raise RuntimeError("loaded entry does not expose model: torch.nn.Module")
    target.load_state_dict(fresh.state_dict(), strict=True)
    device = torch.device(getattr(model, "device", "cpu"))
    target.to(device)
    target.eval()
    return state_dict_sha256(model)


def _shortcut_witnesses(
    cases: list[GateCase],
    predictions: list[int | None],
) -> dict[str, int]:
    return {
        "not_a_mod_p_witness_count": sum(
            prediction == case.expected and case.expected != case.a % case.p
            for case, prediction in zip(cases, predictions)
        ),
        "not_b_mod_p_witness_count": sum(
            prediction == case.expected and case.expected != case.b % case.p
            for case, prediction in zip(cases, predictions)
        ),
        "not_unreduced_residue_product_witness_count": sum(
            prediction == case.expected
            and case.expected != (case.a % case.p) * (case.b % case.p)
            for case, prediction in zip(cases, predictions)
        ),
    }


def evaluate_arm(
    model,
    cases: list[GateCase],
    *,
    decoder,
    output_base: int | str,
    label: str,
) -> dict[str, object]:
    declared, effective = scorer_batch_size(model)
    layout = run_layout(
        model,
        cases,
        order=list(range(len(cases))),
        batch_size=effective,
        label=label,
    )
    outputs = layout.pop("outputs")
    predictions: list[int | None] = []
    cases_receipt = []
    valid_count = 0
    correct = 0
    for case_index, case in enumerate(cases):
        output = outputs[case_index]
        if output_base != 2:
            def manifest_decoder(value, *, base, prime, is_tier_zero):
                return decoder(
                    value,
                    base=output_base,
                    prime=prime,
                    is_tier_zero=is_tier_zero,
                )
        else:
            manifest_decoder = decoder
        valid, value, error = decode_output(manifest_decoder, output, case)
        predictions.append(value)
        valid_count += int(valid)
        exact = bool(valid and value == case.expected)
        correct += int(exact)
        cases_receipt.append(
            {
                **compact_case_identity(case),
                "output_present": output is not None,
                "decoder_valid": valid,
                "decode_error": error,
                "exact": exact,
            }
        )
    witnesses = _shortcut_witnesses(cases, predictions)
    return {
        "label": label,
        "declared_batch_size": declared,
        "effective_batch_size": effective,
        "expected_outputs": len(cases),
        "completed_outputs": layout["completed_outputs"],
        "output_cardinality_exact": layout["cardinality_exact"],
        "batch_checks": layout["batch_checks"],
        "output_sha256": (
            canonical_json_sha256(outputs) if layout["cardinality_exact"] else None
        ),
        "decoder_valid": valid_count,
        "decoder_invalid": len(cases) - valid_count,
        "correct": correct,
        "total": len(cases),
        "shortcut_witness_counts": witnesses,
        "shortcut_checks": {
            name.removesuffix("_witness_count"): count > 0
            for name, count in witnesses.items()
        },
        "cases": cases_receipt,
    }


def paired_gate_passed(
    trained: dict[str, object] | None,
    randomized: dict[str, object] | None,
) -> bool:
    """Qualification requires literal zero sampled exact matches after randomization."""
    if trained is None or randomized is None:
        return False
    return all(
        (
            trained["output_cardinality_exact"],
            randomized["output_cardinality_exact"],
            trained["correct"] == trained["total"],
            all(trained["shortcut_checks"].values()),
            randomized["correct"] == 0,
            trained["total"] == randomized["total"],
            [case["case_sha256"] for case in trained["cases"]]
            == [case["case_sha256"] for case in randomized["cases"]],
        )
    )


def write_receipt(path: Path | None, receipt: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("submission")
    ap.add_argument(
        "--n",
        type=int,
        default=4,
        help="generated distinguishing rows in addition to 20 public boundary rows",
    )
    ap.add_argument(
        "--prime-bits",
        type=int,
        help="generated-row modulus width; default is the checkpoint width",
    )
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--scorer-contract", type=Path, default=DEFAULT_SCORER_CONTRACT)
    ap.add_argument(
        "--scorer-repo",
        type=Path,
        help=(
            "clean checkout of the contract's exact scorer commit; permits "
            "reconstructing an expired historical checkout path without "
            "weakening commit or source-hash checks"
        ),
    )
    ap.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument(
        "--randomize",
        action="store_true",
        help="legacy randomized-only diagnostic; default is paired",
    )
    mode.add_argument(
        "--trained-only",
        action="store_true",
        help="trained-only diagnostic; default is paired",
    )
    ap.add_argument("--case-seed", type=int, default=DEFAULT_CASE_SEED)
    ap.add_argument(
        "--randomization-seed",
        type=int,
        default=DEFAULT_RANDOMIZATION_SEED,
    )
    ap.add_argument("--max-randomized-correct", type=int, default=0)
    args = ap.parse_args()
    if args.n <= 0:
        ap.error("--n must be positive")
    if args.max_randomized_correct < 0:
        ap.error("--max-randomized-correct must be nonnegative")

    if args.randomize:
        execution_mode = "randomized_only_diagnostic"
    elif args.trained_only:
        execution_mode = "trained_only_diagnostic"
    elif args.max_randomized_correct > 0:
        execution_mode = "paired_relaxed_threshold_diagnostic"
    else:
        execution_mode = "paired_qualification"
    runner_path = Path(__file__).resolve()
    helper_path = runner_path.with_name("check_batch_invariance.py")
    source_before = local_source_identity(
        runner_path,
        additional_paths=(helper_path,),
    )
    scorer_identity, decoder, public_generator = load_pinned_scorer(
        args.scorer_contract,
        args.scorer_repo,
    )
    public_cases = select_public_boundary_cases(public_generator())

    artifact_tree_exclusions = receipt_output_exclusions(
        args.submission,
        args.json_out,
    )
    artifact_tree_before = artifact_tree_identity(
        args.submission,
        excluded_paths=artifact_tree_exclusions,
    )

    (
        submission,
        manifest,
        trained_module,
        trained_model,
        artifact_before,
        _artifact_after_trained_load,
    ) = guarded_load_submission(args.submission)
    artifact_tree_after_trained_load = artifact_tree_identity(
        submission,
        excluded_paths=artifact_tree_exclusions,
    )
    if artifact_tree_before != artifact_tree_after_trained_load:
        raise RuntimeError("artifact tree changed during trained-arm load")
    trained_device = force_device(trained_model, args.device)
    checkpoint_width = int(getattr(trained_model, "L", 32))
    prime_bits = args.prime_bits or checkpoint_width
    cases = build_paired_cases(
        public_cases,
        n=args.n,
        prime_bits=prime_bits,
        max_model_width=checkpoint_width,
        seed=args.case_seed,
    )
    case_payloads = [case_payload(case) for case in cases]

    randomized_model = None
    randomized_module = None
    randomized_device = None
    if not args.trained_only:
        (
            random_submission,
            random_manifest,
            randomized_module,
            randomized_model,
            random_artifact_before,
            _random_artifact_after_load,
        ) = guarded_load_submission(args.submission)
        if random_submission != submission or random_manifest != manifest:
            raise RuntimeError("randomized arm resolved a different submission")
        if random_artifact_before != artifact_before:
            raise RuntimeError("artifact identity differs between paired arms")
        random_tree_after_load = artifact_tree_identity(
            submission,
            excluded_paths=artifact_tree_exclusions,
        )
        if random_tree_after_load != artifact_tree_before:
            raise RuntimeError("artifact tree changed during randomized-arm load")
        randomized_device = force_device(randomized_model, args.device)

    trained_state_before = state_dict_sha256(trained_model)
    trained_tensor_identity = model_tensor_identity(trained_model)
    randomized_state_before = None
    randomized_tensor_identity = None
    if randomized_model is not None and randomized_module is not None:
        randomized_state_before = randomize_loaded_model(
            randomized_model,
            randomized_module,
            submission,
            args.randomization_seed,
        )
        randomized_tensor_identity = model_tensor_identity(randomized_model)
        if randomized_state_before == trained_state_before:
            raise RuntimeError("randomized and trained state hashes are identical")

    environment = environment_identity(trained_device)
    randomized_environment = (
        environment_identity(randomized_device)
        if randomized_device is not None
        else None
    )
    receipt: dict[str, object] = {
        "schema_version": 2,
        "status": "running",
        "execution_mode": execution_mode,
        "claim_scope": "sampled_public_and_generated_nonsealed_cases_only",
        "nonclaims": [
            "not_a_proof_that_all_answers_are_encoded_in_weights",
            "not_an_exhaustive_shortcut_exclusion",
            "not_private_set_evidence",
            "not_runtime_qualification",
        ],
        "submission": str(submission),
        "entry_class": manifest["entry_class"],
        "output_base": manifest["output_base"],
        "artifact_sha256_before": artifact_before,
        "artifact_set_sha256": canonical_json_sha256(artifact_before),
        "artifact_tree_identity_before": artifact_tree_before,
        "artifact_tree_identity_sha256": canonical_json_sha256(
            artifact_tree_before
        ),
        "local_source_identity_before": source_before,
        "runner_sha256": source_before["source_sha256"][
            str(runner_path.relative_to(PROJECT_ROOT))
        ],
        "scorer_identity": scorer_identity,
        "scorer_identity_sha256": canonical_json_sha256(scorer_identity),
        "case_set_sha256": canonical_json_sha256(case_payloads),
        "expected_cases_per_arm": len(cases),
        "public_boundary_cases": len(public_cases),
        "generated_distinguishing_cases": args.n,
        "selected_cases": [compact_case_identity(case) for case in cases],
        "checkpoint_width": checkpoint_width,
        "screen_prime_bits": prime_bits,
        "case_seed": args.case_seed,
        "randomization_seed": (
            args.randomization_seed if randomized_model is not None else None
        ),
        "max_randomized_correct": args.max_randomized_correct,
        "trained_state_sha256_before": trained_state_before,
        "randomized_state_sha256_before": randomized_state_before,
        "trained_tensor_identity": trained_tensor_identity,
        "randomized_tensor_identity": randomized_tensor_identity,
        "environment": environment,
        "environment_sha256": canonical_json_sha256(environment),
        "randomized_environment": randomized_environment,
        "randomized_environment_sha256": (
            canonical_json_sha256(randomized_environment)
            if randomized_environment is not None
            else None
        ),
        "trained": None,
        "randomized": None,
        "passed_requested_gate": False,
        "passed_qualification_gate": False,
    }
    write_receipt(args.json_out, receipt)

    try:
        if not args.randomize:
            receipt["trained"] = evaluate_arm(
                trained_model,
                cases,
                decoder=decoder,
                output_base=manifest["output_base"],
                label="trained",
            )
            write_receipt(args.json_out, receipt)
        if randomized_model is not None:
            receipt["randomized"] = evaluate_arm(
                randomized_model,
                cases,
                decoder=decoder,
                output_base=manifest["output_base"],
                label="randomized",
            )
            write_receipt(args.json_out, receipt)
    except BaseException as error:
        receipt["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        )
        receipt["error"] = {"type": type(error).__name__, "message": str(error)}
        write_receipt(args.json_out, receipt)
        raise

    try:
        trained_state_after = state_dict_sha256(trained_model)
        randomized_state_after = (
            state_dict_sha256(randomized_model)
            if randomized_model is not None
            else None
        )
        artifact_after = artifact_identity(submission)
        artifact_tree_after = artifact_tree_identity(
            submission,
            excluded_paths=artifact_tree_exclusions,
        )
        source_after = local_source_identity(
            runner_path,
            additional_paths=(helper_path,),
        )
        scorer_identity_after, _decoder_after, _generator_after = (
            load_pinned_scorer(args.scorer_contract, args.scorer_repo)
        )
    except BaseException as error:
        receipt["status"] = (
            "interrupted" if isinstance(error, KeyboardInterrupt) else "error"
        )
        receipt["error"] = {
            "phase": "post_run_identity_finalization",
            "type": type(error).__name__,
            "message": str(error),
        }
        write_receipt(args.json_out, receipt)
        raise
    artifacts_unchanged = artifact_before == artifact_after
    artifact_tree_unchanged = artifact_tree_before == artifact_tree_after
    sources_unchanged = source_before == source_after
    scorer_unchanged = scorer_identity == scorer_identity_after
    trained_state_unchanged = trained_state_before == trained_state_after
    randomized_state_unchanged = (
        randomized_state_before == randomized_state_after
        if randomized_model is not None
        else True
    )
    qualification_passed = all(
        (
            execution_mode == "paired_qualification",
            paired_gate_passed(
                receipt["trained"],
                receipt["randomized"],
            ),
            artifacts_unchanged,
            artifact_tree_unchanged,
            sources_unchanged,
            scorer_unchanged,
            trained_state_unchanged,
            randomized_state_unchanged,
            environment == randomized_environment,
            trained_tensor_identity == randomized_tensor_identity,
        )
    )

    if execution_mode == "trained_only_diagnostic":
        arm = receipt["trained"]
        requested_passed = bool(
            arm
            and passes_requested_gate(
                randomized=False,
                correct=arm["correct"],
                total=arm["total"],
                max_randomized_correct=args.max_randomized_correct,
                shortcut_checks=arm["shortcut_checks"],
            )
            and arm["output_cardinality_exact"]
        )
    elif execution_mode == "randomized_only_diagnostic":
        arm = receipt["randomized"]
        requested_passed = bool(
            arm
            and passes_requested_gate(
                randomized=True,
                correct=arm["correct"],
                total=arm["total"],
                max_randomized_correct=args.max_randomized_correct,
                shortcut_checks=arm["shortcut_checks"],
            )
            and arm["output_cardinality_exact"]
        )
    elif execution_mode == "paired_qualification":
        requested_passed = qualification_passed
    else:
        trained_arm = receipt["trained"]
        randomized_arm = receipt["randomized"]
        requested_passed = bool(
            trained_arm
            and randomized_arm
            and trained_arm["output_cardinality_exact"]
            and randomized_arm["output_cardinality_exact"]
            and trained_arm["correct"] == trained_arm["total"]
            and all(trained_arm["shortcut_checks"].values())
            and randomized_arm["correct"] <= args.max_randomized_correct
            and [case["case_sha256"] for case in trained_arm["cases"]]
            == [case["case_sha256"] for case in randomized_arm["cases"]]
        )
    requested_passed = bool(
        requested_passed
        and artifacts_unchanged
        and artifact_tree_unchanged
        and sources_unchanged
        and scorer_unchanged
        and trained_state_unchanged
        and randomized_state_unchanged
    )
    receipt.update(
        {
            "status": (
                "completed_exact"
                if qualification_passed
                else "completed_diagnostic"
                if requested_passed
                else "failed"
            ),
            "artifact_sha256_after": artifact_after,
            "artifact_unchanged_during_gate": artifacts_unchanged,
            "artifact_tree_identity_after": artifact_tree_after,
            "artifact_tree_unchanged_during_gate": artifact_tree_unchanged,
            "local_source_identity_after": source_after,
            "sources_unchanged_during_gate": sources_unchanged,
            "scorer_identity_after": scorer_identity_after,
            "scorer_unchanged_during_gate": scorer_unchanged,
            "trained_state_sha256_after": trained_state_after,
            "randomized_state_sha256_after": randomized_state_after,
            "trained_state_unchanged_during_gate": trained_state_unchanged,
            "randomized_state_unchanged_during_gate": randomized_state_unchanged,
            "paired_case_identity_exact": bool(
                receipt["trained"]
                and receipt["randomized"]
                and [case["case_sha256"] for case in receipt["trained"]["cases"]]
                == [case["case_sha256"] for case in receipt["randomized"]["cases"]]
            ),
            "passed_requested_gate": requested_passed,
            "passed_qualification_gate": qualification_passed,
        }
    )
    write_receipt(args.json_out, receipt)
    if args.json_out:
        print(f"receipt={args.json_out}")
    trained_correct = (
        receipt["trained"]["correct"] if receipt["trained"] is not None else None
    )
    randomized_correct = (
        receipt["randomized"]["correct"]
        if receipt["randomized"] is not None
        else None
    )
    print(
        "SUMMARY "
        f"mode={execution_mode} cases={len(cases)} "
        f"trained_correct={trained_correct} randomized_correct={randomized_correct} "
        f"qualification_passed={qualification_passed}"
    )
    return 0 if requested_passed else 1


if __name__ == "__main__":
    sys.exit(main())
