from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"


def load_script(name: str):
    path = SCRIPTS / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"test_gate_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


EQUALITY = load_script("check_batch_invariance")
COLLAPSE = load_script("verify_no_shortcut")


def git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), *arguments],
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def make_fake_pinned_scorer(tmp_path: Path) -> tuple[Path, Path, str]:
    repository = tmp_path / "scorer"
    source_text = {
        name: "\n" for name in EQUALITY.PINNED_SCORER_FILES
    }
    source_text["src/modchallenge/evaluation/decoder.py"] = (
        "def decode_answer(digits, *, base, prime, is_tier_zero=False):\n"
        "    return 7\n"
    )
    source_text["src/modchallenge/testgen/generator.py"] = (
        "def generate_public_test_set():\n"
        "    return []\n"
    )
    for package_init in (
        "src/modchallenge/__init__.py",
        "src/modchallenge/evaluation/__init__.py",
        "src/modchallenge/interface/__init__.py",
        "src/modchallenge/testgen/__init__.py",
    ):
        source_text.setdefault(package_init, "\n")
    for relative_name, contents in source_text.items():
        path = repository / relative_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(contents)

    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    git(repository, "add", ".")
    git(
        repository,
        "-c",
        "user.name=NeuralHorner Test",
        "-c",
        "user.email=tests@invalid.example",
        "commit",
        "-qm",
        "pinned scorer fixture",
    )
    commit = git(repository, "rev-parse", "HEAD")
    declared_name = "src/modchallenge/evaluation/decoder.py"
    declared_digest = hashlib.sha256(
        (repository / declared_name).read_bytes()
    ).hexdigest()
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(
        json.dumps(
            {
                "scorer_repository": "/expired/historical/scorer",
                "scorer_commit": commit,
                "source_sha256": {declared_name: declared_digest},
            }
        )
    )
    return repository, contract_path, commit


def strict_decoder(digits, *, base, prime, is_tier_zero=False):
    assert not is_tier_zero
    actual_base = prime if base == "p" else base
    if not isinstance(digits, list):
        raise ValueError("outer digit row is not a list")
    value = 0
    for digit in digits:
        if isinstance(digit, bool) or not isinstance(digit, int):
            raise ValueError("digit is not an int")
        if not 0 <= digit < actual_base:
            raise ValueError("digit is out of range")
        value = value * actual_base + digit
    if value >= prime:
        raise ValueError("decoded scored-tier value is not reduced")
    return value


def binary_digits(value: int) -> list[int]:
    return [int(digit) for digit in f"{value:b}"]


class ExactMockModel:
    def __init__(self, batch_size: int = 2) -> None:
        self.batch_size = batch_size
        self.preprocessed: list[tuple[str, str]] = []
        self.batch_sizes: list[int] = []

    def max_batch_size(self):
        return self.batch_size

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

    def predict_digits_batch(self, inputs):
        self.batch_sizes.append(len(inputs))
        return [
            binary_digits((int(a) * int(b)) % int(p))
            for a, b, p in inputs
        ]


def small_cases():
    return [
        EQUALITY.GateCase("c0", "unit", None, 0, 8, 9, 17, 4),
        EQUALITY.GateCase("c1", "unit", None, 1, 11, 13, 19, 10),
        EQUALITY.GateCase("c2", "unit", None, 2, 23, 29, 31, 16),
    ]


def test_equality_gate_covers_every_required_layout_with_decimal_inputs() -> None:
    model = ExactMockModel(batch_size=2)
    reload_model = ExactMockModel(batch_size=2)

    result = EQUALITY.evaluate_equality_layouts(
        model,
        lambda: reload_model,
        small_cases(),
        decoder=strict_decoder,
        output_base=2,
        seed=20260801,
        permutations=1,
        require_exact=True,
        require_raw_equality=True,
    )

    assert result["passed_qualification_gate"]
    assert result["output_cardinality_exact"]
    assert result["raw_equality"] and result["decoded_equality"]
    assert set(result["layouts"]) == {
        "singletons",
        "declared_batch",
        "reverse",
        "permutation_0",
        "operand_swap",
        "repeat",
        "reload",
    }
    assert result["layouts"]["declared_batch"]["batch_checks"] == [
        {
            "batch_start": 0,
            "expected": 2,
            "observed": 2,
            "outer_type": "list",
            "outer_list_exact": True,
            "cardinality_exact": True,
            "error": None,
        },
        {
            "batch_start": 2,
            "expected": 1,
            "observed": 1,
            "outer_type": "list",
            "outer_list_exact": True,
            "cardinality_exact": True,
            "error": None,
        },
    ]
    assert all(isinstance(value, str) for _, value in model.preprocessed)
    assert reload_model.batch_sizes == [2, 1]


class GeneratorOutputModel(ExactMockModel):
    def predict_digits_batch(self, inputs):
        return (binary_digits((int(a) * int(b)) % int(p)) for a, b, p in inputs)


class DecimalOutputModel(ExactMockModel):
    def predict_digits_batch(self, inputs):
        self.batch_sizes.append(len(inputs))
        return [
            [int(digit) for digit in str((int(a) * int(b)) % int(p))]
            for a, b, p in inputs
        ]


def test_equality_gate_rejects_unsized_or_wrong_cardinality_outputs() -> None:
    result = EQUALITY.evaluate_equality_layouts(
        GeneratorOutputModel(),
        lambda: ExactMockModel(),
        small_cases(),
        decoder=strict_decoder,
        output_base=2,
        seed=1,
        permutations=1,
        require_exact=True,
        require_raw_equality=True,
    )

    assert not result["output_cardinality_exact"]
    assert not result["passed_requested_gate"]
    first = result["layouts"]["singletons"]["batch_checks"][0][
        "batch_checks"
    ][0]
    assert first["outer_type"] == "generator"
    assert not first["outer_list_exact"]


def test_manifest_output_base_is_used_for_equality_and_collapse_arms() -> None:
    equality = EQUALITY.evaluate_equality_layouts(
        DecimalOutputModel(),
        lambda: DecimalOutputModel(),
        small_cases(),
        decoder=strict_decoder,
        output_base=10,
        seed=1,
        permutations=1,
        require_exact=True,
        require_raw_equality=True,
    )
    arm = COLLAPSE.evaluate_arm(
        DecimalOutputModel(),
        small_cases(),
        decoder=strict_decoder,
        output_base=10,
        label="trained",
    )

    assert equality["passed_qualification_gate"]
    assert arm["correct"] == arm["total"] == len(small_cases())


def test_reload_model_is_created_only_after_repeat_layout() -> None:
    events: list[str] = []

    class EventModel(ExactMockModel):
        def __init__(self, label: str) -> None:
            super().__init__(batch_size=8)
            self.label = label

        def predict_digits_batch(self, inputs):
            events.append(self.label)
            return super().predict_digits_batch(inputs)

    def reload_factory():
        events.append("load_reload")
        return EventModel("reload")

    result = EQUALITY.evaluate_equality_layouts(
        EventModel("active"),
        reload_factory,
        small_cases(),
        decoder=strict_decoder,
        output_base=2,
        seed=1,
        permutations=1,
        require_exact=True,
        require_raw_equality=True,
    )

    assert result["passed_qualification_gate"]
    assert events[-2:] == ["load_reload", "reload"]
    assert "reload" not in events[: events.index("load_reload")]


def test_reload_batch_contract_must_match_to_qualify() -> None:
    result = EQUALITY.evaluate_equality_layouts(
        ExactMockModel(batch_size=2),
        lambda: ExactMockModel(batch_size=3),
        small_cases(),
        decoder=strict_decoder,
        output_base=2,
        seed=1,
        permutations=1,
        require_exact=True,
        require_raw_equality=True,
    )

    assert result["raw_equality"] and result["decoded_equality"]
    assert not result["reload_batch_contract_equal"]
    assert not result["passed_qualification_gate"]


def test_deterministic_orders_are_complete_and_repeatable() -> None:
    first = EQUALITY.deterministic_orders(13, 99, 2)
    second = EQUALITY.deterministic_orders(13, 99, 2)

    assert first == second
    assert first["reverse"] == list(reversed(range(13)))
    assert sorted(first["permutation_0"]) == list(range(13))
    assert first["permutation_0"] != first["permutation_1"]


def _fixture_case(a: int, b: int, p: int, tier_id: int):
    return SimpleNamespace(
        a=str(a),
        b=str(b),
        p=str(p),
        expected=str((a * b) % p),
        tier_id=tier_id,
    )


def test_public_boundary_selection_reaches_every_scored_operand_limit() -> None:
    tiers = []
    for tier_id, operand_bits in EQUALITY.SCORED_TIER_OPERAND_BITS.items():
        rows = [
            _fixture_case(0, 1 << (operand_bits - 2), 17, tier_id),
            _fixture_case(1 << (operand_bits - 2), 0, 17, tier_id),
            _fixture_case(1, 1 << (operand_bits - 2), 17, tier_id),
            _fixture_case(1 << (operand_bits - 2), 1, 17, tier_id),
            _fixture_case((1 << operand_bits) - 1, 3, 17, tier_id),
        ]
        tiers.append(SimpleNamespace(tier_id=tier_id, cases=rows))

    selected = EQUALITY.select_public_boundary_cases(SimpleNamespace(tiers=tiers))

    assert len(selected) == 20
    boundaries = [case for case in selected if "operand-boundary" in case.case_id]
    assert {
        case.tier_id: max(case.a.bit_length(), case.b.bit_length())
        for case in boundaries
    } == EQUALITY.SCORED_TIER_OPERAND_BITS
    assert max(max(case.a.bit_length(), case.b.bit_length()) for case in selected) == 4096


class WrongMockModel(ExactMockModel):
    def predict_digits_batch(self, inputs):
        self.batch_sizes.append(len(inputs))
        return [[0] for _ in inputs]


def distinguishing_cases():
    rows = []
    for index, (a, b, p) in enumerate(((8, 9, 17), (11, 13, 19), (23, 29, 31))):
        expected = (a * b) % p
        assert expected not in (a % p, b % p, (a % p) * (b % p))
        rows.append(
            EQUALITY.GateCase(
                f"d{index}", "unit", None, index, a, b, p, expected
            )
        )
    return rows


def test_paired_collapse_uses_identical_cases_and_requires_both_arms() -> None:
    cases = distinguishing_cases()
    trained = COLLAPSE.evaluate_arm(
        ExactMockModel(),
        cases,
        decoder=strict_decoder,
        output_base=2,
        label="trained",
    )
    randomized = COLLAPSE.evaluate_arm(
        WrongMockModel(),
        cases,
        decoder=strict_decoder,
        output_base=2,
        label="randomized",
    )

    assert trained["correct"] == trained["total"] == len(cases)
    assert randomized["correct"] == 0
    assert all(trained["shortcut_checks"].values())
    assert [case["case_sha256"] for case in trained["cases"]] == [
        case["case_sha256"] for case in randomized["cases"]
    ]
    assert COLLAPSE.paired_gate_passed(
        trained,
        randomized,
    )
    assert not COLLAPSE.paired_gate_passed(
        trained,
        None,
    )
    randomized_perfect = {**randomized, "correct": randomized["total"]}
    assert not COLLAPSE.paired_gate_passed(trained, randomized_perfect)


class TinyCell(torch.nn.Module):
    def __init__(self, width: int = 3) -> None:
        super().__init__()
        self.projection = torch.nn.Linear(width, width)


class TinyEntry:
    def __init__(self) -> None:
        self.model = TinyCell()
        self.device = torch.device("cpu")


def test_randomized_state_hash_is_deterministic_bound_and_distinct(tmp_path) -> None:
    torch.save({"config": {"width": 3}}, tmp_path / "weights.pt")
    module = SimpleNamespace(Cell=TinyCell)
    first = TinyEntry()
    second = TinyEntry()
    trained_hash = EQUALITY.state_dict_sha256(first)

    first_hash = COLLAPSE.randomize_loaded_model(first, module, tmp_path, 12345)
    second_hash = COLLAPSE.randomize_loaded_model(second, module, tmp_path, 12345)

    assert first_hash == second_hash
    assert first_hash == EQUALITY.state_dict_sha256(first)
    assert first_hash != trained_hash


def test_artifact_tree_hash_binds_extra_helpers_but_ignores_receipts(tmp_path) -> None:
    (tmp_path / "manifest.json").write_text('{"entry_class":"model.Entry"}\n')
    (tmp_path / "model.py").write_text("from helper import VALUE\n")
    (tmp_path / "helper.py").write_text("VALUE = 1\n")
    (tmp_path / "weights.pt").write_bytes(b"weights")
    receipts = tmp_path / "receipts"
    receipts.mkdir()
    (receipts / "old.json").write_text("{}\n")

    receipt_output = receipts / "new.json"
    before = EQUALITY.artifact_tree_identity(
        tmp_path,
        excluded_paths=(receipt_output,),
    )
    receipt_output.write_text('{"status":"running"}\n')
    after_receipt = EQUALITY.artifact_tree_identity(
        tmp_path,
        excluded_paths=(receipt_output,),
    )
    (tmp_path / "helper.py").write_text("VALUE = 2\n")
    after_helper = EQUALITY.artifact_tree_identity(tmp_path)

    assert before == after_receipt
    assert before != after_helper
    assert "helper.py" in before["file_sha256"]


def test_two_bit_generated_screen_uses_three_not_two() -> None:
    public = [
        EQUALITY.GateCase("p2", "unit", 1, 0, 1, 1, 2, 1),
        EQUALITY.GateCase("p3", "unit", 1, 1, 2, 2, 3, 1),
    ]

    cases = COLLAPSE.build_paired_cases(
        public,
        n=1,
        prime_bits=2,
        max_model_width=8,
        seed=2026,
    )

    generated = cases[-1]
    assert generated.p == 3
    assert generated.expected not in (
        generated.a % generated.p,
        generated.b % generated.p,
        (generated.a % generated.p) * (generated.b % generated.p),
    )


def test_pinned_decoder_binding_is_hashed_and_rejects_non_list_output() -> None:
    scorer_repository = os.environ.get("NEURAL_HORNER_SCORER_REPO")
    identity, decoder, _generator = EQUALITY.load_pinned_scorer(
        scorer_repository=(
            Path(scorer_repository) if scorer_repository is not None else None
        )
    )

    assert identity["status"] == "verified"
    if scorer_repository is not None:
        assert identity["repository"] == str(Path(scorer_repository).resolve())
    assert "src/modchallenge/evaluation/decoder.py" in identity[
        "qualification_source_sha256"
    ]
    valid, value, error = EQUALITY.decode_output(
        decoder,
        [1, 0],
        EQUALITY.GateCase("x", "unit", None, 0, 1, 2, 17, 2),
    )
    assert valid and value == 2 and error is None
    valid, value, error = EQUALITY.decode_output(
        decoder,
        (1, 0),
        EQUALITY.GateCase("x", "unit", None, 0, 1, 2, 17, 2),
    )
    assert not valid and value is None
    assert error["type"] == "MalformedOutput"


def test_reconstructed_scorer_override_is_rooted_and_commit_bound(
    tmp_path: Path,
) -> None:
    repository, contract_path, commit = make_fake_pinned_scorer(tmp_path)

    identity, decoder, generator = EQUALITY.load_pinned_scorer(
        contract_path,
        repository,
    )

    assert identity["repository_root_exact"]
    assert identity["worktree_clean_including_untracked"]
    assert identity["commit"] == commit
    assert identity["decoder_path"] == str(
        (repository / "src/modchallenge/evaluation/decoder.py").resolve()
    )
    assert decoder([], base=2, prime=3) == 7
    assert generator() == []


def test_scorer_override_rejects_non_root_subdirectory(tmp_path: Path) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    shadow = repository / "shadow"
    shadow.mkdir()

    with pytest.raises(RuntimeError, match="exact Git worktree root"):
        EQUALITY.load_pinned_scorer(contract_path, shadow)


def test_scorer_override_rejects_wrong_commit(tmp_path: Path) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    (repository / "marker.txt").write_text("second commit\n")
    git(repository, "add", "marker.txt")
    git(
        repository,
        "-c",
        "user.name=NeuralHorner Test",
        "-c",
        "user.email=tests@invalid.example",
        "commit",
        "-qm",
        "different commit",
    )

    with pytest.raises(RuntimeError, match="checkout identity mismatch"):
        EQUALITY.load_pinned_scorer(contract_path, repository)


def test_scorer_override_rejects_dirty_tracked_source(tmp_path: Path) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    decoder_path = repository / "src/modchallenge/evaluation/decoder.py"
    decoder_path.write_text(decoder_path.read_text() + "# dirty\n")

    with pytest.raises(RuntimeError, match="checkout identity mismatch"):
        EQUALITY.load_pinned_scorer(contract_path, repository)


def test_scorer_override_rejects_ignored_shadow_source(tmp_path: Path) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    shadow_name = "src/modchallenge/evaluation/shadow.py"
    exclude = repository / ".git/info/exclude"
    exclude.write_text(exclude.read_text() + f"\n/{shadow_name}\n")
    shadow = repository / shadow_name
    shadow.write_text("raise RuntimeError('shadow code executed')\n")
    assert git(repository, "status", "--porcelain", "--untracked-files=all") == ""

    with pytest.raises(RuntimeError, match="uncommitted or cached files"):
        EQUALITY.load_pinned_scorer(contract_path, repository)


def test_scorer_override_rejects_ignored_sibling_import_shadow(
    tmp_path: Path,
) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    shadow_name = "src/sympy.py"
    exclude = repository / ".git/info/exclude"
    exclude.write_text(exclude.read_text() + f"\n/{shadow_name}\n")
    shadow = repository / shadow_name
    shadow.write_text("raise RuntimeError('dependency shadow executed')\n")
    assert git(repository, "status", "--porcelain", "--untracked-files=all") == ""

    with pytest.raises(RuntimeError, match="uncommitted or cached files"):
        EQUALITY.load_pinned_scorer(contract_path, repository)


def test_scorer_override_compares_assume_unchanged_source_to_head(
    tmp_path: Path,
) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    relative_name = "src/modchallenge/evaluation/decoder.py"
    decoder_path = repository / relative_name
    git(repository, "update-index", "--assume-unchanged", relative_name)
    decoder_path.write_text(decoder_path.read_text() + "# hidden mutation\n")
    assert git(repository, "status", "--porcelain", "--untracked-files=all") == ""

    with pytest.raises(RuntimeError, match="working-tree/commit mismatch"):
        EQUALITY.load_pinned_scorer(contract_path, repository)


def test_scorer_override_rejects_declared_hash_mismatch(tmp_path: Path) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    contract = json.loads(contract_path.read_text())
    contract["source_sha256"]["src/modchallenge/evaluation/decoder.py"] = "0" * 64
    contract_path.write_text(json.dumps(contract))

    with pytest.raises(RuntimeError, match="scorer source hash mismatch"):
        EQUALITY.load_pinned_scorer(contract_path, repository)


def test_scorer_override_ignores_git_replacement_objects(tmp_path: Path) -> None:
    repository, contract_path, original_commit = make_fake_pinned_scorer(tmp_path)
    relative_name = "src/modchallenge/evaluation/decoder.py"
    decoder_path = repository / relative_name
    git(repository, "checkout", "-qb", "replacement-fixture")
    decoder_path.write_text(
        "def decode_answer(digits, *, base, prime, is_tier_zero=False):\n"
        "    return 99\n"
    )
    git(repository, "add", relative_name)
    git(
        repository,
        "-c",
        "user.name=NeuralHorner Test",
        "-c",
        "user.email=tests@invalid.example",
        "commit",
        "-qm",
        "replacement object fixture",
    )
    replacement_commit = git(repository, "rev-parse", "HEAD")
    git(repository, "checkout", "-q", "--detach", original_commit)
    git(repository, "replace", original_commit, replacement_commit)
    replaced_view = subprocess.run(
        ["git", "-C", str(repository), "show", f"{original_commit}:{relative_name}"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    assert "return 99" in replaced_view

    identity, decoder, _generator = EQUALITY.load_pinned_scorer(
        contract_path,
        repository,
    )

    assert identity["commit"] == original_commit
    assert decoder([], base=2, prime=3) == 7


def test_scorer_override_discards_preloaded_external_module(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    malicious = SimpleNamespace(
        __file__=str(tmp_path / "external_decoder.py"),
        decode_answer=lambda *_args, **_kwargs: 99,
    )
    monkeypatch.setitem(
        sys.modules,
        "modchallenge.evaluation.decoder",
        malicious,
    )

    identity, decoder, _generator = EQUALITY.load_pinned_scorer(
        contract_path,
        repository,
    )

    assert decoder([], base=2, prime=3) == 7
    assert identity["decoder_path"] != malicious.__file__


def test_scorer_override_rejects_wrong_import_origin(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repository, contract_path, _commit = make_fake_pinned_scorer(tmp_path)
    real_import = EQUALITY.importlib.import_module

    def import_from_wrong_path(name: str):
        if name == "modchallenge.evaluation.decoder":
            return SimpleNamespace(
                __name__=name,
                __file__=str(
                    repository / "src/modchallenge/evaluation/__init__.py"
                ),
                decode_answer=lambda *_args, **_kwargs: 99,
            )
        return real_import(name)

    monkeypatch.setattr(EQUALITY.importlib, "import_module", import_from_wrong_path)
    with pytest.raises(RuntimeError, match="wrong source path"):
        EQUALITY.load_pinned_scorer(contract_path, repository)
