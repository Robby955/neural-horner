# Direct two-pass NeuralHorner study: stopped promotion record

Evidence produced: 2026-08-01 through 2026-08-02

At the time of the study, the submission deadline was August 12, 2026, 23:59
AoE. The pinned official contract allowed a deterministic feedback-free loop
that feeds input tokens through a learned model, while requiring the complete
1,100-problem run to finish within five minutes. Rules and evaluation code were
still experimental, so every receipt below is bound to an immutable scorer
commit rather than asserted as timeless competition state.

The published Dockerfile at current commit `82510bb` installs CPU-only PyTorch,
limits the example run to four CPUs, and does not expose a GPU. The same rules
recommend GPU batching. This hardware contradiction was unresolved and would
remain a deployment blocker for any future candidate: a CUDA timing receipt is
useful engineering evidence, but it does not prove compatibility with the
published CPU sandbox. No organizer clarification or compliance ruling was
requested or issued.

## Artifact record and final decision

| Role | Artifact | Current decision |
| --- | --- | --- |
| incumbent | `TrickyRex/bitserial-modmul-v8@301938693892043d66a8bf6ec60a9c5ab85549d4` | Keep unchanged; the only qualified and submitted artifact. |
| stopped repair | `candidates/direct_two_pass_l2sp_a0875` | F11 vectorized exact-prefix gate passed, but frozen-battery v3 failed at 509/512 attempted rows. Not submitted or qualified. |
| unrun conditional screen | `candidates/direct_two_pass_funcspace_a09375` | Earlier decisive-path result only; current fail-fast retention/F11 screen not run. |

`candidates/direct_two_pass` isolates the schedule with unchanged v8 weights,
but a known F11 x 1 counterexample stops it from occupying a slot.
`candidates/original_three_pass_l2sp_a0875` is the byte-identical-weight control
for the failed repair and is retained for causal evaluation, not as a slot.

Rob authorized one separate Hugging Face testing upload. The exact candidate is
`TrickyRex/neural-horner-direct-l2sp-a0875@b5cee6f1592b1eeb9822e4773e5e294caf013f60`;
its four package files were downloaded and rehashed after upload. It is not a
SAIR slot update or qualification claim.

Rob subsequently confirms that this immutable revision completed a SAIR
playground evaluation with `100/100` on each scored tier 1--10 and `40/100` on
the unscored Tier-0 pure-multiplication diagnostic. He confirms live-playground
v8 also displays `40/100`, so this is no live diagnostic regression. The result
is `ROB-CONFIRMED/UNARCHIVED`: no exact screenshot/log, runtime, or evaluator
identity is preserved here. Older
local open-source-scorer v8 receipts record `60/100`; that historical surface
must not be substituted for the live playground comparison.

The playground result does not survive the retention gate. Frozen-battery v3
subsequently failed at 509/512 attempted rows on fixed-Hamming cases `003`,
`035`, and `110`, with 1,024 planned rows unrun. Its validator found no identity
or cache-provenance blocker. Promotion is therefore `STOPPED`; the immutable
testing upload remains a negative research artifact and was never a SAIR
submission update.

## What changed scientifically

### 1. Remove a learned pass without changing the learned primitive

The v8 cell models one transition

`s' = (2*s + d*x) mod p`.

The incumbent reduces both operands, then scans a fixed-width residue. The new
schedule reduces one raw operand and scans the other raw operand with that
residue as `x`. The exact-integer recurrence therefore uses `A+B` transitions
instead of `A+B+Leff`. Lean now proves both operand orientations. The unproved
bridge remains the learned cell's equality to the integer transition.

On the official public benchmark at scorer commit `82510bb`, a recurrent
bit-work proxy falls by 20.001% across scored tiers. This is an architecture
count, not measured latency. Tier 0 is excluded from the headline because some
diagnostic cases lie outside the candidate's supported modulus range. Receipt:
`receipts/public_schedule_cost_82510.json`.

### 2. Canonicalize operand orientation

The implementation reduces the longer operand and scans the shorter one, using
lexicographic order to break equal-length ties. Swapping `a` and `b` therefore
executes the identical learned trajectory and emits identical raw digits. This
is stronger than merely testing both orientations after the fact.

The nine legacy F11 companions are roughly 4092-4096 bits, so long-reduce routing
keeps their 2049-bit F11 value in the scan phase. Because those cases influenced
the router, they are now development data rather than held-out evidence.

This is not a general F11 repair. When F11 is paired with a shorter companion,
canonical routing reduces F11 and reaches the known bad maximum-width transition.
For `F11 * 1`, the immediate scan of `1` is exact relative to that wrong residue,
so the final answer remains wrong. The unchanged-weight direct candidate is
therefore a stopped architecture ablation. Every repaired candidate must replay
the fixed companion-width diagnostic at 1, 32, 256, 1024, 2048, 2049, 3072,
and 4096 bits, equal-length tie cases, and both input orientations. Those known
cases are development evidence for later candidates, not fresh sealed evidence.

### 3. Test conservative checkpoint interpolation, then retain the failure

The stopped repair candidate interpolates v8 toward the L2-SP `lambda=1e-3`
specialist at `alpha=0.875`. A CPU-fp32 one-step screen is exact on 68/68 rows,
representing 56 unique transition tuples; it explicitly excludes the 12 final
disjoint rows and is not rollout evidence. That historical omission is not the
current F11 qualification remainder. The independent function-space candidate
uses `alpha=0.9375`.

The stronger candidate-owned decisive-path results supersede the injected
two-step probe. Exact-prefix induction checks every transition from the exact
semantic prefix. The historical frozen-v3 receipt records the direct L2-SP candidate as exact on
4,100/4,100 transitions across both `F11 * 1` orientations, with minimum signed
correct-target margin 5.01655. The byte-identical original-schedule control is
exact on 8,196/8,196 transitions across both orientations, with minimum margin
5.01657. Their joint success is strong causal evidence that checkpoint repair,
rather than routing, repairs this case. The independent function-space direct
candidate is exact on an earlier-run decisive path, with minimum margin 4.9199,
but it has no v3 receipt.

Before the retention failure, the historical v3 receipts also recorded the seven other companion widths in both orientations:
14/14 results and 53,840/53,840 transitions, minimum signed margin 5.01657,
exact observed routes, fp32 captured logits, and no artifact/source mutation.
It likewise certifies the three equal-length ties in both orientations: 6/6
results and 24,588/24,588 transitions at the same minimum margin, then all nine
legacy cases: 18/18 results and 110,588/110,588 transitions, minimum margin
7.19223.

The historical strict-union receipt records `status=validated_exact` for four v3
component receipts under one verified scorer checkout/import origin: 20 unique
numerical cases, 40 unique oriented cases, and 193,116/193,116 exact
transitions. The components contribute 4,100 decisive, 53,840 companion,
24,588 tie, and 110,588 legacy transitions. All routes are exact, logits are
fp32, margins are positive, and neither artifacts nor sources mutate. The union
binds runner
`87e0b15e8a234ea03c43f5621d11fc1b83da5a1480d20efbc985c29ad22268f1`,
source set
`0ca78bb1fdfc797ff559aaca0f623351022f92fb0a6208d892050e8832dc6837`,
artifact set
`9b0ba4f1c6ff5ed8ccf7b64b5b173baf10b8881c407ce430ec3338adcc0e06fc`,
and full case set
`54f9d2342fcdbfa6fc21a5d6a6560d13364059be3c48174a6c66a407834a86ee`;
the four component receipt hashes are embedded in the union. Receipt:
`receipts/f11_primary_direct_routebound_v3_union.json`, SHA-256
`5890134071d914c3bd4e77852c7a3379d56b41c30543871e0a7686708b799920`.
This recorded completion of the vectorized exact-prefix induction gate. It is not a literal
sequential, batched, or official-inference receipt, and the later retention
failure makes those prospective promotion gates moot for this candidate.

This union is immutable historical evidence, not currently replayable. Its
runner hash is `87e0b15...` and its bound `held_out_battery.py` hash is
`e1f0568...`; the present files hash to `45287db...` and `3f63dfb...`,
respectively, and the historical snapshots are absent from both this checkout
and its recovery snapshot. The receipt's internal identities and counts remain
valid, but today's sources define a new evidence boundary.

The original-schedule L2-SP control's earlier companion receipt is likewise
exact on 14/14 oriented results and 82,512/82,512 transitions, with minimum
margin 4.9165. Its larger count and slightly lower margin come from the extra
learned operand reduction. It remains diagnostic unless reproduced with a newly
frozen comparable runner.

The decisive four-way causal matrix is therefore:

| Weights | Original three-pass | Canonical direct two-pass |
| --- | --- | --- |
| submitted v8 | 4,097/4,098 exact per orientation; one failure, 642 wrong bits, minimum margin -20.8369 | 2,049/2,050 exact per orientation; one failure, 642 wrong bits, minimum margin -20.8369 |
| L2-SP `alpha=0.875` | 4,098/4,098 exact per orientation; minimum margin 5.0166 | 2,050/2,050 exact per orientation; minimum margin 5.0166 |

The identical learned transition fails under both v8 schedules and passes under
both byte-identical repaired schedules. This isolates checkpoint repair as the
cause of success on the decisive case. Frozen-battery v3 then proves that the
repair does not generalize across the frozen development suite.

Checkpoint compatibility is fail-closed over `L`, configuration, state-dict
order, tensor names, shapes, and dtypes. Derived checkpoints strip stale source
steps and scores; source metadata remains only in `provenance.json`. Full
evaluation, not specialist training probes, decides whether either survives.

## Evidence completed locally

These checks were completed during the August 1--2 study and are reported from
their receipts; they were not rerun for this documentation closeout.

- The candidate and fail-closed research-gate suite passes locally: 90 tests.
- The strict F11 union validator rejects unverified scorer discovery and mixed
  runner/source/artifact/case identities, rechecks the imported scorer origin,
  and enforces exact routes, positive margins, unique cases, and cardinalities.
  The historical v3 decisive, companion, tie, and legacy receipts recorded this
  stronger boundary together as one `validated_exact` 20-case/40-orientation
  union.
- Exact Lean package builds. Both direct schedule theorems and the new composed
  exact-prefix certificate theorem have only Mathlib's standard `propext` and
  `Quot.sound` dependencies; the underlying single-scan induction theorem is
  axiom-free. These are conditional algorithm/trajectory results, not universal
  checkpoint proofs.
- Official scorer commit `82510bba00a1126649bd76dd1a451f14d0b3eb60`:
  116 tests pass and 1 is skipped.
- All four local candidate manifests validate, load at `L=2048` with 470,849
  parameters, and produce zero official static-analysis findings.
- Unchanged-weight candidate: its current artifact/runner/case-bound smoke is
  16/16 exact and bitwise invariant under batching, permutations, and operand
  swaps. Its refreshed p33 no-shortcut screen is 4/4 with all exclusions true
  and 0/4 after deterministic randomization. These are small-width smoke gates;
  exact-prefix induction verifies 2,048 exact candidate transitions before the
  first divergence in each decisive orientation: 2,049/2,050 transitions are
  exact, 642 bits are wrong at global/phase step 2,048, and minimum signed margin
  is -20.8369. This stops the artifact.
- The original three-pass v8 program likewise verifies 4,097/4,098 transitions
  in each orientation and fails the same learned transition with 642 wrong bits
  and minimum signed margin -20.8369. This completes the decisive schedule/
  weight causal matrix with one historically comparable, identity-bound runner.
- The L2-SP candidate's provenance-bound one-step screen is 68/68 exact over 56
  unique transition tuples after excluding exactly 12 final-disjoint rows. Its
  complete decisive path is 4,100/4,100 exact across both orientations under
  v3 candidate-owned prefix induction, minimum signed margin 5.01655. Its v3
  companion receipt adds 14/14 oriented results and 53,840/53,840 exact
  transitions, minimum margin 5.01657; its v3 tie receipt adds 6/6 results and
  24,588/24,588 transitions at the same margin; its v3 legacy receipt adds 18/18
  results and 110,588/110,588 transitions, minimum margin 7.19223. The strict
  union validator certifies the 20-case/40-orientation induction gate.
- The L2-SP and function-space candidates are each 10/10 exact on the same
  small-width singleton/batch/permutation/swap smoke. L2-SP is additionally 4/4
  on a 33-bit no-shortcut screen and 0/4 after deterministic randomization.
- The function-space candidate's small-width batch receipt must match its final
  rebuilt checkpoint; the current receipt does. Its decisive path is
  2,050/2,050 exact in each orientation, minimum signed margin 4.9199.
- The original-schedule L2-SP control has the exact same weight-file SHA-256 as
  the direct L2-SP candidate, reproduces its 68/68 one-step screen, and is 10/10
  exact with decoded-value batch/permutation/swap invariance. Its mixed-width
  raw digit lists differ only in leading-zero length, as expected from the
  incumbent's batch-wide dynamic width. Its decisive path is 8,196/8,196 exact
  across both orientations under v3, minimum signed margin 5.01657. Its older
  seven-companion result is 14/14 and 82,512/82,512, minimum margin 4.9165, but
  remains diagnostic pending v3 replay.

These smoke and vectorized-induction gates did not establish literal sequential
or batched inference, maximum-width retention, fresh sealed generalization, or
five-minute completion on final hardware. The next maximum-width retention gate
produced the validated failure below.

Frozen-battery v1 is `INTERRUPTED`, not passed. Its preserved receipt still says
`running`, but the process ended after the original-orientation Fibonacci family
completed 128/128 exact; the remaining 1,408 rows did not run. Frozen-battery v2
is also `INTERRUPTED`, not passed: it was stopped immediately when preflight
review found that the old runner did not bind its cache inventory into the
receipt. It completed no family and supplies no model-failure evidence.

Frozen-battery v3 is `FAILED`, not interrupted. It stopped fail-closed at
509/512 after the repaired-direct candidate missed fixed-Hamming source indexes
3, 35, and 110; the remaining 1,024 rows did not run. Its one-shot validator
found no identity or cache blocker, so this is a genuine retention failure.
Permanent receipts are under `research/receipts/frozen_battery_v3/`.

The three failures are frozen in
`research/fixtures/fixed_hamming_v3_failures.json`, SHA-256
`4fa3322471e1d4185d53cbd301858d416ba44b8a6ee0d110e1d2a3b8d5c7113a`.
The four-way v8/repaired x original/direct exact-prefix trace is validated at
`research/receipts/fixed_hamming_four_way_v1/validation.json`, SHA-256
`e47583c1e68d2985ffc0814b75473a8cc6c4350ecfb86591b5f9783a98a0d020`:

| Case | v8 original | v8 direct | repaired original | repaired direct | Causal classification |
|---|---:|---:|---:|---:|---|
| source 003 | pass | pass | pass | fail | weight/schedule interaction required |
| source 035 | pass | fail | pass | fail | direct schedule sufficient |
| source 110 | pass | pass | fail | fail | repaired weights sufficient |

Each failing arm has exactly one wrong teacher-state transition. Source 035
fails at the same direct-route transition under both weight sets
(`scan_operand`, global step 4,245); repair worsens the local error from one bit
at margin -2.1741 to seven bits at margin -13.2857. CUDA remains a separate
parity/timing gate only if GPU evaluation is confirmed.

## Fail-closed promotion matrix

This was the predeclared matrix. The L2-SP direct candidate stopped at step 4;
later steps are unmet conditions, not pending evidence or a current run plan.

1. **Identity and compliance**: immutable artifact hashes, current scorer pin,
   scorer import/source digest match, manifest schema, artifact size, official
   static check, no out-of-directory reads.
2. **Program invariants**: unit tests, decoded singleton/batch/reverse/
   permutation/swap equality, preprocessing isolation, and repeated-run
   determinism. Require raw digit and trajectory equality for the canonical
   direct schedule; report benign leading-zero width changes separately for the
   original dynamic-width schedule.
3. **Known wall**: all nine frozen F11 development cases plus eight diagnostic
   companion-width cases, equal-length ties, both orientations, and every
   transition traced; require the exact manifest-program transition count and
   exactly 20 unique numerical cases / 40 oriented results.
4. **Adversarial battery**: v1 and v2 were interrupted; v3 is a validated model
   failure at 509/512 attempted rows. Promotion is stopped.
5. **Fresh generalization**: generate and hash a corrected battery, keep it
   sealed until the recipe, interpolation, and checkpoint are frozen, then open
   it once under the predeclared gate. The legacy battery cannot fill this role.
6. **Official geometry**: public 1100 cases plus at least three fresh seeds at
   100% on scored tiers 1-10. Tier 0 remains diagnostic and must be reported
   separately.
7. **Learned-computation evidence**: randomized-weight and perturbation sweeps
   collapse accuracy; latency scales with token count; no classical shortcut is
   introduced.
8. **Deployment parity**: obtain the organizers' final hardware/image contract,
   then run that exact environment under 300 seconds with headroom. Test the
   CUDA autocast path separately if GPU evaluation is confirmed. Record hardware,
   PyTorch, scorer, seed, artifact, and wall time in every receipt.

Promotion would require every gate. A partial run, Apple MPS smoke, inherited
v8 receipt, or specialist training metric is not a substitute. The current
candidate is stopped and no promotion run is authorized by this document.

## Protocol templates for a new evidence boundary

These templates preserve the study's fail-closed protocol; they cannot reproduce
the historical F11 union because its exact runner/helper snapshots are absent.
Any execution with present sources is a new experiment with new identities, not
a replay. These are not a current benchmark or promotion plan. The
official-evaluation command is a preparation template, not a runnable claim.
Before invoking it, create cache-free immutable copies of the submission package
and detached scorer checkout, plus a cache-free source copy containing
`run_official_eval.py` and `submission_utils.py`. Choose a unique receipt path
under `research/receipts/` that is outside the submission, scorer, and wrapper
copies, plus a fresh empty bytecode-cache prefix so Python cannot read a stale
adjacent `pyc`. Set the six task-specific variables below to those prepared
absolute paths; do not point them at live working directories.

```bash
: "${NEURAL_HORNER_SUBMISSION_COPY:?set a cache-free immutable package copy}"
: "${NEURAL_HORNER_SCORER_COPY:?set a clean detached scorer copy}"
: "${NEURAL_HORNER_WRAPPER_COPY:?set a cache-free wrapper/helper source copy}"
: "${NEURAL_HORNER_OFFICIAL_RECEIPT:?set a unique external research/receipts path}"
: "${NEURAL_HORNER_PYCACHE_PREFIX:?set a fresh empty bytecode-cache prefix}"
: "${NEURAL_HORNER_PYTHON:?set the absolute Python executable path}"

PYTHONDONTWRITEBYTECODE=1 \
PYTHONPYCACHEPREFIX="$NEURAL_HORNER_PYCACHE_PREFIX" \
PYTHONPATH="$NEURAL_HORNER_SCORER_COPY/src" \
"$NEURAL_HORNER_PYTHON" "$NEURAL_HORNER_WRAPPER_COPY/run_official_eval.py" \
  "$NEURAL_HORNER_SUBMISSION_COPY" \
  --scorer-repo "$NEURAL_HORNER_SCORER_COPY" \
  --expected-scorer-sha 82510bba00a1126649bd76dd1a451f14d0b3eb60 \
  --total 1100 --timeout 300 --require-scored-perfect \
  --json-out "$NEURAL_HORNER_OFFICIAL_RECEIPT"

# Future independently authorized rerun only. The prior battery is terminal
# FAILED and candidate promotion is stopped. Prepare a fresh cache-free package
# copy and set this variable to its absolute path.
: "${NEURAL_HORNER_BATTERY_PACKAGE:?set an immutable package copy}"
: "${NEURAL_HORNER_BATTERY_SOURCE_COPY:?set a cache-free research source copy}"
: "${NEURAL_HORNER_BATTERY_RECEIPT:?set a unique external receipt path}"
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$NEURAL_HORNER_SCORER_COPY/src" \
"$NEURAL_HORNER_PYTHON" \
  "$NEURAL_HORNER_BATTERY_SOURCE_COPY/scripts/held_out_battery.py" \
  "$NEURAL_HORNER_BATTERY_PACKAGE" \
  --n 128 --orientation both --require-exact --qualification-l2048-n128 \
  --json-out "$NEURAL_HORNER_BATTERY_RECEIPT"

# Future independently authorized sequential gate only. The source copy must
# contain one newly frozen runner, helper, and research case set. This cannot be
# labeled a replay of the historical v3 union.
: "${NEURAL_HORNER_F11_PACKAGE:?set an immutable package copy}"
: "${NEURAL_HORNER_F11_SOURCE_COPY:?set a cache-free research source copy}"
: "${NEURAL_HORNER_F11_RECEIPT:?set a unique external receipt path}"
: "${NEURAL_HORNER_F11_DEVICE:?set cpu, mps, or cuda from the target contract}"
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH="$NEURAL_HORNER_SCORER_COPY/src" \
"$NEURAL_HORNER_PYTHON" \
  "$NEURAL_HORNER_F11_SOURCE_COPY/scripts/trace_f11_trajectories.py" \
  "$NEURAL_HORNER_F11_PACKAGE" --mode sequential --orientation both \
  --device "$NEURAL_HORNER_F11_DEVICE" --json-out "$NEURAL_HORNER_F11_RECEIPT"
```

A new complete F11 runner requires exactly 20 unique numerical cases and 40
oriented results: nine frozen legacy cases, eight companion widths including the
decisive width-1 case, and three equal-length ties. Its default induction mode
evaluates every transition from the exact semantic prefix: if all candidate
outputs are exact, the literal deterministic rollout follows by induction; if a
failure occurs after an all-exact prefix, that first divergence is also valid
for free-running inference. Post-divergence teacher-prefix evaluations are
labeled separately. Sequentially replay every survivor before promotion; no
borrowed or injected prefix can satisfy either gate.
Select `--device cpu` instead if the confirmed final contract is CPU. The
corresponding current function-space screen has not run. A future study would
repeat the exact matrix for that artifact and its original-schedule control. Do
not tune on a private
or secret-seed result; if interpolation fails, use public counterexamples and
the frozen adversarial suite for the next training run.

These commands do not substitute for the published sandbox. The repository's
current sandbox is CPU-only; do not claim a deployable five-minute result until
the final official hardware contract is confirmed and replayed. The wrapper
records process-level peak memory but cannot observe the model's actual inference
device or dtype; its passing local gate is therefore not the final deployment
gate even when accuracy, geometry, determinism, and wall time pass.

## Counterexample-guided branch (dry-run stage only)

The complete F11 induction gate passed, but the frozen retention battery then
returned a validated candidate failure. Two possible follow-ups are recorded,
not executed. First, a zero-training function-space screen would test
`035 -> 110 -> 003`, then the complete v3 F11 set only if all three pass. Second,
a K-state miner could be designed as a no-training dry run. Neither has run, and
no new training has run. Before any miner work, freeze the three divergence
states, incumbent/F11 retention set, sealed split, candidate-pool construction,
measured-compute accounting, paired same-K random-selection control, and STOP
criteria.

Do not start a broad retrain. First classify the first divergent transitions by
phase, width, state sparsity, signed distance to the modular boundary,
carry/borrow length, subtract count, backend, and minimum correct-bit margin.
Only after a separately authorized dry run survives its predeclared controls
would one retention-constrained semantic-state aggregation experiment from v8
be scientifically defensible: mix equal numbers of exact-prefix states and
detached model-visited residue states from fresh nonsealed rollouts, label both
with the exact transition, and retain incumbent replay plus L2-SP anchoring.
That experiment has not run. Its planned comparison is equal compute and at
least three seeds against IID-only and exact-prefix-only post-training, selected
lexicographically by no retention regression, no fresh-rollout regression, then
maximum minimum correct-bit margin. Mechanistic Group DRO is conditional on
continued failure relocation; unstructured CVaR is not the starting point. Mean
loss is not a promotion metric, and literal hidden-state passing is inapplicable
because the BiGRU hidden state resets at every arithmetic transition.

This design must be compared with XllentAI's pinned-current reported baseline
at `f704813dc64d7c04cc9ba7dbf9a3a281c431628d`, which already uses true Horner
trajectory states, value/bit-length distribution matching, worst-bit margin,
distillation, weight soups, and carry-aware TCNs. A TCN is worth a small
equal-parameter width-matched probe for architectural diversity, not an
immediate 2048-bit training campaign. See `LITERATURE_LEDGER.md`; the XllentAI
results are model-card claims and were not independently rerun here.

Official references:

- https://competition.sair.foundation/competitions/modular-arithmetic-challenge/overview
- https://competition.sair.foundation/competitions/modular-arithmetic-challenge/evaluation-setup
- https://github.com/SAIRcompetition/modular-arithmetic-challenge
