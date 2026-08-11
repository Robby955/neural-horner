# NeuralHorner research record

This directory preserves the evidence and decision trail for the direct-Horner
study. Start with [`FRONTIER_STATE.md`](FRONTIER_STATE.md) for the full ledger
and [`DIRECT_HORNER_RESEARCH.md`](DIRECT_HORNER_RESEARCH.md) for the experimental
design. Claims below are scoped to immutable artifacts and receipts; a passing
probe is never promoted into a broader claim.

## Current decision

- The only qualified and submitted artifact is
  `TrickyRex/bitserial-modmul-v8@301938693892043d66a8bf6ec60a9c5ab85549d4`.
- The direct two-pass L2-SP candidate is **stopped**. It repaired the known F11
  path but failed frozen-battery v3 at 509/512 attempted rows.
- The direct candidate was never substituted for v8 and was not submitted to
  SAIR. Its separate testing upload is research provenance, not a submission.
- No organizer compliance question was sent, and no ruling was requested or
  issued. The local question draft is archival only.
- No training has run after the stopped-candidate decision. The function-space
  screen and K-state miner remain unrun, conditional research ideas.

## Result in one table

| Object | Evidence | Decision |
|---|---|---|
| submitted v8 | official open-source scorer tiers 1--10 = 1.00 across three seeds; structured battery 759/768 on first contact | preserve as the only qualified/submitted incumbent |
| unchanged-weight direct schedule | fails `F11 * 1` at the same learned transition as original-schedule v8 | stopped architecture ablation |
| direct L2-SP `alpha=0.875` | F11 union 193,116/193,116 vectorized exact-prefix transitions; `ROB-CONFIRMED/UNARCHIVED` playground 100/100 on scored tiers; frozen battery v3 FAILED at 509/512 attempted | stopped; not submitted or qualified |
| direct function-space `alpha=0.9375` | earlier decisive-path result only; no v3 retention screen | unrun at the current fail-fast gate |
| K-state counterexample miner | design proposal only | no-training dry run not performed |

The direct schedule removes one learned operand-reduction pass. Its 20.001%
headline is a source-bound recurrent bit-work proxy on the pinned public
benchmark, not measured latency or an end-to-end speedup. Lean proves the exact
integer direct schedule and a conditional exact-prefix theorem; it does not
prove that any trained checkpoint realizes the transition universally.

The F11 union is preserved as an immutable historical receipt. Its internal
identity and count checks validate, but the exact older
`trace_f11_trajectories.py` and `held_out_battery.py` snapshots named by the
receipt are absent from this checkout and its recovery snapshot; both current
files have different hashes. It is therefore not independently replayable from
the present tree. Current runner sources define a new evidence boundary rather
than a replay of the historical union.

## Scorer and rules pin

The research receipts remain pinned to official scorer commit `82510bb`. A live
source refresh on 2026-08-11 found `origin/main` at
`99cac6ef5c2f82e53105ec0ddcbb9b8d37bf6fca`, two commits later. Those commits
add an explicit ban on deterministic operand pre-reduction and update examples;
the audited Docker and evaluation-pipeline files are byte-identical to the
receipt pin. NeuralHorner reduces operands through learned-cell passes rather
than deterministic pre-reduction. This source comparison is not an organizer
compliance ruling, and an old receipt must not be relabeled as evidence from the
new runner. Receipt:
[`receipts/rules_refresh_20260811.json`](receipts/rules_refresh_20260811.json),
SHA-256 `2ae8e5294adf410d2e857b61c2c63050286f9278a03740f839c3cba1af9364b4`.

## Frozen-development result

The 768-row structured battery was disjoint from v8's original training. Thus
v8's 759/768 score is valid **first-contact evidence** for that checkpoint.
Subsequent DAgger, repair, interpolation, and routing choices used the battery,
so it is now frozen development data and cannot be reused as a sealed
post-selection generalization test.

Frozen-battery v3 evaluated the repaired direct candidate in both orientations
with a planned total of 1,536 rows. It completed the first four
original-orientation families and then stopped fail-closed at 509/512 attempted
rows after errors on fixed-Hamming source indexes `003`, `035`, and `110`. The
remaining 1,024 rows were not run. The validator found no artifact, source,
fixture, or cache-provenance mismatch: this is model-failure evidence, not a
runner failure.

- receipt:
  [`receipts/frozen_battery_v3/battery_receipt.json`](receipts/frozen_battery_v3/battery_receipt.json),
  SHA-256 `1b9d94f9c3155f6fefaae34c268ad90fc2630f1cbc0a6cb505e23eec05ff673c`;
- validation:
  [`receipts/frozen_battery_v3/battery_validation.json`](receipts/frozen_battery_v3/battery_validation.json),
  SHA-256 `fb2525975b15ae256a593bcb594b2602f67b41aa2c3a6ddb3e440dbeb1cdcff9`.

Frozen-battery v1 and v2 are `INTERRUPTED`, not passes or model failures. V1
ended after 128 rows with 1,408 unrun; v2 completed no family after a preflight
cache-provenance defect was found.

## Causal result

The three v3 failures were frozen, then replayed under v8/repaired weights
crossed with original/direct schedules:

| Case | v8/original | v8/direct | repaired/original | repaired/direct | Classification |
|---|---:|---:|---:|---:|---|
| `003` | pass | pass | pass | fail | weight/schedule interaction required |
| `035` | pass | fail | pass | fail | direct schedule sufficient |
| `110` | pass | pass | fail | fail | repaired weights sufficient |

Each failing arm contains exactly one wrong exact-teacher transition. Case `035`
fails at the same direct-route transition (`scan_operand`, global step 4,245)
under both weight sets; interpolation worsens the local error from one wrong bit
at margin -2.1741 to seven wrong bits at margin -13.2857. The validated union is
[`receipts/fixed_hamming_four_way_v1/validation.json`](receipts/fixed_hamming_four_way_v1/validation.json),
SHA-256 `e47583c1e68d2985ffc0814b75473a8cc6c4350ecfb86591b5f9783a98a0d020`.

This separates three mechanisms and closes the promotion claim: fixing the F11
transition redistributed error, and removing a pass introduced an independent
vulnerable trajectory. The candidate is not a global repair.

## Playground evidence boundary

The immutable testing upload is
`TrickyRex/neural-horner-direct-l2sp-a0875@b5cee6f1592b1eeb9822e4773e5e294caf013f60`.
Rob confirms that it displayed `100/100` on each scored SAIR playground tier and
`40/100` on unscored Tier 0; he also confirms `40/100` for live-playground v8.
Label this `ROB-CONFIRMED/UNARCHIVED`: the screenshot/log, runtime, and evaluator
identity are not preserved here. Older local v8 receipts showing `60/100` on
Tier 0 belong to a different historical evaluation surface. None of these
playground facts makes the direct candidate submitted or qualified.

## Evidence map

1. [`FRONTIER_STATE.md`](FRONTIER_STATE.md) -- artifact identities, exact claim
   boundaries, receipt hashes, causal conclusions, and stopped state.
2. [`DIRECT_HORNER_RESEARCH.md`](DIRECT_HORNER_RESEARCH.md) -- direct-schedule
   derivation, F11 repair evidence, retention failure, and conditional future
   experiment design.
3. [`LITERATURE_LEDGER.md`](LITERATURE_LEDGER.md) -- mechanism-fit analysis and
   the XllentAI prior-art guard. The shared modulus-conditioned Horner recurrence,
   hard binary state, and held-out-prime transfer are not claimed as novel.
4. [`FRONTIER_RESEARCH_AGENT_PROMPT.md`](FRONTIER_RESEARCH_AGENT_PROMPT.md) --
   historical handoff prompt; the dated ledger and receipts override stale
   instructions in any prompt.
5. [`ORGANIZER_RUNTIME_QUESTION.md`](ORGANIZER_RUNTIME_QUESTION.md) -- historical
   unsent draft. No question was sent and no ruling was requested or issued.

Use the evidence terms literally: **verified**, **partial**, **conditional**,
**failed**, **unverified**, and **stopped**. In particular, do not rewrite
`509/512 attempted with 1,024 unrun` as a completed-battery rate, call the
20.001% proxy latency, or describe an unrun idea as progress.
