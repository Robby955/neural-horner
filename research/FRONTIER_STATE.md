# NeuralHorner frontier state

Evidence cutoff: immutable receipts through 2026-08-02

Disk visibility checkpoint: 2026-08-10. The research tree and receipts survive
locally. External repositories, competition state, and rules were not refreshed
by that checkpoint. A separate official scorer/rules source refresh was completed
on 2026-08-11 as recorded below; other dated external observations remain
historical.

This is the handoff ledger for any new research model or coding agent. Update it
only from immutable receipts. A passing probe is not promotion evidence.

## Objective and evidence boundary

Improve exact modular multiplication and inference efficiency for the SAIR
Modular Arithmetic Challenge without weakening the learned-computation claim.
The study ran in a separate local worktree. The only external research artifact
was the explicitly approved testing upload described below; it was not a SAIR
submission update.

Rob approved one separate public testing upload on 2026-08-01. The stopped
candidate is available at
`TrickyRex/neural-horner-direct-l2sp-a0875@b5cee6f1592b1eeb9822e4773e5e294caf013f60`.
The four candidate files at that immutable revision were downloaded and
SHA-256 reverified against the local package. This upload is not a SAIR update,
qualification result, or promotion.
No organizer compliance question was sent, and no ruling was requested or
issued.

The submitted incumbent remains unchanged:

- Hugging Face: `TrickyRex/bitserial-modmul-v8`
- submitted commit: `301938693892043d66a8bf6ec60a9c5ab85549d4`
- local checkpoint SHA-256:
  `294fbf809ad42bb0ac66dd51303d21ac5dddeca9a081b5790f1af39d36b21609`

The submitted revision still resolves publicly. The repository's current
`main` is `73a442f42489fced315b91873c4df0399e13634c`, which is not the submitted
revision; never substitute moving HF `main` for the pinned SAIR commit.

## Current portfolio

| Role | Artifact | State |
| --- | --- | --- |
| incumbent | submitted v8 | Keep unchanged; this is the only qualified and submitted artifact. Previously validated evidence is not automatically inherited by new programs. |
| stopped repair | `candidates/direct_two_pass_l2sp_a0875` | F11 vectorized exact-prefix induction is `validated_exact` on 193,116/193,116 transitions, but frozen-battery v3 subsequently failed at 509/512 attempted rows. Not submitted or qualified. |
| unrun conditional screen | `candidates/direct_two_pass_funcspace_a09375` | Earlier decisive-path evidence only. The current three-case/F11 fail-fast screen has not run. |
| causal control | `candidates/original_three_pass_l2sp_a0875` | Byte-identical repaired weights under the original program; used to separate schedule and weight effects, not a submission slot. |
| stopped | `candidates/direct_two_pass` | Architecture ablation only. Same v8 weights fail the exact-artifact decisive path in both orientations with 642 wrong bits at the first divergence. |

Only the submitted v8 incumbent is qualified. The L2-SP candidate's completed
F11 induction gate remains material research evidence, but the later validated
retention failure stops promotion. A strong local subproblem result does not
override the broader counterexample.

Rob confirms a SAIR playground run of the immutable testing revision at
`b5cee6f1592b1eeb9822e4773e5e294caf013f60`: tiers 1--10 each displayed
`100/100`, while the unscored Tier-0 pure-multiplication diagnostic displayed
`40/100`. Rob confirms submitted v8 also displays `40/100` on Tier 0 in the live
playground, so there is no live Tier-0 regression. This is
`ROB-CONFIRMED/UNARCHIVED`: no screenshot/log is preserved here, and exact
runtime/evaluator identity remain uncaptured.
Do not conflate this live comparison with older local open-source-scorer v8
receipts, which record `60/100` on their historical evaluation surface.

The stopped candidate's derived checkpoint SHA-256 is
`6d1c8a09e555778fb7961778678a1473e681707ada0d20f1465712313b3e8f01`.
It interpolates v8 toward the L2-SP endpoint at `alpha=0.875`; the endpoint is
`97db605caf949460b06bfb916d6a3ab0039547e6b79c89fbc48a1025a580e421`.
The function-space endpoint SHA-256 is
`2cfe7bb6b7e42a2f2e8863dd6e9122ca51c8c3435180db388e8b338a02c4497f`.
Their original job-temp locations were not durable provenance storage. Exact
hash-verified copies are preserved as
`research/source_checkpoints/l2sp_lambda_1e-3_step1500.pt` and
`research/source_checkpoints/function_space_endpoint.pt`; neither belongs in a
submission package.
The historical selection screen was 68/68 non-final rows exact, representing
only 56 unique transitions, with minimum signed bit-logit margin 1.4953. It
deliberately omitted 12 final-disjoint rows and remains one-step selection
evidence only; those omissions are not the current F11 qualification remainder.

## Verified facts

**Qualification-runner audit, 2026-08-01:** the first-generation inductive
receipts evaluate every candidate-cell transition and replay the manifest
program with an exact recurrence, but they synthesize phase order/counts rather
than comparing the complete observed call sequence. They also record, but do
not gate on, strictly positive margin. The current entry sources match the
expected routes and every passing receipt below has positive margin, so this is
not a reproduced model failure. Those receipts are nevertheless conditional
until a route-bound, positive-margin-gated runner reproduces them. Runner v3 was
frozen for the August study at
`87e0b15e8a234ea03c43f5621d11fc1b83da5a1480d20efbc985c29ad22268f1`.
It binds the helper/scorer source set, checks artifacts before/after load and
execution, validates the observed manifest route, gates positive margin, and
records actual tensor/logit dtypes. Its receipt-bound source-set SHA-256 is
`0ca78bb1fdfc797ff559aaca0f623351022f92fb0a6208d892050e8832dc6837`.
The exact historical runner and `held_out_battery.py` snapshots bound by that
receipt are no longer present in this checkout or its recovery snapshot; the
current files have different hashes. The union is therefore immutable historical
evidence, not independently replayable from the present tree. At execution,
`scripts/validate_f11_receipts.py` rejected unverified scorer discovery,
rechecked the then-current local/scorer/artifact bytes and imported
`modchallenge` origin, and enforced one runner/source/artifact/case identity plus
exact route, margin, and cardinality gates over the receipt union. Those recorded
checks must not be rewritten as a current replay. Any run with today's sources
is a new evidence boundary and requires a new receipt identity.
Also interpret `candidate_prefix_executed=true` by `execution_mode`: in
`vectorized_exact_prefix_induction` it means the checkpoint evaluated every
exact teacher-prefix transition and the rollout follows logically by induction;
it does not mean a literal free-running trajectory was executed. Only
`literal_sequential_replay` satisfies that separate gate. The v3 module-level
singleton wording predates the inductive mode and must not override the receipt.

- Branch: `codex/direct-horner-research-20260801`
- Base commit: `584ccd935ae1124d8a1bd994bd2202fd36ab0346`
- Repository `origin/main` on 2026-08-01:
  `4997f9afd752df715fb73759b6270b2fadb2cfe1`, two README-only prior-art edits
  ahead of this branch. Rebase or merge those documentation commits before
  integration; no research result here assumes they are already present.
- Receipt-bound scorer checkout on 2026-08-01: clean detached
  `82510bba00a1126649bd76dd1a451f14d0b3eb60`.
- Official scorer suite at that pin: 116 passed, 1 skipped.
- **Live source refresh, 2026-08-11:** official scorer/rules `origin/main` was
  `99cac6ef5c2f82e53105ec0ddcbb9b8d37bf6fca`, two commits after receipt pin
  `82510bb`. The diff adds an explicit ban on deterministic operand
  pre-reduction and updates examples; the audited Docker and evaluation-pipeline
  files are byte-identical to the receipt pin. NeuralHorner's operand reductions
  are learned-cell passes, not deterministic pre-reduction. This source
  comparison is not an organizer compliance ruling. All existing receipts
  remain evidence for pinned `82510bb`; do not relabel them as runs against
  `99cac6e`. Refresh receipt:
  `research/receipts/rules_refresh_20260811.json`, SHA-256
  `2ae8e5294adf410d2e857b61c2c63050286f9278a03740f839c3cba1af9364b4`.
- The pinned scorer runtime sources are captured in
  `research/receipts/scorer_runtime_contract_82510.json`. The published
  `modchallenge-sandbox` is a mutable tag, not an immutable digest; its
  Dockerfile installs CPU-only PyTorch 2.5.1 and its wrapper defaults to four
  CPUs, 8 GiB RAM, and 2 GiB tmpfs. No final GPU class, dtype, or immutable
  ranked-evaluation image is published.
- The source-level 300-second inference check is cooperative between batches,
  not hard preemption inside `predict_digits_batch`. The rules describe load
  and determinism as separately bounded, but the pinned pipeline measures and
  logs those phases without enforcing explicit limits. These are verified
  source facts and source mismatches, not claims about an unpublished external
  supervisor.
- The official overview's worked example is defective: it says 52, while
  `(123456789 * 987654321) mod 97 = 6`. This was independently asserted locally
  and is recorded in the runtime-contract receipt.
- The direct exact-integer schedule and its swapped form build in Lean. This
  proves the mathematical recurrence, not that the learned cell realizes it.
- Lean now also proves the exact-prefix certification bridge: agreement of an
  arbitrary cell with the exact transition along one fixed exact trajectory
  implies equality of the deterministic scan, and certified reduction plus
  direct-scan passes compose to the modular product. The scan theorem is
  axiom-free; the composed arithmetic theorem uses only Mathlib's standard
  `propext` and `Quot.sound`. No checkpoint is proved universally correct.
- Scored-tier public-benchmark recurrent bit-work proxy: 20.001% lower for the
  two-pass program. This is a source-bound operation count, not latency.
- The historical v3 receipt records the unchanged-weight direct artifact on
  `F11 * 1`:
  2,049/2,050 transitions are exact per
  orientation, the first divergence is global/phase step 2,048 after a verified
  exact prefix, 642 bits are wrong, and minimum signed correct-target margin is
  -20.8369.
- The historical v3 receipt also records the submitted v8 weights under the original three-pass
  program fail the
  same learned transition after an exact prefix: 4,097/4,098 transitions are
  exact per orientation, one transition fails with 642 wrong bits, and minimum
  signed margin is -20.8369. The first global step is 2,048 or 2,049 depending
  on operand orientation because the original program preserves input order.
- Its refreshed 16/16 batch-invariance and two p33 no-shortcut/randomization
  receipts bind the current artifact, hardened runners, and case sets. They are
  current small-width smoke evidence, not promotion gates; the current `F11 * 1`
  counterexample still stops the artifact.
- The historical v3 receipt records the primary direct L2-SP candidate's complete decisive
  path in both orientations on MPS fp32: 4,100/4,100 transitions, minimum signed
  margin 5.01655, exact observed routes, fp32 captured logits, and no
  artifact/source mutation. This is vectorized exact-prefix induction: every
  exact teacher-prefix transition was evaluated and exactness follows by
  induction, but no literal sequential, batched, or official inference was run.
- The historical v3 receipt also records all seven other companion widths in both
  orientations on MPS fp32: 14/14 results and 53,840/53,840 transitions,
  minimum signed margin 5.01657, exact observed routes, fp32 captured logits,
  and no artifact/source mutation.
- The historical v3 receipt records the three equal-length ties in both orientations on MPS
  fp32: 6/6 results and 24,588/24,588 transitions, minimum signed margin
  5.01657, exact observed routes, and no artifact/source mutation.
- The historical v3 receipt records all nine legacy cases in both orientations on MPS fp32:
  18/18 results and 110,588/110,588 transitions, minimum signed margin 7.19223,
  exact observed routes, fp32 captured logits, and no artifact/source mutation.
- The strict union receipt reports `status=validated_exact` for all four
  historical v3 components: 20 unique numerical cases, 40 unique oriented
  cases, and 193,116/193,116 transitions exact. The components contribute
  4,100 decisive, 53,840 companion, 24,588 tie, and 110,588 legacy transitions;
  all routes are exact, logits are fp32, margins are positive, and no artifact
  or source mutates. The union binds runner
  `87e0b15e8a234ea03c43f5621d11fc1b83da5a1480d20efbc985c29ad22268f1`,
  source set
  `0ca78bb1fdfc797ff559aaca0f623351022f92fb0a6208d892050e8832dc6837`,
  artifact set
  `9b0ba4f1c6ff5ed8ccf7b64b5b173baf10b8881c407ce430ec3338adcc0e06fc`,
  and full case set
  `54f9d2342fcdbfa6fc21a5d6a6560d13364059be3c48174a6c66a407834a86ee`.
  The component receipt hashes are embedded in the union receipt. At execution,
  it rechecked the bound local and scorer bytes, verified scorer commit
  `82510bb`, and checked the imported `modchallenge` origin. Receipt:
  `research/receipts/f11_primary_direct_routebound_v3_union.json`, SHA-256
  `5890134071d914c3bd4e77852c7a3379d56b41c30543871e0a7686708b799920`.
- The L2-SP original/direct controls use byte-identical weights. The historical
  v3 receipt records the original program on 8,196/8,196 decisive transitions, minimum
  margin 5.0166, with the same bound source/artifact checks. Their joint success is
  strong causal evidence that checkpoint repair, not direct routing, repairs
  this case. Both are also 68/68 on the same one-step screen and 10/10 exact as
  decoded values on the small-width smoke. Only the direct program is raw-digit
  invariant across mixed prime widths; the original changes leading-zero width.
- An earlier original-schedule runner clears all seven other companion widths:
  14/14 results and 82,512 transitions, minimum margin 4.9165. This remains
  diagnostic until v3 replay; its larger count and lower margin reflect the
  extra learned operand reduction.
- The function-space direct candidate is exact on its earlier-run decisive path
  with minimum signed margin 4.9199, but has no v3 receipt. It remains an
  independent secondary family, not a qualified submission artifact.
- The 768-case structured battery was disjoint from submitted v8's original
  training and its 759/768 count remains valid first-contact evidence for that
  checkpoint. Later DAgger/repair/routing decisions used it, so it is now frozen
  development data and cannot support a post-selection generalization claim.
- A source audit further narrows that evidence: the 768 rows contain 693 unique
  numerical `(a,b,p)` cases; the alternating family has 53 unique cases among
  128 rows. The legacy family named `product straddles k*p` is not a boundary
  family: its floor-quotient construction yields 0/128 frozen products within
  distance 3 of a modulus multiple, with nearest-distance bit lengths 2040--2047
  (median 2046). Preserve the frozen cases/key for continuity, call them
  quotient-correlated operands, and do not attach an independent-binomial or
  reduction-edge interpretation. A corrected boundary family must be a new,
  versioned set and remain sealed until the method is locked.
- Frozen-battery v1 is `INTERRUPTED`, not passed. Its preserved receipt still
  says `running`, but its process ended after only the original-orientation
  Fibonacci family completed 128/128 exact; the remaining 1,408 rows did not
  run.
- Frozen-battery v2 is also `INTERRUPTED`, not passed. It was stopped
  immediately when preflight review found that its old runner did not bind the
  external cache inventory into the receipt. It completed no family and is not
  model-failure evidence.
- Frozen-battery v3 is `FAILED`, not interrupted. It completed the first four
  original-orientation families, then stopped fail-closed at 509/512 after
  three errors in `fixed Hamming weight W/2` (source indexes 3, 35, and 110).
  The other 1,024 rows were not run. The one-shot validator reports no artifact,
  source, fixture, or cache-provenance mismatch: this is model evidence, not a
  runner failure. Permanent copies are under
  `research/receipts/frozen_battery_v3/`; receipt SHA-256 is
  `1b9d94f9c3155f6fefaae34c268ad90fc2630f1cbc0a6cb505e23eec05ff673c`
  and validation SHA-256 is
  `fb2525975b15ae256a593bcb594b2602f67b41aa2c3a6ddb3e440dbeb1cdcff9`.
- A receipt-bound four-way causal ablation replays those three cases under v8
  and repaired weights crossed with original and direct schedules. Validation
  status is `validated_causal_ablation`:
  - source 003: original-v8 PASS, direct-v8 PASS, original-repaired PASS,
    direct-repaired FAIL -- a weight/schedule interaction is required;
  - source 035: PASS, FAIL, PASS, FAIL -- the direct schedule is sufficient;
  - source 110: PASS, PASS, FAIL, FAIL -- the repaired weights are sufficient.
  Every failing arm has exactly one wrong exact-teacher transition. For source
  035, both direct arms fail the identical teacher transition
  (`scan_operand`, global step 4,245); repair amplifies it from one wrong bit at
  margin -2.1741 to seven wrong bits at margin -13.2857. Fixture SHA-256 is
  `4fa3322471e1d4185d53cbd301858d416ba44b8a6ee0d110e1d2a3b8d5c7113a`;
  union validation SHA-256 is
  `e47583c1e68d2985ffc0814b75473a8cc6c4350ecfb86591b5f9783a98a0d020`.
- Candidate promotion is `STOPPED`. The function-space screen and the
  counterexample-guided K-state miner design are both `UNRUN/CONDITIONAL`. No
  new training has run.

## Stopped branch and conditional future work

1. The cheapest remaining check is a zero-training screen of the existing
   function-space candidate on `035 -> 110 -> 003`, followed by the complete v3
   F11 set only if all three pass. This has not run. Any matched failure closes
   that branch without a broader battery.
2. The K-state counterexample miner is a no-training design proposal, not a
   result. Before any dry run, lock the retention set, sealed split, compute
   accounting, paired random-selection control, and STOP criteria.
3. Do not begin broad training from either idea. Only a predeclared, surviving
   zero-training gate could justify a later experiment.
4. The F11 four-way and fixed-Hamming four-way comparisons are complete.
   Together they show a real Pareto wall: repair fixes F11 but relocates errors,
   while the direct schedule introduces an independent vulnerable trajectory.
   Do not describe the L2-SP checkpoint as a global repair.
5. Any future promotion claim would still need competition-width invariance,
   randomized-weight collapse, a fresh sealed battery, and final-hardware timing.
   None was run for this stopped candidate.
6. The published CPU/GPU runtime ambiguity and schedule-specific compliance
   status remain unresolved. No organizer question was sent and no ruling was
   requested or issued.

## Evidence vocabulary

- **verified**: replayed now with hashes, counts, environment, and expected gate;
- **partial**: a real but insufficient probe, with its missing scope stated;
- **conditional**: depends on an unverified premise or incompatible receipt;
- **failed**: a reproduced counterexample or gate failure;
- **unverified**: no current receipt;
- **stopped**: known evidence is sufficient to prevent promotion.

Never replace these terms with “works,” “fixed,” or “ready” unless every
promotion gate has passed for the exact packaged artifact.

## Prior-art guard

The pinned XllentAI `modular_arithmetic` artifact at
`3d2c226c2382140b890026bdfdd59485daa192ba` also uses a learned,
modulus-conditioned bit-serial Horner recurrence and reports held-out-prime
transfer. Do not claim the shared Horner/double-and-add formulation as novel.
Its model card describes three width-specific MLP cells (about 50M, 114M, and
236M parameters) and public performance through 64-bit primes; its pinned
`model.py` performs Python-side `% p` operand reductions before the learned
scan. NeuralHorner's defensible distinctions are one shared approximately 471K
parameter BiGRU, learned operand reductions, and scaling to the challenge's
larger widths. Source comparison alone is not an organizer compliance
determination. See `LITERATURE_LEDGER.md` for the source-level comparison.

Do not mistake that historical pin for the current nearest baseline. On
2026-08-01, XllentAI Hugging Face `main` resolved to
`f704813dc64d7c04cc9ba7dbf9a3a281c431628d`. Its model card reports two shared
carry-aware TCN weight sets (about 10.7M parameters), public tiers 1-10 at 1.00,
true-trajectory state training, width/distribution matching, worst-bit margin,
distillation, and weight soup. Those claims have not been rerun here, and its
pinned inference source still applies Python-side `% p` to both operands. Any
new training plan must compare against those reported mechanisms rather than
presenting them as new. Refresh moving HF `main`, but cite immutable historical
and current pins separately.
