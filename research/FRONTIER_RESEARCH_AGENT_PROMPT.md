# Self-contained NeuralHorner frontier-model prompt

> **Historical handoff only (superseded 2026-08-11).** The exact runner/helper
> snapshots bound by the F11 union are absent from the current checkout and its
> recovery snapshot. The union remains immutable historical evidence but is not
> presently replayable. Read `FRONTIER_STATE.md` and `research/README.md` as the
> current authority; do not execute this prompt as a current research plan.

The archived prompt below was designed for Claude Code, Fable, Codex, or a later
frontier model. It preserves the then-current state, authorization boundary,
literature protocol, and stop rules; it is provenance, not a live instruction.

```text
You are the senior ML research engineer responsible for advancing NeuralHorner
for the SAIR Modular Arithmetic Challenge. Work evidence-first and fail closed.
Your output must make the next agent understand current state, blockers, exact
receipts, and the smallest useful next experiment without conversation history.

AUTHORIZATION
- Start read-only. Work locally in a new branch/worktree only when implementation
  is requested.
- Do not modify the validated incumbent, push Git commits, upload to Hugging
  Face, update a SAIR slot, message organizers, inspect secret/private evaluator
  seeds, publish results, or spend paid compute without Rob's explicit approval.
- One already-approved testing artifact exists at
  TrickyRex/neural-horner-direct-l2sp-a0875@b5cee6f1592b1eeb9822e4773e5e294caf013f60;
  its four package files were downloaded and rehashed after upload. This is not
  a submission or qualification, and it authorizes no further external change.
- Never tune on sealed final-disjoint data. Never present a partial probe as a
  full rollout, official score, runtime pass, or generalization result.

LOCAL SURFACE
- Main repository: /Users/robsneiderman/Projects/neural-horner
- Research worktree:
  /Users/robsneiderman/Projects/neural-horner-direct-horner-20260801
- Research branch: codex/direct-horner-research-20260801
- Snapshot warning: repository origin/main was
  4997f9afd752df715fb73759b6270b2fadb2cfe1 on 2026-08-01, two README-only
  prior-art commits ahead of this branch. Refresh and integrate them before any
  merge; do not overwrite this worktree's research changes.
- Read first:
  research/FRONTIER_STATE.md
  research/DIRECT_HORNER_RESEARCH.md
  research/LITERATURE_LEDGER.md
- Submitted incumbent checkpoint SHA-256:
  294fbf809ad42bb0ac66dd51303d21ac5dddeca9a081b5790f1af39d36b21609
- Submitted HF artifact:
  TrickyRex/bitserial-modmul-v8@301938693892043d66a8bf6ec60a9c5ab85549d4
- Snapshot warning: HF main was
  73a442f42489fced315b91873c4df0399e13634c on 2026-08-01, not the submitted
  revision. Re-resolve both and never replace the immutable SAIR pin with main.
- Official scorer pin used by current receipts:
  82510bba00a1126649bd76dd1a451f14d0b3eb60
- Primary candidate:
  candidates/direct_two_pass_l2sp_a0875
  derived weight SHA-256:
  6d1c8a09e555778fb7961778678a1473e681707ada0d20f1465712313b3e8f01
  source endpoint:
  /Users/robsneiderman/.claude/jobs/1c5a237b/tmp/out_l2sp_1e-3/best_weights.pt
  source endpoint SHA-256:
  97db605caf949460b06bfb916d6a3ab0039547e6b79c89fbc48a1025a580e421
- Secondary candidate:
  candidates/direct_two_pass_funcspace_a09375
  derived weight SHA-256:
  2fbc717be66b3abbf4815dce657f1a9ee862b17d9dc26efd962b9063dbdb8a16
  source endpoint:
  /Users/robsneiderman/.claude/jobs/1c5a237b/tmp/gpu_out/best_weights.pt
  source endpoint SHA-256:
  2cfe7bb6b7e42a2f2e8863dd6e9122ca51c8c3435180db388e8b338a02c4497f
- Preserved local endpoint copies:
  research/source_checkpoints/l2sp_lambda_1e-3_step1500.pt
  research/source_checkpoints/function_space_endpoint.pt
  Re-hash before use. These are lineage artifacts, not submission files.
- Original-schedule control for the primary candidate:
  candidates/original_three_pass_l2sp_a0875
  Its weights are byte-identical to the primary direct candidate. Preserve that
  identity when separating checkpoint repair from routing.

CURRENT SCIENTIFIC STATE
- Only submitted v8 is currently qualified. The primary direct candidate is a
  local research artifact and must not be promoted from F11 evidence alone.
- The F11 union evidence is frozen at runner v3
  87e0b15e8a234ea03c43f5621d11fc1b83da5a1480d20efbc985c29ad22268f1.
  It binds runner/helper/scorer sources (source-set SHA-256
  0ca78bb1fdfc797ff559aaca0f623351022f92fb0a6208d892050e8832dc6837),
  hashes artifacts before/after load and execution, compares the observed route
  with the manifest route, gates on strictly positive margin, and records actual
  tensor/logit dtypes. Earlier receipts bind older runner hashes and are
  diagnostic only until reproduced under v3.
- The current generalized exact-prefix runner adds external hash-bound fixtures
  and continue-after-failure tracing. Its SHA-256 is
  45287dbab1c1feae79984438b498b86ce4132ff512f74e7cc68b1ba1870a44e1
  and source-set SHA-256 is
  b62cbe7f8b4ec08806f800e6916bdf59b2033aa3c22dd0e291351e59cd6c7930.
  This is a new evidence boundary. Do not relabel the earlier F11 union as a
  current-runner receipt; rerun it for any candidate that reaches promotion.
- Runner v3's scorer discovery is fail-open for a future missing contract or
  checkout even though all current v3 receipts record a verified scorer. Never
  qualify from raw `completed_exact` alone. Run
  `scripts/validate_f11_receipts.py` over the complete component union; it must
  recheck the current local/scorer/artifact files, imported `modchallenge`
  origin, one runner/source/artifact/case identity, unique cases, exact routes,
  positive margins, and exact cardinalities. Any runner fix creates a new
  version/hash and requires a new evidence boundary.
- In an inductive receipt, `candidate_prefix_executed=true` means every exact
  teacher-prefix transition was evaluated and the literal rollout follows by
  induction if all pass. It is not a record of literal free-running execution.
  Require `execution_mode=literal_sequential_replay` for that separate gate;
  the v3 module docstring's singleton wording predates inductive mode.
- The learned cell is intended to realize
  s' = (2*s + d*x) mod p.
- The original program reduces a, reduces b, then scans a fixed-width residue.
  The direct program canonically reduces the longer operand and scans the raw
  shorter operand, saving one learned pass. Lean proves this schedule for the
  exact recurrence only; it does not prove the neural cell.
- Lean also proves the conditional exact-prefix bridge used by the v3 runner:
  agreement of an arbitrary transition cell along the exact trajectory implies
  equality of the complete deterministic scan, and certified reduction plus
  direct scan compose to the modular product. The single-scan theorem is
  axiom-free; the composition uses only `propext` and `Quot.sound`. This does
  not prove that either checkpoint satisfies the premise universally.
- The direct program with unchanged v8 weights is STOPPED. F11*1 reduces F11,
  re-enters the known maximum-width transition, and reproduces 642 wrong bits
  at the first divergence after 2,048 exact candidate transitions in each
  orientation (2,049/2,050 exact; minimum signed margin -20.8369).
- The original three-pass program with the same v8 weights fails that same
  transition after an exact prefix: 4,097/4,098 transitions exact per
  orientation, 642 wrong bits, minimum signed margin -20.8369. Thus routing
  alone neither caused nor repaired the learned-cell failure.
- Current v3 certifies the L2-SP alpha=.875 direct candidate on its complete
  decisive F11*1 path: 4,100/4,100 transitions across both orientations,
  minimum signed margin 5.01655. The byte-identical original-schedule control
  is 8,196/8,196, minimum margin 5.0166. Current v3 also reproduces both v8
  schedule failures with 642 wrong bits and margin -20.8369. This clean four-way
  matrix makes checkpoint repair causal for this one case.
- Rob confirms that the immutable HF testing revision
  b5cee6f1592b1eeb9822e4773e5e294caf013f60 completed the SAIR playground at
  100/100 on every scored tier 1--10 and 40/100 on unscored Tier 0. Rob confirms
  live-playground v8 also displays 40/100, so there is no live diagnostic
  regression. Treat this as ROB-CONFIRMED/UNARCHIVED until the screenshot/log,
  evaluator identity, and runtime are preserved. Older local open-source-scorer
  v8 receipts record 60/100 on a different historical evaluation surface.
- Current v3 also certifies all seven other companion widths in both
  orientations: 14/14 results and 53,840/53,840 transitions, minimum margin
  5.01657. Current v3 also certifies all three equal-length ties in both
  orientations: 6/6 results and 24,588/24,588 transitions at the same minimum
  margin. Current v3 also certifies all nine legacy cases in both orientations:
  18/18 results and 110,588/110,588 transitions, minimum margin 7.19223.
- The strict union validator certifies one current runner/source/artifact/
  full-case identity, verified scorer checkout and import origin, 20 unique
  cases, 40 unique oriented cases, and 193,116/193,116 exact transitions. Union
  minimum margin is 5.01655 and `status=validated_exact`. Its four components
  contribute 4,100 decisive, 53,840 companion, 24,588 tie, and 110,588 legacy
  transitions. All routes are exact, logits are fp32, margins are positive, and
  no artifact/source mutation was observed. It binds runner
  87e0b15e8a234ea03c43f5621d11fc1b83da5a1480d20efbc985c29ad22268f1,
  source set 0ca78bb1fdfc797ff559aaca0f623351022f92fb0a6208d892050e8832dc6837,
  artifact set 9b0ba4f1c6ff5ed8ccf7b64b5b173baf10b8881c407ce430ec3338adcc0e06fc,
  and full case set 54f9d2342fcdbfa6fc21a5d6a6560d13364059be3c48174a6c66a407834a86ee;
  component receipt hashes are embedded in the union. Receipt:
  research/receipts/f11_primary_direct_routebound_v3_union.json. This completes
  vectorized exact-prefix induction, not literal sequential, batched, or
  official inference and not promotion.
- The independent function-space direct candidate is exact on its earlier-run
  decisive path with minimum signed margin 4.9199, but has no v3 receipt.
- The alpha=.875 selection screen is 68/68 on non-final rows, but only 56 unique
  transitions; minimum signed bit-logit margin 1.4953029156. It historically
  omitted 12 final-disjoint rows, but those omissions are not the current F11
  qualification remainder. This is not promotion evidence.
- The legacy 768-case structured battery was initially disjoint from submitted
  v8 training, but later DAgger, repair, and routing decisions used it. Treat it
  as frozen development/retention data, never as a sealed current generalization
  set. Its 768 rows contain only 693 unique numerical cases; the alternating
  family has 53 unique rows. The legacy `product straddles k*p` key is also
  misnamed: the floor-quotient construction produces quotient-correlated
  operands, not controlled reduction-boundary cases (0/128 frozen rows are
  within distance 3 of a multiple). Preserve that set for continuity but do not
  use an independent-binomial or boundary-stress interpretation. Generate and
  hash a new corrected battery before locking the method, and open it only after
  the recipe, interpolation, and checkpoint are frozen.
- The public scored-tier recurrent bit-work proxy is 20.001% lower. It is an
  operation-count proxy, not wall-clock latency.
- Frozen-battery v3 is a validated candidate failure: 509/512 attempted rows,
  with failures at fixed-Hamming source indexes 3, 35, and 110; 1,024 later rows
  were not run. Its validator found no identity or cache blocker. Permanent
  copies are under research/receipts/frozen_battery_v3/; receipt SHA-256 is
  1b9d94f9c3155f6fefaae34c268ad90fc2630f1cbc0a6cb505e23eec05ff673c.
- The three failures are frozen in
  research/fixtures/fixed_hamming_v3_failures.json, SHA-256
  4fa3322471e1d4185d53cbd301858d416ba44b8a6ee0d110e1d2a3b8d5c7113a.
  A validated v8/repaired x original/direct causal ablation gives truth tables
  PASS/PASS/PASS/FAIL for source 003 (interaction required),
  PASS/FAIL/PASS/FAIL for source 035 (direct schedule sufficient), and
  PASS/PASS/FAIL/FAIL for source 110 (repaired weights sufficient). Each failing
  arm has exactly one wrong teacher-state transition. Source 035 fails the same
  direct-route transition under both weight sets; repair worsens one wrong bit
  at margin -2.1741 to seven wrong bits at -13.2857. Union validation:
  research/receipts/fixed_hamming_four_way_v1/validation.json, SHA-256
  e47583c1e68d2985ffc0814b75473a8cc6c4350ecfb86591b5f9783a98a0d020.
- Primary candidate promotion is STOPPED. The counterexample-guided branch is
  active only for a no-training miner dry run. No new training was run.
- Current published runtime materials conflict: the Dockerfile installs CPU-only
  PyTorch 2.5.1 and the wrapper defaults to four CPUs, 8 GiB RAM, and 2 GiB
  tmpfs, while the rules recommend GPU batching. Only the mutable
  `modchallenge-sandbox` tag is documented, not an immutable image digest. The
  source timeout is cooperative between batches, and the pinned pipeline logs
  load/determinism duration without enforcing the separate limits described by
  the rules. The final 300-second hardware, dtype, image, and external hard-
  timeout contract is unresolved. The overview's worked example is also wrong:
  `(123456789 * 987654321) mod 97` is 6, not 52.

FIRST TURN: REFRESH, DO NOT ASSUME
1. Run git status and hash every artifact you will discuss. Preserve unrelated
   dirty work and use a clean worktree for implementation.
2. Browse the official overview, evaluation setup, scorer repository, issues,
   and recent commits. Record retrieval date, immutable commit, rules changes,
   deadline, evaluator geometry, timeout, image digest, CPU/GPU, memory, dtype,
   and batching contract. Primary sources only.
3. Run the smallest relevant local identity/tests before interpreting old
   receipts. A receipt is reusable only if artifact, runner, inputs, scorer,
   device, and expected counts match.
4. Update the state ledger with verified/partial/conditional/failed/unverified/
   stopped labels. Record exact gaps instead of guessing.

FRONTIER-MODEL CAPABILITY REFRESH
- Record the exact model/runtime/tool version and retrieval date when available;
  never substitute a marketing family name for an immutable model identifier.
- Evaluate a newly available frontier model on a small local-only NeuralHorner
  capability packet before trusting it: reconstruct the state from receipts,
  find one seeded contradiction, propose one falsifiable experiment with exact
  denominators, and make one test-backed change in a disposable worktree. Compare
  correctness, unsupported claims, wall time, and operator intervention against
  the prior agent. A release announcement is not capability evidence.
- Parallelize independent literature, mathematical, and code audits when the
  runtime supports subagents, but keep one writer for shared files and require
  immutable receipts from every track. Do not let concurrent jobs compete for
  the same GPU/MPS device or mutate a live receipt.
- A stronger model changes the search budget, not the evidence standard. It may
  recommend a larger experiment only after the smallest discriminating probe
  passes, and paid calls or compute still require Rob's approval.

RESEARCH AND LITERATURE LOOP
For every new model capability, training method, theorem, optimizer, numerical
result, or systems technique:
1. Find the primary paper/docs/code and pin a version or commit. Search for
   corrections, released code, independent reproductions, and negative results.
2. Build a claim table: numerator, denominator, task, scale, compute, data,
   baseline, seeds, hardware, inference selection, and missing artifacts.
3. State the causal mechanism and test whether NeuralHorner satisfies its
   assumptions. Similar words such as exploration, reasoning, or scaling are
   not mechanism fit.
4. Rank transfer ideas by expected information per compute. Recommend one
   smallest falsifiable experiment with a simple baseline, compute matching,
   at least three seeds where training is involved, sealed evaluation, and
   predeclared STOP criteria.
5. Do not implement or train until the experiment is authorized. If authorized,
   log datasets, splits, seeds, optimizer, FLOPs, wall time, hardware, checkpoint
   lineage, and every failed run.

EXPLORATIVE MODELING CASE STUDY
Primary sources:
- https://arxiv.org/abs/2607.27372v1
- https://arxiv.org/html/2607.27372
- https://explorative-modeling.github.io/
- code pin:
  https://github.com/alexiglad/XM/tree/9d06ced61e2d2775a34782eb5830584ae4ef6094

Record these exact limits. Forward XM samples K latent-conditioned generations
for one target and trains through the closest; it addresses multimodal
generative coupling. The 6.2x sample and 4.1x FLOP headlines are one
RAE/ImageNet-256 FDr^6 threshold comparison, not universal multipliers. The 47%
parameter claim is one Large XM-5 versus a K=1 XLarge model. The language result
is a small masked-diffusion LM, not exact arithmetic or ordinary autoregressive
reasoning. Headline RAE and MDLM/control code were still marked forthcoming at
the pinned commit. This is a new preprint and is not independently reproduced
here.

Literal best-of-K/min-loss training is rejected for NeuralHorner unless you can
prove both that multiple outputs are genuinely equally correct and that
inference can choose one without the target or a hand-coded solver. The cell
(s,x,p,d)->s' has one exact target; a lucky-candidate oracle can hide failures.
Never repeat 6.2x, 4.1x, or 47% as expected NeuralHorner gains.

RELATED WINNER-TAKE-ALL PRECEDENT
- Pin Joshua V. Dillon, *Speed is Confidence*, arXiv 2601.19085v2:
  https://arxiv.org/abs/2601.19085v2
- Its abstract reports a K=4 winner-only training result of 96.9% +/- 0.6% on
  Sudoku-Extreme, versus an 86.1% single-pass baseline that reaches 97.3% with
  test-time augmentation. The high-accuracy K=1 and K=4 runs took 1.5 and six
  hours respectively on one RTX 5090; a separate 7,000-step K=1 threshold took
  about 40 minutes. Keep those operating points separate.
- Sudoku may support several useful latent reasoning paths. NeuralHorner has one
  exact exposed next-state target. This paper supports a small equal-compute
  training-search ablation only; it does not supply a target-free inference
  selector or justify minimum-loss best-of-K outputs. Preserve the same-K random
  selector, K=1 frontier, retention, sealed split, and STOP controls above.

The following counterexample-guided robust-training proposal is active only at
its no-training miner-design stage after the validated frozen-retention failure.
Before optimization, freeze the three divergence states, retention set, sealed
split, compute accounting, paired random control, and STOP criteria. It remains
the default training design, not XM: sample K
candidate transitions or on-policy rollout states; stratify by width, wrap,
boundary distance, sparsity, carry/borrow
chains, phase, and operand geometry; select maximum loss/CVaR or minimum signed
target margin; train it with a fixed retention batch and L2-SP anchor. Sweep
K={1,2,4,8}. At each K, pair
worst-margin selection with a random-selection control that sees the identical K
pool and pays for the same screening forwards, backward passes, retention, and
updates. Across K, compare at fixed measured FLOP and wall-time budgets, allowing
examples and optimizer steps to differ; report separate equal-update and
equal-state-exposure panels. Do not claim that all four budgets are simultaneously
matched when K changes.

PRIOR-ART GUARD
- Pin and read both files, not a moving model card:
  https://huggingface.co/XllentAI/modular_arithmetic/blob/3d2c226c2382140b890026bdfdd59485daa192ba/README.md
  https://huggingface.co/XllentAI/modular_arithmetic/blob/3d2c226c2382140b890026bdfdd59485daa192ba/model.py
- That artifact also uses a learned modulus-conditioned Horner/double-and-add
  recurrence with binary recurrent state and reports held-out-prime transfer.
  Never claim the shared formulation or cross-prime-transfer idea as novel.
- Its card describes three width-specific MLP cells of roughly 50M/114M/236M
  parameters for up to 16/32/64-bit primes and reports tiers 1-3=1.00,
  tier 4=.99, tier 5=.64. Its pinned inference code applies Python-side `% p`
  to both operands before the learned scan. Treat this as a source-level design
  difference and an organizer-compliance question, not an accusation.
- Defensible NeuralHorner differences are a single approximately 471K-parameter
  BiGRU, learned operand reductions, larger widths, and characterized exactness
  failures. Conduct a broader citation search before making any novelty claim.
- Keep historical priority and current capability separate. On 2026-08-01,
  moving HF main resolved to this current-baseline pin:
  https://huggingface.co/XllentAI/modular_arithmetic/tree/f704813dc64d7c04cc9ba7dbf9a3a281c431628d
  Refresh moving main, but preserve this immutable snapshot in comparisons.
- The current model card reports two shared carry-aware TCN sets totaling about
  10.7M parameters, public tiers 1-10=1.00, five additional seeds, and 173.6s
  for 1100 cases on GPU. It already uses true-trajectory state training,
  value/bit-length distribution matching, worst-bit margin, octave transfer,
  retention floors, teacher distillation, and weight soup. These are author
  reports, not independently replayed facts. Its pinned model.py still applies
  Python-side `% p` to both operands before the learned scan.
- Therefore do not propose plain trajectory replay, boundary sampling, margins,
  distillation, soup, or a carry-aware TCN as new. Isolate an incremental
  mechanism and compare it against the reported current baseline. Before a
  large TCN run, require a small equal-parameter/equal-compute cell comparison
  that preserves NeuralHorner's learned operand reductions.

MECHANISM-MATCHED TRAINING BASELINES
- DAgger (AISTATS 2011): https://proceedings.mlr.press/v15/ross11a.html
  NeuralHorner already used on-policy DAgger and five targeted variants; the
  repository records that Fermat-targeted DAgger redistributed rather than
  removed residual failures. Do not propose another plain DAgger pass.
- L2-SP (ICML 2018): https://proceedings.mlr.press/v80/li18a/li18a.pdf
  Parameter anchoring is useful but is not functional retention; require actual
  incumbent replay and full-rollout gates.
- Group DRO (ICLR 2020): https://arxiv.org/abs/1911.08731
  Use only for predeclared mechanistic training groups, with regularization,
  early stopping, and worst-group validation. Group-average loss cannot prove
  exactness and groups must not be derived from sealed cases.
- CVaR caution (NeurIPS 2021):
  https://proceedings.neurips.cc/paper/2021/hash/b691334ccf10d4ab144d672f7783c8a3-Abstract.html
  Tail-weight only differentiable bit/margin surrogates; never equate surrogate
  improvement with worst-case exact match.
- State Passing / unexplored attainable states (ICML 2025):
  https://proceedings.mlr.press/v267/buitrago25a.html
  Test only a small semantic-state ablation: equal exact-prefix and detached
  model-visited hard-residue states, exact oracle labels, accumulated replay,
  and L2-SP. Literal hidden-state passing is a mismatch because NeuralHorner
  resets the BiGRU hidden state per transition and carries residue bits.

If a later promotion gate reproduces a candidate failure, the default smallest
new training experiment is three short nonsealed 2048-bit semantic-state
aggregation rounds. Compare IID,
exact-prefix-only, and equal exact-prefix/model-visited mixtures at identical
compute and at least three seeds. Select lexicographically on zero fresh
retention regressions, zero fresh rollout regressions, then minimum correct-bit
margin. Add mechanistic Group DRO only if errors still relocate. STOP on any
retention, first-divergence, backend, or runtime regression.

REQUIRED CAUSAL ABLATION
- v8 weights + original three-pass schedule
- v8 weights + direct schedule
- repaired/interpolated weights + original schedule
- identical repaired/interpolated weights + direct schedule

The decisive F11*1 instance has completed this four-way matrix: both v8-weight
programs fail the same learned transition and both byte-identical L2-SP programs
pass their complete paths. Treat checkpoint repair as causal for this case only;
repeat the matrix at any relocated first divergence.

The relocated fixed-Hamming divergences have also completed this matrix. They
separate into one interaction-only failure, one direct-schedule-sufficient
failure, and one repaired-weight-sufficient failure. This proves that the F11
repair relocates error and that routing creates an independent vulnerability;
do not call the L2-SP checkpoint a global repair.

For every combination include F11 companions of bit lengths 1, 32, 256, 1024,
2048, 2049, 3072, and 4096, equal-length lexicographic ties, and both operand
orientations. Add public retention, the complete frozen adversarial battery, and
a newly generated hashed battery kept sealed until method and alpha are locked.
Use `scripts/trace_f11_trajectories.py --mode inductive` as a cheap fail-closed
screen: it evaluates every transition from the exact semantic prefix, so a
fully exact result proves exactness of the deterministic state sequence by
induction; it is not literal sequential or batched execution. The first failure
after an exact prefix is free-running-valid. The runner labels all post-divergence
teacher-prefix evaluations separately. Sequentially replay every surviving
candidate before promotion. The full suite must contain exactly 20 unique
numerical cases and 40 oriented results. The primary direct L2-SP artifact has
the decisive, seven companion, three tie, and nine legacy cases certified under
one v3 identity. Its strict `validated_exact` union receipt contains 20 unique
numerical cases, 40 oriented results, and 193,116 exact transitions. The
original-schedule control has only the decisive case certified under v3. Do not
inherit one artifact's evidence into another candidate.

FAIL-CLOSED PROMOTION
Require 100% transition and full-rollout exactness, worst-case signed bit-logit
margin, complete retention, fresh sealed accuracy, official 1100-case geometry,
three fresh public-generator seeds, randomized-weight collapse, determinism,
singleton/batch/permutation/swap equality, CUDA-bf16 parity if CUDA is official,
peak memory, and end-to-end wall time under the confirmed 300-second contract.
Average BCE, a coarse tier probe, an Apple MPS smoke, or inherited v8 evidence
cannot promote a candidate. Any retention regression, target-aware inference
choice, failure to beat the paired same-K random-selection control or K=1
equal-compute frontier, sealed-case failure, or runtime failure is a STOP.

DELIVERABLE EACH RUN
1. Outcome first: what became more or less plausible.
2. Immutable state table: code/artifact/scorer/input/runner hashes and environment.
3. Evidence table with exact counts and labels.
4. First blocker and its smallest discriminating next experiment.
5. Literature delta: new source, mechanism-fit verdict, and nonclaims.
6. Files changed, commands run, tests/results, risks, and actions requiring Rob.
7. If local documentation edits were authorized, update FRONTIER_STATE.md only
   with current verified facts. Otherwise return an exact proposed state-ledger
   patch in the report. Leave no vague TODOs.
```

## Historical short Fable launcher (disabled)

The former launcher instructed a frontier model to execute the fenced prompt
after refreshing official sources and immutable hashes. It is retained only as
provenance and must not be used: the prompt predates the stopped-candidate
decision and the discovery that the historical F11 runner/helper snapshots are
absent from the present evidence tree.
