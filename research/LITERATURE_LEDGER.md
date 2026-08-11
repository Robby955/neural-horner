# NeuralHorner literature-transfer ledger

This ledger prevents new papers from becoming untested architecture changes.
For each result, record the primary source, mechanism, exact evidence, mechanism
fit, smallest compute-matched experiment, rejection criteria, and nonclaims.

## 2026-08-01 — Explorative Modeling

Primary sources:

- Alexi Gladstone, Heng Ji, and Yilun Du, *Explorative Modeling*, arXiv
  2607.27372v1, submitted 2026-07-29:
  https://arxiv.org/abs/2607.27372v1
- Project page: https://explorative-modeling.github.io/
- Code inspected at commit
  `9d06ced61e2d2775a34782eb5830584ae4ef6094`:
  https://github.com/alexiglad/XM/tree/9d06ced61e2d2775a34782eb5830584ae4ef6094

### Source-grounded claim table

| Claim | Exact scope | Transfer status |
| --- | --- | --- |
| Forward XM samples `K` latent-conditioned generations for one target and backpropagates through the closest candidate. | Multimodal generative coupling; each extra forward is cheaper than another full forward/backward step under the paper's cost model. | Mechanism mismatch for a deterministic arithmetic cell with one exact target. |
| 6.2x fewer processed samples and 4.1x fewer FLOPs. | One RAE/ImageNet-256 `FDr^6` threshold comparison against the selected baseline, not a cross-domain multiplier. | Do not quote as an expected NeuralHorner gain. |
| 47% parameter efficiency. | One Large XM-5 comparison against a K=1 XLarge model with 47% more parameters. | Not a general parameter-scaling law. |
| Lower-compute SiT result. | 2.5x fewer samples and at most 52% fewer FLOPs in that experiment. | Still generative-image evidence, not exact arithmetic evidence. |
| Language result. | Small masked-diffusion language model: six blocks, width 384, context 256, evaluated on a perplexity-entropy frontier. Autoregressive gains are described as modest and preliminary. | No evidence for exact modular arithmetic or ordinary autoregressive reasoning. |
| Public implementation. | The repository currently marks the headline RAE and masked-diffusion/control code as forthcoming. | Three-day-old preprint; not independently reproduced here. |

### Mechanism-fit verdict

Literal Forward XM is rejected for NeuralHorner. Each transition
`(s, x, p, d) -> s'` has one correct bit vector. A closest-of-`K` training oracle
can reward one lucky output, but the true target is unavailable at inference to
select that output. It would weaken exactness precisely where worst-case
failures matter.

The useful idea is only to scale *search for difficult training states*. Name
the transfer accurately: counterexample-guided robust training or hard-transition
mining, not Explorative Modeling.

### Smallest falsifiable experiment

1. Sample `K={1,2,4,8}` candidate exact transitions or on-policy rollout
   prefixes, stratified by width, wrap/non-wrap, signed distance to `p`, state
   and multiplier Hamming weight, carry/borrow chains, phase, and operand shape.
2. Select maximum loss, CVaR loss, or minimum signed correct-bit margin—the
   opposite direction from XM's minimum-loss candidate.
3. Train that difficult case with a fixed retention batch and L2-SP anchor to
   v8. Compare with uniform replay and the existing L2-SP recipe.
4. At each K, pair hard selection with random selection from the identical K
   pool, using the same screening forwards, backward passes, retention, and
   optimizer updates. Across K, use fixed measured-FLOP and wall-time budgets;
   report equal-update and equal-state-exposure panels separately because all
   budgets cannot be held constant simultaneously. Use at least three seeds.
   Lock method and interpolation alpha before opening the 12 final-disjoint
   transitions or a new hashed sealed battery.
5. Promote only on exact full rollouts, worst-case margin, complete retention,
   fresh sealed cases, randomized-weight collapse, deterministic batching,
   backend parity, and official runtime.

STOP if hard selection does not beat its paired same-K random-selection control
or the `K=1` equal-compute frontier, causes any retention regression, needs
target-aware inference selection, or fails a sealed case.

### Explicit nonclaims

- Explorative Modeling has not been reproduced here.
- Its image or masked-language efficiency ratios do not predict NeuralHorner
  sample, FLOP, parameter, accuracy, or wall-time gains.
- The proposed experiment is inspired by the exploration-budget question but
  changes the selection objective and is not an implementation of XM.

## 2026-08-01 — XllentAI Horner-RNN prior art

Primary sources pinned to
`3d2c226c2382140b890026bdfdd59485daa192ba` (last modified 2026-06-13):

- Model card:
  https://huggingface.co/XllentAI/modular_arithmetic/blob/3d2c226c2382140b890026bdfdd59485daa192ba/README.md
- Inference source:
  https://huggingface.co/XllentAI/modular_arithmetic/blob/3d2c226c2382140b890026bdfdd59485daa192ba/model.py

### Source-grounded comparison

| Topic | Pinned XllentAI artifact | NeuralHorner consequence |
| --- | --- | --- |
| Core formulation | A binary recurrent state applies a learned, modulus-conditioned Horner/double-and-add transition and reports transfer to held-out primes. | The shared Horner formulation and cross-prime-transfer idea are prior art; do not claim them as novel. |
| Architecture and scale | Three width-routed MLP cells, reported at about 50M, 114M, and 236M parameters for at most 16-, 32-, and 64-bit primes. The card reports tiers 1-3 at 1.00, tier 4 at 0.99, and tier 5 at 0.64. | NeuralHorner's defensible distinctions are one shared approximately 471K-parameter BiGRU, learned reductions, and operation at much larger challenge widths. These are differences, not proof of scientific novelty. |
| Forward path | `model.py` line 206 computes `int(a_enc) % p` and `int(b_enc) % p` before the learned scan, although the model card says the scan itself computes no arithmetic. | Compare executable source, not only prose. NeuralHorner's learned reductions remain a meaningful implementation distinction. Whether either design satisfies the current rules is for the organizers to decide. |
| Training distribution | Half of each batch is mined near the comparison boundary; selection uses held-out-prime full-chain accuracy. | Boundary mining is already represented in the nearest prior system. A new hard-transition method must compare against this baseline and isolate what its broader state search adds. |

### Current nearest-baseline snapshot

Keep the historical pin above for priority. Separately, current Hugging Face
`main` resolved on 2026-08-01 to
`f704813dc64d7c04cc9ba7dbf9a3a281c431628d` (last modified 2026-06-22):

- Tree:
  https://huggingface.co/XllentAI/modular_arithmetic/tree/f704813dc64d7c04cc9ba7dbf9a3a281c431628d
- Model card:
  https://huggingface.co/XllentAI/modular_arithmetic/blob/f704813dc64d7c04cc9ba7dbf9a3a281c431628d/README.md
- Evaluation note:
  https://huggingface.co/XllentAI/modular_arithmetic/blob/f704813dc64d7c04cc9ba7dbf9a3a281c431628d/EVALUATION.md
- Inference source:
  https://huggingface.co/XllentAI/modular_arithmetic/blob/f704813dc64d7c04cc9ba7dbf9a3a281c431628d/model.py

This revision is materially stronger than the historical artifact. Its model
card reports two shared carry-aware TCN weight sets totaling about 10.7M
parameters, public tiers 1-10 at 1.00, five additional evaluation seeds at
0.997-1.000 overall, and 173.6 seconds for 1100 cases on an unspecified GPU.
These are source-reported results, not reruns here. The card also documents:

- true-Horner-trajectory state training instead of only uniformly sampled states;
- a value-uniform plus bit-length-uniform prime curriculum to cover changing
  reduction-boundary positions;
- worst-bit margin objectives and gradient accumulation at large widths;
- octave transfer from 1024 to 2048 bits, retention floors, teacher-logit
  distillation, and greedy weight soups;
- a failed all-width single-cell unification and a failed 1024/2048 soup route,
  both useful negative evidence.

The pinned current `model.py` still applies Python-side `% p` to both operands
at line 328 before the learned scan. Its published Docker/hardware question is
the same one that blocks NeuralHorner deployment claims. Treat its score,
runtime, robustness, and compliance statements as model-author reports until
the exact artifact and environment are independently replayed.

### Research consequence of the current baseline

Plain trajectory replay, boundary-focused sampling, worst-bit margins,
distillation, and weight soup are no longer new proposals. The next experiment
must isolate an incremental contribution. The default is a
retention-constrained robust refresh of the existing 471K-parameter GRU:

1. aggregate exact labels on states visited by the current policy, including
   the frozen F11 families and a locked public-retention set;
2. stratify by width, schedule phase, digit, quotient/subtract count,
   carry/borrow length, state sparsity, and signed boundary distance;
3. keep actual replay and L2-SP anchoring mandatory; add CVaR/worst-group
   selection only if errors remain concentrated after distribution matching;
4. compare against uniform replay and ordinary trajectory replay at identical
   examples, optimizer steps, FLOPs, wall time, and at least three seeds;
5. require exact full-rollout retention and a sealed battery before opening any
   final-disjoint transition.

A carry-aware TCN is a credible architecture-diversity study, but first compare
one small width-matched cell against the GRU at equal parameter/compute budgets.
Do not spend on a 2048-bit retrain until that probe demonstrates better
worst-bit margin and chain exactness while preserving learned operand reduction.

### Research and attribution consequence

`origin/main` now records that this pinned repository predates NeuralHorner's
GitHub release. Preserve that attribution when integrating this branch. A paper
or model card should frame NeuralHorner around parameter sharing/efficiency,
learned reduction, scale, exactness diagnostics, and causal schedule/weight
ablations—not invention of learned modular Horner recurrence.

### Explicit nonclaims

- No claim is made here about private XllentAI training runs, unreleased weights,
  organizer compliance, or independent reproduction.
- Reported accuracies and parameter counts are model-card claims tied to the
  pinned artifact, not results rerun in this worktree.
- Source-level differences do not establish priority for every component or a
  publishable novelty claim; a broader citation search remains required.
- The current revision's headline numbers, robustness estimates, and timing are
  not independently verified in this worktree.

## 2026-08-01 — Sequential and robust-training transfer

Only State Passing below is a 2025 result. DAgger, L2-SP, Group DRO, and CVaR
are established methods included for mechanism fit, not novelty.

| Method | Primary-source result | NeuralHorner transfer and limit |
| --- | --- | --- |
| DAgger — Ross, Gordon, and Bagnell, AISTATS 2011: https://proceedings.mlr.press/v15/ross11a.html | Sequential prediction is non-IID because earlier predictions determine later observations. DAgger repeatedly labels states visited by the current policy and aggregates rounds. | Direct fit: thresholded residue predictions become the next transition input and the exact recurrence is a cheap expert. NeuralHorner already uses DAgger and five targeted variants; the repo records that a Fermat-targeted pass closes that family but redistributes failures. The theory does not imply zero error for this nonconvex exact-match setting. |
| L2-SP — Li, Grandvalet, and Davoine, ICML 2018: https://proceedings.mlr.press/v80/li18a/li18a.pdf | Fine-tuning with a quadratic penalty to the pretrained starting point improved the paper's CNN transfer baselines. | Anchor updates to submitted v8, but always retain real incumbent replay and behavioral gates. Parameter proximity is not functional retention, bit-margin preservation, or arithmetic exactness. |
| Group DRO — Sagawa et al., ICLR 2020: https://arxiv.org/abs/1911.08731 | Optimizes worst predefined-group loss; the paper reports that naive Group DRO can fail in overparameterized zero-training-loss regimes and needs regularization/early stopping. | If failures remain clustered after trajectory matching, define training-only mechanistic groups by width, phase, digit, subtract count, carry/borrow length, and boundary distance. Worst-group BCE can still hide one fatal bit and supplies no exactness guarantee; never derive groups from sealed cases. |
| Boosted CVaR Classification — Zhai et al., NeurIPS 2021: https://proceedings.neurips.cc/paper/2021/hash/b691334ccf10d4ab144d672f7783c8a3-Abstract.html | For deterministic classifiers under 0/1 loss, an ERM minimizer also minimizes CVaR 0/1; a different guarantee needs more structure or randomization. | If used, tail-weight a differentiable signed-margin/bit-loss surrogate inside predeclared groups. Never call surrogate-tail improvement worst-case exact-match improvement. |
| State Passing — Buitrago and Gu, ICML 2025: https://proceedings.mlr.press/v267/buitrago25a.html | The paper attributes some recurrent length failures to attainable but unexplored states and reports improvements from post-training on states produced by other sequences or prior chunks. | Conceptually relevant to coverage, but NeuralHorner resets its BiGRU hidden state every arithmetic transition and carries explicit hard residue bits. Test only an equal mix of exact-prefix and detached model-visited semantic residue states with exact labels. The paper gives no evidence for modular arithmetic, this BiGRU, or exact 4096-step rollouts. |

### Ranked experiment

The highest-information incremental ablation is retention-constrained semantic-
state aggregation, not another plain DAgger pass:

1. From v8, run three short aggregation rounds on fresh, nonsealed 2048-bit
   schedules. Sample prefix positions across the full horizon and mechanistic
   strata; label equal exact-prefix and learner-visited residue states with the
   exact transition.
2. Aggregate every round. Train with a fixed incumbent replay set plus L2-SP.
   Compare equal-compute IID-only, exact-prefix-only, and mixed-state controls.
3. Select lexicographically: zero fresh-retention regressions, zero fresh-rollout
   regressions, then largest minimum correct-bit margin. Use at least three seeds
   and keep final-disjoint rows sealed until the recipe, checkpoint, and
   interpolation are frozen.
4. Only if errors still relocate, add mechanistic Group DRO under the same data
   and anchor. Do not start with unstructured CVaR.

STOP on any official-tier or frozen-family regression, a worse first-divergence
count, bf16/CUDA flip, runtime regression, or failure to beat controls at equal
compute. This is a plausible small ablation, not a source-backed expected fix.

## 2026-08-01 — Speed is Confidence

Primary source:

- Joshua V. Dillon, *Speed is Confidence*, arXiv 2601.19085v2, revised
  2026-01-29: https://arxiv.org/abs/2601.19085v2

This result is closer to the proposed exploration question than image-generation
XM because it keeps `K=4` parallel latent reasoning states during training and
backpropagates only through the lowest-loss winner. On Sudoku-Extreme, v2 reports
`96.9% +/- 0.6%` for the resulting single-forward model versus an
`86.1%` single-pass baseline; that baseline reaches `97.3%` with test-time
augmentation. All experiments used one RTX 5090. The abstract reports 1.5 hours
for the high-accuracy K=1 run and six hours for full K=4 WTA; its earlier 7,000-
step K=1 threshold takes about 40 minutes. These are distinct operating points,
not interchangeable training-cost denominators.

Mechanism fit is still weak for NeuralHorner's deployed transition. Sudoku can
support multiple useful internal reasoning trajectories even with a unique final
grid. NeuralHorner exposes one exact next-state bit vector at every step, with no
target-free way to select a lucky candidate at inference. The result supports
testing whether more *training search* can improve one deterministic model, but
does not justify literal minimum-loss best-of-K output selection.

The transfer therefore remains input/state exploration: sample K valid
transition states, select the lowest signed target margin, and train one output
with retention and L2-SP. Pair it with same-K random selection and an across-K
fixed-compute frontier. Do not quote the Sudoku accuracy or training-time ratios
as expected modular-arithmetic gains.

Explicit nonclaims:

- The paper has not been reproduced in this worktree.
- Its result is for a Tiny Recursive Model on Sudoku, not modular arithmetic,
  exact bit transitions, or a 4096-step BiGRU rollout.
- Lowest-loss latent-state WTA is not the same algorithm as worst-margin sampling
  over distinct exact NeuralHorner input states.

## 2026-08-01 — Intra-transition mechanism and local robustness

Primary sources:

- Weiss, Goldberg, and Yahav, *Extracting Automata from Recurrent Neural
  Networks Using Queries and Counterexamples*, ICML 2018:
  https://proceedings.mlr.press/v80/weiss18a.html
- Ko et al., *POPQORN: Quantifying Robustness of Recurrent Neural Networks*,
  ICML 2019: https://proceedings.mlr.press/v97/ko19a.html
- Huang, Geng, and Kolter, *Equilibrium Reasoners: Learning Attractors Enables
  Scalable Reasoning*, arXiv 2605.21488v1:
  https://arxiv.org/abs/2605.21488

### Architecture-fit correction

NeuralHorner does not carry a GRU hidden vector between arithmetic transitions.
Each call runs a two-layer bidirectional GRU from its normal reset state over the
bit positions of one `(s,x,p,d)` transition, thresholds the output logits, and
carries only the explicit predicted residue bits to the next Horner step.
Therefore EqR-style attractor, restart, or multi-step latent-recovery experiments
are a mechanism mismatch and are rejected. POPQORN supports studying local GRU
robustness, and automata extraction motivates counterexample-driven state
abstraction, but neither establishes that NeuralHorner represents carries.

### Smallest falsifiable mechanistic pilot

After the active qualification battery has finished, freeze both v8 and the
repaired candidate and use only existing nonsealed F11-development transitions
plus a predeclared public matched set. Do not train and do not open a sealed or
battery row.

1. Manually instrument the normal-reset GRU recurrence and first require clean
   logits to match the ordinary cell within `1e-6`, exact decoded bits to match,
   and self-patching to have zero effect. Start with four transitions; stop
   before causal analysis if this instrumentation gate fails.
2. For a pilot, freeze 16 bottom-margin and 16 median-margin target positions,
   balanced across orientation and modular-reduction branch. Label each position
   from the exact integer transition with carry-in/out, borrow-in/out, subtract
   count, signed boundary distance, and propagate-run length. These are reference
   labels, not assumed neural semantics.
3. Match natural donors on width, phase, bit position, reduction branch, target
   bit, a frozen local `(s,x,p)` bit window, and hidden norm. For each target use
   a same-carry donor and the closest opposite-carry donor. Patch forward and
   backward recurrent directions separately and recompute only the affected
   within-transition suffix or reverse-prefix. Never roll a patched output into
   another Horner transition and never move hidden vectors across models.
4. Compare opposite-label patches with self-patches, same-label patches, and
   equal-L2 random-direction perturbations using the exact opposite-patch norm.
   Report all-bit signed-margin change, bit flips, and effect versus distance
   from the patched position. Freeze pairs, norms, strata, and the paired
   statistic before viewing outcomes.
5. Scale to 64 bottom-margin plus 64 median-margin positions only if the pilot's
   opposite-label effect exceeds both controls with a predeclared paired 95%
   bootstrap interval excluding zero and has the expected directional footprint.

STOP if clean instrumentation differs, hidden reset is violated, fewer than 16
balanced pilot pairs survive, patch norms differ by more than 1%, selection is
changed after outcomes, the effect is global/nondirectional like random
off-manifold disruption, or any sealed/battery row is accessed. If the repaired
candidate has more same-transition flips or a worse margin curve than v8 at
equal perturbation norm, do not claim that repair improved robustness.

### Explicit nonclaims

- A positive result would be evidence for an intra-transition carry/borrow
  representation, not proof that the network implements the exact algorithm.
- It would not establish multi-step self-correction, an attractor, global
  robustness, or competition qualification.
- A negative result would reject this causal representation claim, not the
  candidate's observed exact outputs.

## Template for the next result

- Date and immutable source/version:
- Mechanism in one paragraph:
- Strongest directly supported result and its denominator:
- What is missing or unreleased:
- NeuralHorner mechanism fit:
- Smallest compute-matched falsification:
- Sealed data and promotion gate:
- STOP conditions:
- Explicit nonclaims:
