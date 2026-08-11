# Canonical direct two-pass NeuralHorner

Status: architecture ablation with a known F11 counterexample; do not submit.

## Candidate

This artifact keeps the submitted v8 transition weights unchanged and replaces
the original three-pass rollout with a swap-invariant two-pass schedule:

1. canonically order the operands by encoded length, then lexicographically;
2. reduce the longer operand with the learned transition and `x = 1`;
3. scan the raw MSB-first bits of the shorter operand with the learned residue
   as `x`.

For the exact recurrence this computes the modular product in either operand
orientation. Canonical ordering makes `(a, b, p)` and `(b, a, p)` execute the
identical learned trajectory.

This routing avoids reducing F11 on the nine legacy cases only because their
companions are longer. Those cases influenced the router and are development
data, not held-out evidence. It is not a general repair: for F11 paired with a
shorter operand, including `F11 * 1`, the unchanged v8 cell reduces F11 and
re-enters the diagnosed 642-wrong-bit maximum-width transition. This checkpoint
line is therefore stopped unless its weights are repaired.

Checkpoint SHA-256:
`294fbf809ad42bb0ac66dd51303d21ac5dddeca9a081b5790f1af39d36b21609`.

## Verified locally on 2026-08-01

- Candidate and fail-closed research-gate suite: `90 passed`.
- Official scorer source: clean detached commit
  `82510bba00a1126649bd76dd1a451f14d0b3eb60`; scorer suite `116 passed, 1 skipped`.
- Current official manifest validation, artifact-size check, static check, and
  model load: pass; zero static findings; below the 20 GB limit.
- Forward-path audit: no classical modular arithmetic, lookup table, or
  compare-against-prime correction detected.
- Current learned-checkpoint smoke receipt: 16/16 exact across prime widths
  2, 3, 31, 32, 33, 63, 64, and 65; singleton/batch/reverse/three permutations
  and operand swaps agree bit-for-bit. The receipt binds the artifact, hardened
  runner, cases, output cardinalities, and environment. This remains a
  small-width smoke, not a promotion gate. See
  `receipts/batch_invariance_smoke.json`.
- Current p33 learned no-shortcut screen: 4/4 exact with every shortcut exclusion
  true; deterministic randomized weights score 0/4. Both receipts bind the
  hardened runner and distinguishing shortcut case.
- Lean exact-recurrence package: `lake build MAC` passes;
  `directModmul_eq` and `directModmul_swap_eq` cover both schedule branches.
- Public-benchmark architecture proxy at official commit `82510bb`: 20.001%
  less dominant recurrent bit-work over scored tiers 1-10. This is not a
  wall-clock timing claim; Tier 0 is excluded from this headline because some
  diagnostic cases lie outside the candidate's supported modulus range.
- Current v3 route/source/artifact-bound MPS-fp32 induction reproduces `F11 * 1`
  for this exact artifact and program in both canonical orientations. Each
  orientation is exact for 2,048 prefix transitions, then produces 642 wrong
  bits at global/phase step 2,048; 2,049 of 2,050 transitions are exact and the
  minimum signed correct-target logit margin is -20.8369. Because every earlier
  transition and the observed manifest route are verified, the first divergence
  is free-running-valid, not a borrowed-state probe. See
  `receipts/f11_decisive_original_mps_fp32_routebound_v3.json` and
  `receipts/f11_decisive_swapped_mps_fp32_routebound_v3.json`. Older receipts
  remain historical support only.

## Promotion blockers

The known `F11 * 1` compositional counterexample blocks promotion regardless of
the small-width passes below.

The canonical `../../model/` artifact and its submitted commit remain unchanged.
No v8 full-scorer, adversarial, timing, bf16-margin, trace-certification, or
randomized-weight receipt is inherited by this candidate.

Before any upload or SAIR slot update, this exact artifact must pass:

- official 1100-case public benchmark and three fresh seeds at 100% on the
  organizer-confirmed final hardware;
- the 768-case legacy adversarial battery in both input orientations;
- all nine frozen F11 cases, the fresh companion-width/equal-length sweep, and
  transition-level traces in both orientations;
- randomized-weight collapse, determinism, preprocessing isolation, and batch
  invariance at competition-scale widths;
- the five-minute full-evaluation budget on the organizers' final hardware;
- CUDA/bf16 parity separately, only if the final contract confirms CUDA.

Apple MPS is too slow for those maximum-width gates. The currently published
sandbox is CPU-only and therefore does not validate the candidate's CUDA path;
organizer confirmation of final evaluation hardware is required. The attempted
55-case official smoke did not reach timed inference before manual interruption,
so it supplies no accuracy or runtime evidence.
