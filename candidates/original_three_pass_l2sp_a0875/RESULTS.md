# Original three-pass NeuralHorner plus L2-SP interpolation

Status: causal control candidate; not submitted and not promotion-ready.

This artifact isolates checkpoint repair from inference routing. It has the
same interpolated checkpoint bytes as
`../direct_two_pass_l2sp_a0875/weights.pt`, but uses the incumbent's original
three-pass program:

1. reduce `a` with the learned cell;
2. reduce `b` with the learned cell;
3. scan the fixed-width residue of `b` using the residue of `a` as `x`.

Checkpoint SHA-256:
`6d1c8a09e555778fb7961778678a1473e681707ada0d20f1465712313b3e8f01`.
The base, endpoint, coefficient, configuration, and source metadata are bound
in `provenance.json`.

## Evidence scope

The cell-level 68-row screen and small-width program invariance screen are run
separately for this exact artifact. Even though the weight file is identical to
the direct candidate, no direct-schedule rollout or runtime receipt is inherited.

Verified locally on 2026-08-01:

- CPU-fp32 one-step screen: 68/68 rows exact over 56 unique transition tuples,
  minimum signed bit-logit margin 1.4953, with exactly 12 final-disjoint rows
  excluded (`transition_screen_cpu.json`).
- Small-width program screen: 10/10 singleton cases exact; every mixed-batch,
  reverse, permutation, and operand-swap result decodes to the same value
  (`batch_invariance_smoke.json`). Raw digit lists differ below 33-bit primes
  because the original program sizes leading-zero output width to the largest
  prime in a mixed batch. That is representation variance, not value variance.
- Minimal package at official scorer commit `82510bb`: manifest, size, static
  analysis, and model load pass; zero static findings, `L=2048`, 470,849
  parameters. The separate forward-path audit also passes.
- The 33-bit no-shortcut screen is 4/4 with trained weights and 0/4 after
  deterministic randomization (`no_shortcut_p33.json` and
  `randomized_collapse_p33.json`). This remains small-width evidence only.
- The current v3 route/source/artifact-bound induction runner certifies the
  complete decisive `F11 * 1` path on MPS fp32 in both input orientations:
  8,196/8,196 transitions, zero failures, and minimum signed correct-target
  logit margin 5.0166
  (`receipts/f11_decisive_both_mps_fp32_routebound_v3.json`). Since the direct
  candidate uses byte-identical weights and also passes its complete decisive
  v3 path, while older comparable original/direct v8 receipts fail the same
  learned transition, this is strong causal evidence that checkpoint repair—not
  routing—is sufficient for this case.
- An earlier runner is exact for all seven other companion widths in both
  orientations: 14/14 results and 82,512/82,512 transitions, minimum margin
  4.9165 (`receipts/f11_companions_both_mps_fp32.json`). This is strong
  diagnostic evidence, but its runner hash predates v3 and it is not inherited
  by the current qualification gate.

This control is required for causal interpretation:

- if both schedules pass with these weights, the checkpoint repair is primary;
- if only the direct schedule passes, routing is primary;
- if only the original schedule passes, the direct raw scan introduces a new
  trajectory failure;
- if neither passes, the one-step repair screen did not compose into rollouts.

## Promotion blockers

The artifact has not passed the remaining 19 F11 cases under v3, literal
sequential confirmation, the frozen and fresh adversarial
batteries, the official 1100 cases, fresh public-generator seeds,
competition-width randomized collapse and batching, backend parity, or the
organizer-confirmed 300-second runtime gate. It cannot occupy a submission slot
until the same fail-closed matrix as the direct candidate passes.
