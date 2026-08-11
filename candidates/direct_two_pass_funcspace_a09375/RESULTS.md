# Direct two-pass interpolation toward a function-space-trained endpoint

Status: high-priority research candidate; not submitted and not promotion-ready.

This artifact uses the same canonical direct two-pass inference program as
`../direct_two_pass` with checkpoint interpolation coefficient `alpha=0.9375`:

- base v8 SHA-256:
  `294fbf809ad42bb0ac66dd51303d21ac5dddeca9a081b5790f1af39d36b21609`;
- function-space step-1500 endpoint SHA-256:
  `2cfe7bb6b7e42a2f2e8863dd6e9122ca51c8c3435180db388e8b338a02c4497f`;
- interpolated output SHA-256:
  `2fbc717be66b3abbf4815dce657f1a9ee862b17d9dc26efd962b9063dbdb8a16`.

`provenance.json` binds the coefficient, source identities, architecture, tensor
count, and parameter count. The derived checkpoint strips stale source steps and
scores. The historical specialist's coarse training probes are provenance
context only; they are not evaluation evidence for this artifact.

## Verified locally on 2026-08-01

- Interpolation and compatibility tests pass as part of the 90-test candidate
  and fail-closed research-gate suite.
- Current official manifest validation, artifact-size check, static check, and
  model load at scorer commit `82510bb`: pass; zero static findings; below the
  20 GB limit.
- Learned-checkpoint smoke receipt: 10/10 exact across prime widths
  2, 3, 31, 32, and 33; singleton/batch/reverse/two permutations agree
  bit-for-bit; all operand swaps agree bit-for-bit. See
  `batch_invariance_smoke.json`.
- Forward-path audit passes.
- CPU-fp32 exact-prefix induction certifies the complete decisive `F11 * 1`
  program path in both canonical input orientations: 2,050/2,050 transitions
  per orientation (4,100 total), zero failures, and minimum signed
  correct-target logit margin 4.9199 (`f11_decisive_inductive_cpu.json`). This
  is candidate-owned full-path evidence under the earlier runner for one case.
  The runner has since been hardened to v3; this candidate has no v3 receipt and
  inherits none from the primary candidate.

## Promotion blockers

This checkpoint has not run the current v3 F11 suite, full public scorer, fresh
seeds, the complete adversarial battery, randomized-weight collapse,
or a timing gate on the final organizer-confirmed hardware. CUDA parity and
timing remain separate gates only if GPU evaluation is confirmed. It must
satisfy the same fail-closed promotion matrix as the unchanged-weight direct
candidate before it can occupy a submission slot.
