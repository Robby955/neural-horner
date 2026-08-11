# Canonical direct two-pass plus L2-SP interpolation

Status: primary research candidate; mirrored to a separate Hugging Face testing
revision, not submitted to SAIR and not promotion-ready.

Immutable testing revision:
`TrickyRex/neural-horner-direct-l2sp-a0875@b5cee6f1592b1eeb9822e4773e5e294caf013f60`.
The downloaded four-file package was SHA-256 reverified after upload.

Rob confirms that the immutable testing revision completed the SAIR playground
with `100/100` on every scored tier 1--10 and `40/100` on the unscored Tier-0
pure-multiplication diagnostic. He confirms live-playground v8 also displays
`40/100`, so the candidate has no live Tier-0 regression. The screenshot/log,
runtime, and evaluator identity cannot yet be archived here because the desktop
thread is unavailable; label this `ROB-CONFIRMED/UNARCHIVED`. Older local
open-source-scorer v8 receipts record `60/100` on a different historical surface.

This artifact uses the canonical direct two-pass inference program with
checkpoint interpolation coefficient `alpha=0.875`:

- base v8 SHA-256:
  `294fbf809ad42bb0ac66dd51303d21ac5dddeca9a081b5790f1af39d36b21609`;
- L2-SP `lambda=1e-3` step-1500 endpoint SHA-256:
  `97db605caf949460b06bfb916d6a3ab0039547e6b79c89fbc48a1025a580e421`;
- interpolated output SHA-256:
  `6d1c8a09e555778fb7961778678a1473e681707ada0d20f1465712313b3e8f01`.

The derived checkpoint contains only `L`, model configuration, and state dict.
Source checkpoint steps, scores, and training fields are retained only as source
provenance in `provenance.json`; they are not claims about this artifact.

## Prior screening context

On CPU fp32, this interpolation was exact on 68/68 repair-screen rows,
representing 56 unique transition tuples. The minimum signed bit-logit margin
was 1.4953. The screen explicitly excluded 12 final-disjoint rows and evaluated
individual transitions, not complete rollouts. It is therefore a candidate
selection signal, not qualification evidence; its 12 omissions are not the
current F11 qualification remainder. The artifact, runner, source
case file, selected rows, environment, and exact counts are bound in
`transition_screen_cpu.json`.

The current qualification runner is v3, SHA-256
`87e0b15e8a234ea03c43f5621d11fc1b83da5a1480d20efbc985c29ad22268f1`.
It binds the runner plus helper/scorer source set, hashes artifacts before load,
after load, and after execution, validates the observed manifest route, gates on
strictly positive margin, and records actual logit/model tensor dtypes. Under
that runner on MPS fp32, the complete decisive `F11 * 1` path is exact in both
orientations: 4,100/4,100 transitions, minimum signed correct-target margin
5.01655, source-set SHA-256 `0ca78bb1fdfc797ff559aaca0f623351022f92fb0a6208d892050e8832dc6837`,
and no artifact/source mutation. See
`receipts/f11_decisive_both_mps_fp32_routebound_v3.json`.

Current v3 also certifies all seven companion widths in both orientations:
14/14 results and 53,840/53,840 transitions, minimum signed margin 5.01657,
the same source-set SHA-256, exact observed routes, fp32 captured logits, and no
artifact/source mutation. See
`receipts/f11_companions_both_mps_fp32_routebound_v3.json`. The current v3
runner also certifies the three equal-length ties in both orientations: 6/6
results and 24,588/24,588 transitions, minimum margin 5.01657, exact routes,
and no artifact/source mutation. See
`receipts/f11_ties_both_mps_fp32_routebound_v3.json`. Finally, current v3
certifies all nine legacy cases in both orientations: 18/18 results and
110,588/110,588 transitions, minimum margin 7.19223, exact routes, and no
artifact/source mutation. See
`receipts/f11_legacy_both_mps_fp32_routebound_v3.json`.

The strict union validator returns `status=validated_exact` for all four current
v3 component receipts under one verified scorer checkout/import origin: 20
unique cases, 40 unique oriented cases, and 193,116/193,116 exact transitions.
The components contribute 4,100 decisive, 53,840 companion, 24,588 tie, and
110,588 legacy transitions. All routes are exact, logits are fp32, margins are
positive, and no artifact/source mutation occurred. The union binds runner
`87e0b15e8a234ea03c43f5621d11fc1b83da5a1480d20efbc985c29ad22268f1`,
source set
`0ca78bb1fdfc797ff559aaca0f623351022f92fb0a6208d892050e8832dc6837`,
artifact set
`9b0ba4f1c6ff5ed8ccf7b64b5b173baf10b8881c407ce430ec3338adcc0e06fc`,
and full case set
`54f9d2342fcdbfa6fc21a5d6a6560d13364059be3c48174a6c66a407834a86ee`;
the four component receipt hashes are embedded in the union. Receipt:
`../../research/receipts/f11_primary_direct_routebound_v3_union.json`, SHA-256
`5890134071d914c3bd4e77852c7a3379d56b41c30543871e0a7686708b799920`.
This completes vectorized exact-prefix induction, not literal sequential,
batched, or official inference and not promotion. Older CPU and injected
two-step receipts remain historical support only.

The small-width batch smoke is 10/10 exact at prime widths 2, 3, 31, 32,
and 33; singleton, batch, reverse, two permutations, and swapped operands agree
bit-for-bit (`batch_invariance_smoke.json`). A separate 33-bit screen is 4/4
with the trained checkpoint and 0/4 after deterministic weight randomization
(`no_shortcut_p33.json` and `randomized_collapse_p33.json`). These tests support
the learned-computation claim only at small width; they do not replace the
competition-scale collapse and invariance gates.

## Required next gates

Only submitted v8 is currently qualified. The L2-SP candidate is now stopped by
a validated frozen-retention failure; no new training ran.
Frozen-battery v1 is `INTERRUPTED`, not passed. Its preserved receipt still says
`running`, but the process ended after the original-orientation Fibonacci family
completed 128/128 exact; the remaining 1,408 rows did not run. Frozen-battery v2
is also `INTERRUPTED`, not passed: it was stopped immediately when preflight
review found that its old runner did not bind cache provenance into the receipt.
It completed no family and supplies no model-failure evidence.

Frozen-battery v3 is `FAILED`, not interrupted: 509/512 attempted rows were
exact, with failures at fixed-Hamming source indexes 3, 35, and 110. The other
1,024 rows were not run. The one-shot validator found no identity or cache
blocker. Receipt SHA-256 is
`1b9d94f9c3155f6fefaae34c268ad90fc2630f1cbc0a6cb505e23eec05ff673c`.

The receipt-bound four-way causal trace validates three distinct mechanisms:
source 003 requires the repaired-weight/direct-schedule interaction; source 035
is caused by the direct schedule and worsened by repair; source 110 is caused by
the repaired weights under either schedule. Union validation SHA-256 is
`e47583c1e68d2985ffc0814b75473a8cc6c4350ecfb86591b5f9783a98a0d020`.

1. Run the K-state miner as a no-training dry run on the three real divergence
   states, with retention, paired random-selection, compute, and STOP rules
   frozen first.
2. Screen the existing function-space candidate on the three cases and complete
   F11 set before authorizing training.
3. Only a candidate that clears both may proceed to the sealed battery, public
   seeds, collapse tests, and final hardware gate.

Until those pass, no claim is made about all-F11 repair, CUDA/bf16 behavior,
fresh sealed generalization, official accuracy, or runtime.
