# bit-serial-v8 (NeuralHorner, dynamic-L) — local scorer result

Same learned cell as v7 (L=2048 weights, warm-started + trained for tier 10), with one
inference change: the per-step state width is sized to the prime's bit-length per batch
(`L_eff`) instead of a fixed 2048. This is exact for the symbolic recurrence because
padded high bits are zero. For the trained bidirectional GRU, the recorded full-width/
dynamic-L comparison is 6/6 output-identical, not a universal theorem. The change removes
the wasted compute the easy tiers spent on a 2048-wide state. Same weights; randomizing
them still collapses every tier to 0. Scorer results are from the official open-source
scorer on a single H100.

## Official scorer (full 1100), 3 independent seeds
- seed a1a1a1a1: htA90 = 10, overall 1.00, tier 10 = 1.00 (completed), wall 170s
- seed b2b2b2b2: htA90 = 10, overall 1.00, tier 10 = 1.00 (completed), wall 163s
- seed c3c3c3c3: htA90 = 10, overall 1.00, tier 10 = 1.00 (completed), wall 174s
- determinism re-run (a1): identical per-tier counts -> deterministic
The published budget is 300s. These runs leave 126--137 seconds of headroom on
this H100 setup; they do not establish timing on different organizer hardware.
Receipts: `receipts/d_*.json`.

## Why it is now under budget (per-tier, baseline -> dynamic-L)
- Baseline (fixed L=2048): inference 305.7s; easy tiers 0-6 ~16s EACH (~110s wasted on padding).
- Dynamic-L: easy tiers 0-6 ~11s TOTAL; tier 10 unchanged (~118s); determinism check halved.
- CUDA graphs were tried and LOST (402s > 383s) -- the per-step cost scales with the state width L,
  not launch dispatch, so graph capture does not help; dynamic-L cuts the per-step compute, which is why it wins.

## What is established
- Accuracy: tiers 1-10 = 1.00 on the official scorer, 3 seeds, deterministic.
- Timing: 163-174s < 300s budget, with margin.
- Learned, not a circuit: randomizing the weights collapses every tier to 0.00 (64/64 -> 0/64).
- Rules-aligned evidence: the public rules permit deterministic token-feeding
  loops when the answer comes from trained parameters; preprocessing is
  per-argument, and the audited forward path contains no symbolic math,
  big-integer modular multiplication, or lookup table. No manual organizer
  compliance ruling has been requested or issued.

## What is NOT yet established
- Full secret-seed leaderboard ranking is theirs to post. Rob reports an automated submission-dashboard score of all ten tiers at 260s under the 300s budget, but the primary output is absent here; label this `ROB-CONFIRMED/UNARCHIVED`, not human verification.
- The bf16/fp32 prose records 0 flipped answers on the tested battery with
  minimum |logit| = 3.017, but the primary output is absent. This is
  `LOCAL-RESULT/UNARCHIVED`; the checking script alone is not a result receipt.
- The organizer compliance status of the hand-coded Horner schedule has not
  received a manual ruling. A clarification draft exists locally but has not
  been posted or sent.
- The Lean package proves the integer algorithm, not the learned network (cell->step bridge open).

## Exactness boundary and later repair result

The submitted v8 checkpoint is not universally exact. On first contact with a
structured 768-row battery whose families were disjoint from v8 training, it
scored 759/768; all nine failures were in the Fermat family. That battery was
subsequently used for repair selection and is now frozen development data, not a
sealed estimate for later candidates.

A later direct two-pass/L2-SP candidate repaired the selected F11 trajectory but
failed a provenance-validated retention run after 509/512 attempted rows, with
three failures in the fixed-Hamming family; the remaining 1,024 rows were not
run. The submitted v8 artifact therefore remains the only qualified submission.
See `../research/FRONTIER_STATE.md` for the receipt-bound causal analysis and
exact nonclaims.
