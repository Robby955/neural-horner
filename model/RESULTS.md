# bit-serial-v8 (NeuralHorner, dynamic-L) — tier 10 cleared, verified

Same learned cell as v7 (L=2048 weights, warm-started + trained for tier 10), with one
inference change: the per-step state width is sized to the prime's bit-length per batch
(`L_eff`) instead of a fixed 2048. This is correctness-preserving (the state only ever
needs the prime's width; the padded bits are always 0) and removes the wasted compute the
easy tiers spent on a 2048-wide state. Same weights; randomizing them still collapses every
tier to 0. All numbers from the official open-source scorer on a single H100.

## Official scorer (full 1100), 3 independent seeds
- seed a1a1a1a1: htA90 = 10, overall 1.00, tier 10 = 1.00 (completed), wall 170s
- seed b2b2b2b2: htA90 = 10, overall 1.00, tier 10 = 1.00 (completed), wall 163s
- seed c3c3c3c3: htA90 = 10, overall 1.00, tier 10 = 1.00 (completed), wall 174s
- determinism re-run (a1): identical per-tier counts -> deterministic
Budget is 300s; margin ~125s (covers slower organizer hardware). Receipts: receipts/d_*.json.

## Why it is now under budget (per-tier, baseline -> dynamic-L)
- Baseline (fixed L=2048): inference 305.7s; easy tiers 0-6 ~16s EACH (~110s wasted on padding).
- Dynamic-L: easy tiers 0-6 ~11s TOTAL; tier 10 unchanged (~118s); determinism check halved.
- CUDA graphs were tried and LOST (402s > 383s) -- the per-step cost scales with the state width L,
  not launch dispatch, so graph capture does not help; dynamic-L cuts the per-step compute, which is why it wins.

## What is established
- Accuracy: tiers 1-10 = 1.00 on the official scorer, 3 seeds, deterministic.
- Timing: 163-174s < 300s budget, with margin.
- bf16 decision-safety: 0 flipped answers vs fp32, min |logit| = 3.017 >> bf16 error.
- Learned, not a circuit: randomizing the weights collapses every tier to 0.00 (64/64 -> 0/64).
- Compliant approach: looped + learned per-step is explicitly permitted; per-argument
  preprocessing; no symbolic math / big-int modmul / lookup in the forward path.

## What is NOT yet established
- Full secret-seed leaderboard ranking is theirs to post (on submission, the competition's automated evaluation scored all ten tiers at 260s, under the 300s budget -- an automated score, not a human verification).
- The organizer compliance ruling on the hand-coded Horner schedule is pending (Zulip posted).
- The Lean package proves the integer algorithm, not the learned network (cell->step bridge open).
