# What this is, what it isn't, and answers to the obvious questions

We share this in the spirit of the Contributor Network: a real result with its limits stated plainly. This is ongoing research.

## What is established (read this first)

- **One qualified/submitted incumbent:** `TrickyRex/bitserial-modmul-v8@301938693892043d66a8bf6ec60a9c5ab85549d4`. No research candidate replaced it.
- **Tiers 1-10 clear** on the official open-source scorer (htA90 = 10, overall 1.00), across 3 independent seeds, deterministic on re-run.
- **Local H100 timing:** 163-174 s per run against the open-source scorer's 300 s limit. This does not establish the same margin on different hardware.
- **Learned, not a circuit:** randomizing the weights collapses every tier from 1.00 to 0.00 (64/64 to 0/64).

## What we did NOT establish

- **Not exact.** The model passes the scorer's random-operand distribution but fails a sparse structured set: on first contact with the now-frozen development battery, the only failing family was Fermat numbers (`2^(2^n) + 1`), power-of-two-adjacent operands (759/768 = 98.83%).
- **Automated leaderboard evaluation = 100%; `ROB-CONFIRMED/UNARCHIVED`; final ranked standing pending.** Rob reports that the submission dashboard scored tiers 1-10 = 100% (htA90 = 10) at 260s, inside the 300s budget. The primary dashboard output is absent here. This is not a human or organizer verification.
- **bf16 primary evidence is not archived.** Repository prose records 0 flips against fp32 on the tested battery with minimum |logit| 3.017, but the primary output is absent. This is `LOCAL-RESULT/UNARCHIVED`; the checking script alone is not a result receipt.
- **No organizer compliance ruling.** The schedule-specific question was never posted or sent; no ruling was requested or issued.
- **The Lean proof covers the algorithm, not the network.** The cell-to-step bridge is open.
- **No successful direct-model repair.** The L2-SP direct candidate failed its frozen-development retention gate at 509/512 attempted rows. It was not submitted.

## Anticipated questions

**You said tier 10 was a wall in the earlier write-up. What changed?**
The earlier fixed-width model plateaued at tier 10. The current entry uses the same learned cell with one inference change: the per-step state width is sized to the prime's bit-length per batch (dynamic-L) instead of a fixed maximum. This is exact for the symbolic recurrence; for the trained bidirectional GRU, full-width and dynamic-L inference agreed on the recorded 6/6 comparison. Dynamic-L then cleared tier 10 to 1.00 on the official scorer across 3 seeds. The capability is still the trained cell, not the schedule: randomizing the weights still collapses every tier to 0.

**Does dynamic-L change the answers, or just the speed?**
The symbolic state only needs the prime's width, so sizing away padded bits removes wasted compute on the easy tiers. For the trained bidirectional GRU, full-width and dynamic-L inference were output-identical on a recorded 6/6 comparison; that is not a theorem for every input. The bf16/fp32 observation is separately battery- and backend-scoped.

**Why did the local H100 run enter the scorer's 300 s limit?**
The fixed-width run spent roughly 16 s on each easy tier on a padded state (about 110 s of waste). Dynamic-L collapses the easy tiers to about 11 s total; the deep tier is unchanged because it genuinely needs the depth. On that H100, net wall-clock went from 383 s to 163--174 s. This comparison does not transfer a margin to other hardware.

**Did you try CUDA graphs for the speedup?**
Yes, and they lost: 402 s against the 383 s baseline. The workload is compute-bound (the per-step GRU runs over the L-wide state), not launch-bound, so graph capture did not help. Dynamic-L attacks the actual wasted compute, which is why it is the right lever. We report the negative result on purpose.

**Is the timing margin safe on the organizers' hardware?**
No general hardware-margin claim is justified. Rob's unarchived submission-dashboard observation is 260 s inside its 300 s limit, while the receipt-backed local H100 runs took 163--174 s. Those are two specific evaluation surfaces, not evidence that the local margin transfers to other hardware.

**Is the bit-serial loop a hand-coded arithmetic algorithm (which the rules forbid)?**
The loop schedule and bit-packing are fixed by hand; the arithmetic, the per-step reduction `s' = (2s + d·x) mod p`, is learned. Randomizing the cell's weights collapses every tier to 0.00, which is the organizers' own anti-cheat condition. The forward path has no `int()%p` on the operands, no symbolic-math library, no lookup table, and no compare-against-`p`. The rules source checked on 2026-08-11 permits deterministic feedback-free token loops while explicitly prohibiting deterministic full-width operand pre-reduction; NeuralHorner's reductions are learned-cell passes. This source comparison is not a schedule-specific organizer ruling: no question was sent and no ruling was requested or issued. The exact pins and hashes are in the [rules-refresh receipt](research/receipts/rules_refresh_20260811.json).

**How is this different from the lookup-table or direct-arithmetic submissions?**
Those declare no training and compute the answer with built-in integer arithmetic or prebuilt tables, so they work for any weights. This entry has ~471K trained parameters and dies when you randomize them. Run the check yourself: `python scripts/verify_no_shortcut.py model --randomize`.

**Was it scored by the competition's evaluation, not just self-run by us?**
Rob reports that the submission dashboard scored tiers 1-10 = 100% in 260 s, inside that evaluation's 300 s limit. The primary dashboard output is absent here, so this is `ROB-CONFIRMED/UNARCHIVED`, not a human verification or a portable hardware-timing result; the final ranked standing remains pending.

**What exactly does the Lean proof establish?**
That the integer double-and-add recurrence the loop imitates equals `(a·b) mod p` for any bit length, with the canonical bound `s < p`. Axioms are `propext` and `Quot.sound` (standard Mathlib); no `sorry`/`admit`. It says nothing about the network's weights; the cell-to-step bridge to the trained network is unproven.

**Why does it fail on Fermat numbers if tiers 1-10 are 1.00?**
The scorer's tiers use a random-operand distribution, where the cell is right. The structured battery deliberately probes families that were disjoint from original v8 training; on first contact, the only one that broke was power-of-two-adjacent operands (`2^(2^n) + 1`). That is the Neural GPU fragility (Price et al. 2016), now characterized rather than hidden. Because later repair attempts were selected against this battery, it is now development data rather than a sealed generalization set.

**Did the direct two-pass candidate fix that failure?**
It fixed the known F11 transition but not the model globally. The L2-SP interpolation cleared a receipt-bound 193,116-transition F11 exact-prefix gate. That gate evaluates exact teacher-prefix transitions and is not a literal sequential, batched, or official-inference receipt. Rob also confirms `100/100` on each scored playground tier and `40/100` on unscored Tier 0, but that result is `ROB-CONFIRMED/UNARCHIVED`: its screenshot/log, runtime, and evaluator identity are not preserved here. The candidate was not submitted or qualified. Frozen-battery v3 then failed at 509/512 attempted rows on fixed-Hamming cases `003`, `035`, and `110`; the other 1,024 planned rows were not run.

**Was the 509/512 run an infrastructure failure?**
No. The run stopped fail-closed on three wrong answers, and its validator found no artifact, source, fixture, or cache-provenance mismatch. It is genuine model-failure evidence. It is also only 512 attempted rows, not a 1,536-row completed result; reporting `509/1536` or treating the unrun rows as failures would be equally wrong.

**What did the four-way ablation show?**
Crossing v8/repaired weights with original/direct schedules gave three different truth tables. Case `003` failed only for repaired weights plus the direct schedule, so an interaction is required. Case `035` failed under both direct schedules, so the direct schedule is sufficient. Case `110` failed under repaired weights under either schedule, so the repair is sufficient. Each failing arm had exactly one wrong teacher-state transition. This is a causal decomposition of a failed repair, not evidence that the candidate nearly qualified.

**Did the direct schedule make inference 20% faster?**
Not established. The exact program removes one learned reduction pass, and a source-bound recurrent bit-work proxy is 20.001% lower over the scored public benchmark. No corresponding end-to-end latency result was measured for the candidate, so 20.001% must not be described as a wall-clock speedup.

**Are there research ideas left?**
Two are recorded but unrun: a zero-training screen of the existing function-space checkpoint on `035 → 110 → 003` and the complete F11 set, and a no-training design stage for a retention-constrained K-state counterexample miner. Neither is a result. No new training has run, and candidate promotion is stopped.

**Is the method novel?**
Not as a shared formulation. XllentAI's immutable historical [`modular_arithmetic` snapshot](https://huggingface.co/XllentAI/modular_arithmetic/tree/3d2c226c2382140b890026bdfdd59485daa192ba), which predates NeuralHorner's public repository, already learned a modulus-conditioned Horner transition with hard binary state and reported held-out-prime transfer. Its [2026-08-01 current pin](https://huggingface.co/XllentAI/modular_arithmetic/tree/f704813dc64d7c04cc9ba7dbf9a3a281c431628d) now reports two shared TCN weight sets spanning tiers 1-10. NeuralHorner's narrower contribution is one compact ~471K-parameter BiGRU checkpoint reused for learned reductions and multiplication without Python-side operand `% p`, together with challenge-width efficiency, first-contact failure localization, causal schedule/checkpoint diagnostics, and the machine-checked integer scaffold.

## How to scrutinize us

The incumbent scorer, determinism, timing, and weight-collapse numbers are backed by receipts under `model/receipts/` and the named scripts. The bf16-margin number is an unarchived local observation, not receipt-backed. Direct-candidate and causal receipts are indexed in [`research/README.md`](research/README.md), including frozen-battery v3 and the fixed-Hamming four-way validation. The Lean sources are in `lean/`; they prove the exact integer schedules and a conditional exact-prefix bridge, not universal checkpoint correctness.
