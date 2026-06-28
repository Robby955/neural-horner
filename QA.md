# What this is, what it isn't, and answers to the obvious questions

We share this in the spirit of the Contributor Network: a real result with its limits stated plainly. This is ongoing research.

## What is established (read this first)

- **Tiers 1-10 clear** on the official open-source scorer (htA90 = 10, overall 1.00), across 3 independent seeds, deterministic on re-run.
- **Under budget:** 163-174 s per run against a 300 s budget (~125 s margin).
- **Learned, not a circuit:** randomizing the weights collapses every tier from 1.00 to 0.00 (64/64 to 0/64).
- **bf16-safe:** 0 flipped answers versus fp32 on the battery; min |logit| = 3.017, well above bf16 error.

## What we did NOT establish

- **Not exact.** The model passes the scorer's random-operand distribution but fails a sparse structured set: on the held-out battery the only failing family is Fermat numbers (`2^(2^n) + 1`), power-of-two-adjacent operands (759/768 = 98.83%).
- **Automated leaderboard evaluation = 100%; final ranked standing pending.** On submission, the competition's automated evaluation scored tiers 1-10 = 100% (htA90 = 10) at 260s, inside the 300s budget. This is an automated score, not a human or organizer verification.
- **Compliance ruling pending.** The ruling on the hand-coded Horner schedule is open (Zulip question posted, no response yet).
- **The Lean proof covers the algorithm, not the network.** The cell-to-step bridge is open.

## Anticipated questions

**You said tier 10 was a wall in the earlier write-up. What changed?**
The earlier fixed-width model plateaued at tier 10. The current entry uses the same learned cell with one inference change: the per-step state width is sized to the prime's bit-length per batch (dynamic-L) instead of a fixed maximum. That is correctness-preserving, since the padded high bits are always zero, and it cleared tier 10 to 1.00 on the official scorer across 3 seeds. The capability is still the trained cell, not the schedule: randomizing the weights still collapses every tier to 0.

**Does dynamic-L change the answers, or just the speed?**
It does not change answers. The state only ever needs the prime's width, so the padded bits carry no information; sizing them away removes wasted compute on the easy tiers. We confirmed determinism: a re-run gives identical per-tier counts, and bf16 decisions match fp32 exactly.

**Why is it suddenly under budget when the fixed-width version was not?**
The fixed-width run spent roughly 16 s on each easy tier on a padded state (about 110 s of waste). Dynamic-L collapses the easy tiers to about 11 s total; the deep tier is unchanged because it genuinely needs the depth. Net wall-clock went from 383 s to 163-174 s.

**Did you try CUDA graphs for the speedup?**
Yes, and they lost: 402 s against the 383 s baseline. The workload is compute-bound (the per-step GRU runs over the L-wide state), not launch-bound, so graph capture did not help. Dynamic-L attacks the actual wasted compute, which is why it is the right lever. We report the negative result on purpose.

**Is the timing margin safe on the organizers' hardware?**
The competition's automated evaluation reported 260s on submission, inside the 300s budget (slower than our 174s, but within budget). That is an automated score, not a human verification.

**Is the bit-serial loop a hand-coded arithmetic algorithm (which the rules forbid)?**
The loop schedule and bit-packing are fixed by hand; the arithmetic, the per-step reduction `s' = (2s + d·x) mod p`, is learned. Randomizing the cell's weights collapses every tier to 0.00, which is the organizers' own anti-cheat condition. The forward path has no `int()%p` on the operands, no symbolic-math library, no lookup table, and no compare-against-`p`. The rules permit looped models whose answer comes from trained parameters. The schedule-specific compliance ruling is still pending (Zulip).

**How is this different from the lookup-table or direct-arithmetic submissions?**
Those declare no training and compute the answer with built-in integer arithmetic or prebuilt tables, so they work for any weights. This entry has ~471K trained parameters and dies when you randomize them. Run the check yourself: `python scripts/verify_no_shortcut.py model --randomize`.

**Was it scored by the competition's evaluation, not just locally?**
Yes -- on submission, the competition's automated evaluation scored tiers 1-10 = 100% (260s, under budget). That is an automated score, not a human or organizer verification; the final ranked standing is theirs to post.

**What exactly does the Lean proof establish?**
That the integer double-and-add recurrence the loop imitates equals `(a·b) mod p` for any bit length, with the canonical bound `s < p`. Axioms are `propext` and `Quot.sound` (standard Mathlib); no `sorry`/`admit`. It says nothing about the network's weights; the cell-to-step bridge to the trained network is unproven.

**Why does it fail on Fermat numbers if tiers 1-10 are 1.00?**
The scorer's tiers use a random-operand distribution, where the cell is right. The held-out battery deliberately probes structured families; the only one that breaks is power-of-two-adjacent operands (`2^(2^n) + 1`). That is the Neural GPU fragility (Price et al. 2016), now characterized rather than hidden.

**Is the method novel?**
Not as a mechanism; it sits in the Neural GPU / looped-model lineage. The contributions are the modulus-conditioning that yields cross-prime transfer at a fixed width (where a monolithic learner reports ~0%, Lauter 2024), clearing all ten scorer tiers within budget with dynamic-L, the power-of-two-adjacent fragility characterization, and the machine-checked integer algorithm.

## How to scrutinize us

Every number has a committed receipt under `model/receipts/` (official eval per seed, determinism re-run, dynamic-L timing). The weight-collapse and bf16-margin checks are `scripts/verify_no_shortcut.py` and `scripts/bf16_margin_check.py`. The Lean proof is in `lean/` (`lake build`).
