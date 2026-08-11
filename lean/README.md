This Lean package proves that the bit-serial Horner recurrence over the exact integer step `(2*s + d*x) mod p` computes natural-number modular reduction and multiplication exactly for any input bit depth. It formalizes the integer ALGORITHM (double-and-add mod p) only; no neural network, weights, or learned cell appear.

`directModmul_eq` and `directModmul_swap_eq` cover both orientations of the two-pass candidate: reduce one operand, then scan the original MSB-first bits of the other with the learned residue as the multiplier. The implementation chooses a swap-invariant canonical orientation, reducing the longer operand (lexicographically larger on equal lengths). Either branch removes the separate residue-multiplication pass at the exact-integer level.

The equality between the learned per-step cell and `step` is NOT proved here and is not proved anywhere yet: it is only empirical, exact on primes < 64 (exhaustive) but failing on a sparse set of large-prime cases. Treating the trained model as "provably exact" is therefore an overclaim until that cell-to-step bridge is established.

`agreesFrom` states the narrower condition actually checked by exact-prefix
trajectory certification: the candidate cell agrees with `step` at each exact
state reached by one fixed scan. `cellHorner_eq_horner_of_agreesFrom` proves by
induction that this local trajectory certificate implies equality of the full
deterministic scan. `cellDirectModmul_eq_of_agreesFrom` composes two such
certificates—reduction followed by the direct raw-bit scan—to prove the final
modular product for that input. This formalizes the receipt-to-rollout logic;
it does not certify any neural checkpoint or generalize beyond the certified
trajectory.
