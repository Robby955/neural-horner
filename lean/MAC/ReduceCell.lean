import MAC.Horner

/-!
# The discrete per-step cell equals the exact modular step

`Horner.lean` proves the loop is exact GIVEN the exact integer step `(2s+dx) % p`.
This file closes the per-step half with a DISCRETE cell that uses only `+`, `≤`, `-`
(no `%`, no division) — the bounded conditional subtraction a small recurrent cell
can realize, and the operation the trained cell matches exhaustively on small primes.

`reduce3 p y` subtracts `p` at most twice; on the range the Horner recurrence actually
visits (`y = 2s + d·x < 3p` for canonical `s,x < p`) it equals `y % p`. Hence the
discrete cell `cellStep` equals `step`, which composes with `modmul_eq`.
-/

/-- Bounded conditional reduction: subtract `p` at most twice. No `%`, no division. -/
def reduce3 (p y : Nat) : Nat :=
  if 2 * p ≤ y then y - 2 * p else if p ≤ y then y - p else y

private lemma sub_eq_mod (p y k : Nat) (hk : k * p ≤ y) (hlt : y - k * p < p) :
    y - k * p = y % p := by
  have heq : (y - k * p) + k * p = y := by omega
  calc y % p = ((y - k * p) + k * p) % p := by rw [heq]
    _ = (y - k * p) % p := by rw [Nat.add_mul_mod_self_right]
    _ = y - k * p := Nat.mod_eq_of_lt hlt

/-- For `y < 3p`, the two-subtraction reducer equals `y mod p`. -/
theorem reduce3_eq_mod {p y : Nat} (hp : 0 < p) (hy : y < 3 * p) :
    reduce3 p y = y % p := by
  unfold reduce3
  split_ifs with h2 h1
  · exact sub_eq_mod p y 2 (by omega) (by omega)
  · simpa using sub_eq_mod p y 1 (by omega) (by omega)
  · exact (Nat.mod_eq_of_lt (by omega)).symm

/-- The discrete per-step cell: shift `2s`, conditional add `d·x`, bounded reduce.
Built only from `+`, `≤`, `-`. -/
def cellStep (p x s : Nat) (d : Bool) : Nat :=
  reduce3 p (2 * s + cond d x 0)

/-- On canonical inputs (`s < p`, `x < p`) the discrete cell equals the exact
modular step. This is the per-step bridge: a `%`-free bounded-subtraction cell
provably computes `(2s + d·x) mod p`. -/
theorem cellStep_eq_step {p x s : Nat} (hp : 0 < p) (hs : s < p) (hx : x < p)
    (d : Bool) : cellStep p x s d = step p x s d := by
  unfold cellStep step
  apply reduce3_eq_mod hp
  cases d <;> simp <;> omega

/-- The discrete cell also stays canonical: its output is `< p`. -/
theorem cellStep_lt {p x s : Nat} (hp : 0 < p) (hs : s < p) (hx : x < p)
    (d : Bool) : cellStep p x s d < p := by
  rw [cellStep_eq_step hp hs hx d, step]
  exact Nat.mod_lt _ hp
