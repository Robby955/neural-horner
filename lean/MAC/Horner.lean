import Mathlib

def bitsToNat (bits : List Bool) : Nat :=
  bits.foldl (fun v b => 2 * v + cond b 1 0) 0

def step (p x s : Nat) (d : Bool) : Nat :=
  (2 * s + (cond d x 0)) % p

def horner (p x : Nat) (bits : List Bool) : Nat :=
  bits.foldl (step p x) 0

def bitsToNatFrom (v : Nat) (bits : List Bool) : Nat :=
  bits.foldl (fun v b => 2 * v + cond b 1 0) v

lemma bitsToNat_eq_bitsToNatFrom_zero (bits : List Bool) :
    bitsToNat bits = bitsToNatFrom 0 bits := by
  rfl

@[simp] lemma bitsToNatFrom_nil (v : Nat) :
    bitsToNatFrom v [] = v := by
  rfl

@[simp] lemma bitsToNatFrom_cons (v : Nat) (b : Bool) (bits : List Bool) :
    bitsToNatFrom v (b :: bits) =
      bitsToNatFrom (2 * v + cond b 1 0) bits := by
  rfl

lemma bitsToNatFrom_append_bit (v : Nat) (bits : List Bool) (b : Bool) :
    bitsToNatFrom v (bits ++ [b]) =
      2 * bitsToNatFrom v bits + cond b 1 0 := by
  induction bits generalizing v with
  | nil => rfl
  | cons c bits ih =>
      simp [bitsToNatFrom]

def toBits : Nat → List Bool
  | 0 => []
  | n + 1 => toBits ((n + 1) / 2) ++ [decide ((n + 1) % 2 = 1)]
termination_by n => n
decreasing_by
  omega

lemma mod_two_bit (n : Nat) :
    cond (decide (n % 2 = 1)) 1 0 = n % 2 := by
  have hlt : n % 2 < 2 := Nat.mod_lt n (by decide)
  by_cases h : n % 2 = 1
  · simp [h]
  · have hzero : n % 2 = 0 := by omega
    simp [hzero]

theorem bitsToNat_toBits (n : Nat) :
    bitsToNat (toBits n) = n := by
  rw [bitsToNat_eq_bitsToNatFrom_zero]
  induction n using Nat.strong_induction_on with
  | h n ih =>
      cases n with
      | zero => simp [toBits, bitsToNatFrom]
      | succ m =>
          have hlt : (m + 1) / 2 < m + 1 := by omega
          rw [toBits]
          calc
            bitsToNatFrom 0 (toBits ((m + 1) / 2) ++ [decide ((m + 1) % 2 = 1)])
                = 2 * bitsToNatFrom 0 (toBits ((m + 1) / 2)) +
                    cond (decide ((m + 1) % 2 = 1)) 1 0 := by
                      exact bitsToNatFrom_append_bit 0 (toBits ((m + 1) / 2))
                        (decide ((m + 1) % 2 = 1))
            _ = 2 * ((m + 1) / 2) + (m + 1) % 2 := by
                      rw [ih ((m + 1) / 2) hlt]
                      exact congrArg (fun t => 2 * ((m + 1) / 2) + t)
                        (mod_two_bit (m + 1))
            _ = m + 1 := by omega

lemma step_mod_eq (p x s v : Nat) (d : Bool) (hs : s = (v * x) % p) :
    step p x s d = ((2 * v + cond d 1 0) * x) % p := by
  subst s
  cases d <;>
    simp [step, Nat.add_mul, Nat.mul_assoc, Nat.mul_mod, Nat.add_mod]

lemma hornerFrom_eq (_hp : 0 < p) (x : Nat) :
    ∀ bits s v, s = (v * x) % p →
      bits.foldl (step p x) s = (bitsToNatFrom v bits * x) % p := by
  intro bits
  induction bits with
  | nil =>
      intro s v hs
      simp [bitsToNatFrom, hs]
  | cons d bits ih =>
      intro s v hs
      exact ih (step p x s d) (2 * v + cond d 1 0)
        (step_mod_eq p x s v d hs)

/--
For any bit-list length, the bit-serial Horner recurrence computes
`bitsToNat bits * x` modulo `p`.
-/
theorem horner_eq {p x : Nat} {bits : List Bool} (hp : 0 < p) :
    horner p x bits = (bitsToNat bits * x) % p := by
  simp [horner, bitsToNat_eq_bitsToNatFrom_zero,
    hornerFrom_eq (p := p) hp x bits 0 0 (by simp)
  ]

theorem reduce_eq {p a : Nat} (hp : 0 < p) :
    horner p 1 (toBits a) = a % p := by
  rw [horner_eq (p := p) (x := 1) (bits := toBits a) hp, bitsToNat_toBits]
  simp

theorem modmul_eq {p a b : Nat} (hp : 0 < p) :
    horner p (a % p) (toBits (b % p)) = (a * b) % p := by
  rw [horner_eq (p := p) (x := a % p) (bits := toBits (b % p)) hp,
    bitsToNat_toBits]
  calc
    ((b % p) * (a % p)) % p = (b * a) % p := by
      simp [Nat.mul_mod]
    _ = (a * b) % p := by rw [Nat.mul_comm]

/--
The direct two-pass schedule reduces `b`, then scans the original bits of `a`.
It computes the same modular product without a separate residue-multiplication pass.
-/
theorem directModmul_eq {p a b : Nat} (hp : 0 < p) :
    horner p (b % p) (toBits a) = (a * b) % p := by
  rw [horner_eq (p := p) (x := b % p) (bits := toBits a) hp,
    bitsToNat_toBits]
  simp [Nat.mul_mod]

/-- The operand-swapped direct schedule computes the identical product. -/
theorem directModmul_swap_eq {p a b : Nat} (hp : 0 < p) :
    horner p (a % p) (toBits b) = (a * b) % p := by
  rw [directModmul_eq (p := p) (a := b) (b := a) hp, Nat.mul_comm]

/-- A transition implementation has the same interface as the exact step. -/
abbrev TransitionCell := Nat → Nat → Nat → Bool → Nat

/-- Run an arbitrary transition cell in the same bit-serial Horner loop. -/
def cellHorner (cell : TransitionCell) (p x : Nat) (bits : List Bool) : Nat :=
  bits.foldl (cell p x) 0

/--
`agreesFrom cell p x s bits` records agreement only along the exact trajectory
starting at `s`. It does not require the cell to be correct on unreachable
states.
-/
def agreesFrom (cell : TransitionCell) (p x : Nat) : Nat → List Bool → Prop
  | _, [] => True
  | s, d :: bits =>
      cell p x s d = step p x s d ∧
        agreesFrom cell p x (step p x s d) bits

lemma foldl_cell_eq_foldl_step_of_agreesFrom
    (cell : TransitionCell) (p x s : Nat) (bits : List Bool)
    (hagree : agreesFrom cell p x s bits) :
    bits.foldl (cell p x) s = bits.foldl (step p x) s := by
  induction bits generalizing s with
  | nil => rfl
  | cons d bits ih =>
      rw [List.foldl_cons, List.foldl_cons]
      rw [hagree.1]
      exact ih (step p x s d) hagree.2

/--
Exact-prefix transition agreement certifies the complete deterministic scan.
This is the formal induction principle used by the trajectory runner.
-/
theorem cellHorner_eq_horner_of_agreesFrom
    (cell : TransitionCell) (p x : Nat) (bits : List Bool)
    (hagree : agreesFrom cell p x 0 bits) :
    cellHorner cell p x bits = horner p x bits := by
  exact foldl_cell_eq_foldl_step_of_agreesFrom cell p x 0 bits hagree

/-- Reduce `b` with the cell, then scan the original bits of `a`. -/
def cellDirectModmul (cell : TransitionCell) (p a b : Nat) : Nat :=
  let reducedB := cellHorner cell p 1 (toBits b)
  cellHorner cell p reducedB (toBits a)

/--
If the learned cell agrees with the exact step along the exact-prefix states of
both direct passes, its deterministic two-pass rollout computes modular
multiplication. The hypotheses are trajectory certificates, not a proof that a
particular neural checkpoint satisfies them universally.
-/
theorem cellDirectModmul_eq_of_agreesFrom
    (cell : TransitionCell) {p a b : Nat} (hp : 0 < p)
    (hreduce : agreesFrom cell p 1 0 (toBits b))
    (hscan : agreesFrom cell p (b % p) 0 (toBits a)) :
    cellDirectModmul cell p a b = (a * b) % p := by
  have hb : cellHorner cell p 1 (toBits b) = b % p := by
    rw [cellHorner_eq_horner_of_agreesFrom cell p 1 (toBits b) hreduce]
    exact reduce_eq hp
  rw [cellDirectModmul, hb]
  rw [cellHorner_eq_horner_of_agreesFrom cell p (b % p) (toBits a) hscan]
  exact directModmul_eq hp

theorem horner_lt {p x : Nat} {bits : List Bool} (hp : 0 < p) :
    horner p x bits < p := by
  rw [horner_eq (p := p) (x := x) (bits := bits) hp]
  exact Nat.mod_lt _ hp

example : bitsToNat [true, false, true] = 5 := by
  native_decide
