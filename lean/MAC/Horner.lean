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

theorem horner_lt {p x : Nat} {bits : List Bool} (hp : 0 < p) :
    horner p x bits < p := by
  rw [horner_eq (p := p) (x := x) (bits := bits) hp]
  exact Nat.mod_lt _ hp

example : bitsToNat [true, false, true] = 5 := by
  native_decide
