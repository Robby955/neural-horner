import MAC.Horner

#eval bitsToNat [true, false, true]

example : bitsToNat [true, false, true] = 5 := by
  native_decide

#check horner_eq
#check reduce_eq
#check modmul_eq
#check directModmul_eq
#check directModmul_swap_eq
#check cellHorner_eq_horner_of_agreesFrom
#check cellDirectModmul_eq_of_agreesFrom
#check horner_lt
#print axioms modmul_eq
#print axioms directModmul_eq
#print axioms directModmul_swap_eq
#print axioms cellHorner_eq_horner_of_agreesFrom
#print axioms cellDirectModmul_eq_of_agreesFrom
