import MAC.Horner

#eval bitsToNat [true, false, true]

example : bitsToNat [true, false, true] = 5 := by
  native_decide

#check horner_eq
#check reduce_eq
#check modmul_eq
#check horner_lt
#print axioms modmul_eq
