-- MULMOD multiplies its factors at 512 bits, in either order.
rw [show Evm.mulmod s0 s1 s2 = Evm.mulmod s1 s0 s2 by simp only [Evm.mulmod, BitVec.mul_comm]]
simp only [Evm.or, Evm.xor, BitVec.xor_self]
rfl
