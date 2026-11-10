-- ADDMOD adds its summands at 512 bits, in either order.
rw [show Evm.addmod s0 s1 s2 = Evm.addmod s1 s0 s2 by simp only [Evm.addmod, BitVec.add_comm]]
simp only [Evm.or, Evm.xor, BitVec.xor_self]
rfl
