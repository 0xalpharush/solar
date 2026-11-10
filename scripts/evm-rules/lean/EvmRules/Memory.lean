import EvmRules.ArithTactic

/-!
# EVM memory

Byte-addressed memory for the physical peephole lane, following the execution specifications'
`MSTORE` and `MLOAD`: a word is stored and loaded big-endian, its most significant byte at the
address. Addresses wrap modulo 2²⁵⁶, where the EVM would run out of gas; the 32 bytes of one
access are still distinct. Memory expansion and its gas are outside this model. The lemmas below
are the identities the peephole rules over one address need; they hold for any other contents.
-/

namespace EvmRules

/-- Memory contents: one byte per word address. -/
abbrev Memory := Word → BitVec 8

namespace Evm

/-- `MSTORE`: byte `j - a` of the stored word, counted from the most significant end, at every
address `j` from `a` through `a + 31`. -/
def mstore (m : Memory) (a v : Word) : Memory := fun j =>
  bif (j - a).ult 32 then (v >>> ((31 - (j - a)) * 8)).setWidth 8 else m j

/-- `MLOAD`: the 32 bytes from `a` on, the first one most significant. -/
def mload (m : Memory) (a : Word) : Word :=
  m (a + 0#256) ++
    m (a + 1#256) ++
    m (a + 2#256) ++
    m (a + 3#256) ++
    m (a + 4#256) ++
    m (a + 5#256) ++
    m (a + 6#256) ++
    m (a + 7#256) ++
    m (a + 8#256) ++
    m (a + 9#256) ++
    m (a + 10#256) ++
    m (a + 11#256) ++
    m (a + 12#256) ++
    m (a + 13#256) ++
    m (a + 14#256) ++
    m (a + 15#256) ++
    m (a + 16#256) ++
    m (a + 17#256) ++
    m (a + 18#256) ++
    m (a + 19#256) ++
    m (a + 20#256) ++
    m (a + 21#256) ++
    m (a + 22#256) ++
    m (a + 23#256) ++
    m (a + 24#256) ++
    m (a + 25#256) ++
    m (a + 26#256) ++
    m (a + 27#256) ++
    m (a + 28#256) ++
    m (a + 29#256) ++
    m (a + 30#256) ++
    m (a + 31#256)

end Evm

theorem add_sub_left (a k : Word) : a + k - a = k := by bv_omega

/-- A load after a store to the same address reads the stored word. -/
theorem mload_mstore (m : Memory) (a v : Word) : Evm.mload (Evm.mstore m a v) a = v := by
  simp only [Evm.mload, Evm.mstore, add_sub_left, BitVec.reduceULT, BitVec.reduceSub,
    BitVec.reduceMul, Bool.cond_true]
  bv_decide

/-- A store to an address hides an earlier store to it. -/
theorem mstore_mstore (m : Memory) (a v w : Word) :
    Evm.mstore (Evm.mstore m a v) a w = Evm.mstore m a w := by
  funext j
  simp only [Evm.mstore]
  cases (j - a).ult 32 <;> rfl

/-- Storing the word just loaded from an address leaves memory unchanged. -/
theorem mstore_mload (m : Memory) (a : Word) : Evm.mstore m a (Evm.mload m a) = m := by
  funext j
  simp only [Evm.mstore]
  cases h : (j - a).ult 32
  · rfl
  · obtain ⟨k, rfl⟩ : ∃ k, j = a + k := ⟨j - a, by bv_omega⟩
    simp only [add_sub_left, BitVec.ult, decide_eq_true_eq] at h ⊢
    have :
        k = 0#256 ∨
        k = 1#256 ∨
        k = 2#256 ∨
        k = 3#256 ∨
        k = 4#256 ∨
        k = 5#256 ∨
        k = 6#256 ∨
        k = 7#256 ∨
        k = 8#256 ∨
        k = 9#256 ∨
        k = 10#256 ∨
        k = 11#256 ∨
        k = 12#256 ∨
        k = 13#256 ∨
        k = 14#256 ∨
        k = 15#256 ∨
        k = 16#256 ∨
        k = 17#256 ∨
        k = 18#256 ∨
        k = 19#256 ∨
        k = 20#256 ∨
        k = 21#256 ∨
        k = 22#256 ∨
        k = 23#256 ∨
        k = 24#256 ∨
        k = 25#256 ∨
        k = 26#256 ∨
        k = 27#256 ∨
        k = 28#256 ∨
        k = 29#256 ∨
        k = 30#256 ∨
        k = 31#256 := by
      bv_omega
    rcases this with h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h |
        h | h | h | h | h | h | h | h | h | h | h | h | h | h | h | h <;>
      subst h <;> simp only [Evm.mload] <;> bv_decide

/-- Rewrite loads and stores of one address with the lemmas above, then prove what remains with
`evm_auto`, with its SAT limit in seconds. -/
macro "evm_memory " n:num : tactic =>
  `(tactic| first
    | (simp only [mload_mstore, mstore_mstore, mstore_mload]; done; evm_proved_by "evm_memory")
    | ((try simp only [mload_mstore, mstore_mstore, mstore_mload]); evm_auto $n))

end EvmRules
