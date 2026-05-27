//! The [`RnsBasis`] trait and its two concrete implementations.
//!
//! [`RnsBasis`] is the §0.2 analogue of [`super::super::zq::modulus::Modulus`]:
//! a `Copy + Eq + Send + Sync + 'static` value type that the higher §0.x
//! primitives use to talk to a composite ring $\mathbb{Z}_Q$. The trait
//! deliberately does **not** extend `Modulus` — the §0.1 trait is `u64`-output
//! bound (see `reduce.rs` "Modulus range constraints"), but $Q = q^{(0)} \cdot
//! q^{(1)}$ exceeds $2^{64}$ for the realistic VIA-C / VIA-B parameter set, so
//! the two layers must remain siblings.
//!
//! ## Two implementations
//!
//! - [`ConstRnsBasis<Q0, Q1>`] — zero-sized, compile-time moduli, with the
//!   coprimality check and the Garner inverse precomputed at monomorphisation
//!   time. Use for the paper's parameter sets (see [`paper`]).
//! - [`DynRnsBasis`] — runtime basis. Panics on coprimality failure during
//!   construction. Use for tests, toy parameters, and JSON-driven test
//!   vectors.

use super::super::zq::modulus::{ConstModulus, DynModulus, Modulus};
use super::reduce::{gcd_u64, mod_inverse_u64};

/// Abstract behaviour shared by every two-prime RNS basis at §0.2.
///
/// All methods are constant-time over secret data — they may branch on the
/// basis values (`q^{(0)}`, `q^{(1)}`, and the precomputed inverse, all public
/// parameters) but never on the inputs $a_0, a_1, x$.
///
/// # Range invariants
///
/// Component values $a_0, a_1$ are always in $[0, q^{(0)})$ and $[0, q^{(1)})$
/// respectively. [`RnsBasis::decompose_u128`] accepts any `u128`;
/// [`RnsBasis::reconstruct`] returns a value in $[0, Q)$ where $Q = q^{(0)}
/// \cdot q^{(1)}$.
///
/// # Performance contract
///
/// Implementations should mark every method `#[inline(always)]` so that for
/// [`ConstRnsBasis`] the compiler folds both moduli and the Garner inverse
/// into immediate operands.
pub trait RnsBasis: Copy + Eq + Send + Sync + 'static {
    /// The first component modulus type (carries $q^{(0)}$).
    type M0: Modulus;
    /// The second component modulus type (carries $q^{(1)}$).
    type M1: Modulus;

    /// Returns the first component modulus instance.
    fn m0(self) -> Self::M0;
    /// Returns the second component modulus instance.
    fn m1(self) -> Self::M1;

    /// $(q^{(0)})^{-1} \bmod q^{(1)}$ — the Garner reconstruction constant.
    ///
    /// For [`ConstRnsBasis`] this is `const`-evaluated; for [`DynRnsBasis`]
    /// it is computed once in [`DynRnsBasis::new`].
    fn q0_inv_mod_q1(self) -> u64;

    /// $Q = q^{(0)} \cdot q^{(1)}$ as `u128`.
    ///
    /// Fits comfortably: every modulus at §0.1 satisfies $q < 2^{63}$ (see
    /// `zq/reduce.rs` "Modulus range constraints"), so the product is at most
    /// $2^{126}$; for paper parameters $Q \le 2^{75}$.
    #[inline(always)]
    fn big_q(self) -> u128 {
        u128::from(self.m0().q()) * u128::from(self.m1().q())
    }

    /// Decompose $x \in \mathbb{Z}$ (given as `u128`) into the residue pair
    /// $\bigl(x \bmod q^{(0)}, \; x \bmod q^{(1)}\bigr)$.
    ///
    /// The output components are always in canonical reduced form. Equivalent
    /// to the canonical $x \mapsto x \bmod Q$ map followed by the CRT
    /// isomorphism.
    #[inline(always)]
    fn decompose_u128(self, x: u128) -> (u64, u64) {
        (self.m0().reduce_u128(x), self.m1().reduce_u128(x))
    }

    /// Garner's 2-prime CRT: reconstruct $x \in [0, Q)$ from the residue pair
    /// $(a_0, a_1) \in [0, q^{(0)}) \times [0, q^{(1)})$.
    ///
    /// # Algorithm
    ///
    /// $$
    /// t \;=\; \bigl((a_1 - a_0) \cdot (q^{(0)})^{-1}\bigr) \bmod q^{(1)},
    /// \quad x \;=\; a_0 + q^{(0)} \cdot t.
    /// $$
    ///
    /// The subtraction is computed via [`Modulus::sub`] on the `m1()` modulus,
    /// which handles the unsigned wrap branchlessly. The lift then fits in
    /// `u128` by the range argument in [`RnsBasis::big_q`].
    ///
    /// The `a0_mod_q1 = m1.reduce_u64(a0)` step before the subtraction is
    /// necessary precisely when `q0 ≥ q1` — then `a0 ∈ [0, q0)` may exceed
    /// `q1` and must be reduced first so that [`Modulus::sub`]'s
    /// `debug_assert!(a < q)` precondition holds. The paper convention places
    /// the smaller prime as `q0`, making this a no-op for paper bases; the
    /// `reconstruct_with_q0_greater_than_q1` test (`basis.rs`) exercises the
    /// reverse-ordering path that [`DynRnsBasis::new`] also accepts.
    #[inline(always)]
    fn reconstruct(self, a0: u64, a1: u64) -> u128 {
        let q0 = self.m0().q();
        let m1 = self.m1();
        // a0 may exceed q^{(1)} (when q0 > q1); reduce first.
        let a0_mod_q1 = m1.reduce_u64(a0);
        let diff = m1.sub(a1, a0_mod_q1);
        let t = m1.mul(diff, self.q0_inv_mod_q1());
        u128::from(a0) + u128::from(q0) * u128::from(t)
    }
}

/// Compile-time RNS basis: both component primes are `const` generic
/// parameters.
///
/// Parameterised directly on the two `u64` consts (rather than on two
/// `Modulus` types) so that the `_CHECK` block can fail-stop the compile when
/// the parameters are inconsistent — there is no way to extract the underlying
/// $Q$ from a generic `M: Modulus` at const-evaluation time.
///
/// Internally uses [`ConstModulus<Q0>`] and [`ConstModulus<Q1>`] as the
/// component-modulus types. Zero-sized: instances exist only to carry the
/// generic parameters into trait dispatch.
///
/// # Compile-time invariants (enforced by the private `_CHECK` block)
///
/// - $Q_0 \ge 2$, $Q_1 \ge 2$.
/// - $Q_0 \ne Q_1$.
/// - $Q_0, Q_1 < 2^{63}$ (matches the §0.1 modulus range bound).
/// - $\gcd(Q_0, Q_1) = 1$ (coprimality, required for CRT).
/// - $(Q_0 \bmod Q_1)^{-1}$ exists modulo $Q_1$ (follows from coprimality;
///   asserted as a defence-in-depth against `mod_inverse_u64` sentinel-`0`s).
///
/// # Example
///
/// ```rust
/// use via_rs::algebra::rns::basis::{ConstRnsBasis, RnsBasis};
/// let b = ConstRnsBasis::<5, 11>;
/// assert_eq!(b.big_q(), 55);
/// assert_eq!(b.decompose_u128(42), (2, 9));    // 42 mod 5, 42 mod 11
/// assert_eq!(b.reconstruct(2, 9), 42);
/// ```
///
/// # Compile-time rejection of invalid bases
///
/// Non-coprime moduli fail to compile because every reconstruction path
/// routes through [`Self::Q0_INV_MOD_Q1`], which triggers the `_CHECK`
/// validation block at monomorphisation:
///
/// ```compile_fail
/// use via_rs::algebra::rns::basis::ConstRnsBasis;
/// // gcd(6, 10) = 2 — fails the coprimality assertion in `_CHECK`.
/// const _: u64 = ConstRnsBasis::<6, 10>::Q0_INV_MOD_Q1;
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct ConstRnsBasis<const Q0: u64, const Q1: u64>;

impl<const Q0: u64, const Q1: u64> ConstRnsBasis<Q0, Q1> {
    /// Compile-time validation block — see the struct docs for the asserted
    /// invariants. Touched by both [`Self::BIG_Q`] and [`Self::Q0_INV_MOD_Q1`]
    /// so that every reconstruction path (which routes through
    /// `Q0_INV_MOD_Q1`) forces these checks at monomorphisation. Without the
    /// `Q0_INV_MOD_Q1` trigger, a user with non-coprime `(Q0, Q1)` would
    /// silently get `Q0_INV_MOD_Q1 = 0` and produce wrong reconstructions —
    /// `BIG_Q` alone is not enough because the default
    /// [`RnsBasis::big_q`] method bypasses it.
    const _CHECK: () = {
        assert!(Q0 >= 2, "ConstRnsBasis: Q0 >= 2");
        assert!(Q1 >= 2, "ConstRnsBasis: Q1 >= 2");
        assert!(Q0 != Q1, "ConstRnsBasis: Q0 != Q1");
        assert!(Q0 < (1u64 << 63), "ConstRnsBasis: Q0 < 2^63");
        assert!(Q1 < (1u64 << 63), "ConstRnsBasis: Q1 < 2^63");
        assert!(
            gcd_u64(Q0, Q1) == 1,
            "ConstRnsBasis: Q0, Q1 must be coprime"
        );
        assert!(
            mod_inverse_u64(Q0 % Q1, Q1) != 0,
            "ConstRnsBasis: inverse of (Q0 mod Q1) modulo Q1 must exist",
        );
    };

    /// $Q = Q_0 \cdot Q_1$ as a compile-time constant.
    pub const BIG_Q: u128 = {
        let () = Self::_CHECK;
        (Q0 as u128) * (Q1 as u128)
    };

    /// Precomputed Garner inverse $(Q_0 \bmod Q_1)^{-1} \bmod Q_1$, evaluated
    /// at compile time. Const-folds into immediate operands at every
    /// reconstruction call site.
    ///
    /// Touches `Self::_CHECK` before computing the inverse so that any
    /// caller reaching this constant (the trait method
    /// [`RnsBasis::q0_inv_mod_q1`] is the canonical path) fires the
    /// coprimality / range invariants at monomorphisation rather than
    /// silently producing the `0` sentinel.
    pub const Q0_INV_MOD_Q1: u64 = {
        let () = Self::_CHECK;
        mod_inverse_u64(Q0 % Q1, Q1)
    };

    /// First component modulus value (also reachable via [`RnsBasis::m0`]).
    pub const Q0: u64 = Q0;
    /// Second component modulus value (also reachable via [`RnsBasis::m1`]).
    pub const Q1: u64 = Q1;
}

impl<const Q0: u64, const Q1: u64> RnsBasis for ConstRnsBasis<Q0, Q1> {
    type M0 = ConstModulus<Q0>;
    type M1 = ConstModulus<Q1>;

    #[inline(always)]
    fn m0(self) -> Self::M0 {
        ConstModulus::<Q0>
    }

    #[inline(always)]
    fn m1(self) -> Self::M1 {
        ConstModulus::<Q1>
    }

    #[inline(always)]
    fn q0_inv_mod_q1(self) -> u64 {
        Self::Q0_INV_MOD_Q1
    }
}

/// Runtime RNS basis: carries the two [`DynModulus`] components alongside the
/// precomputed Garner inverse.
///
/// Use this when the basis is only known at runtime — driven by parsed JSON,
/// paper-quoted toy parameters, or fuzz-target inputs. For paper-pinned
/// production paths, prefer [`ConstRnsBasis`] (zero overhead).
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct DynRnsBasis {
    m0: DynModulus,
    m1: DynModulus,
    q0_inv_mod_q1: u64,
}

impl DynRnsBasis {
    /// Build a runtime RNS basis, validating coprimality and precomputing the
    /// Garner inverse.
    ///
    /// # Panics
    ///
    /// - `m0.q() == m1.q()` (the two primes must be distinct).
    /// - `m0.q() < 2` or `m1.q() < 2`.
    /// - $\gcd(q^{(0)}, q^{(1)}) \ne 1$ (CRT requires coprimality).
    /// - $(q^{(0)} \bmod q^{(1)})^{-1}$ does not exist (should be unreachable
    ///   under coprimality; asserted as defence in depth).
    #[inline]
    pub fn new(m0: DynModulus, m1: DynModulus) -> Self {
        let q0 = m0.q();
        let q1 = m1.q();
        assert!(q0 >= 2, "DynRnsBasis::new: q0 >= 2");
        assert!(q1 >= 2, "DynRnsBasis::new: q1 >= 2");
        assert!(q0 != q1, "DynRnsBasis::new: q0 != q1");
        assert_eq!(
            gcd_u64(q0, q1),
            1,
            "DynRnsBasis::new: q0 and q1 must be coprime",
        );
        let inv = mod_inverse_u64(q0 % q1, q1);
        assert!(
            inv != 0,
            "DynRnsBasis::new: inverse of (q0 mod q1) modulo q1 must exist",
        );
        Self {
            m0,
            m1,
            q0_inv_mod_q1: inv,
        }
    }
}

impl RnsBasis for DynRnsBasis {
    type M0 = DynModulus;
    type M1 = DynModulus;

    #[inline(always)]
    fn m0(self) -> Self::M0 {
        self.m0
    }

    #[inline(always)]
    fn m1(self) -> Self::M1 {
        self.m1
    }

    #[inline(always)]
    fn q0_inv_mod_q1(self) -> u64 {
        self.q0_inv_mod_q1
    }
}

/// Compile-time markers for every two-prime RNS basis that appears in
/// `.docs/primitives.md` Appendix A.
///
/// Only $q_1$ is composite in any realistic VIA / VIA-C / VIA-B parameter set;
/// $q_2$, $q_3$, $q_4$, and $p$ are single-prime / power-of-two and live
/// entirely at §0.1.
pub mod paper {
    use super::ConstRnsBasis;

    /// VIA $q_1 = 268\,369\,921 \cdot 536\,608\,769 \approx 2^{57}$ — the
    /// two-prime RNS basis used by every VIA ciphertext that lives at $q_1$.
    /// See `.docs/primitives.md` §A.1.
    pub type ViaQ1Rns = ConstRnsBasis<268369921, 536608769>;

    /// VIA-C / VIA-B $q_1 = 137\,438\,822\,401 \cdot 274\,810\,798\,081
    /// \approx 2^{75}$ — the only composite modulus in either parameter set.
    /// Used by the LWE-to-RLWE cascade outputs (§5) and the query-encryption
    /// layer (§6.1). See `.docs/primitives.md` §A.1.
    pub type ViaCQ1Rns = ConstRnsBasis<137438822401, 274810798081>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_basis_via_q1_values() {
        let b = paper::ViaQ1Rns::default();
        assert_eq!(b.m0().q(), 268369921);
        assert_eq!(b.m1().q(), 536608769);
        assert_eq!(b.big_q(), 268369921u128 * 536608769u128);
    }

    #[test]
    fn const_basis_via_c_q1_values() {
        let b = paper::ViaCQ1Rns::default();
        assert_eq!(b.m0().q(), 137438822401);
        assert_eq!(b.m1().q(), 274810798081);
        assert_eq!(b.big_q(), 137438822401u128 * 274810798081u128);
    }

    #[test]
    fn const_basis_q0_inv_correct() {
        // VIA: verify (Q0 * Q0_INV_MOD_Q1) mod Q1 == 1.
        let q0 = 268369921u128;
        let q1 = 536608769u128;
        let inv = u128::from(paper::ViaQ1Rns::Q0_INV_MOD_Q1);
        assert_eq!((q0 * inv) % q1, 1);

        // VIA-C: same check at 38-bit primes.
        let q0 = 137438822401u128;
        let q1 = 274810798081u128;
        let inv = u128::from(paper::ViaCQ1Rns::Q0_INV_MOD_Q1);
        assert_eq!((q0 * inv) % q1, 1);
    }

    #[test]
    fn decompose_reconstruct_roundtrip_via_q1() {
        let b = paper::ViaQ1Rns::default();
        let q = b.big_q();
        for x in [
            0u128,
            1,
            42,
            12345678901234567890u128,
            q - 1,
            q / 2,
            q / 3 + 7,
        ] {
            let xr = x % q;
            let (a0, a1) = b.decompose_u128(xr);
            assert!(a0 < b.m0().q() && a1 < b.m1().q());
            assert_eq!(b.reconstruct(a0, a1), xr, "x={xr}");
        }
    }

    #[test]
    fn decompose_reconstruct_roundtrip_via_c_q1() {
        let b = paper::ViaCQ1Rns::default();
        let q = b.big_q();
        for x in [
            0u128,
            1,
            q - 1,
            q / 2,
            q / 7 + 11,
            (1u128 << 70) % q,
            (1u128 << 60) % q,
        ] {
            let (a0, a1) = b.decompose_u128(x);
            assert_eq!(b.reconstruct(a0, a1), x, "x={x}");
        }
    }

    #[test]
    fn decompose_reconstruct_roundtrip_tiny() {
        // Every x in [0, 55) for the toy basis Z_{5 * 11}.
        type Z55 = ConstRnsBasis<5, 11>;
        let b = Z55::default();
        assert_eq!(b.big_q(), 55);
        for x in 0u128..55 {
            let (a0, a1) = b.decompose_u128(x);
            assert!(a0 < 5 && a1 < 11);
            assert_eq!(b.reconstruct(a0, a1), x, "x={x}");
        }
    }

    #[test]
    fn dyn_matches_const_for_via_q1() {
        let c = paper::ViaQ1Rns::default();
        let d = DynRnsBasis::new(DynModulus::new(268369921), DynModulus::new(536608769));
        assert_eq!(c.m0().q(), d.m0().q());
        assert_eq!(c.m1().q(), d.m1().q());
        assert_eq!(c.q0_inv_mod_q1(), d.q0_inv_mod_q1());
        assert_eq!(c.big_q(), d.big_q());
        // Reconstruction agrees.
        for x in [0u128, 1, 42, 999_999, c.big_q() - 1] {
            let (a0, a1) = c.decompose_u128(x);
            let (b0, b1) = d.decompose_u128(x);
            assert_eq!((a0, a1), (b0, b1));
            assert_eq!(c.reconstruct(a0, a1), d.reconstruct(b0, b1));
        }
    }

    #[test]
    fn dyn_matches_const_for_via_c_q1() {
        let c = paper::ViaCQ1Rns::default();
        let d = DynRnsBasis::new(DynModulus::new(137438822401), DynModulus::new(274810798081));
        assert_eq!(c.q0_inv_mod_q1(), d.q0_inv_mod_q1());
        assert_eq!(c.big_q(), d.big_q());
    }

    #[test]
    #[should_panic(expected = "q0 != q1")]
    fn dyn_basis_new_panics_on_equal_primes() {
        DynRnsBasis::new(DynModulus::new(17), DynModulus::new(17));
    }

    #[test]
    #[should_panic(expected = "coprime")]
    fn dyn_basis_new_panics_on_non_coprime() {
        // gcd(6, 10) = 2.
        DynRnsBasis::new(DynModulus::new(6), DynModulus::new(10));
    }

    #[test]
    fn const_basis_default_is_zero_sized() {
        // ConstRnsBasis<Q0, Q1> must be zero-sized so it can be passed by
        // value through every kernel without overhead.
        assert_eq!(core::mem::size_of::<paper::ViaQ1Rns>(), 0);
        assert_eq!(core::mem::size_of::<paper::ViaCQ1Rns>(), 0);
    }

    /// Reconstruction with `q0 > q1` exercises the `a0_mod_q1 =
    /// m1.reduce_u64(a0)` step in Garner — without it, `a1 - a0` would be
    /// computed on a value outside `[0, q1)`. Paper bases happen to satisfy
    /// `q0 < q1` so this path was previously un-exercised. Closes review
    /// item 19.
    #[test]
    fn reconstruct_with_q0_greater_than_q1() {
        // q0 = 11, q1 = 5. Reversed relative to the toy Z_55 basis used
        // elsewhere.
        let b = DynRnsBasis::new(DynModulus::new(11), DynModulus::new(5));
        let q = b.big_q();
        assert_eq!(q, 55);
        for x in 0u128..55 {
            let (a0, a1) = b.decompose_u128(x);
            assert!(a0 < 11 && a1 < 5);
            assert_eq!(b.reconstruct(a0, a1), x, "x = {x}");
        }
    }
}
