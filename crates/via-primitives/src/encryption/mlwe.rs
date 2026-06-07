//! MLWE ciphertext type — `.docs/primitives.md §2.1`.
//!
//! `MLWECiphertext<RANK, N, R>` is the rank-$m$ generalisation of RLWE:
//!
//! - `RANK = 1` corresponds to **RLWE** ($m = 1$, ring $R_{n, q}$).
//! - `RANK = N` with a degree-1 polynomial ring corresponds to **LWE** —
//!   each "mask polynomial" holds a single scalar. This case is realised
//!   by instantiating `R` at `N = 1`, but currently the polynomial backends
//!   require `N >= 2` (see `Poly::_CHECK`); the LWE path will be unlocked
//!   once Layer 5 needs it.
//!
//! Layer 2 here defines only the type, its constructors, and `Zeroize` /
//! `Debug` impls. Operations on MLWEs (the §2.2.5 polynomial-times-MLWE
//! mul) land in Phase 4 of the Layer-2 plan, and the full LWE-to-RLWE
//! cascade in Layer 5.

use core::fmt;

use zeroize::Zeroize;

use crate::algebra::ring::RingPoly;

/// Rank-$m$ MLWE ciphertext: $\bigl((A_0, \ldots, A_{m-1}), \, B\bigr)$
/// encrypting $M$ under the secret key vector
/// $\mathbf{S} = (S_0, \ldots, S_{m-1})$ via
/// $B = \langle \mathbf{A}, \mathbf{S} \rangle + e + M'$.
#[derive(Clone, Copy)]
pub struct MLWECiphertext<const RANK: usize, const N: usize, R: RingPoly<N>> {
    /// The $\mathrm{RANK}$ mask polynomials.
    pub masks: [R; RANK],
    /// The single body polynomial.
    pub body: R,
}

impl<const RANK: usize, const N: usize, R: RingPoly<N>> MLWECiphertext<RANK, N, R> {
    /// Construct an MLWE ciphertext from its mask vector and body.
    #[inline(always)]
    pub fn new(masks: [R; RANK], body: R) -> Self {
        Self { masks, body }
    }
}

impl<const RANK: usize, const N: usize, R: RingPoly<N>> Zeroize for MLWECiphertext<RANK, N, R> {
    fn zeroize(&mut self) {
        for m in &mut self.masks {
            m.zeroize();
        }
        self.body.zeroize();
    }
}

impl<const RANK: usize, const N: usize, R: RingPoly<N>> fmt::Debug for MLWECiphertext<RANK, N, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MLWECiphertext")
            .field("RANK", &RANK)
            .field("N", &N)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Polynomial × MLWE multiplication — §2.2.5.
//
// "Polynomial × MLWE multiplication. Same idea [as plaintext × RLWE],
// componentwise on every mask and on the body. Useful for symbolic
// manipulations like multiplying an MLWE by `X^k` during `Extr` (§5.5)."
// ---------------------------------------------------------------------------

impl<const RANK: usize, const N: usize, R: RingPoly<N>> core::ops::Mul<R>
    for MLWECiphertext<RANK, N, R>
{
    type Output = Self;

    /// $\text{ct} \cdot f = ((f \cdot A_0, \ldots, f \cdot A_{m-1}),
    /// \; f \cdot B)$ — componentwise polynomial multiplication on every
    /// mask and on the body. Decrypts (once the §5 cascade machinery
    /// exists) to $f \cdot M \bmod p$.
    ///
    /// As with [`RLWECiphertext`'s polynomial-times-ciphertext
    /// `Mul<R>`](crate::encryption::RLWECiphertext), `f` should have
    /// small infinity-norm.
    fn mul(self, f: R) -> Self {
        let masks = core::array::from_fn(|i| self.masks[i] * f);
        Self::new(masks, self.body * f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::rns::basis::ConstRnsBasis;
    use crate::algebra::zq::modulus::ConstModulus;

    type SingleR = Poly<4, ConstModulus<17>, Coefficient>;
    type RnsR = PolyRns<4, ConstRnsBasis<5, 11>, Coefficient>;

    #[test]
    fn mlwe_constructs_with_both_backends() {
        let m = ConstModulus::<17>;
        let b = ConstRnsBasis::<5, 11>;
        let single_zero = <SingleR as RingPoly<4>>::zero(m);
        let rns_zero = <RnsR as RingPoly<4>>::zero(b);
        // Rank-3 MLWE over the single-prime ring.
        let _single = MLWECiphertext::<3, 4, SingleR>::new([single_zero; 3], single_zero);
        // Rank-2 MLWE over the RNS ring.
        let _rns = MLWECiphertext::<2, 4, RnsR>::new([rns_zero; 2], rns_zero);
    }

    #[test]
    fn mlwe_zeroize_runs() {
        let m = ConstModulus::<17>;
        let z = <SingleR as RingPoly<4>>::zero(m);
        let mut ct = MLWECiphertext::<2, 4, SingleR>::new([z; 2], z);
        ct.zeroize();
    }

    /// Phase-4 §2.2.5 polynomial-times-MLWE: every mask becomes `f · mask_i`
    /// and the body becomes `f · body`. Tested structurally because Phase 2
    /// gives MLWE no decryption path; that lands when Layer 5's cascade
    /// arrives.
    #[test]
    fn mlwe_mul_polynomial_is_componentwise() {
        let m = ConstModulus::<17>;
        // Build a rank-3 MLWE with hand-chosen mask polynomials and body.
        let mask0: SingleR = Poly::new(m, [1, 2, 3, 4]);
        let mask1: SingleR = Poly::new(m, [5, 6, 7, 8]);
        let mask2: SingleR = Poly::new(m, [9, 10, 11, 12]);
        let body: SingleR = Poly::new(m, [13, 14, 15, 16]);
        let ct = MLWECiphertext::<3, 4, SingleR>::new([mask0, mask1, mask2], body);

        // f = X (single monomial — multiplication is rotation with
        // negacyclic wrap at i = 3 ⟶ 0).
        let f: SingleR = Poly::new(m, [0, 1, 0, 0]);
        let scaled = ct * f;

        let expected_mask0 = mask0 * f;
        let expected_mask1 = mask1 * f;
        let expected_mask2 = mask2 * f;
        let expected_body = body * f;
        assert_eq!(scaled.masks[0], expected_mask0);
        assert_eq!(scaled.masks[1], expected_mask1);
        assert_eq!(scaled.masks[2], expected_mask2);
        assert_eq!(scaled.body, expected_body);
    }
}
