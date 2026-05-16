//! GPU-portable SoA slice kernels for $R_{n, Q}$ under the two-prime RNS
//! decomposition.
//!
//! Mirrors [`crate::algebra::rns::ops`] at the polynomial-ring scale:
//! each kernel takes an [`RnsBasis`] by value plus a pair of flat per-prime
//! `&[u64]` slices, and dispatches to the underlying single-prime
//! [`super::ops`] kernel twice — once per RNS slot.
//!
//! Both kernels assert cross-prime length equality at the top so that any
//! caller passing mismatched per-prime buffer sizes panics with a clear
//! message before the inner kernels see it. The componentwise ops (`add`,
//! `sub`, `neg`, scalar / pointwise `mul`) reuse
//! [`crate::algebra::rns::ops`] directly and are not re-exposed here.

use crate::algebra::rns::basis::RnsBasis;

use super::ops;

/// Negacyclic schoolbook multiplication in $R_{n, Q}$ under the two-prime
/// RNS decomposition. Runs the single-prime
/// [`super::ops::negacyclic_mul_slice`] twice, once per slot.
///
/// # Panics
///
/// Panics on any cross-prime or per-prime length mismatch.
pub fn negacyclic_mul_slice<B: RnsBasis>(
    basis: B,
    dst0: &mut [u64],
    dst1: &mut [u64],
    lhs0: &[u64],
    lhs1: &[u64],
    rhs0: &[u64],
    rhs1: &[u64],
) {
    assert_eq!(
        dst0.len(),
        dst1.len(),
        "negacyclic_mul_slice: cross-prime length mismatch (dst0 vs dst1)",
    );
    ops::negacyclic_mul_slice(basis.m0(), dst0, lhs0, rhs0);
    ops::negacyclic_mul_slice(basis.m1(), dst1, lhs1, rhs1);
}

/// Deterministic negacyclic rotation $X^k$ in $R_{n, Q}$ under the two-prime
/// RNS decomposition. Runs the single-prime [`super::ops::rotate_slice`]
/// twice, once per slot.
///
/// # Panics
///
/// Panics on any cross-prime or per-prime length mismatch.
pub fn rotate_slice<B: RnsBasis>(
    basis: B,
    dst0: &mut [u64],
    dst1: &mut [u64],
    src0: &[u64],
    src1: &[u64],
    k: usize,
) {
    assert_eq!(
        dst0.len(),
        dst1.len(),
        "rotate_slice: cross-prime length mismatch (dst0 vs dst1)",
    );
    ops::rotate_slice(basis.m0(), dst0, src0, k);
    ops::rotate_slice(basis.m1(), dst1, src1, k);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::rns::basis::{ConstRnsBasis, paper};

    type Z55 = ConstRnsBasis<5, 11>;

    #[test]
    fn negacyclic_mul_zero_inputs_per_slot() {
        let b = Z55::default();
        let lhs0 = [0u64; 4];
        let lhs1 = [0u64; 4];
        let rhs0 = [0u64; 4];
        let rhs1 = [0u64; 4];
        let mut dst0 = [1u64; 4];
        let mut dst1 = [1u64; 4];
        negacyclic_mul_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
        assert_eq!(dst0, [0u64; 4]);
        assert_eq!(dst1, [0u64; 4]);
    }

    #[test]
    fn rotate_k_zero_is_copy_per_slot() {
        let b = paper::ViaQ1Rns::default();
        let src0 = [3u64, 5, 11, 7];
        let src1 = [9u64, 1, 8, 2];
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 4];
        rotate_slice(b, &mut dst0, &mut dst1, &src0, &src1, 0);
        assert_eq!(dst0, src0);
        assert_eq!(dst1, src1);
    }

    #[test]
    #[should_panic(expected = "cross-prime length mismatch")]
    fn negacyclic_mul_panics_on_cross_prime_mismatch() {
        let b = Z55::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 3]; // mismatched
        let lhs0 = [0u64; 4];
        let lhs1 = [0u64; 3];
        let rhs0 = [0u64; 4];
        let rhs1 = [0u64; 3];
        negacyclic_mul_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
    }

    #[test]
    #[should_panic(expected = "cross-prime length mismatch")]
    fn rotate_panics_on_cross_prime_mismatch() {
        let b = Z55::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 3]; // mismatched
        let src0 = [0u64; 4];
        let src1 = [0u64; 3];
        rotate_slice(b, &mut dst0, &mut dst1, &src0, &src1, 1);
    }
}
