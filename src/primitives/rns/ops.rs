//! GPU-portable slice kernels for $\mathbb{Z}_Q$ under the two-prime RNS
//! decomposition.
//!
//! Each kernel takes an [`RnsBasis`] by value plus a pair of flat `&[u64]`
//! slices — one per RNS prime — and dispatches to the underlying
//! [`super::super::zq::ops`] kernel twice, once per component. The
//! struct-of-arrays (SoA) layout matches the per-prime contiguous storage that
//! the future polynomial-ring layer (§0.3) will use and that the per-prime NTT
//! (§0.4) needs for $O(n)$ pointwise multiplication.
//!
//! All kernels operate in canonical reduced form: every input must lie within
//! its prime range, and every output stays within its prime range.
//!
//! # In-place use
//!
//! Rust's borrow rules forbid aliasing `&mut [u64]` with `&[u64]`, so the
//! binary kernels cannot accept the same buffers as both `dst` and operand in
//! safe code. For the common `dst += rhs` pattern, copy each source slice into
//! the destination first or expose an in-place variant from the polynomial
//! ring layer (§0.3).
//!
//! # Length mismatch
//!
//! Every kernel panics if any pair of slices it touches differ in length.
//! This is a logic bug at the caller — the polynomial ring infrastructure
//! (§0.3) enforces same-length invariants.

use super::super::zq;
use super::basis::RnsBasis;

/// Coefficient-wise modular add in $\mathbb{Z}_Q$: each output position is
/// `(lhs[i] + rhs[i]) mod Q`, with the operation split per RNS prime.
///
/// # Panics
///
/// Any per-prime length mismatch panics — see module-level docs. Also
/// panics if the two per-prime destinations differ in length, which would
/// otherwise slip past the inner `zq::ops` triple-length checks.
#[inline]
pub fn add_slice<B: RnsBasis>(
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
        "add_slice: cross-prime length mismatch (dst0 vs dst1)",
    );
    zq::ops::add_slice(basis.m0(), dst0, lhs0, rhs0);
    zq::ops::add_slice(basis.m1(), dst1, lhs1, rhs1);
}

/// Coefficient-wise modular sub in $\mathbb{Z}_Q$.
#[inline]
pub fn sub_slice<B: RnsBasis>(
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
        "sub_slice: cross-prime length mismatch (dst0 vs dst1)",
    );
    zq::ops::sub_slice(basis.m0(), dst0, lhs0, rhs0);
    zq::ops::sub_slice(basis.m1(), dst1, lhs1, rhs1);
}

/// Coefficient-wise modular multiply (Hadamard / pointwise product) in
/// $\mathbb{Z}_Q$.
///
/// As at §0.1, this is the pointwise product used inside the NTT-evaluation
/// form of $R_{n, q}$ multiplication (§0.4); it is **not** the polynomial
/// (negacyclic) multiplication of $R_{n, q}$.
#[inline]
pub fn mul_slice<B: RnsBasis>(
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
        "mul_slice: cross-prime length mismatch (dst0 vs dst1)",
    );
    zq::ops::mul_slice(basis.m0(), dst0, lhs0, rhs0);
    zq::ops::mul_slice(basis.m1(), dst1, lhs1, rhs1);
}

/// Coefficient-wise modular negation in $\mathbb{Z}_Q$.
#[inline]
pub fn neg_slice<B: RnsBasis>(
    basis: B,
    dst0: &mut [u64],
    dst1: &mut [u64],
    src0: &[u64],
    src1: &[u64],
) {
    assert_eq!(
        dst0.len(),
        dst1.len(),
        "neg_slice: cross-prime length mismatch (dst0 vs dst1)",
    );
    zq::ops::neg_slice(basis.m0(), dst0, src0);
    zq::ops::neg_slice(basis.m1(), dst1, src1);
}

/// Scalar multiply: `dst[i] = src[i] * scalar mod Q` for each component.
///
/// Takes the scalar as a pre-decomposed pair `(scalar0, scalar1)` so that all
/// `u128` reduction work happens at the API boundary. Callers with a `u128`
/// scalar should decompose first via [`RnsBasis::decompose_u128`].
///
/// # Caller invariant
///
/// `scalar0 < basis.m0().q()` and `scalar1 < basis.m1().q()`. Debug-asserted.
#[inline]
pub fn scalar_mul_slice<B: RnsBasis>(
    basis: B,
    dst0: &mut [u64],
    dst1: &mut [u64],
    src0: &[u64],
    src1: &[u64],
    scalar0: u64,
    scalar1: u64,
) {
    assert_eq!(
        dst0.len(),
        dst1.len(),
        "scalar_mul_slice: cross-prime length mismatch (dst0 vs dst1)",
    );
    zq::ops::scalar_mul_slice(basis.m0(), dst0, src0, scalar0);
    zq::ops::scalar_mul_slice(basis.m1(), dst1, src1, scalar1);
}

/// Garner CRT reconstruction over slices: `dst[i] = reconstruct(src0[i], src1[i])`.
///
/// Used at output boundaries (decoding, decryption) where the composite
/// `u128` value matters; inside the homomorphic pipeline the RNS form is
/// preferred for componentwise arithmetic.
///
/// # Panics
///
/// Panics if `dst.len() != src0.len()` or `dst.len() != src1.len()`.
#[inline]
pub fn reconstruct_slice<B: RnsBasis>(basis: B, dst: &mut [u128], src0: &[u64], src1: &[u64]) {
    assert_eq!(
        dst.len(),
        src0.len(),
        "reconstruct_slice: dst/src0 length mismatch",
    );
    assert_eq!(
        dst.len(),
        src1.len(),
        "reconstruct_slice: dst/src1 length mismatch",
    );
    for ((d, &a0), &a1) in dst.iter_mut().zip(src0).zip(src1) {
        *d = basis.reconstruct(a0, a1);
    }
}

/// RNS decomposition over slices: `(dst0[i], dst1[i]) = decompose(src[i])`.
///
/// Used at input boundaries (e.g. lifting plaintext `u128` values into the
/// homomorphic pipeline).
///
/// # Panics
///
/// Panics if `dst0.len() != src.len()` or `dst1.len() != src.len()`.
#[inline]
pub fn decompose_slice<B: RnsBasis>(basis: B, dst0: &mut [u64], dst1: &mut [u64], src: &[u128]) {
    assert_eq!(
        dst0.len(),
        src.len(),
        "decompose_slice: dst0/src length mismatch",
    );
    assert_eq!(
        dst1.len(),
        src.len(),
        "decompose_slice: dst1/src length mismatch",
    );
    for ((d0, d1), &x) in dst0.iter_mut().zip(dst1.iter_mut()).zip(src) {
        let (v0, v1) = basis.decompose_u128(x);
        *d0 = v0;
        *d1 = v1;
    }
}

#[cfg(test)]
mod tests {
    use super::super::basis::{ConstRnsBasis, DynRnsBasis, paper};
    use super::super::element::RnsZq;
    use super::*;
    use crate::primitives::zq::modulus::DynModulus;

    type Z55 = ConstRnsBasis<5, 11>;

    fn decompose_each<B: RnsBasis>(b: B, xs: &[u128]) -> ([u64; 8], [u64; 8]) {
        let mut d0 = [0u64; 8];
        let mut d1 = [0u64; 8];
        for (i, &x) in xs.iter().enumerate() {
            let (a, c) = b.decompose_u128(x);
            d0[i] = a;
            d1[i] = c;
        }
        (d0, d1)
    }

    #[test]
    fn add_slice_matches_element_op() {
        let b = paper::ViaQ1Rns::default();
        let lhs: [u128; 8] = [0, 1, 42, 999, 1 << 30, 1 << 40, 1 << 50, b.big_q() - 1];
        let rhs: [u128; 8] = [
            0,
            b.big_q() - 1,
            1234,
            5678,
            (1 << 35) + 7,
            (1 << 45) + 11,
            (1 << 55) + 13,
            1,
        ];
        let (lhs0, lhs1) = decompose_each(b, &lhs);
        let (rhs0, rhs1) = decompose_each(b, &rhs);
        let mut dst0 = [0u64; 8];
        let mut dst1 = [0u64; 8];
        add_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
        for i in 0..8 {
            let want = (lhs[i] + rhs[i]) % b.big_q();
            let got = RnsZq::new(b, dst0[i], dst1[i]).to_u128();
            assert_eq!(got, want, "i={i}");
        }
    }

    #[test]
    fn sub_slice_matches_element_op() {
        let b = paper::ViaQ1Rns::default();
        let lhs: [u128; 4] = [10, b.big_q() - 1, 1 << 50, 42];
        let rhs: [u128; 4] = [3, 7, 99, b.big_q() - 1];
        let (lhs0, lhs1) = {
            let mut d0 = [0u64; 4];
            let mut d1 = [0u64; 4];
            for (i, &x) in lhs.iter().enumerate() {
                let (a, c) = b.decompose_u128(x);
                d0[i] = a;
                d1[i] = c;
            }
            (d0, d1)
        };
        let (rhs0, rhs1) = {
            let mut d0 = [0u64; 4];
            let mut d1 = [0u64; 4];
            for (i, &x) in rhs.iter().enumerate() {
                let (a, c) = b.decompose_u128(x);
                d0[i] = a;
                d1[i] = c;
            }
            (d0, d1)
        };
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 4];
        sub_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
        for i in 0..4 {
            let want = (lhs[i] + b.big_q() - rhs[i] % b.big_q()) % b.big_q();
            let got = RnsZq::new(b, dst0[i], dst1[i]).to_u128();
            assert_eq!(got, want, "i={i}");
        }
    }

    #[test]
    fn mul_slice_matches_dyn() {
        let c = paper::ViaQ1Rns::default();
        let d = DynRnsBasis::new(DynModulus::new(268369921), DynModulus::new(536608769));
        let lhs: [u128; 8] = [0, 1, 42, 12345, 1 << 30, 1 << 40, 1 << 50, c.big_q() - 1];
        let rhs: [u128; 8] = [
            0,
            c.big_q() - 1,
            99,
            1 << 20,
            7,
            (1 << 35) + 1,
            (1 << 45) + 1,
            1,
        ];
        let (lhs0, lhs1) = decompose_each(c, &lhs);
        let (rhs0, rhs1) = decompose_each(c, &rhs);
        let mut out_c0 = [0u64; 8];
        let mut out_c1 = [0u64; 8];
        let mut out_d0 = [0u64; 8];
        let mut out_d1 = [0u64; 8];
        mul_slice(c, &mut out_c0, &mut out_c1, &lhs0, &lhs1, &rhs0, &rhs1);
        mul_slice(d, &mut out_d0, &mut out_d1, &lhs0, &lhs1, &rhs0, &rhs1);
        assert_eq!(out_c0, out_d0);
        assert_eq!(out_c1, out_d1);
    }

    #[test]
    fn neg_slice_roundtrip() {
        let b = paper::ViaQ1Rns::default();
        let src_u128: [u128; 4] = [0, 1, 1234, b.big_q() - 1];
        let (src0, src1) = {
            let mut d0 = [0u64; 4];
            let mut d1 = [0u64; 4];
            for (i, &x) in src_u128.iter().enumerate() {
                let (a, c) = b.decompose_u128(x);
                d0[i] = a;
                d1[i] = c;
            }
            (d0, d1)
        };
        let mut neg0 = [0u64; 4];
        let mut neg1 = [0u64; 4];
        let mut back0 = [0u64; 4];
        let mut back1 = [0u64; 4];
        neg_slice(b, &mut neg0, &mut neg1, &src0, &src1);
        neg_slice(b, &mut back0, &mut back1, &neg0, &neg1);
        assert_eq!(back0, src0);
        assert_eq!(back1, src1);
    }

    #[test]
    fn scalar_mul_slice_tiny() {
        let b = Z55::default();
        // Elements at positions 0..4 are 1, 23, 30, 54.
        let src_u128: [u128; 4] = [1, 23, 30, 54];
        let mut src0 = [0u64; 4];
        let mut src1 = [0u64; 4];
        for (i, &x) in src_u128.iter().enumerate() {
            let (a, c) = b.decompose_u128(x);
            src0[i] = a;
            src1[i] = c;
        }
        // Scalar = 7 < q0, q1.
        let (s0, s1) = b.decompose_u128(7);
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 4];
        scalar_mul_slice(b, &mut dst0, &mut dst1, &src0, &src1, s0, s1);
        for i in 0..4 {
            let want = (src_u128[i] * 7) % 55;
            let got = RnsZq::new(b, dst0[i], dst1[i]).to_u128();
            assert_eq!(got, want, "i={i}");
        }
    }

    #[test]
    fn reconstruct_decompose_slice_roundtrip() {
        let b = paper::ViaQ1Rns::default();
        let xs: [u128; 6] = [
            0,
            1,
            1234567890,
            (1u128 << 50) + 7,
            b.big_q() - 1,
            b.big_q() / 2,
        ];
        let mut d0 = [0u64; 6];
        let mut d1 = [0u64; 6];
        decompose_slice(b, &mut d0, &mut d1, &xs);
        let mut back = [0u128; 6];
        reconstruct_slice(b, &mut back, &d0, &d1);
        assert_eq!(back, xs);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn add_slice_panics_on_length_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 4];
        let lhs0 = [0u64; 4];
        let lhs1 = [0u64; 4];
        let rhs0 = [0u64; 3]; // wrong length
        let rhs1 = [0u64; 4];
        add_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
    }

    /// Empty-slice kernels must no-op. Pairs with the zq-side empty-slice
    /// test; closes review item 20 (rns side).
    #[test]
    fn rns_kernels_empty_slice_noop() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0: [u64; 0] = [];
        let mut dst1: [u64; 0] = [];
        let lhs0: [u64; 0] = [];
        let lhs1: [u64; 0] = [];
        let rhs0: [u64; 0] = [];
        let rhs1: [u64; 0] = [];
        add_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
        sub_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
        mul_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
        neg_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1);
        scalar_mul_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, 0, 0);
        let mut dst_u128: [u128; 0] = [];
        reconstruct_slice(b, &mut dst_u128, &lhs0, &lhs1);
        let src_u128: [u128; 0] = [];
        decompose_slice(b, &mut dst0, &mut dst1, &src_u128);
    }

    /// Length-mismatch panics — one per remaining kernel locks the contract
    /// at the rns layer (the underlying zq kernels each verify their own
    /// triple; these tests confirm the RNS wrapper surfaces it). Closes
    /// review item 21 (rns side).
    #[test]
    #[should_panic(expected = "length mismatch")]
    fn sub_slice_panics_on_length_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 4];
        let lhs0 = [0u64; 4];
        let lhs1 = [0u64; 4];
        let rhs0 = [0u64; 3];
        let rhs1 = [0u64; 4];
        sub_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn mul_slice_panics_on_length_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 4];
        let lhs0 = [0u64; 4];
        let lhs1 = [0u64; 4];
        let rhs0 = [0u64; 4];
        let rhs1 = [0u64; 3];
        mul_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn neg_slice_panics_on_length_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 4];
        let src0 = [0u64; 3];
        let src1 = [0u64; 4];
        neg_slice(b, &mut dst0, &mut dst1, &src0, &src1);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn scalar_mul_slice_panics_on_length_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 4];
        let src0 = [0u64; 4];
        let src1 = [0u64; 3];
        scalar_mul_slice(b, &mut dst0, &mut dst1, &src0, &src1, 0, 0);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn reconstruct_slice_panics_on_length_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst = [0u128; 4];
        let src0 = [0u64; 3];
        let src1 = [0u64; 4];
        reconstruct_slice(b, &mut dst, &src0, &src1);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn decompose_slice_panics_on_length_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 3];
        let src = [0u128; 4];
        decompose_slice(b, &mut dst0, &mut dst1, &src);
    }

    /// Cross-prime length mismatch: each prime side is internally
    /// consistent (its triple matches), but the two destinations differ.
    /// Without the new top-of-kernel guard this would have slipped past the
    /// inner `zq::ops` triple-length checks. Closes review item 28.
    #[test]
    #[should_panic(expected = "cross-prime length mismatch")]
    fn add_slice_panics_on_cross_prime_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 3]; // different from dst0.len()
        let lhs0 = [0u64; 4];
        let lhs1 = [0u64; 3];
        let rhs0 = [0u64; 4];
        let rhs1 = [0u64; 3];
        add_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
    }

    #[test]
    #[should_panic(expected = "cross-prime length mismatch")]
    fn sub_slice_panics_on_cross_prime_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 3];
        let lhs0 = [0u64; 4];
        let lhs1 = [0u64; 3];
        let rhs0 = [0u64; 4];
        let rhs1 = [0u64; 3];
        sub_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
    }

    #[test]
    #[should_panic(expected = "cross-prime length mismatch")]
    fn mul_slice_panics_on_cross_prime_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 3];
        let lhs0 = [0u64; 4];
        let lhs1 = [0u64; 3];
        let rhs0 = [0u64; 4];
        let rhs1 = [0u64; 3];
        mul_slice(b, &mut dst0, &mut dst1, &lhs0, &lhs1, &rhs0, &rhs1);
    }

    #[test]
    #[should_panic(expected = "cross-prime length mismatch")]
    fn neg_slice_panics_on_cross_prime_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 3];
        let src0 = [0u64; 4];
        let src1 = [0u64; 3];
        neg_slice(b, &mut dst0, &mut dst1, &src0, &src1);
    }

    #[test]
    #[should_panic(expected = "cross-prime length mismatch")]
    fn scalar_mul_slice_panics_on_cross_prime_mismatch() {
        let b = paper::ViaQ1Rns::default();
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 3];
        let src0 = [0u64; 4];
        let src1 = [0u64; 3];
        scalar_mul_slice(b, &mut dst0, &mut dst1, &src0, &src1, 0, 0);
    }
}
