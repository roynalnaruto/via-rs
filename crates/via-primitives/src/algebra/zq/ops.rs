//! GPU-portable slice kernels for $\mathbb{Z}_q$.
//!
//! Every kernel takes a [`Modulus`] by value plus flat `&[u64]` slices. This
//! is the same shape a CUDA kernel sees — `Modulus` becomes a kernel argument,
//! the slices become device pointers, and the loop becomes a thread-grid
//! launch. The CPU-side `for` body is intentionally the same code we will
//! later vectorise (AVX2 / AVX-512) and lower (CUDA / Metal).
//!
//! All kernels operate in canonical reduced form: every input coefficient must
//! lie in $[0, q)$, and every output coefficient is in $[0, q)$. For scalar
//! ops (`scalar_mul_slice`) the scalar must also be in $[0, q)$.
//!
//! # In-place use
//!
//! Rust's borrow rules forbid aliasing `&mut [u64]` with `&[u64]`, so the
//! binary kernels cannot accept the same buffer as both `dst` and one of
//! the operands in safe code. For the common `dst += rhs` / `dst *= rhs`
//! pattern, copy the source into the destination first or expose an
//! in-place variant from the polynomial ring layer.
//!
//! # Length mismatch
//!
//! Every binary kernel panics if the slice lengths differ. This is a logic
//! bug at the caller — the polynomial ring infrastructure enforces
//! same-length invariants.

use super::modulus::Modulus;

/// Coefficient-wise modular add: `dst[i] = lhs[i] + rhs[i] mod q`.
///
/// # Constant-time
///
/// Yes, over secret coefficient values; the iteration is a fixed-stride pass.
#[inline]
pub fn add_slice<M: Modulus>(modulus: M, dst: &mut [u64], lhs: &[u64], rhs: &[u64]) {
    assert_eq!(dst.len(), lhs.len(), "add_slice: dst/lhs length mismatch");
    assert_eq!(dst.len(), rhs.len(), "add_slice: dst/rhs length mismatch");
    for ((d, &l), &r) in dst.iter_mut().zip(lhs).zip(rhs) {
        *d = modulus.add(l, r);
    }
}

/// Coefficient-wise modular sub: `dst[i] = lhs[i] - rhs[i] mod q`.
#[inline]
pub fn sub_slice<M: Modulus>(modulus: M, dst: &mut [u64], lhs: &[u64], rhs: &[u64]) {
    assert_eq!(dst.len(), lhs.len(), "sub_slice: dst/lhs length mismatch");
    assert_eq!(dst.len(), rhs.len(), "sub_slice: dst/rhs length mismatch");
    for ((d, &l), &r) in dst.iter_mut().zip(lhs).zip(rhs) {
        *d = modulus.sub(l, r);
    }
}

/// Coefficient-wise modular multiply: `dst[i] = lhs[i] * rhs[i] mod q`.
///
/// This is the Hadamard (pointwise) product used inside the NTT-evaluation
/// form of $R_{n, q}$ multiplication; it is **not** the polynomial
/// (negacyclic) multiplication of $R_{n, q}$.
#[inline]
pub fn mul_slice<M: Modulus>(modulus: M, dst: &mut [u64], lhs: &[u64], rhs: &[u64]) {
    assert_eq!(dst.len(), lhs.len(), "mul_slice: dst/lhs length mismatch");
    assert_eq!(dst.len(), rhs.len(), "mul_slice: dst/rhs length mismatch");
    for ((d, &l), &r) in dst.iter_mut().zip(lhs).zip(rhs) {
        *d = modulus.mul(l, r);
    }
}

/// Coefficient-wise modular negation: `dst[i] = -src[i] mod q`.
#[inline]
pub fn neg_slice<M: Modulus>(modulus: M, dst: &mut [u64], src: &[u64]) {
    assert_eq!(dst.len(), src.len(), "neg_slice: dst/src length mismatch");
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = modulus.neg(s);
    }
}

/// Scalar multiply: `dst[i] = src[i] * scalar mod q`.
///
/// Used everywhere a polynomial is scaled by a single ring element — e.g.
/// the gadget product, where each gadget digit multiplies an
/// entire RLev sample.
///
/// # Caller invariant
///
/// `scalar` must be reduced into $[0, q)$. If the caller has an unreduced
/// value, run it through [`Modulus::reduce_u64`] first.
#[inline]
pub fn scalar_mul_slice<M: Modulus>(modulus: M, dst: &mut [u64], src: &[u64], scalar: u64) {
    assert_eq!(
        dst.len(),
        src.len(),
        "scalar_mul_slice: dst/src length mismatch"
    );
    debug_assert!(scalar < modulus.q(), "scalar_mul_slice: scalar must be < q");
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = modulus.mul(s, scalar);
    }
}

/// Centred-lift kernel: `dst[i] = to_centered_i64(src[i], q)`.
///
/// Per-lane wrapper of [`Modulus::to_centered_i64`]. Each input lane
/// must be in canonical $[0, q)$ form (debug-asserted by the trait
/// method). Output lanes lie in
/// $(-\lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$.
///
/// **Not constant-time** over input values. For secret-data inputs
/// (secret-key rekeying), use [`to_centered_i64_ct_slice`].
///
/// # Panics
///
/// Panics if `dst.len() != src.len()`.
#[inline]
pub fn to_centered_i64_slice<M: Modulus>(modulus: M, dst: &mut [i64], src: &[u64]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "to_centered_i64_slice: dst/src length mismatch",
    );
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = modulus.to_centered_i64(s);
    }
}

/// Constant-time centred-lift kernel. Same output as
/// [`to_centered_i64_slice`]; the difference is only timing.
///
/// Use this when the input values are **secrets** (e.g. coefficients
/// of a secret-key polynomial in secret-key rekeying). For RLWE-uniform
/// ciphertext coefficients or plaintext-being-decoded inputs, the
/// non-CT [`to_centered_i64_slice`] is faster and equally safe.
///
/// # Panics
///
/// Panics if `dst.len() != src.len()`.
#[inline]
pub fn to_centered_i64_ct_slice<M: Modulus>(modulus: M, dst: &mut [i64], src: &[u64]) {
    assert_eq!(
        dst.len(),
        src.len(),
        "to_centered_i64_ct_slice: dst/src length mismatch",
    );
    for (d, &s) in dst.iter_mut().zip(src) {
        *d = modulus.to_centered_i64_ct(s);
    }
}

#[cfg(test)]
mod tests {
    use super::super::modulus::{ConstModulus, DynModulus, PowerOfTwoModulus};
    use super::*;

    #[test]
    fn add_slice_matches_scalar() {
        let m = ConstModulus::<17>;
        let lhs = [3u64, 9, 16, 0];
        let rhs = [4u64, 9, 1, 16];
        let mut dst = [0u64; 4];
        add_slice(m, &mut dst, &lhs, &rhs);
        assert_eq!(dst, [7, 1, 0, 16]);
    }

    #[test]
    fn mul_slice_matches_dyn() {
        let c = ConstModulus::<8380417>;
        let d = DynModulus::new(8380417);
        let lhs = [12345u64, 67890, 8380416, 1];
        let rhs = [54321u64, 9876, 1, 8380416];
        let mut out_c = [0u64; 4];
        let mut out_d = [0u64; 4];
        mul_slice(c, &mut out_c, &lhs, &rhs);
        mul_slice(d, &mut out_d, &lhs, &rhs);
        assert_eq!(out_c, out_d);
    }

    #[test]
    fn scalar_mul_slice_pow2() {
        let m = PowerOfTwoModulus::<4>; // q = 16
        let src = [3u64, 5, 7, 15];
        let mut dst = [0u64; 4];
        scalar_mul_slice(m, &mut dst, &src, 3);
        assert_eq!(dst, [9, 15, 5, 13]); // (3*3, 5*3=15, 7*3=21 mod 16 = 5, 15*3=45 mod 16 = 13)
    }

    #[test]
    fn add_slice_via_copy_then_add() {
        // Safe Rust forbids `&mut` and `&` aliasing the same buffer, so the
        // common in-place pattern is: copy source into dst, then call the
        // kernel with `dst` as both destination and one operand-source.
        // We test that copy-then-add produces the expected result.
        let m = ConstModulus::<17>;
        let mut dst = [3u64, 9, 16, 0];
        let rhs = [4u64, 9, 1, 16];
        let lhs = dst; // bytewise copy of dst's current contents
        add_slice(m, &mut dst, &lhs, &rhs);
        assert_eq!(dst, [7, 1, 0, 16]);
    }

    #[test]
    fn neg_slice_roundtrip() {
        let m = ConstModulus::<8380417>;
        let src = [0u64, 1, 12345, 8380416];
        let mut neg = [0u64; 4];
        let mut back = [0u64; 4];
        neg_slice(m, &mut neg, &src);
        neg_slice(m, &mut back, &neg);
        assert_eq!(back, src);
    }

    /// Empty-slice kernels must no-op (no panic, no out-of-bounds). The
    /// polynomial-ring layer will sometimes assemble zero-length
    /// views during recursive decomposition, so this contract matters.
    /// Closes review item 20 (zq side).
    #[test]
    fn zq_kernels_empty_slice_noop() {
        let m = ConstModulus::<17>;
        let mut dst: [u64; 0] = [];
        let lhs: [u64; 0] = [];
        let rhs: [u64; 0] = [];
        add_slice(m, &mut dst, &lhs, &rhs);
        sub_slice(m, &mut dst, &lhs, &rhs);
        mul_slice(m, &mut dst, &lhs, &rhs);
        neg_slice(m, &mut dst, &lhs);
        scalar_mul_slice(m, &mut dst, &lhs, 0);
    }

    /// Length-mismatch panics — one assertion per kernel locks the contract.
    /// `add_slice` is already covered; this adds `sub`, `mul`, `neg`, and
    /// `scalar_mul`. Closes review item 21 (zq side).
    #[test]
    #[should_panic(expected = "sub_slice")]
    fn sub_slice_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0u64; 4];
        let lhs = [0u64; 3]; // wrong length.
        let rhs = [0u64; 4];
        sub_slice(m, &mut dst, &lhs, &rhs);
    }

    #[test]
    #[should_panic(expected = "mul_slice")]
    fn mul_slice_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0u64; 4];
        let lhs = [0u64; 4];
        let rhs = [0u64; 3]; // wrong length.
        mul_slice(m, &mut dst, &lhs, &rhs);
    }

    #[test]
    #[should_panic(expected = "neg_slice")]
    fn neg_slice_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0u64; 4];
        let src = [0u64; 3]; // wrong length.
        neg_slice(m, &mut dst, &src);
    }

    #[test]
    #[should_panic(expected = "scalar_mul_slice")]
    fn scalar_mul_slice_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0u64; 4];
        let src = [0u64; 3]; // wrong length.
        scalar_mul_slice(m, &mut dst, &src, 1);
    }

    /// `to_centered_i64_slice` agrees with per-lane `Modulus::to_centered_i64`.
    #[test]
    fn to_centered_i64_slice_matches_per_lane() {
        let m = ConstModulus::<17>;
        let src = [0u64, 1, 8, 9, 16];
        let mut dst = [0i64; 5];
        to_centered_i64_slice(m, &mut dst, &src);
        assert_eq!(dst, [0i64, 1, 8, -8, -1]);
    }

    /// CT slice matches non-CT slice across a sweep at a representative modulus.
    #[test]
    fn to_centered_i64_ct_slice_matches_non_ct_slice() {
        let m = DynModulus::new(8380417); // VIA-C q_3
        let src = [
            0u64, 1, 4_190_207, 4_190_208, 4_190_209, 8_380_415, 8_380_416,
        ];
        let mut non_ct = [0i64; 7];
        let mut ct = [0i64; 7];
        to_centered_i64_slice(m, &mut non_ct, &src);
        to_centered_i64_ct_slice(m, &mut ct, &src);
        assert_eq!(non_ct, ct);
    }

    #[test]
    #[should_panic(expected = "to_centered_i64_slice")]
    fn to_centered_i64_slice_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0i64; 4];
        let src = [0u64; 3];
        to_centered_i64_slice(m, &mut dst, &src);
    }

    #[test]
    #[should_panic(expected = "to_centered_i64_ct_slice")]
    fn to_centered_i64_ct_slice_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0i64; 4];
        let src = [0u64; 3];
        to_centered_i64_ct_slice(m, &mut dst, &src);
    }
}
