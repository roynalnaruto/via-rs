//! GPU-portable slice kernels for $R_{n, q} = \mathbb{Z}_q\lbrack X\rbrack / (X^n + 1)$.
//!
//! These are the **ring-specific** kernels — operations that depend on the
//! polynomial structure (negacyclic wrap, $X^k$ rotation) rather than the
//! purely coefficient-wise ones (`add`, `sub`, `neg`, scalar / pointwise
//! `mul`). The componentwise ops live at [`crate::algebra::zq::ops`] and
//! are reused directly by the [`Poly`] type — they are not re-exposed here.
//!
//! Every kernel takes a [`Modulus`] by value plus flat `&[u64]` slices.
//! This is the same shape a CUDA kernel sees: `Modulus` becomes a kernel
//! argument, the slices become device pointers, and the loop body is the
//! code we will later vectorise (AVX2 / AVX-512) and lower (CUDA / Metal).
//!
//! All kernels operate in canonical reduced form: every input coefficient
//! must lie in $[0, q)$, and every output coefficient is in $[0, q)$.
//!
//! # Length contract
//!
//! Both kernels expect `dst`, `lhs`, `rhs` (or `dst`, `src`) to share the
//! same length $n$, and treat that $n$ as the negacyclic ring degree
//! (i.e. work in $\mathbb{Z}_q\lbrack X\rbrack / (X^n + 1)$). $n$ must be a power of two
//! at the [`Poly`] layer; the kernels themselves do not validate it because
//! they are happy to operate at any length (e.g. recursive halving steps in
//! a future Karatsuba decomposition). Length mismatches panic at the top
//! of the kernel.
//!
//! # In-place use
//!
//! Rust's borrow rules forbid aliasing `&mut [u64]` with `&[u64]`, so the
//! binary kernels cannot accept the same buffer as both `dst` and one of
//! the operands in safe code. The [`Poly`] operator overloads materialise
//! a fresh destination buffer when needed and call into the kernel.
//!
//! [`Poly`]: super::element::Poly

use crate::algebra::zq::modulus::Modulus;

/// Negacyclic schoolbook multiplication in $R_{n, q}$:
/// $\mathrm{dst} = \mathrm{lhs} \cdot \mathrm{rhs} \bmod (X^n + 1, q)$.
///
/// Computes the polynomial product term-by-term and wraps every overflow
/// $i + j \ge n$ with a sign flip per the negacyclic identity
/// $X^n \equiv -1 \pmod{X^n + 1}$. The destination is overwritten — its
/// prior contents are ignored.
///
/// # Cost
///
/// $O(n^2)$ scalar [`Modulus::mul`] / [`Modulus::add`] / [`Modulus::sub`]
/// calls. For $n = 2048$ at a realistic modulus this is roughly four million
/// modular multiplications — about $10\,\text{ms}$ on a modern CPU with the
/// Barrett kernel. Use the NTT-mediated path via
/// [`super::element::Poly::into_eval`] when the call site is in a hot loop;
/// this schoolbook kernel is intended for setup paths (database encoding,
/// reference correctness checks, tests) where cost transparency matters.
///
/// # Constant-time
///
/// Constant-time over the operand *values* (every inner iteration runs the
/// same code regardless of coefficient content). The wrap branch
/// `if i + j < n` depends only on the public parameter $n$.
///
/// # Panics
///
/// Panics if `dst.len() != lhs.len()` or `dst.len() != rhs.len()`.
pub fn negacyclic_mul_slice<M: Modulus>(m: M, dst: &mut [u64], lhs: &[u64], rhs: &[u64]) {
    assert_eq!(
        dst.len(),
        lhs.len(),
        "negacyclic_mul_slice: dst/lhs length mismatch",
    );
    assert_eq!(
        dst.len(),
        rhs.len(),
        "negacyclic_mul_slice: dst/rhs length mismatch",
    );
    let n = dst.len();
    // Debug-only canonical-form sweep on the input operands. The
    // inner `m.mul(li, rhs[j])` call `debug_assert!`s the same
    // contract per pair, but a top-of-kernel sweep gives a clearer
    // panic message (which lane is out of range) before the O(n^2)
    // loop body runs. Zero release-build cost.
    #[cfg(debug_assertions)]
    for (i, (&l, &r)) in lhs.iter().zip(rhs.iter()).enumerate() {
        debug_assert!(
            l < m.q(),
            "negacyclic_mul_slice: lhs lane {i} = {l} not in [0, q={})",
            m.q(),
        );
        debug_assert!(
            r < m.q(),
            "negacyclic_mul_slice: rhs lane {i} = {r} not in [0, q={})",
            m.q(),
        );
    }
    // Zero the destination — schoolbook accumulates into it.
    for d in dst.iter_mut() {
        *d = 0;
    }
    for i in 0..n {
        let li = lhs[i];
        for j in 0..n {
            let prod = m.mul(li, rhs[j]);
            if i + j < n {
                dst[i + j] = m.add(dst[i + j], prod);
            } else {
                dst[i + j - n] = m.sub(dst[i + j - n], prod);
            }
        }
    }
}

/// Deterministic negacyclic rotation: $\mathrm{dst} = X^k \cdot \mathrm{src}$
/// in $R_{n, q}$.
///
/// Coefficient $\mathrm{src}\lbrack i\rbrack$ lands at $\mathrm{dst}\lbrack (i + k) \bmod n\rbrack$,
/// negated whenever an odd number of negacyclic wraps occurred. Concretely,
/// let $k_\mathrm{eff} = k \bmod 2n$, $k_\mathrm{red} = k_\mathrm{eff} \bmod n$,
/// and $\mathrm{neg} = (k_\mathrm{eff} \ge n)$. Then for each input position
/// $i$:
///
/// - The output position is $i + k_\mathrm{red}$, or $i + k_\mathrm{red} - n$
///   when the sum wraps past $n$.
/// - The sign flips iff *exactly one* of `(i + k_red) >= n` and `neg` holds.
///
/// This implements the `rotate` primitive, the building block of the controlled
/// rotation `CRot`.
///
/// # Constant-time
///
/// Constant-time over the operand *values*. **Not** constant-time over $k$:
/// the modular reduction `k % (2 * n)` and the resulting branches depend on
/// $k$. This is fine because at every protocol call site $k$ is a public
/// parameter — in the rotation it is a loop induction variable, and in `CRot` the
/// per-bit shift $2^i$ is also a loop induction variable (the *encrypted*
/// control bit feeds CMux, not the rotation index).
///
/// **Do not** call this kernel with a $k$ that depends on secret data
/// (a query index, a key-derived value, anything that should not leak
/// through timing). The encrypted-exponent path lives in `CRot`,
/// which composes this rotation with `CMux` over RGSW select bits;
/// callers wanting encrypted rotation should route through that
/// composite rather than passing a secret $k$ here.
///
/// # Panics
///
/// Panics if `dst.len() != src.len()`. Panics if `dst.len() == 0`.
pub fn rotate_slice<M: Modulus>(m: M, dst: &mut [u64], src: &[u64], k: usize) {
    assert_eq!(
        dst.len(),
        src.len(),
        "rotate_slice: dst/src length mismatch",
    );
    let n = dst.len();
    assert!(n > 0, "rotate_slice: zero-length input");
    // Debug-only canonical-form sweep — `m.neg(value)` further down
    // `debug_assert!`s the same contract per lane, but a top-of-
    // kernel sweep yields a clearer panic message (which lane was
    // out of range) before any wrapping logic runs. Zero release-
    // build cost.
    #[cfg(debug_assertions)]
    for (i, &v) in src.iter().enumerate() {
        debug_assert!(
            v < m.q(),
            "rotate_slice: src lane {i} = {v} not in [0, q={})",
            m.q(),
        );
    }
    let k_eff = k % (2 * n);
    let k_red = k_eff % n;
    let neg_global = k_eff >= n;
    for (i, &value) in src.iter().enumerate() {
        let wrapped = i + k_red >= n;
        let out_pos = if wrapped { i + k_red - n } else { i + k_red };
        // `wrapped` and `neg_global` are independent: XOR of the two
        // determines whether the value is negated on its way to `dst`.
        if wrapped ^ neg_global {
            dst[out_pos] = m.neg(value);
        } else {
            dst[out_pos] = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::zq::modulus::{ConstModulus, DynModulus};

    /// Reference negacyclic mul on `u128`s reduced modulo `q` at the end.
    /// The kernel under test should agree element-by-element.
    fn reference_mul(q: u64, lhs: &[u64], rhs: &[u64]) -> [u64; 8] {
        assert_eq!(lhs.len(), 8);
        assert_eq!(rhs.len(), 8);
        let n = 8usize;
        let mut acc = [0i128; 8];
        for i in 0..n {
            for j in 0..n {
                let p = (lhs[i] as i128) * (rhs[j] as i128);
                if i + j < n {
                    acc[i + j] += p;
                } else {
                    acc[i + j - n] -= p;
                }
            }
        }
        let mut out = [0u64; 8];
        let qi = q as i128;
        for k in 0..n {
            let r = acc[k].rem_euclid(qi);
            out[k] = r as u64;
        }
        out
    }

    #[test]
    fn negacyclic_mul_zero_inputs() {
        let m = ConstModulus::<17>;
        let lhs = [0u64; 4];
        let rhs = [0u64; 4];
        let mut dst = [1u64; 4]; // pre-filled to make sure we overwrite
        negacyclic_mul_slice(m, &mut dst, &lhs, &rhs);
        assert_eq!(dst, [0, 0, 0, 0]);
    }

    #[test]
    fn negacyclic_mul_identity_times_anything() {
        let m = ConstModulus::<17>;
        // identity = 1 + 0*X + 0*X^2 + 0*X^3
        let one = [1u64, 0, 0, 0];
        let rhs = [3u64, 5, 11, 7];
        let mut dst = [0u64; 4];
        negacyclic_mul_slice(m, &mut dst, &one, &rhs);
        assert_eq!(dst, rhs);
    }

    #[test]
    fn negacyclic_mul_x_times_x_is_x_squared() {
        let m = ConstModulus::<17>;
        // f = X, g = X (i.e. [0, 1, 0, 0])
        let x = [0u64, 1, 0, 0];
        let mut dst = [0u64; 4];
        negacyclic_mul_slice(m, &mut dst, &x, &x);
        // X * X = X^2 = [0, 0, 1, 0]
        assert_eq!(dst, [0, 0, 1, 0]);
    }

    #[test]
    fn negacyclic_mul_wrap() {
        let m = ConstModulus::<17>;
        // f = X^2, g = X^2 -> X^4 = -1 -> [16, 0, 0, 0] (since 16 == -1 mod 17)
        let f = [0u64, 0, 1, 0];
        let g = [0u64, 0, 1, 0];
        let mut dst = [0u64; 4];
        negacyclic_mul_slice(m, &mut dst, &f, &g);
        assert_eq!(dst, [16, 0, 0, 0]);
    }

    #[test]
    fn negacyclic_mul_specific_pair_matches_reference() {
        let q = 17u64;
        let m = ConstModulus::<17>;
        let lhs = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let rhs = [8u64, 7, 6, 5, 4, 3, 2, 1];
        let mut dst = [0u64; 8];
        negacyclic_mul_slice(m, &mut dst, &lhs, &rhs);
        let want = reference_mul(q, &lhs, &rhs);
        assert_eq!(dst, want);
    }

    #[test]
    fn negacyclic_mul_const_vs_dyn() {
        let q = 8380417u64; // VIA-C q_3
        let c = ConstModulus::<8380417>;
        let d = DynModulus::new(q);
        let lhs = [123456u64, 234567, 345678, 456789, 1, 0, 8380416, 100];
        let rhs = [987654u64, 876543, 765432, 654321, 8380416, 1, 0, 200];
        let mut dst_c = [0u64; 8];
        let mut dst_d = [0u64; 8];
        negacyclic_mul_slice(c, &mut dst_c, &lhs, &rhs);
        negacyclic_mul_slice(d, &mut dst_d, &lhs, &rhs);
        assert_eq!(dst_c, dst_d);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn negacyclic_mul_panics_on_lhs_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0u64; 4];
        let lhs = [0u64; 3];
        let rhs = [0u64; 4];
        negacyclic_mul_slice(m, &mut dst, &lhs, &rhs);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn negacyclic_mul_panics_on_rhs_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0u64; 4];
        let lhs = [0u64; 4];
        let rhs = [0u64; 5];
        negacyclic_mul_slice(m, &mut dst, &lhs, &rhs);
    }

    #[test]
    fn rotate_k_zero_is_copy() {
        let m = ConstModulus::<17>;
        let src = [3u64, 5, 11, 7];
        let mut dst = [0u64; 4];
        rotate_slice(m, &mut dst, &src, 0);
        assert_eq!(dst, src);
    }

    #[test]
    fn rotate_k_lt_n_shifts_with_no_neg() {
        let m = ConstModulus::<17>;
        // Pure left-multiplication by X: [a, b, c, d] -> [-d, a, b, c]
        // because X * (a + bX + cX^2 + dX^3) = aX + bX^2 + cX^3 + dX^4
        //                                    = -d + aX + bX^2 + cX^3.
        let src = [3u64, 5, 11, 7];
        let mut dst = [0u64; 4];
        rotate_slice(m, &mut dst, &src, 1);
        assert_eq!(dst, [m.neg(7), 3, 5, 11]);
    }

    #[test]
    fn rotate_k_eq_n_is_negation() {
        let m = ConstModulus::<17>;
        let src = [3u64, 5, 11, 7];
        let mut dst = [0u64; 4];
        rotate_slice(m, &mut dst, &src, 4);
        for i in 0..4 {
            assert_eq!(dst[i], m.neg(src[i]), "i={i}");
        }
    }

    #[test]
    fn rotate_k_eq_2n_is_identity() {
        let m = ConstModulus::<17>;
        let src = [3u64, 5, 11, 7];
        let mut dst = [0u64; 4];
        rotate_slice(m, &mut dst, &src, 8);
        assert_eq!(dst, src);
    }

    #[test]
    fn rotate_k_gt_2n_reduces_mod_2n() {
        let m = ConstModulus::<17>;
        let src = [3u64, 5, 11, 7];
        let mut dst_a = [0u64; 4];
        let mut dst_b = [0u64; 4];
        rotate_slice(m, &mut dst_a, &src, 1);
        rotate_slice(m, &mut dst_b, &src, 1 + 2 * 4);
        assert_eq!(dst_a, dst_b);
        // Sanity: also equals 1 + 4 * 2 * 100 = 801 mod 8 = 1.
        let mut dst_c = [0u64; 4];
        rotate_slice(m, &mut dst_c, &src, 1 + 8 * 100);
        assert_eq!(dst_a, dst_c);
    }

    #[test]
    fn rotate_matches_multiplication_by_x_pow_k() {
        let m = ConstModulus::<17>;
        let src = [3u64, 5, 11, 7];
        for k in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 13, 50] {
            let mut via_rotate = [0u64; 4];
            rotate_slice(m, &mut via_rotate, &src, k);
            // Build X^k as a polynomial and schoolbook-multiply.
            let mut x_k = [0u64; 4];
            let k_eff = k % 8;
            let k_red = k_eff % 4;
            let neg = k_eff >= 4;
            x_k[k_red] = if neg { m.neg(1) } else { 1 };
            let mut via_mul = [0u64; 4];
            negacyclic_mul_slice(m, &mut via_mul, &src, &x_k);
            assert_eq!(via_rotate, via_mul, "k={k}");
        }
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn rotate_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let mut dst = [0u64; 4];
        let src = [0u64; 3];
        rotate_slice(m, &mut dst, &src, 1);
    }

    /// `negacyclic_mul_slice` debug-asserts every input lane is in
    /// `[0, q)`. Locks the canonical-form contract for direct kernel
    /// callers (the `Poly` wrapper enforces it via type invariant; a
    /// future GPU / SIMD adapter that bypasses the wrapper would slip
    /// without this assert).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "negacyclic_mul_slice: lhs lane")]
    fn negacyclic_mul_slice_debug_asserts_canonical_lhs() {
        let m = ConstModulus::<17>;
        let lhs = [17u64, 0, 0, 0]; // lane 0 = q, out of [0, q)
        let rhs = [1u64, 0, 0, 0];
        let mut dst = [0u64; 4];
        negacyclic_mul_slice(m, &mut dst, &lhs, &rhs);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "negacyclic_mul_slice: rhs lane")]
    fn negacyclic_mul_slice_debug_asserts_canonical_rhs() {
        let m = ConstModulus::<17>;
        let lhs = [1u64, 0, 0, 0];
        let rhs = [0u64, 17, 0, 0]; // lane 1 = q
        let mut dst = [0u64; 4];
        negacyclic_mul_slice(m, &mut dst, &lhs, &rhs);
    }

    /// `rotate_slice` debug-asserts every src lane is in `[0, q)`.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "rotate_slice: src lane")]
    fn rotate_slice_debug_asserts_canonical_src() {
        let m = ConstModulus::<17>;
        let src = [0u64, 0, 17, 0]; // lane 2 = q
        let mut dst = [0u64; 4];
        rotate_slice(m, &mut dst, &src, 1);
    }

    #[test]
    fn rotate_at_paper_modulus_zero_input_unchanged() {
        let q = 8380417u64;
        let d = DynModulus::new(q);
        let src = [0u64; 8];
        let mut dst = [1u64; 8]; // pre-filled; must be fully overwritten
        rotate_slice(d, &mut dst, &src, 5);
        assert_eq!(dst, [0u64; 8]);
    }
}
