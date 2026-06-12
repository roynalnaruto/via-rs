//! GPU-portable coefficient-level kernels for secret-key rekeying.
//!
//! Re-interpret a small-coefficient secret key at a new modulus: given the
//! **already-centred** coefficient vector of $S$ (in $(-q_\text{src}/2,
//! q_\text{src}/2]$), reduce each coefficient mod $q_\text{dst}$. This is
//! well-defined exactly when $\|S\|_\infty < q_\text{dst}/2$, which holds for
//! ternary / bounded-uniform / narrow-Gaussian keys.
//!
//! # Constant-time
//!
//! These kernels consume the **centred lift** of a secret key, so the centring
//! itself (the data-dependent branch) must already have been done in constant
//! time by the caller via [`crate::algebra::ring::abstraction::RingPoly::to_centered_coeffs_ct`].
//! The reduction step here ([`Modulus::reduce_i64`]) is branchless over the
//! coefficient value (it selects via `subtle`), so the whole rekey path stays
//! constant-time over the secret. Do **not** feed a variable-time centred lift
//! into these kernels.
//!
//! # Constant-time centring
//!
//! A variable-time centring branch like `c if c <= q//2 else c - q` would leak
//! the key through timing; the Rust path uses the constant-time
//! [`crate::algebra::ring::abstraction::RingPoly::to_centered_coeffs_ct`]
//! upstream. Both produce the identical centred integer vector, so the rekeyed
//! key is byte-identical — only the timing behaviour differs.

use crate::algebra::zq::modulus::Modulus;

/// Reduce a slice of centred `i64` coefficients into a canonical `u64`
/// destination slice modulo `dst_mod`. The `i64` path is used when the source
/// secret key lives at a single-prime modulus (centred lift is `i64`).
///
/// # Panics
///
/// If `dst.len() != centered_i64.len()`.
///
/// # Constant-time
///
/// Branchless over each coefficient via [`Modulus::reduce_i64`]; see the
/// module-doc for the upstream centring contract.
#[inline]
pub fn rekey_centered_i64_to_modulus_slice<M: Modulus>(
    dst_mod: M,
    dst: &mut [u64],
    centered_i64: &[i64],
) {
    assert_eq!(
        dst.len(),
        centered_i64.len(),
        "rekey_centered_i64_to_modulus_slice: dst/src length mismatch"
    );
    for (d, &c) in dst.iter_mut().zip(centered_i64) {
        *d = dst_mod.reduce_i64(c);
    }
}

/// Reduce a slice of centred `i128` coefficients into a canonical `u64`
/// destination slice modulo `dst_mod`. The `i128` path is used when the source
/// secret key lives at the RNS-composite $q_1$ (centred lift is `i128`); each
/// coefficient is still small (a ternary / narrow key), so it narrows to `i64`
/// before reduction.
///
/// # Panics
///
/// If `dst.len() != centered_i128.len()`. In debug builds, also if any
/// coefficient does not fit in `i64` — a paper-spec secret key is always far
/// below that bound, so an oversize value signals a caller bug.
///
/// # Constant-time
///
/// Branchless over each coefficient via [`Modulus::reduce_i64`] (after the
/// debug-only range check, which is compiled out in release). The centring
/// upstream must be constant-time — see the module-doc.
#[inline]
pub fn rekey_centered_i128_to_modulus_slice<M: Modulus>(
    dst_mod: M,
    dst: &mut [u64],
    centered_i128: &[i128],
) {
    assert_eq!(
        dst.len(),
        centered_i128.len(),
        "rekey_centered_i128_to_modulus_slice: dst/src length mismatch"
    );
    for (d, &c) in dst.iter_mut().zip(centered_i128) {
        debug_assert!(
            c >= i64::MIN as i128 && c <= i64::MAX as i128,
            "rekey_centered_i128_to_modulus_slice: coefficient {c} does not fit in i64"
        );
        *d = dst_mod.reduce_i64(c as i64);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::zq::modulus::ConstModulus;

    #[test]
    fn rekey_i64_zero_maps_to_zero() {
        let mut dst = [0u64; 1];
        rekey_centered_i64_to_modulus_slice(ConstModulus::<97>, &mut dst, &[0]);
        assert_eq!(dst, [0]);
    }

    #[test]
    fn rekey_i64_positive_stays_positive() {
        let mut dst = [0u64; 3];
        rekey_centered_i64_to_modulus_slice(ConstModulus::<97>, &mut dst, &[1, 5, 42]);
        assert_eq!(dst, [1, 5, 42]);
    }

    #[test]
    fn rekey_i64_negative_wraps_correctly() {
        // -1 mod 97 = 96, -2 mod 97 = 95.
        let mut dst = [0u64; 2];
        rekey_centered_i64_to_modulus_slice(ConstModulus::<97>, &mut dst, &[-1, -2]);
        assert_eq!(dst, [96, 95]);
    }

    #[test]
    #[should_panic(expected = "dst/src length mismatch")]
    fn rekey_i64_panics_on_length_mismatch() {
        let mut dst = [0u64; 3];
        rekey_centered_i64_to_modulus_slice(ConstModulus::<97>, &mut dst, &[1, 2]);
    }

    #[test]
    fn rekey_i64_matches_reduce_i64_per_lane() {
        let m = ConstModulus::<97>;
        let inputs = [-48i64, -1, 0, 1, 48];
        let mut dst = [0u64; 5];
        rekey_centered_i64_to_modulus_slice(m, &mut dst, &inputs);
        for (got, &c) in dst.iter().zip(inputs.iter()) {
            assert_eq!(*got, m.reduce_i64(c));
        }
    }

    #[test]
    fn rekey_i128_ternary_matches_i64_path() {
        let m = ConstModulus::<97>;
        let ternary_i64 = [-1i64, 0, 1, -1, 1];
        let ternary_i128 = [-1i128, 0, 1, -1, 1];
        let mut via_i64 = [0u64; 5];
        let mut via_i128 = [0u64; 5];
        rekey_centered_i64_to_modulus_slice(m, &mut via_i64, &ternary_i64);
        rekey_centered_i128_to_modulus_slice(m, &mut via_i128, &ternary_i128);
        assert_eq!(via_i64, via_i128);
    }

    #[test]
    #[should_panic(expected = "dst/src length mismatch")]
    fn rekey_i128_panics_on_length_mismatch() {
        let mut dst = [0u64; 3];
        rekey_centered_i128_to_modulus_slice(ConstModulus::<97>, &mut dst, &[1i128, 2]);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "does not fit in i64")]
    fn rekey_i128_panics_on_oversize_coefficient() {
        let mut dst = [0u64; 1];
        let oversize = [i64::MAX as i128 + 1];
        rekey_centered_i128_to_modulus_slice(ConstModulus::<97>, &mut dst, &oversize);
    }
}
