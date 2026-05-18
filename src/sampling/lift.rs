//! Layer-1 → Layer-0 bridge: lift signed centred-representation coefficients
//! into canonical `[0, q)` under a given [`Modulus`].
//!
//! Three width-typed helpers, one per signed integer type produced by the
//! samplers in §1.3–§1.5:
//!
//! | Helper | Source type | Producing sampler |
//! |---|---|---|
//! | [`lift_centered_i8_into_zq`] | `&[i8]` | §1.3 [`ternary`](crate::sampling::ternary::ternary) |
//! | [`lift_centered_i32_into_zq`] | `&[i32]` | §1.4 [`bounded_uniform`](crate::sampling::bounded::bounded_uniform) |
//! | [`lift_centered_i64_into_zq`] | `&[i64]` | §1.5 [`discrete_gaussian`](crate::sampling::gaussian::discrete_gaussian) and §1.6 [`Distribution::sample_into`](crate::sampling::distribution::Distribution::sample_into) |
//!
//! All three are thin wrappers around [`Modulus::reduce_i64`], which is
//! constant-time over the input sign. That matters for secret-key
//! coefficients: branching on `sign(v)` to choose between `v as u64` and
//! `q - (-v) as u64` would leak Hamming weight through timing.
//!
//! ## Length contract
//!
//! Each helper requires `src.len() == dst.len()` and panics otherwise. This
//! matches the Layer-0 kernel convention (`assert_eq!`, not `Result`).

use crate::algebra::zq::modulus::Modulus;

/// Reduce signed `i8` coefficients (typically ternary samples) into canonical
/// `[0, q)` under `modulus`. Uses [`Modulus::reduce_i64`] (constant-time over
/// the input sign).
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn lift_centered_i8_into_zq<M: Modulus>(modulus: M, src: &[i8], dst: &mut [u64]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "lift_centered_i8_into_zq: src and dst length mismatch"
    );
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = modulus.reduce_i64(*s as i64);
    }
}

/// Reduce signed `i32` coefficients (typically bounded-uniform samples) into
/// canonical `[0, q)` under `modulus`. Uses [`Modulus::reduce_i64`]
/// (constant-time over the input sign).
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn lift_centered_i32_into_zq<M: Modulus>(modulus: M, src: &[i32], dst: &mut [u64]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "lift_centered_i32_into_zq: src and dst length mismatch"
    );
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = modulus.reduce_i64(*s as i64);
    }
}

/// Reduce signed `i64` coefficients (typically Gaussian samples, or the
/// unified output of [`Distribution::sample_into`](crate::sampling::distribution::Distribution::sample_into))
/// into canonical `[0, q)` under `modulus`. Uses [`Modulus::reduce_i64`]
/// (constant-time over the input sign).
///
/// # Panics
///
/// Panics if `src.len() != dst.len()`.
#[inline]
pub fn lift_centered_i64_into_zq<M: Modulus>(modulus: M, src: &[i64], dst: &mut [u64]) {
    assert_eq!(
        src.len(),
        dst.len(),
        "lift_centered_i64_into_zq: src and dst length mismatch"
    );
    for (s, d) in src.iter().zip(dst.iter_mut()) {
        *d = modulus.reduce_i64(*s);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::zq::modulus::{ConstModulus, DynModulus, PowerOfTwoModulus};

    // -----------------------------------------------------------------------
    // Bucket E.1 — spot-value checks against hand-computed expected outputs.
    // -----------------------------------------------------------------------

    #[test]
    fn lift_i8_spot_values_const_modulus_17() {
        let m = ConstModulus::<17>;
        let src: [i8; 5] = [0, 1, -1, 8, -8];
        let mut dst = [0u64; 5];
        lift_centered_i8_into_zq(m, &src, &mut dst);
        // q = 17, so -1 → 16, -8 → 9.
        assert_eq!(dst, [0, 1, 16, 8, 9]);
    }

    #[test]
    fn lift_i32_spot_values_pow2_modulus_12() {
        // q = 4096.
        let m = PowerOfTwoModulus::<12>;
        let src: [i32; 4] = [0, 1, -1, -2048];
        let mut dst = [0u64; 4];
        lift_centered_i32_into_zq(m, &src, &mut dst);
        // -1 → 4095, -2048 → 2048.
        assert_eq!(dst, [0, 1, 4095, 2048]);
    }

    #[test]
    fn lift_i64_spot_values_dyn_modulus_q3() {
        // VIA-C q3 = 8_380_417.
        let m = DynModulus::new(8_380_417);
        let src: [i64; 5] = [0, 1, -1, 1024, -1024];
        let mut dst = [0u64; 5];
        lift_centered_i64_into_zq(m, &src, &mut dst);
        assert_eq!(dst, [0, 1, 8_380_416, 1024, 8_379_393]);
    }

    // -----------------------------------------------------------------------
    // Bucket E.2 — round-trip property across multiple Modulus impls.
    //
    // For values strictly in (-q/2, q/2], lifting then re-centring must yield
    // the original. We test each modulus impl with values within its centred
    // range.
    // -----------------------------------------------------------------------

    /// Per-impl round-trip on a fixed set of centred values.
    fn assert_i64_roundtrip<M: Modulus + Copy>(m: M, values: &[i64]) {
        let n = values.len();
        let mut lifted = [0u64; 32];
        let mut recovered = [0i64; 32];
        let lifted = &mut lifted[..n];
        let recovered = &mut recovered[..n];

        lift_centered_i64_into_zq(m, values, lifted);
        for (out, &raw) in recovered.iter_mut().zip(lifted.iter()) {
            *out = m.to_centered_i64(raw);
        }
        for (v, &back) in values.iter().zip(recovered.iter()) {
            assert_eq!(
                *v,
                back,
                "round-trip mismatch on value {} under q = {}",
                v,
                m.q()
            );
        }
    }

    #[test]
    fn roundtrip_i64_const_modulus_17() {
        let m = ConstModulus::<17>;
        // Centred range for q=17 is (-8, 8] (since q is odd, q/2 = 8).
        let values: [i64; 9] = [-8, -5, -1, 0, 1, 3, 5, 7, 8];
        assert_i64_roundtrip(m, &values);
    }

    #[test]
    fn roundtrip_i64_pow2_modulus_12() {
        let m = PowerOfTwoModulus::<12>;
        // q = 4096 is even, so the centred range is asymmetric: `(-q/2, q/2]`
        // = `[-2047, 2048]`. The Layer-0 `to_centered_i64` threshold is
        // `a > q/2 ⇒ a - q`, so `2048` stays positive and `-2048` is NOT a
        // round-trip fixed point (it lifts to `2048` and centres back to `2048`).
        let values: [i64; 9] = [-2047, -1024, -1, 0, 1, 1024, 2000, 2047, 2048];
        assert_i64_roundtrip(m, &values);
    }

    #[test]
    fn roundtrip_i64_dyn_modulus_q3() {
        let m = DynModulus::new(8_380_417);
        // q ≈ 2^23; centred bound is 4_190_208.
        let values: [i64; 7] = [-4_190_208, -1024, -1, 0, 1, 1024, 4_190_208];
        assert_i64_roundtrip(m, &values);
    }

    #[test]
    fn roundtrip_i8_through_i64_widen() {
        // Lift i8 → u64, recover via to_centered_i64, widen original i8 to i64
        // for comparison.
        let m = ConstModulus::<17>;
        let src: [i8; 5] = [-8, -1, 0, 5, 8];
        let mut lifted = [0u64; 5];
        lift_centered_i8_into_zq(m, &src, &mut lifted);
        for (s, &back_u) in src.iter().zip(lifted.iter()) {
            let back_i = m.to_centered_i64(back_u);
            assert_eq!(*s as i64, back_i);
        }
    }

    #[test]
    fn roundtrip_i32_through_i64_widen() {
        let m = PowerOfTwoModulus::<12>;
        // q = 4096; centred range is `[-2047, 2048]` (asymmetric — see
        // `roundtrip_i64_pow2_modulus_12`).
        let src: [i32; 5] = [-2047, -1, 0, 1024, 2000];
        let mut lifted = [0u64; 5];
        lift_centered_i32_into_zq(m, &src, &mut lifted);
        for (s, &back_u) in src.iter().zip(lifted.iter()) {
            let back_i = m.to_centered_i64(back_u);
            assert_eq!(*s as i64, back_i);
        }
    }

    // -----------------------------------------------------------------------
    // Bucket F — length-mismatch panics.
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "lift_centered_i8_into_zq: src and dst length mismatch")]
    fn lift_i8_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let src = [0i8; 4];
        let mut dst = [0u64; 5];
        lift_centered_i8_into_zq(m, &src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "lift_centered_i32_into_zq: src and dst length mismatch")]
    fn lift_i32_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let src = [0i32; 5];
        let mut dst = [0u64; 4];
        lift_centered_i32_into_zq(m, &src, &mut dst);
    }

    #[test]
    #[should_panic(expected = "lift_centered_i64_into_zq: src and dst length mismatch")]
    fn lift_i64_panics_on_length_mismatch() {
        let m = ConstModulus::<17>;
        let src = [0i64; 3];
        let mut dst = [0u64; 4];
        lift_centered_i64_into_zq(m, &src, &mut dst);
    }

    // -----------------------------------------------------------------------
    // Empty input/output is a no-op (doesn't panic, doesn't write).
    // -----------------------------------------------------------------------

    #[test]
    fn empty_input_is_noop() {
        let m = ConstModulus::<17>;
        let src: [i64; 0] = [];
        let mut dst: [u64; 0] = [];
        lift_centered_i64_into_zq(m, &src, &mut dst);
        // Just assert no panic — no observable side effects to check.
    }
}
