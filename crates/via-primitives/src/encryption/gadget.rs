//! Gadget primitives — `.docs/primitives.md` §2.3, VIA convention.
//!
//! A gadget is a fixed `(B, L)`-parameterised pair of:
//!
//! - a **gadget vector** $\mathbf{g} \in \mathbb{Z}_q^L$ with $g_i =
//!   \mathrm{round}(q / B^{i+1})$ (MSB-first ordering: $g_0$ is the
//!   largest entry), and
//! - a **decomposition** $\mathrm{gdec}(x) \in \mathbb{Z}^L$ that maps any
//!   $x \in \mathbb{Z}_q$ to a tuple of signed-balanced digits in
//!   $(-B/2, B/2]$ satisfying
//!   $x \approx \sum_i d_i \cdot g_i \pmod q$.
//!
//! The "≈" hides a reconstruction error bounded by $\mathrm{round}(q /
//! B^L) / 2$ per coefficient — small enough to be absorbed into RLWE
//! noise.
//!
//! ## Why these primitives matter
//!
//! Layer 2's RLev / RGSW / external-product machinery (Phase 6-7) and
//! Layer 3's key-switching (Phase 8) all reduce to **gadget products**:
//! "multiply by something large" is replaced by "decompose the large
//! thing into a sum of small things, then multiply each small thing by a
//! pre-computed encryption". The decomposed multipliers have
//! infinity-norm $\le B/2$, which is what keeps homomorphic ciphertext
//! noise inside the decryption budget across the protocol's many
//! multiplicative steps.
//!
//! ## VIA convention vs GSW convention
//!
//! VIA's gadget vector is $\mathbf{g} = (q/B, q/B^2, \ldots, q/B^L)$,
//! **not** the standard GSW $\mathbf{g} = (1, B, B^2, \ldots, B^{L-1})$.
//! The two are interchangeable up to a $q / B^L$ scaling, but the noise
//! analysis in the paper (Appendix C) is written for VIA's. Mixing
//! conventions silently degrades the noise bounds.
//!
//! ## API shape
//!
//! - [`gadget_vector_values`] — the `[u128; L]` entries.
//! - [`gadget_scale_into`] — the expensive step, computes per-coefficient
//!   `round(c_centered · B^L / Q)` in a wider-than-`i128` intermediate
//!   when the RNS backend needs it.
//! - [`gadget_extract_lsb_into`] — pulls one base-`B` digit per
//!   coefficient (LSB-first), mutating the scratch buffer in place.
//! - [`gadget_decompose_into`] — convenience over scale + L levels of
//!   extract; writes output in MSB-first order so `out[0]` pairs with
//!   $g_0$ (the largest gadget entry).
//! - [`reconstruct`] — inverse for tests; computes $\sum_i d_i g_i$.
//!
//! The split lets Phase 7's `gadget_product` stream over levels using
//! `O(N)` scratch (`16 KiB` at `N=2048`) rather than materialising the
//! full `[[i64; N]; L]` decomposition (`288 KiB` at `N=2048, L=18`).

use crate::algebra::ring::RingPoly;
use crate::algebra::wide::round_mul_div_u128;

/// Compute the VIA-convention gadget vector entries
/// `[round(q/B), round(q/B²), …, round(q/B^L)]` as raw `u128` values,
/// MSB-first (`out[0]` is the largest entry).
///
/// Returns a `[u128; L]` rather than `[R::Scalar; L]` because:
///
/// 1. For the RNS backend, entries can exceed `u64`; `u128` is the
///    smallest type that fits both backends uniformly.
/// 2. Layer-5 reconstruction and Phase-7 gadget-product call sites lift
///    these into scalar polynomials lazily; centralising the
///    construction here would force every consumer through the same path.
///
/// # Panics
///
/// Debug-asserts `base >= 2` and `L >= 1`. Panics if `B^i` overflows
/// `u128` for any `i ≤ L` — well beyond any realistic VIA parameter
/// (paper worst case is `B^L ≈ 2⁷⁵`).
pub fn gadget_vector_values<const N: usize, R: RingPoly<N>, const L: usize>(
    modulus: R::Modulus,
    base: u64,
) -> [u128; L] {
    debug_assert!(base >= 2, "gadget base must be >= 2");
    debug_assert!(L >= 1, "gadget depth must be >= 1");

    let q = R::modulus_value(modulus);
    let mut out = [0u128; L];
    let mut divisor: u128 = 1;
    for slot in out.iter_mut() {
        divisor = divisor
            .checked_mul(u128::from(base))
            .expect("gadget_vector_values: B^i overflows u128");
        // Python: `(q + divisor // 2) // divisor` — half-up rounding.
        *slot = (q + divisor / 2) / divisor;
    }
    out
}

/// Per-coefficient scale: compute `round(c_centered · B^L / Q)` for
/// every coefficient of `input`, writing the signed `i128` results into
/// `scratch`. The expensive half of decomposition.
///
/// **Single-prime** (`Q ≤ u64::MAX`): the product `|c| · B^L` fits in
/// `i128` directly; no wide arithmetic needed.
///
/// **RNS** (`Q` may be up to ~`2¹²⁶`): the product can reach `~2²⁰⁰` in
/// the abstract; for paper parameters it tops out at `~2¹⁴⁹` (at VIA-C
/// LWE-to-RLWE conversion key with `Q ≈ 2⁷⁵`, `B^L ≈ 2⁷⁵`). We dispatch
/// to `crate::algebra::wide::round_mul_div_u128` (crate-private helper)
/// which uses a 256-bit intermediate.
///
/// # Sign handling
///
/// Matches the Python reference (`gadget.py:84-91`): negative inputs
/// take the absolute value, scale, divide, then negate. This is
/// **round-half-away-from-zero**, distinct from Rust's truncating `/`.
///
/// # Panics
///
/// Debug-asserts `base >= 2` and `depth >= 1`. Panics if `B^depth`
/// overflows `u128`.
pub fn gadget_scale_into<const N: usize, R: RingPoly<N>>(
    input: &R,
    base: u64,
    depth: u8,
    scratch: &mut [i128; N],
) {
    debug_assert!(base >= 2, "gadget base must be >= 2");
    debug_assert!(depth >= 1, "gadget depth must be >= 1");

    let q = R::modulus_value(input.modulus());
    let mut base_pow_l: u128 = 1;
    for _ in 0..depth {
        base_pow_l = base_pow_l
            .checked_mul(u128::from(base))
            .expect("gadget_scale_into: B^L overflows u128");
    }

    let mut centered = [0i128; N];
    input.to_centered_i128_coeffs(&mut centered);

    for (out, &c) in scratch.iter_mut().zip(centered.iter()) {
        let c_abs = c.unsigned_abs();
        let val_abs = round_mul_div_u128(c_abs, base_pow_l, q);
        // Quotient fits in i128 because `c_abs < Q/2` and the quotient
        // is bounded by `B^L`, which fits in u128 by the precondition.
        // For practical parameters `B^L ≤ 2¹²⁶`, well within `i128`.
        debug_assert!(
            val_abs <= i128::MAX as u128,
            "gadget_scale_into: scaled value overflows i128"
        );
        *out = if c >= 0 {
            val_abs as i128
        } else {
            -(val_abs as i128)
        };
    }
}

/// Extract one base-`B` signed-balanced digit per coefficient
/// (LSB-first), mutating `scratch` to remove the extracted contribution.
///
/// # Output range
///
/// Digits satisfy `2·d > -B && 2·d ≤ B`, equivalent to:
///
/// - Even `B`: digit in `(-B/2, B/2]`.
/// - Odd `B`:  digit in `[-(B-1)/2, (B-1)/2]` (symmetric).
///
/// This matches the Python reference `gadget.py:94-99` exactly — the
/// `if digit > B/2 { digit -= B }` rebalance step yields the asymmetric
/// even-B range and the symmetric odd-B range as a unified rule.
///
/// After the call, `scratch[i]` has been advanced by `(val − digit) /
/// B`, preparing the next call to extract the next-higher base-`B` digit.
pub fn gadget_extract_lsb_into<const N: usize>(
    base: u64,
    scratch: &mut [i128; N],
    out: &mut [i64; N],
) {
    debug_assert!(base >= 2, "gadget base must be >= 2");
    let base_i128 = i128::from(base);
    let half_base = base_i128 / 2;

    for (out_slot, val_slot) in out.iter_mut().zip(scratch.iter_mut()) {
        let val = *val_slot;
        // Python's `val % base` returns a non-negative result; rem_euclid matches.
        let mut digit = val.rem_euclid(base_i128);
        if digit > half_base {
            digit -= base_i128;
        }
        // Unified bound: 2d > -B and 2d ≤ B (covers both even and odd B).
        debug_assert!(2 * digit > -base_i128 && 2 * digit <= base_i128);
        debug_assert!(digit.abs() <= i64::MAX as i128, "digit overflows i64");
        *out_slot = digit as i64;
        *val_slot = (val - digit) / base_i128;
    }
}

/// Full decomposition: scale + `L` levels of LSB extraction, with
/// output written **MSB-first** so `out[0]` pairs with the largest
/// gadget entry $g_0$.
///
/// Memory: uses an `N`-sized `i128` scratch on the stack (`16 KiB` at
/// `N=2048`). The output array is filled in reverse during extraction,
/// so no separate buffer for the LSB-first form is needed.
pub fn gadget_decompose_into<const N: usize, R: RingPoly<N>, const L: usize>(
    input: &R,
    base: u64,
    out: &mut [[i64; N]; L],
) {
    debug_assert!(L >= 1, "gadget depth must be >= 1");
    debug_assert!(L <= u8::MAX as usize, "gadget depth must fit in u8");

    let mut scratch = [0i128; N];
    gadget_scale_into::<N, R>(input, base, L as u8, &mut scratch);

    // Extract LSB-first, but write into `out[L - 1 - level]` so the final
    // array is naturally MSB-first.
    for level in 0..L {
        let target = L - 1 - level;
        gadget_extract_lsb_into::<N>(base, &mut scratch, &mut out[target]);
    }
}

/// Inverse of [`gadget_decompose_into`] for tests: computes
/// $\sum_{i=0}^{L-1} d_i \cdot g_i \pmod q$.
///
/// The reconstruction error per coefficient is bounded by
/// $\mathrm{round}(q / B^L) / 2$ — the smallest gadget entry, halved.
/// At paper parameters this is well under the decryption budget.
pub fn reconstruct<const N: usize, R: RingPoly<N>, const L: usize>(
    digits: &[[i64; N]; L],
    modulus: R::Modulus,
    base: u64,
) -> R {
    let g_values = gadget_vector_values::<N, R, L>(modulus, base);

    let mut sum = R::zero(modulus);
    for (level_digits, &g_value) in digits.iter().zip(g_values.iter()) {
        // Lift digit polynomial into R.
        let digit_poly = R::from_centered_i64s(modulus, level_digits);
        // Build the gadget entry as a constant polynomial
        // `(g_value, 0, 0, …, 0)` so we can use the existing
        // `Mul<R, Output = R>` from the trait's supertrait bound.
        let mut g_const = [0u128; N];
        g_const[0] = g_value;
        let g_poly = R::from_u128_coeffs(modulus, &g_const);
        sum += digit_poly * g_poly;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::zq::modulus::{ConstModulus, PowerOfTwoModulus};

    type SinglePolyQ17<const N: usize> = Poly<N, ConstModulus<17>, Coefficient>;
    type SinglePolyQ1024<const N: usize> = Poly<N, PowerOfTwoModulus<10>, Coefficient>;
    type SinglePolyViaCQ2<const N: usize> =
        Poly<N, crate::algebra::zq::modulus::paper::ViaCQ2, Coefficient>;
    type RnsPolyViaCQ1<const N: usize> =
        PolyRns<N, crate::algebra::rns::basis::paper::ViaCQ1Rns, Coefficient>;

    // -----------------------------------------------------------------------
    // gadget_vector_values
    // -----------------------------------------------------------------------

    /// At q=17, B=2, L=4: round(17/2)=9, round(17/4)=4, round(17/8)=2,
    /// round(17/16)=1. Pins the round-vs-ceil distinction.
    #[test]
    fn gadget_vector_at_q17_b2_l4_exact_values() {
        let g = gadget_vector_values::<4, SinglePolyQ17<4>, 4>(ConstModulus, 2);
        assert_eq!(g, [9, 4, 2, 1]);
    }

    /// q = 1024 = 2^10, B = 2, L = 10: exact powers of 2 with no rounding.
    #[test]
    fn gadget_vector_at_q1024_b2_l10_is_exact_powers() {
        let g = gadget_vector_values::<4, SinglePolyQ1024<4>, 10>(PowerOfTwoModulus, 2);
        let expected: [u128; 10] = [512, 256, 128, 64, 32, 16, 8, 4, 2, 1];
        assert_eq!(g, expected);
    }

    /// Paper VIA-C `q₁` (RNS Q) at `(L=18, B=18)` — exercises the
    /// gadget_vector path through `modulus_value` on the RNS backend.
    /// We don't hard-code the full 18-entry array; instead assert
    /// recurrence `g[i] · B ≈ g[i-1]` (within rounding).
    #[test]
    fn gadget_vector_at_via_c_q1_b18_l18_satisfies_recurrence() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let g = gadget_vector_values::<16, RnsPolyViaCQ1<16>, 18>(basis, 18);
        // First entry should be approximately Q / 18.
        let q = <RnsPolyViaCQ1<16> as RingPoly<16>>::modulus_value(basis);
        let expected_first = (q + 9) / 18;
        assert_eq!(g[0], expected_first);
        // Recurrence: g[i] · 18 ≈ g[i-1] (within ±1).
        for i in 1..18 {
            let approx = g[i] * 18;
            assert!(
                approx.abs_diff(g[i - 1]) <= 18,
                "gadget recurrence broken at level {i}"
            );
        }
        // Last entry should be `≈ q / 18^18`, which is small (≈ 1 to a few).
        let final_divisor = 18u128.pow(18);
        let expected_last = (q + final_divisor / 2) / final_divisor;
        assert_eq!(g[17], expected_last);
    }

    // -----------------------------------------------------------------------
    // gadget_decompose + reconstruct round-trip
    // -----------------------------------------------------------------------

    /// Exhaustive over `c ∈ [0, 17)` at `(B=2, L=4)`: every reconstruction
    /// is within `round(17/16)/2 = 0` of the input (since the smallest
    /// gadget entry `g[L-1] = 1`, so error bound is `0.5` → rounds to 0).
    ///
    /// Actually the spec error bound is `round(q/B^L)/2 = round(1)/2 = 1/2`,
    /// so reconstruction can be off by at most ±0. In practice it's exact
    /// for many inputs and within ±1 for the boundary cases.
    #[test]
    fn decompose_reconstruct_q17_b2_l4_exhaustive() {
        // For each c in [0, 17), build a 4-coefficient polynomial with that
        // value in slot 0, zero elsewhere; decompose; reconstruct; check.
        let m = ConstModulus::<17>;
        for c in 0u64..17 {
            let mut coeffs = [0u64; 4];
            coeffs[0] = c;
            let input: SinglePolyQ17<4> = Poly::new(m, coeffs);
            let mut digits = [[0i64; 4]; 4];
            gadget_decompose_into::<4, SinglePolyQ17<4>, 4>(&input, 2, &mut digits);
            let recovered: SinglePolyQ17<4> = reconstruct::<4, SinglePolyQ17<4>, 4>(&digits, m, 2);
            // Centered diff bounded by round(17/16)/2 = 0.5 → ≤ 1 in
            // integer arithmetic (the spec is loose here; in practice
            // we observe exact recovery at this small a parameter set).
            let mut diff_coeffs = [0i64; 4];
            let raw_diff = input - recovered;
            raw_diff.to_centered_coeffs(&mut diff_coeffs);
            for (i, &d) in diff_coeffs.iter().enumerate() {
                assert!(
                    d.abs() <= 1,
                    "reconstruction error exceeded bound at c={c} i={i}: diff={d}"
                );
            }
        }
    }

    /// At VIA-C `q₂` ≈ 2³⁴ with `(L=4, B=24)` (VIA ring-switch params):
    /// reconstruction error ≤ `round(q₂ / 24⁴) / 2`.
    #[test]
    fn decompose_reconstruct_at_via_c_q2_b24_l4_single_prime() {
        let q = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        let coeffs: [u64; 16] = [
            0,
            1,
            100,
            12345,
            1_000_000,
            2_000_000_000,
            17_175_674_000,
            17_175_000_000,
            50,
            99,
            200,
            5_000_000,
            99_999_999,
            8_000_000_000,
            1,
            17_175_674_880,
        ];
        let input: SinglePolyViaCQ2<16> = Poly::new(q, coeffs);
        let mut digits = [[0i64; 16]; 4];
        gadget_decompose_into::<16, SinglePolyViaCQ2<16>, 4>(&input, 24, &mut digits);
        let recovered: SinglePolyViaCQ2<16> =
            reconstruct::<16, SinglePolyViaCQ2<16>, 4>(&digits, q, 24);
        let mut diff_coeffs = [0i64; 16];
        let raw_diff = input - recovered;
        raw_diff.to_centered_coeffs(&mut diff_coeffs);
        let bound: i64 = ((17_175_674_881u128 + 24u128.pow(4) / 2) / 24u128.pow(4) / 2) as i64 + 1;
        for (i, &d) in diff_coeffs.iter().enumerate() {
            assert!(
                d.abs() <= bound,
                "reconstruction error exceeded bound at i={i}: diff={d}, bound={bound}"
            );
        }
    }

    /// **Flagship**: paper-class RNS at `(L=18, B=18, Q ≈ 2⁷⁵)`. The
    /// wider-than-`i128` arithmetic path is exercised end-to-end.
    #[test]
    fn decompose_reconstruct_at_via_c_q1_b18_l18_rns() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        // Build an input by lifting a known set of u128 values via
        // `from_u128_coeffs`. Pick values spread across [0, Q).
        let q = <RnsPolyViaCQ1<16> as RingPoly<16>>::modulus_value(basis);
        let mut coeffs = [0u128; 16];
        for (i, slot) in coeffs.iter_mut().enumerate() {
            // Spread across the modulus.
            *slot = (q / 17) * i as u128;
        }
        let input = <RnsPolyViaCQ1<16> as RingPoly<16>>::from_u128_coeffs(basis, &coeffs);
        let mut digits = [[0i64; 16]; 18];
        gadget_decompose_into::<16, RnsPolyViaCQ1<16>, 18>(&input, 18, &mut digits);
        let recovered: RnsPolyViaCQ1<16> =
            reconstruct::<16, RnsPolyViaCQ1<16>, 18>(&digits, basis, 18);
        let mut diff_coeffs = [0i128; 16];
        let raw_diff = input - recovered;
        raw_diff.to_centered_coeffs(&mut diff_coeffs);
        // Bound: `round(Q/B^L) / 2` for the scale-step rounding, plus
        // `L · B / 4` for the gadget-vector rounding (each g_i differs
        // from the rational `Q / B^{i+1}` by at most 1/2, multiplied by
        // the max digit magnitude `B/2` and summed over `L` levels).
        let b_pow_l: u128 = 18u128.pow(18);
        let g_min: u128 = (q + b_pow_l / 2) / b_pow_l;
        let scale_step_bound: i128 = (g_min / 2) as i128;
        let gadget_rounding_bound: i128 = (18 * 18 / 4) as i128;
        let bound: i128 = scale_step_bound + gadget_rounding_bound + 1;
        for (i, &d) in diff_coeffs.iter().enumerate() {
            assert!(
                d.abs() <= bound,
                "RNS reconstruction error at i={i}: diff={d}, bound={bound}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Digit bounds
    // -----------------------------------------------------------------------

    /// Every digit must satisfy `-B/2 < d ≤ B/2`. Use B=8 (so digit in
    /// `(-4, 4]`) and a non-trivial input.
    #[test]
    fn digit_bounds_b8_l3_at_q1024() {
        let q = PowerOfTwoModulus::<10>;
        let coeffs = [3u64, 100, 511, 999];
        let input: SinglePolyQ1024<4> = Poly::new(q, coeffs);
        let mut digits = [[0i64; 4]; 3];
        gadget_decompose_into::<4, SinglePolyQ1024<4>, 3>(&input, 8, &mut digits);
        for (level, level_digits) in digits.iter().enumerate() {
            for (i, &d) in level_digits.iter().enumerate() {
                assert!(d > -4, "digit underflow at level {level} i={i}: d={d}");
                assert!(d <= 4, "digit overflow at level {level} i={i}: d={d}");
            }
        }
    }

    /// Same on the RNS backend at paper params.
    #[test]
    fn digit_bounds_at_via_c_q1_b18_l4_rns() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let q = <RnsPolyViaCQ1<4> as RingPoly<4>>::modulus_value(basis);
        let coeffs: [u128; 4] = [0, q / 3, q / 2 - 1, q - 1];
        let input = <RnsPolyViaCQ1<4> as RingPoly<4>>::from_u128_coeffs(basis, &coeffs);
        let mut digits = [[0i64; 4]; 4];
        gadget_decompose_into::<4, RnsPolyViaCQ1<4>, 4>(&input, 18, &mut digits);
        for (level, level_digits) in digits.iter().enumerate() {
            for (i, &d) in level_digits.iter().enumerate() {
                assert!(d > -9, "RNS digit underflow at level {level} i={i}: d={d}");
                assert!(d <= 9, "RNS digit overflow at level {level} i={i}: d={d}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Python parity
    // -----------------------------------------------------------------------

    /// Hand-computed against the Python reference at q=17, B=2, L=4.
    ///
    /// For c=5: centered=5, val=round(5·16/17)=5, LSB-first digits
    /// `(1, 0, 1, 0)` (val mod 2 = 1, val=2; 0; 1; 0). Reversed to
    /// MSB-first: `(0, 1, 0, 1)`.
    ///
    /// For c=12: centered=12-17=-5. val=-round(5·16/17)=-5.
    /// LSB-first: val=-5 % 2 = 1 (Euclidean), digit=1, val=(-5-1)/2=-3.
    /// val=-3 % 2 = 1, digit=1, val=(-3-1)/2=-2. val=-2 % 2 = 0,
    /// digit=0, val=-1. val=-1 % 2 = 1, digit=1, val=-1.
    /// Wait — digit=1 > half_base=1 is false. Actually for B=2, half_base=1.
    /// `digit > half_base` is `1 > 1` = false. So digit stays 1, no rebalance.
    /// LSB-first: (1, 1, 0, 1). MSB-first: (1, 0, 1, 1).
    ///
    /// Verify reconstruction: 1·9 + 0·4 + 1·2 + 1·1 = 12. ✓
    #[test]
    fn python_parity_q17_b2_l4_hand_computed() {
        let m = ConstModulus::<17>;
        let input: SinglePolyQ17<4> = Poly::new(m, [5, 12, 0, 8]);
        let mut digits = [[0i64; 4]; 4];
        gadget_decompose_into::<4, SinglePolyQ17<4>, 4>(&input, 2, &mut digits);
        // c=0: centered=5, MSB-first digits (0, 1, 0, 1).
        // c=1: centered=-5, MSB-first digits (1, 0, 1, 1).
        // c=2: centered=0, all-zero digits.
        // c=3: centered=8, val=round(8·16/17)=round(7.529)=8. LSB:
        //   val=8 mod 2=0, val=4. 0, val=2. 0, val=1. 1, val=0.
        //   LSB: (0, 0, 0, 1). MSB: (1, 0, 0, 0). Reconstruct: 9. ✓ (rounded from 8).
        let expected: [[i64; 4]; 4] = [
            // level 0 (MSB, pairs with g[0]=9):
            [0, 1, 0, 1],
            // level 1 (pairs with g[1]=4):
            [1, 0, 0, 0],
            // level 2 (pairs with g[2]=2):
            [0, 1, 0, 0],
            // level 3 (LSB, pairs with g[3]=1):
            [1, 1, 0, 0],
        ];
        for level in 0..4 {
            for i in 0..4 {
                assert_eq!(
                    digits[level][i], expected[level][i],
                    "mismatch at level={level} i={i}"
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // Streaming == batch
    // -----------------------------------------------------------------------

    /// `gadget_scale_into` + level-by-level `gadget_extract_lsb_into`
    /// must produce the same digit polynomials (in LSB order) as
    /// `gadget_decompose_into` returns (in MSB-first order, so reversed).
    #[test]
    fn streaming_extract_matches_batch_decompose() {
        let q = PowerOfTwoModulus::<10>;
        let input: SinglePolyQ1024<4> = Poly::new(q, [42, 137, 999, 3]);

        // Batch: full decompose, MSB-first.
        let mut batch_digits = [[0i64; 4]; 6];
        gadget_decompose_into::<4, SinglePolyQ1024<4>, 6>(&input, 4, &mut batch_digits);

        // Streaming: scale once, extract each LSB into a local buffer,
        // compare against `batch_digits` in reverse-level order.
        let mut scratch = [0i128; 4];
        gadget_scale_into::<4, SinglePolyQ1024<4>>(&input, 4, 6, &mut scratch);
        let mut streaming_lsb = [[0i64; 4]; 6];
        for slot in streaming_lsb.iter_mut() {
            gadget_extract_lsb_into::<4>(4, &mut scratch, slot);
        }

        // The streaming LSB output at `level` must equal the batch MSB output
        // at `L - 1 - level`.
        for (level, lsb_digits) in streaming_lsb.iter().enumerate() {
            assert_eq!(lsb_digits, &batch_digits[6 - 1 - level], "level {level}");
        }
    }
}
