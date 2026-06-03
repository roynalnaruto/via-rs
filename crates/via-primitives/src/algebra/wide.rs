//! Fixed-width 256-bit arithmetic helpers.
//!
//! These exist because §2.3 gadget decomposition on the RNS path needs a
//! single computation — `round(|c_centered| · B^L / Q)` — whose
//! intermediate product overflows `i128`. At paper VIA-C parameters
//! (`Q ≈ 2⁷⁵`, `B^L ≈ 2⁷⁵` for the LWE-to-RLWE conversion-key gadget)
//! the product `|c| · B^L` can reach `2¹⁴⁹` — comfortably inside a
//! 256-bit intermediate, well past the 128-bit limit.
//!
//! Rather than pulling in a `crypto-bigint`-style dependency for one
//! function, we hand-roll a focused 256-bit-by-128-bit divide via a u128
//! schoolbook multiply + binary long division. ~100 LOC, no_std-friendly,
//! verified against tight unit tests at the boundary cases.
//!
//! Crate-internal only; not part of the public API.

const LOW_64: u128 = (1u128 << 64) - 1;

/// Multiply two `u128` values into a 256-bit result, returned as
/// `(lo, hi)` little-endian 128-bit words.
///
/// Schoolbook `u64 × u64 → u128` partial products, four-way carry
/// propagation. Branchless, no overflow.
pub(crate) fn widening_mul_u128(a: u128, b: u128) -> (u128, u128) {
    let a_lo = a & LOW_64;
    let a_hi = a >> 64;
    let b_lo = b & LOW_64;
    let b_hi = b >> 64;

    // Each partial product fits in u128 (u64 × u64 ≤ 2^128 − 2^65 + 1).
    let p00 = a_lo * b_lo;
    let p01 = a_lo * b_hi;
    let p10 = a_hi * b_lo;
    let p11 = a_hi * b_hi;

    // Limb 0: low 64 bits of p00.
    let w0 = p00 & LOW_64;

    // Limb 1: p00.hi + p01.lo + p10.lo + carries from below.
    // Sum of three u64-sized values fits in u128 (≤ 3·(2^64−1) < 2^66).
    let mid1 = (p00 >> 64) + (p01 & LOW_64) + (p10 & LOW_64);
    let w1 = mid1 & LOW_64;
    let carry1 = mid1 >> 64;

    // Limb 2: carry1 + p01.hi + p10.hi + p11.lo. Sum ≤ ~2^66, fits in u128.
    let mid2 = carry1 + (p01 >> 64) + (p10 >> 64) + (p11 & LOW_64);
    let w2 = mid2 & LOW_64;
    let carry2 = mid2 >> 64;

    // Limb 3: carry2 + p11.hi. Fits trivially.
    let w3 = carry2 + (p11 >> 64);

    let lo = w0 | (w1 << 64);
    let hi = w2 | (w3 << 64);
    (lo, hi)
}

/// Divide a 256-bit dividend (`hi`, `lo`) by a 128-bit divisor, returning
/// the `u128` quotient.
///
/// # Preconditions
///
/// - `divisor > 0`.
/// - `hi < divisor` — otherwise the quotient overflows `u128`. Debug-asserted.
///
/// For our use case (gadget scale step), the quotient is bounded by `B^L`
/// which fits in `u128` with margin, and the precondition holds.
///
/// # Algorithm
///
/// Binary long division: 256 iterations, MSB-first, each step shifting
/// the running remainder left by one and conditionally subtracting the
/// divisor. The shift can momentarily produce a value ≥ `2^128`; we
/// track that "overflow bit" implicitly through the subtraction
/// condition.
pub(crate) fn div_u256_by_u128(hi: u128, lo: u128, divisor: u128) -> u128 {
    assert!(divisor > 0, "div_u256_by_u128: divisor must be nonzero");
    // Fast path: dividend fits in u128.
    if hi == 0 {
        return lo / divisor;
    }
    debug_assert!(
        hi < divisor,
        "div_u256_by_u128: quotient overflows u128 (hi >= divisor)"
    );

    let mut remainder: u128 = 0;
    let mut quotient: u128 = 0;

    for i in (0..256).rev() {
        // Bit i of the 256-bit dividend.
        let bit = if i >= 128 {
            (hi >> (i - 128)) & 1
        } else {
            (lo >> i) & 1
        };

        // The bit shifted off the top of `remainder` represents an
        // overflow into the (implicit) bit 128. The invariant
        // `remainder < divisor < 2^128` holds at the top of each
        // iteration, so `overflow_bit` is 0 unless the shift pushed
        // the top bit off.
        let overflow_bit = remainder >> 127;
        remainder = (remainder << 1) | bit;

        // Subtract divisor whenever the effective 129-bit value is ≥
        // divisor. Two cases: (a) overflow_bit set (effective value
        // ≥ 2^128 > divisor); (b) overflow_bit clear and `remainder`
        // already ≥ divisor.
        let should_sub = overflow_bit != 0 || remainder >= divisor;
        if should_sub {
            // `remainder.wrapping_sub(divisor)` gives the correct low
            // 128 bits of (effective − divisor) in either case. The
            // invariant says the result fits in u128 (< divisor).
            remainder = remainder.wrapping_sub(divisor);
            if i < 128 {
                quotient |= 1u128 << i;
            }
            // `i >= 128 && should_sub` would imply a quotient bit ≥ 128,
            // which would overflow u128 — guarded by the `hi < divisor`
            // precondition.
        }
    }
    quotient
}

/// Compute `round(num_abs · scale / divisor)`, where the intermediate
/// product `num_abs · scale` may exceed 128 bits.
///
/// Rounding is half-up (Python's `(num + divisor / 2) // divisor` for
/// non-negative operands; the caller is responsible for sign-handling).
///
/// # Preconditions
///
/// - `divisor > 0`.
/// - `scale < divisor` (typical for gadget decomposition where
///   `scale = B^L` and `B^L < Q` by construction); else the quotient
///   may overflow `u128`.
pub(crate) fn round_mul_div_u128(num_abs: u128, scale: u128, divisor: u128) -> u128 {
    debug_assert!(divisor > 0);
    let (lo, hi) = widening_mul_u128(num_abs, scale);
    // Add divisor / 2 for half-up rounding. May carry into `hi`.
    let half = divisor / 2;
    let (rounded_lo, carry) = lo.overflowing_add(half);
    let rounded_hi = hi + u128::from(carry);
    div_u256_by_u128(rounded_hi, rounded_lo, divisor)
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // widening_mul_u128
    // -----------------------------------------------------------------------

    #[test]
    fn widening_mul_zero_and_one() {
        assert_eq!(widening_mul_u128(0, 0), (0, 0));
        assert_eq!(widening_mul_u128(1, 0), (0, 0));
        assert_eq!(widening_mul_u128(0, 1), (0, 0));
        assert_eq!(widening_mul_u128(1, 1), (1, 0));
        assert_eq!(widening_mul_u128(u128::MAX, 1), (u128::MAX, 0));
    }

    /// `2^64 · 2^64 = 2^128`: lo = 0, hi = 1.
    #[test]
    fn widening_mul_2pow64_squared_is_2pow128() {
        let two_64: u128 = 1u128 << 64;
        let (lo, hi) = widening_mul_u128(two_64, two_64);
        assert_eq!(lo, 0);
        assert_eq!(hi, 1);
    }

    /// `(2^128 − 1)^2 = 2^256 − 2^129 + 1`. Low 128 bits = 1; high = `2^128 − 2`.
    #[test]
    fn widening_mul_u128_max_squared() {
        let (lo, hi) = widening_mul_u128(u128::MAX, u128::MAX);
        assert_eq!(lo, 1);
        assert_eq!(hi, u128::MAX - 1);
    }

    /// At paper VIA-C scale: `|c| ≈ 2^74`, `B^L = 18^18 ≈ 2^75`. The
    /// product is in `[0, 2^150)`, so `hi` is in `[0, 2^22)`.
    #[test]
    fn widening_mul_at_paper_via_c_scale() {
        let c_abs: u128 = (1u128 << 74) - 1;
        let base_pow_l: u128 = 18u128.pow(18);
        let (lo, hi) = widening_mul_u128(c_abs, base_pow_l);
        // Cross-check against a slow reference: convert each operand via
        // u128::checked_mul (would overflow on the real values, so we
        // verify the relation differently). We assert the product fits
        // within the documented bound: hi < 2^25 (generous).
        assert!(hi < (1u128 << 25), "hi = {hi:#x} larger than expected");
        // And reconstruct via division: `(hi · 2^128 + lo) / base_pow_l = c_abs`.
        let recovered = div_u256_by_u128(hi, lo, base_pow_l);
        assert_eq!(recovered, c_abs);
    }

    // -----------------------------------------------------------------------
    // div_u256_by_u128
    // -----------------------------------------------------------------------

    #[test]
    fn div_u256_fast_path_when_hi_is_zero() {
        assert_eq!(div_u256_by_u128(0, 100, 7), 14);
        assert_eq!(div_u256_by_u128(0, u128::MAX, 1), u128::MAX);
        assert_eq!(div_u256_by_u128(0, 0, 1), 0);
    }

    /// `2^128 / 3` — the dividend just exceeds u128 range. Pinned
    /// against the known value `floor(2^128 / 3) = 113427455640312821154458202477256070485`,
    /// with remainder 1 (since `3·q = 2^128 − 1`).
    #[test]
    fn div_u256_2pow128_by_3() {
        let q = div_u256_by_u128(1, 0, 3);
        assert_eq!(q, 113_427_455_640_312_821_154_458_202_477_256_070_485u128);
        // Verify `3·q = 2^128 − 1` via widening_mul.
        let (lo_mul, hi_mul) = widening_mul_u128(q, 3);
        assert_eq!(hi_mul, 0);
        assert_eq!(lo_mul, u128::MAX); // 2^128 − 1
    }

    /// At paper VIA-C LWE-to-RLWE conversion: `Q ≈ 2^75`, scale `≈ 2^75`,
    /// `|c| < Q/2`. The full `round_mul_div_u128` round-trip is exercised
    /// indirectly via the gadget decomposition tests in Phase 5; here we
    /// just pin one concrete value.
    #[test]
    fn div_u256_round_trip_via_widening_mul() {
        // Pick num_abs, divisor, and check (num_abs · divisor) / divisor == num_abs.
        let divisor: u128 = (1u128 << 75) - 1;
        let num_abs: u128 = (1u128 << 74) + 12345;
        let (lo, hi) = widening_mul_u128(num_abs, divisor);
        assert_eq!(div_u256_by_u128(hi, lo, divisor), num_abs);
    }

    // -----------------------------------------------------------------------
    // round_mul_div_u128
    // -----------------------------------------------------------------------

    #[test]
    fn round_mul_div_zero_num_is_zero() {
        assert_eq!(round_mul_div_u128(0, 12345, 67), 0);
    }

    /// `round(2 · 5 / 3) = round(10/3) = round(3.33) = 3`. Verify with
    /// Python-style half-up rounding: (10 + 1) / 3 = 11/3 = 3.
    #[test]
    fn round_mul_div_small_values() {
        assert_eq!(round_mul_div_u128(2, 5, 3), 3);
        // round(5 · 5 / 3) = round(25/3) = round(8.33) = 8. (25 + 1) / 3 = 8.
        assert_eq!(round_mul_div_u128(5, 5, 3), 8);
        // round(2 · 3 / 4) = round(6/4) = round(1.5). Python's half-up: (6 + 2) / 4 = 2.
        assert_eq!(round_mul_div_u128(2, 3, 4), 2);
        // round(1 · 3 / 4) = round(0.75). (3 + 2) / 4 = 1.
        assert_eq!(round_mul_div_u128(1, 3, 4), 1);
        // round(1 · 1 / 4) = round(0.25). (1 + 2) / 4 = 0.
        assert_eq!(round_mul_div_u128(1, 1, 4), 0);
    }

    /// Paper VIA-C: round-trip with `divisor ≈ 2^75`, `scale ≈ 2^75`.
    /// Verifies the whole pipeline (widening mul + add half + div).
    #[test]
    fn round_mul_div_at_paper_via_c_scale() {
        // Q = VIA-C q_1 (paper RNS Q value) ≈ 2^75.
        // q_1 = 137438822401 * 274810798081.
        let q: u128 = 137_438_822_401u128 * 274_810_798_081u128;
        // B^L = 18^18 ≈ 2^75.
        let base_pow_l: u128 = 18u128.pow(18);
        // num_abs = floor(Q / 3) — well inside Q/2 budget.
        let num_abs: u128 = q / 3;
        // Expected: round(num_abs · base_pow_l / Q).
        // num_abs / Q ≈ 1/3, so result ≈ base_pow_l / 3.
        let result = round_mul_div_u128(num_abs, base_pow_l, q);
        let expected_approx = base_pow_l / 3;
        // Rounding error is at most 1 in either direction.
        assert!(
            result.abs_diff(expected_approx) <= 1,
            "result {result} differs from expected ~{expected_approx} by more than 1"
        );
    }
}
