//! Primitive §1.4 — bounded uniform sampler over $[-B, B]$.
//!
//! Per-coefficient call to [`Shake256Prg::uniform_below`] at bound
//! $2B + 1$ followed by a shift by $-B$.
//!
//! The bound is typed `u32` so that $2B + 1$ never overflows the `u64` PRG
//! interface; valid range is `B ∈ [0, u32::MAX]`. Realistic parameter sets use
//! $B \le 2$ (the VIA scheme's secret-key distribution is uniform on
//! $[-2, 2]$), so the `u32` ceiling is generous.

use crate::sampling::prg::Shake256Prg;

/// Fill `out` with coefficients sampled uniformly from $[-\text{bound},
/// \text{bound}]$ (inclusive on both ends — $2 \cdot \text{bound} + 1$ values).
///
/// Each output is one [`Shake256Prg::uniform_below`] draw at bound $2B + 1$,
/// minus $B$. PRG byte budget matches the Python reference's
/// `DeterministicSampler::bounded_uniform_poly(n, bound)` exactly.
///
/// Outputs are signed `i32`. Callers that need the coefficients lifted into a
/// modulus `[0, q)` should go through the `lift_centered_i32_into_zq` helper
/// (Phase 4).
///
/// ## Edge case: `bound == 0`
///
/// Reduces to a single-value distribution returning `0` for every coefficient
/// (via `uniform_below(1)`, which short-circuits without consuming PRG bytes).
/// Useful as a degenerate test fixture; pinned here to keep the byte budget
/// well-defined.
#[inline]
pub fn bounded_uniform(prg: &mut Shake256Prg, bound: u32, out: &mut [i32]) {
    // `bound` is u32 so `2 * bound + 1` ≤ 2^33 - 1, well within u64.
    let range = 2u64 * (bound as u64) + 1;
    let shift = bound as i32;
    for c in out.iter_mut() {
        let v = prg.uniform_below(range);
        // `v` ∈ [0, 2B + 1) fits in i64 trivially; subtraction by `shift` lands
        // in [-B, B], which fits in i32 because `bound: u32` ≤ i32::MAX would
        // be the prerequisite to ever returning B itself, and `bound` is
        // already u32. For `bound > i32::MAX as u32` the subtraction would
        // overflow; debug builds catch it via the cast, release builds wrap.
        *c = (v as i32).wrapping_sub(shift);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First 16 outputs of `DeterministicSampler(b"test").bounded_uniform_poly(16, 2)`.
    const TEST_SEED_BP_B2_N16: [i32; 16] = [-1, 0, 2, 1, 0, 0, 0, -2, 0, 2, -1, 1, -1, -1, 0, 2];

    /// First 12 outputs of `DeterministicSampler(b"test").bounded_uniform_poly(12, 3)`.
    const TEST_SEED_BP_B3_N12: [i32; 12] = [-2, -1, 2, 3, 3, 1, 0, -1, -1, 2, -1, 3];

    /// First 8 outputs of `DeterministicSampler(b"test").bounded_uniform_poly(8, 1024)`.
    /// Exercises a 2-byte randbelow path (bound = 2049, bits = 11).
    const TEST_SEED_BP_B1024_N8: [i32; 8] = [749, 394, -676, 553, 50, 869, -223, -929];

    #[test]
    fn parity_b2_n16() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i32; 16];
        bounded_uniform(&mut prg, 2, &mut out);
        assert_eq!(out, TEST_SEED_BP_B2_N16);
    }

    #[test]
    fn parity_b3_n12() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i32; 12];
        bounded_uniform(&mut prg, 3, &mut out);
        assert_eq!(out, TEST_SEED_BP_B3_N12);
    }

    #[test]
    fn parity_b1024_n8() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i32; 8];
        bounded_uniform(&mut prg, 1024, &mut out);
        assert_eq!(out, TEST_SEED_BP_B1024_N8);
    }

    #[test]
    fn all_outputs_in_range() {
        let mut prg = Shake256Prg::new(b"in-range");
        let mut out = [0i32; 500];
        bounded_uniform(&mut prg, 17, &mut out);
        for &v in &out {
            assert!((-17..=17).contains(&v));
        }
    }

    #[test]
    fn coverage_includes_extremes() {
        // Over 1000 samples with B=2 we should see both -2 and 2.
        let mut prg = Shake256Prg::new(b"extremes");
        let mut out = [0i32; 1000];
        bounded_uniform(&mut prg, 2, &mut out);
        assert!(out.contains(&-2));
        assert!(out.contains(&2));
    }

    #[test]
    fn bound_zero_returns_all_zero_without_byte_draw() {
        let mut prg_a = Shake256Prg::new(b"b0");
        let mut out = [0i32; 8];
        bounded_uniform(&mut prg_a, 0, &mut out);
        assert_eq!(out, [0; 8]);
        // PRG state must be unchanged — `uniform_below(1)` short-circuits.
        let mut prg_b = Shake256Prg::new(b"b0");
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        prg_a.fill_bytes(&mut a);
        prg_b.fill_bytes(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn empty_output_is_noop() {
        let mut prg_a = Shake256Prg::new(b"noop");
        let mut empty: [i32; 0] = [];
        bounded_uniform(&mut prg_a, 5, &mut empty);
        let mut prg_b = Shake256Prg::new(b"noop");
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        prg_a.fill_bytes(&mut a);
        prg_b.fill_bytes(&mut b);
        assert_eq!(a, b);
    }
}
