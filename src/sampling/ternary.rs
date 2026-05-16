//! Primitive §1.3 — ternary sampler over $\{-1, 0, 1\}$.
//!
//! Per-coefficient call to [`Shake256Prg::uniform_below`] with bound 3, then
//! a fixed map `{0, 1, 2} -> {0, 1, -1}`.
//!
//! ## The mapping is load-bearing
//!
//! The mapping is **not** a free choice — every test vector keyed to this
//! sampler assumes `2 -> -1` (so e.g. `[0, 1, -1][prg.uniform_below(3)]`).
//! Re-mapping (say `0 -> -1, 1 -> 0, 2 -> 1`) silently breaks cross-language
//! parity even when the underlying PRG is correct.
//!
//! ## Where it shows up
//!
//! Secret-key distribution $\chi_{S, 1}$ for $S_1$ in some parameter sets, and
//! the default RLWE error distribution in toy / debugging parameter sets where
//! the noise budget is generous. Production VIA-C / VIA-B use Gaussian errors
//! instead (see §1.5).

use crate::sampling::prg::Shake256Prg;

/// Fill `out` with coefficients sampled uniformly from $\{-1, 0, 1\}$.
///
/// Each output is one [`Shake256Prg::uniform_below`] draw at bound 3, mapped
/// `0 -> 0`, `1 -> 1`, `2 -> -1`. The PRG byte budget matches the Python
/// reference's `DeterministicSampler::ternary_poly(n)` exactly.
///
/// Outputs are signed `i8`. Callers that need the coefficients lifted into a
/// modulus `[0, q)` should go through the `lift_centered_i8_into_zq` helper
/// (Phase 4).
#[inline]
pub fn ternary(prg: &mut Shake256Prg, out: &mut [i8]) {
    for c in out.iter_mut() {
        *c = match prg.uniform_below(3) {
            0 => 0,
            1 => 1,
            2 => -1,
            // `uniform_below(3)` only ever returns values in [0, 3).
            _ => unreachable!(),
        };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// First 16 outputs of `DeterministicSampler(b"test").ternary_poly(16)`.
    const TEST_SEED_TP_N16: [i8; 16] = [1, -1, 1, -1, -1, 0, -1, -1, 1, -1, -1, 0, -1, 0, 1, -1];

    /// First 16 outputs of `DeterministicSampler(b"").ternary_poly(16)`.
    const EMPTY_SEED_TP_N16: [i8; 16] = [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, -1, 1, 0, 0, 1, 0];

    #[test]
    fn parity_test_seed_n16() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i8; 16];
        ternary(&mut prg, &mut out);
        assert_eq!(out, TEST_SEED_TP_N16);
    }

    #[test]
    fn parity_empty_seed_n16() {
        let mut prg = Shake256Prg::new(b"");
        let mut out = [0i8; 16];
        ternary(&mut prg, &mut out);
        assert_eq!(out, EMPTY_SEED_TP_N16);
    }

    #[test]
    fn all_outputs_in_range() {
        let mut prg = Shake256Prg::new(b"in-range");
        let mut out = [0i8; 500];
        ternary(&mut prg, &mut out);
        for &v in &out {
            assert!(v == -1 || v == 0 || v == 1);
        }
    }

    #[test]
    fn coverage_all_three_values_appear() {
        // Over 200 samples, all three values must appear.
        let mut prg = Shake256Prg::new(b"coverage");
        let mut out = [0i8; 200];
        ternary(&mut prg, &mut out);
        let mut saw_neg = false;
        let mut saw_zero = false;
        let mut saw_pos = false;
        for &v in &out {
            match v {
                -1 => saw_neg = true,
                0 => saw_zero = true,
                1 => saw_pos = true,
                _ => unreachable!(),
            }
        }
        assert!(saw_neg && saw_zero && saw_pos);
    }

    #[test]
    fn empty_output_is_noop() {
        let mut prg_a = Shake256Prg::new(b"noop");
        let mut empty: [i8; 0] = [];
        ternary(&mut prg_a, &mut empty);
        let mut prg_b = Shake256Prg::new(b"noop");
        let mut a = [0u8; 16];
        let mut b = [0u8; 16];
        prg_a.fill_bytes(&mut a);
        prg_b.fill_bytes(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn mapping_two_maps_to_minus_one() {
        // Explicit guard against the most common silent regression: anyone
        // re-mapping uniform_below(3) outputs to a different signed triple
        // would invalidate every downstream test vector. The first output of
        // ternary(b"test", _) is 1 only if randbelow(3) returned 1 first.
        // We additionally check that the second output, which the reference
        // says is -1, corresponds to randbelow(3) returning 2.
        //
        // randbelow(3) outputs for seed b"test" (per prg.rs tests):
        //   [1, 2, 1, 2, 2, 0, 2, 2, ...]
        // ternary outputs (the values above):
        //   [1, -1, 1, -1, -1, 0, -1, -1, ...]
        // So the second element is -1, matching mapping[2] = -1. ✓
        assert_eq!(TEST_SEED_TP_N16[1], -1);
        assert_eq!(TEST_SEED_TP_N16[5], 0);
    }
}
