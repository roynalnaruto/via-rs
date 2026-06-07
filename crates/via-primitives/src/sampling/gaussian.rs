//! Primitive §1.5 — discrete Gaussian sampler over $\mathbb{Z}$ via the
//! Box-Muller transform.
//!
//! Per sample, draws two $u_{32}$-bounded uniforms $u_1, u_2$ from the PRG
//! and applies
//!
//! $$
//! z \;=\; \sqrt{-2 \,\ln u_1} \,\cdot\, \cos(2 \pi u_2),
//! $$
//!
//! returning $\lfloor z \cdot \sigma \rceil$ (round-half-to-even — matches
//! Python's built-in `round()`).
//!
//! ## The only floating-point primitive in via-rs
//!
//! Every other layer is integer-only. This module pays the f64 cost because
//! the Box-Muller transform is inherently transcendental. All operations are
//! routed through [`libm`] (`log`, `sqrt`, `cos`, `rint`) so the output is
//! deterministic and platform-independent.
//!
//! ## Byte parity with the Python reference
//!
//! The PRG byte budget is fixed at exactly **two** [`Shake256Prg::uniform_below`]
//! calls per emitted sample (each at bound $2^{32}$), regardless of $\sigma$.
//! The Box-Muller sin pair is **not** cached — caching would halve the byte
//! budget and break cross-language test-vector parity. The exact constants are:
//!
//! | Quantity | Value |
//! |---|---|
//! | $u_1$ | `(randbelow(2^32) + 1) / (2^32 + 1)` |
//! | $u_2$ | `randbelow(2^32) / 2^32` |
//! | Rounding | `libm::rint(z * sigma) as i64` (round-half-to-even) |
//!
//! Both `+1`s in $u_1$ are load-bearing: the numerator `+1` pushes the
//! argument of $\ln$ strictly above zero; the denominator `+1` is the
//! asymmetry the reference's `(0, 1]` distribution requires.

use crate::sampling::prg::Shake256Prg;
use core::f64::consts::PI;

/// $2^{32}$ as a `u64` — the `randbelow` bound for both $u_1$ and $u_2$.
const TWO_POW_32: u64 = 1u64 << 32;

/// $2^{32} + 1$ as an exact `f64` — used as the denominator for $u_1$
/// only. The asymmetry with [`TWO_POW_32_F64`] is intentional and matches
/// the Python reference.
const TWO_POW_32_PLUS_1_F64: f64 = 4_294_967_297.0;

/// $2^{32}$ as an exact `f64` — used as the denominator for $u_2$.
const TWO_POW_32_F64: f64 = 4_294_967_296.0;

/// Fill `out` with coefficients sampled from a discrete Gaussian with
/// standard deviation `sigma`, centred on zero.
///
/// Each output sample consumes exactly two [`Shake256Prg::uniform_below`]
/// calls at bound $2^{32}$. Rejection inside `uniform_below` may consume
/// additional bytes; the per-sample expected budget is ~20 bytes.
///
/// The output is `round(z * sigma)` where `round` is round-half-to-even
/// (banker's rounding), implemented via [`libm::rint`]. This matches
/// Python's built-in `round()` for IEEE 754 doubles.
///
/// # Edge cases
///
/// - `sigma == 0.0` → every output is `0`, but the PRG **still advances**
///   (the two `randbelow` draws per sample fire unconditionally).
/// - `sigma < 0.0` → outputs are the negation of `sigma.abs()`'s sequence
///   on the same seed (because Box-Muller's $z$ is symmetric around zero and
///   banker's rounding is symmetric).
/// - `out.is_empty()` → no PRG state change; this is the only short-circuit.
///
/// # Panics
///
/// In debug builds, panics if `sigma` is `NaN` or `±∞`. Release builds
/// match the Python reference and accept any `f64`.
///
/// # Example
///
/// ```
/// use via_primitives::sampling::gaussian::discrete_gaussian;
/// use via_primitives::sampling::prg::Shake256Prg;
///
/// let mut prg = Shake256Prg::new(b"my-seed");
/// let mut out = [0i64; 16];
/// discrete_gaussian(&mut prg, 3.2, &mut out);
/// // For σ = 3.2 the chance of all-zero output is negligible.
/// assert!(out.iter().any(|&v| v != 0));
/// ```
#[inline]
pub fn discrete_gaussian(prg: &mut Shake256Prg, sigma: f64, out: &mut [i64]) {
    debug_assert!(sigma.is_finite(), "sigma must be finite, got {}", sigma);
    for c in out.iter_mut() {
        let rb1 = prg.uniform_below(TWO_POW_32);
        let rb2 = prg.uniform_below(TWO_POW_32);
        // u1 ∈ (0, 1]: the +1 offset pushes log(u1) strictly finite.
        let u1 = (rb1 as f64 + 1.0) / TWO_POW_32_PLUS_1_F64;
        // u2 ∈ [0, 1).
        let u2 = (rb2 as f64) / TWO_POW_32_F64;
        let z = libm::sqrt(-2.0 * libm::log(u1)) * libm::cos(2.0 * PI * u2);
        *c = libm::rint(z * sigma) as i64;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Stage A — banker's-rounding regression guard.
    // -----------------------------------------------------------------------

    #[test]
    fn libm_rint_is_banker_rounding() {
        // Half-integer inputs should round to the nearest even integer.
        // If libm ever switches to round-away-from-zero we want this to fail
        // before the parity vectors do — clearer diagnostic.
        assert_eq!(libm::rint(0.5), 0.0);
        assert_eq!(libm::rint(-0.5), 0.0);
        assert_eq!(libm::rint(1.5), 2.0);
        assert_eq!(libm::rint(2.5), 2.0);
        assert_eq!(libm::rint(-1.5), -2.0);
        assert_eq!(libm::rint(-2.5), -2.0);
        // Non-half-integer values should round to the nearest, unambiguously.
        assert_eq!(libm::rint(0.4), 0.0);
        assert_eq!(libm::rint(0.6), 1.0);
        assert_eq!(libm::rint(-0.4), 0.0);
        assert_eq!(libm::rint(-0.6), -1.0);
    }

    // -----------------------------------------------------------------------
    // Stage B — determinism.
    // -----------------------------------------------------------------------

    #[test]
    fn same_seed_produces_identical_samples() {
        let mut a = Shake256Prg::new(b"determinism");
        let mut b = Shake256Prg::new(b"determinism");
        let mut out_a = [0i64; 32];
        let mut out_b = [0i64; 32];
        discrete_gaussian(&mut a, 3.2, &mut out_a);
        discrete_gaussian(&mut b, 3.2, &mut out_b);
        assert_eq!(out_a, out_b);
    }

    // -----------------------------------------------------------------------
    // Stage C — empty output is a no-op.
    // -----------------------------------------------------------------------

    #[test]
    fn empty_output_does_not_advance_prg() {
        let mut a = Shake256Prg::new(b"noop");
        let mut empty: [i64; 0] = [];
        discrete_gaussian(&mut a, 3.2, &mut empty);
        let mut b = Shake256Prg::new(b"noop");
        let mut buf_a = [0u8; 32];
        let mut buf_b = [0u8; 32];
        a.fill_bytes(&mut buf_a);
        b.fill_bytes(&mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    // -----------------------------------------------------------------------
    // Stage D — byte-exact parity vectors. Generated by running the actual
    // Python reference `DeterministicSampler::discrete_gaussian_poly(n, σ)`
    // on Python 3.13 (/opt/homebrew/bin/python3).
    // -----------------------------------------------------------------------

    /// `DeterministicSampler(b"test").discrete_gaussian_poly(16, 1.0)`.
    const PARITY_TEST_SEED_SIGMA_1_N16: [i64; 16] =
        [-2, 0, -1, -1, 0, 1, -2, 1, 0, 0, 2, 0, -1, 0, 0, 1];

    /// `DeterministicSampler(b"test").discrete_gaussian_poly(16, 3.2)`.
    const PARITY_TEST_SEED_SIGMA_3P2_N16: [i64; 16] =
        [-5, 0, -2, -3, 0, 2, -6, 3, 0, 0, 6, 0, -3, 0, 0, 4];

    /// `DeterministicSampler(b"test").discrete_gaussian_poly(16, 26.0)`.
    const PARITY_TEST_SEED_SIGMA_26_N16: [i64; 16] = [
        -40, 2, -19, -24, -2, 13, -46, 24, 1, 0, 49, -3, -24, -3, 3, 31,
    ];

    /// `DeterministicSampler(b"test").discrete_gaussian_poly(16, 32.0)`.
    const PARITY_TEST_SEED_SIGMA_32_N16: [i64; 16] = [
        -49, 3, -24, -30, -2, 16, -57, 30, 1, 0, 60, -4, -29, -4, 4, 38,
    ];

    /// `DeterministicSampler(b"test").discrete_gaussian_poly(16, 1024.0)`.
    const PARITY_TEST_SEED_SIGMA_1024_N16: [i64; 16] = [
        -1556, 93, -766, -962, -72, 522, -1810, 954, 35, 8, 1936, -128, -930, -131, 119, 1214,
    ];

    /// `DeterministicSampler(b"").discrete_gaussian_poly(8, 1.0)`.
    const PARITY_EMPTY_SEED_SIGMA_1_N8: [i64; 8] = [1, -1, 0, 1, -1, 1, 0, 1];

    #[test]
    fn parity_test_seed_sigma_1_n16() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i64; 16];
        discrete_gaussian(&mut prg, 1.0, &mut out);
        assert_eq!(out, PARITY_TEST_SEED_SIGMA_1_N16);
    }

    #[test]
    fn parity_test_seed_sigma_3p2_n16() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i64; 16];
        discrete_gaussian(&mut prg, 3.2, &mut out);
        assert_eq!(out, PARITY_TEST_SEED_SIGMA_3P2_N16);
    }

    #[test]
    fn parity_test_seed_sigma_26_n16() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i64; 16];
        discrete_gaussian(&mut prg, 26.0, &mut out);
        assert_eq!(out, PARITY_TEST_SEED_SIGMA_26_N16);
    }

    #[test]
    fn parity_test_seed_sigma_32_n16() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i64; 16];
        discrete_gaussian(&mut prg, 32.0, &mut out);
        assert_eq!(out, PARITY_TEST_SEED_SIGMA_32_N16);
    }

    #[test]
    fn parity_test_seed_sigma_1024_n16() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0i64; 16];
        discrete_gaussian(&mut prg, 1024.0, &mut out);
        assert_eq!(out, PARITY_TEST_SEED_SIGMA_1024_N16);
    }

    #[test]
    fn parity_empty_seed_sigma_1_n8() {
        let mut prg = Shake256Prg::new(b"");
        let mut out = [0i64; 8];
        discrete_gaussian(&mut prg, 1.0, &mut out);
        assert_eq!(out, PARITY_EMPTY_SEED_SIGMA_1_N8);
    }

    // -----------------------------------------------------------------------
    // Stage E — byte-budget invariance.
    // -----------------------------------------------------------------------

    #[test]
    fn byte_budget_is_sigma_independent() {
        // Two PRGs seeded identically; sample N values at very different σ.
        // Outputs differ, but the PRG byte position must agree, so the
        // trailing bytes drawn afterwards must be identical.
        let mut a = Shake256Prg::new(b"budget");
        let mut b = Shake256Prg::new(b"budget");
        let mut out_a = [0i64; 16];
        let mut out_b = [0i64; 16];
        discrete_gaussian(&mut a, 1.0, &mut out_a);
        discrete_gaussian(&mut b, 1024.0, &mut out_b);
        let mut tail_a = [0u8; 64];
        let mut tail_b = [0u8; 64];
        a.fill_bytes(&mut tail_a);
        b.fill_bytes(&mut tail_b);
        assert_eq!(tail_a, tail_b);
    }

    // -----------------------------------------------------------------------
    // Stage F — edge cases.
    // -----------------------------------------------------------------------

    #[test]
    fn sigma_zero_returns_all_zero_but_advances_prg() {
        let mut a = Shake256Prg::new(b"sigma0");
        let mut out = [0i64; 16];
        discrete_gaussian(&mut a, 0.0, &mut out);
        assert!(out.iter().all(|&v| v == 0));
        // PRG must have advanced. A fresh PRG with the same seed would produce
        // a different next-byte stream than `a` at this point, since `a` has
        // already consumed ~160 bytes worth of randbelow trajectory.
        let mut b = Shake256Prg::new(b"sigma0");
        let mut t_a = [0u8; 16];
        let mut t_b = [0u8; 16];
        a.fill_bytes(&mut t_a);
        b.fill_bytes(&mut t_b);
        assert_ne!(t_a, t_b);
    }

    #[test]
    fn negative_sigma_negates_outputs() {
        let mut a = Shake256Prg::new(b"negsig");
        let mut b = Shake256Prg::new(b"negsig");
        let mut pos = [0i64; 16];
        let mut neg = [0i64; 16];
        discrete_gaussian(&mut a, 3.2, &mut pos);
        discrete_gaussian(&mut b, -3.2, &mut neg);
        for (p, n) in pos.iter().zip(neg.iter()) {
            assert_eq!(*n, -*p, "negative sigma must negate sample-for-sample");
        }
    }

    #[test]
    fn sigma_very_small_outputs_mostly_zero() {
        let mut prg = Shake256Prg::new(b"small");
        let mut out = [0i64; 1000];
        discrete_gaussian(&mut prg, 0.001, &mut out);
        let zeros = out.iter().filter(|&&v| v == 0).count();
        assert!(zeros >= 990, "expected ≥ 990 zero samples, got {}", zeros);
    }

    #[test]
    fn sigma_extreme_does_not_panic() {
        let mut prg = Shake256Prg::new(b"extreme");
        let mut out = [0i64; 16];
        discrete_gaussian(&mut prg, 1.0e6, &mut out);
        // No saturation expected — sigma = 1e6 with Box-Muller's typical z
        // ∈ [-6, 6] gives outputs in roughly [-6e6, 6e6], well below 2^40.
        for &v in &out {
            assert!(v.unsigned_abs() < (1u64 << 40));
        }
    }

    // -----------------------------------------------------------------------
    // Stage G — distribution sanity.
    // -----------------------------------------------------------------------

    #[test]
    fn empirical_mean_near_zero() {
        let mut prg = Shake256Prg::new(b"mean");
        let n = 10_000usize;
        let sigma = 32.0;
        let mut buf = [0i64; 10_000];
        discrete_gaussian(&mut prg, sigma, &mut buf);
        let sum: i128 = buf.iter().map(|&v| v as i128).sum();
        let mean = sum as f64 / n as f64;
        // For σ=32 at n=10_000, std-err of the mean is σ/√n = 0.32. A 5σ-of-
        // -mean tolerance (1.6) is conservative but tight enough to catch
        // distribution drift.
        assert!(
            mean.abs() < 1.6,
            "|mean| = {} (expected < 1.6 ≈ 0.05·σ)",
            mean
        );
    }

    #[test]
    fn empirical_stddev_near_sigma() {
        let mut prg = Shake256Prg::new(b"stddev");
        let n = 10_000usize;
        let mut buf = [0i64; 10_000];
        let sigma = 32.0;
        discrete_gaussian(&mut prg, sigma, &mut buf);
        let mean = buf.iter().map(|&v| v as f64).sum::<f64>() / n as f64;
        let var = buf
            .iter()
            .map(|&v| {
                let d = v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let sd = libm::sqrt(var);
        assert!(
            (sd - sigma).abs() < 0.05 * sigma,
            "empirical σ = {}, expected ≈ {}",
            sd,
            sigma
        );
    }

    #[test]
    fn tail_within_three_sigma() {
        // Mirror of the Python reference's TestDiscreteGaussian shape test.
        let mut prg = Shake256Prg::new(b"tail");
        let n = 1000usize;
        let sigma = 3.2;
        let mut buf = [0i64; 1000];
        discrete_gaussian(&mut prg, sigma, &mut buf);
        let bound = 3.0 * sigma;
        let inside = buf.iter().filter(|&&v| (v as f64).abs() <= bound).count();
        assert!(
            inside as f64 >= 0.95 * n as f64,
            "{}/{} samples within 3σ",
            inside,
            n
        );
    }
}
