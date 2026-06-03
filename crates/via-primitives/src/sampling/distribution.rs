//! Primitive §1.6 — the [`Distribution`] dispatcher.
//!
//! A typed bundling of `(which distribution, what parameter)` used wherever
//! the protocol needs to sample either a secret-key coefficient or an error
//! coefficient. Replaces the Python reference's
//! `(error_dist: str, error_sigma: Optional[float])` tuple-of-fields pattern
//! with a single value.
//!
//! ## The Layer-1 boundary stops here
//!
//! `Distribution` is **sampling-only**. It knows how to fill a `&mut [i64]`
//! with samples from one of three distributions — and nothing else. It does
//! **not** know about plaintexts, ciphertexts, encoding constants, or secret
//! keys. The composition `(sample error) + (encode plaintext) + (compute
//! ciphertext body)` lives at Layer 2 — typically as methods on the
//! relevant secret-key type. See `.docs/primitives.md` §2.1 / §2.2.
//!
//! Concretely: there is **no** `encrypt`-like method on this enum, by design.
//! Lifting samples into a modulus is also a separate concern, handled by the
//! [`lift`](crate::sampling::lift) module.
//!
//! ## Variants
//!
//! - [`Distribution::Ternary`] — uniform over $\{-1, 0, 1\}$.
//! - [`Distribution::BoundedUniform`] — uniform over $[-B, B]$.
//! - [`Distribution::Gaussian`] — discrete Gaussian with std-dev $\sigma$,
//!   centred on zero. (The only floating-point variant.)

use crate::sampling::gaussian::discrete_gaussian;
use crate::sampling::prg::Shake256Prg;

/// A typed dispatcher over the three sampling distributions used by every
/// key-sampling and error-sampling call site in the protocol.
///
/// See the module docs for the design rationale and Layer-1 boundary
/// guarantees.
#[derive(Copy, Clone, Debug, PartialEq)]
pub enum Distribution {
    /// Uniform over $\{-1, 0, 1\}$. Mirrors §1.3
    /// [`ternary`](crate::sampling::ternary::ternary).
    Ternary,
    /// Uniform over $[-B, B]$ (inclusive on both ends; $2B + 1$ values).
    /// Mirrors §1.4 [`bounded_uniform`](crate::sampling::bounded::bounded_uniform).
    BoundedUniform {
        /// Half-range $B$. $B = 0$ collapses the distribution to a single
        /// value `0` and consumes zero PRG bytes (rides on
        /// [`Shake256Prg::uniform_below`]'s `bound == 1` short-circuit).
        bound: u32,
    },
    /// Discrete Gaussian with standard deviation `sigma`, centred on zero.
    /// Mirrors §1.5 [`crate::sampling::gaussian::discrete_gaussian`].
    Gaussian {
        /// Standard deviation. `sigma == 0.0` produces all-zero output but
        /// **still advances the PRG** (matches the §1.5 contract). Negative
        /// `sigma` negates the output sample-for-sample versus
        /// `abs(sigma)` under the same seed.
        sigma: f64,
    },
}

impl Distribution {
    /// Fill `out` with samples drawn from this distribution under `prg`.
    ///
    /// PRG byte budget matches the underlying §1.3 / §1.4 / §1.5 sampler
    /// byte-for-byte. The `Ternary` and `BoundedUniform` arms inline the
    /// per-coefficient loop (avoiding an intermediate `i8 → i64` or
    /// `i32 → i64` widening pass); the `Gaussian` arm delegates directly
    /// to [`discrete_gaussian`].
    ///
    /// # Panics
    ///
    /// In debug builds, `Distribution::Gaussian { sigma }` panics if `sigma`
    /// is `NaN` or `±∞` (inherited from [`discrete_gaussian`]).
    ///
    /// # Example
    ///
    /// ```
    /// use via_primitives::sampling::{Distribution, Shake256Prg};
    ///
    /// let mut prg = Shake256Prg::new(b"dist-seed");
    /// let mut out = [0i64; 16];
    /// Distribution::Gaussian { sigma: 3.2 }.sample_into(&mut prg, &mut out);
    /// // For σ = 3.2 the chance of all-zero output is negligible.
    /// assert!(out.iter().any(|&v| v != 0));
    /// ```
    #[inline]
    pub fn sample_into(self, prg: &mut Shake256Prg, out: &mut [i64]) {
        match self {
            Distribution::Ternary => {
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
            Distribution::BoundedUniform { bound } => {
                // `bound: u32` keeps `2 * bound + 1` ≤ 2^33 - 1, safely
                // within `u64`. Match the §1.4 sampler's loop exactly.
                let range = 2u64 * (bound as u64) + 1;
                let shift = bound as i64;
                for c in out.iter_mut() {
                    *c = (prg.uniform_below(range) as i64) - shift;
                }
            }
            Distribution::Gaussian { sigma } => {
                discrete_gaussian(prg, sigma, out);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::zq::modulus::ConstModulus;
    use crate::sampling::bounded::bounded_uniform;
    use crate::sampling::gaussian::discrete_gaussian as gaussian_kernel;
    use crate::sampling::lift::lift_centered_i64_into_zq;
    use crate::sampling::ternary::ternary;

    // -----------------------------------------------------------------------
    // Bucket A — dispatch equivalence: Distribution::Variant.sample_into is
    // byte-for-byte equivalent to the corresponding §1.3 / §1.4 / §1.5
    // sampler, widened to i64.
    // -----------------------------------------------------------------------

    #[test]
    fn dispatch_ternary_matches_underlying_sampler() {
        let mut prg_a = Shake256Prg::new(b"dispatch-ternary");
        let mut prg_b = Shake256Prg::new(b"dispatch-ternary");
        let mut via_dispatch = [0i64; 32];
        let mut via_underlying_i8 = [0i8; 32];
        Distribution::Ternary.sample_into(&mut prg_a, &mut via_dispatch);
        ternary(&mut prg_b, &mut via_underlying_i8);
        for (d, u) in via_dispatch.iter().zip(via_underlying_i8.iter()) {
            assert_eq!(*d, *u as i64);
        }
    }

    #[test]
    fn dispatch_bounded_uniform_matches_underlying_sampler() {
        let mut prg_a = Shake256Prg::new(b"dispatch-bounded");
        let mut prg_b = Shake256Prg::new(b"dispatch-bounded");
        let mut via_dispatch = [0i64; 32];
        let mut via_underlying_i32 = [0i32; 32];
        Distribution::BoundedUniform { bound: 17 }.sample_into(&mut prg_a, &mut via_dispatch);
        bounded_uniform(&mut prg_b, 17, &mut via_underlying_i32);
        for (d, u) in via_dispatch.iter().zip(via_underlying_i32.iter()) {
            assert_eq!(*d, *u as i64);
        }
    }

    #[test]
    fn dispatch_gaussian_matches_underlying_sampler() {
        let mut prg_a = Shake256Prg::new(b"dispatch-gaussian");
        let mut prg_b = Shake256Prg::new(b"dispatch-gaussian");
        let mut via_dispatch = [0i64; 32];
        let mut via_underlying = [0i64; 32];
        Distribution::Gaussian { sigma: 3.2 }.sample_into(&mut prg_a, &mut via_dispatch);
        gaussian_kernel(&mut prg_b, 3.2, &mut via_underlying);
        assert_eq!(via_dispatch, via_underlying);
    }

    // -----------------------------------------------------------------------
    // Bucket B — determinism: same seed + same Distribution -> same output.
    // -----------------------------------------------------------------------

    #[test]
    fn determinism_ternary() {
        let mut a = Shake256Prg::new(b"det-tern");
        let mut b = Shake256Prg::new(b"det-tern");
        let mut out_a = [0i64; 16];
        let mut out_b = [0i64; 16];
        Distribution::Ternary.sample_into(&mut a, &mut out_a);
        Distribution::Ternary.sample_into(&mut b, &mut out_b);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn determinism_bounded_uniform() {
        let mut a = Shake256Prg::new(b"det-bnd");
        let mut b = Shake256Prg::new(b"det-bnd");
        let mut out_a = [0i64; 16];
        let mut out_b = [0i64; 16];
        Distribution::BoundedUniform { bound: 5 }.sample_into(&mut a, &mut out_a);
        Distribution::BoundedUniform { bound: 5 }.sample_into(&mut b, &mut out_b);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn determinism_gaussian() {
        let mut a = Shake256Prg::new(b"det-gauss");
        let mut b = Shake256Prg::new(b"det-gauss");
        let mut out_a = [0i64; 16];
        let mut out_b = [0i64; 16];
        Distribution::Gaussian { sigma: 26.0 }.sample_into(&mut a, &mut out_a);
        Distribution::Gaussian { sigma: 26.0 }.sample_into(&mut b, &mut out_b);
        assert_eq!(out_a, out_b);
    }

    // -----------------------------------------------------------------------
    // Bucket C — empty output is a no-op (PRG state preserved).
    // -----------------------------------------------------------------------

    fn assert_empty_noop(dist: Distribution, seed: &[u8]) {
        let mut a = Shake256Prg::new(seed);
        let mut empty: [i64; 0] = [];
        dist.sample_into(&mut a, &mut empty);
        let mut b = Shake256Prg::new(seed);
        let mut buf_a = [0u8; 16];
        let mut buf_b = [0u8; 16];
        a.fill_bytes(&mut buf_a);
        b.fill_bytes(&mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn empty_output_noop_ternary() {
        assert_empty_noop(Distribution::Ternary, b"noop-tern");
    }

    #[test]
    fn empty_output_noop_bounded_uniform() {
        assert_empty_noop(Distribution::BoundedUniform { bound: 7 }, b"noop-bnd");
    }

    #[test]
    fn empty_output_noop_gaussian() {
        assert_empty_noop(Distribution::Gaussian { sigma: 3.2 }, b"noop-gauss");
    }

    // -----------------------------------------------------------------------
    // Bucket D — edge cases that pin asymmetric behaviour established in
    // earlier phases.
    // -----------------------------------------------------------------------

    #[test]
    fn bounded_uniform_bound_zero_does_not_advance_prg() {
        // Pinned from Phase 2: bound=0 collapses to randbelow(1) which
        // short-circuits with NO byte consumption.
        let mut a = Shake256Prg::new(b"bnd0");
        let mut out = [0i64; 16];
        Distribution::BoundedUniform { bound: 0 }.sample_into(&mut a, &mut out);
        assert!(out.iter().all(|&v| v == 0));
        let mut b = Shake256Prg::new(b"bnd0");
        let mut buf_a = [0u8; 32];
        let mut buf_b = [0u8; 32];
        a.fill_bytes(&mut buf_a);
        b.fill_bytes(&mut buf_b);
        assert_eq!(buf_a, buf_b);
    }

    #[test]
    fn gaussian_sigma_zero_does_advance_prg() {
        // Pinned from Phase 3: sigma=0 gives all-zero output but the PRG
        // *does* advance — Box-Muller's randbelow draws fire regardless of σ.
        let mut a = Shake256Prg::new(b"gauss0");
        let mut out = [0i64; 16];
        Distribution::Gaussian { sigma: 0.0 }.sample_into(&mut a, &mut out);
        assert!(out.iter().all(|&v| v == 0));
        let mut b = Shake256Prg::new(b"gauss0");
        let mut buf_a = [0u8; 16];
        let mut buf_b = [0u8; 16];
        a.fill_bytes(&mut buf_a);
        b.fill_bytes(&mut buf_b);
        assert_ne!(buf_a, buf_b);
    }

    #[test]
    fn gaussian_negative_sigma_negates_outputs() {
        // Pinned from Phase 3: σ < 0 produces sample-for-sample negation.
        let mut a = Shake256Prg::new(b"negsig");
        let mut b = Shake256Prg::new(b"negsig");
        let mut pos = [0i64; 16];
        let mut neg = [0i64; 16];
        Distribution::Gaussian { sigma: 3.2 }.sample_into(&mut a, &mut pos);
        Distribution::Gaussian { sigma: -3.2 }.sample_into(&mut b, &mut neg);
        for (p, n) in pos.iter().zip(neg.iter()) {
            assert_eq!(*n, -*p);
        }
    }

    // -----------------------------------------------------------------------
    // Bucket G — end-to-end smoke test: sample via the dispatcher, lift into
    // a Zq modulus, observe the canonical Layer-1 → Layer-0 path Layer-2 will
    // consume.
    // -----------------------------------------------------------------------

    #[test]
    fn end_to_end_ternary_sample_then_lift() {
        let m = ConstModulus::<17>;
        let mut prg = Shake256Prg::new(b"e2e");
        let mut tmp = [0i64; 16];
        let mut out_zq = [0u64; 16];
        Distribution::Ternary.sample_into(&mut prg, &mut tmp);
        lift_centered_i64_into_zq(m, &tmp, &mut out_zq);
        // Every lifted value must be in {0, 1, 16} (the lifted images of
        // {0, 1, -1}).
        for &v in &out_zq {
            assert!(v == 0 || v == 1 || v == 16, "unexpected lifted value {}", v);
        }
        // And at least one of each must appear (with high probability over
        // n = 16). This pins that the dispatcher is actually producing all
        // three ternary values, not just one.
        assert!(out_zq.contains(&0));
        assert!(out_zq.contains(&1));
        assert!(out_zq.contains(&16));
    }
}
