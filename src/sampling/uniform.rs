//! Primitive §1.2 — uniform sampler over $\mathbb{Z}_q$.
//!
//! Per-coefficient call to [`Shake256Prg::uniform_below`] using the modulus'
//! `q` as the rejection bound. Output is canonical `[0, q)`.
//!
//! Typical call sites: the $A$ mask polynomial in every fresh RLWE encryption,
//! plus the auxiliary mask used by some key-switching key constructions.

use crate::algebra::zq::modulus::Modulus;
use crate::sampling::prg::Shake256Prg;

/// Fill `out` with coefficients sampled uniformly from $\{0, 1, \ldots, q - 1\}$
/// under `modulus`.
///
/// Each output coefficient is one independent [`Shake256Prg::uniform_below`]
/// draw at `modulus.q()`. The PRG byte budget matches the Python reference's
/// `DeterministicSampler::uniform_poly(n, q)` exactly.
#[inline]
pub fn uniform_zq<M: Modulus>(modulus: M, prg: &mut Shake256Prg, out: &mut [u64]) {
    let q = modulus.q();
    for c in out.iter_mut() {
        *c = prg.uniform_below(q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::zq::modulus::{ConstModulus, DynModulus};

    /// First 16 outputs of `DeterministicSampler(b"test").uniform_poly(16, 17)`.
    const TEST_SEED_UP_Q17_N16: [u64; 16] =
        [1, 13, 11, 10, 10, 16, 10, 15, 5, 9, 14, 9, 5, 1, 12, 0];

    /// First 8 outputs of `DeterministicSampler(b"test").uniform_poly(8, 8_380_417)`
    /// — paper VIA-C $q_3$.
    const TEST_SEED_UP_Q3_N8: [u64; 8] = [
        7_199_425, 3_980_854, 662_059, 8_268_469, 6_056_624, 7_339_729, 6_004_453, 966_153,
    ];

    /// First 6 outputs of `DeterministicSampler(b"test").uniform_poly(6, 137_438_822_401)`
    /// — paper VIA-C $q_1$ smaller factor (≈ $2^{37}$, exercises the 5-byte path).
    const TEST_SEED_UP_Q1_FACTOR_N6: [u64; 6] = [
        129_770_576_577,
        92_511_284_156,
        122_049_035_818,
        132_706_729_681,
        38_902_040_923,
        100_480_070_262,
    ];

    #[test]
    fn parity_q17_const_modulus() {
        let m = ConstModulus::<17>;
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0u64; 16];
        uniform_zq(m, &mut prg, &mut out);
        assert_eq!(out, TEST_SEED_UP_Q17_N16);
    }

    #[test]
    fn parity_q17_dyn_modulus_matches_const() {
        // Cross-implementation sanity: DynModulus must agree with ConstModulus
        // for the same q.
        let m = DynModulus::new(17);
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0u64; 16];
        uniform_zq(m, &mut prg, &mut out);
        assert_eq!(out, TEST_SEED_UP_Q17_N16);
    }

    #[test]
    fn parity_q3_const_modulus() {
        let m = ConstModulus::<8_380_417>;
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0u64; 8];
        uniform_zq(m, &mut prg, &mut out);
        assert_eq!(out, TEST_SEED_UP_Q3_N8);
    }

    #[test]
    fn parity_q1_factor_dyn_modulus() {
        // Paper q1 factors exceed the i32 range but fit comfortably in u64;
        // only DynModulus carries them at runtime.
        let m = DynModulus::new(137_438_822_401);
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0u64; 6];
        uniform_zq(m, &mut prg, &mut out);
        assert_eq!(out, TEST_SEED_UP_Q1_FACTOR_N6);
    }

    #[test]
    fn all_outputs_in_range() {
        let q = 8_380_417u64;
        let m = ConstModulus::<8_380_417>;
        let mut prg = Shake256Prg::new(b"in-range");
        let mut out = [0u64; 1000];
        uniform_zq(m, &mut prg, &mut out);
        for &v in &out {
            assert!(v < q);
        }
    }

    #[test]
    fn empty_output_is_noop() {
        // No coefficients to fill ⇒ no PRG state change.
        let m = ConstModulus::<17>;
        let mut prg_a = Shake256Prg::new(b"noop");
        let mut empty: [u64; 0] = [];
        uniform_zq(m, &mut prg_a, &mut empty);

        let mut prg_b = Shake256Prg::new(b"noop");
        // Both PRGs should produce identical subsequent bytes.
        let mut a = [0u8; 32];
        let mut b = [0u8; 32];
        prg_a.fill_bytes(&mut a);
        prg_b.fill_bytes(&mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn coverage_at_small_q() {
        // Over many draws at q = 17, every residue must appear at least once.
        let m = ConstModulus::<17>;
        let mut prg = Shake256Prg::new(b"coverage");
        let mut out = [0u64; 1000];
        uniform_zq(m, &mut prg, &mut out);
        let mut seen = [false; 17];
        for &v in &out {
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&b| b), "every residue mod 17 must appear");
    }
}
