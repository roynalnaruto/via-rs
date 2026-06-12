//! Key switching — `.docs/primitives.md` §2.4.
//!
//! Convert an RLWE ciphertext from one secret key to another:
//!
//! $$
//! \mathrm{RLWE}_S(M) \to \mathrm{RLWE}_{S'}(M).
//! $$
//!
//! The conversion uses a **key-switching key** (`ksk`), which is an
//! RLev encryption of the *source* secret key polynomial under the
//! *destination* secret key:
//!
//! $$
//! \mathrm{ksk} \;=\; \mathrm{RLev}_{S'}(S).
//! $$
//!
//! Under that arrangement, the algebra of [`RLevCiphertext::key_switch`]
//! makes the `A · S` term cancel exactly, leaving only `M + small_noise`
//! decryptable under `S'`.
//!
//! Phase 8 is the thinnest phase of Layer 2 — the body of `key_switch`
//! is three operator calls. Layer 3 (ring switching) and Layer 5
//! (LWE-to-RLWE cascade) both build on this primitive.

use crate::algebra::ring::{RingPoly, RingPolyEval};
use crate::sampling::distribution::Distribution;
use crate::sampling::prg::Shake256Prg;

use super::types::{RLWECiphertext, RLevCiphertext, RLevEval, SecretKey};

/// §2.4 — Generate a key-switching key `ksk = RLev_{dst_sk}(src_sk.poly)`:
/// an RLev encryption of the **source** secret-key polynomial under the
/// **destination** secret key.
///
/// Thin wrapper over [`SecretKey::encrypt_rlev`] — provided for
/// protocol-intent clarity (`gen_ksk(src, dst, …)` reads better than
/// `dst.encrypt_rlev(src.poly(), …)` at call sites where the
/// source-versus-destination relationship is part of the protocol
/// semantics).
///
/// Both keys must share the same ring degree `N` and modulus. The
/// gadget depth `L` is the const generic of the return type; `base` is
/// the gadget base `B`.
pub fn gen_ksk<const N: usize, R: RingPoly<N>, const L: usize>(
    src_sk: &SecretKey<N, R>,
    dst_sk: &SecretKey<N, R>,
    base: u64,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
) -> RLevCiphertext<N, R, L> {
    dst_sk.encrypt_rlev::<L>(src_sk.poly(), base, error_dist, prg)
}

impl<const N: usize, R: RingPoly<N>, const L: usize> RLevCiphertext<N, R, L> {
    /// §2.4 — Switch `ct` from the source secret key (under which `ct`
    /// was encrypted) to the destination secret key (the key that
    /// `self` was generated for via [`gen_ksk`]).
    ///
    /// `self` **must** be a key-switching key `RLev_{S'}(S)` —
    /// constructed via [`gen_ksk`]. Calling this on any other RLev
    /// (e.g., the message-encoding RLev produced by `encrypt_rlev` of
    /// an unrelated polynomial) silently produces garbage; there is
    /// no runtime check.
    ///
    /// # Algorithm
    ///
    /// Standard formula (`rlwe.py:427-433`):
    ///
    /// $$
    /// c' \;=\; (0, B) - (A \boxdot \mathrm{ksk}) \;=\; (-A', B - B')
    /// $$
    ///
    /// where `(A', B') = self.gadget_product(&ct.mask, base)`. The two
    /// terms combine so the `A · S` contribution cancels at decryption.
    ///
    /// # Noise growth
    ///
    /// Per-coefficient noise after key switching is bounded by
    /// `e + Σ d_i · e_i + δ · S` where `δ` is the gadget reconstruction
    /// error per coefficient and `e_i` are the per-level RLev errors.
    /// The dominant term `||S||_1 · (g_min/2 + L·B/4)` mirrors the
    /// external-product noise model (Phase 7) but lacks the `m1`
    /// multiplier — so the budget is `N` times less tight than for
    /// external product.
    pub fn key_switch(&self, ct: &RLWECiphertext<N, R>, base: u64) -> RLWECiphertext<N, R>
    where
        R: RingPolyEval<N>,
    {
        let product = self.gadget_product(&ct.mask, base);
        RLWECiphertext::new(-product.mask, ct.body - product.body)
    }
}

impl<const N: usize, R: RingPoly<N> + RingPolyEval<N>, const L: usize> RLevEval<N, R, L> {
    /// Eval-key variant of [`RLevCiphertext::key_switch`] (T7): `self` is the
    /// **pre-transformed** key-switching key, so the per-call `to_eval` of its
    /// samples is skipped. Bit-identical to the coefficient-form `key_switch`
    /// (the NTT is exact). Used by the eval-form conversion cascade.
    pub fn key_switch(&self, ct: &RLWECiphertext<N, R>, base: u64) -> RLWECiphertext<N, R> {
        let product = self.gadget_product(&ct.mask, base);
        RLWECiphertext::new(-product.mask, ct.body - product.body)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::zq::modulus::PowerOfTwoModulus;
    use crate::encryption::encode;

    type SinglePolyQ1024<const N: usize> = Poly<N, PowerOfTwoModulus<10>, Coefficient>;
    type SinglePolyP2<const N: usize> = Poly<N, PowerOfTwoModulus<1>, Coefficient>;
    type SinglePolyP256<const N: usize> = Poly<N, PowerOfTwoModulus<8>, Coefficient>;
    type SinglePolyViaCQ2<const N: usize> =
        Poly<N, crate::algebra::zq::modulus::paper::ViaCQ2, Coefficient>;
    type RnsPolyViaCQ1<const N: usize> =
        PolyRns<N, crate::algebra::rns::basis::paper::ViaCQ1Rns, Coefficient>;

    /// `decrypt(ksk.key_switch(&E_src(m), B), dst_sk).decode(p) == m`.
    /// Toy `(q=1024, p=2, B=2, L=10)` with ternary keys + errors and
    /// **distinct** src/dst seeds.
    #[test]
    fn key_switch_round_trip_at_q1024_p2_ternary() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        const L: usize = 10;
        let mut src_prg = Shake256Prg::new(b"sk-src-toy");
        let mut dst_prg = Shake256Prg::new(b"sk-dst-toy");
        let src_sk =
            SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut src_prg);
        let dst_sk =
            SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut dst_prg);

        let mut ksk_prg = Shake256Prg::new(b"ksk-toy");
        let ksk: RLevCiphertext<4, SinglePolyQ1024<4>, L> =
            gen_ksk(&src_sk, &dst_sk, 2, Distribution::Ternary, &mut ksk_prg);

        let m: SinglePolyP2<4> = Poly::new(p, [1, 0, 1, 1]);
        let mut enc_prg = Shake256Prg::new(b"enc-toy");
        let encoded_m: SinglePolyQ1024<4> = encode(&m, q);
        let rlwe_src = src_sk.encrypt(&encoded_m, Distribution::Ternary, &mut enc_prg);

        let rlwe_dst = ksk.key_switch(&rlwe_src, 2);
        let recovered: SinglePolyP2<4> = dst_sk.decrypt(&rlwe_dst, p);
        for i in 0..4 {
            assert_eq!(
                recovered.coeff(i).to_u64(),
                m.coeff(i).to_u64(),
                "key-switch round-trip diverged at coeff {i}"
            );
        }
    }

    /// Decrypting the switched ciphertext with the **source** secret
    /// key (instead of the destination) should produce a polynomial
    /// that differs from `m` in at least one coefficient — the
    /// `product.mask · (S_src − S_dst)` cross-term is large and
    /// random, so virtually any non-trivial `m` will differ from a
    /// random-looking decoding at multiple coefficients.
    ///
    /// Locks in the "key_switch decrypts under `S'`, not `S`"
    /// invariant; would also fail if `gen_ksk`'s src/dst arguments
    /// were silently swapped.
    #[test]
    fn key_switch_recovers_only_under_destination_key() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        const L: usize = 10;
        const N: usize = 8;
        let mut src_prg = Shake256Prg::new(b"sk-src-wrongkey");
        let mut dst_prg = Shake256Prg::new(b"sk-dst-wrongkey");
        let src_sk =
            SecretKey::<N, SinglePolyQ1024<N>>::keygen(q, Distribution::Ternary, &mut src_prg);
        let dst_sk =
            SecretKey::<N, SinglePolyQ1024<N>>::keygen(q, Distribution::Ternary, &mut dst_prg);

        let mut ksk_prg = Shake256Prg::new(b"ksk-wrongkey");
        let ksk: RLevCiphertext<N, SinglePolyQ1024<N>, L> =
            gen_ksk(&src_sk, &dst_sk, 2, Distribution::Ternary, &mut ksk_prg);

        let m: SinglePolyP2<N> = Poly::new(p, [1, 0, 1, 1, 0, 1, 1, 0]);
        let mut enc_prg = Shake256Prg::new(b"enc-wrongkey");
        let encoded_m: SinglePolyQ1024<N> = encode(&m, q);
        let rlwe_src = src_sk.encrypt(&encoded_m, Distribution::Ternary, &mut enc_prg);

        let rlwe_dst = ksk.key_switch(&rlwe_src, 2);

        // Positive: dst_sk recovers m.
        let recovered_correct: SinglePolyP2<N> = dst_sk.decrypt(&rlwe_dst, p);
        for i in 0..N {
            assert_eq!(
                recovered_correct.coeff(i).to_u64(),
                m.coeff(i).to_u64(),
                "destination decryption diverged at coeff {i}"
            );
        }

        // Negative: src_sk produces a polynomial that differs from m at
        // at least one coefficient. With N=8 and binary m, the
        // probability that all coefficients coincidentally match is
        // ~(1/2)^N = 1/256 — small enough that the test is reliable.
        let recovered_wrong: SinglePolyP2<N> = src_sk.decrypt(&rlwe_dst, p);
        let mut matches = 0usize;
        for i in 0..N {
            if recovered_wrong.coeff(i).to_u64() == m.coeff(i).to_u64() {
                matches += 1;
            }
        }
        assert!(
            matches < N,
            "src_sk should not recover m after key-switch to dst_sk, but all coefficients matched"
        );
    }

    /// Paper-class single-prime: VIA-C `q₂` ≈ 2³⁴, p=256, ring-switch
    /// gadget `(L=8, B=8)` with σ=4 Gaussian errors. Confirms the
    /// algebra at realistic noise levels.
    #[test]
    fn key_switch_at_via_c_q2_p256_ring_switch_gadget_gaussian() {
        let q = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        let p = PowerOfTwoModulus::<8>;
        const L: usize = 8;
        let base: u64 = 8;
        let mut src_prg = Shake256Prg::new(b"sk-src-vc-q2");
        let mut dst_prg = Shake256Prg::new(b"sk-dst-vc-q2");
        let src_sk =
            SecretKey::<16, SinglePolyViaCQ2<16>>::keygen(q, Distribution::Ternary, &mut src_prg);
        let dst_sk =
            SecretKey::<16, SinglePolyViaCQ2<16>>::keygen(q, Distribution::Ternary, &mut dst_prg);

        let mut ksk_prg = Shake256Prg::new(b"ksk-vc-q2");
        let ksk: RLevCiphertext<16, SinglePolyViaCQ2<16>, L> = gen_ksk(
            &src_sk,
            &dst_sk,
            base,
            Distribution::Gaussian { sigma: 4.0 },
            &mut ksk_prg,
        );

        let m_coeffs: [u64; 16] = [
            0, 1, 13, 31, 63, 127, 200, 255, 7, 42, 99, 137, 200, 250, 5, 17,
        ];
        let m: SinglePolyP256<16> = Poly::new(p, m_coeffs);
        let mut enc_prg = Shake256Prg::new(b"enc-vc-q2");
        let encoded_m: SinglePolyViaCQ2<16> = encode(&m, q);
        let rlwe_src = src_sk.encrypt(
            &encoded_m,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );

        let rlwe_dst = ksk.key_switch(&rlwe_src, base);
        let recovered: SinglePolyP256<16> = dst_sk.decrypt(&rlwe_dst, p);
        for (i, &expected) in m_coeffs.iter().enumerate() {
            assert_eq!(
                recovered.coeff(i).to_u64(),
                expected,
                "VIA-C q₂ key-switch diverged at coeff {i}"
            );
        }
    }

    /// **Paper-class RNS** at the gadget params Layer-3 ring-switching
    /// uses: VIA-C `q₁` (Q ≈ 2⁷⁵) at `(L=8, B=8)`, p=2, σ=4 Gaussian.
    /// End-to-end through the RNS gadget-product path inside the key
    /// switch.
    #[test]
    fn key_switch_at_via_c_q1_rns_p2_ring_switch_gadget_gaussian() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let p = PowerOfTwoModulus::<1>;
        const L: usize = 8;
        let base: u64 = 8;
        let mut src_prg = Shake256Prg::new(b"sk-src-vc-q1rns");
        let mut dst_prg = Shake256Prg::new(b"sk-dst-vc-q1rns");
        let src_sk =
            SecretKey::<16, RnsPolyViaCQ1<16>>::keygen(basis, Distribution::Ternary, &mut src_prg);
        let dst_sk =
            SecretKey::<16, RnsPolyViaCQ1<16>>::keygen(basis, Distribution::Ternary, &mut dst_prg);

        let mut ksk_prg = Shake256Prg::new(b"ksk-vc-q1rns");
        let ksk: RLevCiphertext<16, RnsPolyViaCQ1<16>, L> = gen_ksk(
            &src_sk,
            &dst_sk,
            base,
            Distribution::Gaussian { sigma: 4.0 },
            &mut ksk_prg,
        );

        let m_coeffs: [u64; 16] = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0];
        let m: SinglePolyP2<16> = Poly::new(p, m_coeffs);
        let mut enc_prg = Shake256Prg::new(b"enc-vc-q1rns");
        let encoded_m: RnsPolyViaCQ1<16> = encode(&m, basis);
        let rlwe_src = src_sk.encrypt(
            &encoded_m,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );

        let rlwe_dst = ksk.key_switch(&rlwe_src, base);
        let recovered: SinglePolyP2<16> = dst_sk.decrypt(&rlwe_dst, p);
        for (i, &expected) in m_coeffs.iter().enumerate() {
            assert_eq!(
                recovered.coeff(i).to_u64(),
                expected,
                "VIA-C q₁ RNS key-switch diverged at coeff {i}"
            );
        }
    }
}
