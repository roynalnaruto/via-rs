//! RGSW encryption — `.docs/primitives.md` §2.1.
//!
//! An RGSW ciphertext encrypts a message $M$ as a **pair** of RLevs:
//! `RLev_S(-S · M)` and `RLev_S(M)`. This pair is what the external
//! product (Phase 7) consumes to perform homomorphic multiplication
//! against an RLWE ciphertext:
//!
//! $$
//! \mathrm{RGSW}_S(M_1) \boxtimes \mathrm{RLWE}_S(M_2) \to \mathrm{RLWE}_S(M_1 \cdot M_2)
//! $$
//!
//! The two halves can use **different gadget parameters** — paper
//! Tables 5-6 list distinct `(L, B)` for many call sites. The Rust API
//! takes `(base_neg_s_m, base_m)` separately; the const-generic depths
//! `L1`, `L2` are part of the return type.
//!
//! Phase 7's `external_product` will land in this same file.

use crate::algebra::ring::RingPoly;
use crate::sampling::distribution::Distribution;
use crate::sampling::prg::Shake256Prg;

use super::types::{RGSWCiphertext, RLWECiphertext, SecretKey};

impl<const N: usize, R: RingPoly<N>> SecretKey<N, R> {
    /// §2.1 — encrypt `message` as an RGSW: pair of RLevs
    /// `(RLev_S(-S · M), RLev_S(M))`.
    ///
    /// Different gadget bases per half are supported (`base_neg_s_m`
    /// for the `-S · M` half, `base_m` for the `M` half), matching
    /// paper Tables 5-6 where DMux/CMux halves use distinct `(L, B)`.
    /// Pass the same value for both to recover the Python reference's
    /// symmetric behaviour.
    ///
    /// # PRG consumption order
    ///
    /// The `-S · M` half is encrypted **first** (consuming `L1`
    /// per-level RLWE draws), then the `M` half (`L2` draws). Matches
    /// `rlwe.py:328-336`; flipping the order would silently break
    /// Python parity for every RGSW test vector.
    ///
    /// # Constant-time
    ///
    /// The `−(self.poly · *message)` step is constant-time over both
    /// operands: the schoolbook negacyclic mul forwards to
    /// Barrett-reduced `Modulus::mul` (data-independent), and
    /// componentwise negation via `Neg` is branchless. No `_ct`
    /// centred-lift variant is needed here — centred lifts of secret
    /// key material only enter the picture at Phase 8 (secret-key
    /// rekeying).
    pub fn encrypt_rgsw<const L1: usize, const L2: usize>(
        &self,
        message: &R,
        base_neg_s_m: u64,
        base_m: u64,
        error_dist: Distribution,
        prg: &mut Shake256Prg,
    ) -> RGSWCiphertext<N, R, L1, L2> {
        // Compute `-S · M` using the trait's `Mul<R>` and `Neg`.
        // `R: Copy` (via the supertrait bound) so dereferencing is free.
        let neg_s_m: R = -(self.poly * *message);
        // The `-S·M` half first (matches Python ordering).
        let rlev_neg_s_m = self.encrypt_rlev::<L1>(&neg_s_m, base_neg_s_m, error_dist, prg);
        let rlev_m = self.encrypt_rlev::<L2>(message, base_m, error_dist, prg);
        RGSWCiphertext::new(rlev_neg_s_m, rlev_m)
    }
}

// ---------------------------------------------------------------------------
// External product — §2.4.
//
// `RGSW_S(M₁) ⊠ RLWE_S(M₂) → RLWE_S(M₁ · M₂)` — the homomorphic
// multiplication primitive that DMux, CMux, CRot, and key switching
// all compose.
// ---------------------------------------------------------------------------

impl<const N: usize, R: RingPoly<N>, const L1: usize, const L2: usize>
    RGSWCiphertext<N, R, L1, L2>
{
    /// §2.4 — External product: $\mathrm{RGSW}_S(M_1) \boxtimes
    /// \mathrm{RLWE}_S(M_2) \to \mathrm{RLWE}_S(M_1 \cdot M_2)$.
    ///
    /// # Algorithm
    ///
    /// Two gadget products against the two RGSW halves, summed:
    ///
    /// $$
    /// C \boxtimes c = A \boxdot \hat c_1 + B \boxdot \hat c_2
    /// $$
    ///
    /// where `(A, B)` is the input `RLWECiphertext` and `(ĉ_1, ĉ_2)
    /// = (self.neg_s_m, self.m)`. The two gadget products combine to
    /// cancel the `-S · M_1` contribution from the first half, leaving
    /// only the encrypted `M_1 · M_2` plus noise.
    ///
    /// # Bases
    ///
    /// The two halves can use **different gadget bases**, matching the
    /// per-half convention from [`SecretKey::encrypt_rgsw`]. Pass the
    /// same value for both to recover the Python reference's symmetric
    /// behaviour. The depths `L1`, `L2` are implied by the type.
    ///
    /// # Noise growth
    ///
    /// Each gadget product contributes `~||·||_1 · σ_e + L · B / 4`
    /// noise; the two contributions sum. With small-norm `M_1` (the
    /// typical case — encrypted bits or low-weight selector
    /// polynomials in the PIR protocol) the noise stays well inside
    /// the decryption budget at paper parameters.
    pub fn external_product(
        &self,
        ct: &RLWECiphertext<N, R>,
        base_neg_s_m: u64,
        base_m: u64,
    ) -> RLWECiphertext<N, R> {
        let ct1 = self.neg_s_m.gadget_product(&ct.mask, base_neg_s_m);
        let ct2 = self.m.gadget_product(&ct.body, base_m);
        ct1 + ct2 // Phase-4 `Add for RLWECiphertext`.
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::zq::modulus::PowerOfTwoModulus;
    use crate::encryption::gadget::gadget_vector_values;

    type SinglePolyQ1024<const N: usize> = Poly<N, PowerOfTwoModulus<10>, Coefficient>;

    fn const_term_poly<const N: usize, R: RingPoly<N>>(modulus: R::Modulus, value: u128) -> R {
        let mut coeffs = [0u128; N];
        coeffs[0] = value;
        R::from_u128_coeffs(modulus, &coeffs)
    }

    /// RGSW's `m` half is exactly `encrypt_rlev(M, sk, base_m, …)`:
    /// each `samples[i]` decrypts to `g_i · M + small noise`.
    #[test]
    fn encrypt_rgsw_m_half_decrypts_to_g_i_times_m() {
        let q = PowerOfTwoModulus::<10>;
        const L1: usize = 4;
        const L2: usize = 4;
        let mut sk_prg = Shake256Prg::new(b"sk-rgsw-m");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-rgsw-m");

        let message: SinglePolyQ1024<4> = Poly::new(q, [1, 0, 0, 1]);
        let rgsw = sk.encrypt_rgsw::<L1, L2>(&message, 2, 2, Distribution::Ternary, &mut enc_prg);

        let g_values = gadget_vector_values::<4, SinglePolyQ1024<4>, L2>(q, 2);
        for (i, sample) in rgsw.m.samples.iter().enumerate() {
            let g_poly = const_term_poly::<4, SinglePolyQ1024<4>>(q, g_values[i]);
            let expected = message * g_poly;
            let recovered = sk.decrypt_raw(sample);
            let diff = recovered - expected;
            let mut centred = [0i64; 4];
            diff.to_centered_coeffs(&mut centred);
            for (j, &c) in centred.iter().enumerate() {
                assert!(c.abs() <= 4, "m-half sample {i} coeff {j}: noise {c}");
            }
        }
    }

    /// RGSW's `neg_s_m` half encrypts `-S · M`: each `samples[i]`
    /// decrypts to `g_i · (-S · M) + small noise`. Validates the sign
    /// and the `S · M` multiplication.
    #[test]
    fn encrypt_rgsw_neg_s_m_half_decrypts_to_g_i_times_neg_s_m() {
        let q = PowerOfTwoModulus::<10>;
        const L1: usize = 4;
        const L2: usize = 4;
        let mut sk_prg = Shake256Prg::new(b"sk-rgsw-neg");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-rgsw-neg");

        let message: SinglePolyQ1024<4> = Poly::new(q, [1, 0, 0, 1]);
        let rgsw = sk.encrypt_rgsw::<L1, L2>(&message, 2, 2, Distribution::Ternary, &mut enc_prg);

        // Expected encrypted plaintext for each sample: g_i · (-S · M).
        let neg_s_m: SinglePolyQ1024<4> = -(*sk.poly() * message);

        let g_values = gadget_vector_values::<4, SinglePolyQ1024<4>, L1>(q, 2);
        for (i, sample) in rgsw.neg_s_m.samples.iter().enumerate() {
            let g_poly = const_term_poly::<4, SinglePolyQ1024<4>>(q, g_values[i]);
            let expected = neg_s_m * g_poly;
            let recovered = sk.decrypt_raw(sample);
            let diff = recovered - expected;
            let mut centred = [0i64; 4];
            diff.to_centered_coeffs(&mut centred);
            for (j, &c) in centred.iter().enumerate() {
                assert!(c.abs() <= 4, "neg-s-m-half sample {i} coeff {j}: noise {c}");
            }
        }
    }

    /// Build an RGSW at `(L1=2, B1=24)` and `(L2=4, B2=16)` —
    /// asymmetric in both depth and base — and verify both halves
    /// decrypt correctly. Catches a copy-paste bug where one base or
    /// depth accidentally feeds both halves.
    #[test]
    fn encrypt_rgsw_with_distinct_l1_l2_and_distinct_bases() {
        let q = PowerOfTwoModulus::<10>;
        const L1: usize = 2;
        const L2: usize = 4;
        let base_neg_s_m: u64 = 24;
        let base_m: u64 = 16;
        let mut sk_prg = Shake256Prg::new(b"sk-rgsw-asym");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-rgsw-asym");

        let message: SinglePolyQ1024<4> = Poly::new(q, [2, 0, 1, 0]);
        let rgsw = sk.encrypt_rgsw::<L1, L2>(
            &message,
            base_neg_s_m,
            base_m,
            Distribution::Ternary,
            &mut enc_prg,
        );

        // Verify lengths come from the type parameters, not from a
        // mis-shared value.
        assert_eq!(rgsw.neg_s_m.samples.len(), L1);
        assert_eq!(rgsw.m.samples.len(), L2);

        // Verify the `m` half uses `base_m`, not `base_neg_s_m`. The
        // gadget entries differ between bases, so confirming
        // `samples[0]` decrypts to `g_0(base_m) · M` rules out the
        // copy-paste failure mode.
        let g_m = gadget_vector_values::<4, SinglePolyQ1024<4>, L2>(q, base_m);
        let g_m_0 = const_term_poly::<4, SinglePolyQ1024<4>>(q, g_m[0]);
        let expected_m_0 = message * g_m_0;
        let recovered_m_0 = sk.decrypt_raw(&rgsw.m.samples[0]);
        let diff_m_0 = recovered_m_0 - expected_m_0;
        let mut centred = [0i64; 4];
        diff_m_0.to_centered_coeffs(&mut centred);
        for (j, &c) in centred.iter().enumerate() {
            assert!(
                c.abs() <= 4,
                "m-half base_m mismatch at coeff {j}: noise {c}"
            );
        }

        // Verify the `neg_s_m` half uses `base_neg_s_m`.
        let g_ns = gadget_vector_values::<4, SinglePolyQ1024<4>, L1>(q, base_neg_s_m);
        let g_ns_0 = const_term_poly::<4, SinglePolyQ1024<4>>(q, g_ns[0]);
        let neg_s_m_poly: SinglePolyQ1024<4> = -(*sk.poly() * message);
        let expected_ns_0 = neg_s_m_poly * g_ns_0;
        let recovered_ns_0 = sk.decrypt_raw(&rgsw.neg_s_m.samples[0]);
        let diff_ns_0 = recovered_ns_0 - expected_ns_0;
        let mut centred = [0i64; 4];
        diff_ns_0.to_centered_coeffs(&mut centred);
        for (j, &c) in centred.iter().enumerate() {
            assert!(
                c.abs() <= 4,
                "neg-s-m-half base_neg_s_m mismatch at coeff {j}: noise {c}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // external_product — Phase 7
    // -----------------------------------------------------------------------

    /// Phase 7 imports for tests.
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::encryption::encode;

    type SinglePolyP2<const N: usize> = Poly<N, PowerOfTwoModulus<1>, Coefficient>;
    type SinglePolyP256<const N: usize> = Poly<N, PowerOfTwoModulus<8>, Coefficient>;
    type SinglePolyViaCQ2<const N: usize> =
        Poly<N, crate::algebra::zq::modulus::paper::ViaCQ2, Coefficient>;
    type RnsPolyViaCQ1<const N: usize> =
        PolyRns<N, crate::algebra::rns::basis::paper::ViaCQ1Rns, Coefficient>;

    /// `decrypt(rgsw.external_product(&rlwe, base, base), sk).decode(p) == m1 · m2 mod p`.
    /// Toy `(q=1024, p=2, B=2, L=10)` with ternary errors.
    #[test]
    fn external_product_recovers_m1_times_m2_at_q1024_b2() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        const L1: usize = 10;
        const L2: usize = 10;
        let mut sk_prg = Shake256Prg::new(b"sk-ep-toy");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-ep-toy");

        // m1 raw at q (small-norm); m2 plaintext at p, then encoded.
        let m1: SinglePolyQ1024<4> = Poly::new(q, [1, 0, 1, 0]);
        let m2: SinglePolyP2<4> = Poly::new(p, [0, 1, 1, 1]);

        let rgsw = sk.encrypt_rgsw::<L1, L2>(&m1, 2, 2, Distribution::Ternary, &mut enc_prg);
        let encoded_m2: SinglePolyQ1024<4> = encode(&m2, q);
        let rlwe = sk.encrypt(&encoded_m2, Distribution::Ternary, &mut enc_prg);

        let ct = rgsw.external_product(&rlwe, 2, 2);
        let recovered: SinglePolyP2<4> = sk.decrypt(&ct, p);

        // Expected: m1 · m2 mod (p, X^N+1). Both as polynomials at p.
        let m1_at_p: SinglePolyP2<4> = Poly::new(p, [1, 0, 1, 0]);
        let expected = m1_at_p * m2;
        for i in 0..4 {
            assert_eq!(
                recovered.coeff(i).to_u64(),
                expected.coeff(i).to_u64(),
                "coeff {i}: recovered {:?} != expected {:?}",
                recovered.coeff(i).to_u64(),
                expected.coeff(i).to_u64(),
            );
        }
    }

    /// Asymmetric (L1, L2) and (base_neg_s_m, base_m). Confirms each
    /// half's gadget params thread through correctly.
    #[test]
    fn external_product_with_distinct_l1_l2_and_distinct_bases() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        const L1: usize = 5;
        const L2: usize = 10;
        let base_neg_s_m: u64 = 4;
        let base_m: u64 = 2;
        let mut sk_prg = Shake256Prg::new(b"sk-ep-asym");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-ep-asym");

        let m1: SinglePolyQ1024<4> = Poly::new(q, [1, 1, 0, 0]);
        let m2: SinglePolyP2<4> = Poly::new(p, [1, 0, 1, 0]);

        let rgsw = sk.encrypt_rgsw::<L1, L2>(
            &m1,
            base_neg_s_m,
            base_m,
            Distribution::Ternary,
            &mut enc_prg,
        );
        let encoded_m2: SinglePolyQ1024<4> = encode(&m2, q);
        let rlwe = sk.encrypt(&encoded_m2, Distribution::Ternary, &mut enc_prg);

        let ct = rgsw.external_product(&rlwe, base_neg_s_m, base_m);
        let recovered: SinglePolyP2<4> = sk.decrypt(&ct, p);

        let m1_at_p: SinglePolyP2<4> = Poly::new(p, [1, 1, 0, 0]);
        let expected = m1_at_p * m2;
        for i in 0..4 {
            assert_eq!(
                recovered.coeff(i).to_u64(),
                expected.coeff(i).to_u64(),
                "asymmetric ep diverged at coeff {i}"
            );
        }
    }

    /// Paper-class single-prime: VIA-C `q₂`, p=256, CMux-sel gadget
    /// `(L=2, B=81)` for both halves, σ=4 Gaussian.
    #[test]
    fn external_product_at_via_c_q2_p256_paper_gaussian() {
        let q = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        let p = PowerOfTwoModulus::<8>;
        const L1: usize = 2;
        const L2: usize = 2;
        let base: u64 = 81;
        let mut sk_prg = Shake256Prg::new(b"sk-ep-vc-q2");
        let sk =
            SecretKey::<16, SinglePolyViaCQ2<16>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-ep-vc-q2");

        // Small-norm m1 (binary) for tight noise.
        let m1: SinglePolyViaCQ2<16> =
            Poly::new(q, [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]);
        // Full-range m2 at p=256.
        let m2_coeffs: [u64; 16] = [
            0, 1, 13, 31, 63, 127, 200, 255, 7, 42, 99, 137, 200, 250, 5, 17,
        ];
        let m2: SinglePolyP256<16> = Poly::new(p, m2_coeffs);

        let rgsw = sk.encrypt_rgsw::<L1, L2>(
            &m1,
            base,
            base,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let encoded_m2: SinglePolyViaCQ2<16> = encode(&m2, q);
        let rlwe = sk.encrypt(
            &encoded_m2,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );

        let ct = rgsw.external_product(&rlwe, base, base);
        let recovered: SinglePolyP256<16> = sk.decrypt(&ct, p);

        let m1_at_p: SinglePolyP256<16> =
            Poly::new(p, [1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0]);
        let expected = m1_at_p * m2;
        for i in 0..16 {
            assert_eq!(
                recovered.coeff(i).to_u64(),
                expected.coeff(i).to_u64(),
                "VIA-C q₂ external product diverged at coeff {i}"
            );
        }
    }

    /// Paper-class RNS: VIA-C `q₁` (Q ≈ 2⁷⁵), p=2, DMux-ctrl gadget
    /// `(L=2, B=55879)` for both halves, σ=4 Gaussian. Exercises the
    /// RNS gadget-product path through both halves of the external
    /// product.
    #[test]
    fn external_product_at_via_c_q1_rns_p2_paper_gaussian() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let p = PowerOfTwoModulus::<1>;
        const L1: usize = 2;
        const L2: usize = 2;
        let base: u64 = 55879;
        let mut sk_prg = Shake256Prg::new(b"sk-ep-vc-q1rns");
        let sk =
            SecretKey::<16, RnsPolyViaCQ1<16>>::keygen(basis, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-ep-vc-q1rns");

        // Small-norm m1 raw at q₁. Lift binary coefficients via `from_centered_i64s`.
        let m1_centered: [i64; 16] = [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0];
        let m1 = <RnsPolyViaCQ1<16> as RingPoly<16>>::from_centered_i64s(basis, &m1_centered);
        // Binary m2 at p=2.
        let m2_coeffs: [u64; 16] = [1, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0];
        let m2: SinglePolyP2<16> = Poly::new(p, m2_coeffs);

        let rgsw = sk.encrypt_rgsw::<L1, L2>(
            &m1,
            base,
            base,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let encoded_m2: RnsPolyViaCQ1<16> = encode(&m2, basis);
        let rlwe = sk.encrypt(
            &encoded_m2,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );

        let ct = rgsw.external_product(&rlwe, base, base);
        let recovered: SinglePolyP2<16> = sk.decrypt(&ct, p);

        // Expected: m1 · m2 mod (p=2, X¹⁶+1). m1 binary, m2 binary.
        let m1_at_p: SinglePolyP2<16> =
            Poly::new(p, [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0]);
        let expected = m1_at_p * m2;
        for i in 0..16 {
            assert_eq!(
                recovered.coeff(i).to_u64(),
                expected.coeff(i).to_u64(),
                "VIA-C q₁ RNS external product diverged at coeff {i}"
            );
        }
    }
}
