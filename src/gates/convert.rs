//! §4.6 RLWE→RGSW conversion and §4.7 RGSW modulus-switch (Part 3).
//!
//! See `.docs/primitives.md` §4.6-§4.7, `gates.py:257-311`, and
//! `query_comp.py:316-353`.
//!
//! - [`gen_rlwe_to_rgsw_key`] — the conversion key $\mathrm{RLev}_S(S^2)$.
//! - [`rlwe_to_rgsw`] — convert per-gadget-level RLWE ciphertexts to RGSW.
//! - [`mod_switch_rgsw`] — apply the Layer-3
//!   [`crate::switching::mod_switch::mod_switch_sym`] to every constituent RLWE
//!   of an RGSW. This is why §4.7 lives in `gates` (Layer 4), not `encryption`
//!   (Layer 2): importing Layer 3 into Layer 2 would invert the layer order.

use crate::algebra::ring::RingPoly;
use crate::encryption::types::{RGSWCiphertext, RLWECiphertext, RLevCiphertext, SecretKey};
use crate::sampling::distribution::Distribution;
use crate::sampling::prg::Shake256Prg;
use crate::switching::mod_switch::mod_switch_sym;

/// §4.6 — Generate the conversion key $\mathrm{RLev}_S(S^2)$: an RLev
/// encryption of the **square** of the secret-key polynomial under the same
/// key. Thin wrapper over [`SecretKey::encrypt_rlev`], mirroring
/// [`crate::encryption::gen_ksk`]. Returns a bare `RLevCiphertext` (no named
/// wrapper — `gen_ksk` sets the precedent).
///
/// `paper:gates.py:286-291`
///
/// # PRG consumption order
///
/// Delegates to `encrypt_rlev::<L>` (mask then error per level, level 0..L).
/// Reversing would break Python parity for the conversion-key test vectors.
///
/// # Constant-time
///
/// The `S^2 = *sk.poly() * *sk.poly()` schoolbook multiply is data-independent;
/// the squared secret is the plaintext being encrypted, not a branch target.
///
/// # Example
///
/// ```rust
/// use via_rs::algebra::ring::element::Poly;
/// use via_rs::algebra::ring::form::Coefficient;
/// use via_rs::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_rs::encryption::types::{RLevCiphertext, SecretKey};
/// use via_rs::gates::gen_rlwe_to_rgsw_key;
/// use via_rs::sampling::distribution::Distribution;
/// use via_rs::sampling::prg::Shake256Prg;
///
/// type Q = Poly<4, PowerOfTwoModulus<10>, Coefficient>;
/// let q = PowerOfTwoModulus::<10>;
/// let mut prg = Shake256Prg::new(b"gen-conv-key-doc");
/// let sk = SecretKey::<4, Q>::keygen(q, Distribution::Ternary, &mut prg);
/// let key: RLevCiphertext<4, Q, 4> =
///     gen_rlwe_to_rgsw_key(&sk, 2, Distribution::Ternary, &mut prg);
/// assert_eq!(key.samples.len(), 4);
/// ```
pub fn gen_rlwe_to_rgsw_key<const N: usize, R: RingPoly<N>, const L: usize>(
    sk: &SecretKey<N, R>,
    base: u64,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
) -> RLevCiphertext<N, R, L> {
    let s_squared = *sk.poly() * *sk.poly();
    sk.encrypt_rlev::<L>(&s_squared, base, error_dist, prg)
}

/// §4.6 — Convert per-gadget-level RLWE ciphertexts `rlwe_levels[i] =
/// RLWE_S(M·g[i])` into an `RGSW_S(M)` using the conversion key `conv_key =
/// RLev_S(S^2)`.
///
/// For each level `i`: `prod = conv_key ⊡ A_i`; the `neg_s_m` sample is
/// `(B_i + prod.mask, prod.body)`, which decrypts to `-S·M·g[i]` (the
/// §2.4 key-switch identity per gadget level):
///
/// ```text
/// prod.body - (B_i + prod.mask)·S
///   = (E' + A_i·S²) - (A_i·S + E_i + M·g[i])·S - prod.mask·S
///   = E' - E_i·S - S·M·g[i]   (the A_i·S² terms cancel)
///   ≈ -S·M·g[i]
/// ```
///
/// The `m` half (`m_rlev`) is passed **separately**: in production
/// (`query_comp.py:332`) it is `RLevCiphertext::new(rlwe_levels)` (the same
/// ciphertexts); in tests it is a fresh `RLev_S(M)`. The conversion key should
/// use a **finer** gadget (`L_CK > L_OUT`, smaller `base_ck`) so the
/// `~q/base_ck^{L_CK}` approximation error stays under the downstream CMux
/// budget.
///
/// `paper:gates.py:257-311`, `paper:query_comp.py:316-353`
///
/// # Output depth
///
/// Both RGSW halves share depth `L_OUT`. If per-half depths are ever needed
/// (paper Tables 5-6), add an `L_M` const-generic and return
/// `RGSWCiphertext<N, R, L_OUT, L_M>`.
///
/// # Constant-time: No
///
/// Operates on RLWE-uniform ciphertext coefficients; not for key material.
///
/// # Example
///
/// ```rust
/// use via_rs::algebra::ring::RingPoly;
/// use via_rs::algebra::ring::element::Poly;
/// use via_rs::algebra::ring::form::Coefficient;
/// use via_rs::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_rs::encryption::types::{RGSWCiphertext, RLWECiphertext, SecretKey};
/// use via_rs::gates::{gen_rlwe_to_rgsw_key, rlwe_to_rgsw};
/// use via_rs::sampling::distribution::Distribution;
/// use via_rs::sampling::prg::Shake256Prg;
///
/// type Q = Poly<4, PowerOfTwoModulus<17>, Coefficient>;
/// let q = PowerOfTwoModulus::<17>;
/// let mut prg = Shake256Prg::new(b"rlwe-to-rgsw-doc");
/// let sk = SecretKey::<4, Q>::keygen(q, Distribution::Ternary, &mut prg);
///
/// // Trivial zero levels (shape demo).
/// let zero = RLWECiphertext::<4, Q>::new(Q::zero(q), Q::zero(q));
/// let conv_key = gen_rlwe_to_rgsw_key::<4, Q, 8>(&sk, 2, Distribution::Ternary, &mut prg);
/// let m_rlev = sk.encrypt_rlev::<2>(&Q::zero(q), 4, Distribution::Ternary, &mut prg);
///
/// let rgsw: RGSWCiphertext<4, Q, 2, 2> = rlwe_to_rgsw([zero; 2], &conv_key, m_rlev, 2);
/// assert_eq!(rgsw.neg_s_m.samples.len(), 2);
/// assert_eq!(rgsw.m.samples.len(), 2);
/// ```
pub fn rlwe_to_rgsw<const N: usize, R: RingPoly<N>, const L_OUT: usize, const L_CK: usize>(
    rlwe_levels: [RLWECiphertext<N, R>; L_OUT],
    conv_key: &RLevCiphertext<N, R, L_CK>,
    m_rlev: RLevCiphertext<N, R, L_OUT>,
    base_ck: u64,
) -> RGSWCiphertext<N, R, L_OUT, L_OUT> {
    let neg_s_m = core::array::from_fn(|i| {
        let prod = conv_key.gadget_product(&rlwe_levels[i].mask, base_ck);
        RLWECiphertext::new(rlwe_levels[i].body + prod.mask, prod.body)
    });
    RGSWCiphertext::new(RLevCiphertext::new(neg_s_m), m_rlev)
}

/// §4.7 — Modulus-switch an RGSW from ring `R_SRC` to ring `R_DST` by mapping
/// the Layer-3 [`mod_switch_sym`] over every constituent RLWE of both RLev
/// halves. Gadget depths `L1` / `L2` and ring degree `N` are preserved; only
/// the modulus changes.
///
/// In the VIA-C server (`server.py:38-60`) this switches the sel/rot RGSW bits
/// from `q1` to `q2` before CMux / CRot, shrinking coefficients from ~75 bits
/// (RNS `q1`) to ~34 bits (`q2`).
///
/// # Placement
///
/// Lives in `gates` (Layer 4), not `encryption` (Layer 2), because it depends
/// on Layer 3's `mod_switch_sym`.
///
/// # Constant-time: No
///
/// Delegates to `mod_switch_sym` (RLWE-uniform coefficients, not key material).
///
/// # Example
///
/// ```rust
/// use via_rs::algebra::ring::RingPoly;
/// use via_rs::algebra::ring::element::Poly;
/// use via_rs::algebra::ring::form::Coefficient;
/// use via_rs::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_rs::encryption::types::{RGSWCiphertext, RLWECiphertext, RLevCiphertext};
/// use via_rs::gates::mod_switch_rgsw;
///
/// type Src = Poly<4, PowerOfTwoModulus<10>, Coefficient>;
/// type Dst = Poly<4, PowerOfTwoModulus<6>, Coefficient>;
/// let q_src = PowerOfTwoModulus::<10>;
/// let q_dst = PowerOfTwoModulus::<6>;
///
/// let z = <Src as RingPoly<4>>::zero(q_src);
/// let rlwe = RLWECiphertext::<4, Src>::new(z, z);
/// let rlev: RLevCiphertext<4, Src, 2> = RLevCiphertext::new([rlwe; 2]);
/// let rgsw: RGSWCiphertext<4, Src, 2, 2> = RGSWCiphertext::new(rlev, rlev);
///
/// let out: RGSWCiphertext<4, Dst, 2, 2> = mod_switch_rgsw(&rgsw, q_dst);
/// assert_eq!(out.neg_s_m.samples.len(), 2);
/// assert_eq!(out.m.samples.len(), 2);
/// ```
#[allow(non_camel_case_types)]
pub fn mod_switch_rgsw<
    const N: usize,
    R_SRC: RingPoly<N>,
    R_DST: RingPoly<N>,
    const L1: usize,
    const L2: usize,
>(
    rgsw: &RGSWCiphertext<N, R_SRC, L1, L2>,
    dst_mod: R_DST::Modulus,
) -> RGSWCiphertext<N, R_DST, L1, L2> {
    let neg_s_m = RLevCiphertext::new(core::array::from_fn(|i| {
        mod_switch_sym(&rgsw.neg_s_m.samples[i], dst_mod)
    }));
    let m = RLevCiphertext::new(core::array::from_fn(|i| {
        mod_switch_sym(&rgsw.m.samples[i], dst_mod)
    }));
    RGSWCiphertext::new(neg_s_m, m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::zq::modulus::{DynModulus, PowerOfTwoModulus};
    use crate::encryption::gadget::gadget_vector_values;
    use crate::encryption::rlwe::encode;

    // ----- gen_rlwe_to_rgsw_key -----

    /// The key is exactly `encrypt_rlev(S^2)` — verify by reconstructing with an
    /// identically-seeded PRG (locks the body computation + PRG order).
    #[test]
    fn gen_rlwe_to_rgsw_key_equals_encrypt_rlev_of_s_squared() {
        type Q = Poly<4, PowerOfTwoModulus<10>, Coefficient>;
        const L: usize = 10;
        let q = PowerOfTwoModulus::<10>;
        let mut sk_prg = Shake256Prg::new(b"genck-sk");
        let sk = SecretKey::<4, Q>::keygen(q, Distribution::Ternary, &mut sk_prg);

        let mut prg1 = Shake256Prg::new(b"genck-enc");
        let key = gen_rlwe_to_rgsw_key::<4, Q, L>(&sk, 2, Distribution::Ternary, &mut prg1);

        let mut prg2 = Shake256Prg::new(b"genck-enc");
        let s2 = *sk.poly() * *sk.poly();
        let expected = sk.encrypt_rlev::<L>(&s2, 2, Distribution::Ternary, &mut prg2);

        assert_eq!(key.samples, expected.samples);
    }

    // ----- rlwe_to_rgsw -----

    /// Mirror `test_gates.py::TestRLWEToRGSW`: build per-level RLWE(M·g[i]),
    /// convert to RGSW(M=1), then check `external_product(rgsw, RLWE(M')) ==
    /// RLWE(M')`. Functional smoke test (its own seeds), complementary to the
    /// byte-parity KAT.
    #[test]
    fn rlwe_to_rgsw_produces_usable_rgsw() {
        type Q = Poly<64, DynModulus, Coefficient>;
        type P = Poly<64, DynModulus, Coefficient>;
        const N: usize = 64;
        const L_OUT: usize = 3;
        const L_CK: usize = 16;
        let q = DynModulus::new(65537);
        let p = DynModulus::new(2);
        let base = 4u64;
        let ck_base = 2u64;

        let mut sk_prg = Shake256Prg::new(b"conv-key");
        let sk = SecretKey::<N, Q>::keygen(q, Distribution::Ternary, &mut sk_prg);

        // m_poly = constant 1.
        let mut one = [0u128; N];
        one[0] = 1;
        let m_poly = <Q as RingPoly<N>>::from_u128_coeffs(q, &one);

        // Per-level RLWE(M·g[i]) with separate seeds.
        let g = gadget_vector_values::<N, Q, L_OUT>(q, base);
        let level_seeds: [&[u8]; L_OUT] = [b"conv-rlwe-0", b"conv-rlwe-1", b"conv-rlwe-2"];
        let rlwe_levels: [RLWECiphertext<N, Q>; L_OUT] = core::array::from_fn(|i| {
            let mut gc = [0u128; N];
            gc[0] = g[i];
            let g_poly = <Q as RingPoly<N>>::from_u128_coeffs(q, &gc);
            let scaled = m_poly * g_poly;
            let mut prg = Shake256Prg::new(level_seeds[i]);
            sk.encrypt(&scaled, Distribution::Ternary, &mut prg)
        });

        let mut ck_prg = Shake256Prg::new(b"conv-key-rlev");
        let conv_key =
            gen_rlwe_to_rgsw_key::<N, Q, L_CK>(&sk, ck_base, Distribution::Ternary, &mut ck_prg);

        let mut m_prg = Shake256Prg::new(b"conv-m-rlev");
        let m_rlev = sk.encrypt_rlev::<L_OUT>(&m_poly, base, Distribution::Ternary, &mut m_prg);

        let rgsw = rlwe_to_rgsw::<N, Q, L_OUT, L_CK>(rlwe_levels, &conv_key, m_rlev, ck_base);
        assert_eq!(rgsw.neg_s_m.samples.len(), L_OUT);
        assert_eq!(rgsw.m.samples.len(), L_OUT);

        // External product with RLWE(M') should give RLWE(1·M') = RLWE(M').
        let msg_prime: P = Poly::new(p, {
            let mut c = [0u64; N];
            c[0] = 1;
            c[1] = 1;
            c
        });
        let mut ep_prg = Shake256Prg::new(b"conv-rlwe-prime");
        let ct_prime = sk.encrypt(&encode(&msg_prime, q), Distribution::Ternary, &mut ep_prg);
        let result = rgsw.external_product(&ct_prime, base, base);
        let recovered: P = sk.decrypt(&result, p);
        assert_eq!(recovered, msg_prime);
    }

    /// Paper-class RNS shape/compile test: query compression runs `rlwe_to_rgsw`
    /// at the q1 RNS modulus.
    #[test]
    fn rlwe_to_rgsw_at_viac_q1rns() {
        type Q<const N: usize> =
            PolyRns<N, crate::algebra::rns::basis::paper::ViaCQ1Rns, Coefficient>;
        const N: usize = 16;
        const L_OUT: usize = 3;
        const L_CK: usize = 8;
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let mut sk_prg = Shake256Prg::new(b"q1rns-sk");
        let sk = SecretKey::<N, Q<N>>::keygen(basis, Distribution::Ternary, &mut sk_prg);

        let zero = <Q<N> as RingPoly<N>>::zero(basis);
        let zero_rlwe = RLWECiphertext::<N, Q<N>>::new(zero, zero);
        let rlwe_levels = [zero_rlwe; L_OUT];

        let mut ck_prg = Shake256Prg::new(b"q1rns-ck");
        let conv_key =
            gen_rlwe_to_rgsw_key::<N, Q<N>, L_CK>(&sk, 2, Distribution::Ternary, &mut ck_prg);
        let mut m_prg = Shake256Prg::new(b"q1rns-m");
        let m_rlev = sk.encrypt_rlev::<L_OUT>(&zero, 4, Distribution::Ternary, &mut m_prg);

        let rgsw = rlwe_to_rgsw::<N, Q<N>, L_OUT, L_CK>(rlwe_levels, &conv_key, m_rlev, 2);
        assert_eq!(rgsw.neg_s_m.samples.len(), L_OUT);
        assert_eq!(rgsw.m.samples.len(), L_OUT);
    }

    // ----- mod_switch_rgsw -----

    /// Every switched coefficient must be `< q_dst`; depths preserved.
    #[test]
    fn mod_switch_rgsw_output_coefficients_below_dst_modulus() {
        type Src<const N: usize> = Poly<N, PowerOfTwoModulus<16>, Coefficient>;
        type Dst<const N: usize> = Poly<N, PowerOfTwoModulus<8>, Coefficient>;
        const N: usize = 8;
        let q_src = PowerOfTwoModulus::<16>;
        let q_dst = PowerOfTwoModulus::<8>;
        let mut sk_prg = Shake256Prg::new(b"msr-shape-sk");
        let sk = SecretKey::<N, Src<N>>::keygen(q_src, Distribution::Ternary, &mut sk_prg);
        let mut c = [0u128; N];
        c[0] = 1;
        let m = <Src<N> as RingPoly<N>>::from_u128_coeffs(q_src, &c);
        let mut rgsw_prg = Shake256Prg::new(b"msr-shape-rgsw");
        let rgsw = sk.encrypt_rgsw::<2, 3>(&m, 2, 2, Distribution::Ternary, &mut rgsw_prg);

        let switched: RGSWCiphertext<N, Dst<N>, 2, 3> = mod_switch_rgsw(&rgsw, q_dst);
        assert_eq!(switched.neg_s_m.samples.len(), 2);
        assert_eq!(switched.m.samples.len(), 3);
        let q_dst_val = <Dst<N> as RingPoly<N>>::modulus_value(q_dst);
        for sample in switched
            .neg_s_m
            .samples
            .iter()
            .chain(switched.m.samples.iter())
        {
            for poly in [&sample.mask, &sample.body] {
                let mut coeffs = [0u128; N];
                poly.to_u128_coeffs(&mut coeffs);
                for v in coeffs {
                    assert!(v < q_dst_val, "coefficient {v} >= q_dst {q_dst_val}");
                }
            }
        }
    }

    /// A switched RGSW must still work in `external_product` at the new modulus.
    #[test]
    fn mod_switch_rgsw_external_product_still_works_after_switch() {
        use crate::switching::rekey::rekey_secret_key;
        type Src<const N: usize> = Poly<N, PowerOfTwoModulus<16>, Coefficient>;
        type Dst<const N: usize> = Poly<N, PowerOfTwoModulus<12>, Coefficient>;
        type P<const N: usize> = Poly<N, PowerOfTwoModulus<1>, Coefficient>;
        const N: usize = 8;
        const L: usize = 16; // base^L = 2^16 = q_src
        let q_src = PowerOfTwoModulus::<16>;
        let q_dst = PowerOfTwoModulus::<12>;
        let p = PowerOfTwoModulus::<1>;

        let mut sk_prg = Shake256Prg::new(b"msr-ep-sk");
        let sk_src = SecretKey::<N, Src<N>>::keygen(q_src, Distribution::Ternary, &mut sk_prg);

        // RGSW(M) with binary M at q_src.
        let m1_coeffs: [u64; N] = [1, 0, 1, 0, 1, 1, 0, 0];
        let m1: P<N> = Poly::new(p, m1_coeffs);
        let mut m1_q = [0u128; N];
        for (i, &c) in m1_coeffs.iter().enumerate() {
            m1_q[i] = c as u128;
        }
        let m1_src = <Src<N> as RingPoly<N>>::from_u128_coeffs(q_src, &m1_q);
        let mut rgsw_prg = Shake256Prg::new(b"msr-ep-rgsw");
        let rgsw = sk_src.encrypt_rgsw::<L, L>(&m1_src, 2, 2, Distribution::Ternary, &mut rgsw_prg);

        let switched: RGSWCiphertext<N, Dst<N>, L, L> = mod_switch_rgsw(&rgsw, q_dst);

        // Destination key + fresh RLWE(M') at q_dst.
        let sk_dst: SecretKey<N, Dst<N>> = rekey_secret_key(&sk_src, q_dst);
        let m2: P<N> = Poly::new(p, [1, 1, 0, 1, 0, 0, 1, 0]);
        let mut enc_prg = Shake256Prg::new(b"msr-ep-enc");
        let ct_prime = sk_dst.encrypt(&encode(&m2, q_dst), Distribution::Ternary, &mut enc_prg);

        let result = switched.external_product(&ct_prime, 2, 2);
        let recovered: P<N> = sk_dst.decrypt(&result, p);
        let expected = m1 * m2;
        assert_eq!(recovered, expected);
    }

    /// Paper-class shape test: VIA-C server switches RGSW bits q1(RNS) → q2.
    #[test]
    fn mod_switch_rgsw_viac_q1rns_to_q2() {
        type Q1<const N: usize> =
            PolyRns<N, crate::algebra::rns::basis::paper::ViaCQ1Rns, Coefficient>;
        type Q2<const N: usize> = Poly<N, crate::algebra::zq::modulus::paper::ViaCQ2, Coefficient>;
        const N: usize = 16;
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let q2 = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        let mut sk_prg = Shake256Prg::new(b"msr-q1q2-sk");
        let sk = SecretKey::<N, Q1<N>>::keygen(basis, Distribution::Ternary, &mut sk_prg);
        let mut c = [0u128; N];
        c[0] = 1;
        let m = <Q1<N> as RingPoly<N>>::from_u128_coeffs(basis, &c);
        let mut rgsw_prg = Shake256Prg::new(b"msr-q1q2-rgsw");
        let rgsw = sk.encrypt_rgsw::<2, 2>(&m, 2, 2, Distribution::Ternary, &mut rgsw_prg);

        // Production gate: VIA-C server.py:38-60 performs exactly this switch.
        let switched: RGSWCiphertext<N, Q2<N>, 2, 2> = mod_switch_rgsw(&rgsw, q2);
        let q2_val = <Q2<N> as RingPoly<N>>::modulus_value(q2);
        for sample in switched
            .neg_s_m
            .samples
            .iter()
            .chain(switched.m.samples.iter())
        {
            let mut coeffs = [0u128; N];
            sample.mask.to_u128_coeffs(&mut coeffs);
            for v in coeffs {
                assert!(v < q2_val);
            }
        }
    }
}
