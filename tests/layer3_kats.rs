//! Cross-language KAT parity for Layer-3 switching primitives (§3.1–§3.4).
//!
//! Each test reproduces — in Rust, with the same seed and sampling order — a
//! setup that `.references/via-spec/scripts/gen_layer3_kats.py` ran in Python,
//! then asserts byte-for-byte equality against the generated `data::*`
//! constants. The `kat_gen_rsk_prg_order` test is the critical PRG-ordering
//! lock: it asserts the full D×L×N2 mask/body byte stream, so any divergence
//! in the four nested `gen_rsk` loops (j-outer, level-inner, mask-before-error)
//! surfaces immediately.
//!
//! Regenerate the constants with `just regen-kats`.

use via_rs::algebra::ring::abstraction::RingPoly;
use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::PowerOfTwoModulus;
use via_rs::encryption::rlwe::encode;
use via_rs::encryption::types::{RLWECiphertext, SecretKey};
use via_rs::sampling::distribution::Distribution;
use via_rs::sampling::prg::Shake256Prg;
use via_rs::switching::mod_switch::{mod_switch_asym, mod_switch_sym};
use via_rs::switching::rekey::rekey_secret_key;
use via_rs::switching::ring_switch::{RingSwitchKey, gen_rsk, ring_switch};

mod data {
    include!("data/layer3_kats.rs");
}

// TOY parameter backends (power-of-two moduli, matching the Python generator).
type Q2<const N: usize> = Poly<N, PowerOfTwoModulus<32>, Coefficient>;
type Q3<const N: usize> = Poly<N, PowerOfTwoModulus<16>, Coefficient>;
type Q4<const N: usize> = Poly<N, PowerOfTwoModulus<12>, Coefficient>;
type Pp<const N: usize> = Poly<N, PowerOfTwoModulus<4>, Coefficient>; // p = 16

const N1: usize = 64;
const N2: usize = 16;
const BASE: u64 = 4;
const DEPTH: usize = 8;
const D: usize = 4;

/// Assert that `poly`'s canonical coefficients equal `expected`.
fn assert_coeffs<const N: usize, R: RingPoly<N>>(poly: &R, expected: &[u64]) {
    let mut got = [0u128; N];
    poly.to_u128_coeffs(&mut got);
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(got[i], u128::from(e), "coeff {i}");
    }
}

fn u128_arr<const N: usize>(src: &[u64; N]) -> [u128; N] {
    core::array::from_fn(|i| u128::from(src[i]))
}

#[test]
fn kat_mod_switch_sym() {
    let q2 = PowerOfTwoModulus::<32>;
    let q3 = PowerOfTwoModulus::<16>;
    let mut prg = Shake256Prg::new(b"layer3-kat-mod-switch-sym");
    let mask = <Q2<N1> as RingPoly<N1>>::random_uniform(q2, &mut prg);
    let body = <Q2<N1> as RingPoly<N1>>::random_uniform(q2, &mut prg);
    let ct = RLWECiphertext::new(mask, body);
    let out: RLWECiphertext<N1, Q3<N1>> = mod_switch_sym(&ct, q3);
    assert_coeffs(&out.mask, &data::MOD_SWITCH_SYM_MASK_AT_Q3);
    assert_coeffs(&out.body, &data::MOD_SWITCH_SYM_BODY_AT_Q3);
}

#[test]
fn kat_mod_switch_asym() {
    let q2 = PowerOfTwoModulus::<32>;
    let q3 = PowerOfTwoModulus::<16>;
    let q4 = PowerOfTwoModulus::<12>;
    let mut prg = Shake256Prg::new(b"layer3-kat-mod-switch-asym");
    let mask = <Q2<N1> as RingPoly<N1>>::random_uniform(q2, &mut prg);
    let body = <Q2<N1> as RingPoly<N1>>::random_uniform(q2, &mut prg);
    let ct = RLWECiphertext::new(mask, body);
    let out = mod_switch_asym::<N1, Q2<N1>, Q3<N1>, Q4<N1>>(&ct, q3, q4);
    assert_coeffs(&out.mask, &data::MOD_SWITCH_ASYM_MASK_AT_Q3);
    assert_coeffs(&out.body, &data::MOD_SWITCH_ASYM_BODY_AT_Q4);
}

/// Build the ring-switch key from the `gen-rsk-rsk` seed exactly as the
/// Python generator does (S1, S2 keygen, then gen_rsk on the same stream).
fn build_gen_rsk_rsk() -> RingSwitchKey<N1, N2, Q3<N2>, DEPTH, D> {
    let q3 = PowerOfTwoModulus::<16>;
    let mut prg = Shake256Prg::new(b"layer3-kat-gen-rsk-rsk");
    let s1 = SecretKey::<N1, Q3<N1>>::keygen(q3, Distribution::Ternary, &mut prg);
    let s2 = SecretKey::<N2, Q3<N2>>::keygen(q3, Distribution::Ternary, &mut prg);
    gen_rsk(&s1, &s2, BASE, Distribution::Ternary, &mut prg)
}

#[test]
fn kat_gen_rsk_first_sample() {
    let rsk = build_gen_rsk_rsk();
    let first = &rsk.samples[0].samples[0];
    assert_coeffs(&first.mask, &data::GEN_RSK_J0_L0_MASK);
    assert_coeffs(&first.body, &data::GEN_RSK_J0_L0_BODY);
}

#[test]
fn kat_gen_rsk_prg_order() {
    let rsk = build_gen_rsk_rsk();
    let mut mask_stream: Vec<u64> = Vec::new();
    let mut body_stream: Vec<u64> = Vec::new();
    for j in 0..D {
        for level in 0..DEPTH {
            let sample = &rsk.samples[j].samples[level];
            let mut m = [0u128; N2];
            let mut b = [0u128; N2];
            sample.mask.to_u128_coeffs(&mut m);
            sample.body.to_u128_coeffs(&mut b);
            mask_stream.extend(m.iter().map(|&v| v as u64));
            body_stream.extend(b.iter().map(|&v| v as u64));
        }
    }
    assert_eq!(
        mask_stream.as_slice(),
        data::GEN_RSK_FULL_MASK_STREAM.as_slice()
    );
    assert_eq!(
        body_stream.as_slice(),
        data::GEN_RSK_FULL_BODY_STREAM.as_slice()
    );
}

#[test]
fn kat_ring_switch_apply() {
    let q3 = PowerOfTwoModulus::<16>;
    let pm = PowerOfTwoModulus::<4>;
    let mut prg = Shake256Prg::new(b"layer3-kat-ring-switch");
    let s1 = SecretKey::<N1, Q3<N1>>::keygen(q3, Distribution::Ternary, &mut prg);
    let s2 = SecretKey::<N2, Q3<N2>>::keygen(q3, Distribution::Ternary, &mut prg);
    let rsk: RingSwitchKey<N1, N2, Q3<N2>, DEPTH, D> =
        gen_rsk(&s1, &s2, BASE, Distribution::Ternary, &mut prg);

    let m: [u128; N1] = core::array::from_fn(|i| (i % 16) as u128);
    let pt = <Pp<N1> as RingPoly<N1>>::from_u128_coeffs(pm, &m);
    let encoded: Q3<N1> = encode(&pt, q3);
    let ct = s1.encrypt(&encoded, Distribution::Ternary, &mut prg);

    let out = ring_switch(&ct, &rsk, BASE);
    assert_coeffs(&out.mask, &data::RING_SWITCH_OUT_MASK);
    assert_coeffs(&out.body, &data::RING_SWITCH_OUT_BODY);
}

#[test]
fn kat_rekey_secret_key() {
    let q2 = PowerOfTwoModulus::<32>;
    let q3 = PowerOfTwoModulus::<16>;
    let mut prg = Shake256Prg::new(b"layer3-kat-rekey");
    let sk = SecretKey::<N1, Q2<N1>>::keygen(q2, Distribution::Ternary, &mut prg);
    let rekeyed: SecretKey<N1, Q3<N1>> = rekey_secret_key(&sk, q3);
    assert_coeffs(rekeyed.poly(), &data::REKEY_SK_Q3_COEFFS);
}

#[test]
fn kat_decrypt_asymmetric_recovery() {
    let q3 = PowerOfTwoModulus::<16>;
    let q4 = PowerOfTwoModulus::<12>;
    let pm = PowerOfTwoModulus::<4>;
    let mut prg = Shake256Prg::new(b"layer3-kat-decrypt-asym");
    let s2 = SecretKey::<N2, Q3<N2>>::keygen(q3, Distribution::Ternary, &mut prg);

    let m: [u64; N2] = core::array::from_fn(|i| ((3 * i + 1) % 16) as u64);
    let pt = <Pp<N2> as RingPoly<N2>>::from_u128_coeffs(pm, &u128_arr(&m));
    let encoded: Q3<N2> = encode(&pt, q3);
    let ct = s2.encrypt(&encoded, Distribution::Ternary, &mut prg);

    // Body-only rescale: mask q3 -> q3 (identity), body q3 -> q4.
    let ms = mod_switch_asym::<N2, Q3<N2>, Q3<N2>, Q4<N2>>(&ct, q3, q4);
    let recovered: Pp<N2> = s2.decrypt_asymmetric(&ms, q3, q4, pm);

    assert_coeffs(&recovered, &data::DECRYPT_ASYM_RECOVERED);
    // Recovery is exact: the recovered plaintext equals the input.
    assert_coeffs(&recovered, &data::DECRYPT_ASYM_INPUT_PLAINTEXT);
}
