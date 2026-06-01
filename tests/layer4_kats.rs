//! Cross-language KAT parity for Layer-4 homomorphic gates (§4.1–§4.7).
//!
//! Each test reproduces — in Rust, with the same seed and PRG draw order — a
//! setup that `.references/via-spec/scripts/gen_layer4_kats.py` ran in Python,
//! then asserts byte-for-byte equality against the generated `data::*`
//! constants. `kat_crot_forward` / `kat_crot_slot_extract` (plus the data-level
//! `kat_crot_directions_differ`) are the direction-locking tests: identical key
//! and bit draws, opposite rotation directions, byte-distinct outputs.
//!
//! Regenerate the constants with `just regen-kats-layer4`.

use via_rs::algebra::ring::abstraction::RingPoly;
use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::gadget::gadget_vector_values;
use via_rs::encryption::rlwe::encode;
use via_rs::encryption::types::{RGSWCiphertext, RLWECiphertext, SecretKey};
use via_rs::gates::{
    CRotDir, cmux, cmux_tree, crot, dmux, dmux_tree, mod_switch_rgsw, rlwe_to_rgsw, rotate,
};
use via_rs::sampling::distribution::Distribution;
use via_rs::sampling::prg::Shake256Prg;

mod data {
    include!("data/layer4_kats.rs");
}

// TOY parameters, matching gen_layer4_kats.py.
const N: usize = 64;
const Q: u64 = 65537;
const P: u64 = 2;
const Q_SMALL: u64 = 4096;
const BASE: u64 = 4;
const DEPTH: usize = 3;
const CK_DEPTH: usize = 16;
const CK_BASE: u64 = 2;

type R = Poly<64, DynModulus, Coefficient>;
type Pt = Poly<64, DynModulus, Coefficient>;

fn q() -> DynModulus {
    DynModulus::new(Q)
}
fn p() -> DynModulus {
    DynModulus::new(P)
}

fn assert_coeffs(poly: &R, expected: &[u64], label: &str) {
    let mut coeffs = [0u128; N];
    poly.to_u128_coeffs(&mut coeffs);
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(coeffs[i], e as u128, "{label}: coeff {i} mismatch");
    }
}

fn bit_poly(b: u64) -> R {
    let mut c = [0u128; N];
    c[0] = b as u128;
    <R as RingPoly<N>>::from_u128_coeffs(q(), &c)
}

fn msg_poly(coeffs: &[u64]) -> Pt {
    let mut c = [0u64; N];
    c[..coeffs.len()].copy_from_slice(coeffs);
    Pt::new(p(), c)
}

fn enc(coeffs: &[u64], sk: &SecretKey<N, R>, prg: &mut Shake256Prg) -> RLWECiphertext<N, R> {
    let encoded: R = encode(&msg_poly(coeffs), q());
    sk.encrypt(&encoded, Distribution::Ternary, prg)
}

fn rgsw_bit(
    b: u64,
    sk: &SecretKey<N, R>,
    prg: &mut Shake256Prg,
) -> RGSWCiphertext<N, R, DEPTH, DEPTH> {
    sk.encrypt_rgsw::<DEPTH, DEPTH>(&bit_poly(b), BASE, BASE, Distribution::Ternary, prg)
}

#[test]
fn kat_rotate() {
    let mut prg = Shake256Prg::new(b"layer4-kat-rotate");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let ct = enc(&[1], &sk, &mut prg);
    let out = rotate(&ct, 3);
    assert_coeffs(&out.mask, &data::ROTATE_K3_MASK, "rotate mask");
    assert_coeffs(&out.body, &data::ROTATE_K3_BODY, "rotate body");
}

#[test]
fn kat_cmux() {
    let mut prg = Shake256Prg::new(b"layer4-kat-cmux");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let rgsw1 = rgsw_bit(1, &sk, &mut prg);
    let ct0 = enc(&[1], &sk, &mut prg);
    let ct1 = enc(&[0, 1], &sk, &mut prg);
    let out = cmux(&rgsw1, &ct0, &ct1, BASE, BASE);
    assert_coeffs(&out.mask, &data::CMUX_SELECT1_MASK, "cmux mask");
    assert_coeffs(&out.body, &data::CMUX_SELECT1_BODY, "cmux body");
}

#[test]
fn kat_dmux() {
    let mut prg = Shake256Prg::new(b"layer4-kat-dmux");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let rgsw1 = rgsw_bit(1, &sk, &mut prg);
    let ct = enc(&[1, 1], &sk, &mut prg);
    let (r0, r1) = dmux(&rgsw1, &ct, BASE, BASE);
    assert_coeffs(&r0.mask, &data::DMUX_BIT1_RESULT0_MASK, "dmux r0 mask");
    assert_coeffs(&r0.body, &data::DMUX_BIT1_RESULT0_BODY, "dmux r0 body");
    assert_coeffs(&r1.mask, &data::DMUX_BIT1_RESULT1_MASK, "dmux r1 mask");
    assert_coeffs(&r1.body, &data::DMUX_BIT1_RESULT1_BODY, "dmux r1 body");
}

#[test]
fn kat_cmux_tree() {
    let mut prg = Shake256Prg::new(b"layer4-kat-cmux-tree");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let bits: [RGSWCiphertext<N, R, DEPTH, DEPTH>; 2] =
        core::array::from_fn(|_| rgsw_bit(1, &sk, &mut prg));
    let mut inputs: [RLWECiphertext<N, R>; 4] = core::array::from_fn(|i| {
        let mut coeffs = [0u64; N];
        coeffs[i] = 1;
        let encoded: R = encode(&Pt::new(p(), coeffs), q());
        sk.encrypt(&encoded, Distribution::Ternary, &mut prg)
    });
    let out = cmux_tree(&bits, &mut inputs, BASE, BASE);
    assert_coeffs(&out.mask, &data::CMUX_TREE_IDX3_MASK, "cmux_tree mask");
    assert_coeffs(&out.body, &data::CMUX_TREE_IDX3_BODY, "cmux_tree body");
}

#[test]
fn kat_dmux_tree() {
    let mut prg = Shake256Prg::new(b"layer4-kat-dmux-tree");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let b0 = rgsw_bit(1, &sk, &mut prg);
    let b1 = rgsw_bit(0, &sk, &mut prg);
    let ct = enc(&[1, 1], &sk, &mut prg);
    let mut out = [ct; 4];
    dmux_tree(&[b0, b1], ct, &mut out, BASE, BASE);
    assert_coeffs(
        &out[0].mask,
        &data::DMUX_TREE_IDX2_OUT0_MASK,
        "dmux_tree out0 mask",
    );
    assert_coeffs(
        &out[0].body,
        &data::DMUX_TREE_IDX2_OUT0_BODY,
        "dmux_tree out0 body",
    );
    assert_coeffs(
        &out[1].mask,
        &data::DMUX_TREE_IDX2_OUT1_MASK,
        "dmux_tree out1 mask",
    );
    assert_coeffs(
        &out[1].body,
        &data::DMUX_TREE_IDX2_OUT1_BODY,
        "dmux_tree out1 body",
    );
    assert_coeffs(
        &out[2].mask,
        &data::DMUX_TREE_IDX2_OUT2_MASK,
        "dmux_tree out2 mask",
    );
    assert_coeffs(
        &out[2].body,
        &data::DMUX_TREE_IDX2_OUT2_BODY,
        "dmux_tree out2 body",
    );
    assert_coeffs(
        &out[3].mask,
        &data::DMUX_TREE_IDX2_OUT3_MASK,
        "dmux_tree out3 mask",
    );
    assert_coeffs(
        &out[3].body,
        &data::DMUX_TREE_IDX2_OUT3_BODY,
        "dmux_tree out3 body",
    );
}

#[test]
fn kat_crot_forward() {
    let mut prg = Shake256Prg::new(b"layer4-kat-crot-forward");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let bits: [RGSWCiphertext<N, R, DEPTH, DEPTH>; 2] =
        core::array::from_fn(|_| rgsw_bit(1, &sk, &mut prg));
    let ct = enc(&[1], &sk, &mut prg);
    let out = crot(CRotDir::Forward, &bits, ct, BASE, BASE);
    assert_coeffs(&out.mask, &data::CROT_FORWARD_ROT3_MASK, "crot fwd mask");
    assert_coeffs(&out.body, &data::CROT_FORWARD_ROT3_BODY, "crot fwd body");
}

#[test]
fn kat_crot_slot_extract() {
    let mut prg = Shake256Prg::new(b"layer4-kat-crot-slot-extract");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let bits: [RGSWCiphertext<N, R, DEPTH, DEPTH>; 2] =
        core::array::from_fn(|_| rgsw_bit(1, &sk, &mut prg));
    let ct = enc(&[1], &sk, &mut prg);
    let out = crot(CRotDir::SlotExtract, &bits, ct, BASE, BASE);
    assert_coeffs(
        &out.mask,
        &data::CROT_SLOT_EXTRACT_ROT3_MASK,
        "crot slot mask",
    );
    assert_coeffs(
        &out.body,
        &data::CROT_SLOT_EXTRACT_ROT3_BODY,
        "crot slot body",
    );
}

/// The two CRot directions must produce byte-distinct outputs (the lock for the
/// parameterised `CRotDir` decision).
#[test]
fn kat_crot_directions_differ() {
    assert_ne!(
        data::CROT_FORWARD_ROT3_MASK,
        data::CROT_SLOT_EXTRACT_ROT3_MASK
    );
    assert_ne!(
        data::CROT_FORWARD_ROT3_BODY,
        data::CROT_SLOT_EXTRACT_ROT3_BODY
    );
}

#[test]
fn kat_rlwe_to_rgsw() {
    let mut prg = Shake256Prg::new(b"layer4-kat-rlwe-to-rgsw");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let m_poly = bit_poly(1); // constant 1
    let g = gadget_vector_values::<N, R, DEPTH>(q(), BASE);
    let ct_levels: [RLWECiphertext<N, R>; DEPTH] = core::array::from_fn(|i| {
        let mut gc = [0u128; N];
        gc[0] = g[i];
        let g_poly = <R as RingPoly<N>>::from_u128_coeffs(q(), &gc);
        let scaled = m_poly * g_poly;
        sk.encrypt(&scaled, Distribution::Ternary, &mut prg)
    });
    let s_squared = *sk.poly() * *sk.poly();
    let conv_key =
        sk.encrypt_rlev::<CK_DEPTH>(&s_squared, CK_BASE, Distribution::Ternary, &mut prg);
    let m_rlev = sk.encrypt_rlev::<DEPTH>(&m_poly, BASE, Distribution::Ternary, &mut prg);
    let ct_prime = enc(&[1, 1], &sk, &mut prg);

    let rgsw = rlwe_to_rgsw::<N, R, DEPTH, CK_DEPTH>(ct_levels, &conv_key, m_rlev, CK_BASE);
    assert_coeffs(
        &rgsw.neg_s_m.samples[0].mask,
        &data::RLWE_TO_RGSW_NEG_S_M_L0_MASK,
        "r2r neg_s_m[0] mask",
    );
    assert_coeffs(
        &rgsw.neg_s_m.samples[0].body,
        &data::RLWE_TO_RGSW_NEG_S_M_L0_BODY,
        "r2r neg_s_m[0] body",
    );

    let ep = rgsw.external_product(&ct_prime, BASE, BASE);
    assert_coeffs(
        &ep.mask,
        &data::RLWE_TO_RGSW_EXT_PRODUCT_MASK,
        "r2r ext_product mask",
    );
    assert_coeffs(
        &ep.body,
        &data::RLWE_TO_RGSW_EXT_PRODUCT_BODY,
        "r2r ext_product body",
    );
}

#[test]
fn kat_mod_switch_rgsw() {
    let mut prg = Shake256Prg::new(b"layer4-kat-mod-switch-rgsw");
    let sk = SecretKey::<N, R>::keygen(q(), Distribution::Ternary, &mut prg);
    let rgsw = rgsw_bit(1, &sk, &mut prg);
    let q_small = DynModulus::new(Q_SMALL);
    let switched: RGSWCiphertext<N, R, DEPTH, DEPTH> = mod_switch_rgsw(&rgsw, q_small);
    assert_coeffs(
        &switched.neg_s_m.samples[0].mask,
        &data::MOD_SWITCH_RGSW_NEG_S_M_L0_MASK,
        "msr neg_s_m[0] mask",
    );
    assert_coeffs(
        &switched.neg_s_m.samples[0].body,
        &data::MOD_SWITCH_RGSW_NEG_S_M_L0_BODY,
        "msr neg_s_m[0] body",
    );
    assert_coeffs(
        &switched.m.samples[0].mask,
        &data::MOD_SWITCH_RGSW_M_L0_MASK,
        "msr m[0] mask",
    );
    assert_coeffs(
        &switched.m.samples[0].body,
        &data::MOD_SWITCH_RGSW_M_L0_BODY,
        "msr m[0] body",
    );
}
