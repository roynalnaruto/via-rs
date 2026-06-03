//! Cross-language KAT parity for the Layer-5 MLWE LWE→RLWE conversion cascade
//! (§5.1–§5.4).
//!
//! Each test reproduces — in Rust, with the same per-component seeds and PRG
//! draw order — a setup that
//! `.references/via-spec/scripts/gen_layer5_kats.py` ran in Python, then asserts
//! byte-for-byte equality against the generated `data::*` constants.
//!
//! `kat_encrypt_lwe` locks the `n`-masks-then-error PRG order of `encrypt_lwe`;
//! `kat_gen_lwe_to_rlwe_key` locks the step/group/j key-generation order; and
//! `kat_conv_d` is the first cross-language check of the `conv_step` /
//! `key_switch` sign convention. §5.5 `extr` has no KAT (absent from the Python
//! reference — see `src/conversion/extract.rs` for its Rust round-trip tests).
//!
//! Regenerate the constants with `just regen-kats-layer5`.

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{
    conv_step, embed_mlwe, encrypt_lwe, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8, rlwe_to_mlwe,
};
use via_primitives::encryption::MLWECiphertext;
use via_primitives::encryption::rlwe::encode;
use via_primitives::encryption::types::SecretKey;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;

mod data {
    include!("data/layer5_kats.rs");
}

// TOY parameters, matching gen_layer5_kats.py.
const Q: u64 = 65537;
const P: u64 = 16;
const BASE: u64 = 8;
const DEPTH: usize = 6;
const MESSAGE: u64 = 5;

type R8 = Poly<8, DynModulus, Coefficient>;
type R4 = Poly<4, DynModulus, Coefficient>;

fn q() -> DynModulus {
    DynModulus::new(Q)
}
fn p() -> DynModulus {
    DynModulus::new(P)
}

/// Assert the first `expected.len()` coefficients of a single-prime poly match.
fn assert_poly<const D: usize>(
    poly: &Poly<D, DynModulus, Coefficient>,
    expected: &[u64],
    label: &str,
) {
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(poly.coeff(i).to_u64(), e, "{label}: coeff {i} mismatch");
    }
}

#[test]
fn kat_embed_mlwe() {
    let key_prg = &mut Shake256Prg::new(b"layer5-kat-embed-key");
    let sk = SecretKey::<4, R4>::keygen(q(), Distribution::Ternary, key_prg);
    let enc_prg = &mut Shake256Prg::new(b"layer5-kat-embed-enc");
    let msg = R4::new(p(), [1, 2, 3, 4]);
    let encoded = encode::<4, R4, R4>(&msg, q());
    let ct = sk.encrypt(&encoded, Distribution::Ternary, enc_prg);
    let embedded: MLWECiphertext<1, 8, R8> = embed_mlwe(&rlwe_to_mlwe(&ct));
    assert_poly(&embedded.masks[0], &data::EMBED_MLWE_MASK0, "embed mask0");
    assert_poly(&embedded.body, &data::EMBED_MLWE_BODY, "embed body");
}

#[test]
fn kat_encrypt_lwe() {
    let key_prg = &mut Shake256Prg::new(b"layer5-kat-encrypt-lwe-key");
    let sk = SecretKey::<8, R8>::keygen(q(), Distribution::Ternary, key_prg);
    let enc_prg = &mut Shake256Prg::new(b"layer5-kat-encrypt-lwe-enc");
    let lwe = encrypt_lwe(&sk, MESSAGE, P, Distribution::Ternary, enc_prg);
    for (i, &e) in data::ENCRYPT_LWE_MASKS.iter().enumerate() {
        assert_eq!(lwe.masks[i].coeff(0).to_u64(), e, "encrypt_lwe mask {i}");
    }
    assert_eq!(
        lwe.body.coeff(0).to_u64(),
        data::ENCRYPT_LWE_BODY[0],
        "encrypt_lwe body"
    );
}

#[test]
fn kat_conv_d() {
    let key_prg = &mut Shake256Prg::new(b"layer5-kat-conv-key");
    let sk = SecretKey::<8, R8>::keygen(q(), Distribution::Ternary, key_prg);
    let enc_prg = &mut Shake256Prg::new(b"layer5-kat-conv-enc");
    let lwe = encrypt_lwe(&sk, MESSAGE, P, Distribution::Ternary, enc_prg);
    let gen_prg = &mut Shake256Prg::new(b"layer5-kat-conv-keygen");
    let key = gen_lwe_to_rlwe_key_n8::<_, DEPTH>(&sk, BASE, Distribution::Ternary, gen_prg);
    // One Conv₂ step (8,1) -> (4,2), consuming step 0's 8 keys.
    let out = conv_step::<8, 1, 4, 2, _, _, DEPTH>(&lwe, &key.keys_2, BASE);
    assert_poly(&out.masks[0], &data::CONV_D_OUT_MASK0, "conv_d mask0");
    assert_poly(&out.body, &data::CONV_D_OUT_BODY, "conv_d body");
}

#[test]
fn kat_gen_lwe_to_rlwe_key() {
    let key_prg = &mut Shake256Prg::new(b"layer5-kat-keygen-key");
    let sk = SecretKey::<8, R8>::keygen(q(), Distribution::Ternary, key_prg);
    let gen_prg = &mut Shake256Prg::new(b"layer5-kat-keygen-gen");
    let key = gen_lwe_to_rlwe_key_n8::<_, DEPTH>(&sk, BASE, Distribution::Ternary, gen_prg);
    // step_keys[0].samples[0].mask == keys_2[0].samples[0].mask (degree 2).
    assert_poly(
        &key.keys_2[0].samples[0].mask,
        &data::LWE2RLWE_KEY_STEP0_S0_L0_MASK,
        "gen_lwe_to_rlwe_key step0 s0 l0 mask",
    );
}

#[test]
fn kat_lwe_to_rlwe() {
    let key_prg = &mut Shake256Prg::new(b"layer5-kat-cascade-key");
    let sk = SecretKey::<8, R8>::keygen(q(), Distribution::Ternary, key_prg);
    let enc_prg = &mut Shake256Prg::new(b"layer5-kat-cascade-enc");
    let lwe = encrypt_lwe(&sk, MESSAGE, P, Distribution::Ternary, enc_prg);
    let gen_prg = &mut Shake256Prg::new(b"layer5-kat-cascade-keygen");
    let key = gen_lwe_to_rlwe_key_n8::<_, DEPTH>(&sk, BASE, Distribution::Ternary, gen_prg);
    let rlwe = lwe_to_rlwe_n8::<_, DEPTH>(&lwe, &key, BASE);
    assert_poly(&rlwe.mask, &data::LWE2RLWE_OUT_MASK, "lwe_to_rlwe mask");
    assert_poly(&rlwe.body, &data::LWE2RLWE_OUT_BODY, "lwe_to_rlwe body");
}
