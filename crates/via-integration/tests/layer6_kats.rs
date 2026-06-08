//! Cross-language KAT parity for the Layer-6 VIA-C protocol composites
//! (§4.2 QueryComp, §4.3 RespComp).
//!
//! Each test reproduces — in Rust, with the same per-component seeds and PRG
//! draw order — a setup that
//! `.references/via-spec/scripts/gen_layer6_kats.py` ran in Python, then asserts
//! byte-for-byte equality against the generated `data::*` constants.
//!
//! These live in `via-integration` (not `via-primitives`) because the composites
//! span crates: `kat_query_comp` drives the real `via-client` query path and
//! `kat_resp_comp` the `via-server` RespComp.
//!
//! **Paper-over-spec.** `kat_resp_comp` checks the **paper's asymmetric** Figure-7
//! path (`mod_switch_sym → ring_switch → mod_switch_asym`), which is what Rust
//! `resp_comp` implements. The Python reference `resp_comp.py` is deliberately
//! *symmetric* and would NOT match — so the generator reconstructs the paper path
//! from the reference's primitives instead (see the script's PAPER-over-spec
//! note). Where a VIA variant diverges from the Python spec, the paper wins.
//!
//! Regenerate the constants with `just regen-kats-layer6`.

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{encrypt_lwe_raw, gen_lwe_to_rlwe_key_n8};
use via_primitives::encryption::rlwe::encode;
use via_primitives::encryption::types::SecretKey;
use via_primitives::gates::gen_rlwe_to_rgsw_key;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{KeyDist, PIRParams};
use via_server::resp_comp;

mod data {
    include!("data/layer6_kats.rs");
}

// TOY parameters, matching gen_layer6_kats.py and client_server_e2e.rs.
const N1: usize = 8;
const N2: usize = 4;
const D: usize = 2; // d = N1 / N2
const L_QUERY: usize = 7;
const L_CK: usize = 7;
const L_RSK: usize = 8;

const Q1: u64 = 1 << 36;
const Q2: u64 = 1 << 28;
const Q3: u64 = 1 << 20;
const Q4: u64 = 1 << 12;
const P: u64 = 16;

const B_QUERY: u64 = 64;
const CK_BASE: u64 = 64;
const B_RSK: u64 = 8;

const NUM_ROWS: usize = 2; // I
const NUM_COLS: usize = 2; // J

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;
type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;

/// Assert `poly.coeff(i) == expected[i]` for every `expected[i]`.
fn assert_coeffs<const DEG: usize>(
    poly: &Poly<DEG, DynModulus, Coefficient>,
    expected: &[u64],
    label: &str,
) {
    for (i, &e) in expected.iter().enumerate() {
        assert_eq!(poly.coeff(i).to_u64(), e, "{label}: coeff {i} mismatch");
    }
}

/// The shared toy `PIRParams` (identical to `client_server_e2e.rs`).
fn toy_params() -> PIRParams {
    PIRParams::new(
        N1, N2, Q1 as u128, Q2, Q3, Q4, P, //
        B_QUERY, L_QUERY, B_QUERY, L_QUERY, B_RSK, L_RSK, //
        KeyDist::Ternary, KeyDist::Ternary, 1, None, None, None, 40,
    )
}

/// KAT A — `encrypt_lwe_raw` (Δ-free raw LWE): locks the n-masks-then-error PRG
/// order at a non-trivial raw message (the query path encrypts `b·g_i` directly).
#[test]
fn kat_encrypt_lwe_raw() {
    let q1 = DynModulus::new(Q1);
    let key_prg = &mut Shake256Prg::new(b"layer6-kat-elr-key");
    let sk = SecretKey::<N1, R8>::keygen(q1, Distribution::Ternary, key_prg);
    let enc_prg = &mut Shake256Prg::new(b"layer6-kat-elr-enc");
    let message: u128 = 0xDEAD_BEEFu128 % (Q1 as u128);
    let lwe = encrypt_lwe_raw(&sk, message, Distribution::Ternary, enc_prg);
    for (i, &e) in data::ENCRYPT_LWE_RAW_MASKS.iter().enumerate() {
        assert_eq!(lwe.masks[i].coeff(0).to_u64(), e, "encrypt_lwe_raw mask {i}");
    }
    assert_eq!(
        lwe.body.coeff(0).to_u64(),
        data::ENCRYPT_LWE_RAW_BODY[0],
        "encrypt_lwe_raw body"
    );
}

/// KAT D — `gen_rlwe_to_rgsw_key` = RLev_S(S²): locks the S² = s·s (negacyclic)
/// computation and the `encrypt_rlev` draw order used by QueryCompSetup.
#[test]
fn kat_gen_rlwe_to_rgsw_key() {
    let q1 = DynModulus::new(Q1);
    let key_prg = &mut Shake256Prg::new(b"layer6-kat-ck-key");
    let sk = SecretKey::<N1, R8>::keygen(q1, Distribution::Ternary, key_prg);
    let gen_prg = &mut Shake256Prg::new(b"layer6-kat-ck-gen");
    let key = gen_rlwe_to_rgsw_key::<N1, R8, L_CK>(&sk, CK_BASE, Distribution::Ternary, gen_prg);
    assert_coeffs(
        &key.samples[0].mask,
        &data::RLWE_TO_RGSW_KEY_S0_MASK,
        "rlwe_to_rgsw_key sample0 mask",
    );
    assert_coeffs(
        &key.samples[0].body,
        &data::RLWE_TO_RGSW_KEY_S0_BODY,
        "rlwe_to_rgsw_key sample0 body",
    );
}

/// KAT C — `query_comp` via the **real** client path. The client's `S1` is the
/// first keygen of `setup`, so seeding setup with `"qc-setup"` reproduces the
/// Python `keygen("qc-setup")`; `query("qc-query")` then mirrors `query_comp`.
/// Index 5 → (α,β,γ) = (0,1,1): a 0 DMux bit and 1 CMux/CRot bits.
#[test]
fn kat_query_comp() {
    let q1 = DynModulus::new(Q1);
    let q3 = DynModulus::new(Q3);
    let setup_prg = &mut Shake256Prg::new(b"layer6-kat-qc-setup");
    let (client, _pp) = ToyClient::setup(
        q1,
        q3,
        toy_params(),
        NUM_ROWS,
        NUM_COLS,
        CK_BASE,
        Distribution::Ternary,
        Distribution::Ternary,
        Distribution::Ternary,
        setup_prg,
        |sk, base, dist, prg| {
            Box::new(gen_lwe_to_rlwe_key_n8::<DynModulus, L_CK>(
                sk, base, dist, prg,
            ))
        },
        |sk1, sk2, dist, prg| {
            let q3_mod = RingPoly::modulus(sk2.poly());
            let s1_q3 = rekey_secret_key::<N1, R8, R8>(sk1, q3_mod);
            gen_rsk::<N1, N2, R8, R4, L_RSK, D>(&s1_q3, sk2, B_RSK, dist, prg)
        },
    )
    .expect("client setup");

    let query_prg = &mut Shake256Prg::new(b"layer6-kat-qc-query");
    let query = client.query(5, query_prg).expect("client query");

    // 3 bits × L_QUERY levels = 21 LWE ciphertexts, ct-major mask layout.
    assert_eq!(query.ciphertexts.len(), 21, "query length");
    let mut mask_idx = 0;
    for (c, ct) in query.ciphertexts.iter().enumerate() {
        for (m, mask) in ct.masks.iter().enumerate() {
            assert_eq!(
                mask.coeff(0).to_u64(),
                data::QUERY_COMP_MASKS[mask_idx],
                "query ct {c} mask {m}"
            );
            mask_idx += 1;
        }
        assert_eq!(
            ct.body.coeff(0).to_u64(),
            data::QUERY_COMP_BODIES[c],
            "query ct {c} body"
        );
    }
}

/// KAT B — `resp_comp` on the **paper** asymmetric path (Figure 7). The generated
/// answer is `(mask @ q3, body @ q4)`; the Python side reconstructs this path
/// (its `resp_comp.py` is symmetric — see the module docs). This is the first
/// cross-language lock of `mod_switch_sym → ring_switch → mod_switch_asym`.
#[test]
fn kat_resp_comp() {
    let q2 = DynModulus::new(Q2);
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);

    // S1 @ q2 (answer key), S2 @ q3 (target key).
    let s1_prg = &mut Shake256Prg::new(b"layer6-kat-rc-s1");
    let sk1 = SecretKey::<N1, R8>::keygen(q2, Distribution::Ternary, s1_prg);
    let s2_prg = &mut Shake256Prg::new(b"layer6-kat-rc-s2");
    let sk2 = SecretKey::<N2, R4>::keygen(q3, Distribution::Ternary, s2_prg);

    // Representative answer ciphertext at q2 under S1 (constant term = 3).
    let enc_prg = &mut Shake256Prg::new(b"layer6-kat-rc-enc");
    let msg = R8::new(p, [3, 0, 0, 0, 0, 0, 0, 0]);
    let ct = sk1.encrypt(&encode::<N1, R8, R8>(&msg, q2), Distribution::Ternary, enc_prg);

    // Ring-switch key S1→S2 @ q3 (rekey S1: q2→q3 first, then gen_rsk).
    let s1_q3 = rekey_secret_key::<N1, R8, R8>(&sk1, q3);
    let rsk_prg = &mut Shake256Prg::new(b"layer6-kat-rc-rsk");
    let rsk = gen_rsk::<N1, N2, R8, R4, L_RSK, D>(&s1_q3, &sk2, B_RSK, Distribution::Ternary, rsk_prg);

    // Paper path: sym q2→q3 → ring_switch n1→n2 @ q3 → asym q3→q4 (body only).
    let answer = resp_comp::<N1, N2, R8, R8, R4, R4, L_RSK, D>(&ct, &rsk, q3, q4, B_RSK);

    assert_coeffs(&answer.mask, &data::RESP_COMP_MASK_Q3, "resp_comp mask @ q3");
    assert_coeffs(&answer.body, &data::RESP_COMP_BODY_Q4, "resp_comp body @ q4");
}
