//! **Empirical validation of Yue's DMux/CMux noise-bound bug.**
//!
//! The paper's Lemmas C.1/C.2 bound the DMux external-product
//! approximate-decomposition residual as `~ n·q²/B^{2ℓ}` — with **no** secret
//! factor. Yue's claim (and the paper's own C.3/[45] lemmas) is that the term
//! actually scales with the secret second moment `E[s²]`, so the published
//! bound only holds for a small/binary secret and is too aggressive for a
//! large-variance (Gaussian) key.
//!
//! This test settles it empirically at the SECURE dimensions (n1=4096, q1≈2^74.7
//! RNS, erratum DMux gadget B=18073, ℓ=2). For these parameters the residual
//! dwarfs every other DMux noise term, so the measured DMux output-noise
//! variance *is* the residual. We run one DMux per secret distribution of
//! increasing `E[s²]` and watch the variance:
//!
//! - paper C.1/C.2 (`×1`) ⇒ variance **constant** across secrets,
//! - Yue (`×E[s²]`)        ⇒ variance **∝ E[s²]**.
//!
//! Run (heavy n4096 RNS): `cargo test -p via-primitives --release --features alloc
//! --test dmux_noise_empirical -- --ignored --nocapture`

#![cfg(feature = "alloc")]

use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::ring::rns_element::PolyRns;
use via_primitives::algebra::rns::basis::paper::ViaSecQ1Rns;
use via_primitives::algebra::zq::modulus::paper::ViaSecP;
use via_primitives::encryption::rlwe::encode;
use via_primitives::encryption::{RLWECiphertext, SecretKey};
use via_primitives::gates::dmux;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;

const N1: usize = 4096;
const BASE: u64 = 18073; // erratum DMux base (SECURE_PARAMS.gadget_base_1)
const L: usize = 2; // erratum DMux gadget length (= L_QUERY)
const ERR_SIGMA: f64 = 4.0; // fixed encryption error — only the SECRET varies

type R1 = PolyRns<N1, ViaSecQ1Rns, Coefficient>;
type Rp = Poly<N1, ViaSecP, Coefficient>;

/// Sample-variance (over the N1 coefficients) of the center-lifted noise in a
/// ciphertext that should decrypt to `expected` (Δ-encoded at q1).
fn noise_variance(sk: &SecretKey<N1, R1>, ct: &RLWECiphertext<N1, R1>, expected: &R1) -> f64 {
    let raw = sk.decrypt_raw(ct);
    let noise = raw - *expected;
    let mut centered = [0i128; N1];
    noise.to_centered_coeffs(&mut centered);
    let sum_sq: f64 = centered.iter().map(|&c| (c as f64) * (c as f64)).sum();
    sum_sq / (N1 as f64)
}

/// One DMux at the secure params under a secret drawn from `secret_dist`;
/// returns the measured output-noise variance of the selected branch.
fn dmux_noise(secret_dist: Distribution, seed: &[u8]) -> f64 {
    let q1 = ViaSecQ1Rns::default();
    let p = ViaSecP::default();
    let mut prg = Shake256Prg::new(seed);

    // Secret with the chosen distribution (this is the variable under test).
    let sk = SecretKey::<N1, R1>::keygen(q1, secret_dist, &mut prg);

    // RGSW encrypting the control bit 1 (constant-term-1 polynomial).
    let mut one = [0u128; N1];
    one[0] = 1;
    let one_poly = R1::from_u128_coeffs(q1, &one);
    let rgsw = sk.encrypt_rgsw::<L, L>(
        &one_poly,
        BASE,
        BASE,
        Distribution::Gaussian { sigma: ERR_SIGMA },
        &mut prg,
    );

    // A fresh RLWE encryption of a small message (Δ-scaled at q1).
    let msg: Rp = Poly::new(
        p,
        core::array::from_fn(|i| if i < 4 { (i as u64) + 1 } else { 0 }),
    );
    let encoded: R1 = encode(&msg, q1);
    let ct = sk.encrypt(
        &encoded,
        Distribution::Gaussian { sigma: ERR_SIGMA },
        &mut prg,
    );

    // bit = 1 ⇒ the "selected" branch r1 ≈ Enc(message); measure its noise.
    let (_r0, r1) = dmux(&rgsw, &ct, BASE, BASE);
    noise_variance(&sk, &r1, &encoded)
}

#[test]
#[ignore = "heavy n4096 RNS DMux — run with --release -- --ignored --nocapture"]
fn dmux_residual_scales_with_secret_variance() {
    // n4096 RNS objects are large; run on a generous stack like the e2e tests.
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(run_experiment)
        .expect("spawn")
        .join()
        .expect("experiment thread panicked");
}

fn run_experiment() {
    // The analytic residual base (paper ×1 value): (n1+1)·q1²/(12·B^{2ℓ}).
    let q1f = 173_964_607_489f64 * 173_964_656_641f64;
    let residual_base = (N1 as f64 + 1.0) * q1f * q1f / (12.0 * (BASE as f64).powi(2 * L as i32));

    // Secret distributions with increasing E[s²].
    let cases: [(&str, Distribution, f64); 4] = [
        ("ternary", Distribution::Ternary, 2.0 / 3.0),
        ("gauss σ=4", Distribution::Gaussian { sigma: 4.0 }, 16.0),
        ("gauss σ=16", Distribution::Gaussian { sigma: 16.0 }, 256.0),
        ("gauss σ=32", Distribution::Gaussian { sigma: 32.0 }, 1024.0),
    ];

    println!("\nDMux output-noise at n1={N1}, q1≈2^74.7, B={BASE}, ℓ={L}");
    println!("analytic residual base (paper ×1) = {residual_base:.3e}\n");
    println!(
        "{:<11} {:>8} {:>12} {:>12} {:>10}",
        "secret", "E[s^2]", "measured var", "var/base", "ratio→tern"
    );

    let mut results = vec![];
    for (name, dist, e2) in cases {
        // average two seeds for a steadier estimate
        let v = (dmux_noise(dist, b"dmux-noise-a") + dmux_noise(dist, b"dmux-noise-b")) / 2.0;
        results.push((name, e2, v));
    }

    let tern_var = results[0].2;
    for (name, e2, v) in &results {
        println!(
            "{:<11} {:>8.2} {:>12.3e} {:>12.3} {:>10.1}",
            name,
            e2,
            v,
            v / residual_base,
            v / tern_var
        );
    }

    // --- Decisive checks ---------------------------------------------------
    // (1) var/base ≈ E[s²] for each secret (Yue's factor), within 3×.
    for (name, e2, v) in &results {
        let ratio = v / residual_base;
        assert!(
            ratio > e2 / 3.0 && ratio < e2 * 3.0,
            "{name}: var/base = {ratio:.3} not ≈ E[s²] = {e2} (Yue's factor)"
        );
    }
    // (2) the σ=32 secret produces ≫ 100× the ternary noise — refuting the
    //     paper's "constant in the secret" (×1) bound, which predicts ≈1×.
    let (_, _, gauss32_var) = results[3];
    assert!(
        gauss32_var / tern_var > 100.0,
        "σ=32 / ternary variance ratio = {:.1}; paper ×1 predicts ≈1, Yue predicts ≈1536",
        gauss32_var / tern_var
    );

    // (3) NEGATIVE WITNESS (per Rohit's PR review): for the paper-prescribed
    //     Gaussian secret (σ=32), the measured DMux residual EXCEEDS the residual
    //     the paper's own C.1/C.2 bound budgets for it (`residual_base`, the ×1
    //     value) by ~1000× — i.e. the published bound is violated at the gate.
    //     This is the primitive-level half of "error not within budget for paper
    //     params"; the emergent decryption-failure (P_fail > 2^-40 once this is
    //     scaled through the I=2^11 recursion) is asserted analytically in
    //     `via-estimator`'s `yue_correction_breaks_paper_gaussian`. The
    //     conversion itself is NOT a valid negative witness — it keeps ~46-bit
    //     headroom even at σ=32 (see `conversion_noise_secure`).
    assert!(
        gauss32_var / residual_base > 100.0,
        "σ=32 residual is {:.0}× the paper-budgeted residual — bound NOT violated?",
        gauss32_var / residual_base
    );
}
