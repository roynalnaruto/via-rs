//! **Yue's recommended test #1: is the LWE→RLWE conversion's output noise
//! budget large enough?** (https://github.com/0xalizk/via-rs/issues/3)
//!
//! Yue's intuition: VIA-C's whole correctness argument rests on the
//! LWE→RLWE conversion being low-noise — "if test 1 works well with a decent
//! noise budget, VIA-C should work when DMux/CMux use generous parameters."
//! This measures it at the SECURE dimensions (n1=4096, q1≈2^74.7 RNS, ternary
//! keys, conversion gadget B=18, ℓ=18 — which fully covers q1): encrypt an LWE,
//! run the real n4096 cascade conversion, decrypt the RLWE, and report the
//! center-lifted noise against the decode budget Δ/2 = ⌈q1/p⌉/2.
//!
//! Run: `cargo test -p via-primitives --release --features alloc
//!       --test conversion_noise_secure -- --ignored --nocapture`

#![cfg(feature = "alloc")]

use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::ring::rns_element::PolyRns;
use via_primitives::algebra::rns::basis::paper::ViaSecQ1Rns;
use via_primitives::conversion::{
    encrypt_lwe, gen_lwe_to_rlwe_key_rns_n4096_boxed, lwe_to_rlwe_rns_n4096,
};
use via_primitives::encryption::SecretKey;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;

const N1: usize = 4096;
const P: u64 = 16; // ViaSecP plaintext modulus
const CK_BASE: u64 = 18; // conversion gadget base (18^18 ≈ 2^75 ⊇ q1)
const L_CK: usize = 18; // conversion gadget length
const MESSAGE: u64 = 5;

type R1 = PolyRns<N1, ViaSecQ1Rns, Coefficient>;

#[test]
#[ignore = "secure-scale n4096 RNS conversion (~54 MB key) — run with --release -- --ignored --nocapture"]
fn lwe_to_rlwe_conversion_noise_budget() {
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(run)
        .expect("spawn")
        .join()
        .expect("conversion-noise thread panicked");
}

fn run() {
    let q1 = ViaSecQ1Rns::default();
    let q1_val = <R1 as RingPoly<N1>>::modulus_value(q1); // u128, ≈2^74.7
    let delta = q1_val.div_ceil(u128::from(P)); // Δ = ⌈q1/p⌉
    let budget = delta / 2; // decode threshold Δ/2

    // Ternary keys + ternary error, exactly as via-rs runs.
    let mut prg = Shake256Prg::new(b"secure-conv-noise");
    let sk = SecretKey::<N1, R1>::keygen(q1, Distribution::Ternary, &mut prg);

    // LWE encrypting Δ·MESSAGE under S1, the n4096 cascade key, one conversion.
    let lwe = encrypt_lwe::<N1, R1>(&sk, MESSAGE, P, Distribution::Ternary, &mut prg);
    let key = gen_lwe_to_rlwe_key_rns_n4096_boxed::<ViaSecQ1Rns, L_CK>(
        &sk,
        CK_BASE,
        Distribution::Ternary,
        &mut prg,
    );
    let rlwe = lwe_to_rlwe_rns_n4096::<ViaSecQ1Rns, L_CK>(&lwe, &key, CK_BASE);

    // The message lands in the constant coefficient; everything else is noise.
    let raw = sk.decrypt_raw(&rlwe);
    let dm = (delta * u128::from(MESSAGE)) % q1_val;
    let expected = R1::from_u128_coeffs(q1, &core::array::from_fn(|i| if i == 0 { dm } else { 0 }));
    let noise = raw - expected;

    let mut centered = [0i128; N1];
    noise.to_centered_coeffs(&mut centered);
    let inf_norm = centered.iter().map(|c| c.unsigned_abs()).max().unwrap_or(0);
    let var = centered
        .iter()
        .map(|&c| (c as f64) * (c as f64))
        .sum::<f64>()
        / (N1 as f64);

    let log2 = |x: f64| if x > 0.0 { x.log2() } else { f64::NEG_INFINITY };
    let headroom_bits = log2(budget as f64) - log2(inf_norm as f64);

    println!(
        "\nLWE→RLWE conversion noise @ SECURE params (n1={N1}, q1≈2^74.7, B={CK_BASE}, ℓ={L_CK}, ternary)"
    );
    println!(
        "  Δ = ⌈q1/p⌉ ≈ 2^{:.1}   decode budget Δ/2 ≈ 2^{:.1}",
        log2(delta as f64),
        log2(budget as f64)
    );
    println!(
        "  noise ‖·‖∞ ≈ 2^{:.1}   σ ≈ 2^{:.1}",
        log2(inf_norm as f64),
        0.5 * log2(var)
    );
    println!("  headroom: {headroom_bits:.1} bits below the decode budget");

    // The conversion must decode (noise < Δ/2) with a wide margin — Yue's
    // "decent noise budget" condition. The margin is enormous here (~50 bits),
    // which is exactly why the conversion is the cheap part of VIA-C.
    assert!(
        inf_norm < budget,
        "conversion noise {inf_norm} exceeds decode budget {budget}"
    );
    assert!(
        headroom_bits > 30.0,
        "conversion budget headroom only {headroom_bits:.1} bits"
    );
}
