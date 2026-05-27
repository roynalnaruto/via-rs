//! Fuzz: `rgsw.external_product(rlwe).decrypt(p) == m1 · m2 mod (p, X^N+1)`.
//!
//! End-to-end homomorphic-multiplication round-trip on the single-prime
//! backend. Exercises the full Phase-3 to Phase-7 pipeline under random
//! inputs:
//!
//! - Phase 3: `SecretKey::keygen`, `encode`, `encrypt`, `decrypt`.
//! - Phase 5: `gadget_vector_values`, `gadget_scale_into`,
//!   `gadget_extract_lsb_into`, the wide-arithmetic `round_mul_div_u128`.
//! - Phase 6: `encrypt_rgsw`.
//! - Phase 7: `gadget_product` (twice, via `external_product`), `Add`
//!   on `RLWECiphertext`.
//!
//! Targeted regression classes:
//!
//! - LSB/MSB swap in `gadget_product`'s digit-to-sample pairing —
//!   would silently corrupt every multiplication.
//! - Base/depth threading bug in `external_product` (one half's base
//!   accidentally used for the other).
//! - Noise blow-up under non-trivial plaintexts.
//!
//! Run with `cargo +nightly fuzz run encryption_external_product_homomorphism`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::{SecretKey, encode};
use via_rs::sampling::{Distribution, Shake256Prg};

// `N = 4` keeps both noise-budget terms achievable across a meaningful
// fraction of the (q, p, B) fuzz space — at `N = 8` the worst-case
// `N² · L · B · p` constraint forces q above ~2 MiB, leaving only the
// paper-class single-prime moduli.
const N: usize = 4;
const L1: usize = 4;
const L2: usize = 4;

/// `(q, p, B)` combinations where the noise budget after a small-norm
/// external product comfortably exceeds the bound. Curated so the
/// fuzzer doesn't spend cycles on parameter sets that can never
/// round-trip.
const KNOWN_Q: &[u64] = &[
    1024,
    65537,
    1_048_583,      // ~2^20
    8_380_417,      // VIA-C q_3
    17_175_674_881, // VIA-C q_2
];

const KNOWN_P: &[u64] = &[2, 16, 256];

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    q_index: u8,
    p_index: u8,
    base: u8,
    /// Per-coefficient bit picks for m1 (binary plaintext, small norm).
    m1_bits: u32,
    /// Per-coefficient values for m2 (reduced mod p inside the body).
    m2_coeffs: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let sk_len = u.int_in_range::<usize>(1..=32)?;
        let mut sk_seed = vec![0u8; sk_len];
        u.fill_buffer(&mut sk_seed)?;
        let enc_len = u.int_in_range::<usize>(1..=32)?;
        let mut enc_seed = vec![0u8; enc_len];
        u.fill_buffer(&mut enc_seed)?;
        let q_index = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
        let p_index = u.int_in_range::<u8>(0..=(KNOWN_P.len() as u8 - 1))?;
        // Base ∈ [2, 64] — keep B^L bounded, leave plenty of noise headroom.
        let base = u.int_in_range::<u8>(2..=64)?;
        let m1_bits: u32 = u.arbitrary()?;
        let mut m2_coeffs = [0u64; N];
        for slot in m2_coeffs.iter_mut() {
            *slot = u.arbitrary()?;
        }
        Ok(Input {
            sk_seed,
            enc_seed,
            q_index,
            p_index,
            base,
            m1_bits,
            m2_coeffs,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_value = KNOWN_Q[input.q_index as usize];
    let p_value = KNOWN_P[input.p_index as usize];
    let base = u64::from(input.base);
    let b_pow_l: u128 = (base as u128).pow(L1 as u32);

    // Feasibility gate. The dominant noise after `external_product`
    // is `δ · s · m1` where `δ ≈ round(q/B^L)/2 + L·B/4` is the
    // gadget-reconstruction error (Phase-5 derivation). For the
    // decoder's round-to-Δ/2 step not to wrap, the decode-scaled
    // noise must stay under `1/2`:
    //
    //     N² · (q/(2·B^L) + L·B/4) · p / q  <  1/4   (2x safety)
    //
    // Splitting into two terms gives two constraints (each must hold
    // with 2x safety margin):
    //
    //   (A) `B^L > 4 · N² · p`    — bounds the `g_min/2` term.
    //   (B) `q   > 2 · N² · L · B · p` — bounds the `L·B/4` slack term.
    //
    // Both worst-case `||s||_1 = ||m1||_1 = N`.
    let n_sq: u128 = (N as u128) * (N as u128);
    let p128 = u128::from(p_value);
    let required_b_pow_l: u128 = 4 * n_sq * p128;
    let required_q: u128 = 2 * n_sq * (L1 as u128) * (base as u128) * p128;
    if b_pow_l < required_b_pow_l || u128::from(q_value) < required_q {
        return;
    }
    // Sanity: don't degenerate the gadget.
    if b_pow_l > u128::from(q_value) {
        return;
    }

    let q = DynModulus::new(q_value);
    let p = DynModulus::new(p_value);

    // Build m1 as binary (small-norm); at most ||m1||_1 = N.
    let mut m1_coeffs = [0u64; N];
    for (i, slot) in m1_coeffs.iter_mut().enumerate() {
        *slot = u64::from((input.m1_bits >> i) & 1);
    }
    let m1: Poly<N, DynModulus, Coefficient> = Poly::new(q, m1_coeffs);

    let mut m2_coeffs_at_p = [0u64; N];
    for (out, &raw) in m2_coeffs_at_p.iter_mut().zip(input.m2_coeffs.iter()) {
        *out = raw % p_value;
    }
    let m2: Poly<N, DynModulus, Coefficient> = Poly::new(p, m2_coeffs_at_p);

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, Poly<N, DynModulus, Coefficient>>::keygen(
        q,
        Distribution::Ternary,
        &mut sk_prg,
    );
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);

    let rgsw = sk.encrypt_rgsw::<L1, L2>(&m1, base, base, Distribution::Ternary, &mut enc_prg);
    let encoded_m2: Poly<N, DynModulus, Coefficient> = encode(&m2, q);
    let rlwe = sk.encrypt(&encoded_m2, Distribution::Ternary, &mut enc_prg);

    let ct = rgsw.external_product(&rlwe, base, base);
    let recovered: Poly<N, DynModulus, Coefficient> = sk.decrypt(&ct, p);

    // Compute expected: m1 · m2 mod (p, X^N+1). Both polynomials at p.
    let m1_at_p: Poly<N, DynModulus, Coefficient> = Poly::new(p, m1_coeffs);
    let expected = m1_at_p * m2;
    for i in 0..N {
        assert_eq!(
            recovered.coeff(i).to_u64(),
            expected.coeff(i).to_u64(),
            "external-product round-trip diverged at coeff {i}; q={q_value} p={p_value} base={base}"
        );
    }
});
