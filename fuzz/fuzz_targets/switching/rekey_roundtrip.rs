//! Fuzz: §3.4 secret-key rekeying preserves coefficients and decryptability.
//!
//! Build a ternary key at `q_src`, rekey it to `q_dst` (§3.4), then encrypt
//! and decrypt under the rekeyed key — the plaintext must round-trip, and
//! every rekeyed coefficient must lie in `[0, q_dst)`. Catches a centring or
//! reduction regression that would shift a key coefficient (e.g. storing
//! `-1` as `q_src - 1` instead of re-centring before reducing mod `q_dst`).
//!
//! Single-prime `DynModulus` carrier, `N = 16`.
//!
//! Run with `cargo +nightly fuzz run switching_rekey_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::abstraction::RingPoly;
use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::{SecretKey, encode};
use via_rs::sampling::{Distribution, Shake256Prg};
use via_rs::switching::rekey::rekey_secret_key;

const N: usize = 16;
type R = Poly<N, DynModulus, Coefficient>;

/// Source / destination moduli; `q_dst` must comfortably exceed
/// `2 * ||S||_inf = 2` so the centred ternary key re-interprets cleanly.
const KNOWN_Q: &[u64] = &[1024, 65536, 8_380_417, 2_147_352_577];
const KNOWN_P: &[u64] = &[2, 16];

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    src_idx: u8,
    dst_idx: u8,
    p_idx: u8,
    plaintext: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let sl = u.int_in_range::<usize>(1..=32)?;
        let mut sk_seed = vec![0u8; sl];
        u.fill_buffer(&mut sk_seed)?;
        let el = u.int_in_range::<usize>(1..=32)?;
        let mut enc_seed = vec![0u8; el];
        u.fill_buffer(&mut enc_seed)?;
        let src_idx = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
        let dst_idx = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
        let p_idx = u.int_in_range::<u8>(0..=(KNOWN_P.len() as u8 - 1))?;
        let mut plaintext = [0u64; N];
        for slot in &mut plaintext {
            *slot = u.arbitrary()?;
        }
        Ok(Input {
            sk_seed,
            enc_seed,
            src_idx,
            dst_idx,
            p_idx,
            plaintext,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_src = KNOWN_Q[input.src_idx as usize];
    let q_dst = KNOWN_Q[input.dst_idx as usize];
    let p_value = KNOWN_P[input.p_idx as usize];
    // Need q_dst >= p (so encode is well-defined) and large enough for a
    // ternary key (|coeff| <= 1 << q_dst/2 trivially holds for all KNOWN_Q).
    if q_dst < p_value {
        return;
    }

    let qs = DynModulus::new(q_src);
    let qd = DynModulus::new(q_dst);
    let p = DynModulus::new(p_value);

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk_src = SecretKey::<N, R>::keygen(qs, Distribution::Ternary, &mut sk_prg);
    let sk_dst: SecretKey<N, R> = rekey_secret_key(&sk_src, qd);

    // Every rekeyed coefficient is canonical in [0, q_dst).
    let mut coeffs = [0u128; N];
    sk_dst.poly().to_u128_coeffs(&mut coeffs);
    for (i, &c) in coeffs.iter().enumerate() {
        assert!(
            c < u128::from(q_dst),
            "rekeyed coeff {i} = {c} >= q_dst={q_dst}"
        );
    }

    // The centred coefficients are unchanged by the rekey (same integers).
    let mut src_c = [0i64; N];
    let mut dst_c = [0i64; N];
    sk_src.poly().to_centered_coeffs(&mut src_c);
    sk_dst.poly().to_centered_coeffs(&mut dst_c);
    assert_eq!(src_c, dst_c, "rekey changed the centred coefficient vector");

    // Encrypt-decrypt round-trip under the rekeyed key.
    let mut lanes = [0u64; N];
    for (slot, &raw) in lanes.iter_mut().zip(input.plaintext.iter()) {
        *slot = raw % p_value;
    }
    let plaintext = R::new(p, lanes);
    let encoded: R = encode(&plaintext, qd);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let ct = sk_dst.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
    let recovered: R = sk_dst.decrypt(&ct, p);
    for (i, &expected) in lanes.iter().enumerate() {
        assert_eq!(
            recovered.coeff(i).to_u64(),
            expected,
            "rekey round-trip diverged at i={i}; q_src={q_src} q_dst={q_dst} p={p_value}",
        );
    }
});
