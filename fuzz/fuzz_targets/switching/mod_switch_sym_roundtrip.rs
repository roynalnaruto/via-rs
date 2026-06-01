//! Fuzz: §3.1 symmetric ModSwitch must preserve the plaintext.
//!
//! Encrypt at `q_src`, `mod_switch_sym` down to `q_dst`, rekey the secret key
//! to `q_dst` (§3.4) and decrypt — the recovered plaintext must equal the
//! original whenever the post-switch noise stays inside the destination
//! decoding budget. Catches rescale rounding-direction regressions and the
//! mask/body asymmetry that a wrong `RescaleConsts` would introduce.
//!
//! Single-prime `DynModulus` carrier, `N = 16`.
//!
//! Run with `cargo +nightly fuzz run switching_mod_switch_sym_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::types::RLWECiphertext;
use via_rs::encryption::{SecretKey, encode};
use via_rs::sampling::{Distribution, Shake256Prg};
use via_rs::switching::mod_switch::mod_switch_sym;
use via_rs::switching::rekey::rekey_secret_key;

const N: usize = 16;
type R = Poly<N, DynModulus, Coefficient>;

/// Sorted modulus list; the target picks `q_src` strictly larger than `q_dst`.
const KNOWN_Q: &[u64] = &[65536, 8_380_417, 2_147_352_577, 17_175_674_881];
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
    // Only a genuine down-switch is interesting.
    if q_dst >= q_src {
        return;
    }
    let p_value = KNOWN_P[input.p_idx as usize];
    let delta_dst = q_dst.div_ceil(p_value);

    // Feasibility: ternary error tail = 1, scaled by q_dst/q_src (< 1), plus
    // the mask/body rescale rounding (bounded by ~N/2 from `||S||_1`). Gate
    // conservatively so the assertion only fires for real regressions.
    let scaled_err = 1u64; // ternary tail * q_dst/q_src rounds to <= 1
    if 2 * (scaled_err + N as u64) >= delta_dst {
        return;
    }

    let qs = DynModulus::new(q_src);
    let qd = DynModulus::new(q_dst);
    let p = DynModulus::new(p_value);

    let mut lanes = [0u64; N];
    for (slot, &raw) in lanes.iter_mut().zip(input.plaintext.iter()) {
        *slot = raw % p_value;
    }
    let plaintext = R::new(p, lanes);

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(qs, Distribution::Ternary, &mut sk_prg);
    let encoded: R = encode(&plaintext, qs);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let ct: RLWECiphertext<N, R> = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);

    let switched: RLWECiphertext<N, R> = mod_switch_sym(&ct, qd);
    let sk_dst: SecretKey<N, R> = rekey_secret_key(&sk, qd);
    let recovered: R = sk_dst.decrypt(&switched, p);

    for (i, &expected) in lanes.iter().enumerate() {
        assert_eq!(
            recovered.coeff(i).to_u64(),
            expected,
            "mod_switch_sym round-trip diverged at i={i}; q_src={q_src} q_dst={q_dst} p={p_value}",
        );
    }
});
