//! Fuzz: §6.1 `encrypt_lwe_raw` Δ-equivalence.
//!
//! `encrypt_lwe_raw(sk, Δ·m)` must decrypt to `m` for every `m ∈ [0, p)` — i.e.
//! the Δ-free raw encryption used by query compression agrees with the
//! Δ-encoded `encrypt_lwe` when handed a pre-scaled message. Catches a wrong
//! body assembly, a key/mask dot-product error, or a `u128` truncation.
//!
//! Run with `cargo +nightly fuzz run protocol_encrypt_lwe_raw_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{decrypt_lwe, encrypt_lwe_raw};
use via_primitives::encryption::SecretKey;
use via_primitives::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
type R = Poly<N, DynModulus, Coefficient>;

const KNOWN_Q: &[u64] = &[65537, 786433, 8_380_417];
const P: u64 = 16;

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    q_idx: u8,
    message: u64,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed = |u: &mut Unstructured<'a>| -> arbitrary::Result<Vec<u8>> {
            let l = u.int_in_range::<usize>(1..=32)?;
            let mut s = vec![0u8; l];
            u.fill_buffer(&mut s)?;
            Ok(s)
        };
        Ok(Input {
            sk_seed: seed(u)?,
            enc_seed: seed(u)?,
            q_idx: u.arbitrary()?,
            message: u.arbitrary()?,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_val = KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()];
    let q = DynModulus::new(q_val);
    let m = input.message % P;
    // Δ = ⌈q/p⌉; raw-encrypt the pre-scaled Δ·m so decrypt recovers m.
    let delta = (q_val as u128).div_ceil(P as u128);

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(q, Distribution::Ternary, &mut sk_prg);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let ct = encrypt_lwe_raw(&sk, delta * m as u128, Distribution::Ternary, &mut enc_prg);

    assert_eq!(ct.masks.len(), N, "encrypt_lwe_raw rank must be n");
    assert_eq!(decrypt_lwe(&ct, &sk, P), m, "raw Δ-equivalence diverged");
});
