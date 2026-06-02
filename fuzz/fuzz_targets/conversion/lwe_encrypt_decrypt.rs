//! Fuzz: §5.1 `encrypt_lwe` / `decrypt_lwe` round-trip.
//!
//! `decrypt_lwe(encrypt_lwe(m)) == m` for every scalar `m ∈ [0, p)`. Catches a
//! wrong Δ, a body-assembly error, or a key/mask mismatch in the dot product.
//!
//! Run with `cargo +nightly fuzz run conversion_lwe_encrypt_decrypt`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::conversion::{decrypt_lwe, encrypt_lwe};
use via_rs::encryption::SecretKey;
use via_rs::sampling::{Distribution, Shake256Prg};

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
    // p must divide into q with room; all KNOWN_Q ≫ p so the single error term
    // is far inside Δ/2 = q/(2p).
    let q = DynModulus::new(q_val);
    let message = input.message % P;

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(q, Distribution::Ternary, &mut sk_prg);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let ct = encrypt_lwe(&sk, message, P, Distribution::Ternary, &mut enc_prg);

    assert_eq!(ct.masks.len(), N, "encrypt_lwe rank must be n");
    assert_eq!(decrypt_lwe(&ct, &sk, P), message, "lwe round-trip diverged");
});
