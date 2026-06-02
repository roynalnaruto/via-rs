//! Fuzz: §5.3 the full `lwe_to_rlwe_n8` cascade preserves the message.
//!
//! `decrypt(lwe_to_rlwe(encrypt_lwe(m)))` recovers `m` at coefficient 0 of the
//! degree-8 RLWE (the rest zero). Catches a per-step wiring error, a wrong
//! key-field association, or accumulated-noise misanalysis across the 3 steps.
//!
//! Run with `cargo +nightly fuzz run conversion_cascade_roundtrip`.

#![no_main]
#![allow(clippy::needless_range_loop)]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::conversion::{encrypt_lwe, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8};
use via_rs::encryption::types::SecretKey;
use via_rs::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
const DEPTH: usize = 8;
const STEPS: u64 = 3; // log2(8)
type R8 = Poly<N, DynModulus, Coefficient>;

const KNOWN_Q: &[u64] = &[65537, 786433, 8_380_417];
const BASES: &[u64] = &[2, 4, 8];
const P: u64 = 16;

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    gen_seed: Vec<u8>,
    q_idx: u8,
    base_idx: u8,
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
            gen_seed: seed(u)?,
            q_idx: u.arbitrary()?,
            base_idx: u.arbitrary()?,
            message: u.arbitrary()?,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_val = KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()];
    let base = BASES[input.base_idx as usize % BASES.len()];

    // Feasibility: 3 chained key-switch steps must stay inside Δ/2 = q/(2p).
    let tail = q_val / base.pow(DEPTH as u32);
    let noise = STEPS * (DEPTH as u64 * base + tail);
    if 8 * noise >= q_val / (2 * P) {
        return;
    }

    let q = DynModulus::new(q_val);
    let p = DynModulus::new(P);
    let message = input.message % P;

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R8>::keygen(q, Distribution::Ternary, &mut sk_prg);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let lwe = encrypt_lwe(&sk, message, P, Distribution::Ternary, &mut enc_prg);
    let mut gen_prg = Shake256Prg::new(&input.gen_seed);
    let key = gen_lwe_to_rlwe_key_n8::<_, DEPTH>(&sk, base, Distribution::Ternary, &mut gen_prg);

    let rlwe = lwe_to_rlwe_n8::<_, DEPTH>(&lwe, &key, base);
    let recovered: R8 = sk.decrypt(&rlwe, p);
    assert_eq!(recovered.coeff(0).to_u64(), message, "cascade diverged");
    for i in 1..N {
        assert_eq!(recovered.coeff(i).to_u64(), 0, "cascade slot {i} nonzero");
    }
});
