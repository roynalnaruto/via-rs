//! Fuzz: §5.2 a single `conv_step` preserves the message.
//!
//! `conv_step` on an `(8,1)`-LWE yields a `(4,2)`-MLWE that decrypts under the
//! projected key vector `π_j^{8→2}(S)` to the original scalar at coefficient 0.
//! Catches a slot-accumulation, embed, or key-switch sign error.
//!
//! Run with `cargo +nightly fuzz run conversion_conv_step_roundtrip`.

#![no_main]
#![allow(clippy::needless_range_loop)]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{conv_step, encrypt_lwe, gen_conv_step_key};
use via_primitives::encryption::decode;
use via_primitives::encryption::types::SecretKey;
use via_primitives::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
const DEPTH: usize = 8;
type R8 = Poly<N, DynModulus, Coefficient>;
type R2 = Poly<2, DynModulus, Coefficient>;

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

    // Feasibility: one key-switch step's noise must stay inside Δ/2 = q/(2p).
    let tail = q_val / base.pow(DEPTH as u32);
    let noise = DEPTH as u64 * base + tail;
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
    let keys =
        gen_conv_step_key::<N, 1, 2, N, R8, DEPTH>(&sk, base, Distribution::Ternary, &mut gen_prg);

    let out = conv_step::<N, 1, 4, 2, _, _, DEPTH>(&lwe, &keys, base);

    let key_vec: [R2; 4] = core::array::from_fn(|j| sk.poly().project_at::<2>(j));
    let mut acc = out.body;
    for k in 0..4 {
        acc -= out.masks[k] * key_vec[k];
    }
    let recovered: R2 = decode(&acc, p);
    assert_eq!(recovered.coeff(0).to_u64(), message, "conv_step diverged");
    assert_eq!(recovered.coeff(1).to_u64(), 0, "conv_step slot 1 nonzero");
});
