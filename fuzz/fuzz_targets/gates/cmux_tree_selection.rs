//! Fuzz: §4.3 `cmux_tree` selects `inputs[index]` (LSB-first bits).
//!
//! With `m = 2` select bits encoding `index ∈ [0,4)`, the tree must return the
//! ciphertext at that index. Guards the in-place reduction and the bit ordering.
//!
//! Run with `cargo +nightly fuzz run gates_cmux_tree_selection`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::RingPoly;
use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::types::{RGSWCiphertext, RLWECiphertext};
use via_rs::encryption::{SecretKey, encode};
use via_rs::gates::cmux_tree;
use via_rs::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
const DEPTH: usize = 8;
type R = Poly<N, DynModulus, Coefficient>;

const KNOWN_Q: &[u64] = &[65537, 786433, 8_380_417];
const BASES: &[u64] = &[2, 4];
const P: u64 = 2;

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    rgsw_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    index: u8,
    q_idx: u8,
    base_idx: u8,
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
            rgsw_seed: seed(u)?,
            enc_seed: seed(u)?,
            index: u.int_in_range::<u8>(0..=3)?,
            q_idx: u.arbitrary()?,
            base_idx: u.arbitrary()?,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_val = KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()];
    let base = BASES[input.base_idx as usize % BASES.len()];
    let index = input.index as usize; // in [0, 4)

    // Two CMux levels of noise plus the reconstruction tail.
    let tail = q_val / base.pow(DEPTH as u32);
    let noise = 2 * 2 * (DEPTH as u64 * base + tail);
    if 8 * noise >= q_val / P {
        return;
    }

    let q = DynModulus::new(q_val);
    let p = DynModulus::new(P);

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(q, Distribution::Ternary, &mut sk_prg);

    // Select bits = LSB-first binary of `index`.
    let mut rgsw_prg = Shake256Prg::new(&input.rgsw_seed);
    let bits: [RGSWCiphertext<N, R, DEPTH, DEPTH>; 2] = core::array::from_fn(|i| {
        let mut bc = [0u128; N];
        bc[0] = ((index >> i) & 1) as u128;
        let bp = <R as RingPoly<N>>::from_u128_coeffs(q, &bc);
        sk.encrypt_rgsw::<DEPTH, DEPTH>(&bp, base, base, Distribution::Ternary, &mut rgsw_prg)
    });

    // inputs[i] = message with `1` at coefficient `i`.
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let mut inputs: [RLWECiphertext<N, R>; 4] = core::array::from_fn(|i| {
        let mut m = [0u64; N];
        m[i] = 1;
        let encoded: R = encode(&R::new(p, m), q);
        sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg)
    });

    let out = cmux_tree(&bits, &mut inputs, base, base);
    let rec: R = sk.decrypt(&out, p);
    for i in 0..N {
        let expected = if i == index { 1 } else { 0 };
        assert_eq!(
            rec.coeff(i).to_u64(),
            expected,
            "cmux_tree index={index} diverged at i={i}"
        );
    }
});
