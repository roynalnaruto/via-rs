//! Fuzz: §7 VIA-B `Repack` reconstructs the paper interleave.
//!
//! For any two messages `M0, M1 ∈ R_{8,p}` and any modulus/key seed,
//! `Repack_4({Enc(M0), Enc(M1)})` decrypts under `S` to the interleave at
//! coefficients 0,2,4,6 = `[M0_0, M1_0, M0_4, M1_4]` (the
//! `repack_n8_t2_reconstructs` property under fuzzed inputs), using the dedicated
//! key oracle (= the cascade's own conv-step keys, §3.5). The strongest net for
//! the P1/P2 crux.
//!
//! Run with `cargo +nightly fuzz run --features via-b conversion_repack_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{gen_repack_keys_n8_t2, repack_n8_t2};
use via_primitives::encryption::types::SecretKey;
use via_primitives::encryption::{decode, encode};
use via_primitives::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
const P: u64 = 16;
const L: usize = 8;
const BASE: u64 = 8;
const KNOWN_Q: &[u64] = &[65537, 786433, 8_380_417];
type R8 = Poly<N, DynModulus, Coefficient>;

#[derive(Debug)]
struct Input {
    key_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    q_idx: u8,
    m0: [u64; N],
    m1: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed = |u: &mut Unstructured<'a>| -> arbitrary::Result<Vec<u8>> {
            let l = u.int_in_range::<usize>(1..=32)?;
            let mut s = vec![0u8; l];
            u.fill_buffer(&mut s)?;
            Ok(s)
        };
        let key_seed = seed(u)?;
        let enc_seed = seed(u)?;
        let q_idx = u.arbitrary()?;
        let mut m0 = [0u64; N];
        let mut m1 = [0u64; N];
        for x in &mut m0 {
            *x = u.arbitrary()?;
        }
        for x in &mut m1 {
            *x = u.arbitrary()?;
        }
        Ok(Input {
            key_seed,
            enc_seed,
            q_idx,
            m0,
            m1,
        })
    }
}

fuzz_target!(|input: Input| {
    let q = DynModulus::new(KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()]);
    let p = DynModulus::new(P);
    let m0: [u64; N] = core::array::from_fn(|i| input.m0[i] % P);
    let m1: [u64; N] = core::array::from_fn(|i| input.m1[i] % P);

    let mut prg = Shake256Prg::new(&input.key_seed);
    let sk = SecretKey::<N, R8>::keygen(q, Distribution::Ternary, &mut prg);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let c0 = sk.encrypt(
        &encode::<N, R8, R8>(&R8::new(p, m0), q),
        Distribution::Ternary,
        &mut enc_prg,
    );
    let c1 = sk.encrypt(
        &encode::<N, R8, R8>(&R8::new(p, m1), q),
        Distribution::Ternary,
        &mut enc_prg,
    );

    let keys = gen_repack_keys_n8_t2::<DynModulus, L>(&sk, BASE, Distribution::Ternary, &mut prg);
    let out = repack_n8_t2(&[c0, c1], &keys, BASE);

    let mut acc = out.body;
    acc -= out.mask * *sk.poly();
    let rec: R8 = decode::<N, R8, R8>(&acc, p);
    assert_eq!(rec.coeff(0).to_u64(), m0[0], "repack slot 0 (M0_0)");
    assert_eq!(rec.coeff(2).to_u64(), m1[0], "repack slot 2 (M1_0)");
    assert_eq!(rec.coeff(4).to_u64(), m0[4], "repack slot 4 (M0_4)");
    assert_eq!(rec.coeff(6).to_u64(), m1[4], "repack slot 6 (M1_4)");
});
