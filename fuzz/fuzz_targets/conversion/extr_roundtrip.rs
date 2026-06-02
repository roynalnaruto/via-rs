//! Fuzz: §5.5 `extr` extracts the right coefficients.
//!
//! `D = 1` (classical sample extraction) recovers `M[0]`; `D = 2` recovers
//! `π_0^{8→2}(M) = (M[0], M[4])`, each under the matching projected key vector.
//! `extr` adds no noise (pure index move), so only the host RLWE's encryption
//! noise is in play. Catches a reversed projection slot or a missing `·X`.
//!
//! Run with `cargo +nightly fuzz run conversion_extr_roundtrip`.

#![no_main]
#![allow(clippy::needless_range_loop)]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::conversion::extr;
use via_rs::encryption::types::SecretKey;
use via_rs::encryption::{MLWECiphertext, decode, encode};
use via_rs::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
type R8 = Poly<N, DynModulus, Coefficient>;
type R1 = Poly<1, DynModulus, Coefficient>;
type R2 = Poly<2, DynModulus, Coefficient>;

const KNOWN_Q: &[u64] = &[65537, 786433, 8_380_417];
const P: u64 = 16;

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    q_idx: u8,
    msg: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed = |u: &mut Unstructured<'a>| -> arbitrary::Result<Vec<u8>> {
            let l = u.int_in_range::<usize>(1..=32)?;
            let mut s = vec![0u8; l];
            u.fill_buffer(&mut s)?;
            Ok(s)
        };
        let sk_seed = seed(u)?;
        let enc_seed = seed(u)?;
        let q_idx = u.arbitrary()?;
        let mut msg = [0u64; N];
        for slot in &mut msg {
            *slot = u.arbitrary()?;
        }
        Ok(Input {
            sk_seed,
            enc_seed,
            q_idx,
            msg,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_val = KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()];
    let q = DynModulus::new(q_val);
    let p = DynModulus::new(P);

    let mut m = [0u64; N];
    for i in 0..N {
        m[i] = input.msg[i] % P;
    }

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R8>::keygen(q, Distribution::Ternary, &mut sk_prg);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let encoded: R8 = encode(&R8::new(p, m), q);
    let rlwe = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);

    // D = 1: sample extraction → M[0] under (S_0, …, S_7).
    let lwe: MLWECiphertext<8, 1, R1> = extr::<N, 1, 8, _>(&rlwe);
    let key1: [R1; 8] = core::array::from_fn(|j| sk.poly().project_at::<1>(j));
    let mut acc1 = lwe.body;
    for k in 0..8 {
        acc1 -= lwe.masks[k] * key1[k];
    }
    let rec1: R1 = decode(&acc1, p);
    assert_eq!(rec1.coeff(0).to_u64(), m[0], "extr d=1 diverged");

    // D = 2: → (M[0], M[4]) under (π_0(S), …, π_3(S)).
    let mlwe: MLWECiphertext<4, 2, R2> = extr::<N, 2, 4, _>(&rlwe);
    let key2: [R2; 4] = core::array::from_fn(|j| sk.poly().project_at::<2>(j));
    let mut acc2 = mlwe.body;
    for k in 0..4 {
        acc2 -= mlwe.masks[k] * key2[k];
    }
    let rec2: R2 = decode(&acc2, p);
    assert_eq!(rec2.coeff(0).to_u64(), m[0], "extr d=2 slot 0 diverged");
    assert_eq!(rec2.coeff(1).to_u64(), m[4], "extr d=2 slot 1 diverged");
});
