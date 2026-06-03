//! Fuzz: §5.1 `embed_mlwe` preserves the message under the embedded key.
//!
//! Since $\iota_0$ is a ring homomorphism, embedding an RLWE (as a rank-1 MLWE)
//! from degree 4 to degree 8 must decrypt under $\iota_0(S)$ to $\iota_0(M)$:
//! `m[i]` lands at coefficient `2i`, odd positions are zero. Catches a wrong
//! embed stride or slot.
//!
//! Run with `cargo +nightly fuzz run conversion_embed_mlwe_roundtrip`.

#![no_main]
#![allow(clippy::needless_range_loop)]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{embed_mlwe, mlwe_to_rlwe, rlwe_to_mlwe};
use via_primitives::encryption::types::SecretKey;
use via_primitives::encryption::{MLWECiphertext, encode};
use via_primitives::sampling::{Distribution, Shake256Prg};

const N: usize = 4;
const NL: usize = 8;
type R4 = Poly<N, DynModulus, Coefficient>;
type R8 = Poly<NL, DynModulus, Coefficient>;

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
    let sk = SecretKey::<N, R4>::keygen(q, Distribution::Ternary, &mut sk_prg);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let encoded: R4 = encode(&R4::new(p, m), q);
    let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);

    let embedded: MLWECiphertext<1, NL, R8> = embed_mlwe(&rlwe_to_mlwe(&ct));
    let embedded_rlwe = mlwe_to_rlwe(&embedded);
    let sk8 = SecretKey::<NL, R8>::from_poly(sk.poly().embed_at::<NL>(0));
    let recovered: R8 = sk8.decrypt(&embedded_rlwe, p);

    for i in 0..N {
        assert_eq!(
            recovered.coeff(2 * i).to_u64(),
            m[i],
            "embed: m[{i}] not at 2i"
        );
        assert_eq!(
            recovered.coeff(2 * i + 1).to_u64(),
            0,
            "embed: odd slot nonzero"
        );
    }
});
