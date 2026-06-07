//! Fuzz: §4.4 `crot` rotates correctly in both directions.
//!
//! On a unit message `X^0`, `Forward` with bits encoding `γ` yields `X^{γ}`
//! (coefficient 1 at slot `γ mod N`); `SlotExtract` yields `X^{-γ}`
//! (coefficient 1 at slot `(N-γ) mod N`, since negation is a noop at p=2).
//!
//! Run with `cargo +nightly fuzz run gates_crot_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::encryption::types::RGSWCiphertext;
use via_primitives::encryption::{SecretKey, encode};
use via_primitives::gates::{CRotDir, crot};
use via_primitives::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
const DEPTH: usize = 8;
const BITS: usize = 2; // gamma in [0, 4)
type R = Poly<N, DynModulus, Coefficient>;

const KNOWN_Q: &[u64] = &[65537, 786433, 8_380_417];
const BASES: &[u64] = &[2, 4];
const P: u64 = 2;

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    rgsw_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    gamma: u8,
    slot_extract: bool,
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
            gamma: u.int_in_range::<u8>(0..=3)?,
            slot_extract: u.arbitrary()?,
            q_idx: u.arbitrary()?,
            base_idx: u.arbitrary()?,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_val = KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()];
    let base = BASES[input.base_idx as usize % BASES.len()];
    let gamma = input.gamma as usize; // in [0, 4)

    // BITS CMux levels of noise plus the reconstruction tail.
    let tail = q_val / base.pow(DEPTH as u32);
    let noise = BITS as u64 * 2 * (DEPTH as u64 * base + tail);
    if 8 * noise >= q_val / P {
        return;
    }

    let q = DynModulus::new(q_val);
    let p = DynModulus::new(P);

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(q, Distribution::Ternary, &mut sk_prg);

    let mut rgsw_prg = Shake256Prg::new(&input.rgsw_seed);
    let bits: [RGSWCiphertext<N, R, DEPTH, DEPTH>; BITS] = core::array::from_fn(|i| {
        let mut bc = [0u128; N];
        bc[0] = ((gamma >> i) & 1) as u128;
        let bp = <R as RingPoly<N>>::from_u128_coeffs(q, &bc);
        sk.encrypt_rgsw::<DEPTH, DEPTH>(&bp, base, base, Distribution::Ternary, &mut rgsw_prg)
    });

    // Unit message X^0.
    let mut m = [0u64; N];
    m[0] = 1;
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let encoded: R = encode(&R::new(p, m), q);
    let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);

    let dir = if input.slot_extract {
        CRotDir::SlotExtract
    } else {
        CRotDir::Forward
    };
    let out = crot(dir, &bits, ct, base, base);
    let rec: R = sk.decrypt(&out, p);

    let target_slot = if input.slot_extract {
        (N - gamma) % N
    } else {
        gamma % N
    };
    for i in 0..N {
        let expected = if i == target_slot { 1 } else { 0 };
        assert_eq!(
            rec.coeff(i).to_u64(),
            expected,
            "crot slot_extract={} gamma={gamma} diverged at i={i}",
            input.slot_extract,
        );
    }
});
