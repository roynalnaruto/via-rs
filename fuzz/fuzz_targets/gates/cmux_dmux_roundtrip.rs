//! Fuzz: §4.1 CMux selects, §4.2 DMux routes.
//!
//! `cmux(RGSW(b), ct0, ct1)` must decrypt to `ct_b`'s message; `dmux(RGSW(b),
//! ct0)` must place `ct0`'s message at output `b` and zero at the other. Catches
//! external-product sign/half errors.
//!
//! Run with `cargo +nightly fuzz run gates_cmux_dmux_roundtrip`.

#![no_main]
// The per-coefficient checks index `.coeff(i)` (a method, not a slice), so a
// range loop is the clearest form.
#![allow(clippy::needless_range_loop)]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::RingPoly;
use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::{SecretKey, encode};
use via_rs::gates::{cmux, dmux};
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
    bit: bool,
    q_idx: u8,
    base_idx: u8,
    msg0: [u64; N],
    msg1: [u64; N],
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
        let rgsw_seed = seed(u)?;
        let enc_seed = seed(u)?;
        let bit = u.arbitrary()?;
        let q_idx = u.arbitrary()?;
        let base_idx = u.arbitrary()?;
        let mut msg0 = [0u64; N];
        let mut msg1 = [0u64; N];
        for slot in &mut msg0 {
            *slot = u.arbitrary()?;
        }
        for slot in &mut msg1 {
            *slot = u.arbitrary()?;
        }
        Ok(Input {
            sk_seed,
            rgsw_seed,
            enc_seed,
            bit,
            q_idx,
            base_idx,
            msg0,
            msg1,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_val = KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()];
    let base = BASES[input.base_idx as usize % BASES.len()];

    // Feasibility: external product noise (2 RGSW halves, DEPTH levels) plus the
    // gadget reconstruction tail must stay well inside the p=2 budget q/2.
    let tail = q_val / base.pow(DEPTH as u32);
    let noise = 2 * (DEPTH as u64 * base + tail);
    if 8 * noise >= q_val / P {
        return;
    }

    let q = DynModulus::new(q_val);
    let p = DynModulus::new(P);

    let mut m0 = [0u64; N];
    let mut m1 = [0u64; N];
    for i in 0..N {
        m0[i] = input.msg0[i] % P;
        m1[i] = input.msg1[i] % P;
    }

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(q, Distribution::Ternary, &mut sk_prg);

    let mut bc = [0u128; N];
    bc[0] = input.bit as u128;
    let bit_poly = <R as RingPoly<N>>::from_u128_coeffs(q, &bc);
    let mut rgsw_prg = Shake256Prg::new(&input.rgsw_seed);
    let rgsw = sk.encrypt_rgsw::<DEPTH, DEPTH>(
        &bit_poly,
        base,
        base,
        Distribution::Ternary,
        &mut rgsw_prg,
    );

    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let enc0: R = encode(&R::new(p, m0), q);
    let ct0 = sk.encrypt(&enc0, Distribution::Ternary, &mut enc_prg);
    let enc1: R = encode(&R::new(p, m1), q);
    let ct1 = sk.encrypt(&enc1, Distribution::Ternary, &mut enc_prg);

    // CMux: bit=0 -> ct0, bit=1 -> ct1.
    let selected = cmux(&rgsw, &ct0, &ct1, base, base);
    let rec: R = sk.decrypt(&selected, p);
    let expected = if input.bit { &m1 } else { &m0 };
    for i in 0..N {
        assert_eq!(rec.coeff(i).to_u64(), expected[i], "cmux diverged at i={i}");
    }

    // DMux on ct0: bit=0 -> (m0, 0), bit=1 -> (0, m0).
    let (r0, r1) = dmux(&rgsw, &ct0, base, base);
    let dec0: R = sk.decrypt(&r0, p);
    let dec1: R = sk.decrypt(&r1, p);
    for i in 0..N {
        let (e0, e1) = if input.bit { (0, m0[i]) } else { (m0[i], 0) };
        assert_eq!(dec0.coeff(i).to_u64(), e0, "dmux r0 diverged at i={i}");
        assert_eq!(dec1.coeff(i).to_u64(), e1, "dmux r1 diverged at i={i}");
    }
});
