//! Fuzz: §4.7 `mod_switch_rgsw` output coefficients are all `< q_dst`.
//!
//! A pure range check (no decryption): every constituent RLWE coefficient of
//! both RLev halves must be reduced into `[0, q_dst)`. Catches a wrong rescale
//! direction or an off-by-one in the modulus-switch mapping.
//!
//! Run with `cargo +nightly fuzz run gates_mod_switch_rgsw_shape`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::RingPoly;
use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::SecretKey;
use via_rs::encryption::types::RGSWCiphertext;
use via_rs::gates::mod_switch_rgsw;
use via_rs::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
const DEPTH: usize = 4;
type R = Poly<N, DynModulus, Coefficient>;

const KNOWN_Q: &[u64] = &[65537, 786433, 8_380_417, 2_147_352_577];

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    rgsw_seed: Vec<u8>,
    src_idx: u8,
    dst_idx: u8,
    bit: bool,
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
            src_idx: u.arbitrary()?,
            dst_idx: u.arbitrary()?,
            bit: u.arbitrary()?,
            base_idx: u.arbitrary()?,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_src = KNOWN_Q[input.src_idx as usize % KNOWN_Q.len()];
    let q_dst = KNOWN_Q[input.dst_idx as usize % KNOWN_Q.len()];
    // Only a genuine down-switch is interesting.
    if q_dst >= q_src {
        return;
    }
    let base = [2u64, 4, 8][input.base_idx as usize % 3];

    let qs = DynModulus::new(q_src);
    let qd = DynModulus::new(q_dst);

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(qs, Distribution::Ternary, &mut sk_prg);
    let mut bc = [0u128; N];
    bc[0] = input.bit as u128;
    let bit_poly = <R as RingPoly<N>>::from_u128_coeffs(qs, &bc);
    let mut rgsw_prg = Shake256Prg::new(&input.rgsw_seed);
    let rgsw = sk.encrypt_rgsw::<DEPTH, DEPTH>(
        &bit_poly,
        base,
        base,
        Distribution::Ternary,
        &mut rgsw_prg,
    );

    let switched: RGSWCiphertext<N, R, DEPTH, DEPTH> = mod_switch_rgsw(&rgsw, qd);

    for sample in switched
        .neg_s_m
        .samples
        .iter()
        .chain(switched.m.samples.iter())
    {
        for poly in [&sample.mask, &sample.body] {
            let mut coeffs = [0u128; N];
            poly.to_u128_coeffs(&mut coeffs);
            for v in coeffs {
                assert!(
                    v < q_dst as u128,
                    "mod_switch_rgsw: coeff {v} >= q_dst {q_dst} (q_src={q_src})",
                );
            }
        }
    }
});
