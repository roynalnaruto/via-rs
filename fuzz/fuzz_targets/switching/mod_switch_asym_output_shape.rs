//! Fuzz: §3.2 asymmetric ModSwitch output-shape invariant.
//!
//! For an arbitrary ciphertext at `q_src`, `mod_switch_asym` to
//! `(q_mask, q_body)` must produce a mask whose every coefficient is `<
//! q_mask` and a body whose every coefficient is `< q_body`. This catches a
//! rescale that forgets to reduce into the destination modulus (e.g. an
//! off-by-one letting a coefficient equal `q_dst`).
//!
//! Single-prime `DynModulus` carrier, `N = 16`.
//!
//! Run with `cargo +nightly fuzz run switching_mod_switch_asym_output_shape`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::abstraction::RingPoly;
use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::types::RLWECiphertext;
use via_rs::switching::mod_switch::mod_switch_asym;

const N: usize = 16;
type R = Poly<N, DynModulus, Coefficient>;

const KNOWN_Q: &[u64] = &[65536, 8_380_417, 2_147_352_577, 17_175_674_881];

#[derive(Debug)]
struct Input {
    src_idx: u8,
    mask_idx: u8,
    body_idx: u8,
    mask_coeffs: [u64; N],
    body_coeffs: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let src_idx = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
        let mask_idx = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
        let body_idx = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
        let mut mask_coeffs = [0u64; N];
        let mut body_coeffs = [0u64; N];
        for slot in &mut mask_coeffs {
            *slot = u.arbitrary()?;
        }
        for slot in &mut body_coeffs {
            *slot = u.arbitrary()?;
        }
        Ok(Input {
            src_idx,
            mask_idx,
            body_idx,
            mask_coeffs,
            body_coeffs,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_src = KNOWN_Q[input.src_idx as usize];
    let q_mask = KNOWN_Q[input.mask_idx as usize];
    let q_body = KNOWN_Q[input.body_idx as usize];
    // Down-switch on both legs is the documented use; skip up-switches.
    if q_mask >= q_src || q_body >= q_src {
        return;
    }

    let qs = DynModulus::new(q_src);
    let qm = DynModulus::new(q_mask);
    let qb = DynModulus::new(q_body);

    let mut mask_lanes = [0u64; N];
    let mut body_lanes = [0u64; N];
    for (slot, &raw) in mask_lanes.iter_mut().zip(input.mask_coeffs.iter()) {
        *slot = raw % q_src;
    }
    for (slot, &raw) in body_lanes.iter_mut().zip(input.body_coeffs.iter()) {
        *slot = raw % q_src;
    }
    let ct = RLWECiphertext::<N, R>::new(R::new(qs, mask_lanes), R::new(qs, body_lanes));

    let out = mod_switch_asym::<N, R, R, R>(&ct, qm, qb);

    let mut m = [0u128; N];
    let mut b = [0u128; N];
    out.mask.to_u128_coeffs(&mut m);
    out.body.to_u128_coeffs(&mut b);
    for (i, &v) in m.iter().enumerate() {
        assert!(
            v < u128::from(q_mask),
            "mask coeff {i} = {v} >= q_mask={q_mask}"
        );
    }
    for (i, &v) in b.iter().enumerate() {
        assert!(
            v < u128::from(q_body),
            "body coeff {i} = {v} >= q_body={q_body}"
        );
    }
});
