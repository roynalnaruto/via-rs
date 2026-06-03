//! Fuzz: every digit produced by `gadget_decompose_into` must lie in
//! the signed-balanced range `(-B/2, B/2]`.
//!
//! Targeted regressions: any change to the LSB-first extraction loop
//! that breaks the `digit > B/2 → digit -= B` rebalance step would
//! produce unbalanced digits outside the range.
//!
//! Cheap and fast — no reconstruction step, just digit-by-digit bound
//! checks.
//!
//! Run with `cargo +nightly fuzz run encryption_gadget_digit_bounds`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::encryption::gadget_decompose_into;

const N: usize = 8;
const L: usize = 4;

const KNOWN_Q: &[u64] = &[
    17,
    257,
    1024,
    65537,
    1_048_583,
    8_380_417,
    2_147_352_577,
    17_175_674_881,
];

#[derive(Debug)]
struct Input {
    q_index: u8,
    base: u8,
    coeffs: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let q_index = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
        // Base in [2, 64]; B^L ≤ 2²⁴, so no feasibility gate needed.
        let base = u.int_in_range::<u8>(2..=64)?;
        let mut coeffs = [0u64; N];
        for slot in coeffs.iter_mut() {
            *slot = u.arbitrary()?;
        }
        Ok(Input {
            q_index,
            base,
            coeffs,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_value = KNOWN_Q[input.q_index as usize];
    let base = u64::from(input.base);
    let base_i64 = base as i64;

    let q = DynModulus::new(q_value);
    let mut reduced = [0u64; N];
    for (out, &raw) in reduced.iter_mut().zip(input.coeffs.iter()) {
        *out = raw % q_value;
    }
    let input_poly = Poly::<N, DynModulus, Coefficient>::new(q, reduced);

    let mut digits = [[0i64; N]; L];
    gadget_decompose_into::<N, Poly<N, DynModulus, Coefficient>, L>(&input_poly, base, &mut digits);

    // Unified bound: `2·d > -B && 2·d ≤ B` — covers even `B` (digits in
    // `(-B/2, B/2]`) and odd `B` (digits in `[-(B−1)/2, (B−1)/2]`).
    for (level, level_digits) in digits.iter().enumerate() {
        for (i, &d) in level_digits.iter().enumerate() {
            assert!(
                2 * d > -base_i64,
                "digit underflow at level={level} i={i}: d={d}, base={base}"
            );
            assert!(
                2 * d <= base_i64,
                "digit overflow at level={level} i={i}: d={d}, base={base}"
            );
        }
    }
});
