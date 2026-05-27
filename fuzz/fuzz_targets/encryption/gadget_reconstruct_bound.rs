//! Fuzz: `reconstruct(gadget_decompose_into(input))` recovers `input`
//! within the spec error bound for any `(q, B, m)` the fuzzer fits
//! through a single-prime `Poly<8, DynModulus, Coefficient>` carrier.
//!
//! Targeted regressions:
//!
//! - Wrong rounding direction in [`gadget_scale_into`] (truncating `/`
//!   instead of half-away-from-zero) — would silently double the
//!   reconstruction error at negative inputs.
//! - Sign mishandling in the scale step — would invert half of the
//!   recovered coefficients.
//! - Off-by-one in the LSB-first → MSB-first reversal inside
//!   [`gadget_decompose_into`] — would misalign every digit with its
//!   gadget entry, blowing up reconstruction error.
//! - Wide-arithmetic helper bug in [`algebra::wide`] — triggered when
//!   `B^L · |c|` exceeds `i128`. Even at single-prime `q ≤ 2³⁴`, with
//!   `B^L ≤ 2²⁴` and `|c| < 2³³`, product ≤ 2⁵⁷ — fits in i128 but still
//!   routes through the helper.
//!
//! Single-prime backend only; the RNS path is covered by the
//! `decompose_reconstruct_at_via_c_q1_b18_l18_rns` unit test, which
//! exercises the wide-arithmetic divide at its `Q ≈ 2⁷⁵` worst case.
//!
//! Run with `cargo +nightly fuzz run encryption_gadget_reconstruct_bound`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::{gadget_decompose_into, reconstruct};

const N: usize = 8;
const L: usize = 4;

/// Ciphertext moduli to fuzz against — mix of toy and paper-class.
const KNOWN_Q: &[u64] = &[
    17,             // smallest viable
    257,            // larger toy
    1024,           // power-of-two q
    65537,          // medium prime
    1_048_583,      // ~2^20 prime
    8_380_417,      // VIA-C q_3
    2_147_352_577,  // VIA q_3
    17_175_674_881, // VIA-C q_2
];

#[derive(Debug)]
struct Input {
    q_index: u8,
    /// Base `B`, restricted to `[2, 64]` so `B^L ≤ 2²⁴` and the
    /// reconstruction bound stays meaningful.
    base: u8,
    coeffs: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let q_index = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
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
    let b_pow_l: u128 = (base as u128).pow(L as u32);

    // Feasibility gate: `B^L > q` makes the gadget vector degenerate
    // (smallest entries round to zero), so the reconstruction error
    // bound becomes uselessly loose. Skip — these inputs aren't
    // interesting to the round-trip assertion.
    if b_pow_l > u128::from(q_value) {
        return;
    }

    let q = DynModulus::new(q_value);

    // Reduce input coefficients into `[0, q)`.
    let mut reduced = [0u64; N];
    for (out, &raw) in reduced.iter_mut().zip(input.coeffs.iter()) {
        *out = raw % q_value;
    }
    let input_poly = Poly::<N, DynModulus, Coefficient>::new(q, reduced);

    let mut digits = [[0i64; N]; L];
    gadget_decompose_into::<N, Poly<N, DynModulus, Coefficient>, L>(&input_poly, base, &mut digits);
    let recovered: Poly<N, DynModulus, Coefficient> =
        reconstruct::<N, Poly<N, DynModulus, Coefficient>, L>(&digits, q, base);

    // Spec bound: `round(q/B^L)/2` for the scale-step rounding, plus
    // `L · B / 4` for the gadget-vector rounding (each `g_i` differs
    // from rational `q / B^{i+1}` by ≤ 1/2, multiplied by max digit
    // magnitude `B/2`, summed over `L` levels). +2 slack for integer
    // rounding artefacts.
    let g_min: u128 = (u128::from(q_value) + b_pow_l / 2) / b_pow_l;
    let bound: i64 = (g_min / 2) as i64 + (L as i64 * base as i64 / 4) + 2;

    let mut diff_coeffs = [0i64; N];
    let raw_diff = input_poly - recovered;
    raw_diff.to_centered_coeffs(&mut diff_coeffs);
    for (i, &d) in diff_coeffs.iter().enumerate() {
        assert!(
            d.abs() <= bound,
            "reconstruction error at i={i}: diff={d}, bound={bound}, q={q_value} base={base}"
        );
    }
});
