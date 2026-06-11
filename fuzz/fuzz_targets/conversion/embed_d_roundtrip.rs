//! Fuzz: §7 VIA-B `embed_d` interleaves `d` inputs at slots `0..d-1`.
//!
//! For `d = 2` rank-1 `(1, 8)`-MLWEs with arbitrary coefficients/modulus,
//! `embed_d` produces a `(1, 16)`-MLWE whose mask/body place input-`s` coefficient
//! `i` at position `2·i + s` (the `embed_d_interleaves_two_inputs_at_slots`
//! property). Pure index reshape — no keys, no noise — so this is an exact
//! coefficient check under fuzzed inputs.
//!
//! Run with `cargo +nightly fuzz run --features via-b conversion_embed_d_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::embed_d;
use via_primitives::encryption::MLWECiphertext;

const N: usize = 8;
const NL: usize = 16; // N · d, d = 2
const KNOWN_Q: &[u64] = &[65537, 786433, 8_380_417];
type R8 = Poly<N, DynModulus, Coefficient>;
type R16 = Poly<NL, DynModulus, Coefficient>;

#[derive(Debug)]
struct Input {
    q_idx: u8,
    a0: [u64; N],
    b0: [u64; N],
    a1: [u64; N],
    b1: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let arr = |u: &mut Unstructured<'a>| -> arbitrary::Result<[u64; N]> {
            let mut a = [0u64; N];
            for x in &mut a {
                *x = u.arbitrary()?;
            }
            Ok(a)
        };
        let q_idx = u.arbitrary()?;
        Ok(Input {
            q_idx,
            a0: arr(u)?,
            b0: arr(u)?,
            a1: arr(u)?,
            b1: arr(u)?,
        })
    }
}

fuzz_target!(|input: Input| {
    let qv = KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()];
    let q = DynModulus::new(qv);
    let red = |a: &[u64; N]| -> [u64; N] { core::array::from_fn(|i| a[i] % qv) };
    let (a0, b0, a1, b1) = (
        red(&input.a0),
        red(&input.b0),
        red(&input.a1),
        red(&input.b1),
    );

    let ct0 = MLWECiphertext::<1, N, R8>::new([R8::new(q, a0)], R8::new(q, b0));
    let ct1 = MLWECiphertext::<1, N, R8>::new([R8::new(q, a1)], R8::new(q, b1));
    let out: MLWECiphertext<1, NL, R16> = embed_d::<1, N, NL, R8, R16>(&[ct0, ct1]);

    for i in 0..N {
        assert_eq!(out.body.coeff(2 * i).to_u64(), b0[i], "body slot 0, i={i}");
        assert_eq!(
            out.body.coeff(2 * i + 1).to_u64(),
            b1[i],
            "body slot 1, i={i}"
        );
        assert_eq!(
            out.masks[0].coeff(2 * i).to_u64(),
            a0[i],
            "mask slot 0, i={i}"
        );
        assert_eq!(
            out.masks[0].coeff(2 * i + 1).to_u64(),
            a1[i],
            "mask slot 1, i={i}"
        );
    }
});
