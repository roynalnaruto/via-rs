//! Fuzz: Barrett reduction (and the mask path) versus the naive
//! `(x % q) as u64` reference, for every `Modulus` impl.
//!
//! Run with `cargo +nightly fuzz run zq_reduce`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::zq::modulus::{DynModulus, Modulus};

/// Paper-pinned moduli, plus a small set of representative odd composites,
/// gives the fuzzer a strong starting point for corpus expansion.
const KNOWN_MODULI: &[u64] = &[
    2,
    3,
    5,
    17,
    257, // tiny primes
    16,
    256,
    4096,
    32768, // powers of two: p, q_4
    8380417,
    2147352577, // VIA-C / VIA q_3 primes
    17175674881,
    34359214081, // q_2 primes
    137438822401,
    274810798081, // VIA-C q_1 RNS primes
    268369921,
    536608769, // VIA q_1 RNS primes
];

#[derive(Debug)]
struct FuzzModulus(DynModulus);

impl<'a> Arbitrary<'a> for FuzzModulus {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let pick_known: bool = u.arbitrary()?;
        let q = if pick_known {
            *u.choose(KNOWN_MODULI)?
        } else {
            // Random odd in [3, 2^38]; covers VIA-C's largest §0.1 modulus.
            let raw = u.int_in_range::<u64>(3..=(1u64 << 38))?;
            raw | 1
        };
        Ok(FuzzModulus(DynModulus::new(q)))
    }
}

#[derive(Debug, Arbitrary)]
struct Input {
    modulus: FuzzModulus,
    x: u128,
}

fuzz_target!(|input: Input| {
    let m = input.modulus.0;
    let q = m.q();
    let got = m.reduce_u128(input.x);
    let want = (input.x % u128::from(q)) as u64;
    assert_eq!(got, want, "reduce_u128 disagrees for q={q}, x={}", input.x);
});
