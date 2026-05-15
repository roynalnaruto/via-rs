//! Fuzz: encode/decode round-trips for the `Zq<M>` wrapper.
//!
//! - `Zq::new(m, x).to_u64() == x mod q`
//! - `Zq::from_i64(m, x).to_centered_i64()` round-trips for centred values.
//! - `to_centered_i64` always falls in $(-\lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::zq::element::Zq;
use via_rs::primitives::zq::modulus::{DynModulus, Modulus};

const KNOWN_MODULI: &[u64] = &[
    2,
    3,
    5,
    17,
    257,
    16,
    256,
    4096,
    32768,
    8380417,
    2147352577,
    17175674881,
    34359214081,
    137438822401,
    274810798081,
    268369921,
    536608769,
];

#[derive(Debug)]
struct FuzzModulus(DynModulus);

impl<'a> Arbitrary<'a> for FuzzModulus {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let pick_known: bool = u.arbitrary()?;
        let q = if pick_known {
            *u.choose(KNOWN_MODULI)?
        } else {
            u.int_in_range::<u64>(3..=(1u64 << 38))? | 1
        };
        Ok(FuzzModulus(DynModulus::new(q)))
    }
}

#[derive(Debug, Arbitrary)]
struct Input {
    modulus: FuzzModulus,
    raw_u: u64,
    raw_i: i64,
}

fuzz_target!(|input: Input| {
    let m = input.modulus.0;
    let q = m.q();
    let q128 = u128::from(q);

    // u64 -> Zq round-trip via canonical form.
    let z = Zq::new(m, input.raw_u);
    assert!(z.to_u64() < q);
    assert_eq!(u128::from(z.to_u64()), u128::from(input.raw_u) % q128);

    // i64 -> Zq -> centered i64 in range.
    let zi = Zq::from_i64(m, input.raw_i);
    assert!(zi.to_u64() < q);
    let centred = zi.to_centered_i64();
    let half = (q / 2) as i64;
    let lo = -half;
    let hi = half;
    // Range: (-floor(q/2), floor(q/2)] when q even, [-(q-1)/2, (q-1)/2] when q odd.
    // The implementation uses `a <= q/2 ? a : a - q`, so for even q the bound is inclusive at hi
    // and exclusive at lo (since q/2 maps to q/2, and q/2 + 1 maps to q/2 + 1 - q = -(q/2 - 1)).
    assert!(centred >= lo && centred <= hi, "centred={centred}, q={q}");

    // Centring round-trips: re-encoding the centred value recovers the same Zq.
    let again = Zq::from_i64(m, centred);
    assert_eq!(zi.to_u64(), again.to_u64());
});
