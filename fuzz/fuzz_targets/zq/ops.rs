//! Fuzz: `add` / `sub` / `mul` / `neg` agree with reference `u128` arithmetic,
//! for arbitrary moduli and inputs reduced into $[0, q)$.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::zq::modulus::{DynModulus, Modulus};

const KNOWN_MODULI: &[u64] = &[
    16,
    256,
    4096,
    32768,
    17,
    257,
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
    a: u64,
    b: u64,
}

fuzz_target!(|input: Input| {
    let m = input.modulus.0;
    let q = m.q();
    // Reduce inputs into canonical [0, q) so the operator preconditions hold.
    let a = m.reduce_u64(input.a);
    let b = m.reduce_u64(input.b);
    let q128 = u128::from(q);

    // add
    let want_add = ((u128::from(a) + u128::from(b)) % q128) as u64;
    assert_eq!(m.add(a, b), want_add, "add: q={q}, a={a}, b={b}");

    // sub: ((a - b) mod q)
    let want_sub = ((u128::from(a) + q128 - u128::from(b)) % q128) as u64;
    assert_eq!(m.sub(a, b), want_sub, "sub: q={q}, a={a}, b={b}");

    // mul
    let want_mul = ((u128::from(a) * u128::from(b)) % q128) as u64;
    assert_eq!(m.mul(a, b), want_mul, "mul: q={q}, a={a}, b={b}");

    // neg
    let want_neg = if a == 0 { 0 } else { q - a };
    assert_eq!(m.neg(a), want_neg, "neg: q={q}, a={a}");
});
