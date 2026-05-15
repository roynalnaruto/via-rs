//! Fuzz: `Poly`'s operator overloads (add / sub / neg / scalar mul, plus
//! pointwise mul on `Evaluation` form) agree with `zq::ops` slice kernels
//! applied directly to the same buffers. Locks the SoA / wrapper-level
//! equivalence.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::element::Poly;
use via_rs::primitives::ring::form::{Coefficient, Evaluation};
use via_rs::primitives::zq::modulus::{DynModulus, Modulus};
use via_rs::primitives::zq::ops as zq_ops;

const N: usize = 16;

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
    lhs: [u64; N],
    rhs: [u64; N],
    scalar: u64,
}

fuzz_target!(|input: Input| {
    let m = input.modulus.0;
    let mut lhs = input.lhs;
    let mut rhs = input.rhs;
    for v in lhs.iter_mut() {
        *v = m.reduce_u64(*v);
    }
    for v in rhs.iter_mut() {
        *v = m.reduce_u64(*v);
    }

    // Coefficient-form: add / sub / neg / scalar_mul match zq_ops kernels.
    let pa: Poly<N, _, Coefficient> = Poly::new(m, lhs);
    let pb: Poly<N, _, Coefficient> = Poly::new(m, rhs);

    let mut want = [0u64; N];
    zq_ops::add_slice(m, &mut want, &lhs, &rhs);
    assert_eq!((pa + pb).values(), &want, "coeff add");

    let mut want = [0u64; N];
    zq_ops::sub_slice(m, &mut want, &lhs, &rhs);
    assert_eq!((pa - pb).values(), &want, "coeff sub");

    let mut want = [0u64; N];
    zq_ops::neg_slice(m, &mut want, &lhs);
    assert_eq!((-pa).values(), &want, "coeff neg");

    let s = m.reduce_u64(input.scalar);
    let mut want = [0u64; N];
    zq_ops::scalar_mul_slice(m, &mut want, &lhs, s);
    assert_eq!((pa * input.scalar).values(), &want, "coeff scalar_mul");

    // Evaluation-form: pointwise mul matches `zq_ops::mul_slice`.
    let ea: Poly<N, _, Evaluation> = Poly::new(m, lhs);
    let eb: Poly<N, _, Evaluation> = Poly::new(m, rhs);
    let mut want = [0u64; N];
    zq_ops::mul_slice(m, &mut want, &lhs, &rhs);
    assert_eq!((ea * eb).values(), &want, "eval pointwise mul");
});
