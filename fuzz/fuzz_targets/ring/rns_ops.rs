//! Fuzz: `PolyRns` operator overloads (add / sub / neg / scalar mul,
//! pointwise mul on `Evaluation` form) agree with `rns::ops` slice kernels
//! applied directly to the same per-slot buffers.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::form::{Coefficient, Evaluation};
use via_rs::algebra::ring::rns_element::PolyRns;
use via_rs::algebra::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_rs::algebra::rns::ops as rns_ops;
use via_rs::algebra::zq::modulus::{DynModulus, Modulus};

const N: usize = 16;

const KNOWN_PAIRS: &[(u64, u64)] = &[
    (268369921, 536608769),
    (137438822401, 274810798081),
    (5, 11),
    (7, 13),
    (17, 257),
];

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichBasis {
    ViaQ1,
    ViaCQ1,
    Dyn,
}

#[derive(Debug)]
struct Input {
    which: WhichBasis,
    dyn_pair_idx: usize,
    lhs: [u128; N],
    rhs: [u128; N],
    scalar: u64,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut lhs = [0u128; N];
        let mut rhs = [0u128; N];
        for v in lhs.iter_mut() {
            *v = u.arbitrary()?;
        }
        for v in rhs.iter_mut() {
            *v = u.arbitrary()?;
        }
        Ok(Self {
            which: WhichBasis::arbitrary(u)?,
            dyn_pair_idx: u.int_in_range::<usize>(0..=(KNOWN_PAIRS.len() - 1))?,
            lhs,
            rhs,
            scalar: u.arbitrary()?,
        })
    }
}

fn check<B: RnsBasis>(b: B, lhs: &[u128; N], rhs: &[u128; N], scalar: u64) {
    let mut lhs0 = [0u64; N];
    let mut lhs1 = [0u64; N];
    let mut rhs0 = [0u64; N];
    let mut rhs1 = [0u64; N];
    for i in 0..N {
        let (a, c) = b.decompose_u128(lhs[i]);
        lhs0[i] = a;
        lhs1[i] = c;
        let (a, c) = b.decompose_u128(rhs[i]);
        rhs0[i] = a;
        rhs1[i] = c;
    }
    let pa: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, lhs);
    let pr: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, rhs);

    // add: each slot matches the underlying rns::ops::add_slice.
    let mut want0 = [0u64; N];
    let mut want1 = [0u64; N];
    rns_ops::add_slice(b, &mut want0, &mut want1, &lhs0, &lhs1, &rhs0, &rhs1);
    let sum = pa + pr;
    assert_eq!(sum.values0(), &want0, "coeff add slot0");
    assert_eq!(sum.values1(), &want1, "coeff add slot1");

    // sub
    let mut want0 = [0u64; N];
    let mut want1 = [0u64; N];
    rns_ops::sub_slice(b, &mut want0, &mut want1, &lhs0, &lhs1, &rhs0, &rhs1);
    let diff = pa - pr;
    assert_eq!(diff.values0(), &want0, "coeff sub slot0");
    assert_eq!(diff.values1(), &want1, "coeff sub slot1");

    // neg
    let mut want0 = [0u64; N];
    let mut want1 = [0u64; N];
    rns_ops::neg_slice(b, &mut want0, &mut want1, &lhs0, &lhs1);
    let neg = -pa;
    assert_eq!(neg.values0(), &want0, "coeff neg slot0");
    assert_eq!(neg.values1(), &want1, "coeff neg slot1");

    // scalar_mul: reduce scalar per slot first.
    let s0 = b.m0().reduce_u64(scalar);
    let s1 = b.m1().reduce_u64(scalar);
    let mut want0 = [0u64; N];
    let mut want1 = [0u64; N];
    rns_ops::scalar_mul_slice(b, &mut want0, &mut want1, &lhs0, &lhs1, s0, s1);
    let smul = pa * scalar;
    assert_eq!(smul.values0(), &want0, "coeff scalar_mul slot0");
    assert_eq!(smul.values1(), &want1, "coeff scalar_mul slot1");

    // Evaluation form pointwise mul ↔ rns_ops::mul_slice.
    let ea: PolyRns<N, B, Evaluation> = PolyRns::new(b, lhs0, lhs1);
    let eb: PolyRns<N, B, Evaluation> = PolyRns::new(b, rhs0, rhs1);
    let mut want0 = [0u64; N];
    let mut want1 = [0u64; N];
    rns_ops::mul_slice(b, &mut want0, &mut want1, &lhs0, &lhs1, &rhs0, &rhs1);
    let pmul = ea * eb;
    assert_eq!(pmul.values0(), &want0, "eval mul slot0");
    assert_eq!(pmul.values1(), &want1, "eval mul slot1");
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichBasis::ViaQ1 => check(
            paper::ViaQ1Rns::default(),
            &input.lhs,
            &input.rhs,
            input.scalar,
        ),
        WhichBasis::ViaCQ1 => check(
            paper::ViaCQ1Rns::default(),
            &input.lhs,
            &input.rhs,
            input.scalar,
        ),
        WhichBasis::Dyn => {
            let (q0, q1) = KNOWN_PAIRS[input.dyn_pair_idx];
            let basis = DynRnsBasis::new(DynModulus::new(q0), DynModulus::new(q1));
            check(basis, &input.lhs, &input.rhs, input.scalar);
        }
    }
});
