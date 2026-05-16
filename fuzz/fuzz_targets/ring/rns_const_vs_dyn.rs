//! Fuzz: paper `ConstRnsBasis` agrees with `DynRnsBasis::new(...)` across
//! the full `PolyRns` operator surface (add / sub / neg / mul /
//! scalar mul / mul_x_pow).

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::ring::rns_element::PolyRns;
use via_rs::algebra::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_rs::algebra::zq::modulus::{DynModulus, Modulus};

const N: usize = 8;

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichConst {
    ViaQ1,
    ViaCQ1,
}

#[derive(Debug)]
struct Input {
    which: WhichConst,
    lhs: [u128; N],
    rhs: [u128; N],
    scalar: u64,
    k: usize,
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
            which: WhichConst::arbitrary(u)?,
            lhs,
            rhs,
            scalar: u.arbitrary()?,
            k: u.arbitrary()?,
        })
    }
}

fn check<C: RnsBasis>(c: C, lhs: &[u128; N], rhs: &[u128; N], scalar: u64, k: usize) {
    let d = DynRnsBasis::new(DynModulus::new(c.m0().q()), DynModulus::new(c.m1().q()));

    let pa_c: PolyRns<N, C, Coefficient> = PolyRns::from_u128_array(c, lhs);
    let pb_c: PolyRns<N, C, Coefficient> = PolyRns::from_u128_array(c, rhs);
    let pa_d: PolyRns<N, DynRnsBasis, Coefficient> = PolyRns::from_u128_array(d, lhs);
    let pb_d: PolyRns<N, DynRnsBasis, Coefficient> = PolyRns::from_u128_array(d, rhs);

    // add
    let sc = pa_c + pb_c;
    let sd = pa_d + pb_d;
    assert_eq!(sc.values0(), sd.values0(), "add s0");
    assert_eq!(sc.values1(), sd.values1(), "add s1");

    // sub
    let sc = pa_c - pb_c;
    let sd = pa_d - pb_d;
    assert_eq!(sc.values0(), sd.values0(), "sub s0");
    assert_eq!(sc.values1(), sd.values1(), "sub s1");

    // neg
    let nc = -pa_c;
    let nd = -pa_d;
    assert_eq!(nc.values0(), nd.values0(), "neg s0");
    assert_eq!(nc.values1(), nd.values1(), "neg s1");

    // mul
    let mc = pa_c * pb_c;
    let md = pa_d * pb_d;
    assert_eq!(mc.values0(), md.values0(), "mul s0");
    assert_eq!(mc.values1(), md.values1(), "mul s1");

    // scalar_mul
    let sc = pa_c * scalar;
    let sd = pa_d * scalar;
    assert_eq!(sc.values0(), sd.values0(), "scalar s0");
    assert_eq!(sc.values1(), sd.values1(), "scalar s1");

    // mul_x_pow
    let rc = pa_c.mul_x_pow(k);
    let rd = pa_d.mul_x_pow(k);
    assert_eq!(rc.values0(), rd.values0(), "rot s0");
    assert_eq!(rc.values1(), rd.values1(), "rot s1");
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichConst::ViaQ1 => check(
            paper::ViaQ1Rns::default(),
            &input.lhs,
            &input.rhs,
            input.scalar,
            input.k,
        ),
        WhichConst::ViaCQ1 => check(
            paper::ViaCQ1Rns::default(),
            &input.lhs,
            &input.rhs,
            input.scalar,
            input.k,
        ),
    }
});
