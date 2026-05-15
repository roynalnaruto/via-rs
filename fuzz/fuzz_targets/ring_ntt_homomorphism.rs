//! Fuzz: schoolbook negacyclic `f * g` equals NTT-mediated
//! `(f.into_eval() * g.into_eval()).into_coeff()` at small $N$. Cross-
//! validates the NTT body against the §0.3 schoolbook oracle.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::element::Poly;
use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::zq::modulus::{ConstModulus, Modulus, paper};

const N: usize = 8;

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichConst {
    Q17,
    ViaCQ3,
    ViaQ3,
    ViaCQ2,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichConst,
    lhs: [u64; N],
    rhs: [u64; N],
}

fn check<C: Modulus>(c: C, lhs: [u64; N], rhs: [u64; N])
where
    C: via_rs::primitives::ring::ntt::NttFriendly<N>,
{
    let f: Poly<N, C, Coefficient> = Poly::new(c, lhs);
    let g: Poly<N, C, Coefficient> = Poly::new(c, rhs);
    let schoolbook = f * g;
    let via_ntt = (f.into_eval() * g.into_eval()).into_coeff();
    assert_eq!(via_ntt, schoolbook);
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichConst::Q17 => check(ConstModulus::<17>, input.lhs, input.rhs),
        WhichConst::ViaCQ3 => check(paper::ViaCQ3::default(), input.lhs, input.rhs),
        WhichConst::ViaQ3 => check(paper::ViaQ3::default(), input.lhs, input.rhs),
        WhichConst::ViaCQ2 => check(paper::ViaCQ2::default(), input.lhs, input.rhs),
    }
});
