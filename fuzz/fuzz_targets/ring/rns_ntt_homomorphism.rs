//! Fuzz: PolyRns schoolbook negacyclic mul == NTT-mediated pointwise mul
//! round-trip. Per-slot NTT vs per-slot schoolbook, in lockstep.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::ring::ntt::NttFriendly;
use via_primitives::algebra::ring::rns_element::PolyRns;
use via_primitives::algebra::rns::basis::{RnsBasis, paper};

const N: usize = 4;

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichBasis {
    ViaQ1,
    ViaCQ1,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichBasis,
    lhs: [u128; N],
    rhs: [u128; N],
}

fn check<B>(b: B, lhs: &[u128; N], rhs: &[u128; N])
where
    B: RnsBasis,
    B::M0: NttFriendly<N>,
    B::M1: NttFriendly<N>,
{
    let f: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, lhs);
    let g: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, rhs);
    let schoolbook = f * g;
    let via_ntt = (f.into_eval() * g.into_eval()).into_coeff();
    assert_eq!(via_ntt, schoolbook);
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichBasis::ViaQ1 => check(paper::ViaQ1Rns::default(), &input.lhs, &input.rhs),
        WhichBasis::ViaCQ1 => check(paper::ViaCQ1Rns::default(), &input.lhs, &input.rhs),
    }
});
