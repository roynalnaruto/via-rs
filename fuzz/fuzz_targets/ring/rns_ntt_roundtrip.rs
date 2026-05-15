//! Fuzz: `PolyRns::into_eval().into_coeff()` is the identity at small $N$
//! across the two paper RNS bases. Validates per-slot NTT chaining.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::ring::ntt::NttFriendly;
use via_rs::primitives::ring::rns_element::PolyRns;
use via_rs::primitives::rns::basis::{RnsBasis, paper};

const N: usize = 4;

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichBasis {
    ViaQ1,
    ViaCQ1,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichBasis,
    values: [u128; N],
}

fn check<B>(b: B, values: &[u128; N])
where
    B: RnsBasis,
    B::M0: NttFriendly<N>,
    B::M1: NttFriendly<N>,
{
    let p: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, values);
    let back = p.into_eval().into_coeff();
    assert_eq!(back, p);
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichBasis::ViaQ1 => check(paper::ViaQ1Rns::default(), &input.values),
        WhichBasis::ViaCQ1 => check(paper::ViaCQ1Rns::default(), &input.values),
    }
});
