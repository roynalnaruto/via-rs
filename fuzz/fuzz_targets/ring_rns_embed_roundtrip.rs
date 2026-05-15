//! Fuzz: `PolyRns::embed_at(j).project_at(j) == identity` across paper
//! RNS bases. RNS analogue of `ring_embed_roundtrip`.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::ring::rns_element::PolyRns;
use via_rs::primitives::rns::basis::{RnsBasis, paper};

const N_SMALL: usize = 4;
const N_LARGE: usize = 16;
const D: usize = N_LARGE / N_SMALL;

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichBasis {
    ViaQ1,
    ViaCQ1,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichBasis,
    values: [u128; N_SMALL],
    slot: u8,
}

fn check<B: RnsBasis>(b: B, values: &[u128; N_SMALL], slot: usize) {
    let f: PolyRns<N_SMALL, B, Coefficient> = PolyRns::from_u128_array(b, values);
    let slot = slot % D;
    let big: PolyRns<N_LARGE, B, Coefficient> = f.embed_at::<N_LARGE>(slot);
    let back: PolyRns<N_SMALL, B, Coefficient> = big.project_at::<N_SMALL>(slot);
    assert_eq!(back, f);
}

fuzz_target!(|input: Input| {
    let slot = input.slot as usize;
    match input.which {
        WhichBasis::ViaQ1 => check(paper::ViaQ1Rns::default(), &input.values, slot),
        WhichBasis::ViaCQ1 => check(paper::ViaCQ1Rns::default(), &input.values, slot),
    }
});
