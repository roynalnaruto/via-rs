//! Fuzz: $\pi_{j'} \circ \iota_j = 0$ for $j' \ne j$ on `PolyRns`.
//! After embedding at slot `j`, projecting at any *other* slot
//! yields the zero polynomial — slots are disjoint, per RNS slot.
//! RNS analogue of `ring_embed_slot_isolation`.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::ring::rns_element::PolyRns;
use via_rs::primitives::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_rs::primitives::zq::modulus::DynModulus;

const N_SMALL: usize = 4;
const N_LARGE: usize = 16;
const D: usize = N_LARGE / N_SMALL;

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

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichBasis,
    dyn_pair_idx: u8,
    values: [u128; N_SMALL],
    slot: u8,
}

fn check<B: RnsBasis>(b: B, values: &[u128; N_SMALL], slot: usize) {
    let f: PolyRns<N_SMALL, B, Coefficient> = PolyRns::from_u128_array(b, values);
    let zero: PolyRns<N_SMALL, B, Coefficient> = PolyRns::zero(b);
    let slot = slot % D;
    let big: PolyRns<N_LARGE, B, Coefficient> = f.embed_at::<N_LARGE>(slot);
    for jp in 0..D {
        if jp == slot {
            continue;
        }
        let back: PolyRns<N_SMALL, B, Coefficient> = big.project_at::<N_SMALL>(jp);
        assert_eq!(back, zero, "embed {slot}, project {jp}");
    }
}

fuzz_target!(|input: Input| {
    let slot = input.slot as usize;
    match input.which {
        WhichBasis::ViaQ1 => check(paper::ViaQ1Rns::default(), &input.values, slot),
        WhichBasis::ViaCQ1 => check(paper::ViaCQ1Rns::default(), &input.values, slot),
        WhichBasis::Dyn => {
            let idx = (input.dyn_pair_idx as usize) % KNOWN_PAIRS.len();
            let (q0, q1) = KNOWN_PAIRS[idx];
            let basis = DynRnsBasis::new(DynModulus::new(q0), DynModulus::new(q1));
            check(basis, &input.values, slot);
        }
    }
});
