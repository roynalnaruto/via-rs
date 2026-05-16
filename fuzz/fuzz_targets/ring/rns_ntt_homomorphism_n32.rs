//! Fuzz: per-slot NTT round-trip homomorphism at larger $N$.
//!
//! Companion to `ring_rns_ntt_homomorphism` (which pins $N = 4$). Lifts the
//! same property — `(f * g)` via per-slot schoolbook agrees with the
//! NTT-mediated pointwise product — to $N = 32$, exercising more butterfly
//! stages (5 rounds vs 2). A regression that only manifests after a few
//! Cooley–Tukey rounds (off-by-one in twiddle indexing, half-stride
//! miscalculation, etc.) wouldn't trigger at $N = 4$ but would here.
//!
//! Kept as a separate target so the fuzz runner can budget it independently
//! from the $N = 4$ baseline (input size is $\sim$1 KiB vs $\sim$128 B).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::ring::ntt::NttFriendly;
use via_rs::algebra::ring::rns_element::PolyRns;
use via_rs::algebra::rns::basis::{RnsBasis, paper};

const N: usize = 32;

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
