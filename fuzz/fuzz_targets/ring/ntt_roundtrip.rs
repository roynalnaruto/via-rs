//! Fuzz: `Poly::into_eval().into_coeff()` is the identity at small $N$
//! across paper-friendly moduli. Verifies the NTT bijection.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::element::Poly;
use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::zq::modulus::{ConstModulus, Modulus, paper};

// Tiny N keeps fuzz iterations fast; paper-N=2048 is exercised in the
// integration test in element.rs.
const N: usize = 8;

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichConst {
    /// `ConstModulus<17>` — tiny prime, fastest iterations. NTT-friendly for N ∈ {2, 4, 8}.
    Q17,
    /// VIA-C $q_3 \approx 2^{23}$.
    ViaCQ3,
    /// VIA $q_3 \approx 2^{31}$.
    ViaQ3,
    /// VIA-C $q_2 \approx 2^{34}$.
    ViaCQ2,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichConst,
    values: [u64; N],
}

fn check<C>(c: C, values: [u64; N])
where
    C: Modulus + via_rs::primitives::ring::ntt::NttFriendly<N>,
{
    let f: Poly<N, C, Coefficient> = Poly::new(c, values);
    let back = f.into_eval().into_coeff();
    assert_eq!(back, f);
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichConst::Q17 => check(ConstModulus::<17>, input.values),
        WhichConst::ViaCQ3 => check(paper::ViaCQ3::default(), input.values),
        WhichConst::ViaQ3 => check(paper::ViaQ3::default(), input.values),
        WhichConst::ViaCQ2 => check(paper::ViaCQ2::default(), input.values),
    }
});
