//! Fuzz: `ConstRnsBasis<Q0, Q1>` and `DynRnsBasis::new(DynModulus::new(Q0),
//! DynModulus::new(Q1))` must produce identical outputs for every paper-pinned
//! basis. This is the §0.2 analogue of `zq_const_vs_dyn`.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_primitives::algebra::rns::element::RnsZq;
use via_primitives::algebra::zq::modulus::DynModulus;

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichBasis {
    /// VIA paper q_1 (≈ 2^57).
    ViaQ1,
    /// VIA-C / VIA-B paper q_1 (≈ 2^75) — the largest Q in any parameter set.
    ViaCQ1,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichBasis,
    x: u128,
    y: u128,
    scalar: u64,
}

fn check<C, D>(c: C, d: D, x: u128, y: u128, scalar: u64)
where
    C: RnsBasis,
    D: RnsBasis,
{
    // Sanity: both bases describe the same composite modulus.
    assert_eq!(c.big_q(), d.big_q());
    assert_eq!(c.q0_inv_mod_q1(), d.q0_inv_mod_q1());

    // Decomposition and reconstruction agree.
    let (cx0, cx1) = c.decompose_u128(x);
    let (dx0, dx1) = d.decompose_u128(x);
    assert_eq!((cx0, cx1), (dx0, dx1));
    assert_eq!(c.reconstruct(cx0, cx1), d.reconstruct(dx0, dx1));

    // Operator overloads agree (we route through the RnsZq wrapper).
    let xc = RnsZq::from_u128(c, x);
    let yc = RnsZq::from_u128(c, y);
    let xd = RnsZq::from_u128(d, x);
    let yd = RnsZq::from_u128(d, y);
    assert_eq!((xc + yc).to_u128(), (xd + yd).to_u128(), "add");
    assert_eq!((xc - yc).to_u128(), (xd - yd).to_u128(), "sub");
    assert_eq!((xc * yc).to_u128(), (xd * yd).to_u128(), "mul");
    assert_eq!((-xc).to_u128(), (-xd).to_u128(), "neg");
    assert_eq!((xc * scalar).to_u128(), (xd * scalar).to_u128(), "mul-u64");
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichBasis::ViaQ1 => {
            let c = paper::ViaQ1Rns::default();
            let d = DynRnsBasis::new(DynModulus::new(268369921), DynModulus::new(536608769));
            check(c, d, input.x, input.y, input.scalar);
        }
        WhichBasis::ViaCQ1 => {
            let c = paper::ViaCQ1Rns::default();
            let d = DynRnsBasis::new(DynModulus::new(137438822401), DynModulus::new(274810798081));
            check(c, d, input.x, input.y, input.scalar);
        }
    }
});
