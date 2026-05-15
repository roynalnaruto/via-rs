//! Fuzz: `Poly::to_centered_coeffs` then re-reduce via
//! `Modulus::reduce_i64` is the identity. Also `PolyRns::to_centered_coeffs`
//! then `from_u128_array((c + Q) as u128 % Q)` recovers the original.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::element::Poly;
use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::ring::rns_element::PolyRns;
use via_rs::primitives::rns::basis::{RnsBasis, paper as rns_paper};
use via_rs::primitives::zq::modulus::{ConstModulus, DynModulus, Modulus, paper};

const N: usize = 8;

const KNOWN_MODULI: &[u64] = &[
    16, 256, 4096, 17, 257, 8380417, 2147352577, 17175674881, 34359214081, 137438822401,
    274810798081,
];

#[derive(Debug)]
struct FuzzModulus(DynModulus);

impl<'a> Arbitrary<'a> for FuzzModulus {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let pick_known: bool = u.arbitrary()?;
        let q = if pick_known {
            *u.choose(KNOWN_MODULI)?
        } else {
            u.int_in_range::<u64>(3..=(1u64 << 38))? | 1
        };
        Ok(FuzzModulus(DynModulus::new(q)))
    }
}

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichLayer {
    SinglePrimeQ17,
    SinglePrimeViaCQ3,
    SinglePrimeDyn,
    RnsViaQ1,
    RnsViaCQ1,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichLayer,
    dyn_mod: FuzzModulus,
    values_u64: [u64; N],
    values_u128: [u128; N],
}

fn check_single<M: Modulus>(m: M, values: [u64; N]) {
    let f: Poly<N, M, Coefficient> = Poly::new(m, values);
    let mut centred = [0i64; N];
    f.to_centered_coeffs(&mut centred);
    let mut back = [0u64; N];
    for (b, &c) in back.iter_mut().zip(centred.iter()) {
        *b = m.reduce_i64(c);
    }
    let back_poly: Poly<N, M, Coefficient> = Poly::new(m, back);
    assert_eq!(back_poly, f);
}

fn check_rns<B: RnsBasis>(b: B, values: &[u128; N]) {
    let f: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, values);
    let mut centred = [0i128; N];
    f.to_centered_coeffs(&mut centred);
    let q = b.big_q();
    let q_i = q as i128;
    let mut back = [0u128; N];
    for (slot, &c) in back.iter_mut().zip(centred.iter()) {
        let r = c.rem_euclid(q_i);
        *slot = r as u128;
    }
    let back_poly: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, &back);
    assert_eq!(back_poly, f);
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichLayer::SinglePrimeQ17 => check_single(ConstModulus::<17>, input.values_u64),
        WhichLayer::SinglePrimeViaCQ3 => check_single(paper::ViaCQ3::default(), input.values_u64),
        WhichLayer::SinglePrimeDyn => check_single(input.dyn_mod.0, input.values_u64),
        WhichLayer::RnsViaQ1 => check_rns(rns_paper::ViaQ1Rns::default(), &input.values_u128),
        WhichLayer::RnsViaCQ1 => check_rns(rns_paper::ViaCQ1Rns::default(), &input.values_u128),
    }
});
