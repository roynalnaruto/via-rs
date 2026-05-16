//! Fuzz: paper-modulus `ConstModulus<Q>` agrees with `DynModulus::new(Q)`
//! across the full `Poly` operator surface (add / sub / neg / mul /
//! scalar mul / mul_x_pow).

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::{DynModulus, Modulus, paper};

const N: usize = 8;

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichConst {
    ViaCQ3,
    ViaQ3,
    ViaCQ2,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichConst,
    lhs: [u64; N],
    rhs: [u64; N],
    scalar: u64,
    k: usize,
}

fn check<C: Modulus>(c: C, d: DynModulus, lhs: [u64; N], rhs: [u64; N], scalar: u64, k: usize) {
    let pa_c: Poly<N, _, Coefficient> = Poly::new(c, lhs);
    let pb_c: Poly<N, _, Coefficient> = Poly::new(c, rhs);
    let pa_d: Poly<N, _, Coefficient> = Poly::new(d, lhs);
    let pb_d: Poly<N, _, Coefficient> = Poly::new(d, rhs);

    assert_eq!((pa_c + pb_c).values(), (pa_d + pb_d).values(), "add");
    assert_eq!((pa_c - pb_c).values(), (pa_d - pb_d).values(), "sub");
    assert_eq!((-pa_c).values(), (-pa_d).values(), "neg");
    assert_eq!((pa_c * pb_c).values(), (pa_d * pb_d).values(), "mul");
    assert_eq!(
        (pa_c * scalar).values(),
        (pa_d * scalar).values(),
        "scalar_mul",
    );
    assert_eq!(
        pa_c.mul_x_pow(k).values(),
        pa_d.mul_x_pow(k).values(),
        "mul_x_pow",
    );
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichConst::ViaCQ3 => {
            let c = paper::ViaCQ3::default();
            let d = DynModulus::new(c.q());
            check(c, d, input.lhs, input.rhs, input.scalar, input.k);
        }
        WhichConst::ViaQ3 => {
            let c = paper::ViaQ3::default();
            let d = DynModulus::new(c.q());
            check(c, d, input.lhs, input.rhs, input.scalar, input.k);
        }
        WhichConst::ViaCQ2 => {
            let c = paper::ViaCQ2::default();
            let d = DynModulus::new(c.q());
            check(c, d, input.lhs, input.rhs, input.scalar, input.k);
        }
    }
});
