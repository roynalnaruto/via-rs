//! Fuzz: `pack_slots → unpack_slots` is identity at the `Poly` API
//! across $d$ random slot polys.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::element::Poly;
use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::zq::modulus::{ConstModulus, DynModulus, Modulus, paper};

const N_SMALL: usize = 4;
const N_LARGE: usize = 16;
const D: usize = N_LARGE / N_SMALL;

const KNOWN_MODULI: &[u64] = &[
    16, 256, 17, 257, 8380417, 2147352577, 17175674881, 34359214081, 137438822401, 274810798081,
];

#[derive(Debug)]
struct FuzzModulus(DynModulus);

impl<'a> Arbitrary<'a> for FuzzModulus {
    fn arbitrary(u: &mut arbitrary::Unstructured<'a>) -> arbitrary::Result<Self> {
        let q = *u.choose(KNOWN_MODULI)?;
        Ok(FuzzModulus(DynModulus::new(q)))
    }
}

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichConst {
    Q17,
    ViaCQ3,
    Dyn,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichConst,
    dyn_mod: FuzzModulus,
    slots: [[u64; N_SMALL]; D],
}

fn check<M: Modulus>(m: M, slots_vals: [[u64; N_SMALL]; D]) {
    let slot_polys: [Poly<N_SMALL, M, Coefficient>; D] =
        core::array::from_fn(|j| Poly::new(m, slots_vals[j]));
    let packed: Poly<N_LARGE, M, Coefficient> = Poly::pack_slots::<N_LARGE>(m, &slot_polys);
    let mut back: [Poly<N_SMALL, M, Coefficient>; D] = [Poly::zero(m); D];
    packed.unpack_slots::<N_SMALL>(&mut back);
    for j in 0..D {
        assert_eq!(back[j], slot_polys[j], "slot j={j}");
    }
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichConst::Q17 => check(ConstModulus::<17>, input.slots),
        WhichConst::ViaCQ3 => check(paper::ViaCQ3::default(), input.slots),
        WhichConst::Dyn => check(input.dyn_mod.0, input.slots),
    }
});
