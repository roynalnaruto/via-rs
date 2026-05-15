//! Fuzz: `Poly::embed_at(j).project_at(j) == identity` for every slot,
//! across paper-friendly moduli. Confirms the bijection $\pi_j \circ
//! \iota_j = \mathrm{id}$ at the `Poly` API.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::element::Poly;
use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::zq::modulus::{ConstModulus, DynModulus, Modulus, paper};

// (N_small, N_large) = (8, 32) — d = 4 slots.
const N_SMALL: usize = 8;
const N_LARGE: usize = 32;

const KNOWN_MODULI: &[u64] = &[
    16,
    256,
    4096,
    32768,
    17,
    257,
    8380417,
    2147352577,
    17175674881,
    34359214081,
    137438822401,
    274810798081,
    268369921,
    536608769,
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
enum WhichConst {
    Q17,
    ViaCQ3,
    Dyn,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichConst,
    dyn_mod: FuzzModulus,
    values: [u64; N_SMALL],
    slot: u8,
}

fn check<M: Modulus>(m: M, values: [u64; N_SMALL], slot: usize) {
    let f: Poly<N_SMALL, M, Coefficient> = Poly::new(m, values);
    let d = N_LARGE / N_SMALL;
    let slot = slot % d;
    let big: Poly<N_LARGE, M, Coefficient> = f.embed_at::<N_LARGE>(slot);
    let back: Poly<N_SMALL, M, Coefficient> = big.project_at::<N_SMALL>(slot);
    assert_eq!(back, f);
}

fuzz_target!(|input: Input| {
    let slot = input.slot as usize;
    match input.which {
        WhichConst::Q17 => check(ConstModulus::<17>, input.values, slot),
        WhichConst::ViaCQ3 => check(paper::ViaCQ3::default(), input.values, slot),
        WhichConst::Dyn => check(input.dyn_mod.0, input.values, slot),
    }
});
