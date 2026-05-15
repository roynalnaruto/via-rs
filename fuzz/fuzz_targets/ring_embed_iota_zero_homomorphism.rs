//! Fuzz: $\iota_0$ is a ring homomorphism — the only algebraic property
//! of $\iota$ beyond permutation. For random $f, g \in R_{N', q}$:
//!
//! ```text
//! pi_0^{N -> N'}(iota_0^{N' -> N}(f) * iota_0^{N' -> N}(g)) == f * g
//! ```
//!
//! Where both multiplications are schoolbook negacyclic in their
//! respective rings. A regression in the embed/project indexing
//! (off-by-one in the stride, swap of slots) that didn't show up in
//! the round-trip / slot-isolation fuzz targets would surface here.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::element::Poly;
use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::zq::modulus::{ConstModulus, DynModulus, Modulus, paper};

// (N_SMALL, N_LARGE) = (4, 16) — d = 4 slots. Small enough that the
// O(N^2) schoolbook negacyclic mul at N=16 is cheap (~256 mod-mul per
// fuzz iter); large enough that d=4 exercises a non-trivial stride.
const N_SMALL: usize = 4;
const N_LARGE: usize = 16;

const KNOWN_MODULI: &[u64] = &[
    16, 256, 4096, 32768, 17, 257, 8380417, 2147352577, 17175674881, 34359214081, 137438822401,
    274810798081, 268369921, 536608769,
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
    f_vals: [u64; N_SMALL],
    g_vals: [u64; N_SMALL],
}

fn check<M: Modulus>(m: M, f_vals: [u64; N_SMALL], g_vals: [u64; N_SMALL]) {
    let f: Poly<N_SMALL, M, Coefficient> = Poly::new(m, f_vals);
    let g: Poly<N_SMALL, M, Coefficient> = Poly::new(m, g_vals);
    let prod_small = f * g;

    let fe: Poly<N_LARGE, M, Coefficient> = f.embed_at::<N_LARGE>(0);
    let ge: Poly<N_LARGE, M, Coefficient> = g.embed_at::<N_LARGE>(0);
    let prod_large = fe * ge;
    let back: Poly<N_SMALL, M, Coefficient> = prod_large.project_at::<N_SMALL>(0);

    assert_eq!(back, prod_small);
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichConst::Q17 => check(ConstModulus::<17>, input.f_vals, input.g_vals),
        WhichConst::ViaCQ3 => check(paper::ViaCQ3::default(), input.f_vals, input.g_vals),
        WhichConst::Dyn => check(input.dyn_mod.0, input.f_vals, input.g_vals),
    }
});
