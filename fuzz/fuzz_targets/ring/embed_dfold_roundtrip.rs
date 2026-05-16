//! Fuzz: `pack_slots → unpack_slots` is identity at the `Poly` API
//! across $d$ random slot polys, parameterised over a small enum of
//! shapes so the production-relevant $d$ ratios (paper VIA-C's
//! $d = n_1/n_2 = 4$, VIA-B's smaller-$n_3$ shapes) all get fuzz
//! coverage. Originally hardcoded $d = 4$ only.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::{ConstModulus, DynModulus, Modulus, paper};

const KNOWN_MODULI: &[u64] = &[
    16,
    256,
    17,
    257,
    8380417,
    2147352577,
    17175674881,
    34359214081,
    137438822401,
    274810798081,
];

/// Largest `N_LARGE` across the shapes below; sized for the raw bytes
/// buffer that we re-shape per case. Keep an eye on this if a future
/// shape pushes the buffer above ~512 bytes — fuzz inputs are bounded.
const MAX_N_LARGE: usize = 32;

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

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichShape {
    /// $(N_\mathrm{small}, N_\mathrm{large}, d) = (4, 16, 4)$. Paper VIA-C
    /// $n_1 / n_2$ ratio.
    D4,
    /// $(2, 16, 8)$ — same `N_LARGE`, half the slot degree.
    D8,
    /// $(2, 32, 16)$ — wider $d$, closer to VIA-B's tiny-$n_3$ regime.
    D16,
    // VIA-B's $n_3 = 1$ case (slot degree 1, $d = n_1$) cannot be
    // fuzzed at the typed `Poly` API: the `_CHECK` block in
    // `Poly<N, M, F>` requires `N >= 2`. The slice-level kernel
    // (`reshape::pack_slots_slice`) does support $n_\text{small} = 1$
    // and is locked by the unit test
    // `pack_then_unpack_identity_n_small_1_n_large_8` in `reshape.rs`;
    // a slice-level fuzz target for that path would be a separate
    // addition.
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichConst,
    shape: WhichShape,
    dyn_mod: FuzzModulus,
    raw: [u64; MAX_N_LARGE],
}

/// Generic round-trip check. Caller supplies the slot polys as a fixed
/// `[[u64; N_SMALL]; D]` array; the `N_SMALL * D == N_LARGE` constraint
/// is checked at runtime (the const-generic relationship would need
/// more compiler support to express).
fn check<M: Modulus, const N_SMALL: usize, const N_LARGE: usize, const D: usize>(
    m: M,
    slots_vals: [[u64; N_SMALL]; D],
) {
    debug_assert_eq!(N_SMALL * D, N_LARGE);
    let slot_polys: [Poly<N_SMALL, M, Coefficient>; D] =
        core::array::from_fn(|j| Poly::new(m, slots_vals[j]));
    let packed: Poly<N_LARGE, M, Coefficient> = Poly::pack_slots::<N_LARGE>(m, &slot_polys);
    let mut back: [Poly<N_SMALL, M, Coefficient>; D] = [Poly::zero(m); D];
    packed.unpack_slots::<N_SMALL>(&mut back);
    for (j, (b, s)) in back.iter().zip(slot_polys.iter()).enumerate() {
        assert_eq!(b, s, "slot j={j}");
    }
}

/// Reshape a flat `[u64; MAX_N_LARGE]` slice into a per-shape slot grid.
fn reshape<const N_SMALL: usize, const D: usize>(raw: &[u64; MAX_N_LARGE]) -> [[u64; N_SMALL]; D] {
    let mut out = [[0u64; N_SMALL]; D];
    for j in 0..D {
        for i in 0..N_SMALL {
            out[j][i] = raw[j * N_SMALL + i];
        }
    }
    out
}

fn dispatch_shape<M: Modulus>(m: M, shape: WhichShape, raw: &[u64; MAX_N_LARGE]) {
    match shape {
        WhichShape::D4 => check::<M, 4, 16, 4>(m, reshape::<4, 4>(raw)),
        WhichShape::D8 => check::<M, 2, 16, 8>(m, reshape::<2, 8>(raw)),
        WhichShape::D16 => check::<M, 2, 32, 16>(m, reshape::<2, 16>(raw)),
    }
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichConst::Q17 => dispatch_shape(ConstModulus::<17>, input.shape, &input.raw),
        WhichConst::ViaCQ3 => dispatch_shape(paper::ViaCQ3::default(), input.shape, &input.raw),
        WhichConst::Dyn => dispatch_shape(input.dyn_mod.0, input.shape, &input.raw),
    }
});
