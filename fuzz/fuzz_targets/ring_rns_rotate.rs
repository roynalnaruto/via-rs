//! Fuzz: `PolyRns::mul_x_pow(k)` and the underlying `rns_ops::rotate_slice`
//! agree with a `u128` schoolbook-times-$X^k$ reference. RNS analogue of
//! the single-prime `ring_rotate`.
//!
//! Validates that the per-slot dispatch of the negacyclic rotation is
//! consistent with the full $\mathbb{Z}_Q$ semantics — a regression that
//! flips a wrap or swaps slots would surface here but not in the
//! existing `ring_rns_negacyclic_mul` target (which checks schoolbook
//! mul, not rotation specifically).
//!
//! Constrained per-lane inputs (`u32`-wide) so the u128 reference stays
//! well within `u128` for the paper-sized basis $Q \approx 2^{75}$.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::ring::rns_element::PolyRns;
use via_rs::primitives::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_rs::primitives::zq::modulus::DynModulus;

const N: usize = 8;

const KNOWN_PAIRS: &[(u64, u64)] = &[
    (5, 11),
    (7, 13),
    (17, 257),
    (268369921, 536608769),
    (137438822401, 274810798081),
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
    src: [u32; N],
    k: usize,
}

/// Reference: for each $i$, the coefficient $\mathrm{src}[i] \cdot X^i$
/// becomes $\mathrm{src}[i] \cdot X^{i + k}$, with $X^n \equiv -1$
/// applied to keep degrees in $[0, n)$.
fn reference_rotate_u128(q: u128, src: &[u128; N], k: usize) -> [u128; N] {
    let k_eff = k % (2 * N);
    let k_red = k_eff % N;
    let neg = k_eff >= N;
    let mut dst = [0u128; N];
    for (i, &v_raw) in src.iter().enumerate() {
        let wrapped = i + k_red >= N;
        let out_pos = if wrapped { i + k_red - N } else { i + k_red };
        let v = v_raw % q;
        dst[out_pos] = if wrapped ^ neg {
            if v == 0 { 0 } else { q - v }
        } else {
            v
        };
    }
    dst
}

fn check<B: RnsBasis>(b: B, src_u32: &[u32; N], k: usize) {
    let q = b.big_q();
    let mut src_u128 = [0u128; N];
    for (i, &v) in src_u32.iter().enumerate() {
        src_u128[i] = u128::from(v) % q;
    }
    let p: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, &src_u128);
    let got = p.mul_x_pow(k);
    let want = reference_rotate_u128(q, &src_u128, k);
    for (i, &w) in want.iter().enumerate() {
        assert_eq!(got.coeff(i).to_u128(), w, "i={i} k={k} q={q}");
    }
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichBasis::ViaQ1 => check(paper::ViaQ1Rns::default(), &input.src, input.k),
        WhichBasis::ViaCQ1 => check(paper::ViaCQ1Rns::default(), &input.src, input.k),
        WhichBasis::Dyn => {
            let idx = (input.dyn_pair_idx as usize) % KNOWN_PAIRS.len();
            let (q0, q1) = KNOWN_PAIRS[idx];
            let basis = DynRnsBasis::new(DynModulus::new(q0), DynModulus::new(q1));
            check(basis, &input.src, input.k);
        }
    }
});
