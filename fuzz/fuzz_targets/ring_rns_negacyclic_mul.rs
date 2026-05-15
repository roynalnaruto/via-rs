//! Fuzz: `PolyRns::Mul` (coefficient form, schoolbook negacyclic per slot)
//! agrees with a `u128` reference schoolbook in $\mathbb{Z}_Q[X] /
//! (X^N + 1)$. Validates that RNS reconstruction + slot-wise multiplication
//! agree with full $\mathbb{Z}_Q$ multiplication.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::ring::rns_element::PolyRns;
use via_rs::primitives::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_rs::primitives::zq::modulus::DynModulus;

const N: usize = 4;

const KNOWN_PAIRS: &[(u64, u64)] = &[
    (5, 11),
    (7, 13),
    (17, 257),
    (268369921, 536608769),
];

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichBasis {
    ViaQ1,
    Dyn,
}

/// Constrained per-lane inputs (`u32`-wide) keep the reference schoolbook
/// product safely inside `i128` for the paper-sized basis $Q \approx 2^{57}$:
/// each $f_i g_j$ is at most $2^{64}$, summed over $N^2 = 16$ pairs, well
/// below $2^{127}$.
#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichBasis,
    dyn_pair_idx: u8,
    lhs: [u32; N],
    rhs: [u32; N],
}

fn check<B: RnsBasis>(b: B, lhs_u32: &[u32; N], rhs_u32: &[u32; N]) {
    let q = b.big_q();
    let mut lhs = [0u128; N];
    let mut rhs = [0u128; N];
    for i in 0..N {
        lhs[i] = u128::from(lhs_u32[i]) % q;
        rhs[i] = u128::from(rhs_u32[i]) % q;
    }
    let f: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, &lhs);
    let g: PolyRns<N, B, Coefficient> = PolyRns::from_u128_array(b, &rhs);
    let got = f * g;

    // Reference schoolbook in i128 with explicit negacyclic wrap, then mod Q.
    let qi = q as i128;
    let mut acc = [0i128; N];
    for i in 0..N {
        for j in 0..N {
            let p = (lhs[i] as i128) * (rhs[j] as i128);
            if i + j < N {
                acc[i + j] += p;
            } else {
                acc[i + j - N] -= p;
            }
        }
    }
    for i in 0..N {
        let want = acc[i].rem_euclid(qi) as u128;
        assert_eq!(got.coeff(i).to_u128(), want, "i={i}, q={q}");
    }
}

fuzz_target!(|input: Input| {
    match input.which {
        WhichBasis::ViaQ1 => check(paper::ViaQ1Rns::default(), &input.lhs, &input.rhs),
        WhichBasis::Dyn => {
            let idx = (input.dyn_pair_idx as usize) % KNOWN_PAIRS.len();
            let (q0, q1) = KNOWN_PAIRS[idx];
            let basis = DynRnsBasis::new(DynModulus::new(q0), DynModulus::new(q1));
            check(basis, &input.lhs, &input.rhs);
        }
    }
});
