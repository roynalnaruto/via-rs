//! Fuzz: decompose then reconstruct must round-trip for every two-prime
//! RNS basis we care about.
//!
//! - `basis.reconstruct(basis.decompose_u128(x))` == `x mod (q0 * q1)`.
//! - The two components produced by `decompose_u128` lie in their respective
//!   prime ranges.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_rs::algebra::zq::modulus::{DynModulus, Modulus};

/// Pairs of distinct, coprime u64s used to seed the dynamic basis arm.
///
/// Includes both paper products (so the corpus exercises the const-vs-dyn
/// parity at 38-bit primes) plus a small handful of toy bases for sanity.
const KNOWN_PAIRS: &[(u64, u64)] = &[
    (268369921, 536608769),       // VIA q_1
    (137438822401, 274810798081), // VIA-C / VIA-B q_1
    (5, 11),
    (7, 13),
    (17, 257),
    (8380417, 2147352577),
];

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichBasis {
    /// VIA paper q_1.
    ViaQ1,
    /// VIA-C / VIA-B paper q_1.
    ViaCQ1,
    /// Runtime basis (one of `KNOWN_PAIRS` or a freshly generated coprime pair).
    Dyn,
}

#[derive(Debug)]
struct FuzzInput {
    which: WhichBasis,
    /// Index into `KNOWN_PAIRS` when we pick from the corpus; ignored
    /// otherwise.
    dyn_pair_idx: usize,
    /// Used to derive a "fresh" coprime pair when we don't pick from the
    /// corpus.
    dyn_a: u64,
    /// Used to derive a "fresh" coprime pair when we don't pick from the
    /// corpus.
    dyn_b: u64,
    pick_known_dyn: bool,
    x: u128,
}

impl<'a> Arbitrary<'a> for FuzzInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            which: WhichBasis::arbitrary(u)?,
            dyn_pair_idx: u.int_in_range::<usize>(0..=(KNOWN_PAIRS.len() - 1))?,
            dyn_a: u.int_in_range::<u64>(3..=(1u64 << 38))?,
            dyn_b: u.int_in_range::<u64>(3..=(1u64 << 38))?,
            pick_known_dyn: u.arbitrary()?,
            x: u.arbitrary()?,
        })
    }
}

/// Naive `u128 % u128` reference (the standard library does this, but we want
/// to be explicit about the contract).
fn reference_reduce(x: u128, big_q: u128) -> u128 {
    x % big_q
}

fn check<B: RnsBasis>(b: B, x: u128) {
    let q = b.big_q();
    let xr = reference_reduce(x, q);
    let (a0, a1) = b.decompose_u128(x);
    assert!(a0 < b.m0().q(), "a0={a0}, q0={}", b.m0().q());
    assert!(a1 < b.m1().q(), "a1={a1}, q1={}", b.m1().q());
    // Decomposition is well-defined modulo Q.
    let (a0r, a1r) = b.decompose_u128(xr);
    assert_eq!((a0, a1), (a0r, a1r));
    // Reconstruction recovers x mod Q.
    assert_eq!(b.reconstruct(a0, a1), xr, "x={x}, Q={q}");
}

fn dyn_basis_from_pair(a: u64, b: u64) -> Option<DynRnsBasis> {
    if a < 2 || b < 2 || a == b {
        return None;
    }
    // Make the pair coprime by stepping b up until gcd == 1.
    let mut bb = b;
    let mut steps: u32 = 0;
    while gcd(a, bb) != 1 {
        bb = bb.wrapping_add(1);
        if bb < 2 || bb == a {
            return None;
        }
        steps += 1;
        if steps > 16 {
            return None;
        }
    }
    Some(DynRnsBasis::new(DynModulus::new(a), DynModulus::new(bb)))
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

fuzz_target!(|input: FuzzInput| {
    match input.which {
        WhichBasis::ViaQ1 => check(paper::ViaQ1Rns::default(), input.x),
        WhichBasis::ViaCQ1 => check(paper::ViaCQ1Rns::default(), input.x),
        WhichBasis::Dyn => {
            let basis = if input.pick_known_dyn {
                let (q0, q1) = KNOWN_PAIRS[input.dyn_pair_idx];
                DynRnsBasis::new(DynModulus::new(q0), DynModulus::new(q1))
            } else {
                match dyn_basis_from_pair(input.dyn_a, input.dyn_b) {
                    Some(b) => b,
                    None => return,
                }
            };
            check(basis, input.x);
        }
    }
});
