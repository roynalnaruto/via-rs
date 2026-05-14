//! Fuzz: SoA slice kernels (`add_slice`, `sub_slice`, `mul_slice`,
//! `neg_slice`, `scalar_mul_slice`, `decompose_slice`, `reconstruct_slice`)
//! must agree element-by-element with the per-element [`RnsZq`] operations.
//! Catches any divergence between the bulk and scalar paths.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_rs::primitives::rns::element::RnsZq;
use via_rs::primitives::rns::ops;
use via_rs::primitives::zq::modulus::DynModulus;

const MAX_LEN: usize = 16;

const KNOWN_PAIRS: &[(u64, u64)] = &[
    (268369921, 536608769),
    (137438822401, 274810798081),
    (5, 11),
    (7, 13),
    (17, 257),
];

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichBasis {
    ViaQ1,
    ViaCQ1,
    Dyn,
}

#[derive(Debug)]
struct FuzzInput {
    which: WhichBasis,
    dyn_pair_idx: usize,
    /// Length of the slice; clamped to `[1, MAX_LEN]`.
    len: u8,
    lhs: [u128; MAX_LEN],
    rhs: [u128; MAX_LEN],
    scalar: u64,
}

impl<'a> Arbitrary<'a> for FuzzInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut lhs = [0u128; MAX_LEN];
        let mut rhs = [0u128; MAX_LEN];
        for slot in &mut lhs {
            *slot = u.arbitrary()?;
        }
        for slot in &mut rhs {
            *slot = u.arbitrary()?;
        }
        Ok(Self {
            which: WhichBasis::arbitrary(u)?,
            dyn_pair_idx: u.int_in_range::<usize>(0..=(KNOWN_PAIRS.len() - 1))?,
            len: u.int_in_range::<u8>(1..=(MAX_LEN as u8))?,
            lhs,
            rhs,
            scalar: u.arbitrary()?,
        })
    }
}

fn check<B: RnsBasis>(b: B, lhs: &[u128], rhs: &[u128], scalar: u64) {
    let n = lhs.len();
    assert_eq!(rhs.len(), n);

    let mut lhs0 = [0u64; MAX_LEN];
    let mut lhs1 = [0u64; MAX_LEN];
    let mut rhs0 = [0u64; MAX_LEN];
    let mut rhs1 = [0u64; MAX_LEN];
    ops::decompose_slice(b, &mut lhs0[..n], &mut lhs1[..n], lhs);
    ops::decompose_slice(b, &mut rhs0[..n], &mut rhs1[..n], rhs);

    // add
    let mut out0 = [0u64; MAX_LEN];
    let mut out1 = [0u64; MAX_LEN];
    ops::add_slice(
        b,
        &mut out0[..n],
        &mut out1[..n],
        &lhs0[..n],
        &lhs1[..n],
        &rhs0[..n],
        &rhs1[..n],
    );
    for i in 0..n {
        let want = RnsZq::from_u128(b, lhs[i]) + RnsZq::from_u128(b, rhs[i]);
        assert_eq!(out0[i], want.value0(), "add.a0 i={i}");
        assert_eq!(out1[i], want.value1(), "add.a1 i={i}");
    }

    // sub
    ops::sub_slice(
        b,
        &mut out0[..n],
        &mut out1[..n],
        &lhs0[..n],
        &lhs1[..n],
        &rhs0[..n],
        &rhs1[..n],
    );
    for i in 0..n {
        let want = RnsZq::from_u128(b, lhs[i]) - RnsZq::from_u128(b, rhs[i]);
        assert_eq!(out0[i], want.value0(), "sub.a0 i={i}");
        assert_eq!(out1[i], want.value1(), "sub.a1 i={i}");
    }

    // mul
    ops::mul_slice(
        b,
        &mut out0[..n],
        &mut out1[..n],
        &lhs0[..n],
        &lhs1[..n],
        &rhs0[..n],
        &rhs1[..n],
    );
    for i in 0..n {
        let want = RnsZq::from_u128(b, lhs[i]) * RnsZq::from_u128(b, rhs[i]);
        assert_eq!(out0[i], want.value0(), "mul.a0 i={i}");
        assert_eq!(out1[i], want.value1(), "mul.a1 i={i}");
    }

    // neg
    ops::neg_slice(b, &mut out0[..n], &mut out1[..n], &lhs0[..n], &lhs1[..n]);
    for i in 0..n {
        let want = -RnsZq::from_u128(b, lhs[i]);
        assert_eq!(out0[i], want.value0(), "neg.a0 i={i}");
        assert_eq!(out1[i], want.value1(), "neg.a1 i={i}");
    }

    // scalar_mul — reduce scalar through each modulus first.
    let s0 = b.m0().reduce_u64(scalar);
    let s1 = b.m1().reduce_u64(scalar);
    ops::scalar_mul_slice(
        b,
        &mut out0[..n],
        &mut out1[..n],
        &lhs0[..n],
        &lhs1[..n],
        s0,
        s1,
    );
    for i in 0..n {
        let want = RnsZq::from_u128(b, lhs[i]) * scalar;
        assert_eq!(out0[i], want.value0(), "scalar_mul.a0 i={i}");
        assert_eq!(out1[i], want.value1(), "scalar_mul.a1 i={i}");
    }

    // reconstruct_slice
    let mut back = [0u128; MAX_LEN];
    ops::reconstruct_slice(b, &mut back[..n], &lhs0[..n], &lhs1[..n]);
    let q = b.big_q();
    for i in 0..n {
        assert_eq!(back[i], lhs[i] % q, "reconstruct i={i}");
    }
}

// Bring `Modulus::reduce_u64` into scope for `scalar_mul`'s scalar reduction.
use via_rs::primitives::zq::modulus::Modulus;

fuzz_target!(|input: FuzzInput| {
    let n = input.len as usize;
    let lhs = &input.lhs[..n];
    let rhs = &input.rhs[..n];
    match input.which {
        WhichBasis::ViaQ1 => check(paper::ViaQ1Rns::default(), lhs, rhs, input.scalar),
        WhichBasis::ViaCQ1 => check(paper::ViaCQ1Rns::default(), lhs, rhs, input.scalar),
        WhichBasis::Dyn => {
            let (q0, q1) = KNOWN_PAIRS[input.dyn_pair_idx];
            let basis = DynRnsBasis::new(DynModulus::new(q0), DynModulus::new(q1));
            check(basis, lhs, rhs, input.scalar);
        }
    }
});
