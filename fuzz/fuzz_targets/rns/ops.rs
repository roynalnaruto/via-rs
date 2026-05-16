//! Fuzz: componentwise `RnsZq` arithmetic agrees with reference `u128`
//! arithmetic, for paper bases and arbitrary coprime-pair runtime bases.
//!
//! - `(RnsZq::from_u128(x) + RnsZq::from_u128(y)).to_u128() == (x + y) mod Q`.
//! - Same for `-`, `*`, unary `-`, and `* u64`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::rns::basis::{DynRnsBasis, RnsBasis, paper};
use via_rs::algebra::rns::element::RnsZq;
use via_rs::algebra::zq::modulus::{DynModulus, Modulus};

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
    x: u128,
    y: u128,
    scalar: u64,
}

impl<'a> Arbitrary<'a> for FuzzInput {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        Ok(Self {
            which: WhichBasis::arbitrary(u)?,
            dyn_pair_idx: u.int_in_range::<usize>(0..=(KNOWN_PAIRS.len() - 1))?,
            x: u.arbitrary()?,
            y: u.arbitrary()?,
            scalar: u.arbitrary()?,
        })
    }
}

fn check_ops<B: RnsBasis>(b: B, x: u128, y: u128, scalar: u64) {
    let q = b.big_q();
    let xr = x % q;
    let yr = y % q;
    let xz = RnsZq::from_u128(b, xr);
    let yz = RnsZq::from_u128(b, yr);

    // add
    let want_add = (xr + yr) % q;
    assert_eq!((xz + yz).to_u128(), want_add, "add: x={xr}, y={yr}");

    // sub: ((xr + Q) - yr) mod Q. xr + Q fits in u128 since Q < 2^126.
    let want_sub = (xr + q - yr) % q;
    assert_eq!((xz - yz).to_u128(), want_sub, "sub: x={xr}, y={yr}");

    // mul. xr * yr may overflow u128 only if both are within ~2^64 of Q.
    // For paper bases Q < 2^75 so xr*yr < 2^150 — overflows!
    // We compute (xr * yr) mod Q via Russian-peasant style: cast to per-prime
    // residues, multiply per component, then reconstruct. This is exactly what
    // the trait does, so we cross-check against the slow per-modulus reference.
    let (a0, a1) = b.decompose_u128(xr);
    let (c0, c1) = b.decompose_u128(yr);
    let m0 = b.m0();
    let m1 = b.m1();
    let want_a0 = m0_mul(m0.q(), a0, c0);
    let want_a1 = m0_mul(m1.q(), a1, c1);
    let want_mul_pair = (want_a0, want_a1);
    let got_pair = ((xz * yz).value0(), (xz * yz).value1());
    assert_eq!(got_pair, want_mul_pair, "mul: x={xr}, y={yr}");

    // neg
    let want_neg = if xr == 0 { 0 } else { q - xr };
    assert_eq!((-xz).to_u128(), want_neg, "neg: x={xr}");

    // scalar mul (Mul<u64>): xr * scalar mod Q, computed per-prime.
    let s0 = m0_mod(m0.q(), scalar);
    let s1 = m0_mod(m1.q(), scalar);
    let want_scalar_a0 = m0_mul(m0.q(), a0, s0);
    let want_scalar_a1 = m0_mul(m1.q(), a1, s1);
    let scaled = xz * scalar;
    assert_eq!(
        scaled.value0(),
        want_scalar_a0,
        "scalar.a0: scalar={scalar}"
    );
    assert_eq!(
        scaled.value1(),
        want_scalar_a1,
        "scalar.a1: scalar={scalar}"
    );
}

/// Naive modular multiply: `(a * b) mod q` using u128 intermediates.
fn m0_mul(q: u64, a: u64, b: u64) -> u64 {
    ((u128::from(a) * u128::from(b)) % u128::from(q)) as u64
}

/// Naive `x mod q`.
fn m0_mod(q: u64, x: u64) -> u64 {
    x % q
}

fuzz_target!(|input: FuzzInput| {
    match input.which {
        WhichBasis::ViaQ1 => check_ops(paper::ViaQ1Rns::default(), input.x, input.y, input.scalar),
        WhichBasis::ViaCQ1 => {
            check_ops(paper::ViaCQ1Rns::default(), input.x, input.y, input.scalar)
        }
        WhichBasis::Dyn => {
            let (q0, q1) = KNOWN_PAIRS[input.dyn_pair_idx];
            let basis = DynRnsBasis::new(DynModulus::new(q0), DynModulus::new(q1));
            check_ops(basis, input.x, input.y, input.scalar);
        }
    }
});
