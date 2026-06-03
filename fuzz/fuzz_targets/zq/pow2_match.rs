//! Fuzz: `PowerOfTwoModulus<L>` and `ConstModulus<{1 << L}>` must agree.
//!
//! Catches accidental divergences between the specialised mask reduction and
//! the generic Barrett reduction when the modulus is a power of two.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::zq::modulus::{ConstModulus, Modulus, PowerOfTwoModulus};

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichLog2 {
    /// VIA-C / VIA-B $p = 16$.
    Log4,
    /// VIA $p = 256$.
    Log8,
    /// VIA-C / VIA-B $q_4 = 4096$.
    Log12,
    /// VIA $q_4 = 32768$.
    Log15,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichLog2,
    a: u64,
    b: u64,
    x: u128,
}

fuzz_target!(|input: Input| {
    macro_rules! check {
        ($log:expr) => {{
            let pow = PowerOfTwoModulus::<{ $log }>;
            let cst = ConstModulus::<{ 1u64 << $log }>;
            let a = pow.reduce_u64(input.a);
            let b = pow.reduce_u64(input.b);
            assert_eq!(pow.add(a, b), cst.add(a, b));
            assert_eq!(pow.sub(a, b), cst.sub(a, b));
            assert_eq!(pow.mul(a, b), cst.mul(a, b));
            assert_eq!(pow.neg(a), cst.neg(a));
            assert_eq!(pow.reduce_u128(input.x), cst.reduce_u128(input.x));
        }};
    }

    match input.which {
        WhichLog2::Log4 => check!(4),
        WhichLog2::Log8 => check!(8),
        WhichLog2::Log12 => check!(12),
        WhichLog2::Log15 => check!(15),
    }
});
