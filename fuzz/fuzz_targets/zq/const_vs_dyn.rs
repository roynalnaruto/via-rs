//! Fuzz: `ConstModulus<Q>` and `DynModulus::new(Q)` must produce identical
//! outputs for every paper-pinned $Q$.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::zq::modulus::{ConstModulus, DynModulus, Modulus};

#[derive(Debug, Arbitrary, Clone, Copy)]
enum WhichModulus {
    /// VIA $q_2$ prime ($\approx 2^{35}$).
    ViaQ2,
    /// VIA $q_3$ prime ($\approx 2^{31}$).
    ViaQ3,
    /// VIA-C / VIA-B $q_2$ prime ($\approx 2^{34}$).
    ViaCQ2,
    /// VIA-C / VIA-B $q_3$ prime ($\approx 2^{23}$).
    ViaCQ3,
    /// VIA $q_1$ first RNS prime.
    ViaQ1P0,
    /// VIA-C / VIA-B $q_1$ second RNS prime ($\approx 2^{38}$ — largest at §0.1).
    ViaCQ1P1,
}

#[derive(Debug, Arbitrary)]
struct Input {
    which: WhichModulus,
    a: u64,
    b: u64,
    x: u128,
}

fuzz_target!(|input: Input| {
    macro_rules! check {
        ($q:expr) => {{
            let c = ConstModulus::<{ $q }>;
            let d = DynModulus::new($q);
            let a_c = c.reduce_u64(input.a);
            let b_c = c.reduce_u64(input.b);
            let a_d = d.reduce_u64(input.a);
            let b_d = d.reduce_u64(input.b);
            assert_eq!(a_c, a_d);
            assert_eq!(b_c, b_d);
            assert_eq!(c.add(a_c, b_c), d.add(a_d, b_d));
            assert_eq!(c.sub(a_c, b_c), d.sub(a_d, b_d));
            assert_eq!(c.mul(a_c, b_c), d.mul(a_d, b_d));
            assert_eq!(c.neg(a_c), d.neg(a_d));
            assert_eq!(c.reduce_u128(input.x), d.reduce_u128(input.x));
        }};
    }

    match input.which {
        WhichModulus::ViaQ2 => check!(34359214081),
        WhichModulus::ViaQ3 => check!(2147352577),
        WhichModulus::ViaCQ2 => check!(17175674881),
        WhichModulus::ViaCQ3 => check!(8380417),
        WhichModulus::ViaQ1P0 => check!(268369921),
        WhichModulus::ViaCQ1P1 => check!(274810798081),
    }
});
