//! Fuzz: `ring::ops::rotate_slice` and `Poly::mul_x_pow` agree with a brute
//! reference that moves each coefficient explicitly and applies the
//! negacyclic sign rule.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::primitives::ring::element::Poly;
use via_rs::primitives::ring::form::Coefficient;
use via_rs::primitives::ring::ops;
use via_rs::primitives::zq::modulus::{DynModulus, Modulus};

const N: usize = 8;

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
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let pick_known: bool = u.arbitrary()?;
        let q = if pick_known {
            *u.choose(KNOWN_MODULI)?
        } else {
            u.int_in_range::<u64>(3..=(1u64 << 38))? | 1
        };
        Ok(FuzzModulus(DynModulus::new(q)))
    }
}

#[derive(Debug, Arbitrary)]
struct Input {
    modulus: FuzzModulus,
    src: [u64; N],
    k: usize,
}

/// Brute reference: build X^k as a polynomial and schoolbook-multiply.
fn reference_rotate(m: DynModulus, src: &[u64; N], k: usize) -> [u64; N] {
    let q = m.q();
    let k_eff = k % (2 * N);
    let k_red = k_eff % N;
    let neg = k_eff >= N;
    let mut dst = [0u64; N];
    for (i, &v) in src.iter().enumerate() {
        let wrapped = i + k_red >= N;
        let out_pos = if wrapped { i + k_red - N } else { i + k_red };
        dst[out_pos] = if wrapped ^ neg {
            if v == 0 { 0 } else { q - v }
        } else {
            v
        };
    }
    dst
}

fuzz_target!(|input: Input| {
    let m = input.modulus.0;
    let mut src = input.src;
    for v in src.iter_mut() {
        *v = m.reduce_u64(*v);
    }
    let want = reference_rotate(m, &src, input.k);

    // Direct kernel.
    let mut got = [0u64; N];
    ops::rotate_slice(m, &mut got, &src, input.k);
    assert_eq!(got, want, "ops: k={}, src={src:?}", input.k);

    // Via Poly::mul_x_pow.
    let p: Poly<N, _, Coefficient> = Poly::new(m, src);
    let p_rot = p.mul_x_pow(input.k);
    assert_eq!(p_rot.values(), &want, "Poly: k={}, src={src:?}", input.k);
});
