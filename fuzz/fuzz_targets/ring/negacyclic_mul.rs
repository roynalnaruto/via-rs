//! Fuzz: `ring::ops::negacyclic_mul_slice` agrees with a `u128` schoolbook
//! reference that explicitly applies the negacyclic wrap `X^N = -1`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::ops;
use via_rs::algebra::zq::modulus::{DynModulus, Modulus};

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
    lhs: [u64; N],
    rhs: [u64; N],
}

fn reference_mul(q: u64, lhs: &[u64; N], rhs: &[u64; N]) -> [u64; N] {
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
    let mut out = [0u64; N];
    let qi = q as i128;
    for k in 0..N {
        out[k] = acc[k].rem_euclid(qi) as u64;
    }
    out
}

fuzz_target!(|input: Input| {
    let m = input.modulus.0;
    let q = m.q();
    let mut lhs = input.lhs;
    let mut rhs = input.rhs;
    for v in lhs.iter_mut() {
        *v = m.reduce_u64(*v);
    }
    for v in rhs.iter_mut() {
        *v = m.reduce_u64(*v);
    }
    let mut got = [0u64; N];
    ops::negacyclic_mul_slice(m, &mut got, &lhs, &rhs);
    let want = reference_mul(q, &lhs, &rhs);
    assert_eq!(got, want, "q={q}, lhs={lhs:?}, rhs={rhs:?}");
});
