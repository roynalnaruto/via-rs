//! Fuzz: §4.6 converted RGSW is usable in `external_product`.
//!
//! Convert per-level `RLWE(1·g[i])` into `RGSW(1)` via the `RLev_S(S^2)`
//! conversion key, then check `RGSW(1) ⊠ RLWE(M') == RLWE(M')`. Catches a wrong
//! `neg_s_m` construction (sign / field order) in the conversion.
//!
//! Run with `cargo +nightly fuzz run gates_rlwe_to_rgsw_usable`.

#![no_main]
// The per-coefficient checks index `.coeff(i)` (a method, not a slice), so a
// range loop is the clearest form.
#![allow(clippy::needless_range_loop)]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::encryption::gadget::gadget_vector_values;
use via_primitives::encryption::types::RLWECiphertext;
use via_primitives::encryption::{SecretKey, encode};
use via_primitives::gates::{gen_rlwe_to_rgsw_key, rlwe_to_rgsw};
use via_primitives::sampling::{Distribution, Shake256Prg};

const N: usize = 8;
const L_OUT: usize = 3;
const L_CK: usize = 16;
const BASE: u64 = 4;
const CK_BASE: u64 = 2;
type R = Poly<N, DynModulus, Coefficient>;

const KNOWN_Q: &[u64] = &[65537, 786433];
const P: u64 = 2;

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    level_seed: Vec<u8>,
    ck_seed: Vec<u8>,
    m_seed: Vec<u8>,
    prime_seed: Vec<u8>,
    q_idx: u8,
    msg_prime: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed = |u: &mut Unstructured<'a>| -> arbitrary::Result<Vec<u8>> {
            let l = u.int_in_range::<usize>(1..=32)?;
            let mut s = vec![0u8; l];
            u.fill_buffer(&mut s)?;
            Ok(s)
        };
        let sk_seed = seed(u)?;
        let level_seed = seed(u)?;
        let ck_seed = seed(u)?;
        let m_seed = seed(u)?;
        let prime_seed = seed(u)?;
        let q_idx = u.arbitrary()?;
        let mut msg_prime = [0u64; N];
        for slot in &mut msg_prime {
            *slot = u.arbitrary()?;
        }
        Ok(Input {
            sk_seed,
            level_seed,
            ck_seed,
            m_seed,
            prime_seed,
            q_idx,
            msg_prime,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_val = KNOWN_Q[input.q_idx as usize % KNOWN_Q.len()];

    // Conversion error ~ (q / ck_base^L_CK) * ||S^2|| * N, plus the external
    // product reconstruction tail q / BASE^L_OUT. Gate conservatively.
    let conv_err = (q_val / CK_BASE.pow(L_CK as u32) + 1) * (N as u64) * (N as u64);
    let ext_tail = q_val / BASE.pow(L_OUT as u32);
    if 8 * (conv_err + ext_tail) >= q_val / P {
        return;
    }

    let q = DynModulus::new(q_val);
    let p = DynModulus::new(P);

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(q, Distribution::Ternary, &mut sk_prg);

    // m_poly = constant 1.
    let mut one = [0u128; N];
    one[0] = 1;
    let m_poly = <R as RingPoly<N>>::from_u128_coeffs(q, &one);

    // Per-level RLWE(1 * g[i]).
    let g = gadget_vector_values::<N, R, L_OUT>(q, BASE);
    let mut level_prg = Shake256Prg::new(&input.level_seed);
    let ct_levels: [RLWECiphertext<N, R>; L_OUT] = core::array::from_fn(|i| {
        let mut gc = [0u128; N];
        gc[0] = g[i];
        let g_poly = <R as RingPoly<N>>::from_u128_coeffs(q, &gc);
        let scaled = m_poly * g_poly;
        sk.encrypt(&scaled, Distribution::Ternary, &mut level_prg)
    });

    let mut ck_prg = Shake256Prg::new(&input.ck_seed);
    let conv_key =
        gen_rlwe_to_rgsw_key::<N, R, L_CK>(&sk, CK_BASE, Distribution::Ternary, &mut ck_prg);
    let mut m_prg = Shake256Prg::new(&input.m_seed);
    let m_rlev = sk.encrypt_rlev::<L_OUT>(&m_poly, BASE, Distribution::Ternary, &mut m_prg);

    let rgsw = rlwe_to_rgsw::<N, R, L_OUT, L_CK>(ct_levels, &conv_key, m_rlev, CK_BASE);

    // External product with RLWE(M') must recover M' (since M = 1).
    let mut m = [0u64; N];
    for i in 0..N {
        m[i] = input.msg_prime[i] % P;
    }
    let mut prime_prg = Shake256Prg::new(&input.prime_seed);
    let ct_prime = sk.encrypt(
        &encode(&R::new(p, m), q),
        Distribution::Ternary,
        &mut prime_prg,
    );

    let result = rgsw.external_product(&ct_prime, BASE, BASE);
    let rec: R = sk.decrypt(&result, p);
    for i in 0..N {
        assert_eq!(
            rec.coeff(i).to_u64(),
            m[i],
            "rlwe_to_rgsw usability diverged at i={i}"
        );
    }
});
