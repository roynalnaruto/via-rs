//! Fuzz: §6.2 RespComp paper-path round-trip.
//!
//! On the paper's asymmetric Figure-7 path (`mod_switch_sym → ring_switch →
//! mod_switch_asym`), compressing a fresh low-noise answer ciphertext and
//! recovering it with `decrypt_asymmetric` must return the constant term that
//! survives ring switching. Also pins the `(mask @ q3, body @ q4)` shape.
//! Catches a modulus/degree slip in the 3-step compression or a rounding bug.
//!
//! Run with `cargo +nightly fuzz run protocol_resp_comp_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::encryption::SecretKey;
use via_primitives::encryption::rlwe::encode;
use via_primitives::sampling::{Distribution, Shake256Prg};
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_server::resp_comp;

const N1: usize = 8;
const N2: usize = 4;
const D: usize = 2;
const L_RSK: usize = 8;

const Q2: u64 = 1 << 28;
const Q3: u64 = 1 << 20;
const Q4: u64 = 1 << 12;
const P: u64 = 16;
const B_RSK: u64 = 8;

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;

#[derive(Debug)]
struct Input {
    s1_seed: Vec<u8>,
    s2_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    rsk_seed: Vec<u8>,
    message: u64,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed = |u: &mut Unstructured<'a>| -> arbitrary::Result<Vec<u8>> {
            let l = u.int_in_range::<usize>(1..=32)?;
            let mut s = vec![0u8; l];
            u.fill_buffer(&mut s)?;
            Ok(s)
        };
        Ok(Input {
            s1_seed: seed(u)?,
            s2_seed: seed(u)?,
            enc_seed: seed(u)?,
            rsk_seed: seed(u)?,
            message: u.arbitrary()?,
        })
    }
}

fuzz_target!(|input: Input| {
    let q2 = DynModulus::new(Q2);
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);
    let c = input.message % P;

    // S1 @ q2 (answer key), S2 @ q3 (target key).
    let mut s1_prg = Shake256Prg::new(&input.s1_seed);
    let sk1 = SecretKey::<N1, R8>::keygen(q2, Distribution::Ternary, &mut s1_prg);
    let mut s2_prg = Shake256Prg::new(&input.s2_seed);
    let sk2 = SecretKey::<N2, R4>::keygen(q3, Distribution::Ternary, &mut s2_prg);

    // Fresh answer ciphertext @ q2 under S1, constant term = c.
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let mut coeffs = [0u64; N1];
    coeffs[0] = c;
    let msg = R8::new(p, coeffs);
    let ct = sk1.encrypt(
        &encode::<N1, R8, R8>(&msg, q2),
        Distribution::Ternary,
        &mut enc_prg,
    );

    // Ring-switch key S1→S2 @ q3 (rekey S1: q2→q3, then gen_rsk).
    let s1_q3 = rekey_secret_key::<N1, R8, R8>(&sk1, q3);
    let mut rsk_prg = Shake256Prg::new(&input.rsk_seed);
    let rsk = gen_rsk::<N1, N2, R8, R4, L_RSK, D>(
        &s1_q3,
        &sk2,
        B_RSK,
        Distribution::Ternary,
        &mut rsk_prg,
    );

    // T7: `resp_comp` consumes the eval-form ring-switch key (derived once,
    // bit-identical to the coeff key via exact NTT — the round-trip is unchanged).
    let rsk_eval = rsk.to_eval();
    let answer = resp_comp::<N1, N2, R8, R4, R4, L_RSK, D>(&ct, &rsk_eval, q3, q4, B_RSK);

    // Shape: mask reduced mod q3, body mod q4.
    for i in 0..N2 {
        assert!(answer.mask.coeff(i).to_u64() < Q3, "mask coeff {i} ≥ q3");
        assert!(answer.body.coeff(i).to_u64() < Q4, "body coeff {i} ≥ q4");
    }

    let recovered: R4 = sk2.decrypt_asymmetric::<R4, R4, R4>(&answer, q3, q4, p);
    assert_eq!(
        recovered.coeff(0).to_u64(),
        c,
        "resp_comp round-trip diverged"
    );
});
