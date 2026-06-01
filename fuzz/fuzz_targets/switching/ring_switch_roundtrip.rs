//! Fuzz: §3.3 RingSwitch must recover the slot-0 projection of the plaintext.
//!
//! Generate a ring-switch key from `(S1, S2)`, encrypt a plaintext under `S1`
//! in `R_{N1, q}`, ring-switch to `R_{N2, q}`, and decrypt under `S2`; the
//! recovered polynomial must equal `pi_0^{N1 -> N2}(plaintext)` (coefficient
//! `i` is plaintext coefficient `d * i`, `d = N1 / N2`). Catches the
//! negacyclic `X^{-j}` identity and the gadget-product accumulation.
//!
//! Small dimensions (`N1 = 16, N2 = 4, D = 4, L = 8`) keep iterations fast.
//! The gadget depth `L = 8` bounds the per-coefficient reconstruction error at
//! `q / B^L / 2`; even at the smallest base (`B = 2` ⇒ `q / 2^8 ≈ 2^23`) the
//! accumulated `D`-gadget-product noise stays far below `Δ/2 = q/(2p)` with the
//! large fixed `q` and `p = 2`. (At `L = 4`, `B = 2` the tail `q / 2^4 ≈ 2^27`
//! overflows the budget — a noise limit, not a code bug.) Recovery is thus
//! mathematically guaranteed and any failure is a real regression.
//!
//! Run with `cargo +nightly fuzz run switching_ring_switch_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::DynModulus;
use via_rs::encryption::{SecretKey, encode};
use via_rs::sampling::{Distribution, Shake256Prg};
use via_rs::switching::ring_switch::{RingSwitchKey, gen_rsk, ring_switch};

const N1: usize = 16;
const N2: usize = 4;
const D: usize = 4;
const L: usize = 8;
type R1 = Poly<N1, DynModulus, Coefficient>;
type R2 = Poly<N2, DynModulus, Coefficient>;
type Pt1 = Poly<N1, DynModulus, Coefficient>;
type Pt2 = Poly<N2, DynModulus, Coefficient>;

/// Large modulus + `p = 2`: Δ ≈ q/2 dwarfs the worst-case ring-switch noise
/// (D gadget products at depth L with ternary errors), so recovery is exact.
const Q: u64 = 2_147_352_577; // VIA q_3 ≈ 2^31
const P: u64 = 2;

#[derive(Debug)]
struct Input {
    key_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    base_idx: u8,
    plaintext: [u64; N1],
}

const BASES: &[u64] = &[2, 4, 8, 16];

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let kl = u.int_in_range::<usize>(1..=32)?;
        let mut key_seed = vec![0u8; kl];
        u.fill_buffer(&mut key_seed)?;
        let el = u.int_in_range::<usize>(1..=32)?;
        let mut enc_seed = vec![0u8; el];
        u.fill_buffer(&mut enc_seed)?;
        let base_idx = u.int_in_range::<u8>(0..=(BASES.len() as u8 - 1))?;
        let mut plaintext = [0u64; N1];
        for slot in &mut plaintext {
            *slot = u.arbitrary()?;
        }
        Ok(Input {
            key_seed,
            enc_seed,
            base_idx,
            plaintext,
        })
    }
}

fuzz_target!(|input: Input| {
    let base = BASES[input.base_idx as usize];
    let q = DynModulus::new(Q);
    let p = DynModulus::new(P);

    let mut key_prg = Shake256Prg::new(&input.key_seed);
    let s1 = SecretKey::<N1, R1>::keygen(q, Distribution::Ternary, &mut key_prg);
    let s2 = SecretKey::<N2, R2>::keygen(q, Distribution::Ternary, &mut key_prg);
    let rsk: RingSwitchKey<N1, N2, R2, L, D> =
        gen_rsk(&s1, &s2, base, Distribution::Ternary, &mut key_prg);

    let mut lanes = [0u64; N1];
    for (slot, &raw) in lanes.iter_mut().zip(input.plaintext.iter()) {
        *slot = raw % P;
    }
    let plaintext = Pt1::new(p, lanes);
    let encoded: R1 = encode(&plaintext, q);

    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let ct = s1.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);

    let switched = ring_switch(&ct, &rsk, base);
    let recovered: Pt2 = s2.decrypt(&switched, p);

    let d = N1 / N2;
    for i in 0..N2 {
        assert_eq!(
            recovered.coeff(i).to_u64(),
            lanes[d * i],
            "ring_switch round-trip diverged at i={i}; base={base}",
        );
    }
});
