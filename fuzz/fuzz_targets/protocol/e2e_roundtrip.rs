//! Fuzz: §6 VIA-C client↔server end-to-end round-trip.
//!
//! For any index and any record table, `recover(answer(query(index))) ==
//! record[index]` through the real `Client::{setup,query,recover}` and
//! `Server::{setup,answer}` at toy single-prime params. The strongest invariant
//! in the suite — it exercises QueryComp → DMux → ModSwitch → FirstDim → CMux →
//! CRot → RespComp as one pipeline across many key seeds and selections.
//!
//! Run with `cargo +nightly fuzz run protocol_e2e_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{LweToRlweKeyN8, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8};
use via_primitives::sampling::{Distribution, Shake256Prg};
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{KeyDist, PIRParams};
use via_server::Server;

const N1: usize = 8;
const N2: usize = 4;
const D: usize = 2;
const L_QUERY: usize = 7;
const L_CK: usize = 7;
const L_RSK: usize = 8;

const Q1: u64 = 1 << 36;
const Q2: u64 = 1 << 28;
const Q3: u64 = 1 << 20;
const Q4: u64 = 1 << 12;
const P: u64 = 16;

const B_QUERY: u64 = 64;
const CK_BASE: u64 = 64;
const B_RSK: u64 = 8;

const NUM_ROWS: usize = 2;
const NUM_COLS: usize = 2;
const NUM_RECORDS: usize = D * NUM_ROWS * NUM_COLS; // 8

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;
type K = LweToRlweKeyN8<DynModulus, L_CK>;
type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;
type ToyServer = Server<K, N1, N2, R8, R8, R4, R4, R8, L_QUERY, L_CK, L_RSK, D>;

#[derive(Debug)]
struct Input {
    seed: Vec<u8>,
    index: usize,
    records: [[u64; N2]; NUM_RECORDS],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let l = u.int_in_range::<usize>(1..=32)?;
        let mut seed = vec![0u8; l];
        u.fill_buffer(&mut seed)?;
        let index = u.int_in_range::<usize>(0..=NUM_RECORDS - 1)?;
        let mut records = [[0u64; N2]; NUM_RECORDS];
        for rec in &mut records {
            for coeff in rec.iter_mut() {
                *coeff = u.int_in_range::<u64>(0..=P - 1)?;
            }
        }
        Ok(Input { seed, index, records })
    }
}

fn toy_params() -> PIRParams {
    PIRParams::new(
        N1, N2, Q1 as u128, Q2, Q3, Q4, P, //
        B_QUERY, L_QUERY, B_QUERY, L_QUERY, B_RSK, L_RSK, //
        KeyDist::Ternary, KeyDist::Ternary, 1, None, None, None, 40,
    )
}

fuzz_target!(|input: Input| {
    let q1 = DynModulus::new(Q1);
    let q2 = DynModulus::new(Q2);
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);
    let mut prg = Shake256Prg::new(&input.seed);

    let (client, pp) = ToyClient::setup(
        q1,
        q3,
        toy_params(),
        NUM_ROWS,
        NUM_COLS,
        CK_BASE,
        Distribution::Ternary,
        Distribution::Ternary,
        Distribution::Ternary,
        &mut prg,
        |sk, base, dist, prg| {
            Box::new(gen_lwe_to_rlwe_key_n8::<DynModulus, L_CK>(
                sk, base, dist, prg,
            ))
        },
        |sk1, sk2, dist, prg| {
            let q3_mod = RingPoly::modulus(sk2.poly());
            let s1_q3 = rekey_secret_key::<N1, R8, R8>(sk1, q3_mod);
            gen_rsk::<N1, N2, R8, R4, L_RSK, D>(&s1_q3, sk2, B_RSK, dist, prg)
        },
    );

    let records: Vec<R4> = input.records.iter().map(|c| R4::new(p, *c)).collect();
    let server = ToyServer::setup::<R4>(&records, pp, q1, q2, q3, q4, p);

    let query = client.query(input.index, &mut prg);
    let answer = server
        .answer::<R8, _>(&query, lwe_to_rlwe_n8::<DynModulus, L_CK>)
        .expect("server answer");
    let recovered: R4 = client.recover::<R4, R4, R4>(&answer, q3, q4, p);

    assert_eq!(recovered, records[input.index], "e2e diverged at index {}", input.index);
});
