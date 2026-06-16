//! Fuzz: §7 VIA-B client↔server batch round-trip.
//!
//! For any index batch and record table, `recover_batch(answer_batch(batch_query(
//! idxs)))[t] == record[idxs[t]]` through the real `Client::{setup, batch_query,
//! recover_batch}` + `Server::answer_batch` at toy params (n1=8, n2=4, n3=2, T=2;
//! `repack_n8_t2`). Exercises the whole VIA-B pipeline — including the q1→q2
//! cascade-key mod-switch + the depth-2 repack + the strided de-interleave — under
//! fuzzed selections and many key seeds.
//!
//! Run with `cargo +nightly fuzz run --features via-b protocol_answer_batch_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{
    LweToRlweKeyN8, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8_eval,
    repack_keys_n8_t2_from_cascade_modswitched, repack_n8_t2,
};
use via_primitives::encryption::types::RLWECiphertext;
use via_primitives::sampling::{Distribution, Shake256Prg};
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{KeyDist, PIRParams};
use via_server::ViaBServer;

const N1: usize = 8;
const N2: usize = 4;
const N3: usize = 2;
const T: usize = 2;
const D: usize = 2;
const D3: usize = N1 / N3; // 4
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
const NUM_RECORDS: usize = D3 * NUM_ROWS * NUM_COLS; // 16

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;
type R2 = Poly<N3, DynModulus, Coefficient>;
type K = LweToRlweKeyN8<DynModulus, L_CK>;
type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;
type ToyBServer = ViaBServer<K, N1, N2, N3, R8, R8, R4, R4, L_QUERY, L_CK, L_RSK, D>;

#[derive(Debug)]
struct Input {
    seed: Vec<u8>,
    idxs: [usize; T],
    records: [[u64; N3]; NUM_RECORDS],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let l = u.int_in_range::<usize>(1..=32)?;
        let mut seed = vec![0u8; l];
        u.fill_buffer(&mut seed)?;
        let mut idxs = [0usize; T];
        for i in &mut idxs {
            *i = u.int_in_range::<usize>(0..=NUM_RECORDS - 1)?;
        }
        let mut records = [[0u64; N3]; NUM_RECORDS];
        for rec in &mut records {
            for c in rec.iter_mut() {
                *c = u.int_in_range::<u64>(0..=P - 1)?;
            }
        }
        Ok(Input {
            seed,
            idxs,
            records,
        })
    }
}

fn params() -> PIRParams {
    PIRParams::new_b(
        N1,
        N2,
        Q1 as u128,
        Q2,
        Q3,
        Q4,
        P,
        B_QUERY,
        L_QUERY,
        B_QUERY,
        L_QUERY,
        B_RSK,
        L_RSK,
        KeyDist::Ternary,
        KeyDist::Ternary,
        1,
        None,
        None,
        None,
        40,
        N3,
        T,
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
        params(),
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
    )
    .expect("client setup");

    let records: Vec<R2> = input.records.iter().map(|c| R2::new(p, *c)).collect();
    let server = ToyBServer::setup::<R8, R2>(&records, pp, q1, q2, q3, q4, p);

    let batch = client
        .batch_query::<T, N3>(&input.idxs, &mut prg)
        .expect("batch_query");
    let answer = server
        .answer_batch::<R8, T, _, _>(
            &batch,
            |rotateds: &[RLWECiphertext<N1, R8>], k: &K| {
                let keys_q2 = repack_keys_n8_t2_from_cascade_modswitched(k, q2);
                let arr: &[_; T] = rotateds.try_into().expect("T rotated ciphertexts");
                repack_n8_t2(arr, &keys_q2, CK_BASE)
            },
            lwe_to_rlwe_n8_eval::<DynModulus, L_CK>,
        )
        .expect("answer_batch");
    let recovered: Vec<R2> = client
        .recover_batch::<R4, R4, R4, N3, T>(&answer, q3, q4, p)
        .expect("recover_batch");

    for t in 0..T {
        assert_eq!(
            recovered[t], records[input.idxs[t]],
            "batch diverged at slot {t} (idx {})",
            input.idxs[t]
        );
    }
});
