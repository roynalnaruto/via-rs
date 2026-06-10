//! Toy single-prime **client ↔ server** end-to-end test (P4 capstone).
//!
//! Exercises the whole VIA-C protocol through the real public APIs of both
//! crates — `Client::{setup, query, recover}` and `Server::{setup, answer}` —
//! with nothing inlined:
//!
//! ```text
//! Client::setup ─(PublicParams)→ Server::setup
//! Client::query ─(CompressedQuery)→ Server::answer ─(answer)→ Client::recover
//! ```
//!
//! and asserts `recover == record[index]`. This is the strongest proof that the
//! P4 client and the P3 server agree on every shared convention (bit ordering,
//! gadget base, PRG-fed key layout, the q1≫q2≫q3≫q4 modulus chain). Params are
//! the same single-prime set proven to close noise in `via-server`'s `e2e_toy`.

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{LweToRlweKeyN8, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8};
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{KeyDist, PIRParams};
use via_server::ViaCServer;

const N1: usize = 8;
const N2: usize = 4;
const D: usize = 2; // d = N1 / N2
const L_QUERY: usize = 7;
const L_CK: usize = 7;
const L_RSK: usize = 8;

const Q1: u64 = 1 << 36;
const Q2: u64 = 1 << 28;
const Q3: u64 = 1 << 20;
const Q4: u64 = 1 << 12;
const P: u64 = 16;

const B_QUERY: u64 = 64; // DMux/CMux gadget base (= gadget_base_1/2)
const CK_BASE: u64 = 64; // cascade + conversion-key base
const B_RSK: u64 = 8; // ring-switch base

const NUM_ROWS: usize = 2; // I
const NUM_COLS: usize = 2; // J

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;
type K = LweToRlweKeyN8<DynModulus, L_CK>;
type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;
type ToyServer = ViaCServer<K, N1, N2, R8, R8, R4, R4, R8, L_QUERY, L_CK, L_RSK, D>;

/// A distinct record per flat index (so selection is genuinely tested).
fn record(m: usize, p: DynModulus) -> R4 {
    let m = m as u64;
    Poly::new(
        p,
        [(m + 1) % P, (2 * m + 1) % P, (3 * m + 1) % P, (m + 5) % P],
    )
}

fn toy_params() -> PIRParams {
    PIRParams::new(
        N1,
        N2,
        Q1 as u128,
        Q2,
        Q3,
        Q4,
        P, //
        B_QUERY,
        L_QUERY,
        B_QUERY,
        L_QUERY,
        B_RSK,
        L_RSK, //
        KeyDist::Ternary,
        KeyDist::Ternary,
        1,
        None,
        None,
        None,
        40,
    )
}

/// Full protocol round-trip for `index`; returns `(recovered, expected)`.
fn round_trip(index: usize) -> (R4, R4) {
    let q1 = DynModulus::new(Q1);
    let q2 = DynModulus::new(Q2);
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);
    let mut prg = Shake256Prg::new(b"via-c-client-server-e2e");

    // --- Client setup → (Client, PublicParams) ---------------------------
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
            // rekey S1 → q3 (S2's modulus) before gen_rsk — concrete types so
            // the private RekeySource bound resolves.
            let q3_mod = RingPoly::modulus(sk2.poly());
            let s1_q3 = rekey_secret_key::<N1, R8, R8>(sk1, q3_mod);
            gen_rsk::<N1, N2, R8, R4, L_RSK, D>(&s1_q3, sk2, B_RSK, dist, prg)
        },
    )
    .expect("client setup");

    // --- Server setup (consumes the client's PublicParams) ---------------
    let records: Vec<R4> = (0..D * NUM_ROWS * NUM_COLS).map(|m| record(m, p)).collect();
    let server = ToyServer::setup::<R4>(&records, pp, q1, q2, q3, q4, p);

    // --- Query → Answer → Recover ----------------------------------------
    let query = client.query(index, &mut prg).expect("client query");
    let answer = server
        .answer::<R8, _>(&query, lwe_to_rlwe_n8::<DynModulus, L_CK>)
        .expect("server answer");
    let recovered: R4 = client
        .recover::<R4, R4, R4>(&answer, q3, q4, p)
        .expect("client recover");

    (recovered, records[index])
}

/// Index 0 — all query bits zero (noise closure in isolation).
#[test]
fn client_server_e2e_index_0() {
    let (got, want) = round_trip(0);
    assert_eq!(got, want, "recover(query(0)) must equal record[0]");
}

/// Index 5 = γ·4 + α·2 + β with (α,β,γ) = (0,1,1) — DMux off, CMux + CRot on.
#[test]
fn client_server_e2e_index_5() {
    let (got, want) = round_trip(5);
    assert_eq!(got, want, "recover(query(5)) must equal record[5]");
}

/// Index 7 = (α,β,γ) = (1,1,1) — every gate selecting.
#[test]
fn client_server_e2e_index_7() {
    let (got, want) = round_trip(7);
    assert_eq!(got, want, "recover(query(7)) must equal record[7]");
}

/// Every index in range recovers its own record (full coverage).
#[test]
fn client_server_e2e_all_indices() {
    for index in 0..(D * NUM_ROWS * NUM_COLS) {
        let (got, want) = round_trip(index);
        assert_eq!(
            got, want,
            "recover(query({index})) must equal record[{index}]"
        );
    }
}
