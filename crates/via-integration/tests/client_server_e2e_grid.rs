//! Toy VIA-C client↔server e2e on a **larger asymmetric grid** `I=4, J=8`
//! (`#[ignore]`).
//!
//! [`client_server_e2e`](super) runs the minimal **square** `2×2` grid, where the
//! DMux and CMux trees collapse to a single level (`num_dmux = num_cmux = 1`).
//! This module re-runs the identical VIA-C round-trip — same public APIs, same
//! single-prime toy params — at `I=4, J=8` (`num_dmux = log₂4 = 2`,
//! `num_cmux = log₂8 = 3`), so it additionally exercises:
//!
//!   - **multi-level DMux (depth 2) + CMux (depth 3)** tree recursions, which are
//!     degenerate single nodes at `2×2`;
//!   - the **multi-bit client index → (row α, col β, record γ) decomposition** and
//!     the `query_decomp` partition into `(num_dmux, num_cmux, num_crot)` RGSW
//!     bits — single-bit per axis at `2×2`;
//!   - **`I ≠ J`**, so a row/column transposition (structurally invisible on a
//!     square grid) is caught;
//!   - **`first_dim` accumulation** over a non-trivial `4×8` matrix.
//!
//! It queries **every** index (all `d·I·J = 64` records — every one of the 32
//! cells, both γ slots). `#[ignore]` (run with
//! `cargo test -p via-integration -- --ignored`): the 64 full round-trips are
//! cheap at n8 but kept off the default path; the `2×2` `client_server_e2e`
//! remains the fast default-CI gate. Params are otherwise the single-prime toy
//! set from `client_server_e2e`.

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

const NUM_ROWS: usize = 4; // I — num_dmux = log₂ I = 2 (was 2 → 1 level)
const NUM_COLS: usize = 8; // J — num_cmux = log₂ J = 3 (was 2 → 1 level)

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;
type K = LweToRlweKeyN8<DynModulus, L_CK>;
type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;
type ToyServer = ViaCServer<K, N1, N2, R8, R8, R4, R4, L_QUERY, L_CK, L_RSK, D>;

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

/// A fully set-up client + server over the `4×8` toy DB, plus the recover-side
/// moduli. Built once and reused across all 64 queries (`query`/`answer`/`recover`
/// take `&self`), so the sweep continues a single PRG stream.
struct Harness {
    client: ToyClient,
    server: ToyServer,
    q3: DynModulus,
    q4: DynModulus,
    p: DynModulus,
    prg: Shake256Prg,
}

fn build() -> Harness {
    let q1 = DynModulus::new(Q1);
    let q2 = DynModulus::new(Q2);
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);
    let mut prg = Shake256Prg::new(b"via-c-client-server-e2e-grid");

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
    )
    .expect("client setup");

    let records: Vec<R4> = (0..D * NUM_ROWS * NUM_COLS).map(|m| record(m, p)).collect();
    let server = ToyServer::setup::<R8, R4>(&records, pp, q1, q2, q3, q4, p);

    Harness {
        client,
        server,
        q3,
        q4,
        p,
        prg,
    }
}

/// One full round-trip for `index`; returns `(recovered, expected)`.
fn run_query(h: &mut Harness, index: usize) -> (R4, R4) {
    let query = h.client.query(index, &mut h.prg).expect("client query");
    let answer = h
        .server
        .answer::<R8, _>(&query, lwe_to_rlwe_n8::<DynModulus, L_CK>)
        .expect("server answer");
    let recovered: R4 = h
        .client
        .recover::<R4, R4, R4>(&answer, h.q3, h.q4, h.p)
        .expect("client recover");
    (recovered, record(index, h.p))
}

/// Every index recovers its own record on the `4×8` grid — each of the 32 cells
/// `(α, β)` is hit by both of its `d = 2` records (`γ = 0, 1`). `index = γ·(I·J)
/// + α·J + β`, so the sweep walks every multi-level DMux/CMux selection path.
#[test]
#[ignore = "larger 4×8 grid (multi-level DMux/CMux); run with -- --ignored"]
fn client_server_e2e_grid_every_cell() {
    let mut h = build();
    for index in 0..(D * NUM_ROWS * NUM_COLS) {
        let (got, want) = run_query(&mut h, index);
        let (alpha, beta, gamma) = (
            (index / NUM_COLS) % NUM_ROWS,
            index % NUM_COLS,
            index / (NUM_ROWS * NUM_COLS),
        );
        assert_eq!(
            got, want,
            "recover(query({index})) must equal record[{index}] (cell α={alpha}, β={beta}, γ={gamma})"
        );
    }
}
