//! Toy VIA-B batch e2e on a **larger asymmetric grid** `I=4, J=8` (`#[ignore]`).
//!
//! [`batch_e2e_toy`](super) runs the batch path on the minimal **square** `2×2`
//! grid, where the DMux and CMux trees collapse to a single level
//! (`num_dmux = num_cmux = 1`). This module re-runs the identical VIA-B batch
//! round-trip — same `ViaBToyParams<64,16,2,8>` toy params, same `repack_n64_t8`
//! — at `I=4, J=8` (`num_dmux = log₂4 = 2`, `num_cmux = log₂8 = 3`). Combined
//! with `batch_e2e_toy`'s already-non-degenerate `T=8` repack, this is the most
//! complete toy exercise of the VIA-B server: per prefix it now drives
//! **multi-level DMux (depth 2) + CMux (depth 3)** plus `num_crot = 5`, the
//! multi-bit `(α, β, γ)` decomposition, an `I ≠ J` grid (catches a row/col
//! transposition), and `first_dim` over a `4×8` matrix — then the depth-5
//! `repack` + strided deinterleave on top.
//!
//! **One `T=8` batch only.** `answer_batch` runs `T` cascade-heavy
//! `answer_through_crot` prefixes; at `4×8` each prefix carries more
//! DMux/CMux/CRot bits than the `2×2` test, so a single batch keeps the runtime
//! bounded. The 8 indices are chosen to hit **8 distinct cells across all 4 rows
//! and all 8 columns** (`index = γ·(I·J) + α·J + β`), with varied within-cell
//! slots and distinct records (distinct mod p), so one batch maximally probes the
//! asymmetric grid.
//!
//! `#[ignore]` — run with
//! `cargo test -p via-integration --features via-b -- --ignored via_b_grid`. The
//! `2×2` `batch_e2e_toy` remains the default-CI gate. Modulus flow (q1 > q2, the
//! cascade-key reuse mod-switched into the repack) is identical to `batch_e2e_toy`.
#![cfg(feature = "via-b")]

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{LweToRlweKeyN64, gen_lwe_to_rlwe_key_n64};
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{KeyDist, PIRParams};
use via_server::{ServerConfig, ViaBServer};

const N1: usize = 64;
const N2: usize = 16;
const N3: usize = 2; // record degree (VIA-B)
const T: usize = 8; // batch count (T·N3 = 16 = N2)
const D: usize = 4; // d = N1 / N2 (ring-switch ratio)
const D3: usize = N1 / N3; // records per cell = 32
const L_QUERY: usize = 7;
// n64 cascade gadget: base 8, 12 digits — see batch_e2e_toy.rs.
const L_CK: usize = 12;
const L_RSK: usize = 8;

const Q1: u64 = 1 << 36;
const Q2: u64 = 1 << 28;
const Q3: u64 = 1 << 20;
const Q4: u64 = 1 << 12;
const P: u64 = 16;

const B_QUERY: u64 = 64;
const CK_BASE: u64 = 8; // n64 cascade base (pairs with L_CK = 12)
const B_RSK: u64 = 8;

const NUM_ROWS: usize = 4; // I — num_dmux = log₂ I = 2 (was 2 → 1 level)
const NUM_COLS: usize = 8; // J — num_cmux = log₂ J = 3 (was 2 → 1 level)

type R64 = Poly<N1, DynModulus, Coefficient>;
type R16 = Poly<N2, DynModulus, Coefficient>;
type R2 = Poly<N3, DynModulus, Coefficient>; // the record ring at n3
type K = LweToRlweKeyN64<DynModulus, L_CK>;
type ToyClient = Client<N1, N2, R64, R16, L_QUERY, L_CK, L_RSK, D>;
type ToyBServer = ViaBServer<K, N1, N2, N3, R64, R64, R16, R16, L_QUERY, L_CK, L_RSK, D>;

/// A distinct degree-n3 record per flat index.
fn record(m: usize, p: DynModulus) -> R2 {
    let m = m as u64;
    Poly::new(p, [(m + 1) % P, (2 * m + 3) % P])
}

struct Harness {
    client: ToyClient,
    server: ToyBServer,
    q3: DynModulus,
    q4: DynModulus,
    p: DynModulus,
    prg: Shake256Prg,
}

fn toy_params() -> PIRParams {
    PIRParams::new_b(
        N1,
        N2,
        Q1 as u128,
        Q2, // metadata only; runtime q2_mod = q1 (see batch_e2e_toy.rs)
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

/// Set up the client + server over a DB of `d3·I·J = 1024` degree-n3 records.
fn build() -> Harness {
    let q1 = DynModulus::new(Q1);
    let q2 = DynModulus::new(Q2);
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);
    let mut prg = Shake256Prg::new(b"via-b-batch-e2e-grid");

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
            Box::new(gen_lwe_to_rlwe_key_n64::<DynModulus, L_CK>(
                sk, base, dist, prg,
            ))
        },
        |sk1, sk2, dist, prg| {
            let q3_mod = RingPoly::modulus(sk2.poly());
            let s1_q3 = rekey_secret_key::<N1, R64, R64>(sk1, q3_mod);
            gen_rsk::<N1, N2, R64, R16, L_RSK, D>(&s1_q3, sk2, B_RSK, dist, prg)
        },
    )
    .expect("client setup");

    let records: Vec<R2> = (0..D3 * NUM_ROWS * NUM_COLS)
        .map(|m| record(m, p))
        .collect();
    let server = ToyBServer::setup::<R64, R2>(ServerConfig::new(pp, q1, q2, q3, q4), &records, p)
        .expect("server setup");

    Harness {
        client,
        server,
        q3,
        q4,
        p,
        prg,
    }
}

/// One batch round-trip for `idxs`; returns `(recovered, expected)` records.
fn run_batch(h: &mut Harness, idxs: &[usize; T]) -> (Vec<R2>, Vec<R2>) {
    let batch = h
        .client
        .batch_query::<T, N3>(idxs, &mut h.prg)
        .expect("batch_query");
    // cascade + repack are the server backend's behaviour now.
    let answer = h.server.answer_batch::<T>(&batch).expect("answer_batch");
    let recovered: Vec<R2> = h
        .client
        .recover_batch::<R16, R16, R16, N3, T>(&answer, h.q3, h.q4, h.p)
        .expect("recover_batch");

    let expected: Vec<R2> = idxs.iter().map(|&i| record(i, h.p)).collect();
    (recovered, expected)
}

/// A single `T=8` batch whose indices hit 8 distinct cells `(α, β)` spanning all
/// 4 rows and all 8 columns (`index = γ·32 + α·8 + β`), with varied within-cell
/// slots `γ` and distinct records — each recovers in batch order through the
/// multi-level (DMux d2 / CMux d3) selection + depth-5 repack + strided
/// deinterleave.
#[test]
#[ignore = "larger 4×8 grid (multi-level DMux/CMux); run with --features via-b -- --ignored"]
fn via_b_grid_batch_roundtrip() {
    let mut h = build();
    // cells (α,β): (0,0)(1,1)(2,2)(3,3)(0,4)(1,5)(2,6)(3,7); γ: 0,2,…,14; distinct mod p.
    let idxs = [0usize, 73, 146, 219, 260, 333, 406, 479];
    let (got, want) = run_batch(&mut h, &idxs);
    assert_eq!(got.len(), T, "T recovered records");
    for t in 0..T {
        assert_eq!(
            got[t], want[t],
            "batch slot {t} must recover record[{}]",
            idxs[t]
        );
    }
}
