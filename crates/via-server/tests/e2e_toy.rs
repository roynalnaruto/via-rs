//! Toy single-prime **end-to-end** gate for the VIA-C answer pipeline (P3
//! Task 31): `query → answer → recover == record`.
//!
//! This is the primary integration gate for `via-server`. It builds the query
//! keys and issues a query **directly via `via-primitives`** — it does NOT
//! depend on `via-client` (the `client ⊥ server` invariant). It mirrors the
//! reference `test_e2e.py` structure as a pure-Rust integration test.
//!
//! ## Query construction (the client's job, inlined here)
//!
//! Each query bit `b` becomes `L_QUERY` raw LWEs `encrypt_lwe_raw(s1, b·g[i])`,
//! with `g = (⌈q1/B⌉, …, ⌈q1/Bᴸ⌉)` the VIA gadget vector at the query base `B`.
//! The cascade preserves the value, so these assemble into `RLev_{S1}(b)` →
//! `RGSW_{S1}(b)` inside `query_decomp`. Bit order matches the pipeline:
//! DMux (MSB-first, `log₂I`) then CMux (LSB-first, `log₂J`) then CRot
//! (LSB-first, `log₂d`). For the toy `I=J=d=2`, each group is a single bit, so
//! the target `(i, j, k)` maps to `record[k·I·J + i·J + j]` after the slot-`k`
//! rotation and the ring-switch π₀ projection.

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{
    LweToRlweKeyN8, encrypt_lwe_raw, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8,
};
use via_primitives::encryption::gadget_vector_values;
use via_primitives::encryption::types::SecretKey;
use via_primitives::gates::gen_rlwe_to_rgsw_key;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{CompressedQuery, KeyDist, PIRParams, PublicParams, QueryCompressionKey};
use via_server::{ViaCServer, answer_one_query, setup_db};

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

const B_QUERY: u64 = 64; // DMux/CMux gadget base (b1 = b2)
const CK_BASE: u64 = 64; // cascade + conversion-key gadget base
const B_RSK: u64 = 8; // ring-switch gadget base

const NUM_ROWS: usize = 2; // I
const NUM_COLS: usize = 2; // J

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;
type Cascade = LweToRlweKeyN8<DynModulus, L_CK>;

/// A fixed, distinct record per flat index (so selection is actually tested).
fn record(m: usize, p: DynModulus) -> R4 {
    let m = m as u64;
    Poly::new(
        p,
        [(m + 1) % P, (2 * m + 1) % P, (3 * m + 1) % P, (m + 5) % P],
    )
}

/// Run the full pipeline selecting `(i, j, k)` and return `(recovered, expected)`.
///
/// `via_server == true` routes through the [`Server`] struct (`setup` +
/// `answer`); `false` calls [`setup_db`] + [`answer_one_query`] directly. Both
/// paths must agree.
fn run_e2e(target_i: usize, target_j: usize, target_k: usize, via_server: bool) -> (R4, R4) {
    let q1 = DynModulus::new(Q1);
    let q2 = DynModulus::new(Q2);
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);
    let mut prg = Shake256Prg::new(b"via-c-e2e-toy");

    // --- Keys ------------------------------------------------------------
    let s1 = SecretKey::<N1, R8>::keygen(q1, Distribution::Ternary, &mut prg);
    let s2 = SecretKey::<N2, R4>::keygen(q3, Distribution::Ternary, &mut prg);
    let cascade_key =
        gen_lwe_to_rlwe_key_n8::<_, L_CK>(&s1, CK_BASE, Distribution::Ternary, &mut prg);
    let conv_key =
        gen_rlwe_to_rgsw_key::<N1, R8, L_CK>(&s1, CK_BASE, Distribution::Ternary, &mut prg);
    let qck = QueryCompressionKey::new(Box::new(cascade_key), Box::new(conv_key));
    let s1_q3 = rekey_secret_key::<N1, R8, R8>(&s1, q3);
    let rsk =
        gen_rsk::<N1, N2, R8, R4, L_RSK, D>(&s1_q3, &s2, B_RSK, Distribution::Ternary, &mut prg);

    let params = PIRParams::new(
        N1,
        N2,
        Q1 as u128,
        Q2,
        Q3,
        Q4,
        P, //
        B_QUERY,
        L_QUERY, // gadget 1 (DMux @ q1)
        B_QUERY,
        L_QUERY, // gadget 2 (CMux/CRot @ q2)
        B_RSK,
        L_RSK, // gadget rsk
        KeyDist::Ternary,
        KeyDist::Ternary,
        1,
        None,
        None,
        None,
        40,
    );
    let pp = PublicParams::new(
        qck,
        Box::new(rsk),
        params,
        NUM_ROWS,
        NUM_COLS,
        CK_BASE,
        L_CK,
    );

    // --- Database: d·I·J = 8 distinct records ----------------------------
    let records: Vec<R4> = (0..D * NUM_ROWS * NUM_COLS).map(|m| record(m, p)).collect();
    let db = setup_db::<N1, N2, R8, R4>(&records, NUM_ROWS, NUM_COLS, p);

    // --- Query: target (i, j, k) as gadget-scaled LWE bits ---------------
    let g = gadget_vector_values::<N1, R8, L_QUERY>(q1, B_QUERY);
    let mut lwes = Vec::new();
    for &bit in &[target_i, target_j, target_k] {
        for &gi in g.iter() {
            let val = (bit as u128) * gi;
            lwes.push(encrypt_lwe_raw(&s1, val, Distribution::Ternary, &mut prg));
        }
    }
    let query = CompressedQuery::new(lwes);

    // --- Answer + recover ------------------------------------------------
    let answer = if via_server {
        let server =
            ViaCServer::<Cascade, N1, N2, R8, R8, R4, R4, R8, L_QUERY, L_CK, L_RSK, D>::setup::<R4>(
                &records, pp, q1, q2, q3, q4, p,
            );
        server
            .answer::<R8, _>(&query, lwe_to_rlwe_n8::<DynModulus, L_CK>)
            .expect("server answer")
    } else {
        answer_one_query::<N1, N2, R8, R8, R8, R4, R4, R8, Cascade, L_QUERY, L_CK, L_RSK, D, _>(
            &query,
            &pp,
            &db,
            q1,
            q2,
            q3,
            q4,
            lwe_to_rlwe_n8::<DynModulus, L_CK>,
        )
        .expect("answer")
    };
    let recovered: R4 = s2.decrypt_asymmetric(&answer, q3, q4, p);

    let flat = target_k * NUM_ROWS * NUM_COLS + target_i * NUM_COLS + target_j;
    (recovered, records[flat])
}

/// Index 0 (all query bits zero) — isolates noise closure from bit-selection.
#[test]
fn e2e_toy_recovers_index_000() {
    let (got, want) = run_e2e(0, 0, 0, false);
    assert_eq!(got, want, "select (0,0,0) → record[0]");
}

/// DMux bit set (`i=1`) — selects row 1 → `record[2]`.
#[test]
fn e2e_toy_recovers_index_100() {
    let (got, want) = run_e2e(1, 0, 0, false);
    assert_eq!(got, want, "select (i=1,j=0,k=0) → record[2]");
}

/// CMux bit set (`j=1`) — selects column 1 → `record[1]`.
#[test]
fn e2e_toy_recovers_index_010() {
    let (got, want) = run_e2e(0, 1, 0, false);
    assert_eq!(got, want, "select (i=0,j=1,k=0) → record[1]");
}

/// CRot bit set (`k=1`) — extracts slot 1 → `record[4]`.
#[test]
fn e2e_toy_recovers_index_001() {
    let (got, want) = run_e2e(0, 0, 1, false);
    assert_eq!(got, want, "select (i=0,j=0,k=1) → record[4]");
}

/// All bits set (`i=j=k=1`) — `record[7]`; exercises every gate together.
#[test]
fn e2e_toy_recovers_index_111() {
    let (got, want) = run_e2e(1, 1, 1, false);
    assert_eq!(got, want, "select (1,1,1) → record[7]");
}

/// Same pipeline routed through the [`Server`] struct (`setup` + `answer`):
/// select `(i=0,j=1,k=1)` → `record[5]`. Locks that `Server` delegates to
/// `answer_one_query` identically and stays decrypt-correct.
#[test]
fn e2e_toy_via_server_struct_recovers_record() {
    let (got, want) = run_e2e(0, 1, 1, true);
    assert_eq!(got, want, "Server::answer select (0,1,1) → record[5]");
}
