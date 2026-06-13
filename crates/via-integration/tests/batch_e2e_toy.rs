//! Toy single-prime **VIA-B batch** client↔server e2e capstone.
//!
//! Validates the whole VIA-B batch protocol through the real public APIs —
//! `Client::{setup, batch_query, recover_batch}` and `Server::answer_batch` — at
//! the canonical toy preset `ViaBToyParams<64,16,2,8>` (n1=64, n2=16, n3=2, T=8;
//! `repack_n64_t8`, depth 5), with nothing inlined. Asserts every one of the T
//! batched records recovers.
//!
//! T=8 is deliberately **non-degenerate**: it drives a real 3-bit bit-reversal in
//! the repack tree, an 8-wide strided deinterleave (`record_t =
//! recovered.project_at::<N3>(t)`), `num_crot = log2(N1/N3) = 5` rotation bits, a
//! `d = N1/N2 = 4` ring-switch fold, and a depth-5 repack — the index/stride/noise
//! mechanics that the minimal n8/T2 tuple leaves as the identity (n8/T2 stays
//! covered by the `repack_n8_t2` unit tests and the paper batch e2e).
//!
//! ## Modulus flow (q1 > q2) — the cascade-key reuse across moduli
//!
//! The repack reuses the **q1** cascade keys but operates on the post-CRot
//! ciphertexts, which VIA-C places at **q2**; a `key_switch`
//! across q1 ≠ q2 is a modulus mismatch. So the server **mod-switches the q1
//! cascade-key suffix → q2** internally
//! ([`repack_keys_n64_t8_from_cascade_modswitched`]) before repacking — the client
//! still ships only the q1 keys, so the no-new-offline-payload property holds. (Toy is
//! single-prime; the paper's RNS-q1 → single-prime-q2 *cross-type* mod-switch is
//! the remaining extension.)
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
// n64 cascade gadget: base 8, 12 digits (8^12 ≈ 2^36 ≥ q1). The 6-step n64
// cascade needs the finer base-8 decomposition that `cascade_full_roundtrip_n64`
// pins — the n8 toy's coarse base-64/L=7 is too noisy at this depth.
const L_CK: usize = 12;
const L_RSK: usize = 8;

// The real descending modulus chain q1 > q2 > q3 > q4 (params + runtime). The
// repack reuses the q1 cascade keys mod-switched to q2 (see file doc).
const Q1: u64 = 1 << 36;
const Q2: u64 = 1 << 28;
const Q3: u64 = 1 << 20;
const Q4: u64 = 1 << 12;
const P: u64 = 16;

const B_QUERY: u64 = 64;
const CK_BASE: u64 = 8; // n64 cascade base (pairs with L_CK = 12)
const B_RSK: u64 = 8;

const NUM_ROWS: usize = 2;
const NUM_COLS: usize = 2;

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

/// A fully set-up client + server over the toy DB, plus the recover-side moduli.
/// Built once per test (the n64 cascade keygen is the dominant cost) and reused
/// across every batch — `batch_query`/`answer_batch`/`recover_batch` all take
/// `&self`, so distinct batches just continue the one PRG stream.
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
        Q2, // metadata only; runtime q2_mod = q1 (see const docs)
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

/// Set up the client + server over a DB of d3·I·J degree-n3 records (N_REC = N3).
fn build() -> Harness {
    let q1 = DynModulus::new(Q1);
    let q2 = DynModulus::new(Q2); // the real q2 < q1; cascade keys mod-switched q1→q2 for repack
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);
    let mut prg = Shake256Prg::new(b"via-b-batch-e2e-toy");

    // --- Client setup (identical to VIA-C; batch is client-side methods) --
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
    // cascade + repack (cascade-key reuse across q1 ≠ q2) are the server backend's
    // behaviour now; `answer_batch::<T>` wires them.
    let answer = h.server.answer_batch::<T>(&batch).expect("answer_batch");
    let recovered: Vec<R2> = h
        .client
        .recover_batch::<R16, R16, R16, N3, T>(&answer, h.q3, h.q4, h.p)
        .expect("recover_batch");

    let expected: Vec<R2> = idxs.iter().map(|&i| record(i, h.p)).collect();
    (recovered, expected)
}

/// Full VIA-B batch round-trip: each of the T=8 batched indices recovers its own
/// degree-n3 record, in batch order (deinterleave = `project_at::<N3>(t)`). The
/// indices span all four cells (I·J) and are distinct mod p so the records are
/// distinct (a misrouted slot cannot pass by value-collision).
#[test]
fn via_b_toy_batch_roundtrip() {
    let mut h = build();
    let idxs = [3usize, 11, 20, 38, 72, 89, 106, 127];
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

/// Coverage: T=8 batches spanning distinct cells (I·J) and within-cell record
/// slots (d3 = N1/N3 = 32), all recovering correctly — guards both the selection
/// (CRot at the finer n3 granularity, 5 rotation bits) and the 8-wide strided
/// de-interleave order. The two batches together cover all 16 residues mod p.
#[test]
fn via_b_toy_batch_covers_cells_and_slots() {
    let mut h = build();
    for idxs in [
        [0usize, 17, 34, 51, 68, 85, 102, 119],
        [8, 25, 42, 59, 76, 93, 110, 127],
    ] {
        let (got, want) = run_batch(&mut h, &idxs);
        for t in 0..T {
            assert_eq!(got[t], want[t], "batch {idxs:?} slot {t}");
        }
    }
}
