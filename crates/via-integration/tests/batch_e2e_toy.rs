//! Toy single-prime **VIA-B batch** client↔server e2e (P5 capstone).
//!
//! Validates the whole VIA-B batch protocol through the real public APIs —
//! `Client::{setup, batch_query, recover_batch}` and `Server::answer_batch` — at
//! a minimal toy tuple (n1=8, n2=4, n3=2, T=2; `repack_n8_t2`, depth 2), with
//! nothing inlined. Asserts every one of the T batched records recovers. This is
//! the first end-to-end exercise of the CRot → Extr → Repack → RespComp →
//! deinterleave path, and the empirical gate for P4's strided-deinterleave
//! derivation (`record_t = recovered.project_at::<N3>(t)`).
//!
//! ## Modulus flow (q1 > q2) — the §3.5 key reuse across moduli
//!
//! The repack reuses the **q1** cascade keys (§3.5) but operates on the post-CRot
//! ciphertexts, which VIA-C places at **q2** (`.docs/via-b.md` §4); a `key_switch`
//! across q1 ≠ q2 is a modulus mismatch. So the server **mod-switches the q1
//! cascade-key suffix → q2** internally
//! ([`repack_keys_n8_t2_from_cascade_modswitched`]) before repacking — the client
//! still ships only the q1 keys, so §3.5's no-new-offline-payload holds. (Toy is
//! single-prime; the paper's RNS-q1 → single-prime-q2 *cross-type* mod-switch is
//! the remaining extension.)
#![cfg(feature = "via-b")]

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{
    LweToRlweKeyN8, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8,
    repack_keys_n8_t2_from_cascade_modswitched, repack_n8_t2,
};
use via_primitives::encryption::types::RLWECiphertext;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{KeyDist, PIRParams};
use via_server::ViaBServer;

const N1: usize = 8;
const N2: usize = 4;
const N3: usize = 2; // record degree (VIA-B)
const T: usize = 2; // batch count (T·N3 = 4 = N2)
const D: usize = 2; // d = N1 / N2 (ring-switch ratio)
const D3: usize = N1 / N3; // records per cell = 4
const L_QUERY: usize = 7;
const L_CK: usize = 7;
const L_RSK: usize = 8;

// The real descending modulus chain q1 > q2 > q3 > q4 (params + runtime). The
// repack reuses the q1 cascade keys mod-switched to q2 (see file doc).
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

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;
type R2 = Poly<N3, DynModulus, Coefficient>; // the record ring at n3
type K = LweToRlweKeyN8<DynModulus, L_CK>;
type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;
type ToyBServer = ViaBServer<K, N1, N2, N3, R8, R8, R4, R4, R8, L_QUERY, L_CK, L_RSK, D>;

/// A distinct degree-n3 record per flat index.
fn record(m: usize, p: DynModulus) -> R2 {
    let m = m as u64;
    Poly::new(p, [(m + 1) % P, (2 * m + 3) % P])
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

/// Batch round-trip for `idxs`; returns `(recovered, expected)` records.
fn batch_round_trip(idxs: &[usize; T]) -> (Vec<R2>, Vec<R2>) {
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

    // --- Server setup: a DB of d3·I·J degree-n3 records (N_REC = N3) ------
    let records: Vec<R2> = (0..D3 * NUM_ROWS * NUM_COLS)
        .map(|m| record(m, p))
        .collect();
    let server = ToyBServer::setup::<R2>(&records, pp, q1, q2, q3, q4, p);

    // --- Batch query → answer_batch → recover_batch -----------------------
    let batch = client
        .batch_query::<T, N3>(idxs, &mut prg)
        .expect("batch_query");
    let answer = server
        .answer_batch::<R8, T, _, _>(
            &batch,
            // Repack closure (§3.5 across q1 ≠ q2): mod-switch the q1 cascade key
            // suffix → q2, then repack the q2 post-CRot ciphertexts at base CK_BASE.
            |rotateds: &[RLWECiphertext<N1, R8>], k: &K| {
                let arr: &[_; T] = rotateds.try_into().expect("T rotated ciphertexts");
                let keys_q2 = repack_keys_n8_t2_from_cascade_modswitched(k, q2);
                repack_n8_t2(arr, &keys_q2, CK_BASE)
            },
            lwe_to_rlwe_n8::<DynModulus, L_CK>,
        )
        .expect("answer_batch");
    let recovered: Vec<R2> = client
        .recover_batch::<R4, R4, R4, N3, T>(&answer, q3, q4, p)
        .expect("recover_batch");

    let expected: Vec<R2> = idxs.iter().map(|&i| record(i, p)).collect();
    (recovered, expected)
}

/// Full VIA-B batch round-trip: each of the T batched indices recovers its own
/// degree-n3 record, in batch order (deinterleave = `project_at::<N3>(t)`).
#[test]
fn via_b_toy_batch_roundtrip() {
    let idxs = [3usize, 11usize];
    let (got, want) = batch_round_trip(&idxs);
    assert_eq!(got.len(), T, "T recovered records");
    for t in 0..T {
        assert_eq!(
            got[t], want[t],
            "batch slot {t} must recover record[{}]",
            idxs[t]
        );
    }
}

/// Coverage: several batches spanning distinct cells (I·J) and within-cell record
/// slots (d3 = N1/N3 = 4), all recovering correctly — guards both the selection
/// (CRot at the finer n3 granularity) and the de-interleave order.
#[test]
fn via_b_toy_batch_covers_cells_and_slots() {
    for idxs in [[0usize, 15], [7, 8], [1, 14], [5, 9]] {
        let (got, want) = batch_round_trip(&idxs);
        for t in 0..T {
            assert_eq!(got[t], want[t], "batch {idxs:?} slot {t}");
        }
    }
}
