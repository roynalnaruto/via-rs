//! **Paper-scale VIA-B batch** client↔server e2e (`#[ignore]`).
//!
//! The production paper path end to end through the real public APIs:
//! `Client::{setup, batch_query, recover_batch}` + `Server::answer_batch`, at the
//! paper rings/moduli (Appendix B): `n1 = 2048`, `n2 = 512`, `q1 ≈ 2^75` (2-prime
//! RNS) ≫ `q2 ≈ 2^34` (single-prime) ≫ `q3 ≈ 2^23` ≫ `q4 = 2^12`, `p = 16`. The
//! repack runs at the single-prime `q2` (`repack_poly_2048_t8`); its q2 key is
//! derived from the RNS-`q1` cascade key by the **boxed cross-type** mod-switch
//! `repack_keys_poly_2048_t8_from_rns_cascade_boxed` (the §3.5 key reuse — the
//! client ships only the q1 key, the server derives q2).
//!
//! ## Batch size
//!
//! `T = 8` with `n3 = N2/T = 64` (the record-fit boundary `T·n3 = n2`). The
//! *production* batch is `T = 256` (`repack_poly_2048_t256`), but that runs
//! `answer_through_crot` 256× (≈3 h at paper scale); `T = 8` exercises the same
//! crypto — real degree, real RNS-q1→single-prime-q2 cross-type key, the depth-5
//! recursion, full deinterleave — in minutes. The `T = 256` repack itself is the
//! GO of `repack_poly_2048_t256_spike`.
//!
//! `#[ignore]` — heavy (the n2048 RNS cascade per query). Run with:
//! `cargo test -p via-integration --features via-b --release -- --ignored via_b_paper_batch`
#![cfg(feature = "via-b")]

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::rns::basis::paper::ViaCQ1Rns;
use via_primitives::algebra::zq::modulus::paper::{ViaCP, ViaCQ2, ViaCQ3, ViaCQ4};
use via_primitives::conversion::{
    LweToRlweKeyRnsN2048, gen_lwe_to_rlwe_key_rns_n2048_boxed, lwe_to_rlwe_rns_n2048_eval,
    repack_keys_poly_2048_t8_from_rns_cascade_boxed, repack_poly_2048_t8,
};
use via_primitives::encryption::types::RLWECiphertext;
use via_primitives::params::{ViaCPolyP, ViaCPolyQ1Rns, ViaCPolyQ2, ViaCPolyQ3, ViaCPolyQ4};
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{KeyDist, PIRParams};
use via_server::ViaBServer;

const N1: usize = 2048;
const N2: usize = 512;
const N3: usize = 64; // record degree = N2/T (record-fit boundary)
const T: usize = 8; // batch count
const D: usize = 4; // d = N1/N2 (ring-switch ratio)
const D3: usize = N1 / N3; // records per cell = 32
const L_QUERY: usize = 2;
const L_CK: usize = 18;
const L_RSK: usize = 8;
const CK_BASE: u64 = 18;
const NUM_ROWS: usize = 2; // I
const NUM_COLS: usize = 2; // J
const STACK_MB: usize = 16;

// Paper ring instantiation (Appendix B).
type R1 = ViaCPolyQ1Rns<N1>; // S1 @ q1-RNS, n1
type R2 = ViaCPolyQ2<N1>; // q2 @ n1 (single-prime) — the post-CRot / repack ring
type R3N1 = ViaCPolyQ3<N1>; // q3 @ n1 (mod_switch_sym intermediate + rekey target)
type R3N2 = ViaCPolyQ3<N2>; // q3 @ n2 (S2 ring + answer mask)
type R4N2 = ViaCPolyQ4<N2>; // q4 @ n2 (answer body)
type RpN1 = ViaCPolyP<N1>; // p @ n1 (DB embed target)
type Rec = ViaCPolyP<N3>; // p @ n3 (records)
type Rp512 = ViaCPolyP<N2>; // p @ n2 (recover's degree-n2 plaintext)
type K = LweToRlweKeyRnsN2048<ViaCQ1Rns, L_CK>;

type PaperBClient = Client<N1, N2, R1, R3N2, L_QUERY, L_CK, L_RSK, D>;
type PaperBServer = ViaBServer<K, N1, N2, N3, R1, R2, R3N2, R4N2, L_QUERY, L_CK, L_RSK, D>;

/// A distinct degree-n3 record per flat index.
fn record(m: usize) -> Rec {
    let m = m as u64;
    let coeffs: [u64; N3] = core::array::from_fn(|j| ((m + 1) * (j as u64 + 1)) % 16);
    Poly::new(ViaCP::default(), coeffs)
}

fn paper_params() -> PIRParams {
    PIRParams::new_b(
        N1,
        N2,
        137_438_822_401u128 * 274_810_798_081,
        17_175_674_881,
        8_380_417,
        4096,
        16,
        55879,
        L_QUERY, // gadget_base_1 (DMux @ q1)
        81,
        L_QUERY, // gadget_base_2 (CMux/CRot @ q2)
        8,
        L_RSK, // gadget_base_rsk
        KeyDist::Ternary,
        KeyDist::Ternary,
        26,
        None,
        None,
        None,
        128,
        N3,
        T,
    )
}

/// Full paper-scale VIA-B batch round-trip; returns `(recovered, expected)`.
fn batch_round_trip(idxs: &[usize; T]) -> (Vec<Rec>, Vec<Rec>) {
    let q1 = ViaCQ1Rns::default();
    let q2 = ViaCQ2::default();
    let q3 = ViaCQ3::default();
    let q4 = ViaCQ4::default();
    let p = ViaCP::default();
    let mut prg = Shake256Prg::new(b"via-b-paper-batch-e2e");

    // --- Client setup (RNS-q1 cascade; same shape as the VIA-C paper e2e) ----
    let (client, pp) = PaperBClient::setup(
        q1,
        q3,
        paper_params(),
        NUM_ROWS,
        NUM_COLS,
        CK_BASE,
        Distribution::Ternary,
        Distribution::Ternary,
        Distribution::Ternary,
        &mut prg,
        gen_lwe_to_rlwe_key_rns_n2048_boxed::<ViaCQ1Rns, L_CK>,
        |sk1, sk2, dist, prg| {
            let q3_mod = RingPoly::modulus(sk2.poly());
            let s1_q3 = rekey_secret_key::<N1, R1, R3N1>(sk1, q3_mod);
            gen_rsk::<N1, N2, R3N1, R3N2, L_RSK, D>(&s1_q3, sk2, 8, dist, prg)
        },
    )
    .expect("client setup");

    // --- Server setup: a DB of d3·I·J degree-n3 records (N_REC = N3) ---------
    let records: Vec<Rec> = (0..D3 * NUM_ROWS * NUM_COLS).map(record).collect();
    let server = PaperBServer::setup::<RpN1, Rec>(&records, pp, q1, q2, q3, q4, p);

    // --- Batch query → answer_batch → recover_batch --------------------------
    let batch = client
        .batch_query::<T, N3>(idxs, &mut prg)
        .expect("batch_query");
    let answer = server
        .answer_batch::<R3N1, T, _, _>(
            &batch,
            // Repack at single-prime q2: derive the q2 key (boxed, heap) from the
            // RNS-q1 cascade key by cross-type mod-switch (§3.5), then pack.
            |rotateds: &[RLWECiphertext<N1, R2>], k: &K| {
                let q2_key = repack_keys_poly_2048_t8_from_rns_cascade_boxed(k, q2);
                let arr: &[_; T] = rotateds.try_into().expect("T rotated ciphertexts");
                repack_poly_2048_t8(arr, &*q2_key, CK_BASE)
            },
            lwe_to_rlwe_rns_n2048_eval::<ViaCQ1Rns, L_CK>,
        )
        .expect("answer_batch");
    let recovered: Vec<Rec> = client
        .recover_batch::<R3N2, R4N2, Rp512, N3, T>(&answer, q3, q4, p)
        .expect("recover_batch");

    let expected: Vec<Rec> = idxs.iter().map(|&i| record(i)).collect();
    (recovered, expected)
}

/// The production paper batch round-trip: each of the T batched degree-n3 records
/// recovers, end to end at paper rings/moduli with the cross-type q2 key.
#[test]
#[ignore = "paper-scale n2048 RNS VIA-B batch (T=8) — heavy; run with --release -- --ignored"]
fn via_b_paper_batch_roundtrip() {
    std::thread::Builder::new()
        .stack_size(STACK_MB << 20)
        .spawn(|| {
            let idxs = [3usize, 17, 42, 5, 88, 100, 7, 120];
            let (got, want) = batch_round_trip(&idxs);
            assert_eq!(got.len(), T, "T recovered records");
            for t in 0..T {
                assert_eq!(
                    got[t], want[t],
                    "paper batch slot {t} must recover record[{}]",
                    idxs[t]
                );
            }
        })
        .expect("spawn paper batch thread")
        .join()
        .expect("paper batch e2e thread panicked");
}
