//! **Paper-scale** in-memory client ↔ server e2e (P4, double-confirmation).
//!
//! Same protocol round-trip as `client_server_e2e`, but at the paper VIA-C
//! parameters (Appendix B): `n1 = 2048`, `n2 = 512`, `d = 4`, `q1 ≈ 2^75`
//! (two-prime RNS) `≫ q2 ≈ 2^34 ≫ q3 ≈ 2^23 ≫ q4 = 2^12`, `p = 16`, with the
//! Table-6 gadget bases (DMux 55879, CMux/CRot 81, conversion-key 18,
//! ring-switch 8). This is *not* a wire test — objects are passed in-memory;
//! it exists only to confirm the protocol and the noise budget close at
//! paper-scale ring dimensions and moduli.
//!
//! `#[ignore]` — heavy: the n2048 RNS cascade key is ~24.75 MB and the
//! schoolbook O(n²) pipeline at n=2048 runs for minutes. Run with:
//!
//! ```text
//! cargo test -p via-integration --release -- --ignored
//! ```

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::rns::basis::paper::ViaCQ1Rns;
use via_primitives::algebra::zq::modulus::paper::{ViaCP, ViaCQ2, ViaCQ3, ViaCQ4};
use via_primitives::conversion::{
    LweToRlweKeyRnsN2048, gen_lwe_to_rlwe_key_rns_n2048_boxed, lwe_to_rlwe_rns_n2048_eval,
};
use via_primitives::params::{ViaCPolyP, ViaCPolyQ1Rns, ViaCPolyQ2, ViaCPolyQ3, ViaCPolyQ4};
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{KeyDist, PIRParams, PublicParams};
use via_server::ViaCServer;

const N1: usize = 2048;
const N2: usize = 512;
const D: usize = 4; // d = N1 / N2
const L_QUERY: usize = 2;
const L_CK: usize = 18;
const L_RSK: usize = 8;
const CK_BASE: u64 = 18;

const NUM_ROWS: usize = 2; // I
const NUM_COLS: usize = 2; // J

// Paper ring instantiation (Appendix B).
type R1 = ViaCPolyQ1Rns<N1>; // S1 @ q1-RNS, n1
type R2N1 = ViaCPolyQ2<N1>; // q2 @ n1 (DMux output / FirstDim)
type R3N1 = ViaCPolyQ3<N1>; // q3 @ n1 (mod_switch_sym intermediate + rekey target)
type R3N2 = ViaCPolyQ3<N2>; // q3 @ n2 (S2 ring + answer mask)
type R4N2 = ViaCPolyQ4<N2>; // q4 @ n2 (answer body)
type RpN1 = ViaCPolyP<N1>; // p @ n1 (DB cell embed target)
type Rec = ViaCPolyP<N2>; // p @ n2 (records)
type K = LweToRlweKeyRnsN2048<ViaCQ1Rns, L_CK>;

type PaperClient = Client<N1, N2, R1, R3N2, L_QUERY, L_CK, L_RSK, D>;
type PaperServer = ViaCServer<K, N1, N2, R1, R2N1, R3N2, R4N2, L_QUERY, L_CK, L_RSK, D>;
type PaperPp = PublicParams<K, N1, N2, R1, R3N2, L_QUERY, L_CK, L_RSK, D>;

/// Client `setup` (the keygen-heavy phase) factored out for reuse by both the
/// full round-trip and the setup-only stack-attribution test.
fn paper_setup(prg: &mut Shake256Prg) -> (PaperClient, PaperPp) {
    PaperClient::setup(
        ViaCQ1Rns::default(),
        ViaCQ3::default(),
        paper_params(),
        NUM_ROWS,
        NUM_COLS,
        CK_BASE,
        Distribution::Ternary,
        Distribution::Ternary,
        Distribution::Ternary,
        prg,
        gen_lwe_to_rlwe_key_rns_n2048_boxed::<ViaCQ1Rns, L_CK>,
        |sk1, sk2, dist, prg| {
            let q3_mod = RingPoly::modulus(sk2.poly());
            let s1_q3 = rekey_secret_key::<N1, R1, R3N1>(sk1, q3_mod);
            gen_rsk::<N1, N2, R3N1, R3N2, L_RSK, D>(&s1_q3, sk2, 8, dist, prg)
        },
    )
    .expect("client setup")
}

fn paper_params() -> PIRParams {
    PIRParams::new(
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
    )
}

/// A distinct record per flat index (first few coeffs vary; rest zero).
fn record(m: usize, p: ViaCP) -> Rec {
    let coeffs: [u64; N2] =
        core::array::from_fn(|j| if j < 4 { ((m + 1 + j) % 16) as u64 } else { 0 });
    Poly::new(p, coeffs)
}

/// Full paper-scale protocol round-trip for `index`; returns `(recovered, expected)`.
fn round_trip(index: usize) -> (Rec, Rec) {
    let q1 = ViaCQ1Rns::default();
    let q2 = ViaCQ2::default();
    let q3 = ViaCQ3::default();
    let q4 = ViaCQ4::default();
    let p = ViaCP::default();
    let mut prg = Shake256Prg::new(b"via-c-paper-scale-e2e");

    // --- Client setup ----------------------------------------------------
    let (client, pp) = paper_setup(&mut prg);

    // --- Server setup ----------------------------------------------------
    let records: Vec<Rec> = (0..D * NUM_ROWS * NUM_COLS).map(|m| record(m, p)).collect();
    let server = PaperServer::setup::<RpN1, Rec>(&records, pp, q1, q2, q3, q4, p);

    // --- Query → Answer → Recover ----------------------------------------
    let query = client.query(index, &mut prg).expect("client query");
    let answer = server
        .answer::<R3N1, _>(&query, lwe_to_rlwe_rns_n2048_eval::<ViaCQ1Rns, L_CK>)
        .expect("server answer");
    let recovered: Rec = client
        .recover::<R3N2, R4N2, Rec>(&answer, q3, q4, p)
        .expect("client recover");

    (recovered, records[index])
}

/// Index 15 = (α,β,γ) = (1,1,3): DMux + CMux + both CRot bits all selecting —
/// the full gate path at paper-scale moduli/depths.
#[test]
#[ignore = "paper-scale n2048 RNS pipeline — heavy; run with --release -- --ignored"]
fn client_server_e2e_paper_scale_index_15() {
    // Default 6 MB stack (2× margin over the measured ~3 MB tightest). Override
    // with VIA_E2E_STACK_MB=N to bisect without recompiling.
    //
    // The `perf/client-keygen` work cut the requirement from ~12 MB to ~3 MB by
    // heap-building the per-S1 keys (conv-key RLev, ring-switch key) and the
    // cascade key per-sample, so keygen no longer moves ~MiB blocks by value
    // (setup-only is now ~2 MB). The residual ~3 MB is the answer phase, driven
    // by the depth-18 RNS cascade operation (~2 MB) — the summed O(N) stack
    // scratch along its `cascade → conv_step → key_switch → gadget_product`
    // chain, not data liveness or inlining (both ruled out by measurement).
    // Pushing under the 2 MB default would mean heap-ifying that hot-path
    // scratch, a poor time-for-stack trade in the compute-bound answer; a small
    // explicit stack is the right call. A spawn is needed regardless (the
    // default test thread is ~2 MB).
    let stack_mb: usize = std::env::var("VIA_E2E_STACK_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    std::thread::Builder::new()
        .stack_size(stack_mb << 20)
        .spawn(|| {
            let (got, want) = round_trip(15);
            assert_eq!(
                got, want,
                "paper-scale recover(query(15)) must equal record[15]"
            );
        })
        .expect("spawn paper-scale thread")
        .join()
        .expect("paper-scale e2e thread panicked");
}

/// Setup-only stack attribution: run just `Client::setup` (the keygen phase)
/// and black-box the result, so its stack can be bisected in isolation
/// (`VIA_E2E_STACK_MB`). Comparing this against the full pipeline tells us how
/// much of the requirement is keygen vs the answer pipeline.
#[test]
#[ignore = "paper-scale setup-only stack attribution; run with --release -- --ignored"]
fn client_setup_only_paper_scale() {
    let stack_mb: usize = std::env::var("VIA_E2E_STACK_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(6);
    std::thread::Builder::new()
        .stack_size(stack_mb << 20)
        .spawn(|| {
            let mut prg = Shake256Prg::new(b"via-c-paper-setup-only");
            let (client, pp) = paper_setup(&mut prg);
            core::hint::black_box((&client, &pp));
        })
        .expect("spawn setup-only thread")
        .join()
        .expect("setup-only thread panicked");
}
