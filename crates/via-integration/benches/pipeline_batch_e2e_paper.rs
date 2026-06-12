//! Paper-scale VIA-B **full `answer_batch` e2e** benchmark (n2048 RNS, Appendix B).
//!
//! Complements `pipeline_batch_paper` (which times the two *isolated* new VIA-B
//! costs — `q2_key_build`, `repack` — at the production T=256) by timing the
//! whole batch answer pipeline end to end through the real `Server::answer_batch`:
//! `answer_through_crot × T` (the variant-common prefix) → `Repack_{N2}` →
//! `RespComp`, at paper rings/moduli.
//!
//! Batch size **T=8** with `N3 = N2/T = 64` (the record-fit boundary `T·N3 = N2`)
//! — exactly the tractable choice of `tests/batch_e2e_paper.rs`: the production
//! T=256 runs `answer_through_crot` 256× (~3 h under criterion), while T=8
//! exercises the same crypto (real degree, the RNS-q1→single-prime-q2 cross-type
//! key, the full prefix + repack + resp_comp) in ~minutes.
//!
//! The grid is env-driven (`VIA_BENCH_ROWS`/`VIA_BENCH_COLS`, default 2×2) so the
//! `/benchmark` CI workflow runs it at the larger 8×16 grid alongside
//! `pipeline_paper`. The bench-id `paper_batch/03_answer_batch_e2e` groups it with
//! the other VIA-B rows in the comparison table.
//!
//! The whole criterion flow runs on a 32 MiB-stack thread: the depth-18 RNS
//! cascade's O(N) scratch needs more than the default main-thread stack
//! (`tests/batch_e2e_paper.rs` uses 16 MiB), and this keeps the bench
//! self-contained — no external `ulimit`, works locally and in CI alike.
//!
//! Run: `cargo bench -p via-integration --features via-b --bench pipeline_batch_e2e_paper`.
//! Gated on `via-b`; under default the bench binary is a no-op `main`.
#![allow(missing_docs)]

#[cfg(feature = "via-b")]
mod b {
    use criterion::{Criterion, black_box};
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
    use via_protocol::{BatchedQuery, KeyDist, PIRParams};
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

    type R1 = ViaCPolyQ1Rns<N1>; // S1 @ q1-RNS, n1
    type R2 = ViaCPolyQ2<N1>; // q2 @ n1 (post-CRot / repack ring)
    type R3N1 = ViaCPolyQ3<N1>; // q3 @ n1 (mod_switch_sym intermediate)
    type R3N2 = ViaCPolyQ3<N2>; // q3 @ n2 (S2 ring + answer mask)
    type R4N2 = ViaCPolyQ4<N2>; // q4 @ n2 (answer body)
    type RpN1 = ViaCPolyP<N1>; // p @ n1 (DB embed target)
    type Rec = ViaCPolyP<N3>; // p @ n3 (records)
    type K = LweToRlweKeyRnsN2048<ViaCQ1Rns, L_CK>;
    type PaperBClient = Client<N1, N2, R1, R3N2, L_QUERY, L_CK, L_RSK, D>;
    type PaperBServer = ViaBServer<K, N1, N2, N3, R1, R2, R3N2, R4N2, L_QUERY, L_CK, L_RSK, D>;
    type Batch = BatchedQuery<N1, 1, <R1 as RingPoly<N1>>::Projected<1>>;

    /// Paper grid dim (I or J) from env — default 2 preserves the local 2×2 run.
    /// The `/benchmark` workflow sets `VIA_BENCH_ROWS=8`, `VIA_BENCH_COLS=16`.
    fn grid_dim(var: &str, default: usize) -> usize {
        let v = std::env::var(var)
            .ok()
            .and_then(|s| s.trim().parse::<usize>().ok())
            .unwrap_or(default);
        assert!(
            v.is_power_of_two(),
            "{var}={v} must be a power of two (DMux/CMux trees)"
        );
        v
    }

    /// A distinct degree-n3 record per flat index (mirrors `tests/batch_e2e_paper.rs`).
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

    struct Fixture {
        server: PaperBServer,
        batch: Batch,
        q2: ViaCQ2,
    }

    /// Build the heavy fixture once (untimed): client+server setup at the given
    /// grid, plus the batched query for the fixed indices.
    fn build_fixture(num_rows: usize, num_cols: usize, idxs: &[usize; T]) -> Fixture {
        let q1 = ViaCQ1Rns::default();
        let q2 = ViaCQ2::default();
        let q3 = ViaCQ3::default();
        let q4 = ViaCQ4::default();
        let p = ViaCP::default();
        let mut prg = Shake256Prg::new(b"via-b-bench-batch-e2e");

        let (client, pp) = PaperBClient::setup(
            q1,
            q3,
            paper_params(),
            num_rows,
            num_cols,
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

        let records: Vec<Rec> = (0..D3 * num_rows * num_cols).map(record).collect();
        let server = PaperBServer::setup::<RpN1, Rec>(&records, pp, q1, q2, q3, q4, p);

        let batch = client
            .batch_query::<T, N3>(idxs, &mut prg)
            .expect("batch_query");
        Fixture { server, batch, q2 }
    }

    pub fn batch_e2e_paper_benches(c: &mut Criterion) {
        let num_rows = grid_dim("VIA_BENCH_ROWS", 2);
        let num_cols = grid_dim("VIA_BENCH_COLS", 2);
        // Indices valid for any grid >= 2×2 (range D3·I·J >= 128; max idx = 120).
        let idxs: [usize; T] = [3, 17, 42, 5, 88, 100, 7, 120];
        let fx = build_fixture(num_rows, num_cols, &idxs);

        // Repack at single-prime q2: derive the q2 key (boxed/heap) from the RNS-q1
        // cascade key by cross-type mod-switch (§3.5), then pack — the injected
        // closure `Server::answer_batch` calls between the prefix and resp_comp.
        let q2 = fx.q2;
        let repack = |rotateds: &[RLWECiphertext<N1, R2>], k: &K| {
            let q2_key = repack_keys_poly_2048_t8_from_rns_cascade_boxed(k, q2);
            let arr: &[_; T] = rotateds.try_into().expect("T rotated ciphertexts");
            repack_poly_2048_t8(arr, &*q2_key, CK_BASE)
        };
        let cascade = lwe_to_rlwe_rns_n2048_eval::<ViaCQ1Rns, L_CK>;

        c.bench_function("paper_batch/03_answer_batch_e2e", |b| {
            b.iter(|| {
                black_box(
                    fx.server
                        .answer_batch::<R3N1, T, _, _>(&fx.batch, &repack, cascade)
                        .expect("answer_batch"),
                )
            })
        });
    }

    /// Reduced sampling — one batch answer is ~seconds.
    pub fn batch_criterion() -> Criterion {
        Criterion::default().sample_size(10).configure_from_args()
    }
}

// Custom harness (replaces criterion_main!): run the whole criterion flow on a
// 32 MiB-stack thread so the depth-18 RNS cascade's O(N) scratch (≈ the 16 MiB
// `tests/batch_e2e_paper.rs` uses, plus the fixture build + criterion overhead)
// never overflows the default main-thread stack — locally and in CI alike.
#[cfg(feature = "via-b")]
fn main() {
    std::thread::Builder::new()
        .stack_size(32 << 20)
        .spawn(|| {
            let mut c = b::batch_criterion();
            b::batch_e2e_paper_benches(&mut c);
            c.final_summary();
        })
        .expect("spawn bench thread")
        .join()
        .expect("bench thread panicked");
}

#[cfg(not(feature = "via-b"))]
fn main() {}
