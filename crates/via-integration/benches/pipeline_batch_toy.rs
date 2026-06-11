//! Toy VIA-B **batch** per-phase + full-pipeline benchmarks (n1=8, n2=4, n3=2,
//! T=2; `repack_n8_t2`). Mirrors `pipeline_toy.rs` for the batch path: the client
//! `batch_query`, the **isolated repack** (the one new VIA-B cost), the full
//! `answer_batch` (T prefixes + repack + RespComp), `recover_batch`, and e2e.
//!
//! Run: `cargo bench -p via-integration --features via-b --bench pipeline_batch_toy`.
//! Gated on `via-b`; under default the bench binary is a no-op `main`.
#![allow(missing_docs)]

#[cfg(feature = "via-b")]
mod b {
    use criterion::{BatchSize, Criterion, black_box};
    use via_client::Client;
    use via_primitives::algebra::ring::RingPoly;
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::zq::modulus::DynModulus;
    use via_primitives::conversion::{
        LweToRlweKeyN8, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8, repack_keys_n8_t2_from_cascade,
        repack_n8_t2,
    };
    use via_primitives::encryption::types::RLWECiphertext;
    use via_primitives::sampling::distribution::Distribution;
    use via_primitives::sampling::prg::Shake256Prg;
    use via_primitives::switching::gen_rsk;
    use via_primitives::switching::rekey::rekey_secret_key;
    use via_protocol::{BatchedQuery, KeyDist, PIRParams, PublicParams};
    use via_server::{answer_batch, answer_through_crot};

    const N1: usize = 8;
    const N2: usize = 4;
    const N3: usize = 2;
    const T: usize = 2;
    const D: usize = 2;
    const D3: usize = N1 / N3;
    const L_QUERY: usize = 7;
    const L_CK: usize = 7;
    const L_RSK: usize = 8;
    const Q1: u64 = 1 << 36;
    const Q2: u64 = 1 << 28; // params metadata; runtime q2 = q1 (see batch_e2e_toy.rs)
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
    type R2 = Poly<N3, DynModulus, Coefficient>;
    type K = LweToRlweKeyN8<DynModulus, L_CK>;
    type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;
    type ToyPp = PublicParams<K, N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;

    const IDXS: [usize; T] = [3, 11];

    fn record(m: usize, p: DynModulus) -> R2 {
        let m = m as u64;
        Poly::new(p, [(m + 1) % P, (2 * m + 3) % P])
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

    /// Repack closure: borrow the cascade-key suffix (§3.5) + base CK_BASE.
    fn repack(rotateds: &[RLWECiphertext<N1, R8>], k: &K) -> RLWECiphertext<N1, R8> {
        let arr: &[_; T] = rotateds.try_into().expect("T rotated ciphertexts");
        repack_n8_t2(arr, &repack_keys_n8_t2_from_cascade(k), CK_BASE)
    }

    struct Fixture {
        client: ToyClient,
        pp: ToyPp,
        encoded_db: Vec<Vec<R8>>,
        batch: BatchedQuery<N1, 1, <R8 as RingPoly<N1>>::Projected<1>>,
        q1: DynModulus,
        q3: DynModulus,
        q4: DynModulus,
        p: DynModulus,
    }

    fn build_fixture() -> Fixture {
        let q1 = DynModulus::new(Q1);
        let q3 = DynModulus::new(Q3);
        let p = DynModulus::new(P);
        let mut prg = Shake256Prg::new(b"via-b-bench-batch-toy");

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

        // The encoded DB (free `setup_db`, N_REC = N3): d3·I·J degree-n3 records.
        let records: Vec<R2> = (0..D3 * NUM_ROWS * NUM_COLS)
            .map(|m| record(m, p))
            .collect();
        let encoded_db = via_server::setup_db::<N1, N3, R8, R2>(&records, NUM_ROWS, NUM_COLS, p);
        let batch = client
            .batch_query::<T, N3>(&IDXS, &mut prg)
            .expect("batch_query");

        Fixture {
            client,
            pp,
            encoded_db,
            batch,
            q1,
            q3,
            q4: DynModulus::new(Q4),
            p,
        }
    }

    pub fn batch_benches(c: &mut Criterion) {
        let fx = build_fixture();
        let q2 = fx.q1; // runtime q2 = q1
        let cascade = lwe_to_rlwe_n8::<DynModulus, L_CK>;
        let cascade_key: &K = &fx.pp.query_comp_key.lwe_to_rlwe_key;

        // Run answer_batch once (untimed) to obtain the answer for recover_batch.
        let run_answer_batch = || {
            answer_batch::<N1, N2, N3, T, R8, R8, R8, R4, R4, R8, K, L_QUERY, L_CK, L_RSK, D, _, _>(
                &fx.batch,
                &fx.pp,
                &fx.encoded_db,
                fx.q1,
                q2,
                fx.q3,
                fx.q4,
                repack,
                cascade,
            )
            .expect("answer_batch")
        };
        let answer = run_answer_batch();

        // Build the T post-CRot ciphertexts (untimed) to isolate the repack.
        let rotated: Vec<RLWECiphertext<N1, R8>> = fx
            .batch
            .queries
            .iter()
            .map(|q| {
                answer_through_crot::<N1, N2, N3, R8, R8, R4, R8, K, L_QUERY, L_CK, L_RSK, D, _>(
                    q,
                    &fx.pp,
                    &fx.encoded_db,
                    fx.q1,
                    q2,
                    cascade,
                )
                .expect("answer_through_crot")
            })
            .collect();

        c.bench_function("batch/01_batch_query", |b| {
            b.iter_batched(
                || Shake256Prg::new(b"via-b-bench-bq"),
                |mut prg| black_box(fx.client.batch_query::<T, N3>(&IDXS, &mut prg)),
                BatchSize::SmallInput,
            )
        });
        // The one new VIA-B step, isolated: pack T post-CRot cts into one.
        c.bench_function("batch/02_repack", |b| {
            b.iter(|| black_box(repack(&rotated, cascade_key)))
        });
        // Full answer = T × (steps 1–6) + repack + RespComp once.
        c.bench_function("batch/03_answer_batch", |b| {
            b.iter(|| black_box(run_answer_batch()))
        });
        c.bench_function("batch/04_recover_batch", |b| {
            b.iter(|| {
                black_box(
                    fx.client
                        .recover_batch::<R4, R4, R4, N3, T>(&answer, fx.q3, fx.q4, fx.p),
                )
            })
        });
        c.bench_function("batch/05_e2e_full", |b| {
            b.iter_batched(
                || Shake256Prg::new(b"via-b-bench-e2e"),
                |mut prg| {
                    let batch = fx.client.batch_query::<T, N3>(&IDXS, &mut prg).unwrap();
                    let ans = answer_batch::<
                        N1,
                        N2,
                        N3,
                        T,
                        R8,
                        R8,
                        R8,
                        R4,
                        R4,
                        R8,
                        K,
                        L_QUERY,
                        L_CK,
                        L_RSK,
                        D,
                        _,
                        _,
                    >(
                        &batch,
                        &fx.pp,
                        &fx.encoded_db,
                        fx.q1,
                        q2,
                        fx.q3,
                        fx.q4,
                        repack,
                        cascade,
                    )
                    .unwrap();
                    black_box(
                        fx.client
                            .recover_batch::<R4, R4, R4, N3, T>(&ans, fx.q3, fx.q4, fx.p),
                    )
                },
                BatchSize::SmallInput,
            )
        });
    }
}

#[cfg(feature = "via-b")]
criterion::criterion_group!(benches, b::batch_benches);
#[cfg(feature = "via-b")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "via-b"))]
fn main() {}
