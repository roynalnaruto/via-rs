//! Paper-scale (n2048 RNS, Appendix B) per-step + full-pipeline benchmarks —
//! the authoritative, opt-in suite (reduced sampling; minutes per run).
//!
//! Same isolation technique as `pipeline_toy` (build each step's input once by
//! replaying the prior steps, time only that step), at the real paper params
//! where the depth-18 RNS cascade dominates. Run: `just bench-paper`.
#![allow(missing_docs)] // criterion_group! generates undocumented public items

use std::time::Duration;

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::rns::basis::paper::ViaCQ1Rns;
use via_primitives::algebra::zq::modulus::paper::{ViaCP, ViaCQ2, ViaCQ3, ViaCQ4};
use via_primitives::conversion::{
    LweToRlweKeyRnsN2048, gen_lwe_to_rlwe_key_rns_n2048_boxed, lwe_to_rlwe_rns_n2048,
};
use via_primitives::encryption::types::{RGSWCiphertext, RLWECiphertext};
use via_primitives::gates::{CRotDir, cmux_tree, crot, dmux_tree, mod_switch_rgsw};
use via_primitives::params::{ViaCPolyP, ViaCPolyQ1Rns, ViaCPolyQ2, ViaCPolyQ3, ViaCPolyQ4};
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::mod_switch::mod_switch_sym;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{CompressedQuery, KeyDist, PIRParams, PublicParams};
use via_server::{answer_one_query, first_dim, query_decomp, resp_comp, setup_db};

// ── Paper params (mirror crates/via-integration/tests/client_server_e2e_paper.rs) ──
const N1: usize = 2048;
const N2: usize = 512;
const D: usize = 4;
const L_QUERY: usize = 2;
const L_CK: usize = 18;
const L_RSK: usize = 8;
const CK_BASE: u64 = 18;
const NUM_ROWS: usize = 2;
const NUM_COLS: usize = 2;
const INDEX: usize = 15; // (α,β,γ)=(1,1,3) — every gate selecting.

type R1 = ViaCPolyQ1Rns<N1>; // S1 @ q1-RNS, n1
type R2N1 = ViaCPolyQ2<N1>; // q2 @ n1
type R3N1 = ViaCPolyQ3<N1>; // q3 @ n1 (resp_comp intermediate)
type R3N2 = ViaCPolyQ3<N2>; // q3 @ n2 (S2 ring + answer mask)
type R4N2 = ViaCPolyQ4<N2>; // q4 @ n2 (answer body)
type RpN1 = ViaCPolyP<N1>; // p @ n1 (DB cell embed target)
type Rec = ViaCPolyP<N2>; // p @ n2 (records / recovered)
type K = LweToRlweKeyRnsN2048<ViaCQ1Rns, L_CK>;
type PaperClient = Client<N1, N2, R1, R3N2, L_QUERY, L_CK, L_RSK, D>;
type PaperPp = PublicParams<K, N1, N2, R1, R3N2, L_QUERY, L_CK, L_RSK, D>;

fn record(m: usize, p: ViaCP) -> Rec {
    let coeffs: [u64; N2] =
        core::array::from_fn(|j| if j < 4 { ((m + 1 + j) % 16) as u64 } else { 0 });
    via_primitives::algebra::ring::element::Poly::new(p, coeffs)
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
        L_QUERY,
        81,
        L_QUERY,
        8,
        L_RSK,
        KeyDist::Ternary,
        KeyDist::Ternary,
        26,
        None,
        None,
        None,
        128,
    )
}

fn gen_rsk_closure(
    sk1: &via_primitives::encryption::types::SecretKey<N1, R1>,
    sk2: &via_primitives::encryption::types::SecretKey<N2, R3N2>,
    dist: Distribution,
    prg: &mut Shake256Prg,
) -> via_primitives::switching::RingSwitchKey<N1, N2, R3N2, L_RSK, D> {
    let q3_mod = RingPoly::modulus(sk2.poly());
    let s1_q3 = rekey_secret_key::<N1, R1, R3N1>(sk1, q3_mod);
    gen_rsk::<N1, N2, R3N1, R3N2, L_RSK, D>(&s1_q3, sk2, 8, dist, prg)
}

struct Fixture {
    client: PaperClient,
    pp: PaperPp,
    encoded_db: Vec<Vec<RpN1>>,
    query: CompressedQuery<N1, 1, <R1 as RingPoly<N1>>::Projected<1>>,
    q1: ViaCQ1Rns,
    q2: ViaCQ2,
    q3: ViaCQ3,
    q4: ViaCQ4,
    p: ViaCP,
}

fn build_fixture() -> Fixture {
    let (q1, q2, q3, q4, p) = (
        ViaCQ1Rns::default(),
        ViaCQ2::default(),
        ViaCQ3::default(),
        ViaCQ4::default(),
        ViaCP::default(),
    );
    let mut prg = Shake256Prg::new(b"via-c-bench-paper");
    let (client, pp) = PaperClient::setup(
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
        gen_rsk_closure,
    )
    .expect("client setup");
    let records: Vec<Rec> = (0..D * NUM_ROWS * NUM_COLS).map(|m| record(m, p)).collect();
    let encoded_db = setup_db::<N1, N2, RpN1, Rec>(&records, NUM_ROWS, NUM_COLS, p);
    let query = client.query(INDEX, &mut prg).expect("client query");
    Fixture {
        client,
        pp,
        encoded_db,
        query,
        q1,
        q2,
        q3,
        q4,
        p,
    }
}

fn paper_benches(c: &mut Criterion) {
    let fx = build_fixture();
    let cascade = lwe_to_rlwe_rns_n2048::<ViaCQ1Rns, L_CK>;
    let b1 = fx.pp.params.gadget_base_1;
    let b2 = fx.pp.params.gadget_base_2;
    let b_rsk = fx.pp.params.gadget_base_rsk;
    let ck_base = fx.pp.ck_base;
    // Paper: I=J=2 (1 bit each), d=4 (2 bits).
    let (num_dmux, num_cmux, num_crot) = (1usize, 1usize, 2usize);

    let run_qd = || {
        query_decomp::<N1, R1, K, L_QUERY, L_CK, _>(
            &fx.query.ciphertexts,
            &fx.pp.query_comp_key,
            num_dmux,
            num_cmux,
            num_crot,
            ck_base,
            ck_base,
            cascade,
        )
    };
    let run_dmux = |dq: &via_protocol::DecompressedQuery<N1, R1, L_QUERY>| {
        let mut delta_coeffs = [0u128; N1];
        delta_coeffs[0] = fx.pp.params.delta();
        let trivial = RLWECiphertext::trivial(fx.q1, &R1::from_u128_coeffs(fx.q1, &delta_coeffs));
        let zero_q1 = RLWECiphertext::new(R1::zero(fx.q1), R1::zero(fx.q1));
        let mut out: Vec<RLWECiphertext<N1, R1>> = vec![zero_q1; NUM_ROWS];
        dmux_tree(&dq.dmux_bits, trivial, &mut out, b1, b1);
        out
    };
    let run_modswitch = |dmux_out: &[RLWECiphertext<N1, R1>]| {
        dmux_out
            .iter()
            .map(|ct| mod_switch_sym::<N1, R1, R2N1>(ct, fx.q2))
            .collect::<Vec<RLWECiphertext<N1, R2N1>>>()
    };
    let run_first_dim = |switched: &[RLWECiphertext<N1, R2N1>]| {
        let db_q2: Vec<Vec<R2N1>> = fx
            .encoded_db
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        let mut cc = [0u128; N1];
                        cell.to_u128_coeffs(&mut cc);
                        R2N1::from_u128_coeffs(fx.q2, &cc)
                    })
                    .collect()
            })
            .collect();
        first_dim::<N1, R2N1>(switched, &db_q2, fx.q2)
    };
    let run_cmux = |dq: &via_protocol::DecompressedQuery<N1, R1, L_QUERY>,
                    mut fd: Vec<RLWECiphertext<N1, R2N1>>| {
        let cmux_q2: Vec<RGSWCiphertext<N1, R2N1, L_QUERY, L_QUERY>> = dq
            .cmux_bits
            .iter()
            .map(|rgsw| mod_switch_rgsw::<N1, R1, R2N1, L_QUERY, L_QUERY>(rgsw, fx.q2))
            .collect();
        cmux_tree(&cmux_q2, &mut fd, b2, b2)
    };
    let run_crot = |dq: &via_protocol::DecompressedQuery<N1, R1, L_QUERY>,
                    selected: RLWECiphertext<N1, R2N1>| {
        let crot_q2: Vec<RGSWCiphertext<N1, R2N1, L_QUERY, L_QUERY>> = dq
            .crot_bits
            .iter()
            .map(|rgsw| mod_switch_rgsw::<N1, R1, R2N1, L_QUERY, L_QUERY>(rgsw, fx.q2))
            .collect();
        crot(CRotDir::SlotExtract, &crot_q2, selected, b2, b2)
    };
    let run_resp_comp = |rotated: &RLWECiphertext<N1, R2N1>| {
        resp_comp::<N1, N2, R2N1, R3N1, R3N2, R4N2, L_RSK, D>(
            rotated,
            &fx.pp.ring_switch_key,
            fx.q3,
            fx.q4,
            b_rsk,
        )
    };

    // Build the chain ONCE (untimed) to obtain each step's input.
    let dq = run_qd();
    let dmux_out = run_dmux(&dq);
    let switched = run_modswitch(&dmux_out);
    let fd_results = run_first_dim(&switched);
    let selected = run_cmux(&dq, fd_results.clone());
    let rotated = run_crot(&dq, selected);
    let answer = run_resp_comp(&rotated);

    c.bench_function("paper/00_client_setup", |b| {
        b.iter_batched(
            || Shake256Prg::new(b"via-c-bench-paper-setup"),
            |mut prg| {
                black_box(PaperClient::setup(
                    fx.q1,
                    fx.q3,
                    paper_params(),
                    NUM_ROWS,
                    NUM_COLS,
                    CK_BASE,
                    Distribution::Ternary,
                    Distribution::Ternary,
                    Distribution::Ternary,
                    &mut prg,
                    gen_lwe_to_rlwe_key_rns_n2048_boxed::<ViaCQ1Rns, L_CK>,
                    gen_rsk_closure,
                ))
            },
            BatchSize::SmallInput,
        )
    });
    c.bench_function("paper/00_client_query", |b| {
        b.iter_batched(
            || Shake256Prg::new(b"via-c-bench-paper-query"),
            |mut prg| black_box(fx.client.query(INDEX, &mut prg)),
            BatchSize::SmallInput,
        )
    });

    c.bench_function("paper/01_query_decomp", |b| b.iter(|| black_box(run_qd())));
    c.bench_function("paper/02_dmux", |b| b.iter(|| black_box(run_dmux(&dq))));
    c.bench_function("paper/03_mod_switch", |b| {
        b.iter(|| black_box(run_modswitch(&dmux_out)))
    });
    c.bench_function("paper/04_first_dim", |b| {
        b.iter(|| black_box(run_first_dim(&switched)))
    });
    c.bench_function("paper/05_cmux", |b| {
        b.iter_batched(
            || fd_results.clone(),
            |fd| black_box(run_cmux(&dq, fd)),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("paper/06_crot", |b| {
        b.iter_batched(
            || selected,
            |sel| black_box(run_crot(&dq, sel)),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("paper/07_resp_comp", |b| {
        b.iter(|| black_box(run_resp_comp(&rotated)))
    });

    c.bench_function("paper/08_client_recover", |b| {
        b.iter(|| {
            black_box(
                fx.client
                    .recover::<R3N2, R4N2, Rec>(&answer, fx.q3, fx.q4, fx.p),
            )
        })
    });
    c.bench_function("paper/09_e2e_full", |b| {
        b.iter_batched(
            || Shake256Prg::new(b"via-c-bench-paper-e2e"),
            |mut prg| {
                let query = fx.client.query(INDEX, &mut prg).expect("client query");
                let ans = answer_one_query::<
                    N1,
                    N2,
                    R1,
                    R2N1,
                    R3N1,
                    R3N2,
                    R4N2,
                    RpN1,
                    K,
                    L_QUERY,
                    L_CK,
                    L_RSK,
                    D,
                    _,
                >(
                    &query,
                    &fx.pp,
                    &fx.encoded_db,
                    fx.q1,
                    fx.q2,
                    fx.q3,
                    fx.q4,
                    cascade,
                )
                .unwrap();
                black_box(
                    fx.client
                        .recover::<R3N2, R4N2, Rec>(&ans, fx.q3, fx.q4, fx.p),
                )
            },
            BatchSize::SmallInput,
        )
    });
}

// Reduced sampling — one paper answer is ~tens of seconds.
fn paper_criterion() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(15))
}

criterion_group! {
    name = benches;
    config = paper_criterion();
    targets = paper_benches
}
criterion_main!(benches);
