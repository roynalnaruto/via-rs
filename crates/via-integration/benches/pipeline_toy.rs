//! Fast (toy single-prime, n8/n4) per-step + full-pipeline benchmarks for the
//! VIA-C client↔server protocol — the frequent regression suite.
//!
//! Each of the 7 answer steps plus `Client::{setup, query, recover}` is timed in
//! isolation: the step's input is built **once** (untimed) by replaying the
//! prior steps, mirroring `answer_one_query`; the bench then times only that
//! step (cloning the input per-iteration for the steps that consume/mutate it).
//!
//! Run: `just bench` · baseline compare: `just bench-save NAME` / `just bench-cmp NAME`.
#![allow(missing_docs)] // criterion_group! generates undocumented public items

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::conversion::{LweToRlweKeyN8, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8};
use via_primitives::encryption::types::{RGSWCiphertext, RLWECiphertext};
use via_primitives::gates::{CRotDir, cmux_tree, crot, dmux_tree, mod_switch_rgsw};
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::mod_switch::mod_switch_sym;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{CompressedQuery, KeyDist, PIRParams, PublicParams};
use via_server::{answer_one_query, first_dim, query_decomp, resp_comp, setup_db};

// ── Toy params (mirror crates/via-integration/tests/client_server_e2e.rs) ──
const N1: usize = 8;
const N2: usize = 4;
const D: usize = 2;
const L_QUERY: usize = 7;
const L_CK: usize = 7;
const L_RSK: usize = 8;
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
const INDEX: usize = 5; // (α,β,γ)=(0,1,1) — DMux off, CMux + CRot on.

type R8 = Poly<N1, DynModulus, Coefficient>;
type R4 = Poly<N2, DynModulus, Coefficient>;
type K = LweToRlweKeyN8<DynModulus, L_CK>;
type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;
type ToyPp = PublicParams<K, N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;

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
    )
}

fn gen_cascade(
    sk: &via_primitives::encryption::types::SecretKey<N1, R8>,
    base: u64,
    dist: Distribution,
    prg: &mut Shake256Prg,
) -> Box<K> {
    Box::new(gen_lwe_to_rlwe_key_n8::<DynModulus, L_CK>(
        sk, base, dist, prg,
    ))
}

fn gen_rsk_closure(
    sk1: &via_primitives::encryption::types::SecretKey<N1, R8>,
    sk2: &via_primitives::encryption::types::SecretKey<N2, R4>,
    dist: Distribution,
    prg: &mut Shake256Prg,
) -> via_primitives::switching::RingSwitchKey<N1, N2, R4, L_RSK, D> {
    let q3_mod = RingPoly::modulus(sk2.poly());
    let s1_q3 = rekey_secret_key::<N1, R8, R8>(sk1, q3_mod);
    gen_rsk::<N1, N2, R8, R4, L_RSK, D>(&s1_q3, sk2, B_RSK, dist, prg)
}

struct Fixture {
    client: ToyClient,
    pp: ToyPp,
    encoded_db: Vec<Vec<R8>>,
    query: CompressedQuery<N1, 1, <R8 as RingPoly<N1>>::Projected<1>>,
    q1: DynModulus,
    q2: DynModulus,
    q3: DynModulus,
    q4: DynModulus,
    p: DynModulus,
}

fn build_fixture() -> Fixture {
    let q1 = DynModulus::new(Q1);
    let q2 = DynModulus::new(Q2);
    let q3 = DynModulus::new(Q3);
    let q4 = DynModulus::new(Q4);
    let p = DynModulus::new(P);
    let mut prg = Shake256Prg::new(b"via-c-bench-toy");

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
        gen_cascade,
        gen_rsk_closure,
    )
    .expect("client setup");
    let records: Vec<R4> = (0..D * NUM_ROWS * NUM_COLS).map(|m| record(m, p)).collect();
    let encoded_db = setup_db::<N1, N2, R8, R4>(&records, NUM_ROWS, NUM_COLS, p);
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

fn toy_benches(c: &mut Criterion) {
    let fx = build_fixture();
    let cascade = lwe_to_rlwe_n8::<DynModulus, L_CK>;
    let b1 = fx.pp.params.gadget_base_1;
    let b2 = fx.pp.params.gadget_base_2;
    let b_rsk = fx.pp.params.gadget_base_rsk;
    let ck_base = fx.pp.ck_base;
    // Toy: I=J=d=2 → one bit each.
    let (num_dmux, num_cmux, num_crot) = (1usize, 1usize, 1usize);

    // ── per-step "run" helpers: each = exactly what answer_one_query's
    // stepN debug_span scopes (used for both the untimed chain build and the
    // timed bench body). ──────────────────────────────────────────────────
    let run_qd = || {
        query_decomp::<N1, R8, K, L_QUERY, L_CK, _>(
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
    let run_dmux = |dq: &via_protocol::DecompressedQuery<N1, R8, L_QUERY>| {
        let mut delta_coeffs = [0u128; N1];
        delta_coeffs[0] = fx.pp.params.delta();
        let trivial = RLWECiphertext::trivial(fx.q1, &R8::from_u128_coeffs(fx.q1, &delta_coeffs));
        let zero_q1 = RLWECiphertext::new(R8::zero(fx.q1), R8::zero(fx.q1));
        let mut out: Vec<RLWECiphertext<N1, R8>> = vec![zero_q1; NUM_ROWS];
        dmux_tree(&dq.dmux_bits, trivial, &mut out, b1, b1);
        out
    };
    let run_modswitch = |dmux_out: &[RLWECiphertext<N1, R8>]| {
        dmux_out
            .iter()
            .map(|ct| mod_switch_sym::<N1, R8, R8>(ct, fx.q2))
            .collect::<Vec<RLWECiphertext<N1, R8>>>()
    };
    let run_first_dim = |switched: &[RLWECiphertext<N1, R8>]| {
        let db_q2: Vec<Vec<R8>> = fx
            .encoded_db
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        let mut cc = [0u128; N1];
                        cell.to_u128_coeffs(&mut cc);
                        R8::from_u128_coeffs(fx.q2, &cc)
                    })
                    .collect()
            })
            .collect();
        first_dim::<N1, R8>(switched, &db_q2, fx.q2)
    };
    let run_cmux = |dq: &via_protocol::DecompressedQuery<N1, R8, L_QUERY>,
                    mut fd: Vec<RLWECiphertext<N1, R8>>| {
        let cmux_q2: Vec<RGSWCiphertext<N1, R8, L_QUERY, L_QUERY>> = dq
            .cmux_bits
            .iter()
            .map(|rgsw| mod_switch_rgsw::<N1, R8, R8, L_QUERY, L_QUERY>(rgsw, fx.q2))
            .collect();
        cmux_tree(&cmux_q2, &mut fd, b2, b2)
    };
    let run_crot = |dq: &via_protocol::DecompressedQuery<N1, R8, L_QUERY>,
                    selected: RLWECiphertext<N1, R8>| {
        let crot_q2: Vec<RGSWCiphertext<N1, R8, L_QUERY, L_QUERY>> = dq
            .crot_bits
            .iter()
            .map(|rgsw| mod_switch_rgsw::<N1, R8, R8, L_QUERY, L_QUERY>(rgsw, fx.q2))
            .collect();
        crot(CRotDir::SlotExtract, &crot_q2, selected, b2, b2)
    };
    let run_resp_comp = |rotated: &RLWECiphertext<N1, R8>| {
        resp_comp::<N1, N2, R8, R8, R4, R4, L_RSK, D>(
            rotated,
            &fx.pp.ring_switch_key,
            fx.q3,
            fx.q4,
            b_rsk,
        )
    };

    // ── Build the chain ONCE (untimed) to obtain each step's input. ───────
    let dq = run_qd();
    let dmux_out = run_dmux(&dq);
    let switched = run_modswitch(&dmux_out);
    let fd_results = run_first_dim(&switched);
    let selected = run_cmux(&dq, fd_results.clone());
    let rotated = run_crot(&dq, selected);
    let answer = run_resp_comp(&rotated);

    // ── Client steps ──────────────────────────────────────────────────────
    c.bench_function("toy/00_client_setup", |b| {
        b.iter_batched(
            || Shake256Prg::new(b"via-c-bench-toy-setup"),
            |mut prg| {
                black_box(ToyClient::setup(
                    fx.q1,
                    fx.q3,
                    toy_params(),
                    NUM_ROWS,
                    NUM_COLS,
                    CK_BASE,
                    Distribution::Ternary,
                    Distribution::Ternary,
                    Distribution::Ternary,
                    &mut prg,
                    gen_cascade,
                    gen_rsk_closure,
                ))
            },
            BatchSize::SmallInput,
        )
    });
    c.bench_function("toy/00_client_query", |b| {
        b.iter_batched(
            || Shake256Prg::new(b"via-c-bench-toy-query"),
            |mut prg| black_box(fx.client.query(INDEX, &mut prg)),
            BatchSize::SmallInput,
        )
    });

    // ── 7 answer steps ────────────────────────────────────────────────────
    c.bench_function("toy/01_query_decomp", |b| b.iter(|| black_box(run_qd())));
    c.bench_function("toy/02_dmux", |b| b.iter(|| black_box(run_dmux(&dq))));
    c.bench_function("toy/03_mod_switch", |b| {
        b.iter(|| black_box(run_modswitch(&dmux_out)))
    });
    c.bench_function("toy/04_first_dim", |b| {
        b.iter(|| black_box(run_first_dim(&switched)))
    });
    c.bench_function("toy/05_cmux", |b| {
        b.iter_batched(
            || fd_results.clone(),
            |fd| black_box(run_cmux(&dq, fd)),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("toy/06_crot", |b| {
        b.iter_batched(
            || selected,
            |sel| black_box(run_crot(&dq, sel)),
            BatchSize::SmallInput,
        )
    });
    c.bench_function("toy/07_resp_comp", |b| {
        b.iter(|| black_box(run_resp_comp(&rotated)))
    });

    // ── Client recover + full pipeline ────────────────────────────────────
    c.bench_function("toy/08_client_recover", |b| {
        b.iter(|| black_box(fx.client.recover::<R4, R4, R4>(&answer, fx.q3, fx.q4, fx.p)))
    });
    c.bench_function("toy/09_e2e_full", |b| {
        b.iter_batched(
            || Shake256Prg::new(b"via-c-bench-toy-e2e"),
            |mut prg| {
                let query = fx.client.query(INDEX, &mut prg).expect("client query");
                let ans = answer_one_query::<
                    N1,
                    N2,
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
                black_box(fx.client.recover::<R4, R4, R4>(&ans, fx.q3, fx.q4, fx.p))
            },
            BatchSize::SmallInput,
        )
    });
}

criterion_group!(benches, toy_benches);
criterion_main!(benches);
