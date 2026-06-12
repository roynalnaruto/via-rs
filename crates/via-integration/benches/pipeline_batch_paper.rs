//! Paper-scale VIA-B **batch** benchmarks (n2048 RNS) — the two new
//! VIA-B server costs in isolation:
//!
//! - `paper_batch/01_q2_key_build` — derive the ~11.25 MiB single-prime q2 repack
//!   key from the RNS-q1 cascade key by the boxed cross-type mod-switch
//!   (`repack_keys_poly_2048_t256_from_rns_cascade_boxed`).
//! - `paper_batch/02_repack` — `repack_poly_2048_t256` on T=256 ciphertexts @ q2.
//!
//! The per-query `answer_through_crot` (T cascade-heavy prefixes) is deliberately
//! NOT benched here — its per-step cost is already `pipeline_paper`, and T of them
//! is hours under criterion. Inputs are built once (untimed); reduced sampling.
//!
//! Run: `cargo bench -p via-integration --features via-b --bench pipeline_batch_paper`.
//! Gated on `via-b`; under default the bench binary is a no-op `main`.
#![allow(missing_docs)]

#[cfg(feature = "via-b")]
mod b {
    use criterion::{Criterion, black_box};
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::ring::rns_element::PolyRns;
    use via_primitives::algebra::rns::basis::paper::ViaCQ1Rns;
    use via_primitives::algebra::zq::modulus::ConstModulus;
    use via_primitives::algebra::zq::modulus::paper::ViaCQ2;
    use via_primitives::conversion::{
        LweToRlweKeyRnsN2048, RepackKeysPoly2048T256, gen_lwe_to_rlwe_key_rns_n2048_boxed,
        repack_keys_poly_2048_t256_from_rns_cascade_boxed, repack_poly_2048_t256,
    };
    use via_primitives::encryption::encode;
    use via_primitives::encryption::types::{RLWECiphertext, SecretKey};
    use via_primitives::sampling::distribution::Distribution;
    use via_primitives::sampling::prg::Shake256Prg;
    use via_primitives::switching::rekey::rekey_secret_key;

    const N1: usize = 2048;
    const T: usize = 256;
    const L_CK: usize = 18;
    const CK_BASE: u64 = 18;
    const VAL: u64 = 7;

    type Rq1 = PolyRns<N1, ViaCQ1Rns, Coefficient>;
    type Rq2 = Poly<N1, ViaCQ2, Coefficient>;
    type P1 = Poly<N1, ConstModulus<16>, Coefficient>;
    type K = LweToRlweKeyRnsN2048<ViaCQ1Rns, L_CK>;

    struct Fixture {
        cascade: Box<K>,
        q2_key: Box<RepackKeysPoly2048T256<ViaCQ2, L_CK>>,
        inputs: Vec<RLWECiphertext<N1, Rq2>>,
        q2: ViaCQ2,
    }

    fn build_fixture() -> Fixture {
        let basis = ViaCQ1Rns::default();
        let q2 = ViaCQ2::default();
        let p = ConstModulus::<16>;
        let mut prg = Shake256Prg::new(b"via-b-bench-batch-paper");
        let sk1 = SecretKey::<N1, Rq1>::keygen(basis, Distribution::Ternary, &mut prg);

        let cascade = gen_lwe_to_rlwe_key_rns_n2048_boxed::<ViaCQ1Rns, L_CK>(
            &sk1,
            CK_BASE,
            Distribution::Ternary,
            &mut prg,
        );
        let q2_key = repack_keys_poly_2048_t256_from_rns_cascade_boxed(&cascade, q2);

        // S1 reinterpreted at single-prime q2; T inputs @ q2 (VAL in coeff 0).
        let sk_q2 = rekey_secret_key::<N1, Rq1, Rq2>(&sk1, q2);
        let mut inputs: Vec<RLWECiphertext<N1, Rq2>> = Vec::with_capacity(T);
        for _ in 0..T {
            let mut coeffs = [0u64; N1];
            coeffs[0] = VAL;
            let m = encode::<N1, Rq2, P1>(&Poly::new(p, coeffs), q2);
            inputs.push(sk_q2.encrypt(&m, Distribution::Ternary, &mut prg));
        }

        Fixture {
            cascade,
            q2_key,
            inputs,
            q2,
        }
    }

    pub fn batch_paper_benches(c: &mut Criterion) {
        let fx = build_fixture();

        // The one new server-side key derivation: RNS-q1 cascade → single-prime
        // q2 repack key, boxed/heap (per iteration).
        c.bench_function("paper_batch/01_q2_key_build", |b| {
            b.iter(|| {
                black_box(repack_keys_poly_2048_t256_from_rns_cascade_boxed(
                    &*fx.cascade,
                    fx.q2,
                ))
            })
        });

        // The depth-10 repack at single-prime q2.
        let arr: &[_; T] = (&fx.inputs[..]).try_into().expect("T inputs");
        c.bench_function("paper_batch/02_repack", |b| {
            b.iter(|| black_box(repack_poly_2048_t256(arr, &*fx.q2_key, CK_BASE)))
        });
    }
}

#[cfg(feature = "via-b")]
criterion::criterion_group! {
    name = benches;
    config = criterion::Criterion::default().sample_size(10);
    targets = b::batch_paper_benches
}
#[cfg(feature = "via-b")]
criterion::criterion_main!(benches);

#[cfg(not(feature = "via-b"))]
fn main() {}
