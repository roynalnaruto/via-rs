//! Kernel-level micro-benchmarks for `via-primitives` (criterion).
//!
//! Complements the end-to-end pipeline benches in `via-integration` by isolating
//! the primitives the pipeline reaches only transitively. The headline measure
//! is the §0.4 NTT win on the multiplicative core: the eval-backed
//! `gadget_product` (real NTT at NTT-friendly moduli) vs a schoolbook reference,
//! at the paper ring degree `N = 2048` on both backends (single-prime `q₂`, RNS
//! `q₁`), across the gadget depths `L ∈ {2, 8, 18}` used by the paper gates. A
//! raw single-multiply comparison (schoolbook vs forward+pointwise+inverse NTT)
//! provides context.
//!
//! Run: `just bench-primitives` (or `cargo bench -p via-primitives`).
#![allow(missing_docs)] // criterion_group! generates undocumented public items

use std::time::Duration;

use criterion::measurement::WallTime;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, black_box, criterion_group, criterion_main,
};

use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::ring::rns_element::PolyRns;
use via_primitives::algebra::rns::basis::paper::ViaCQ1Rns;
use via_primitives::algebra::zq::modulus::paper::ViaCQ2;
use via_primitives::encryption::{
    RLWECiphertext, RLevCiphertext, SecretKey, gadget_extract_lsb_into, gadget_scale_into,
};
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;

const N: usize = 2048;

type SinglePoly = Poly<N, ViaCQ2, Coefficient>;
type RnsPoly = PolyRns<N, ViaCQ1Rns, Coefficient>;

/// Schoolbook reference: the pre-NTT coefficient-form gadget-product loop. The
/// production `gadget_product` (eval-backed) is benchmarked against this at the
/// **same** NTT-friendly modulus, so the timing delta is purely the NTT win.
fn schoolbook_gadget_product<const M: usize, R: RingPoly<M>, const L: usize>(
    rlev: &RLevCiphertext<M, R, L>,
    plaintext: &R,
    base: u64,
) -> RLWECiphertext<M, R> {
    let modulus = plaintext.modulus();
    let mut scratch = [0i128; M];
    gadget_scale_into::<M, R>(plaintext, base, L as u8, &mut scratch);
    let mut result_mask = R::zero(modulus);
    let mut result_body = R::zero(modulus);
    let mut digit_buf = [0i64; M];
    for k in 0..L {
        gadget_extract_lsb_into::<M>(base, &mut scratch, &mut digit_buf);
        let digit_poly = R::from_centered_i64s(modulus, &digit_buf);
        let sample = &rlev.samples[L - 1 - k];
        result_mask += digit_poly * sample.mask;
        result_body += digit_poly * sample.body;
    }
    RLWECiphertext::new(result_mask, result_body)
}

/// A deterministic small-norm (ternary-ish) polynomial — a realistic plaintext
/// / message. Values do not affect the timing of the multiply path.
fn small_centered(seed: u64) -> [i64; N] {
    let mut out = [0i64; N];
    let mut x = seed | 1;
    for v in out.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = ((x >> 61) as i64) % 3 - 1;
    }
    out
}

/// A deterministic full-width set of coefficients (each reduced mod q on build).
fn full_u128(seed: u64) -> [u128; N] {
    let mut out = [0u128; N];
    let mut x = seed | 1;
    for v in out.iter_mut() {
        x = x
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *v = u128::from(x >> 1);
    }
    out
}

fn tune(g: &mut BenchmarkGroup<'_, WallTime>) {
    // The L=18 RNS schoolbook case is ~0.7 s/call; keep the run bounded while
    // still statistically meaningful for a one-off measurement.
    g.sample_size(10);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(4));
}

fn add_single<const L: usize>(g: &mut BenchmarkGroup<'_, WallTime>, base: u64) {
    let q = ViaCQ2::default();
    let mut prg = Shake256Prg::new(b"bench-gp-single");
    let sk = SecretKey::<N, SinglePoly>::keygen(q, Distribution::Ternary, &mut prg);
    let message = <SinglePoly as RingPoly<N>>::from_centered_i64s(q, &small_centered(0xA1));
    let rlev = sk.encrypt_rlev::<L>(
        &message,
        base,
        Distribution::Gaussian { sigma: 3.2 },
        &mut prg,
    );
    let pt = <SinglePoly as RingPoly<N>>::from_centered_i64s(q, &small_centered(0xB2));

    g.bench_function(BenchmarkId::new("schoolbook", L), |b| {
        b.iter(|| {
            black_box(schoolbook_gadget_product(
                black_box(&rlev),
                black_box(&pt),
                base,
            ))
        })
    });
    g.bench_function(BenchmarkId::new("ntt", L), |b| {
        b.iter(|| black_box(rlev.gadget_product(black_box(&pt), base)))
    });
}

fn add_rns<const L: usize>(g: &mut BenchmarkGroup<'_, WallTime>, base: u64) {
    let basis = ViaCQ1Rns::default();
    let mut prg = Shake256Prg::new(b"bench-gp-rns");
    let sk = SecretKey::<N, RnsPoly>::keygen(basis, Distribution::Ternary, &mut prg);
    let message = <RnsPoly as RingPoly<N>>::from_centered_i64s(basis, &small_centered(0xC3));
    let rlev = sk.encrypt_rlev::<L>(
        &message,
        base,
        Distribution::Gaussian { sigma: 3.2 },
        &mut prg,
    );
    let pt = <RnsPoly as RingPoly<N>>::from_centered_i64s(basis, &small_centered(0xD4));

    g.bench_function(BenchmarkId::new("schoolbook", L), |b| {
        b.iter(|| {
            black_box(schoolbook_gadget_product(
                black_box(&rlev),
                black_box(&pt),
                base,
            ))
        })
    });
    g.bench_function(BenchmarkId::new("ntt", L), |b| {
        b.iter(|| black_box(rlev.gadget_product(black_box(&pt), base)))
    });
}

fn bench_gadget_product(c: &mut Criterion) {
    // Binary gadget (base = 2) across L — timing is base-independent, so a
    // uniform base isolates the effect of the depth L (= number of samples).
    {
        let mut g = c.benchmark_group("gadget_product/single_q2_n2048");
        tune(&mut g);
        add_single::<2>(&mut g, 2);
        add_single::<8>(&mut g, 2);
        add_single::<18>(&mut g, 2);
        g.finish();
    }
    {
        let mut g = c.benchmark_group("gadget_product/rns_q1_n2048");
        tune(&mut g);
        add_rns::<2>(&mut g, 2);
        add_rns::<8>(&mut g, 2);
        add_rns::<18>(&mut g, 2);
        g.finish();
    }
}

fn bench_ring_mul(c: &mut Criterion) {
    let q = ViaCQ2::default();
    let a = <SinglePoly as RingPoly<N>>::from_u128_coeffs(q, &full_u128(0x11));
    let b = <SinglePoly as RingPoly<N>>::from_u128_coeffs(q, &full_u128(0x22));

    let mut g = c.benchmark_group("ring_mul/single_q2_n2048");
    g.sample_size(30);
    g.warm_up_time(Duration::from_millis(500));
    g.measurement_time(Duration::from_secs(4));
    g.bench_function("schoolbook", |bn| {
        bn.iter(|| black_box(black_box(a) * black_box(b)))
    });
    g.bench_function("ntt_mediated", |bn| {
        bn.iter(|| {
            let ae = black_box(a).into_eval();
            let be = black_box(b).into_eval();
            black_box((ae * be).into_coeff())
        })
    });
    g.finish();
}

criterion_group!(benches, bench_gadget_product, bench_ring_mul);
criterion_main!(benches);
