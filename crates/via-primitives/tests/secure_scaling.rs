//! Empirical n2048 → n4096 scaling for the ≥120-bit "secure" instantiation.
//!
//! Quantifies the performance-profile change from the paper's n1=2048 VIA-C
//! parameters to the n1=4096 secure parameters (see `docs/params-bench.html`).
//! Measures (a) the production eval-backed multiplicative kernel
//! `gadget_product` at conversion depth L=18 on the q1-RNS ring, at both ring
//! degrees; and (b) the conversion cascade key sizes via `size_of`.
//!
//! Requires `--features alloc` (the cascade key types are alloc-gated). Timing
//! test is `#[ignore]`:
//! `cargo test -p via-primitives --release --features alloc --test secure_scaling -- --ignored --nocapture`
#![cfg(feature = "alloc")]

use std::time::Instant;

use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::ring::rns_element::PolyRns;
use via_primitives::algebra::rns::basis::paper::{ViaCQ1Rns, ViaSecQ1Rns};
use via_primitives::conversion::{LweToRlweKeyRnsN2048, LweToRlweKeyRnsN4096};
use via_primitives::encryption::SecretKey;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;

const L: usize = 18;
const BASE: u64 = 11; // erratum conversion base

fn small_centered<const N: usize>(seed: u64) -> [i64; N] {
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

/// Monomorphic timing for one (N, basis) pair — concrete types satisfy the
/// `NttFriendly` bound the eval-backed `gadget_product` needs.
macro_rules! time_gp {
    ($N:literal, $basis:ty, $iters:expr) => {{
        type P = PolyRns<$N, $basis, Coefficient>;
        let basis = <$basis>::default();
        let mut prg = Shake256Prg::new(b"secure-scaling");
        let sk = SecretKey::<$N, P>::keygen(basis, Distribution::Ternary, &mut prg);
        let message = <P as RingPoly<$N>>::from_centered_i64s(basis, &small_centered::<$N>(0xC3));
        let rlev = sk.encrypt_rlev::<L>(
            &message,
            BASE,
            Distribution::Gaussian { sigma: 3.2 },
            &mut prg,
        );
        let pt = <P as RingPoly<$N>>::from_centered_i64s(basis, &small_centered::<$N>(0xD4));
        for _ in 0..2 {
            std::hint::black_box(rlev.gadget_product(std::hint::black_box(&pt), BASE));
        }
        let t0 = Instant::now();
        for _ in 0..$iters {
            std::hint::black_box(rlev.gadget_product(std::hint::black_box(&pt), BASE));
        }
        t0.elapsed().as_secs_f64() / f64::from($iters as u32) * 1e6
    }};
}

#[test]
#[ignore = "timing; run with --release --features alloc -- --ignored --nocapture"]
fn gadget_product_n2048_vs_n4096() {
    // Big stack: the n4096 RLev (~2.3 MB) + gadget-product scratch overflow the
    // default test-thread stack.
    std::thread::Builder::new()
        .stack_size(256 << 20)
        .spawn(|| {
            let us_2048 = time_gp!(2048, ViaCQ1Rns, 200u32);
            let us_4096 = time_gp!(4096, ViaSecQ1Rns, 200u32);
            println!("\n=== eval-backed gadget_product (L=18, q1-RNS) ===");
            println!("  n=2048 : {us_2048:8.1} us/op");
            println!("  n=4096 : {us_4096:8.1} us/op");
            println!(
                "  ratio  : {:.2}x  (analytic n.log n = 2.18x)",
                us_4096 / us_2048
            );
        })
        .expect("spawn")
        .join()
        .expect("timing thread panicked");
}

#[test]
fn cascade_key_sizes() {
    let s2048 = core::mem::size_of::<LweToRlweKeyRnsN2048<ViaCQ1Rns, L>>();
    let s4096 = core::mem::size_of::<LweToRlweKeyRnsN4096<ViaSecQ1Rns, L>>();
    println!("\n=== LWE->RLWE conversion cascade key size (L=18) ===");
    println!("  n=2048 : {:.2} MB", s2048 as f64 / (1u64 << 20) as f64);
    println!("  n=4096 : {:.2} MB", s4096 as f64 / (1u64 << 20) as f64);
    println!("  ratio  : {:.2}x", s4096 as f64 / s2048 as f64);
    assert!(s4096 > s2048, "n4096 cascade key must exceed n2048");
}
