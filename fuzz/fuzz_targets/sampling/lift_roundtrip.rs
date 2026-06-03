//! Fuzz: `lift_centered_i64_into_zq` then `Modulus::to_centered_i64` is the
//! identity for any signed value strictly inside the centred range
//! `(-q/2, q/2]`.
//!
//! Also exercises the canonical Layer-1 → Layer-0 pipeline: sample via
//! `Distribution::sample_into`, lift through a modulus, centre back, assert
//! equal. Catches regressions in either `Modulus::reduce_i64` or the lift
//! kernel that would silently break secret-key handling.
//!
//! Run with `cargo +nightly fuzz run sampling_lift_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::zq::modulus::{DynModulus, Modulus};
use via_primitives::sampling::{Distribution, Shake256Prg, lift_centered_i64_into_zq};

const KNOWN_MODULI: &[u64] = &[
    17,
    257,
    4096,
    32768,
    8_380_417,      // VIA-C q_3
    2_147_352_577,  // VIA q_3
    17_175_674_881, // VIA-C q_2
    34_359_214_081, // VIA q_2
];

#[derive(Debug)]
enum DistChoice {
    Ternary,
    BoundedUniform { bound: u32 },
    GaussianSmall { sigma: f64 },
}

impl<'a> Arbitrary<'a> for DistChoice {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        // Distribution selection: 0 = ternary, 1 = bounded, 2 = gaussian.
        let pick: u8 = u.int_in_range(0..=2)?;
        Ok(match pick {
            0 => DistChoice::Ternary,
            1 => {
                // Bound capped well below q/2 of the smallest modulus (17/2 = 8).
                let bound = u.int_in_range::<u32>(0..=7)?;
                DistChoice::BoundedUniform { bound }
            }
            _ => {
                // Sigma small enough that all samples fit in the centred range
                // of the smallest tested modulus with overwhelming probability.
                // σ = 0.5 → 6σ tail ≈ 3 < q/2 = 8.
                let sigma_raw: u8 = u.arbitrary()?;
                let sigma = (sigma_raw as f64) / 256.0 + 0.0; // [0.0, ~1.0)
                DistChoice::GaussianSmall { sigma }
            }
        })
    }
}

impl DistChoice {
    fn into_distribution(self) -> Distribution {
        match self {
            DistChoice::Ternary => Distribution::Ternary,
            DistChoice::BoundedUniform { bound } => Distribution::BoundedUniform { bound },
            DistChoice::GaussianSmall { sigma } => Distribution::Gaussian { sigma },
        }
    }
}

#[derive(Debug)]
struct Input {
    seed: Vec<u8>,
    q_index: u8,
    dist: DistChoice,
    n: u16,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed_len = u.int_in_range::<usize>(0..=64)?;
        let mut seed = vec![0u8; seed_len];
        u.fill_buffer(&mut seed)?;
        let q_index = u.int_in_range::<u8>(0..=(KNOWN_MODULI.len() as u8 - 1))?;
        let dist = DistChoice::arbitrary(u)?;
        // Cap n so the fuzzer iterates quickly.
        let n = u.int_in_range::<u16>(0..=128)?;
        Ok(Input {
            seed,
            q_index,
            dist,
            n,
        })
    }
}

fuzz_target!(|input: Input| {
    let q = KNOWN_MODULI[input.q_index as usize];
    let m = DynModulus::new(q);
    let half_q = (q / 2) as i64;

    let mut prg = Shake256Prg::new(&input.seed);
    let n = input.n as usize;
    let mut sampled = vec![0i64; n];
    input
        .dist
        .into_distribution()
        .sample_into(&mut prg, &mut sampled);

    // Round-trip: lift, centre back, assert equal — but only for samples
    // whose magnitude fits strictly inside `(-q/2, q/2]`. For even q the
    // centred range is `[-q/2 + 1, q/2]`; values `≤ -q/2` are not fixed
    // points. Skip them for the fuzz invariant — they're expected to
    // wrap. (The chosen distributions are configured so this is rare.)
    let mut lifted = vec![0u64; n];
    lift_centered_i64_into_zq(m, &sampled, &mut lifted);
    for (&orig, &lift_u) in sampled.iter().zip(lifted.iter()) {
        assert!(lift_u < q, "lifted value {} >= q {}", lift_u, q);
        // Check the round-trip only inside the centred range.
        if orig > -half_q && orig <= half_q {
            let back = m.to_centered_i64(lift_u);
            assert_eq!(
                orig, back,
                "round-trip failed for {} under q = {} (lifted {}, back {})",
                orig, q, lift_u, back,
            );
        }
    }
});
