//! Fuzz: `SecretKey::keygen` + `encode` + `SecretKey::encrypt` +
//! `SecretKey::decrypt` should round-trip the plaintext for every input
//! the fuzzer can fit through a single-prime `Poly<16, DynModulus,
//! Coefficient>` carrier at the parameter ranges enumerated below.
//!
//! The targeted regression classes:
//!
//! - PRG-order regression in [`SecretKey::encrypt`] — would manifest as
//!   silent decryption failures whenever the error distribution actually
//!   produces a non-zero `e`.
//! - Off-by-one in `encode`'s $\Delta = \lceil q / p \rceil$ — would
//!   round-trip at zero-error parameters but break under any noise.
//! - Sign-handling regression in `decode`'s `i128::div_euclid` — would
//!   misbehave at exactly the negative-numerator boundary cases noise
//!   provokes.
//!
//! Single-prime backend only; the RNS path uses parallel arithmetic that
//! would benefit from its own target (deferred).
//!
//! Run with `cargo +nightly fuzz run encryption_rlwe_roundtrip`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::element::Poly;
use via_rs::algebra::ring::form::Coefficient;
use via_rs::algebra::zq::modulus::{DynModulus, Modulus};
use via_rs::encryption::{SecretKey, encode};
use via_rs::sampling::{Distribution, Shake256Prg};

/// Fixed ring degree for the fuzz target. Const-generic constraints
/// force a compile-time choice; 16 keeps iterations fast (~4 KiB poly
/// arithmetic) while still exercising negacyclic wrap.
const N: usize = 16;

/// `q` values where `q/p` is generously large for every `p` in
/// [`KNOWN_P`]. Pre-filtered so the round-trip is mathematically
/// guaranteed at every error level the fuzzer can request.
const KNOWN_Q: &[u64] = &[
    1024,
    4096,
    32768,
    65536,
    8_380_417,      // VIA-C q_3
    2_147_352_577,  // VIA q_3
    17_175_674_881, // VIA-C q_2
];

/// Plaintext moduli covering the paper's range (`p = 16` for VIA-C,
/// `p = 256` for VIA) plus the single-bit `p = 2` floor.
const KNOWN_P: &[u64] = &[2, 16, 256];

/// Sanity bound on per-coefficient post-decrypt noise — chosen to sit
/// above the worst-case `8σ` tail of any Gaussian variant in
/// [`DistChoice`], so the assertion only fires for genuinely structured
/// (i.e., buggy) noise.
const NOISE_BUDGET_OUTPUT: i64 = 64;

#[derive(Debug)]
enum DistChoice {
    Ternary,
    BoundedUniform { bound: u32 },
    GaussianSmall { sigma_index: u8 },
}

impl<'a> Arbitrary<'a> for DistChoice {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let pick: u8 = u.int_in_range(0..=2)?;
        Ok(match pick {
            0 => DistChoice::Ternary,
            1 => DistChoice::BoundedUniform {
                bound: u.int_in_range(0..=4)?,
            },
            _ => DistChoice::GaussianSmall {
                sigma_index: u.int_in_range(0..=3)?,
            },
        })
    }
}

impl DistChoice {
    fn to_distribution(&self) -> Distribution {
        match self {
            DistChoice::Ternary => Distribution::Ternary,
            DistChoice::BoundedUniform { bound } => Distribution::BoundedUniform { bound: *bound },
            DistChoice::GaussianSmall { sigma_index } => {
                let sigma = match sigma_index {
                    0 => 1.0,
                    1 => 2.0,
                    2 => 4.0,
                    _ => 6.0,
                };
                Distribution::Gaussian { sigma }
            }
        }
    }

    /// A conservative upper bound on `|e_i|` across the entire vector
    /// the distribution will produce — used as the feasibility gate for
    /// the round-trip assertion. For Ternary / BoundedUniform this is
    /// an exact maximum; for Gaussian it's the `8σ` truncation tail (an
    /// `8σ` event fires at probability `~1.2e-15` per coefficient, well
    /// below any reasonable fuzz iteration count even multiplied by
    /// `N = 16`).
    fn max_abs_coefficient(&self) -> u64 {
        match self {
            DistChoice::Ternary => 1,
            DistChoice::BoundedUniform { bound } => u64::from(*bound),
            DistChoice::GaussianSmall { sigma_index } => match sigma_index {
                0 => 8,  // σ = 1.0
                1 => 16, // σ = 2.0
                2 => 32, // σ = 4.0
                _ => 48, // σ = 6.0
            },
        }
    }
}

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    q_index: u8,
    p_index: u8,
    key_dist: DistChoice,
    error_dist: DistChoice,
    plaintext_coeffs: [u64; N],
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let sk_seed_len = u.int_in_range::<usize>(1..=32)?;
        let mut sk_seed = vec![0u8; sk_seed_len];
        u.fill_buffer(&mut sk_seed)?;

        let enc_seed_len = u.int_in_range::<usize>(1..=32)?;
        let mut enc_seed = vec![0u8; enc_seed_len];
        u.fill_buffer(&mut enc_seed)?;

        let q_index = u.int_in_range::<u8>(0..=(KNOWN_Q.len() as u8 - 1))?;
        let p_index = u.int_in_range::<u8>(0..=(KNOWN_P.len() as u8 - 1))?;
        let key_dist = DistChoice::arbitrary(u)?;
        let error_dist = DistChoice::arbitrary(u)?;

        // Pull a raw [u64; N]; the body of the fuzz target reduces each
        // lane mod p before encoding.
        let mut plaintext_coeffs = [0u64; N];
        for slot in &mut plaintext_coeffs {
            *slot = u.arbitrary()?;
        }

        Ok(Input {
            sk_seed,
            enc_seed,
            q_index,
            p_index,
            key_dist,
            error_dist,
            plaintext_coeffs,
        })
    }
}

fuzz_target!(|input: Input| {
    let q_value = KNOWN_Q[input.q_index as usize];
    let p_value = KNOWN_P[input.p_index as usize];

    // Feasibility gate: the round-trip can only recover the plaintext
    // when every error coefficient stays strictly inside
    // `Δ/2 = ⌈q / p⌉ / 2`. We require `tail < Δ/2` (i.e.,
    // `2·tail < Δ`) so that even at the truncation boundary the
    // post-decrypt rounding picks the correct multiple of `Δ`. Inputs
    // that violate this are not interesting to the fuzzer — there's
    // nothing the implementation can do to make them round-trip — so
    // we discard them up front.
    let delta = q_value.div_ceil(p_value);
    let error_tail = input.error_dist.max_abs_coefficient();
    if 2 * error_tail >= delta {
        return;
    }

    let q = DynModulus::new(q_value);
    let p = DynModulus::new(p_value);

    // Reduce each plaintext lane into `[0, p)`.
    let mut plaintext_lanes = [0u64; N];
    for (slot, &raw) in plaintext_lanes
        .iter_mut()
        .zip(input.plaintext_coeffs.iter())
    {
        *slot = raw % p_value;
    }
    let plaintext = Poly::<N, DynModulus, Coefficient>::new(p, plaintext_lanes);

    // Keygen + encode + encrypt + decrypt.
    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, Poly<N, DynModulus, Coefficient>>::keygen(
        q,
        input.key_dist.to_distribution(),
        &mut sk_prg,
    );
    let encoded: Poly<N, DynModulus, Coefficient> = encode(&plaintext, q);

    let mut enc_prg = Shake256Prg::new(&input.enc_seed);
    let ct = sk.encrypt(&encoded, input.error_dist.to_distribution(), &mut enc_prg);

    // Independent noise check: `decrypt_raw − encoded`, centred, must be
    // small. Catches an encrypt that produces structured noise (e.g., a
    // wrong-order PRG that ate plaintext bytes into the mask).
    let raw = sk.decrypt_raw(&ct);
    let mut raw_minus_encoded = [0u64; N];
    let raw_vals = raw.values();
    let enc_vals = encoded.values();
    for ((out, &r), &e) in raw_minus_encoded
        .iter_mut()
        .zip(raw_vals.iter())
        .zip(enc_vals.iter())
    {
        *out = q.sub(r, e);
    }
    let noise_poly =
        unsafe { Poly::<N, DynModulus, Coefficient>::from_reduced_unchecked(q, raw_minus_encoded) };
    let mut noise_centred = [0i64; N];
    noise_poly.to_centered_coeffs(&mut noise_centred);
    for (i, &c) in noise_centred.iter().enumerate() {
        assert!(
            c.abs() <= NOISE_BUDGET_OUTPUT,
            "noise at i={i} = {c} exceeded budget {NOISE_BUDGET_OUTPUT}; q={q_value} p={p_value}"
        );
    }

    // Full round-trip equality.
    let recovered: Poly<N, DynModulus, Coefficient> = sk.decrypt(&ct, p);
    for (i, &expected) in plaintext_lanes.iter().enumerate() {
        assert_eq!(
            recovered.coeff(i).to_u64(),
            expected,
            "round-trip diverged at i={i}; q={q_value} p={p_value}",
        );
    }
});
