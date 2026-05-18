//! Fuzz: `Shake256Prg::uniform_below(bound)` always returns a value in
//! `[0, bound)` and never panics for `bound > 0`.
//!
//! Run with `cargo +nightly fuzz run sampling_uniform_below_range`.
//!
//! Invariants verified:
//! - `bound == 1` returns `0` (special-cased; pinned by Phase 1).
//! - For `bound > 1`, every emitted value is `< bound`.
//! - Repeated draws against the same bound do not panic (no infinite-loop
//!   regressions in the rejection path).
//! - The `bits == 64` branch (when `bound > 2^63`) does not panic.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::sampling::Shake256Prg;

#[derive(Debug)]
struct Input {
    seed: Vec<u8>,
    bound: u64,
    n_draws: u32,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed_len = u.int_in_range::<usize>(0..=64)?;
        let mut seed = vec![0u8; seed_len];
        u.fill_buffer(&mut seed)?;

        // Avoid bound == 0 (would panic per spec). Bound covers the full
        // u64 range so we exercise the bits == 64 path.
        let bound = u.int_in_range::<u64>(1..=u64::MAX)?;

        // Cap draws so the fuzz target finishes promptly even when bound
        // is tiny and rejection probability is high.
        let n_draws = u.int_in_range::<u32>(0..=256)?;

        Ok(Input {
            seed,
            bound,
            n_draws,
        })
    }
}

fuzz_target!(|input: Input| {
    let mut prg = Shake256Prg::new(&input.seed);
    for _ in 0..input.n_draws {
        let v = prg.uniform_below(input.bound);
        assert!(v < input.bound, "{} >= bound {}", v, input.bound);
        if input.bound == 1 {
            assert_eq!(v, 0);
        }
    }
});
