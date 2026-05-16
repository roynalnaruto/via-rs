//! Fuzz: `embed_at_slice` zeroes every non-slot position of its
//! destination. The kernel writes `n_small` source values into slots
//! `{d*i + j}` and is contracted to leave every other position at zero
//! — including when the caller hands it a buffer pre-populated with
//! non-zero bytes.
//!
//! The `Poly::embed_at` wrapper always allocates a fresh
//! zero-initialised destination, so the wrapper-level fuzz target
//! `ring_embed_roundtrip` can't catch a regression that drops the
//! kernel's internal `dst.fill(0)` pass. This slice-level target
//! adversarially pre-fills `dst` so any "optimization" that skips the
//! zeroing surfaces as a contract violation.

#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;

use via_rs::algebra::ring::reshape::embed_at_slice;

// (n_small, n_large) = (4, 16), d=4. Small enough for fast fuzz
// iterations; large enough for d > 2 so the non-slot positions cover
// more than half the destination.
const N_SMALL: usize = 4;
const N_LARGE: usize = 16;

#[derive(Debug, Arbitrary)]
struct Input {
    src: [u64; N_SMALL],
    prefill: [u64; N_LARGE],
    slot: u8,
}

fuzz_target!(|input: Input| {
    let d = N_LARGE / N_SMALL;
    let slot = (input.slot as usize) % d;

    // Adversarially pre-populate dst.
    let mut dst: [u64; N_LARGE] = input.prefill;

    embed_at_slice(&input.src, &mut dst, slot);

    for (k, &v) in dst.iter().enumerate() {
        if k % d == slot {
            // Slot position: holds src[k / d].
            assert_eq!(
                v,
                input.src[k / d],
                "slot k={k}: expected src[{}]={}, got {}",
                k / d,
                input.src[k / d],
                v,
            );
        } else {
            // Non-slot position: must be zeroed even if `prefill[k]` was non-zero.
            assert_eq!(
                v, 0,
                "non-slot k={k}: pre-fill {} survived; embed_at_slice did not zero",
                input.prefill[k],
            );
        }
    }
});
