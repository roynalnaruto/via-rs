//! Fuzz: `Shake256Prg::fill_bytes` is associative across chunk boundaries.
//!
//! Reading a single `N`-byte buffer must equal reading the same `N` bytes in
//! arbitrary chunks. Catches off-by-one bugs in the buffered SHAKE-256 read
//! path at the 136-byte block boundary.
//!
//! Run with `cargo +nightly fuzz run sampling_prg_buffer_boundary`.
//!
//! Invariants verified:
//! - Two `Shake256Prg`s seeded identically produce the same `N` bytes,
//!   regardless of whether the read is one call or many.
//! - Chunk sizes in `[0, total]` are tolerated (zero-length chunks are
//!   no-ops).

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_rs::sampling::Shake256Prg;

/// Cap the seed length at the PRG's accepted maximum (64 bytes) and the
/// total read at 512 bytes; pick chunk sizes that sum to `total`.
#[derive(Debug)]
struct Input {
    seed: Vec<u8>,
    chunks: Vec<usize>,
    total: usize,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed_len = u.int_in_range::<usize>(0..=64)?;
        let mut seed = vec![0u8; seed_len];
        u.fill_buffer(&mut seed)?;

        let total = u.int_in_range::<usize>(0..=512)?;

        // Build a chunk schedule that sums to exactly `total`. We sample
        // up to 32 chunks; the last chunk soaks up the remainder.
        let max_chunks = u.int_in_range::<usize>(1..=32)?;
        let mut chunks = Vec::with_capacity(max_chunks);
        let mut remaining = total;
        for _ in 0..max_chunks - 1 {
            if remaining == 0 {
                break;
            }
            let take = u.int_in_range::<usize>(0..=remaining)?;
            chunks.push(take);
            remaining -= take;
        }
        chunks.push(remaining);

        Ok(Input {
            seed,
            chunks,
            total,
        })
    }
}

fuzz_target!(|input: Input| {
    // Reading via one big call.
    let mut prg_single = Shake256Prg::new(&input.seed);
    let mut single = vec![0u8; input.total];
    prg_single.fill_bytes(&mut single);

    // Reading via the chunk schedule.
    let mut prg_chunks = Shake256Prg::new(&input.seed);
    let mut chunked = vec![0u8; input.total];
    let mut offset = 0;
    for &take in &input.chunks {
        let end = offset + take;
        prg_chunks.fill_bytes(&mut chunked[offset..end]);
        offset = end;
    }
    assert_eq!(offset, input.total, "chunk schedule did not sum to total");

    assert_eq!(single, chunked, "split read disagrees with single read");
});
