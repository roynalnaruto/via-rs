//! Test-only utilities shared across the crate's unit tests.
//!
//! Declared `#[cfg(test)]` in `lib.rs`, so this module is entirely absent from
//! the shipping build and never touches the `no_std` / no-alloc surface. It
//! hosts helpers that would otherwise be copy-pasted across several
//! `#[cfg(test)] mod tests` blocks.

use rand_core::RngCore;

/// SplitMix64 — a small, well-characterised PRG used in the uniformity tests.
/// Avoids pulling in `rand_chacha` or similar as a dev-dependency for a handful
/// of tests. Previously duplicated verbatim in the `zq`, `rns`, and `ring`
/// element test modules; consolidated here once the third caller appeared.
pub(crate) struct SplitMix64(u64);

impl SplitMix64 {
    pub(crate) fn new(seed: u64) -> Self {
        Self(seed)
    }
}

impl RngCore for SplitMix64 {
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
    fn fill_bytes(&mut self, dst: &mut [u8]) {
        for chunk in dst.chunks_mut(8) {
            let bytes = self.next_u64().to_le_bytes();
            chunk.copy_from_slice(&bytes[..chunk.len()]);
        }
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), rand_core::Error> {
        self.fill_bytes(dst);
        Ok(())
    }
}
