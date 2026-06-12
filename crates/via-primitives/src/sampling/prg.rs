//! SHAKE-256 counter-mode deterministic PRG.
//!
//! Every randomized operation in the protocol — mask sampling, error sampling,
//! key generation — pulls its bytes through this PRG. The contract is:
//!
//! 1. Seed bytes are absorbed verbatim, not pre-hashed (so the seed itself is
//!    visible in the first block's input).
//! 2. Output is partitioned into **136-byte blocks** (the SHAKE-256 rate).
//! 3. Block $k$ is the first 136 bytes of
//!    $\text{SHAKE-256}(\text{seed} \,\|\, \text{u64\_le}(k))$.
//! 4. A fresh hasher is built per block (per-block re-hash); the hasher state
//!    is never reused across blocks.
//! 5. The counter is a `u64`, encoded as 8 little-endian bytes.
//!
//! Any conforming implementation that mirrors those rules must produce
//! byte-identical output for the same seed. This is the floor of the
//! cross-language test-vector reproducibility contract.
//!
//! ## Why a fixed-cap seed buffer
//!
//! The crate is `#![no_std]` and avoids `alloc`. The seed is stored inline in
//! a fixed `[u8; 64]` buffer with a `seed_len: u8` discriminator; constructing
//! a [`Shake256Prg`] from a seed longer than 64 bytes panics. 64 bytes covers
//! every seed length the spec calls for (which uses ≤ 32-byte seeds).
//!
//! ## Determinism vs. timing
//!
//! The PRG output and byte-budget are entirely a function of the seed and the
//! number of bytes drawn so far. [`Shake256Prg::uniform_below`] uses rejection
//! sampling — the number of attempts is data-dependent, but the data is
//! PRG-output, not a secret. Higher layers that need timing-independence over
//! a secret should not branch on PRG state directly.

use rand_core::{CryptoRng, RngCore};
use sha3::{
    Shake256,
    digest::{ExtendableOutput, Update, XofReader},
};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// SHAKE-256 rate, in bytes. Each squeeze block produced by the PRG is exactly
/// this many bytes.
const BLOCK_LEN: usize = 136;

/// Maximum seed length supported by [`Shake256Prg::new`].
const SEED_CAP: usize = 64;

/// Deterministic SHAKE-256 counter-mode PRG.
///
/// Conceptually an infinite byte stream parameterised by a seed. Bytes are
/// produced in 136-byte blocks via SHAKE-256, with each block keyed by the
/// concatenation of the seed and an incrementing 64-bit little-endian counter.
///
/// The struct stores its state inline (no heap allocation) and zeroes itself
/// on drop.
#[derive(Zeroize, ZeroizeOnDrop)]
pub struct Shake256Prg {
    /// Seed bytes, left-aligned. Only the first `seed_len` bytes are
    /// meaningful; the rest is uninitialised filler that we never read.
    seed: [u8; SEED_CAP],
    /// Number of meaningful seed bytes. Always ≤ `SEED_CAP`.
    seed_len: u8,
    /// Next block index to absorb. Each refill computes
    /// `SHAKE-256(seed || u64_le(counter))` and then increments.
    counter: u64,
    /// Most recently squeezed 136-byte block. Initialised to zero; the first
    /// read triggers a refill because `pos` starts at `BLOCK_LEN`.
    buffer: [u8; BLOCK_LEN],
    /// Read cursor into `buffer`. Range `0..=BLOCK_LEN`. When `pos == BLOCK_LEN`
    /// the buffer is logically empty and the next read triggers a refill.
    pos: u8,
}

impl Shake256Prg {
    /// Construct a fresh PRG seeded with `seed`.
    ///
    /// # Panics
    ///
    /// Panics if `seed.len() > 64`. The spec's test vectors all use
    /// seeds of at most 32 bytes, so the 64-byte cap is generous.
    ///
    /// # Example
    ///
    /// ```
    /// use via_primitives::sampling::Shake256Prg;
    /// let mut prg = Shake256Prg::new(b"my-seed");
    /// let mut bytes = [0u8; 32];
    /// prg.fill_bytes(&mut bytes);
    /// assert!(bytes.iter().any(|&b| b != 0));
    /// ```
    #[inline]
    pub fn new(seed: &[u8]) -> Self {
        assert!(
            seed.len() <= SEED_CAP,
            "seed length {} exceeds cap {}",
            seed.len(),
            SEED_CAP,
        );
        let mut buf = [0u8; SEED_CAP];
        buf[..seed.len()].copy_from_slice(seed);
        Self {
            seed: buf,
            seed_len: seed.len() as u8,
            counter: 0,
            buffer: [0u8; BLOCK_LEN],
            // pos == BLOCK_LEN means "buffer empty"; the first read refills.
            pos: BLOCK_LEN as u8,
        }
    }

    /// Refill `self.buffer` with the next 136-byte squeeze block.
    ///
    /// Builds a fresh `Shake256` hasher per call (per-block re-hash); the
    /// hasher is **not** reused across blocks, matching the spec's framing of
    /// each block as `SHAKE-256(seed || u64_le(counter))`.
    #[inline]
    fn refill(&mut self) {
        let mut hasher = Shake256::default();
        hasher.update(&self.seed[..self.seed_len as usize]);
        hasher.update(&self.counter.to_le_bytes());
        let mut reader = hasher.finalize_xof();
        reader.read(&mut self.buffer);
        // Wrapping not expected to occur in practice (2^64 blocks ≈ 2.5e21
        // bytes), but use wrapping_add to keep `no_std` builds panic-free
        // under release semantics where overflow would otherwise wrap silently.
        self.counter = self.counter.wrapping_add(1);
        self.pos = 0;
    }

    /// Fill `out` with the next `out.len()` pseudorandom bytes.
    ///
    /// Bytes are drawn sequentially from the internal 136-byte buffer; the
    /// buffer is refilled as needed. The byte stream is identical to the
    /// concatenation of `SHAKE-256(seed || u64_le(0)) || SHAKE-256(seed ||
    /// u64_le(1)) || …`, truncated to `out.len()` bytes from the current
    /// position.
    #[inline]
    pub fn fill_bytes(&mut self, out: &mut [u8]) {
        let mut written = 0;
        while written < out.len() {
            if self.pos as usize >= BLOCK_LEN {
                self.refill();
            }
            let buf_pos = self.pos as usize;
            let available = BLOCK_LEN - buf_pos;
            let take = core::cmp::min(out.len() - written, available);
            out[written..written + take].copy_from_slice(&self.buffer[buf_pos..buf_pos + take]);
            // `pos + take` ≤ BLOCK_LEN = 136 ≤ u8::MAX, so the cast is safe.
            self.pos = (buf_pos + take) as u8;
            written += take;
        }
    }

    /// Sample uniformly from `[0, bound)` using rejection on a power-of-two
    /// mask.
    ///
    /// Reads `ceil(bit_length(bound) / 8)` bytes from the PRG per attempt and
    /// masks down to the next power of two above (or equal to) `bound`,
    /// rejecting any draw that lands in `[bound, 2^bit_length(bound))`. This
    /// is the same procedure as `int.from_bytes(..., "little") & mask`,
    /// so the rejection trajectory — and therefore the PRG byte budget — is
    /// byte-identical.
    ///
    /// Special case: `bound == 1` returns `0` without consuming any PRG bytes,
    /// matching the early-return path.
    ///
    /// # Panics
    ///
    /// Panics if `bound == 0`.
    ///
    /// # Example
    ///
    /// ```
    /// use via_primitives::sampling::Shake256Prg;
    /// let mut prg = Shake256Prg::new(b"example");
    /// for _ in 0..16 {
    ///     let v = prg.uniform_below(17);
    ///     assert!(v < 17);
    /// }
    /// ```
    #[inline]
    pub fn uniform_below(&mut self, bound: u64) -> u64 {
        assert!(bound > 0, "bound must be positive");
        if bound == 1 {
            return 0;
        }
        // For `bound > 1`, `bits` lies in `[1, 64]`. This equals
        // `bound.bit_length()`, which is `64 - bound.leading_zeros()` here.
        let bits = 64 - bound.leading_zeros();
        let nbytes = bits.div_ceil(8) as usize;
        let mask: u64 = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        // Zero-padded little-endian buffer; only the low `nbytes` are refilled
        // per attempt, so the high bytes stay zero across iterations.
        let mut buf = [0u8; 8];
        loop {
            self.fill_bytes(&mut buf[..nbytes]);
            let value = u64::from_le_bytes(buf) & mask;
            if value < bound {
                return value;
            }
        }
    }
}

impl RngCore for Shake256Prg {
    #[inline]
    fn next_u32(&mut self) -> u32 {
        let mut b = [0u8; 4];
        Self::fill_bytes(self, &mut b);
        u32::from_le_bytes(b)
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        let mut b = [0u8; 8];
        Self::fill_bytes(self, &mut b);
        u64::from_le_bytes(b)
    }

    #[inline]
    fn fill_bytes(&mut self, dest: &mut [u8]) {
        Self::fill_bytes(self, dest);
    }

    #[inline]
    fn try_fill_bytes(&mut self, dest: &mut [u8]) -> Result<(), rand_core::Error> {
        Self::fill_bytes(self, dest);
        Ok(())
    }
}

impl CryptoRng for Shake256Prg {}

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Determinism — two PRGs with the same seed produce identical streams.
    // -----------------------------------------------------------------------

    #[test]
    fn same_seed_produces_identical_bytes() {
        let mut a = Shake256Prg::new(b"determinism-test");
        let mut b = Shake256Prg::new(b"determinism-test");
        let mut out_a = [0u8; 500];
        let mut out_b = [0u8; 500];
        a.fill_bytes(&mut out_a);
        b.fill_bytes(&mut out_b);
        assert_eq!(out_a, out_b);
    }

    #[test]
    fn different_seed_produces_different_bytes() {
        let mut a = Shake256Prg::new(b"seed-alpha");
        let mut b = Shake256Prg::new(b"seed-beta");
        let mut out_a = [0u8; 64];
        let mut out_b = [0u8; 64];
        a.fill_bytes(&mut out_a);
        b.fill_bytes(&mut out_b);
        assert_ne!(out_a, out_b);
    }

    #[test]
    fn empty_seed_is_valid() {
        let mut p = Shake256Prg::new(b"");
        let mut out = [0u8; 32];
        p.fill_bytes(&mut out);
        // Just assert non-panic and non-all-zero output (sanity).
        assert!(out.iter().any(|&b| b != 0));
    }

    // -----------------------------------------------------------------------
    // Byte-stream parity.
    //
    // The constants below are the first bytes of the concatenated
    // `SHAKE-256(seed || u64_le(ctr))` block stream, starting at counter zero.
    // Any drift here is a cross-language reproducibility failure.
    // -----------------------------------------------------------------------

    /// SHAKE-256(`b"" || u64_le(0)`) — first 32 bytes.
    const EMPTY_SEED_FIRST_32: [u8; 32] = [
        0x11, 0x91, 0x41, 0xdc, 0xe8, 0x98, 0x07, 0x09, 0x60, 0x95, 0xd9, 0x72, 0x9b, 0x0d, 0xa8,
        0x04, 0x81, 0xa4, 0x92, 0x49, 0x8e, 0x23, 0x53, 0x46, 0xef, 0xc5, 0x8a, 0xa7, 0x33, 0x35,
        0xa3, 0x51,
    ];

    /// SHAKE-256(`b"test" || u64_le(0)`) — first 32 bytes.
    const TEST_SEED_FIRST_32: [u8; 32] = [
        0xc1, 0xda, 0xed, 0x36, 0xbe, 0xbc, 0x2b, 0x1a, 0x8a, 0xb5, 0x2a, 0x7e, 0xb0, 0x6a, 0x5c,
        0xd1, 0xfe, 0xef, 0xe5, 0x9e, 0x5b, 0x09, 0xbe, 0x0e, 0x29, 0x76, 0x32, 0x14, 0x65, 0xf7,
        0x21, 0xb3,
    ];

    /// SHAKE-256(`b"test" || u64_le(0)`) bytes 134..=137 — straddles the
    /// first/second block boundary. Bytes 134, 135 belong to block 0; bytes
    /// 136, 137 are the first two bytes of SHAKE-256(`b"test" || u64_le(1)`).
    const TEST_SEED_BOUNDARY_134_138: [u8; 4] = [0xc2, 0x39, 0xe1, 0xbb];

    #[test]
    fn byte_stream_parity_empty_seed_first_32() {
        let mut prg = Shake256Prg::new(b"");
        let mut out = [0u8; 32];
        prg.fill_bytes(&mut out);
        assert_eq!(out, EMPTY_SEED_FIRST_32);
    }

    #[test]
    fn byte_stream_parity_test_seed_first_32() {
        let mut prg = Shake256Prg::new(b"test");
        let mut out = [0u8; 32];
        prg.fill_bytes(&mut out);
        assert_eq!(out, TEST_SEED_FIRST_32);
    }

    #[test]
    fn byte_stream_parity_block_boundary_134_to_137() {
        let mut prg = Shake256Prg::new(b"test");
        // Skip the first 134 bytes of block 0.
        let mut skip = [0u8; 134];
        prg.fill_bytes(&mut skip);
        // Now read 4 bytes that straddle the block-0 → block-1 boundary.
        let mut out = [0u8; 4];
        prg.fill_bytes(&mut out);
        assert_eq!(out, TEST_SEED_BOUNDARY_134_138);
    }

    #[test]
    fn split_reads_concatenate_to_one_big_read() {
        // Reading n bytes in one go should match reading the same total in
        // several smaller chunks (covers block boundaries indirectly).
        let mut p_a = Shake256Prg::new(b"split-test");
        let mut a_all = [0u8; 300];
        p_a.fill_bytes(&mut a_all);

        let mut p_b = Shake256Prg::new(b"split-test");
        let mut b_all = [0u8; 300];
        // Chunk sizes chosen to cross both the 136 and 272 boundaries.
        let chunks = [1, 134, 1, 5, 130, 1, 28];
        let mut offset = 0;
        for chunk in chunks {
            p_b.fill_bytes(&mut b_all[offset..offset + chunk]);
            offset += chunk;
        }
        // Last chunk to fill the remainder.
        p_b.fill_bytes(&mut b_all[offset..]);
        assert_eq!(a_all, b_all);
    }

    // -----------------------------------------------------------------------
    // uniform_below — edge cases and parity.
    //
    // The expected sequences below were generated by running
    // `randbelow(bound)` repeatedly on the same seed.
    // -----------------------------------------------------------------------

    /// First 8 outputs of `randbelow(3)` on seed `b"test"`.
    const TEST_SEED_RANDBELOW_3_FIRST_8: [u64; 8] = [1, 2, 1, 2, 2, 0, 2, 2];

    /// First 8 outputs of `randbelow(256)` on seed `b"test"`.
    const TEST_SEED_RANDBELOW_256_FIRST_8: [u64; 8] = [193, 237, 190, 43, 42, 176, 229, 190];

    /// First 4 outputs of `randbelow(8380417)` on seed `b"test"`
    /// (the VIA-C $q_3$ modulus, ≈ $2^{23}$).
    const TEST_SEED_RANDBELOW_Q3_FIRST_4: [u64; 4] = [7_199_425, 3_980_854, 662_059, 8_268_469];

    /// First 4 outputs of `randbelow(2**32)` on seed `b"test"`.
    /// Exercises the 33-bit mask path (bits = 33, nbytes = 5).
    const TEST_SEED_RANDBELOW_2P32_FIRST_4: [u64; 4] =
        [921_557_697, 1_789_951_530, 3_857_710_801, 3_356_614_788];

    /// First 4 outputs of `randbelow(2**32 + 1)` on seed `b"test"`.
    /// Same bit_length as `2**32`, so the byte-budget path is identical and the
    /// expected values match.
    const TEST_SEED_RANDBELOW_2P32_PLUS_1_FIRST_4: [u64; 4] =
        [921_557_697, 1_789_951_530, 3_857_710_801, 3_356_614_788];

    #[test]
    fn uniform_below_bound_1_returns_zero_without_byte_draw() {
        let mut prg = Shake256Prg::new(b"check-state");
        // First, snapshot the PRG byte position by drawing 0 bytes then 1.
        let baseline_pos = prg.pos;
        let baseline_counter = prg.counter;
        for _ in 0..100 {
            assert_eq!(prg.uniform_below(1), 0);
        }
        // No state change.
        assert_eq!(prg.pos, baseline_pos);
        assert_eq!(prg.counter, baseline_counter);
    }

    #[test]
    fn uniform_below_3_parity() {
        let mut prg = Shake256Prg::new(b"test");
        for (i, &expected) in TEST_SEED_RANDBELOW_3_FIRST_8.iter().enumerate() {
            let got = prg.uniform_below(3);
            assert_eq!(got, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn uniform_below_256_parity() {
        let mut prg = Shake256Prg::new(b"test");
        for (i, &expected) in TEST_SEED_RANDBELOW_256_FIRST_8.iter().enumerate() {
            let got = prg.uniform_below(256);
            assert_eq!(got, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn uniform_below_q3_parity() {
        let mut prg = Shake256Prg::new(b"test");
        for (i, &expected) in TEST_SEED_RANDBELOW_Q3_FIRST_4.iter().enumerate() {
            let got = prg.uniform_below(8_380_417);
            assert_eq!(got, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn uniform_below_2p32_parity() {
        let mut prg = Shake256Prg::new(b"test");
        for (i, &expected) in TEST_SEED_RANDBELOW_2P32_FIRST_4.iter().enumerate() {
            let got = prg.uniform_below(1u64 << 32);
            assert_eq!(got, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn uniform_below_2p32_plus_1_parity() {
        let mut prg = Shake256Prg::new(b"test");
        for (i, &expected) in TEST_SEED_RANDBELOW_2P32_PLUS_1_FIRST_4.iter().enumerate() {
            let got = prg.uniform_below((1u64 << 32) + 1);
            assert_eq!(got, expected, "mismatch at index {}", i);
        }
    }

    #[test]
    fn uniform_below_respects_bound() {
        let mut prg = Shake256Prg::new(b"range-test");
        for _ in 0..1000 {
            let v = prg.uniform_below(17);
            assert!(v < 17);
        }
    }

    #[test]
    fn uniform_below_large_bound_no_panic() {
        let mut prg = Shake256Prg::new(b"large-bound");
        // Bound = u64::MAX exercises the bits == 64 branch.
        let v = prg.uniform_below(u64::MAX);
        assert!(v < u64::MAX);
    }

    #[test]
    fn uniform_below_paper_q1_factor_parity() {
        // The smaller of the VIA-C q1 RNS primes: 137_438_822_401
        // (≈ 2^37). This exercises a 5-byte uniform_below path.
        let mut prg = Shake256Prg::new(b"test");
        let q = 137_438_822_401u64;
        for _ in 0..100 {
            let v = prg.uniform_below(q);
            assert!(v < q);
        }
    }

    // -----------------------------------------------------------------------
    // Construction edges.
    // -----------------------------------------------------------------------

    #[test]
    #[should_panic(expected = "exceeds cap")]
    fn new_panics_on_oversized_seed() {
        let too_long = [0u8; SEED_CAP + 1];
        let _ = Shake256Prg::new(&too_long);
    }

    #[test]
    fn new_accepts_64_byte_seed_at_cap() {
        let at_cap = [0xa5u8; SEED_CAP];
        let mut prg = Shake256Prg::new(&at_cap);
        let mut out = [0u8; 16];
        prg.fill_bytes(&mut out);
    }

    #[test]
    #[should_panic(expected = "bound must be positive")]
    fn uniform_below_panics_on_zero_bound() {
        let mut prg = Shake256Prg::new(b"");
        let _ = prg.uniform_below(0);
    }
}
