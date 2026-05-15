//! GPU-portable slice kernels for primitive §0.5 — ring embedding and
//! projection between $R_{n', q}$ and $R_{n, q}$ where $n' \mid n$.
//!
//! Four kernels implement the four maps defined in `.docs/primitives.md`
//! §0.5 (full derivation at `.docs/0.5-ring-embedding-and-projection.md`):
//!
//! - [`embed_at_slice`] — $\iota_j^{n' \to n}$, single-slot embed. Places
//!   each source coefficient at stride $d = n / n'$, offset $j$.
//! - [`project_at_slice`] — $\pi_j^{n \to n'}$, single-slot project.
//!   Reads every $d$-th source coefficient starting at $j$.
//! - [`pack_slots_slice`] — $\iota^{n' \to n}$, $d$-fold pack. Treats
//!   the input as $d$ concatenated length-$n'$ polys and interleaves
//!   them into the output. Pure permutation.
//! - [`unpack_slots_slice`] — $\pi^{n \to n'}$, $d$-fold unpack. Inverse
//!   permutation of [`pack_slots_slice`].
//!
//! ## No arithmetic
//!
//! Unlike [`crate::primitives::ring::ops`] and
//! [`crate::primitives::zq::ops`], these kernels perform **no modular
//! arithmetic** — they are pure coefficient permutations. The
//! `Modulus`-by-value argument is therefore absent. Every output lane is
//! either zero (single-slot embed only) or an exact copy of some input
//! lane.
//!
//! ## Constant-time
//!
//! The access pattern depends only on slice lengths and the slot index
//! $j$, all of which are public protocol parameters at every call site
//! (see §3.3 ring-switch key gen, §4.4 CRot slot extraction, §5.5
//! `Extr`, §6.3 `SetupDB`, §7.1/§7.4 VIA-B repacking). Coefficient
//! *values* are never branched on. The kernels are fully constant-time
//! over secret data.
//!
//! ## Length contract
//!
//! - Single-slot kernels: `src.len()` is $n_\text{small}$, `dst.len()`
//!   is $n_\text{large}$, with $n_\text{large}$ a positive multiple of
//!   $n_\text{small}$. The slot index satisfies `slot < d = n_large /
//!   n_small`.
//! - $d$-fold kernels: `src.len() == dst.len()` is $n_\text{large}$.
//!   The smaller-ring degree $n_\text{small}$ is passed as an explicit
//!   parameter and must satisfy `n_large % n_small == 0`.
//!
//! Length mismatches panic at the top of the kernel.

// ---------------------------------------------------------------------------
// Single-slot kernels — $\iota_j$ and $\pi_j$
// ---------------------------------------------------------------------------

/// $\iota_j^{n_\text{small} \to n_\text{large}}(f)$ — single-slot
/// embedding. Writes `src[i]` to `dst[d \cdot i + \text{slot}]` for
/// $i \in [0, n_\text{small})$ and zeros every other position of `dst`.
///
/// # Panics
///
/// Panics if `dst.len() < src.len()` or `dst.len() % src.len() != 0` or
/// `slot >= d` where $d = \mathrm{dst.len()} / \mathrm{src.len()}$.
pub fn embed_at_slice(src: &[u64], dst: &mut [u64], slot: usize) {
    let n_small = src.len();
    let n_large = dst.len();
    assert!(
        n_large >= n_small,
        "embed_at_slice: dst.len() < src.len() ({n_large} < {n_small})",
    );
    assert!(n_small > 0, "embed_at_slice: zero-length src is degenerate",);
    assert!(
        n_large.is_multiple_of(n_small),
        "embed_at_slice: n_large must be a multiple of n_small",
    );
    let d = n_large / n_small;
    assert!(
        slot < d,
        "embed_at_slice: slot {slot} out of range (d = {d})",
    );
    // Zero the destination first; we'll overwrite `n_small` of its
    // positions below.
    for v in dst.iter_mut() {
        *v = 0;
    }
    for (i, &v) in src.iter().enumerate() {
        dst[d * i + slot] = v;
    }
}

/// $\pi_j^{n_\text{large} \to n_\text{small}}(g)$ — single-slot
/// projection. Writes `src[d \cdot i + \text{slot}]` to `dst[i]` for
/// $i \in [0, n_\text{small})$.
///
/// # Panics
///
/// Panics if `src.len() < dst.len()` or `src.len() % dst.len() != 0` or
/// `slot >= d`.
pub fn project_at_slice(src: &[u64], dst: &mut [u64], slot: usize) {
    let n_small = dst.len();
    let n_large = src.len();
    assert!(
        n_large >= n_small,
        "project_at_slice: src.len() < dst.len() ({n_large} < {n_small})",
    );
    assert!(
        n_small > 0,
        "project_at_slice: zero-length dst is degenerate",
    );
    assert!(
        n_large.is_multiple_of(n_small),
        "project_at_slice: n_large must be a multiple of n_small",
    );
    let d = n_large / n_small;
    assert!(
        slot < d,
        "project_at_slice: slot {slot} out of range (d = {d})",
    );
    for (i, dst_slot) in dst.iter_mut().enumerate() {
        *dst_slot = src[d * i + slot];
    }
}

// ---------------------------------------------------------------------------
// $d$-fold kernels — $\iota$ and $\pi$
// ---------------------------------------------------------------------------

/// $\iota^{n_\text{small} \to n_\text{large}}$ — $d$-fold packing.
///
/// Interprets `src` as $d$ concatenated length-$n_\text{small}$ slot
/// polynomials (the $j$-th poly occupying `src[j*n_small ..
/// (j+1)*n_small]`) and writes the packed result into `dst`. The
/// permutation is `dst[d*i + j] = src[j*n_small + i]`. Both slices
/// have length $n_\text{large} = d \cdot n_\text{small}$.
///
/// # Panics
///
/// Panics if `src.len() != dst.len()` or `dst.len() % n_small != 0` or
/// `n_small == 0`.
pub fn pack_slots_slice(src: &[u64], dst: &mut [u64], n_small: usize) {
    let n_large = dst.len();
    assert_eq!(
        src.len(),
        n_large,
        "pack_slots_slice: src/dst length mismatch ({} vs {n_large})",
        src.len(),
    );
    assert!(n_small > 0, "pack_slots_slice: n_small must be positive",);
    assert!(
        n_large.is_multiple_of(n_small),
        "pack_slots_slice: n_large must be a multiple of n_small",
    );
    let d = n_large / n_small;
    for j in 0..d {
        let src_base = j * n_small;
        for i in 0..n_small {
            dst[d * i + j] = src[src_base + i];
        }
    }
}

/// $\pi^{n_\text{large} \to n_\text{small}}$ — $d$-fold unpacking.
///
/// Inverse permutation of [`pack_slots_slice`]: `dst[j*n_small + i] =
/// src[d*i + j]`. Reads the packed `src` and writes $d$ concatenated
/// slot polys into `dst`.
///
/// # Panics
///
/// Same length contract as [`pack_slots_slice`].
pub fn unpack_slots_slice(src: &[u64], dst: &mut [u64], n_small: usize) {
    let n_large = src.len();
    assert_eq!(
        dst.len(),
        n_large,
        "unpack_slots_slice: src/dst length mismatch ({n_large} vs {})",
        dst.len(),
    );
    assert!(n_small > 0, "unpack_slots_slice: n_small must be positive",);
    assert!(
        n_large.is_multiple_of(n_small),
        "unpack_slots_slice: n_large must be a multiple of n_small",
    );
    let d = n_large / n_small;
    for j in 0..d {
        let dst_base = j * n_small;
        for i in 0..n_small {
            dst[dst_base + i] = src[d * i + j];
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed embedding at $n' = 2, n = 4$. With $d = 2$ and
    /// `src = [a, b]`:
    /// - $\iota_0(\text{src}) = [a, 0, b, 0]$ (slot 0 = even positions).
    /// - $\iota_1(\text{src}) = [0, a, 0, b]$ (slot 1 = odd positions).
    #[test]
    fn embed_at_hand_computed_n_small_2_n_large_4() {
        let src = [7u64, 11];
        let mut dst = [99u64; 4]; // pre-filled to confirm zeroing
        embed_at_slice(&src, &mut dst, 0);
        assert_eq!(dst, [7, 0, 11, 0]);
        embed_at_slice(&src, &mut dst, 1);
        assert_eq!(dst, [0, 7, 0, 11]);
    }

    /// Hand-computed projection at $n = 4, n' = 2, d = 2$. With
    /// `src = [a, b, c, d]`:
    /// - $\pi_0(\text{src}) = [a, c]$ (even positions).
    /// - $\pi_1(\text{src}) = [b, d]$ (odd positions).
    #[test]
    fn project_at_hand_computed_n_large_4_n_small_2() {
        let src = [3u64, 5, 7, 11];
        let mut dst = [0u64; 2];
        project_at_slice(&src, &mut dst, 0);
        assert_eq!(dst, [3, 7]);
        project_at_slice(&src, &mut dst, 1);
        assert_eq!(dst, [5, 11]);
    }

    /// Round-trip $\pi_j \circ \iota_j = \mathrm{id}$ at $(n', n) = (2, 4)$.
    #[test]
    fn roundtrip_n_small_2_n_large_4() {
        let src = [3u64, 5];
        for j in 0..2usize {
            let mut large = [0u64; 4];
            embed_at_slice(&src, &mut large, j);
            let mut back = [0u64; 2];
            project_at_slice(&large, &mut back, j);
            assert_eq!(back, src, "j={j}");
        }
    }

    /// Round-trip at $(n', n) = (4, 16)$ covering every slot $j \in [0, 4)$.
    #[test]
    fn roundtrip_n_small_4_n_large_16() {
        let src = [3u64, 5, 11, 7];
        for j in 0..4usize {
            let mut large = [0u64; 16];
            embed_at_slice(&src, &mut large, j);
            let mut back = [0u64; 4];
            project_at_slice(&large, &mut back, j);
            assert_eq!(back, src, "j={j}");
        }
    }

    /// Round-trip at $(n', n) = (8, 32)$.
    #[test]
    fn roundtrip_n_small_8_n_large_32() {
        let src = [3u64, 5, 11, 7, 9, 1, 4, 13];
        for j in 0..4usize {
            let mut large = [0u64; 32];
            embed_at_slice(&src, &mut large, j);
            let mut back = [0u64; 8];
            project_at_slice(&large, &mut back, j);
            assert_eq!(back, src, "j={j}");
        }
    }

    /// Slot disjointness: $\pi_{j'} \circ \iota_j = 0$ for $j' \ne j$.
    #[test]
    fn project_at_other_slot_is_zero() {
        let src = [10u64, 11, 12, 13];
        let zero = [0u64; 4];
        for j in 0..4usize {
            let mut large = [0u64; 16];
            embed_at_slice(&src, &mut large, j);
            for jp in 0..4usize {
                if jp == j {
                    continue;
                }
                let mut back = [0u64; 4];
                project_at_slice(&large, &mut back, jp);
                assert_eq!(back, zero, "embed at {j}, project at {jp}");
            }
        }
    }

    /// $d$-fold round-trip at $(n', n) = (4, 16)$.
    #[test]
    fn pack_then_unpack_identity_n_small_4_n_large_16() {
        let src: [u64; 16] = core::array::from_fn(|i| (i as u64) + 1);
        let mut packed = [0u64; 16];
        pack_slots_slice(&src, &mut packed, 4);
        let mut back = [0u64; 16];
        unpack_slots_slice(&packed, &mut back, 4);
        assert_eq!(back, src);
    }

    /// $d$-fold round-trip at $(n', n) = (2, 4)$.
    #[test]
    fn pack_then_unpack_identity_n_small_2_n_large_4() {
        let src = [1u64, 2, 3, 4];
        let mut packed = [0u64; 4];
        pack_slots_slice(&src, &mut packed, 2);
        let mut back = [0u64; 4];
        unpack_slots_slice(&packed, &mut back, 2);
        assert_eq!(back, src);
    }

    /// $d$-fold round-trip at $(n', n) = (1, 8)$ — extreme: every record
    /// is a single coefficient, $d = 8$. This is the VIA-B $n_3 = 1$
    /// shape.
    #[test]
    fn pack_then_unpack_identity_n_small_1_n_large_8() {
        let src = [10u64, 20, 30, 40, 50, 60, 70, 80];
        let mut packed = [0u64; 8];
        pack_slots_slice(&src, &mut packed, 1);
        let mut back = [0u64; 8];
        unpack_slots_slice(&packed, &mut back, 1);
        assert_eq!(back, src);
        // With n_small = 1, d = 8, every input element becomes its own
        // slot. Packing is a no-op: pack[d*0 + j] = src[j*1 + 0] = src[j].
        assert_eq!(packed, src);
    }

    /// $d$-fold pack should agree with per-slot single-slot embeds
    /// summed position-wise. Slots are *disjoint by construction*
    /// (embed_j writes to positions ≡ j (mod d) and zero elsewhere),
    /// so adding the per-slot embeds reconstructs the d-fold pack
    /// without overflow risk on raw `u64`. The earlier version used
    /// bitwise OR for the same reason, but addition reads more
    /// faithfully as a disjoint-union operation and survives any
    /// future test-value change.
    #[test]
    fn pack_slots_matches_per_slot_embed() {
        // Source layout: d=4 slots of n_small=4 lanes each, into n_large=16.
        let mut concat = [0u64; 16];
        for j in 0..4usize {
            for i in 0..4usize {
                concat[j * 4 + i] = ((j * 100) + i + 1) as u64;
            }
        }
        let mut packed_via_dfold = [0u64; 16];
        pack_slots_slice(&concat, &mut packed_via_dfold, 4);

        let mut packed_via_singles = [0u64; 16];
        for j in 0..4usize {
            let mut slot_src = [0u64; 4];
            slot_src.copy_from_slice(&concat[j * 4..(j + 1) * 4]);
            let mut tmp = [0u64; 16];
            embed_at_slice(&slot_src, &mut tmp, j);
            for (acc, &v) in packed_via_singles.iter_mut().zip(tmp.iter()) {
                *acc += v;
            }
        }
        assert_eq!(packed_via_dfold, packed_via_singles);
    }

    /// Unpack via $d$-fold should agree with per-slot single-slot
    /// projects concatenated.
    #[test]
    fn unpack_slots_matches_per_slot_project() {
        let packed: [u64; 16] = core::array::from_fn(|i| (i as u64) + 1);
        let mut unpacked_via_dfold = [0u64; 16];
        unpack_slots_slice(&packed, &mut unpacked_via_dfold, 4);

        let mut unpacked_via_singles = [0u64; 16];
        for j in 0..4usize {
            let mut dst_slot = [0u64; 4];
            project_at_slice(&packed, &mut dst_slot, j);
            unpacked_via_singles[j * 4..(j + 1) * 4].copy_from_slice(&dst_slot);
        }
        assert_eq!(unpacked_via_dfold, unpacked_via_singles);
    }

    #[test]
    fn embed_at_slot_zero_index_pattern() {
        // For j = 0 with n_small=4, n_large=16, d=4:
        // dst[d*i] = src[i], dst[k % d != 0] = 0.
        let src = [3u64, 5, 11, 7];
        let mut dst = [0u64; 16];
        embed_at_slice(&src, &mut dst, 0);
        for (k, &v) in dst.iter().enumerate() {
            if k % 4 == 0 {
                assert_eq!(v, src[k / 4]);
            } else {
                assert_eq!(v, 0);
            }
        }
    }

    #[test]
    #[should_panic(expected = "dst.len() < src.len()")]
    fn embed_at_panics_on_dst_smaller_than_src() {
        let src = [0u64; 8];
        let mut dst = [0u64; 4];
        embed_at_slice(&src, &mut dst, 0);
    }

    #[test]
    #[should_panic(expected = "n_large must be a multiple")]
    fn embed_at_panics_on_non_divisible_lengths() {
        let src = [0u64; 4];
        let mut dst = [0u64; 6]; // 6 % 4 != 0
        embed_at_slice(&src, &mut dst, 0);
    }

    #[test]
    #[should_panic(expected = "slot")]
    fn embed_at_panics_on_out_of_range_slot() {
        let src = [0u64; 4];
        let mut dst = [0u64; 8]; // d = 2; slot < 2 required
        embed_at_slice(&src, &mut dst, 2);
    }

    #[test]
    #[should_panic(expected = "slot")]
    fn project_at_panics_on_out_of_range_slot() {
        let src = [0u64; 8];
        let mut dst = [0u64; 4]; // d = 2; slot < 2 required
        project_at_slice(&src, &mut dst, 2);
    }

    #[test]
    #[should_panic(expected = "length mismatch")]
    fn pack_slots_panics_on_length_mismatch() {
        let src = [0u64; 16];
        let mut dst = [0u64; 8];
        pack_slots_slice(&src, &mut dst, 4);
    }

    #[test]
    #[should_panic(expected = "n_large must be a multiple")]
    fn pack_slots_panics_on_non_divisible_lengths() {
        let src = [0u64; 12];
        let mut dst = [0u64; 12];
        pack_slots_slice(&src, &mut dst, 5); // 12 % 5 != 0
    }
}
