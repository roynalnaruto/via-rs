//! RNS-aware SoA wrappers for primitive §0.5 — `PolyRns` analogues of
//! the four kernels in [`super::reshape`].
//!
//! Each kernel calls the corresponding single-prime kernel twice, once
//! per RNS slot. The basis `B: RnsBasis` is unused at the kernel level
//! (the maps are pure permutations independent of the moduli) but kept
//! as an argument for shape-consistency with the rest of the RNS kernel
//! family — see [`crate::algebra::rns::ops`] and
//! [`super::rns_ops`].
//!
//! All kernels enforce cross-prime length equality at the top.

use crate::algebra::rns::basis::RnsBasis;

use super::reshape;

/// $\iota_j^{n_\text{small} \to n_\text{large}}$ per RNS slot.
///
/// # Panics
///
/// Panics on cross-prime length mismatch or any of the single-prime
/// kernel panics.
pub fn embed_at_slice<B: RnsBasis>(
    _basis: B,
    src0: &[u64],
    src1: &[u64],
    dst0: &mut [u64],
    dst1: &mut [u64],
    slot: usize,
) {
    assert_eq!(
        src0.len(),
        src1.len(),
        "embed_at_slice: cross-prime src length mismatch",
    );
    assert_eq!(
        dst0.len(),
        dst1.len(),
        "embed_at_slice: cross-prime dst length mismatch",
    );
    reshape::embed_at_slice(src0, dst0, slot);
    reshape::embed_at_slice(src1, dst1, slot);
}

/// $\pi_j^{n_\text{large} \to n_\text{small}}$ per RNS slot.
pub fn project_at_slice<B: RnsBasis>(
    _basis: B,
    src0: &[u64],
    src1: &[u64],
    dst0: &mut [u64],
    dst1: &mut [u64],
    slot: usize,
) {
    assert_eq!(
        src0.len(),
        src1.len(),
        "project_at_slice: cross-prime src length mismatch",
    );
    assert_eq!(
        dst0.len(),
        dst1.len(),
        "project_at_slice: cross-prime dst length mismatch",
    );
    reshape::project_at_slice(src0, dst0, slot);
    reshape::project_at_slice(src1, dst1, slot);
}

/// $d$-fold packing $\iota^{n_\text{small} \to n_\text{large}}$ per RNS
/// slot.
pub fn pack_slots_slice<B: RnsBasis>(
    _basis: B,
    src0: &[u64],
    src1: &[u64],
    dst0: &mut [u64],
    dst1: &mut [u64],
    n_small: usize,
) {
    assert_eq!(
        src0.len(),
        src1.len(),
        "pack_slots_slice: cross-prime src length mismatch",
    );
    assert_eq!(
        dst0.len(),
        dst1.len(),
        "pack_slots_slice: cross-prime dst length mismatch",
    );
    reshape::pack_slots_slice(src0, dst0, n_small);
    reshape::pack_slots_slice(src1, dst1, n_small);
}

/// $d$-fold unpacking $\pi^{n_\text{large} \to n_\text{small}}$ per RNS
/// slot.
pub fn unpack_slots_slice<B: RnsBasis>(
    _basis: B,
    src0: &[u64],
    src1: &[u64],
    dst0: &mut [u64],
    dst1: &mut [u64],
    n_small: usize,
) {
    assert_eq!(
        src0.len(),
        src1.len(),
        "unpack_slots_slice: cross-prime src length mismatch",
    );
    assert_eq!(
        dst0.len(),
        dst1.len(),
        "unpack_slots_slice: cross-prime dst length mismatch",
    );
    reshape::unpack_slots_slice(src0, dst0, n_small);
    reshape::unpack_slots_slice(src1, dst1, n_small);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::rns::basis::{ConstRnsBasis, paper};

    type Z55 = ConstRnsBasis<5, 11>;

    #[test]
    fn rns_embed_project_roundtrip_per_slot() {
        let b = Z55::default();
        let src0 = [1u64, 2];
        let src1 = [3u64, 4];
        for j in 0..2usize {
            let mut dst0 = [0u64; 4];
            let mut dst1 = [0u64; 4];
            embed_at_slice(b, &src0, &src1, &mut dst0, &mut dst1, j);
            let mut back0 = [0u64; 2];
            let mut back1 = [0u64; 2];
            project_at_slice(b, &dst0, &dst1, &mut back0, &mut back1, j);
            assert_eq!(back0, src0, "j={j} slot0");
            assert_eq!(back1, src1, "j={j} slot1");
        }
    }

    #[test]
    fn rns_pack_unpack_identity() {
        let b = paper::ViaQ1Rns::default();
        let src0: [u64; 8] = core::array::from_fn(|i| (i as u64) + 1);
        let src1: [u64; 8] = core::array::from_fn(|i| (i as u64) * 7 + 3);
        let mut packed0 = [0u64; 8];
        let mut packed1 = [0u64; 8];
        pack_slots_slice(b, &src0, &src1, &mut packed0, &mut packed1, 2);
        let mut back0 = [0u64; 8];
        let mut back1 = [0u64; 8];
        unpack_slots_slice(b, &packed0, &packed1, &mut back0, &mut back1, 2);
        assert_eq!(back0, src0);
        assert_eq!(back1, src1);
    }

    #[test]
    #[should_panic(expected = "cross-prime")]
    fn rns_embed_at_panics_on_cross_prime_dst_mismatch() {
        let b = Z55::default();
        let src0 = [0u64; 2];
        let src1 = [0u64; 2];
        let mut dst0 = [0u64; 4];
        let mut dst1 = [0u64; 6]; // mismatched
        embed_at_slice(b, &src0, &src1, &mut dst0, &mut dst1, 0);
    }
}
