//! GPU-portable coefficient-level kernels for modulus switching.
//!
//! Following the low-level kernel-shape convention (see
//! [`crate::algebra::zq::ops`]): every kernel takes plain-old-data constants
//! by value plus flat slices, so the same body lowers to a CUDA / Metal
//! thread-grid launch. [`RescaleConsts`] is `Copy` and holds the three
//! precomputed values a rescale loop needs (`q_src`, `q_dst`, `q_src/2`),
//! mirroring how a kernel argument would carry them.
//!
//! # Constant-time: No
//!
//! The rescale arithmetic is applied to RLWE-uniform ciphertext coefficients,
//! which leak nothing about secrets through timing under the RLWE assumption.
//! These kernels are *not* constant-time over their inputs and must not be used
//! on secret-key material; the rekeying path has its own constant-time kernels
//! in [`super::rekey`].

use crate::algebra::zq::modulus::Modulus;

/// Precomputed constants for a single `q_src → q_dst` coefficient rescale.
///
/// Holds `q_src`, `q_dst`, and `q_src / 2` (the rounding bias) as `u128` so
/// the multiply-add-divide fits the widest paper modulus. `Copy` so it can be
/// passed by value into every kernel / orchestrator without re-deriving the
/// half.
#[derive(Copy, Clone, Debug)]
pub struct RescaleConsts {
    /// Source modulus $q_\text{src}$.
    pub q_src: u128,
    /// Destination modulus $q_\text{dst}$.
    pub q_dst: u128,
    /// Precomputed rounding bias $\lfloor q_\text{src} / 2 \rfloor$.
    pub q_src_half: u128,
}

impl RescaleConsts {
    /// Build the constants for a `q_src → q_dst` rescale. `const fn` so paper
    /// parameter sets can precompute it at compile time.
    pub const fn new(q_src: u128, q_dst: u128) -> Self {
        Self {
            q_src,
            q_dst,
            q_src_half: q_src / 2,
        }
    }

    /// Round-to-nearest rescale of a single coefficient:
    /// $\lfloor c \cdot q_\text{dst} / q_\text{src} \rceil$, computed with
    /// integer arithmetic (round-half-up via the `q_src_half` bias). The
    /// result lies in $[0, q_\text{dst}]$ — the upper endpoint is reduced by
    /// the caller's `from_u128_coeffs` / `reduce_u128`.
    ///
    /// # Overflow safety
    ///
    /// At paper params $c < q_\text{src} \le 2^{75}$ and
    /// $q_\text{dst} \le 2^{35}$, so the product is $\le 2^{110}$ and fits in
    /// `u128` with margin.
    #[inline]
    pub fn scale(&self, c: u128) -> u128 {
        (c * self.q_dst + self.q_src_half) / self.q_src
    }
}

/// Rescale a slice of canonical `u128` source coefficients into a `u64`
/// destination slice reduced modulo `dst_modulus`. Coefficient-parallel: each
/// lane is independent, matching the GPU thread-grid shape.
///
/// # Panics
///
/// If `src.len() != dst.len()` (a caller logic bug — the ring layer enforces
/// equal lengths).
///
/// # Constant-time: No
///
/// Routes through [`Modulus::reduce_u128`] (Barrett; data-dependent only on
/// public modulus) applied to RLWE-uniform coefficients — see the module-doc.
/// See `src/algebra/zq/modulus.rs:151-175` for the reduction's timing notes.
#[inline]
pub fn rescale_slice_u128<M: Modulus>(
    consts: RescaleConsts,
    src: &[u128],
    dst_modulus: M,
    dst: &mut [u64],
) {
    assert_eq!(
        src.len(),
        dst.len(),
        "rescale_slice_u128: src/dst length mismatch"
    );
    for (d, &c) in dst.iter_mut().zip(src) {
        *d = dst_modulus.reduce_u128(consts.scale(c));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::zq::modulus::ConstModulus;

    #[test]
    fn new_initializes_fields() {
        let c = RescaleConsts::new(100, 10);
        assert_eq!(c.q_src, 100);
        assert_eq!(c.q_dst, 10);
        assert_eq!(c.q_src_half, 50);
    }

    #[test]
    fn scale_zero_maps_to_zero() {
        let c = RescaleConsts::new(100, 10);
        assert_eq!(c.scale(0), 0);
    }

    #[test]
    fn scale_halfway_rounds_up() {
        // q_src = 100, q_dst = 10. c = 5 → 5*10/100 = 0.5 → rounds to 1.
        let c = RescaleConsts::new(100, 10);
        assert_eq!(c.scale(5), 1);
    }

    #[test]
    fn scale_top_rounds_to_q_dst() {
        // c = q_src - 1 = 99 → (99*10 + 50)/100 = 1040/100 = 10 = q_dst.
        let c = RescaleConsts::new(100, 10);
        assert_eq!(c.scale(99), 10);
    }

    #[test]
    fn scale_large_modulus_overflow_safe() {
        // q_src ≈ 2^75, q_dst ≈ 2^35, c just below q_src: product ≈ 2^110.
        let q_src: u128 = 1 << 75;
        let q_dst: u128 = 1 << 35;
        let c = RescaleConsts::new(q_src, q_dst);
        let coeff = q_src - 1;
        // Expected ≈ round(coeff * q_dst / q_src) = q_dst (top rounds up).
        assert_eq!(c.scale(coeff), q_dst);
    }

    #[test]
    fn rescale_consts_is_copy() {
        fn takes_copy<T: Copy>(_: T) {}
        let c = RescaleConsts::new(8, 4);
        takes_copy(c);
        // Still usable after the copy.
        assert_eq!(c.q_src, 8);
    }

    #[test]
    fn rescale_slice_smoke() {
        // q_src = 16, q_dst = 4. Coeffs [0, 8, 15] → [0, 2, 4 mod 4 = 0].
        let consts = RescaleConsts::new(16, 4);
        let src = [0u128, 8, 15];
        let mut dst = [0u64; 3];
        rescale_slice_u128(consts, &src, ConstModulus::<4>, &mut dst);
        // 0 → 0; 8 → (32+8)/16 = 2; 15 → (60+8)/16 = 4 → 0 mod 4.
        assert_eq!(dst, [0, 2, 0]);
    }

    #[test]
    #[should_panic(expected = "src/dst length mismatch")]
    fn rescale_slice_length_mismatch() {
        let consts = RescaleConsts::new(16, 4);
        let src = [0u128, 8];
        let mut dst = [0u64; 3];
        rescale_slice_u128(consts, &src, ConstModulus::<4>, &mut dst);
    }
}
