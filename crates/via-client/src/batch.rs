//! §7 — VIA-B batch client helper: the recovered-answer de-interleave.
//!
//! Gated `#[cfg(feature = "via-b")]` at the [`crate`] boundary. The batch
//! actions [`Client::batch_query`](crate::Client::batch_query) and
//! [`Client::recover_batch`](crate::Client::recover_batch) live on
//! [`crate::Client`] (they need its private key material); this module holds the
//! pure de-interleave map `recover_batch` applies.
//!
//! `paper:via.pdf §4.5 (VIA-B client-side)`

use alloc::vec::Vec;
use via_primitives::algebra::ring::RingPoly;

/// De-interleave the degree-`N2` plaintext from a VIA-B batch recover into the
/// `T` record polynomials of degree `N3`.
///
/// `Repack_{n2}` interleaves the `T` records so that — after RespComp's ring
/// switch to degree `N2` — record `t`'s `N3` coefficients land at the **strided**
/// slot set `{ t + (N2/N3)·k : k ∈ [N3] }`. That is exactly `project_at::<N3>(t)`
/// (the same strided projection [`extr`](via_primitives::conversion::extr) uses),
/// so record `t` is `recovered.project_at::<N3>(t)` for `t ∈ [T]` — the inverse
/// of the repack interleave `ι_0^{N2→N1} ∘ ι^{N3→N2}` (NOT a contiguous window).
///
/// # Panics (compile-time)
///
/// `N3 < 1`, `N2 < N3`, `T < 1`, `T·N3 > N2`, or `N2 % N3 != 0`.
///
/// # Constant-time: No
///
/// Index manipulation on public data only.
///
/// `paper:via.pdf §4.5; slot map derived from the §3.4 repack interleave`
pub fn deinterleave_batch<const N2: usize, const N3: usize, const T: usize, R>(
    recovered: &R,
) -> Vec<R::Projected<N3>>
where
    R: RingPoly<N2>,
{
    const {
        assert!(N3 >= 1, "deinterleave_batch: N3 must be >= 1");
        assert!(N2 >= N3, "deinterleave_batch: N2 must be >= N3");
        assert!(T > 0, "deinterleave_batch: T must be > 0");
        assert!(
            T * N3 <= N2,
            "deinterleave_batch: T * N3 must be <= N2 (record-fit invariant)"
        );
        assert!(
            N2.is_multiple_of(N3),
            "deinterleave_batch: N2 must be divisible by N3"
        );
    }
    // Record t occupies projection slot t (stride N2/N3); `project_at` extracts it.
    (0..T).map(|t| recovered.project_at::<N3>(t)).collect()
}

#[cfg(all(test, feature = "via-b"))]
mod tests {
    use super::*;
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::zq::modulus::DynModulus;

    /// `deinterleave_batch` strided map at `N2=4, N3=2, T=2` (stride `N2/N3 = 2`):
    /// record `t = project_at::<2>(t) = [poly[t], poly[t+2]]`. So `poly [1,2,3,4]`
    /// → record0 `[1,3]`, record1 `[2,4]` — the inverse of the §3.4 repack
    /// interleave, NOT a contiguous `[1,2]/[3,4]` window.
    #[test]
    fn deinterleave_batch_strided_windows() {
        type R4 = Poly<4, DynModulus, Coefficient>;
        let p = DynModulus::new(16);
        let poly = R4::from_u128_coeffs(p, &[1, 2, 3, 4]);

        let records = deinterleave_batch::<4, 2, 2, R4>(&poly);
        assert_eq!(records.len(), 2, "T=2 records");
        assert_eq!(records[0].coeff(0).to_u64(), 1, "record0 coeff0 = poly[0]");
        assert_eq!(records[0].coeff(1).to_u64(), 3, "record0 coeff1 = poly[2]");
        assert_eq!(records[1].coeff(0).to_u64(), 2, "record1 coeff0 = poly[1]");
        assert_eq!(records[1].coeff(1).to_u64(), 4, "record1 coeff1 = poly[3]");
    }
}
