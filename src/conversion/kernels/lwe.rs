//! GPU-portable constant-time kernel for the §5.1 LWE body dot product.
//!
//! POD by value + flat slices (the Layer-0 kernel shape; see
//! [`crate::algebra::zq::ops`]); the same body lowers to a CUDA / Metal
//! reduction. The single-prime orchestrator calls [`dot_residues`] once; the
//! RNS orchestrator calls it once per prime residue lane (mirroring how the
//! §0.5 RNS reshape wrappers fan a slice kernel over both primes).
//!
//! # Constant-time: Yes (over the key)
//!
//! The mask residues `a` are RLWE-uniform (public); the key residues `s` are
//! secret. [`dot_residues`] runs a **data-independent** loop — always
//! `a.len()` multiply-accumulates, no early exit and no secret-indexed branch —
//! so it leaks nothing about `s` through control flow (the only data-dependent
//! cost is the hardware `%`, the standard caveat shared with every reduce in
//! [`crate::switching::kernels`]).

/// $\bigl(\sum_i a_i \cdot s_i\bigr) \bmod q$ over flat residue slices — one
/// single-prime modulus, or one RNS prime's residue lane.
///
/// Both `a` (the LWE mask scalars, RLWE-uniform) and `s` (the secret-key
/// coefficient vector) are passed as canonical residues in $[0, q)$, exactly as
/// the Python reference computes the body
/// (`pir/primitives/mlwe.py:149-153`, `sk_coeffs` taken mod $q$).
///
/// The running accumulator is reduced mod $q$ **each step**, so with
/// $q < 2^{64}$ every intermediate stays below $q^2 + q < 2^{128}$ and `u128`
/// never overflows regardless of `a.len()`.
///
/// # Panics
///
/// if `a.len() != s.len()`.
#[inline]
pub fn dot_residues(a: &[u64], s: &[u64], q: u64) -> u64 {
    assert!(
        a.len() == s.len(),
        "dot_residues: a.len() ({}) != s.len() ({})",
        a.len(),
        s.len(),
    );
    let q = u128::from(q);
    let mut acc: u128 = 0;
    for (&ai, &si) in a.iter().zip(s.iter()) {
        acc = (acc + u128::from(ai) * u128::from(si)) % q;
    }
    acc as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Naïve reference: full-width sum then a single reduction.
    fn reference(a: &[u64], s: &[u64], q: u64) -> u64 {
        let mut acc: u128 = 0;
        for (&ai, &si) in a.iter().zip(s.iter()) {
            acc += u128::from(ai) * u128::from(si);
        }
        (acc % u128::from(q)) as u64
    }

    #[test]
    fn dot_residues_matches_reference() {
        let q = 65537u64;
        let a = [12345u64, 65536, 0, 1, 40000, 7, 65535, 9001];
        let s = [1u64, 65536, 12, 0, 3, 65535, 2, 5]; // ternary residues incl. q-1 = -1
        assert_eq!(dot_residues(&a, &s, q), reference(&a, &s, q));
    }

    #[test]
    fn dot_residues_zero_key_is_zero() {
        let q = 65537u64;
        let a = [1u64, 2, 3, 4];
        let s = [0u64; 4];
        assert_eq!(dot_residues(&a, &s, q), 0);
    }

    #[test]
    fn dot_residues_q_minus_one_is_negation() {
        // s = (q-1) ≡ -1, so Σ a_i·s_i ≡ -Σ a_i (mod q).
        let q = 97u64;
        let a = [10u64, 20, 30];
        let s = [q - 1, q - 1, q - 1];
        let want = (3 * q - (10 + 20 + 30)) % q; // -(60) mod 97
        assert_eq!(dot_residues(&a, &s, q), want);
    }

    #[test]
    #[should_panic(expected = "a.len()")]
    fn dot_residues_length_mismatch_panics() {
        let _ = dot_residues(&[1, 2], &[1], 7);
    }
}
