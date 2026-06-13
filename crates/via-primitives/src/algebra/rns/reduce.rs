//! Const-fn helpers — `gcd_u64` and the extended-Euclidean
//! `mod_inverse_u64`. Used by [`super::basis::ConstRnsBasis`] at compile time
//! to validate that the two moduli are coprime and to precompute the Garner
//! reconstruction inverse $(q^{(0)})^{-1} \bmod q^{(1)}$.
//!
//! These mirror the role of [`super::super::zq::reduce`] for the single-prime layer: small,
//! `const fn`-friendly primitives that the higher layers fold into immediates.
//!
//! # Why `i128` intermediates?
//!
//! The textbook extended-Euclidean recurrence keeps coefficients bounded by
//! the larger input modulus, but the per-step product $q \cdot y$ can grow up
//! to $\approx \text{modulus}^2$. For our 38-bit VIA-C / VIA-B primes that is
//! $\approx 2^{76}$ — well past `i64::MAX`. `i128` headroom is comfortable for
//! every modulus here (the largest is
//! $274\,810\,798\,081 \approx 2^{38}$, so products stay under $2^{77}$).

/// Greatest common divisor of two `u64` values via the Euclidean algorithm.
///
/// # Invariants
///
/// `gcd_u64(0, 0) == 0`. For any `a, b` with $a + b > 0$, the result is in
/// $[1, \max(a, b)]$.
///
/// `const fn` so that [`super::basis::ConstRnsBasis`] can call it from
/// `_CHECK` at monomorphisation time.
#[inline]
pub const fn gcd_u64(a: u64, b: u64) -> u64 {
    let mut x = a;
    let mut y = b;
    while y != 0 {
        let t = y;
        y = x % y;
        x = t;
    }
    x
}

/// Modular inverse via the extended Euclidean algorithm: returns the unique
/// $r \in [0, n)$ with $r \cdot x \equiv 1 \pmod{n}$, or **`0`** when
/// $\gcd(x, n) \ne 1$ (i.e. no inverse exists).
///
/// # Input handling
///
/// - `n` must be `≥ 2`. `n ∈ {0, 1}` returns the `0` sentinel.
/// - `x` may be **any** `u64` — the algorithm's first Euclidean step
///   effectively reduces `x mod n`, so callers do not need to pre-reduce.
///   Passing `x ≥ n` and passing `x % n` produce the same result.
/// - Returns `0` when `x == 0` or `gcd(x mod n, n) > 1` (no inverse exists).
///   Also returns `0` for `x == n` (the gcd test fails).
///
/// The `0` sentinel lets callers use a `const`-evaluated
/// `assert!(mod_inverse_u64(...) != 0, ...)` to fail-fast at compile time —
/// see [`super::basis::ConstRnsBasis`]'s `_CHECK` block.
///
/// # Algorithm
///
/// Standard extended Euclidean: maintain `(old_r, r)` and `(old_s, s)`
/// satisfying `old_s · x + old_t · n = old_r`. When `r == 0`, `old_r` is
/// $\gcd(x, n)$; if that equals `1`, `old_s mod n` is the inverse.
///
/// Intermediates use `i128` — see the module docs for the overflow rationale.
#[inline]
pub const fn mod_inverse_u64(x: u64, n: u64) -> u64 {
    if n < 2 {
        return 0;
    }
    if x == 0 {
        return 0;
    }
    let mut old_r: i128 = x as i128;
    let mut r: i128 = n as i128;
    let mut old_s: i128 = 1;
    let mut s: i128 = 0;
    while r != 0 {
        let q = old_r / r;
        let new_r = old_r - q * r;
        old_r = r;
        r = new_r;
        let new_s = old_s - q * s;
        old_s = s;
        s = new_s;
    }
    if old_r != 1 {
        return 0;
    }
    let n_i = n as i128;
    let mut result = old_s % n_i;
    if result < 0 {
        result += n_i;
    }
    result as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gcd_basic() {
        assert_eq!(gcd_u64(0, 0), 0);
        assert_eq!(gcd_u64(0, 7), 7);
        assert_eq!(gcd_u64(7, 0), 7);
        assert_eq!(gcd_u64(1, 1), 1);
        assert_eq!(gcd_u64(12, 18), 6);
        assert_eq!(gcd_u64(18, 12), 6);
        assert_eq!(gcd_u64(17, 13), 1); // coprime primes
    }

    #[test]
    fn gcd_paper_primes_is_one() {
        assert_eq!(gcd_u64(268369921, 536608769), 1);
        assert_eq!(gcd_u64(137438822401, 274810798081), 1);
    }

    #[test]
    fn mod_inverse_small_primes() {
        // Z_17: every i in [1, 17) has an inverse.
        for i in 1u64..17 {
            let inv = mod_inverse_u64(i, 17);
            assert!(inv > 0 && inv < 17, "inv={inv}, i={i}");
            assert_eq!((i * inv) % 17, 1, "i={i}, inv={inv}");
        }
    }

    #[test]
    fn mod_inverse_returns_zero_on_non_coprime() {
        // gcd(6, 9) = 3, so 6 has no inverse mod 9.
        assert_eq!(mod_inverse_u64(6, 9), 0);
        // gcd(4, 8) = 4.
        assert_eq!(mod_inverse_u64(4, 8), 0);
        // x = 0 has no inverse.
        assert_eq!(mod_inverse_u64(0, 17), 0);
    }

    #[test]
    fn mod_inverse_returns_zero_on_bad_modulus() {
        assert_eq!(mod_inverse_u64(1, 0), 0);
        assert_eq!(mod_inverse_u64(1, 1), 0);
    }

    #[test]
    fn mod_inverse_via_q1_primes() {
        // Garner needs (Q0 mod Q1)^{-1} mod Q1.
        let q0: u64 = 268369921;
        let q1: u64 = 536608769;
        let inv = mod_inverse_u64(q0 % q1, q1);
        assert!(inv > 0 && inv < q1);
        // Verify (q0 * inv) mod q1 == 1.
        assert_eq!(
            ((q0 as u128) * (inv as u128)) % (q1 as u128),
            1u128,
            "inv={inv}",
        );
    }

    #[test]
    fn mod_inverse_via_c_q1_primes() {
        // The 38-bit regime where Respire's i64 implementation would overflow.
        let q0: u64 = 137438822401;
        let q1: u64 = 274810798081;
        let inv = mod_inverse_u64(q0 % q1, q1);
        assert!(inv > 0 && inv < q1);
        assert_eq!(
            ((q0 as u128) * (inv as u128)) % (q1 as u128),
            1u128,
            "inv={inv}",
        );
    }

    /// Touch the const evaluator: if `mod_inverse_u64` weren't a real `const fn`
    /// this would fail to compile. Locks the const-fn contract used by
    /// [`super::basis::ConstRnsBasis::_CHECK`].
    #[test]
    fn mod_inverse_is_const_evaluable() {
        const INV: u64 = mod_inverse_u64(2, 17);
        assert_eq!((2 * INV) % 17, 1);
    }

    /// The docstring claims callers can enforce `x < n` themselves, but the
    /// implementation actually accepts `x ≥ n` and returns the inverse of
    /// `x mod n`. Pin that behaviour. Closes review item 6.
    #[test]
    fn mod_inverse_handles_x_greater_than_or_equal_n() {
        // 17 mod 11 = 6; 6^{-1} mod 11 = 2 (since 6 * 2 = 12 ≡ 1 mod 11).
        let inv = mod_inverse_u64(17, 11);
        assert_eq!(inv, 2);
        assert_eq!((17 * inv) % 11, 1);
        // Larger x: x = 3*n + r with r coprime to n.
        let inv = mod_inverse_u64(3 * 17 + 4, 17);
        // 3*17 + 4 mod 17 = 4; 4^{-1} mod 17 = 13 (4 * 13 = 52 = 3*17 + 1).
        assert_eq!(inv, 13);
    }

    /// `mod_inverse_u64(x, x)` for `x ≥ 2` must return 0 — the algorithm
    /// reaches `old_r = x`, fails the `old_r == 1` check, and falls through
    /// to the sentinel. The existing zero-return tests cover `n ∈ {0, 1}`
    /// and `gcd(x, n) > 1` cases but never `x == n`. Closes review item 7.
    #[test]
    fn mod_inverse_x_equals_n_returns_zero() {
        for n in [2u64, 3, 17, 8380417, 274_810_798_081] {
            assert_eq!(mod_inverse_u64(n, n), 0, "n={n}");
        }
    }
}
