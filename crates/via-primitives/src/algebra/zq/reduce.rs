//! Reduction kernels for $\mathbb{Z}_q$ — primitive §0.1 of `.docs/primitives.md`.
//!
//! Two reduction algorithms cover every modulus that appears in VIA / VIA-C /
//! VIA-B at this layer:
//!
//! - **Barrett reduction** for odd-prime moduli. Pre-compute
//!   $\mu = \lfloor 2^{128} / q \rfloor$ once per modulus, then each
//!   reduction is a $128 \times 128 \to 256$ multiply (we keep only the high
//!   128 bits) followed by one conditional subtract.
//! - **Mask reduction** for power-of-two moduli $q = 2^L$: trivially
//!   $x \bmod q = x \mathbin{\\&} (q - 1)$.
//!
//! Both paths are branchless over secret data. The conditional subtracts use
//! [`subtle::ConditionallySelectable`] so an attacker who can observe timing
//! cannot learn anything about the value being reduced (only its modulus,
//! which is public).
//!
//! # Modulus range constraints
//!
//! The §0.1 layer is designed for $q < 2^{63}$ so that lazy-reduction
//! intermediates in $[0, 2q)$ still fit in `u64`. Every modulus in
//! `.docs/primitives.md` Appendix A satisfies $q \le 2^{38}$, with the
//! largest being VIA-C's $q_1$ second RNS prime $\approx 2^{38}$.

use subtle::{Choice, ConditionallySelectable};

/// Compute the Barrett constant $\mu = \lfloor 2^{128} / q \rfloor$ for a
/// non-trivial modulus $q$.
///
/// `const fn` so that [`ConstModulus`](super::modulus::ConstModulus) can
/// fold the precomputation at compile time.
///
/// # Invariants
///
/// Input: $q \in [2, 2^{63})$. Output: $\mu = \lfloor 2^{128} / q \rfloor$
/// with $\mu < 2^{127}$ (since $q \ge 2$).
///
/// # Panics
///
/// Panics in `const` evaluation if either bound is violated:
/// - `q < 2` — no residue ring ($q = 0$) or a trivial one ($q = 1$, always
///   returns 0; callers should short-circuit).
/// - `q >= 2^63` — would break the §0.1 modulus range contract (see the
///   module-level "Modulus range constraints" note): the unreduced sum
///   `a + b` in [`super::modulus::Modulus::add`] could overflow `u64`, and
///   the Barrett slack proof (`q_hat ∈ {⌊x/q⌋ − 1, ⌊x/q⌋}`) no longer
///   holds. Power-of-two moduli at $q = 2^{63}$ are still valid via
///   [`super::modulus::PowerOfTwoModulus<63>`]'s mask path, which does not
///   go through this constant.
///
/// # Algorithm
///
/// Write $2^{128} = q \cdot k + r$ with $r \in [0, q)$. We want $k$.
/// `u128::MAX / q` returns $\lfloor (2^{128} - 1) / q \rfloor$:
/// - When $r \ge 1$, this equals $k$.
/// - When $r = 0$ (only possible for $q = 2^L$), this equals $k - 1$, so we
///   add one.
#[inline]
pub const fn barrett_mu(q: u64) -> u128 {
    assert!(q >= 2, "barrett_mu requires q >= 2");
    assert!(
        q < (1u64 << 63),
        "barrett_mu requires q < 2^63 (§0.1 modulus range contract)",
    );
    let q = q as u128;
    let approx = u128::MAX / q;
    // r = 2^128 mod q, computed as ((2^128 - 1) mod q + 1) mod q.
    let r = (u128::MAX % q + 1) % q;
    if r == 0 { approx + 1 } else { approx }
}

/// Compute the high 128 bits of $x \cdot \mu$ where $x, \mu$ are `u128`.
///
/// Equivalent to $\lfloor (x \cdot \mu) / 2^{128} \rfloor$.
///
/// # Invariants
///
/// No overflow or panic for any inputs (each intermediate product is at most
/// $(2^{64} - 1)^2 < 2^{128}$).
///
/// # Algorithm
///
/// Split $x = x_1 \cdot 2^{64} + x_0$ and $\mu = m_1 \cdot 2^{64} + m_0$.
/// Then $x \cdot \mu = x_0 m_0 + (x_0 m_1 + x_1 m_0) \cdot 2^{64} + x_1 m_1 \cdot 2^{128}$.
/// We sum the four cross-products at the correct shifted positions and
/// extract bits $[128, 256)$.
#[inline]
pub const fn umul128_hi(x: u128, mu: u128) -> u128 {
    let x_lo = (x as u64) as u128;
    let x_hi = x >> 64;
    let m_lo = (mu as u64) as u128;
    let m_hi = mu >> 64;

    // Each is at most $(2^{64} - 1)^2$, fits in u128.
    let p00 = x_lo * m_lo;
    let p01 = x_lo * m_hi;
    let p10 = x_hi * m_lo;
    let p11 = x_hi * m_hi;

    // Carry into bit 128: low 64 bits of p01 and p10 align with bits 64..128
    // of p00. The sum (p00 >> 64) + (p01 lo 64) + (p10 lo 64) can occupy up
    // to 66 bits; its high 64 bits are the carry into the [128, 256) window.
    let mask64: u128 = (1u128 << 64) - 1;
    let carry = ((p00 >> 64) + (p01 & mask64) + (p10 & mask64)) >> 64;

    // High 128 bits: p11 contributes [128, 256) directly; the high 64 bits of
    // p01 and p10 contribute [128, 192); plus the carry.
    p11 + (p01 >> 64) + (p10 >> 64) + carry
}

/// Branchless Barrett reduction.
///
/// Given $x \in [0, 2^{128})$ and a modulus $q$ with precomputed
/// $\mu = \lfloor 2^{128} / q \rfloor$, returns the unique
/// $r \in [0, q)$ with $r \equiv x \pmod{q}$.
///
/// # Invariants
///
/// Output is in $[0, q)$. Requires $q \ge 2$ and $q < 2^{63}$ — the latter
/// ensures the lazy intermediate fits in `u64`.
///
/// # Constant-time
///
/// Timing is independent of $x$ and depends only on $q$ (a public parameter).
///
/// # Algorithm
///
/// Compute $\hat q = \lfloor x \mu / 2^{128} \rfloor$ via [`umul128_hi`].
/// Standard Barrett analysis shows $\hat q \in \\{\lfloor x/q \rfloor - 1, \lfloor x/q \rfloor\\}$
/// when $\mu = \lfloor 2^{128} / q \rfloor$ and $x < 2^{128}$, so the residue
/// $r = x - \hat q \cdot q$ lies in $[0, 2q)$ and a single conditional
/// subtract suffices.
#[inline]
pub fn barrett_reduce(x: u128, q: u64, mu: u128) -> u64 {
    debug_assert!(q >= 2, "barrett_reduce requires q >= 2");
    debug_assert!(q < (1u64 << 63), "barrett_reduce requires q < 2^63");
    let q_hat = umul128_hi(x, mu);
    // q_hat may exceed u64 for small q, but we only care about (q_hat * q) mod 2^128.
    let r = x.wrapping_sub(q_hat.wrapping_mul(q as u128)) as u64;
    cond_sub(r, q)
}

/// Branchless conditional subtract: returns `x - q` if `x >= q`, else `x`.
///
/// Used as the final step of Barrett reduction and as the modular addition
/// kernel: $(a + b) \bmod q$ for $a, b \in [0, q)$ first computes the unreduced
/// sum (which is in $[0, 2q)$) and then calls this helper.
///
/// # Invariants
///
/// Output is in $[0, \max(x, q))$. When the precondition $x < 2q$ holds, the
/// output is in $[0, q)$.
///
/// # Constant-time
///
/// Branchless: uses [`subtle::ConditionallySelectable`] on the borrow flag of
/// `overflowing_sub`.
#[inline]
pub fn cond_sub(x: u64, q: u64) -> u64 {
    let (diff, borrow) = x.overflowing_sub(q);
    // borrow = 1 iff x < q (we shouldn't subtract); pick `x` then.
    // borrow = 0 iff x >= q; pick `diff`.
    u64::conditional_select(&diff, &x, Choice::from(u8::from(borrow)))
}

/// Branchless conditional add: returns `x + q` if `cond` is set, else `x`.
///
/// Used as the modular subtraction kernel: $(a - b) \bmod q$ for
/// $a, b \in [0, q)$ first computes the wrapping difference, then adds $q$
/// exactly when the subtraction underflowed.
///
/// # Constant-time
///
/// Branchless under [`subtle::ConditionallySelectable`].
#[inline]
pub fn cond_add(x: u64, q: u64, cond: Choice) -> u64 {
    let sum = x.wrapping_add(q);
    u64::conditional_select(&x, &sum, cond)
}

/// Power-of-two mask reduction: $x \bmod 2^L = x \mathbin{\\&} (2^L - 1)$.
///
/// Used by [`PowerOfTwoModulus`](super::modulus::PowerOfTwoModulus) for the
/// $q_4$ and $p$ moduli in VIA / VIA-C / VIA-B.
///
/// # Invariants
///
/// Input: any `u128`. Output: in $[0, 2^L)$. Requires $L < 64$ so the
/// result fits in `u64`.
#[inline]
pub const fn mask_reduce(x: u128, log2_q: u32) -> u64 {
    debug_assert!(
        log2_q >= 1,
        "mask_reduce requires log2_q >= 1 (q = 1 is trivial)"
    );
    debug_assert!(log2_q < 64, "mask_reduce requires log2_q < 64");
    let mask = (1u128 << log2_q) - 1;
    (x & mask) as u64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference: $\mu = \lfloor 2^{128} / q \rfloor$ computed via u128 division.
    fn reference_mu(q: u64) -> u128 {
        // Cannot represent 2^128 directly; use the same construction.
        let q = q as u128;
        let approx = u128::MAX / q;
        let r = (u128::MAX % q + 1) % q;
        if r == 0 { approx + 1 } else { approx }
    }

    #[test]
    fn barrett_mu_matches_reference() {
        for &q in &[
            2u64,
            16,
            256,
            4096,
            32768,
            8380417,
            2147352577,
            17175674881,
            34359214081,
            137438822401,
            274810798081,
            268369921,
            536608769,
        ] {
            assert_eq!(barrett_mu(q), reference_mu(q), "q = {q}");
        }
    }

    #[test]
    fn umul128_hi_basic() {
        // (2^64) * (2^64) = 2^128, so high = 1.
        assert_eq!(umul128_hi(1u128 << 64, 1u128 << 64), 1);
        // (2^127) * (2^127) = 2^254, so high = 2^126.
        assert_eq!(umul128_hi(1u128 << 127, 1u128 << 127), 1u128 << 126);
        // Smaller values: 7 * 11 = 77, high = 0.
        assert_eq!(umul128_hi(7, 11), 0);
    }

    #[test]
    fn barrett_reduce_matches_mod() {
        for &q in &[3u64, 17, 257, 8380417, 274810798081] {
            let mu = barrett_mu(q);
            for x in [
                0u128,
                1,
                (q as u128) - 1,
                q as u128,
                (q as u128) * 2,
                (q as u128) * 17 + 5,
                u128::MAX / 2,
                u128::MAX,
            ] {
                let got = barrett_reduce(x, q, mu);
                let want = (x % (q as u128)) as u64;
                assert_eq!(got, want, "q = {q}, x = {x}");
            }
        }
    }

    #[test]
    fn cond_sub_branchless() {
        assert_eq!(cond_sub(10, 7), 3);
        assert_eq!(cond_sub(7, 7), 0);
        assert_eq!(cond_sub(5, 7), 5);
    }

    #[test]
    fn mask_reduce_powers_of_two() {
        assert_eq!(mask_reduce(0xDEADBEEFu128, 4), 0xF);
        assert_eq!(mask_reduce(0xDEADBEEFu128, 12), 0xEEF);
        assert_eq!(mask_reduce(u128::MAX, 32), u32::MAX as u64);
    }

    /// `barrett_mu` accepts `q` just under the §0.1 upper bound (`2^63 - 1`,
    /// the largest representable non-pow2 modulus). Pairs with the panic
    /// tests below to pin both boundaries.
    #[test]
    fn barrett_mu_accepts_just_below_bound() {
        let q = (1u64 << 63) - 1; // 2^63 - 1, non-pow2.
        let mu = barrett_mu(q);
        // Sanity-check against the same reference used elsewhere.
        assert_eq!(mu, reference_mu(q));
        // Reduction round-trip at an extreme x.
        let got = barrett_reduce(u128::MAX, q, mu);
        let want = (u128::MAX % u128::from(q)) as u64;
        assert_eq!(got, want);
    }

    /// `barrett_mu(2^63)` must panic — the slack proof for
    /// [`barrett_reduce`] (and the §0.1 `Modulus::add` contract) require
    /// `q < 2^63`. Pow2 `q = 2^63` is still valid via the mask path
    /// ([`super::modulus::PowerOfTwoModulus<63>`]), which does not call
    /// `barrett_mu`.
    #[test]
    #[should_panic(expected = "q < 2^63")]
    fn barrett_mu_panics_on_q_at_2_63() {
        let _ = barrett_mu(1u64 << 63);
    }

    /// Even one above the bound must panic — exercises the strict inequality.
    #[test]
    #[should_panic(expected = "q < 2^63")]
    fn barrett_mu_panics_on_q_above_2_63() {
        let _ = barrett_mu((1u64 << 63) | 1);
    }

    /// `umul128_hi` carry-propagation across all 256 product bits. The full-
    /// range case `(u128::MAX, u128::MAX)` is the most adversarial input — the
    /// per-lane sums hit `2^64` exactly, which must contribute `1` of carry
    /// into the high half. Closes review item 2.
    #[test]
    fn umul128_hi_full_range() {
        // (2^128 − 1)^2 = 2^256 − 2·2^128 + 1, so the high 128 bits are
        // 2^128 − 2 = u128::MAX − 1.
        assert_eq!(umul128_hi(u128::MAX, u128::MAX), u128::MAX - 1);
    }

    /// `barrett_reduce` at the upper slack boundary `q < 2^63`. The slack
    /// proof `q_hat ∈ {⌊x/q⌋ − 1, ⌊x/q⌋}` is tightest here; this pins the
    /// worst case for the §0.1 modulus range. Closes review item 3 — existing
    /// tests stopped at ~2^38 and the fuzz target generates q ≤ 2^38, so this
    /// regime was un-exercised by both unit and fuzz coverage.
    #[test]
    fn barrett_reduce_slack_boundary_near_2_63() {
        let q = (1u64 << 63) - 1; // i64::MAX — odd, non-pow2, max non-pow2 q.
        let mu = barrett_mu(q);
        let qu = u128::from(q);
        for x in [
            0u128,
            1,
            qu - 1,
            qu,
            qu + 1,
            qu * 2 - 1,
            qu * 2,
            qu * 17 + 5,
            (qu * (qu - 1)).saturating_sub(1), // near q^2
            u128::MAX / 2,
            u128::MAX - 1,
            u128::MAX,
        ] {
            let got = barrett_reduce(x, q, mu);
            let want = (x % qu) as u64;
            assert_eq!(got, want, "x={x}");
        }
    }

    /// Direct test of [`cond_add`] — the existing coverage is only via
    /// `Modulus::sub`'s borrow path. Closes review item 14.
    #[test]
    fn cond_add_branchless() {
        // No-op when the condition is 0.
        assert_eq!(cond_add(5, 7, Choice::from(0)), 5);
        // Adds when the condition is 1.
        assert_eq!(cond_add(5, 7, Choice::from(1)), 12);
        // x = 0 path.
        assert_eq!(cond_add(0, 17, Choice::from(1)), 17);
        assert_eq!(cond_add(0, 17, Choice::from(0)), 0);
        // Wrapping behaviour is intentional: x + q is computed via
        // wrapping_add. Pin that contract too.
        assert_eq!(
            cond_add(u64::MAX, 1, Choice::from(1)),
            0,
            "cond_add uses wrapping_add by design",
        );
    }
}
