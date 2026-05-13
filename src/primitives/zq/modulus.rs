//! The [`Modulus`] trait and its three concrete implementations.
//!
//! [`Modulus`] is the contract that the higher §0.x primitives use to talk to
//! $\mathbb{Z}_q$. The trait surface is small — two required methods
//! ([`Modulus::q`], [`Modulus::reduce_u128`]) plus seven provided methods
//! that derive from them — and value-typed (`Copy + Send + Sync + 'static`)
//! so that a [`Modulus`] can be passed through a function boundary the same
//! way a CUDA kernel receives its `__constant__` arguments — no virtual
//! dispatch, no heap.
//!
//! ## Three implementations
//!
//! - [`ConstModulus<Q>`] — zero-sized, compile-time modulus, Barrett
//!   constants computed by `const fn` at monomorphization time. Use for the
//!   paper's parameter sets (see [`paper`]).
//! - [`PowerOfTwoModulus<LOG2_Q>`] — zero-sized, compile-time
//!   $q = 2^{\text{LOG2\\_Q}}$, reduction is a single mask. Use for $q_4$ and
//!   the plaintext modulus $p$.
//! - [`DynModulus`] — runtime modulus carrying its precomputed Barrett
//!   constants. Use for tests, toy parameters, and JSON-driven test vectors.

use subtle::{Choice, ConditionallySelectable};

use super::reduce::{barrett_mu, barrett_reduce, cond_add, cond_sub, mask_reduce};

/// Abstract behaviour shared by every modulus type at §0.1.
///
/// All methods are constant-time over secret data — they may branch on the
/// modulus value (which is a public parameter of the scheme) but never on the
/// inputs $a, b, x$.
///
/// # Range invariants
///
/// Unless otherwise stated, every method takes inputs in $[0, q)$ and produces
/// outputs in $[0, q)$. The two exceptions are [`Modulus::reduce_u128`] (which
/// accepts any `u128`) and [`Modulus::reduce_u64`] / [`Modulus::reduce_i64`]
/// (which accept any value of the underlying integer type).
///
/// # Performance contract
///
/// Implementations should mark every method `#[inline(always)]`. For
/// [`ConstModulus`] and [`PowerOfTwoModulus`] the compiler is then expected to
/// fold the modulus and the Barrett constant into immediate operands; the
/// generated code matches a hand-tuned single-modulus impl.
pub trait Modulus: Copy + Eq + Send + Sync + 'static {
    /// Returns the modulus $q$.
    fn q(self) -> u64;

    /// Reduces an unsigned 128-bit value modulo $q$.
    ///
    /// Returns the unique $r \in [0, q)$ with $r \equiv x \pmod{q}$.
    fn reduce_u128(self, x: u128) -> u64;

    /// Modular addition $(a + b) \bmod q$.
    ///
    /// # Invariants
    ///
    /// Inputs: $a, b \in [0, q)$. Output: $\in [0, q)$.
    ///
    /// Requires $q < 2^{63}$ so that the unreduced sum fits in `u64`.
    #[inline(always)]
    fn add(self, a: u64, b: u64) -> u64 {
        cond_sub(a.wrapping_add(b), self.q())
    }

    /// Modular subtraction $(a - b) \bmod q$.
    ///
    /// # Invariants
    ///
    /// Inputs: $a, b \in [0, q)$. Output: $\in [0, q)$.
    #[inline(always)]
    fn sub(self, a: u64, b: u64) -> u64 {
        let (diff, borrow) = a.overflowing_sub(b);
        cond_add(diff, self.q(), Choice::from(u8::from(borrow)))
    }

    /// Modular negation $(-a) \bmod q$.
    ///
    /// # Invariants
    ///
    /// Input: $a \in [0, q)$. Output: $\in [0, q)$. Maps $0 \mapsto 0$.
    #[inline(always)]
    fn neg(self, a: u64) -> u64 {
        // Branchless: q - a if a != 0, else 0. Use overflowing_sub and select.
        let (diff, _borrow) = self.q().overflowing_sub(a);
        let is_zero = Choice::from(u8::from(a == 0));
        u64::conditional_select(&diff, &0u64, is_zero)
    }

    /// Modular multiplication $(a \cdot b) \bmod q$.
    ///
    /// # Invariants
    ///
    /// Inputs: $a, b \in [0, q)$. Output: $\in [0, q)$. The intermediate
    /// $a \cdot b$ is computed in `u128`, so no overflow occurs for any
    /// $q \le 2^{64} - 1$.
    #[inline(always)]
    fn mul(self, a: u64, b: u64) -> u64 {
        self.reduce_u128(u128::from(a) * u128::from(b))
    }

    /// Reduce an arbitrary `u64` value into $[0, q)$.
    #[inline(always)]
    fn reduce_u64(self, x: u64) -> u64 {
        // Use reduce_u128 so we cover x in [q, 2^64) uniformly.
        self.reduce_u128(u128::from(x))
    }

    /// Reduce an arbitrary `i64` value into $[0, q)$.
    ///
    /// Used by the ternary / bounded-uniform / discrete-Gaussian samplers
    /// (§1.x), which produce small signed integers that must be lifted into
    /// $\mathbb{Z}_q$.
    #[inline(always)]
    fn reduce_i64(self, x: i64) -> u64 {
        // `i64::unsigned_abs()` correctly handles `i64::MIN` (returns `2^63`).
        let magnitude = self.reduce_u64(x.unsigned_abs());
        if x >= 0 {
            magnitude
        } else {
            self.neg(magnitude)
        }
    }

    /// Centered representation $\tilde a \in (-q/2, q/2]$ with
    /// $\tilde a \equiv a \pmod q$ — see `.docs/primitives.md` §0.6.
    ///
    /// # Invariants
    ///
    /// Input: $a \in [0, q)$. Output: $\in (-\lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$.
    /// Specifically, returns $a$ when $a \le \lfloor q/2 \rfloor$, else $a - q$.
    ///
    /// # Not constant-time
    ///
    /// The comparison `a > q/2` branches on a value derived from the input.
    /// Callers handling secret data should not use this helper directly;
    /// it is intended for decoding boundaries (paper §2.2 `Dec`, §3.1
    /// `ModSwitch`) where the value is about to be revealed.
    #[inline(always)]
    fn to_centered_i64(self, a: u64) -> i64 {
        let q = self.q();
        if a <= q / 2 {
            a as i64
        } else {
            -((q - a) as i64)
        }
    }
}

/// Compile-time modulus: $q$ is a `const` generic parameter.
///
/// Zero-sized — instances exist only to carry the generic parameter into trait
/// dispatch. The associated [`ConstModulus::MU`] is the Barrett constant
/// $\mu = \lfloor 2^{128} / Q \rfloor$, evaluated at compile time.
///
/// # Example
///
/// ```rust
/// use via_rs::primitives::zq::modulus::{ConstModulus, Modulus};
/// let m = ConstModulus::<17>;
/// assert_eq!(m.add(10, 12), 5); // (10 + 12) mod 17 = 22 mod 17 = 5
/// assert_eq!(m.mul(5, 7), 1);   // (5 * 7) mod 17 = 35 mod 17 = 1
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct ConstModulus<const Q: u64>;

impl<const Q: u64> ConstModulus<Q> {
    /// Precomputed Barrett constant for $Q$ — see [`barrett_mu`].
    ///
    /// Compile-time evaluated; const-folds into immediate operands at every
    /// call site.
    pub const MU: u128 = barrett_mu(Q);

    /// The modulus value (also available via [`Modulus::q`]).
    pub const Q: u64 = Q;
}

impl<const Q: u64> Modulus for ConstModulus<Q> {
    #[inline(always)]
    fn q(self) -> u64 {
        Q
    }

    #[inline(always)]
    fn reduce_u128(self, x: u128) -> u64 {
        barrett_reduce(x, Q, Self::MU)
    }
}

/// Compile-time power-of-two modulus: $q = 2^{\text{LOG2\\_Q}}$.
///
/// Zero-sized. Reduction is a single bitwise AND with $(2^{\text{LOG2\\_Q}} - 1)$;
/// no Barrett constants needed.
///
/// Used for $q_4 \in \\{2^{12}, 2^{15}\\}$ and $p \in \\{16, 256\\}$ in the
/// realistic VIA / VIA-C / VIA-B parameter sets.
///
/// # Example
///
/// ```rust
/// use via_rs::primitives::zq::modulus::{PowerOfTwoModulus, Modulus};
/// let m = PowerOfTwoModulus::<4>; // q = 16
/// assert_eq!(m.q(), 16);
/// assert_eq!(m.add(10, 12), 6); // (10 + 12) mod 16
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct PowerOfTwoModulus<const LOG2_Q: u32>;

impl<const LOG2_Q: u32> PowerOfTwoModulus<LOG2_Q> {
    /// Compile-time guard: `LOG2_Q` must be in `[1, 64)` so that `1u64 << LOG2_Q`
    /// neither overflows `u64` nor produces the degenerate trivial ring
    /// $\mathbb{Z}_1$. Forced to evaluate at monomorphisation; an invalid
    /// `LOG2_Q` fails the compile.
    const _CHECK: () = {
        assert!(LOG2_Q >= 1, "PowerOfTwoModulus requires LOG2_Q >= 1");
        assert!(LOG2_Q < 64, "PowerOfTwoModulus requires LOG2_Q < 64");
    };

    /// The modulus value $2^{\text{LOG2\\_Q}}$.
    pub const Q: u64 = {
        // Touch _CHECK to force its evaluation when Q is referenced.
        let () = Self::_CHECK;
        1u64 << LOG2_Q
    };
    /// Reduction mask $2^{\text{LOG2\\_Q}} - 1$.
    pub const MASK: u64 = Self::Q - 1;
}

impl<const LOG2_Q: u32> Modulus for PowerOfTwoModulus<LOG2_Q> {
    #[inline(always)]
    fn q(self) -> u64 {
        Self::Q
    }

    #[inline(always)]
    fn reduce_u128(self, x: u128) -> u64 {
        mask_reduce(x, LOG2_Q)
    }

    #[inline(always)]
    fn add(self, a: u64, b: u64) -> u64 {
        // Wrap-around plus mask is enough; no conditional subtract needed.
        a.wrapping_add(b) & Self::MASK
    }

    #[inline(always)]
    fn sub(self, a: u64, b: u64) -> u64 {
        a.wrapping_sub(b) & Self::MASK
    }

    #[inline(always)]
    fn neg(self, a: u64) -> u64 {
        a.wrapping_neg() & Self::MASK
    }

    #[inline(always)]
    fn reduce_u64(self, x: u64) -> u64 {
        x & Self::MASK
    }
}

/// Runtime modulus: carries the modulus value alongside its precomputed
/// Barrett constants.
///
/// Use this when the modulus is only known at runtime — driven by parsed
/// JSON, paper-quoted toy parameters, or fuzz-target inputs. For paper-pinned
/// production paths, prefer [`ConstModulus`] (zero overhead).
///
/// `DynModulus::new` detects power-of-two moduli and switches to mask
/// reduction internally — equivalent in result to [`PowerOfTwoModulus`].
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct DynModulus {
    q: u64,
    mu: u128,
    /// `Some(L)` when `q = 2^L`; selects the mask reduction path.
    log2_q_if_pow2: Option<u32>,
}

impl DynModulus {
    /// Build a runtime modulus, precomputing Barrett constants if needed.
    ///
    /// # Panics
    ///
    /// Panics if `q < 2`.
    #[inline]
    pub const fn new(q: u64) -> Self {
        assert!(q >= 2, "DynModulus requires q >= 2");
        if q.is_power_of_two() {
            Self {
                q,
                mu: 0, // unused; reduce_u128 routes to mask path
                log2_q_if_pow2: Some(q.trailing_zeros()),
            }
        } else {
            Self {
                q,
                mu: barrett_mu(q),
                log2_q_if_pow2: None,
            }
        }
    }
}

impl Modulus for DynModulus {
    #[inline(always)]
    fn q(self) -> u64 {
        self.q
    }

    #[inline(always)]
    fn reduce_u128(self, x: u128) -> u64 {
        match self.log2_q_if_pow2 {
            Some(log2) => mask_reduce(x, log2),
            None => barrett_reduce(x, self.q, self.mu),
        }
    }
}

/// Compile-time markers for every modulus that appears in
/// `.docs/primitives.md` Appendix A.
///
/// These let production code carry the modulus *in its type*, so the
/// monomorphic call sites get full inlining and const-folding of the Barrett
/// constants. Each marker is a [`ConstModulus`] or [`PowerOfTwoModulus`] with
/// the canonical paper value baked in.
pub mod paper {
    use super::{ConstModulus, PowerOfTwoModulus};

    /// VIA $q_1$ first RNS prime, $268369921 \approx 2^{28}$. NTT-friendly
    /// for $n_1 = 2048$ (satisfies $q \equiv 1 \pmod{2 n_1}$).
    pub type ViaQ1P0 = ConstModulus<268369921>;
    /// VIA $q_1$ second RNS prime, $536608769 \approx 2^{29}$. NTT-friendly.
    pub type ViaQ1P1 = ConstModulus<536608769>;
    /// VIA $q_2 = 34359214081 \approx 2^{35}$.
    pub type ViaQ2 = ConstModulus<34359214081>;
    /// VIA $q_3 = 2147352577 \approx 2^{31}$.
    pub type ViaQ3 = ConstModulus<2147352577>;
    /// VIA $q_4 = 2^{15}$.
    pub type ViaQ4 = PowerOfTwoModulus<15>;
    /// VIA plaintext modulus $p = 256 = 2^8$.
    pub type ViaP = PowerOfTwoModulus<8>;

    /// VIA-C / VIA-B $q_1$ first RNS prime, $137438822401 \approx 2^{37}$.
    pub type ViaCQ1P0 = ConstModulus<137438822401>;
    /// VIA-C / VIA-B $q_1$ second RNS prime, $274810798081 \approx 2^{38}$.
    pub type ViaCQ1P1 = ConstModulus<274810798081>;
    /// VIA-C / VIA-B $q_2 = 17175674881 \approx 2^{34}$.
    pub type ViaCQ2 = ConstModulus<17175674881>;
    /// VIA-C / VIA-B $q_3 = 8380417 \approx 2^{23}$.
    pub type ViaCQ3 = ConstModulus<8380417>;
    /// VIA-C / VIA-B $q_4 = 2^{12}$.
    pub type ViaCQ4 = PowerOfTwoModulus<12>;
    /// VIA-C / VIA-B plaintext modulus $p = 16 = 2^4$.
    pub type ViaCP = PowerOfTwoModulus<4>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn const_modulus_basic_ops() {
        let m = ConstModulus::<17>;
        assert_eq!(m.q(), 17);
        assert_eq!(m.add(10, 12), 5);
        assert_eq!(m.sub(3, 10), 10);
        assert_eq!(m.mul(5, 7), 1);
        assert_eq!(m.neg(5), 12);
        assert_eq!(m.neg(0), 0);
    }

    #[test]
    fn pow2_modulus_basic_ops() {
        let m = PowerOfTwoModulus::<4>; // q = 16
        assert_eq!(m.q(), 16);
        assert_eq!(m.add(10, 12), 6);
        assert_eq!(m.sub(3, 10), 9);
        assert_eq!(m.mul(5, 7), 3); // 35 mod 16
        assert_eq!(m.neg(5), 11);
        assert_eq!(m.neg(0), 0);
    }

    #[test]
    fn dyn_matches_const_for_prime() {
        let c = ConstModulus::<8380417>;
        let d = DynModulus::new(8380417);
        for (a, b) in [
            (0u64, 0u64),
            (1, 1),
            (8380416, 1),
            (12345, 67890),
            (1 << 22, 1 << 22),
        ] {
            assert_eq!(c.add(a, b), d.add(a, b));
            assert_eq!(c.sub(a, b), d.sub(a, b));
            assert_eq!(c.mul(a, b), d.mul(a, b));
        }
    }

    #[test]
    fn dyn_matches_pow2_for_power_of_two() {
        let pow = PowerOfTwoModulus::<12>; // q = 4096
        let dyn_ = DynModulus::new(4096);
        for (a, b) in [(0u64, 0u64), (1, 1), (4095, 1), (1234, 567), (2048, 2048)] {
            assert_eq!(pow.add(a, b), dyn_.add(a, b));
            assert_eq!(pow.sub(a, b), dyn_.sub(a, b));
            assert_eq!(pow.mul(a, b), dyn_.mul(a, b));
        }
    }

    #[test]
    fn reduce_i64_handles_negatives() {
        let m = ConstModulus::<17>;
        assert_eq!(m.reduce_i64(0), 0);
        assert_eq!(m.reduce_i64(3), 3);
        assert_eq!(m.reduce_i64(-3), 14); // 17 - 3
        assert_eq!(m.reduce_i64(20), 3); // 20 mod 17
        assert_eq!(m.reduce_i64(-20), 14); // -(20 mod 17) mod 17 = -3 mod 17 = 14
    }

    #[test]
    fn to_centered_i64_range() {
        let m = ConstModulus::<17>;
        assert_eq!(m.to_centered_i64(0), 0);
        assert_eq!(m.to_centered_i64(8), 8);
        assert_eq!(m.to_centered_i64(9), -8); // 9 - 17
        assert_eq!(m.to_centered_i64(16), -1);
    }

    #[test]
    fn paper_moduli_values() {
        assert_eq!(paper::ViaCQ3::Q, 8380417);
        assert_eq!(paper::ViaCQ1P1::Q, 274810798081);
        assert_eq!(paper::ViaCQ4::Q, 4096);
        assert_eq!(paper::ViaCP::Q, 16);
    }
}
