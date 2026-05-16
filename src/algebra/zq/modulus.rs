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
    /// Inputs: $a, b \in [0, q)$. Output: $\in [0, q)$. The precondition is
    /// `debug_assert!`'d — release builds trust the caller. Use
    /// [`Modulus::reduce_u64`] first if the inputs are unreduced.
    ///
    /// Requires $q < 2^{63}$ so that the unreduced sum fits in `u64`.
    #[inline(always)]
    fn add(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q(), "Modulus::add: a >= q");
        debug_assert!(b < self.q(), "Modulus::add: b >= q");
        cond_sub(a.wrapping_add(b), self.q())
    }

    /// Modular subtraction $(a - b) \bmod q$.
    ///
    /// # Invariants
    ///
    /// Inputs: $a, b \in [0, q)$. Output: $\in [0, q)$. Precondition
    /// `debug_assert!`'d.
    #[inline(always)]
    fn sub(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q(), "Modulus::sub: a >= q");
        debug_assert!(b < self.q(), "Modulus::sub: b >= q");
        let (diff, borrow) = a.overflowing_sub(b);
        cond_add(diff, self.q(), Choice::from(u8::from(borrow)))
    }

    /// Modular negation $(-a) \bmod q$.
    ///
    /// # Invariants
    ///
    /// Input: $a \in [0, q)$. Output: $\in [0, q)$. Maps $0 \mapsto 0$.
    /// Precondition `debug_assert!`'d.
    #[inline(always)]
    fn neg(self, a: u64) -> u64 {
        debug_assert!(a < self.q(), "Modulus::neg: a >= q");
        // Branchless: q - a if a != 0, else 0. Use overflowing_sub and select.
        let (diff, _borrow) = self.q().overflowing_sub(a);
        let is_zero = Choice::from(u8::from(a == 0));
        u64::conditional_select(&diff, &0u64, is_zero)
    }

    /// Modular multiplication $(a \cdot b) \bmod q$.
    ///
    /// # Invariants
    ///
    /// Inputs: $a, b \in [0, q)$. Output: $\in [0, q)$. Precondition
    /// `debug_assert!`'d. The intermediate $a \cdot b$ is computed in
    /// `u128`, so no overflow occurs for any $q \le 2^{64} - 1$.
    #[inline(always)]
    fn mul(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < self.q(), "Modulus::mul: a >= q");
        debug_assert!(b < self.q(), "Modulus::mul: b >= q");
        self.reduce_u128(u128::from(a) * u128::from(b))
    }

    /// Reduce an arbitrary `u64` value into $[0, q)$.
    ///
    /// Inputs in `[q, 2^64)` are explicitly accepted — samplers (§1.x) and
    /// raw-bytes-to-Zq paths pass unreduced u64s. The default routes through
    /// [`Modulus::reduce_u128`] (Barrett), which is correct for the full
    /// u64 range under the §0.1 `q < 2^63` contract; the
    /// [`PowerOfTwoModulus`] specialisation collapses to an unconditional
    /// mask. Distinct from [`Modulus::add`] / `sub` / `mul`, which
    /// `debug_assert!` that their inputs are already reduced.
    #[inline(always)]
    fn reduce_u64(self, x: u64) -> u64 {
        self.reduce_u128(u128::from(x))
    }

    /// Reduce an arbitrary `i64` value into $[0, q)$.
    ///
    /// Used by the ternary / bounded-uniform / discrete-Gaussian samplers
    /// (§1.x), which produce small signed integers that must be lifted into
    /// $\mathbb{Z}_q$. The *sign* of those samples is secret data (it
    /// determines a coefficient of the secret key or error polynomial), so
    /// this lift must not leak it through timing.
    ///
    /// # Constant-time
    ///
    /// Constant-time over `x`: both branches (magnitude and its negation
    /// modulo `q`) are computed unconditionally and the result is selected
    /// via [`subtle::ConditionallySelectable`]. `i64::unsigned_abs` is
    /// itself branchless in `std` and correctly maps `i64::MIN` to `2^63`.
    #[inline(always)]
    fn reduce_i64(self, x: i64) -> u64 {
        let magnitude = self.reduce_u64(x.unsigned_abs());
        let neg = self.neg(magnitude);
        let is_negative = Choice::from(x.is_negative() as u8);
        u64::conditional_select(&magnitude, &neg, is_negative)
    }

    /// Centered representation $\tilde a \in (-q/2, q/2]$ with
    /// $\tilde a \equiv a \pmod q$ — see `.docs/primitives.md` §0.6.
    ///
    /// # Invariants
    ///
    /// Input: $a \in [0, q)$ (`debug_assert!`'d). Output:
    /// $\in (-\lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$.
    /// Specifically, returns $a$ when $a \le \lfloor q/2 \rfloor$, else $a - q$.
    ///
    /// # Not constant-time
    ///
    /// The comparison `a > q/2` branches on a value derived from the input.
    /// Callers handling secret data should not use this helper directly;
    /// it is intended for decoding boundaries (paper §2.2 `Dec`, §3.1
    /// `ModSwitch`) where the value is about to be revealed. For
    /// secret-key coefficient centring (§3.4 rekeying), use the
    /// constant-time companion [`Self::to_centered_i64_ct`].
    #[inline(always)]
    fn to_centered_i64(self, a: u64) -> i64 {
        debug_assert!(a < self.q(), "Modulus::to_centered_i64: a >= q");
        let q = self.q();
        if a <= q / 2 {
            a as i64
        } else {
            -((q - a) as i64)
        }
    }

    /// Constant-time variant of [`Self::to_centered_i64`].
    ///
    /// Same output as [`Self::to_centered_i64`]; the difference is only
    /// the timing behaviour. Branchless: both candidate outputs (the
    /// positive `a as i64` and the negative `-((q - a) as i64)`) are
    /// computed unconditionally, and the choice is made via a
    /// [`subtle::ConditionallySelectable`] cmov driven by
    /// [`subtle::ConstantTimeGreater`] on `u64`.
    ///
    /// # Invariants
    ///
    /// Input: $a \in [0, q)$ (`debug_assert!`'d). Output:
    /// $\in (-\lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$.
    ///
    /// # Constant-time
    ///
    /// CT over the input value $a$. The access pattern depends only
    /// on the public parameter $q$. Use this whenever centring a
    /// **secret** coefficient — specifically the §3.4 secret-key
    /// rekeying step, where $S$'s small non-uniform coefficients
    /// would otherwise leak Hamming-weight information through
    /// timing/side-channels.
    #[inline(always)]
    fn to_centered_i64_ct(self, a: u64) -> i64 {
        use subtle::{ConditionallySelectable, ConstantTimeGreater};
        debug_assert!(a < self.q(), "Modulus::to_centered_i64_ct: a >= q");
        let q = self.q();
        let half = q / 2;
        // Compute both branches unconditionally.
        let pos = a as i64;
        let neg = -((q - a) as i64);
        let is_negative = a.ct_gt(&half);
        i64::conditional_select(&pos, &neg, is_negative)
    }
}

/// Compile-time modulus: $q$ is a `const` generic parameter.
///
/// Zero-sized — instances exist only to carry the generic parameter into trait
/// dispatch. The associated [`ConstModulus::MU`] is the Barrett constant
/// $\mu = \lfloor 2^{128} / Q \rfloor$, evaluated at compile time.
///
/// # Compile-time invariants
///
/// `Q ∈ [2, 2^63)`. For $Q = 2^{63}$ use [`PowerOfTwoModulus<63>`] instead —
/// the mask reduction path does not need the Barrett slack and is correct
/// at that boundary. Violations fail at monomorphisation via [`Self::_CHECK`],
/// which is reached from every trait method ([`Modulus::q`] for the add /
/// sub / neg path, [`Self::MU`] → [`barrett_mu`] for mul / reduce).
///
/// # Pow2 `Q`: prefer [`PowerOfTwoModulus`]
///
/// `ConstModulus<{1u64 << L}>` *compiles* — `_CHECK` is satisfied — but it
/// silently uses Barrett reduction even though a single mask AND would
/// suffice. Use [`PowerOfTwoModulus<L>`] for power-of-two moduli; the
/// paper aliases already route correctly (`ViaQ4 = PowerOfTwoModulus<15>`,
/// `ViaCP = PowerOfTwoModulus<4>`, etc.).
///
/// # Example
///
/// ```rust
/// use via_rs::algebra::zq::modulus::{ConstModulus, Modulus};
/// let m = ConstModulus::<17>;
/// assert_eq!(m.add(10, 12), 5); // (10 + 12) mod 17 = 22 mod 17 = 5
/// assert_eq!(m.mul(5, 7), 1);   // (5 * 7) mod 17 = 35 mod 17 = 1
/// ```
///
/// # Compile-time rejection of out-of-range `Q`
///
/// ```compile_fail
/// use via_rs::algebra::zq::modulus::{ConstModulus, Modulus};
/// // Q = 2^63 violates the §0.1 modulus range contract; barrett_mu refuses
/// // to const-evaluate, so any use of `ConstModulus::<{1u64 << 63}>::MU`
/// // fails to compile. Use `PowerOfTwoModulus<63>` instead.
/// const _: u128 = ConstModulus::<{ 1u64 << 63 }>::MU;
/// ```
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct ConstModulus<const Q: u64>;

impl<const Q: u64> ConstModulus<Q> {
    /// Compile-time validation block — see the struct docs. Touched by
    /// [`Modulus::q`] (for the add / sub / neg path) and by [`Self::MU`]
    /// (for the mul / reduce path, indirectly via [`barrett_mu`]) so every
    /// monomorphised use of `ConstModulus<Q>` fires the range check.
    const _CHECK: () = {
        assert!(Q >= 2, "ConstModulus: Q >= 2");
        assert!(
            Q < 1u64 << 63,
            "ConstModulus: Q < 2^63 (§0.1 modulus range contract); for Q = 2^63 use PowerOfTwoModulus<63>",
        );
    };

    /// Precomputed Barrett constant for $Q$ — see [`barrett_mu`].
    ///
    /// Compile-time evaluated; const-folds into immediate operands at every
    /// call site. Touches [`Self::_CHECK`] so the range invariants fire even
    /// for trait paths that never reach [`barrett_mu`].
    pub const MU: u128 = {
        let () = Self::_CHECK;
        barrett_mu(Q)
    };

    /// The modulus value (also available via [`Modulus::q`]).
    pub const Q: u64 = Q;
}

impl<const Q: u64> Modulus for ConstModulus<Q> {
    #[inline(always)]
    fn q(self) -> u64 {
        // Force `_CHECK` evaluation at monomorphisation so the add / sub /
        // neg path (which never touches `Self::MU`) still validates `Q`.
        let () = Self::_CHECK;
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
/// use via_rs::algebra::zq::modulus::{PowerOfTwoModulus, Modulus};
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
        debug_assert!(a < Self::Q, "PowerOfTwoModulus::add: a >= q");
        debug_assert!(b < Self::Q, "PowerOfTwoModulus::add: b >= q");
        // Wrap-around plus mask is enough; no conditional subtract needed.
        a.wrapping_add(b) & Self::MASK
    }

    #[inline(always)]
    fn sub(self, a: u64, b: u64) -> u64 {
        debug_assert!(a < Self::Q, "PowerOfTwoModulus::sub: a >= q");
        debug_assert!(b < Self::Q, "PowerOfTwoModulus::sub: b >= q");
        a.wrapping_sub(b) & Self::MASK
    }

    #[inline(always)]
    fn neg(self, a: u64) -> u64 {
        debug_assert!(a < Self::Q, "PowerOfTwoModulus::neg: a >= q");
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

    /// `reduce_i64(i64::MIN)` exercises the `unsigned_abs → neg` path at its
    /// most adversarial input — `i64::MIN.unsigned_abs() == 2^63`, one bit
    /// above the largest representable signed magnitude. Closes the
    /// `.docs/review.md` item 8 gap (and verifies the constant-time
    /// rewrite preserves value semantics on the boundary).
    #[test]
    fn reduce_i64_min_extreme() {
        // Several moduli to triangulate: a small prime, a paper $q_3$, and
        // a pow2 (which routes through the mask `reduce_u64` override).
        for q in [17u64, 8380417, 4096] {
            let m = DynModulus::new(q);
            let got = m.reduce_i64(i64::MIN);
            let want = i64::MIN.rem_euclid(q as i64) as u64;
            assert_eq!(got, want, "q = {q}");
        }
        // Pin the well-known case: 2^63 mod 17 = 9, so -2^63 mod 17 = 8.
        assert_eq!(ConstModulus::<17>.reduce_i64(i64::MIN), 8);
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

    /// Pow2 `q = 2^63` is valid: the mask path doesn't need Barrett slack.
    /// Documents the boundary policy paired with the `barrett_mu` panic
    /// tests in `reduce.rs`.
    #[test]
    fn pow2_modulus_at_log63_boundary() {
        let m = PowerOfTwoModulus::<63>;
        assert_eq!(m.q(), 1u64 << 63);
        // Worst-case add: (q-1) + (q-1) = 2q - 2; for q = 2^63 the unreduced
        // sum is 2^64 - 2, which fits in u64 without wrapping. Mask to q-1.
        let a = (1u64 << 63) - 1;
        assert_eq!(m.add(a, a), (1u64 << 63) - 2);
        assert_eq!(m.sub(0, 1), (1u64 << 63) - 1);
        assert_eq!(m.mul(a, 2), (1u64 << 63) - 2); // ((2^63 - 1) * 2) mod 2^63
        assert_eq!(m.neg(1), (1u64 << 63) - 1);
    }

    /// `DynModulus::new` at the pow2 boundary `q = 2^63` succeeds and agrees
    /// with [`PowerOfTwoModulus<63>`]. Confirms the early-return branch is
    /// not over-fenced.
    #[test]
    fn dyn_modulus_pow2_at_2_63_boundary_works() {
        let pow = PowerOfTwoModulus::<63>;
        let dyn_ = DynModulus::new(1u64 << 63);
        assert_eq!(dyn_.q(), 1u64 << 63);
        for (a, b) in [
            (0u64, 0u64),
            (1, 1),
            ((1u64 << 63) - 1, 1),
            ((1u64 << 63) - 1, (1u64 << 63) - 1),
            (1u64 << 62, 1u64 << 62),
        ] {
            assert_eq!(pow.add(a, b), dyn_.add(a, b));
            assert_eq!(pow.sub(a, b), dyn_.sub(a, b));
            assert_eq!(pow.mul(a, b), dyn_.mul(a, b));
        }
    }

    /// `DynModulus::new` for a non-pow2 modulus at-or-above `2^63` must
    /// reject at construction — without this the non-pow2 Barrett path
    /// would silently break the §0.1 add / mul correctness.
    #[test]
    #[should_panic(expected = "q < 2^63")]
    fn dyn_modulus_panics_on_non_pow2_q_at_2_63() {
        let _ = DynModulus::new((1u64 << 63) | 1);
    }

    /// `to_centered_i64` at the smallest legal modulus `q = 2`: residues
    /// `{0, 1}` centre to `{0, 1}` (the (-1, 1] interval includes 1). Closes
    /// review item 9 (lower boundary).
    #[test]
    fn to_centered_i64_q2_minimum() {
        let m = DynModulus::new(2);
        assert_eq!(m.to_centered_i64(0), 0);
        assert_eq!(m.to_centered_i64(1), 1);
    }

    /// `to_centered_i64` at an even q (`q = 256`, the VIA plaintext modulus).
    /// The boundary case: `a = 128 = q/2` maps to `128` (≤ q/2 path), and
    /// `a = 129` maps to `129 - 256 = -127`. Closes review item 9 (even-q
    /// boundary).
    #[test]
    fn to_centered_i64_even_q_boundary() {
        let m = DynModulus::new(256);
        assert_eq!(m.to_centered_i64(0), 0);
        assert_eq!(m.to_centered_i64(128), 128);
        assert_eq!(m.to_centered_i64(129), -127);
        assert_eq!(m.to_centered_i64(255), -1);
    }

    /// `PowerOfTwoModulus<1>` is the smallest legal pow2 modulus (`q = 2`).
    /// Closes review item 12 (lower boundary; the upper boundary `<63>` is
    /// pinned by `pow2_modulus_at_log63_boundary` above).
    #[test]
    fn pow2_modulus_at_log1_boundary() {
        let m = PowerOfTwoModulus::<1>;
        assert_eq!(m.q(), 2);
        // Z_2 arithmetic: addition is XOR.
        assert_eq!(m.add(0, 0), 0);
        assert_eq!(m.add(0, 1), 1);
        assert_eq!(m.add(1, 1), 0);
        assert_eq!(m.sub(0, 1), 1);
        assert_eq!(m.mul(1, 1), 1);
        assert_eq!(m.neg(0), 0);
        assert_eq!(m.neg(1), 1);
    }

    /// `PowerOfTwoModulus::mul` uses the default `Modulus::mul` body, which
    /// routes through `reduce_u128` → `mask_reduce` for the pow2
    /// specialisation. The existing tests only exercise this indirectly via
    /// `dyn_matches_pow2_for_power_of_two`. Closes review item 13.
    #[test]
    fn pow2_modulus_mul_direct() {
        let m = PowerOfTwoModulus::<12>; // q = 4096 (VIA-C q_4).
        // (q - 1) * (q - 1) = q^2 - 2q + 1 ≡ 1 (mod q).
        assert_eq!(m.mul(4095, 4095), 1);
        // 5 * 1000 = 5000 mod 4096 = 904.
        assert_eq!(m.mul(5, 1000), 904);
        // Both zero is zero.
        assert_eq!(m.mul(0, 4095), 0);
    }

    /// Direct test of `Modulus::sub`'s borrow path (`a < b`) at the modulus
    /// level. Existing coverage only goes through `Zq::sub`. Closes review
    /// item 15.
    #[test]
    fn modulus_sub_borrow_path() {
        let m = DynModulus::new(17);
        assert_eq!(m.sub(3, 10), 10); // (3 - 10) mod 17 = -7 mod 17 = 10
        assert_eq!(m.sub(0, 1), 16);
        assert_eq!(m.sub(0, 16), 1);
        // Paper q_3 boundary.
        let m = DynModulus::new(8380417);
        assert_eq!(m.sub(0, 1), 8380416);
        assert_eq!(m.sub(1, 2), 8380416);
    }

    /// `Modulus::add` near the u64-sum boundary: with `q` just under `2^63`
    /// and `a = b = q - 1`, the unreduced sum `a + b = 2q - 2 ≈ 2^64 - 4`
    /// fits in u64 without wrapping and `cond_sub` must trigger. Closes
    /// review item 16.
    #[test]
    fn modulus_add_near_u64_sum_boundary() {
        let q = (1u64 << 63) - 1; // largest non-pow2 q under §0.1 contract.
        let m = DynModulus::new(q);
        let a = q - 1;
        // 2(q − 1) mod q = q − 2.
        assert_eq!(m.add(a, a), q - 2);
        // Just under the boundary.
        assert_eq!(m.add(a, 1), 0);
        assert_eq!(m.add(a, 0), a);
    }

    /// `Modulus::add` `debug_assert!` precondition fires on unreduced input.
    /// Locks the contract added in item 24: every release-build call site is
    /// expected to pass `a, b ∈ [0, q)`; debug builds enforce it. (Only
    /// included under `cfg(debug_assertions)` since the assert is compiled
    /// out in release.)
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Modulus::add: a >= q")]
    fn modulus_add_debug_asserts_lhs_in_range() {
        let m = DynModulus::new(17);
        let _ = m.add(17, 5); // a == q violates the precondition.
    }

    /// `Modulus::mul` for inputs close to `q - 1` at the largest paper prime
    /// (VIA-C / VIA-B q_1 second RNS prime, ≈ 2^38). The product `a · b ≈
    /// q^2 ≈ 2^76` exercises Barrett at full u128 width. Closes review item 17.
    #[test]
    fn modulus_mul_near_q_squared_paper_prime() {
        let q = 274_810_798_081u64; // ViaCQ1P1, ≈ 2^38.
        let m = DynModulus::new(q);
        let qu = u128::from(q);
        // (q - 1)^2 ≡ 1 (mod q).
        assert_eq!(m.mul(q - 1, q - 1), 1);
        // (q - 1)(q - 2) ≡ 2 (mod q).
        assert_eq!(m.mul(q - 1, q - 2), 2);
        // Cross-check a handful against u128 reference.
        for (a, b) in [(q - 1, q / 2), (q / 3, q - 7), (q - 17, q - 19)] {
            let got = m.mul(a, b);
            let want = ((u128::from(a) * u128::from(b)) % qu) as u64;
            assert_eq!(got, want, "a={a}, b={b}");
        }
    }

    /// CT centred lift produces identical output to the non-CT version
    /// across a sweep of representative inputs and moduli.
    #[test]
    fn to_centered_i64_ct_matches_non_ct_small_moduli() {
        for q in [2u64, 3, 17, 256, 4096, 8380417] {
            let m = DynModulus::new(q);
            for a in 0..q {
                let want = m.to_centered_i64(a);
                let got = m.to_centered_i64_ct(a);
                assert_eq!(got, want, "q={q}, a={a}");
            }
        }
    }

    /// CT centred lift at a paper prime, sweeping a handful of
    /// boundary-relevant inputs (0, q/2, q/2+1, q-1, and random
    /// midpoints). Full sweep would be too slow at $q \approx 2^{38}$.
    #[test]
    fn to_centered_i64_ct_matches_non_ct_paper_prime() {
        let q = 274_810_798_081u64; // VIA-C q_1 second RNS prime, ~2^38
        let m = DynModulus::new(q);
        let half = q / 2;
        for &a in &[
            0,
            1,
            half - 1,
            half,
            half + 1,
            q - 1,
            12_345_678_901u64,
            q - 12_345_678_901u64,
        ] {
            let want = m.to_centered_i64(a);
            let got = m.to_centered_i64_ct(a);
            assert_eq!(got, want, "q={q}, a={a}");
        }
    }

    /// Explicit hand-computed boundary checks for the CT path at q=17.
    #[test]
    fn to_centered_i64_ct_boundary_q17() {
        let m = ConstModulus::<17>;
        assert_eq!(m.to_centered_i64_ct(0), 0);
        assert_eq!(m.to_centered_i64_ct(8), 8); // q/2
        assert_eq!(m.to_centered_i64_ct(9), -8); // q/2 + 1
        assert_eq!(m.to_centered_i64_ct(16), -1); // q - 1
    }
}
