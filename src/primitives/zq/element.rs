//! Single-value ergonomic wrapper [`Zq<M>`].
//!
//! [`Zq`] carries a reduced `u64` coefficient and (for runtime moduli) the
//! [`Modulus`] context that defines it. The wrapper implements the usual
//! arithmetic operators (`+`, `-`, `*`, unary `-`, and the `*_assign` family)
//! plus [`subtle::ConditionallySelectable`] and [`zeroize::Zeroize`]. For
//! batch arithmetic on polynomial coefficient vectors, prefer the
//! [`ops`](super::ops) kernels: they avoid the per-element wrapper overhead
//! and lower cleanly to SIMD / GPU later.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand_core::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use zeroize::Zeroize;

use super::modulus::Modulus;

/// An element of $\mathbb{Z}_q$ paired with its modulus context.
///
/// The wrapper carries the modulus by value (not by reference) so that
/// [`Zq`] remains `Copy`. For zero-sized moduli ([`ConstModulus`] and
/// [`PowerOfTwoModulus`](super::modulus::PowerOfTwoModulus)) the wrapper has
/// the same layout as a bare `u64`; for [`DynModulus`](super::modulus::DynModulus)
/// the wrapper grows by `sizeof(DynModulus)` (~32 bytes) — acceptable for
/// element-level ergonomics, but use the [`ops`](super::ops) slice kernels for
/// polynomial-sized data.
///
/// # Invariants
///
/// The stored `value` is always in $[0, q)$. Constructors enforce this via
/// the modulus's reduction kernel; the operator overloads preserve it.
///
/// [`ConstModulus`]: super::modulus::ConstModulus
/// [`DynModulus`]: super::modulus::DynModulus
#[derive(Copy, Clone)]
pub struct Zq<M: Modulus> {
    value: u64,
    modulus: M,
}

impl<M: Modulus> Zq<M> {
    /// Construct a [`Zq`] from a `u64`, reducing into $[0, q)$ if needed.
    ///
    /// Equivalent to $x \bmod q$ in the canonical representation.
    #[inline(always)]
    pub fn new(modulus: M, value: u64) -> Self {
        let reduced = modulus.reduce_u64(value);
        // SAFETY: `reduce_u64` always returns a value in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(modulus, reduced) }
    }

    /// Construct a [`Zq`] from a signed `i64`, lifting into $[0, q)$.
    ///
    /// Useful for samplers that produce signed integers (ternary, bounded
    /// uniform, discrete Gaussian — see §1.3-1.5).
    #[inline(always)]
    pub fn from_i64(modulus: M, value: i64) -> Self {
        let reduced = modulus.reduce_i64(value);
        // SAFETY: `reduce_i64` always returns a value in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(modulus, reduced) }
    }

    /// Construct a [`Zq`] from a `u64` that is **already in canonical reduced
    /// form** $[0, q)$.
    ///
    /// Marked `unsafe` not because misuse can cause memory-safety UB
    /// (`Zq` is a plain `u64` wrapper), but because misuse silently corrupts
    /// downstream cryptographic arithmetic — a class of bug we want every
    /// caller to acknowledge explicitly.
    ///
    /// # Safety
    ///
    /// Caller must guarantee `value < modulus.q()`. Use [`Zq::new`] if you
    /// cannot prove that locally.
    #[inline(always)]
    pub const unsafe fn from_reduced_unchecked(modulus: M, value: u64) -> Self {
        Self { value, modulus }
    }

    /// The zero element $0 \in \mathbb{Z}_q$.
    #[inline(always)]
    pub fn zero(modulus: M) -> Self {
        // SAFETY: `0 < q` for every valid modulus (`q >= 2`).
        unsafe { Self::from_reduced_unchecked(modulus, 0) }
    }

    /// The one element $1 \in \mathbb{Z}_q$.
    ///
    /// # Panics in debug
    ///
    /// Asserts $q \ge 2$ in debug builds, so the value `1` is in $[0, q)$.
    #[inline(always)]
    pub fn one(modulus: M) -> Self {
        debug_assert!(modulus.q() >= 2, "Zq::one requires q >= 2");
        // SAFETY: `1 < q` when `q >= 2`, which is the documented contract.
        unsafe { Self::from_reduced_unchecked(modulus, 1) }
    }

    /// The underlying `u64` value in canonical $[0, q)$ form.
    #[inline(always)]
    pub const fn to_u64(self) -> u64 {
        self.value
    }

    /// The centred representation $\tilde a \in (-\lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$.
    ///
    /// See [`Modulus::to_centered_i64`] for the invariants and the constant-time note.
    #[inline(always)]
    pub fn to_centered_i64(self) -> i64 {
        self.modulus.to_centered_i64(self.value)
    }

    /// The modulus this element is associated with.
    #[inline(always)]
    pub const fn modulus(self) -> M {
        self.modulus
    }

    /// Sample a uniformly random element of $\mathbb{Z}_q$.
    ///
    /// Uses rejection sampling on the smallest power-of-two cover of $[0, q)$.
    /// On average draws fewer than two `u64` words per sample for any modulus
    /// (worst case is when $q$ is just above a power of two; even then the
    /// expected reject rate stays below $1$).
    ///
    /// # Constant-time
    ///
    /// Rejection sampling has data-independent rejection rate (depends only
    /// on $q$, a public parameter). The accepted value is uniformly random,
    /// so its bytes carry no information about the sampler's internal state.
    pub fn random<R: RngCore + ?Sized>(modulus: M, rng: &mut R) -> Self {
        let q = modulus.q();
        debug_assert!(q >= 2, "Zq::random requires q >= 2");
        let bits = 64 - (q - 1).leading_zeros();
        let mask = if bits == 64 {
            u64::MAX
        } else {
            (1u64 << bits) - 1
        };
        loop {
            let mut buf = [0u8; 8];
            rng.fill_bytes(&mut buf);
            let candidate = u64::from_le_bytes(buf) & mask;
            if candidate < q {
                // SAFETY: rejection branch guarantees `candidate < q`.
                return unsafe { Self::from_reduced_unchecked(modulus, candidate) };
            }
        }
    }
}

impl<M: Modulus> Add for Zq<M> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Zq::add: modulus mismatch");
        let v = self.modulus.add(self.value, rhs.value);
        // SAFETY: `Modulus::add` returns a value in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(self.modulus, v) }
    }
}

impl<M: Modulus> Sub for Zq<M> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Zq::sub: modulus mismatch");
        let v = self.modulus.sub(self.value, rhs.value);
        // SAFETY: `Modulus::sub` returns a value in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(self.modulus, v) }
    }
}

impl<M: Modulus> Mul for Zq<M> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Zq::mul: modulus mismatch");
        let v = self.modulus.mul(self.value, rhs.value);
        // SAFETY: `Modulus::mul` returns a value in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(self.modulus, v) }
    }
}

impl<M: Modulus> Mul<u64> for Zq<M> {
    type Output = Self;
    /// Multiply by an arbitrary `u64` scalar (does not require pre-reduction).
    ///
    /// Performed as a single `u128` reduction: `value * scalar < q * 2^64 < 2^128`
    /// whenever `q < 2^63`, which matches our §0.1 modulus bound.
    #[inline(always)]
    fn mul(self, scalar: u64) -> Self {
        let v = self
            .modulus
            .reduce_u128(u128::from(self.value) * u128::from(scalar));
        // SAFETY: `reduce_u128` returns a value in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(self.modulus, v) }
    }
}

impl<M: Modulus> Neg for Zq<M> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let v = self.modulus.neg(self.value);
        // SAFETY: `Modulus::neg` returns a value in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(self.modulus, v) }
    }
}

impl<M: Modulus> AddAssign for Zq<M> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<M: Modulus> SubAssign for Zq<M> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<M: Modulus> MulAssign for Zq<M> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<M: Modulus> PartialEq for Zq<M> {
    /// Equal iff the reduced values agree **and** the moduli agree. For
    /// zero-sized moduli the modulus check is a no-op.
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value && self.modulus == other.modulus
    }
}

impl<M: Modulus> Eq for Zq<M> {}

impl<M: Modulus> ConstantTimeEq for Zq<M> {
    /// Constant-time equality on the *value* component.
    ///
    /// The modulus is a public parameter; this comparison is meaningful only
    /// when the caller has already established that the two operands share a
    /// modulus. The default ([`PartialEq`]) implementation enforces the
    /// modulus match in non-constant time; use this when both operands are
    /// known to live in the same ring.
    #[inline(always)]
    fn ct_eq(&self, other: &Self) -> Choice {
        self.value.ct_eq(&other.value)
    }
}

impl<M: Modulus> ConditionallySelectable for Zq<M> {
    /// Select `b` when `choice` is set, else `a`. Both operands must share
    /// the same modulus; the resulting [`Zq`] inherits that modulus.
    #[inline(always)]
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        assert!(
            a.modulus == b.modulus,
            "Zq::conditional_select: modulus mismatch"
        );
        let v = u64::conditional_select(&a.value, &b.value, choice);
        // SAFETY: `v` is one of `a.value` or `b.value`, each in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(a.modulus, v) }
    }
}

impl<M: Modulus> Zeroize for Zq<M> {
    /// Zero the coefficient. The modulus is a public parameter and is
    /// intentionally **not** wiped.
    #[inline(always)]
    fn zeroize(&mut self) {
        self.value.zeroize();
    }
}

impl<M: Modulus> Hash for Zq<M> {
    /// Hash on the value and the modulus's `q`. Two `Zq` instances with the
    /// same value but different moduli hash differently, mirroring [`PartialEq`].
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
        self.modulus.q().hash(state);
    }
}

impl<M: Modulus> fmt::Debug for Zq<M> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Zq({} mod {})", self.value, self.modulus.q())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::zq::modulus::{ConstModulus, DynModulus, PowerOfTwoModulus};

    #[test]
    fn ops_const_modulus() {
        let m = ConstModulus::<17>;
        let a = Zq::new(m, 10);
        let b = Zq::new(m, 12);
        assert_eq!((a + b).to_u64(), 5);
        assert_eq!((a - b).to_u64(), 15);
        assert_eq!((a * b).to_u64(), 1); // 120 mod 17 = 1
        assert_eq!((-a).to_u64(), 7);
        assert_eq!((a * 3u64).to_u64(), 13); // 30 mod 17
    }

    #[test]
    fn ops_dyn_modulus() {
        let m = DynModulus::new(17);
        let a = Zq::new(m, 10);
        let b = Zq::new(m, 12);
        assert_eq!((a + b).to_u64(), 5);
        assert_eq!((a * b).to_u64(), 1);
    }

    #[test]
    fn pow2_zero_one() {
        let m = PowerOfTwoModulus::<4>;
        assert_eq!(Zq::zero(m).to_u64(), 0);
        assert_eq!(Zq::one(m).to_u64(), 1);
    }

    #[test]
    fn random_in_range() {
        // Use a tiny deterministic RNG by hand to avoid pulling in `rand`.
        struct Counter(u64);
        impl RngCore for Counter {
            fn next_u32(&mut self) -> u32 {
                self.0 = self.0.wrapping_add(1);
                self.0 as u32
            }
            fn next_u64(&mut self) -> u64 {
                self.0 = self.0.wrapping_add(1);
                self.0
            }
            fn fill_bytes(&mut self, dst: &mut [u8]) {
                for chunk in dst.chunks_mut(8) {
                    self.0 = self.0.wrapping_add(1);
                    let bytes = self.0.to_le_bytes();
                    chunk.copy_from_slice(&bytes[..chunk.len()]);
                }
            }
            fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), rand_core::Error> {
                self.fill_bytes(dst);
                Ok(())
            }
        }
        let m = ConstModulus::<17>;
        let mut rng = Counter(0);
        for _ in 0..256 {
            let z = Zq::random(m, &mut rng);
            assert!(z.to_u64() < 17);
        }
    }

    #[test]
    fn conditional_select_picks_b_when_set() {
        let m = ConstModulus::<17>;
        let a = Zq::new(m, 3);
        let b = Zq::new(m, 11);
        let pick_a = Zq::conditional_select(&a, &b, Choice::from(0));
        let pick_b = Zq::conditional_select(&a, &b, Choice::from(1));
        assert_eq!(pick_a.to_u64(), 3);
        assert_eq!(pick_b.to_u64(), 11);
    }

    #[test]
    fn zeroize_clears_value() {
        let m = ConstModulus::<17>;
        let mut z = Zq::new(m, 13);
        z.zeroize();
        assert_eq!(z.to_u64(), 0);
    }

    /// `Zq::new(m, q)` — a value exactly at the modulus must reduce to 0.
    /// Trivial edge case but not previously pinned; closes review item 10.
    #[test]
    fn zq_new_at_modulus_reduces_to_zero() {
        let m = ConstModulus::<17>;
        assert_eq!(Zq::new(m, 17).to_u64(), 0);
        let m = DynModulus::new(8380417);
        assert_eq!(Zq::new(m, 8380417).to_u64(), 0);
        let m = PowerOfTwoModulus::<4>;
        assert_eq!(Zq::new(m, 16).to_u64(), 0);
    }

    /// Distributivity `(a + b) * c = a*c + b*c` at both a small prime and a
    /// paper prime. Single sanity check that locks the ring axioms against
    /// any future kernel regression. Closes review item 23 (Zq side).
    #[test]
    fn zq_ring_axiom_distributivity() {
        for q in [17u64, 8380417, 274_810_798_081] {
            let m = DynModulus::new(q);
            let a = Zq::new(m, 12345 % q);
            let b = Zq::new(m, 67890 % q);
            let c = Zq::new(m, 13579 % q);
            assert_eq!((a + b) * c, (a * c) + (b * c), "q={q}");
        }
    }

    /// `Zq::add` between two `DynModulus` instances with different `q` must
    /// panic via the modulus-equality assert. Locks the cross-modulus
    /// guardrail. Closes review item 22 (Zq side).
    #[test]
    #[should_panic(expected = "modulus mismatch")]
    fn zq_add_panics_on_modulus_mismatch() {
        let m17 = DynModulus::new(17);
        let m19 = DynModulus::new(19);
        let a = Zq::new(m17, 5);
        let b = Zq::new(m19, 3);
        let _ = a + b;
    }

    /// `Zq::random` distribution-uniformity smoke test. Uses SplitMix64
    /// (a well-known low-bias PRG) as the underlying byte source and runs a
    /// Pearson chi-squared statistic on 10 000 samples mod 17. The 99%
    /// threshold for χ² with 16 d.f. is ~32; we set a generous 50 to avoid
    /// flaky tests while still catching gross bias (e.g. accidentally
    /// truncating the high bits via a too-small mask). Closes review item 18
    /// (Zq side).
    #[test]
    fn zq_random_uniformity_chi_squared() {
        let m = ConstModulus::<17>;
        let mut rng = SplitMix64::new(0xCAFEF00DD15EA5E5);
        let mut counts = [0u64; 17];
        const N: u64 = 10_000;
        for _ in 0..N {
            let z = Zq::random(m, &mut rng);
            counts[z.to_u64() as usize] += 1;
        }
        let expected = N as f64 / 17.0;
        let chi2: f64 = counts
            .iter()
            .map(|&o| {
                let d = o as f64 - expected;
                d * d / expected
            })
            .sum();
        assert!(chi2 < 50.0, "chi^2 = {chi2}, counts = {counts:?}");
    }

    /// SplitMix64 — a small, well-characterised PRG used in the uniformity
    /// tests. Avoids pulling in `rand_chacha` or similar as a dev-dep for
    /// just two tests. (Duplicated in `rns::element::tests`; if a third
    /// caller appears, lift it into a shared `test_util` module.)
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
    }

    impl RngCore for SplitMix64 {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn fill_bytes(&mut self, dst: &mut [u8]) {
            for chunk in dst.chunks_mut(8) {
                let bytes = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        }
        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(dst);
            Ok(())
        }
    }
}
