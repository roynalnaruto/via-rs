//! Single-value ergonomic wrapper [`RnsZq<B>`].
//!
//! [`RnsZq`] carries the two reduced `u64` components $(a_0, a_1)$ that
//! represent an element of $\mathbb{Z}_Q$ under the CRT decomposition
//! $Q = q^{(0)} \cdot q^{(1)}$, plus the [`RnsBasis`] context that defines
//! them. The wrapper implements the usual arithmetic operators (`+`, `-`,
//! `*`, unary `-`, and the `*_assign` family) plus
//! [`subtle::ConditionallySelectable`] and [`zeroize::Zeroize`], all acting
//! componentwise.
//!
//! For batch arithmetic on polynomial coefficient vectors prefer the
//! [`ops`](super::ops) kernels: they avoid the per-element wrapper overhead
//! and lower cleanly to SIMD / GPU later.

use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand_core::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use zeroize::Zeroize;

use super::super::zq::element::Zq;
use super::super::zq::modulus::Modulus;
use super::basis::RnsBasis;

/// An element of $\mathbb{Z}_Q$ paired with its two-prime RNS basis.
///
/// Carries the basis by value (not by reference) so that [`RnsZq`] remains
/// `Copy`. For zero-sized bases ([`ConstRnsBasis`](super::basis::ConstRnsBasis))
/// the wrapper occupies just two `u64`s; for
/// [`DynRnsBasis`](super::basis::DynRnsBasis) it grows by `sizeof(DynRnsBasis)`
/// (~80 bytes) — acceptable for element-level ergonomics but not for
/// polynomial-sized data; use the [`ops`](super::ops) slice kernels for that.
///
/// # Invariants
///
/// The stored `value0` is always in $[0, q^{(0)})$ and `value1` is always in
/// $[0, q^{(1)})$. Constructors enforce this via the underlying [`Modulus`]
/// reduction kernels; the operator overloads preserve it.
#[derive(Copy, Clone)]
pub struct RnsZq<B: RnsBasis> {
    value0: u64,
    value1: u64,
    basis: B,
}

impl<B: RnsBasis> RnsZq<B> {
    /// Construct an [`RnsZq`] from the two raw `u64` components, reducing each
    /// into its prime range if needed.
    ///
    /// Equivalent to $(v_0 \bmod q^{(0)}, v_1 \bmod q^{(1)})$.
    #[inline(always)]
    pub fn new(basis: B, value0: u64, value1: u64) -> Self {
        let v0 = basis.m0().reduce_u64(value0);
        let v1 = basis.m1().reduce_u64(value1);
        // SAFETY: `reduce_u64` always returns a value in `[0, q)`.
        unsafe { Self::from_reduced_unchecked(basis, v0, v1) }
    }

    /// Construct an [`RnsZq`] from an unsigned `u128`, decomposing into the
    /// two component residues.
    #[inline(always)]
    pub fn from_u128(basis: B, x: u128) -> Self {
        let (v0, v1) = basis.decompose_u128(x);
        // SAFETY: `decompose_u128` returns components in canonical reduced form.
        unsafe { Self::from_reduced_unchecked(basis, v0, v1) }
    }

    /// Construct an [`RnsZq`] from a signed `i128`, lifting into $[0, Q)$.
    ///
    /// Useful at boundaries where samplers or centred representations produce
    /// signed integers that must be lifted into $\mathbb{Z}_Q$.
    ///
    /// # Constant-time
    ///
    /// Constant-time over `x`: both branches (magnitude and its negation
    /// modulo `Q`) are computed unconditionally and the result is selected
    /// via [`subtle::ConditionallySelectable`]. `i128::unsigned_abs` is
    /// itself branchless in `std` and correctly maps `i128::MIN` to
    /// `2^127`. Mirrors the §0.1 [`Modulus::reduce_i64`] discipline; the
    /// sign of `x` may be secret sampler output and must not leak.
    #[inline(always)]
    pub fn from_i128(basis: B, x: i128) -> Self {
        let magnitude = Self::from_u128(basis, x.unsigned_abs());
        let neg = -magnitude;
        let is_negative = Choice::from(x.is_negative() as u8);
        Self::conditional_select(&magnitude, &neg, is_negative)
    }

    /// Construct an [`RnsZq`] from two `u64`s that are **already in canonical
    /// reduced form** ($v_0 < q^{(0)}$ and $v_1 < q^{(1)}$).
    ///
    /// Marked `unsafe` not because misuse can cause memory-safety UB
    /// ([`RnsZq`] is a plain pair-of-`u64` wrapper), but because misuse
    /// silently corrupts downstream cryptographic arithmetic — a class of bug
    /// we want every caller to acknowledge explicitly.
    ///
    /// # Safety
    ///
    /// Caller must guarantee `v0 < basis.m0().q()` and `v1 < basis.m1().q()`.
    /// Use [`RnsZq::new`] if you cannot prove that locally.
    #[inline(always)]
    pub const unsafe fn from_reduced_unchecked(basis: B, value0: u64, value1: u64) -> Self {
        Self {
            value0,
            value1,
            basis,
        }
    }

    /// The zero element $0 \in \mathbb{Z}_Q$.
    #[inline(always)]
    pub fn zero(basis: B) -> Self {
        // SAFETY: `0 < q` for every component modulus (both q >= 2).
        unsafe { Self::from_reduced_unchecked(basis, 0, 0) }
    }

    /// The one element $1 \in \mathbb{Z}_Q$.
    ///
    /// # Panics in debug
    ///
    /// Asserts both component moduli are at least 2 in debug builds.
    #[inline(always)]
    pub fn one(basis: B) -> Self {
        debug_assert!(basis.m0().q() >= 2, "RnsZq::one requires q0 >= 2");
        debug_assert!(basis.m1().q() >= 2, "RnsZq::one requires q1 >= 2");
        // SAFETY: `1 < q` when `q >= 2`.
        unsafe { Self::from_reduced_unchecked(basis, 1, 1) }
    }

    /// The first component value in canonical $[0, q^{(0)})$ form.
    #[inline(always)]
    pub const fn value0(self) -> u64 {
        self.value0
    }

    /// The second component value in canonical $[0, q^{(1)})$ form.
    #[inline(always)]
    pub const fn value1(self) -> u64 {
        self.value1
    }

    /// The basis this element is associated with.
    #[inline(always)]
    pub const fn basis(self) -> B {
        self.basis
    }

    /// The reconstructed `u128` value in canonical $[0, Q)$ form via Garner.
    #[inline(always)]
    pub fn to_u128(self) -> u128 {
        self.basis.reconstruct(self.value0, self.value1)
    }

    /// Sample a uniformly random element of $\mathbb{Z}_Q$ by independently
    /// drawing each component from its prime range.
    ///
    /// Equivalent in distribution to $(x \bmod q^{(0)}, x \bmod q^{(1)})$ for
    /// $x$ uniform on $[0, Q)$ — the CRT bijection preserves uniformity.
    pub fn random<R: RngCore + ?Sized>(basis: B, rng: &mut R) -> Self {
        let v0 = Zq::random(basis.m0(), rng).to_u64();
        let v1 = Zq::random(basis.m1(), rng).to_u64();
        // SAFETY: `Zq::random` returns a value in `[0, q)` for each component.
        unsafe { Self::from_reduced_unchecked(basis, v0, v1) }
    }
}

impl<B: RnsBasis> Add for RnsZq<B> {
    type Output = Self;
    #[inline(always)]
    fn add(self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "RnsZq::add: basis mismatch");
        let v0 = self.basis.m0().add(self.value0, rhs.value0);
        let v1 = self.basis.m1().add(self.value1, rhs.value1);
        // SAFETY: `Modulus::add` returns each component in canonical range.
        unsafe { Self::from_reduced_unchecked(self.basis, v0, v1) }
    }
}

impl<B: RnsBasis> Sub for RnsZq<B> {
    type Output = Self;
    #[inline(always)]
    fn sub(self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "RnsZq::sub: basis mismatch");
        let v0 = self.basis.m0().sub(self.value0, rhs.value0);
        let v1 = self.basis.m1().sub(self.value1, rhs.value1);
        // SAFETY: `Modulus::sub` returns each component in canonical range.
        unsafe { Self::from_reduced_unchecked(self.basis, v0, v1) }
    }
}

impl<B: RnsBasis> Mul for RnsZq<B> {
    type Output = Self;
    #[inline(always)]
    fn mul(self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "RnsZq::mul: basis mismatch");
        let v0 = self.basis.m0().mul(self.value0, rhs.value0);
        let v1 = self.basis.m1().mul(self.value1, rhs.value1);
        // SAFETY: `Modulus::mul` returns each component in canonical range.
        unsafe { Self::from_reduced_unchecked(self.basis, v0, v1) }
    }
}

impl<B: RnsBasis> Mul<u64> for RnsZq<B> {
    type Output = Self;
    /// Multiply by an arbitrary `u64` scalar.
    ///
    /// The scalar is reduced through each component modulus before the
    /// componentwise multiply, so callers need not pre-reduce.
    #[inline(always)]
    fn mul(self, scalar: u64) -> Self {
        let s0 = self.basis.m0().reduce_u64(scalar);
        let s1 = self.basis.m1().reduce_u64(scalar);
        let v0 = self.basis.m0().mul(self.value0, s0);
        let v1 = self.basis.m1().mul(self.value1, s1);
        // SAFETY: `Modulus::mul` returns each component in canonical range.
        unsafe { Self::from_reduced_unchecked(self.basis, v0, v1) }
    }
}

impl<B: RnsBasis> Neg for RnsZq<B> {
    type Output = Self;
    #[inline(always)]
    fn neg(self) -> Self {
        let v0 = self.basis.m0().neg(self.value0);
        let v1 = self.basis.m1().neg(self.value1);
        // SAFETY: `Modulus::neg` returns each component in canonical range.
        unsafe { Self::from_reduced_unchecked(self.basis, v0, v1) }
    }
}

impl<B: RnsBasis> AddAssign for RnsZq<B> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<B: RnsBasis> SubAssign for RnsZq<B> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<B: RnsBasis> MulAssign for RnsZq<B> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<B: RnsBasis> PartialEq for RnsZq<B> {
    /// Equal iff both component values agree **and** the bases agree. For
    /// zero-sized bases the basis check is a no-op.
    #[inline(always)]
    fn eq(&self, other: &Self) -> bool {
        self.value0 == other.value0 && self.value1 == other.value1 && self.basis == other.basis
    }
}

impl<B: RnsBasis> Eq for RnsZq<B> {}

impl<B: RnsBasis> ConstantTimeEq for RnsZq<B> {
    /// Constant-time equality on both component values combined with bitwise AND.
    ///
    /// The basis is a public parameter; this comparison is meaningful only
    /// when the caller has already established that the two operands share a
    /// basis. The default [`PartialEq`] implementation enforces the basis
    /// match in non-constant time; use this when both operands are known to
    /// live in the same ring.
    #[inline(always)]
    fn ct_eq(&self, other: &Self) -> Choice {
        self.value0.ct_eq(&other.value0) & self.value1.ct_eq(&other.value1)
    }
}

impl<B: RnsBasis> ConditionallySelectable for RnsZq<B> {
    /// Select `b` when `choice` is set, else `a`. Both operands must share the
    /// same basis; the resulting [`RnsZq`] inherits that basis.
    #[inline(always)]
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        assert!(
            a.basis == b.basis,
            "RnsZq::conditional_select: basis mismatch",
        );
        let v0 = u64::conditional_select(&a.value0, &b.value0, choice);
        let v1 = u64::conditional_select(&a.value1, &b.value1, choice);
        // SAFETY: each selected value is one of `a.valueN` or `b.valueN`,
        // each already in canonical reduced form.
        unsafe { Self::from_reduced_unchecked(a.basis, v0, v1) }
    }
}

impl<B: RnsBasis> Zeroize for RnsZq<B> {
    /// Zero both components. The basis is a public parameter and is
    /// intentionally **not** wiped.
    #[inline(always)]
    fn zeroize(&mut self) {
        self.value0.zeroize();
        self.value1.zeroize();
    }
}

impl<B: RnsBasis> Hash for RnsZq<B> {
    /// Hash on the values and on the basis's component moduli. Two [`RnsZq`]
    /// instances with the same component values but different bases hash
    /// differently, mirroring [`PartialEq`].
    #[inline(always)]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value0.hash(state);
        self.value1.hash(state);
        self.basis.m0().q().hash(state);
        self.basis.m1().q().hash(state);
    }
}

impl<B: RnsBasis> fmt::Debug for RnsZq<B> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RnsZq({} mod {}, {} mod {})",
            self.value0,
            self.basis.m0().q(),
            self.value1,
            self.basis.m1().q(),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::rns::basis::{ConstRnsBasis, DynRnsBasis, paper};
    use crate::primitives::zq::modulus::DynModulus;

    type Z55 = ConstRnsBasis<5, 11>;

    #[test]
    fn ops_const_basis_tiny() {
        let b = Z55::default();
        let x = RnsZq::from_u128(b, 23);
        let y = RnsZq::from_u128(b, 17);
        assert_eq!(x.to_u128(), 23);
        assert_eq!(y.to_u128(), 17);
        assert_eq!((x + y).to_u128(), 40); // 23 + 17 mod 55
        assert_eq!((x - y).to_u128(), 6); // 23 - 17 mod 55
        assert_eq!((x * y).to_u128(), (23u128 * 17) % 55); // 391 mod 55 = 6
        assert_eq!((-x).to_u128(), 32); // -23 mod 55
    }

    #[test]
    fn ops_const_basis_via_q1() {
        let b = paper::ViaQ1Rns::default();
        let q = b.big_q();
        let x_raw: u128 = 12345678901234567890;
        let y_raw: u128 = 9876543210987654321;
        let x = RnsZq::from_u128(b, x_raw);
        let y = RnsZq::from_u128(b, y_raw);
        let xr = x_raw % q;
        let yr = y_raw % q;
        assert_eq!(x.to_u128(), xr);
        assert_eq!(y.to_u128(), yr);
        assert_eq!((x + y).to_u128(), (xr + yr) % q);
        assert_eq!((x - y).to_u128(), (xr + q - yr) % q);
        // Multiplication may overflow u128 for both factors ≈ q < 2^57, so the
        // intermediate xr * yr fits in u128 (each < 2^57).
        assert_eq!((x * y).to_u128(), (xr * yr) % q);
        assert_eq!((-x).to_u128(), (q - xr) % q);
    }

    #[test]
    fn ops_dyn_basis_matches_const() {
        let c = paper::ViaQ1Rns::default();
        let d = DynRnsBasis::new(DynModulus::new(268369921), DynModulus::new(536608769));
        let x_raw: u128 = 999_999_999_999;
        let y_raw: u128 = 123_456_789_012;
        let xc = RnsZq::from_u128(c, x_raw);
        let yc = RnsZq::from_u128(c, y_raw);
        let xd = RnsZq::from_u128(d, x_raw);
        let yd = RnsZq::from_u128(d, y_raw);
        assert_eq!((xc + yc).to_u128(), (xd + yd).to_u128());
        assert_eq!((xc - yc).to_u128(), (xd - yd).to_u128());
        assert_eq!((xc * yc).to_u128(), (xd * yd).to_u128());
        assert_eq!((-xc).to_u128(), (-xd).to_u128());
    }

    #[test]
    fn mul_u64_scalar_tiny() {
        let b = Z55::default();
        let x = RnsZq::from_u128(b, 23);
        // Scalar 7 < q0, q1 — no reduction needed.
        assert_eq!((x * 7u64).to_u128(), (23 * 7) % 55);
        // Scalar 100 > both q0 and q1 — reduction kicks in.
        assert_eq!((x * 100u64).to_u128(), (23 * 100) % 55);
    }

    #[test]
    fn from_i128_signed() {
        let b = Z55::default();
        // -3 in Z_55 is 52.
        assert_eq!(RnsZq::from_i128(b, -3).to_u128(), 52);
        // -60 in Z_55 is (-60 mod 55) = 50.
        assert_eq!(RnsZq::from_i128(b, -60).to_u128(), 50);
        // 0 is 0.
        assert_eq!(RnsZq::from_i128(b, 0).to_u128(), 0);
        // 100 in Z_55 is 45.
        assert_eq!(RnsZq::from_i128(b, 100).to_u128(), 45);
    }

    /// `from_i128(i128::MIN)` exercises the `unsigned_abs → neg` path on
    /// the most adversarial signed input — `i128::MIN.unsigned_abs() ==
    /// 2^127`. Closes the `.docs/review.md` item 11 gap and verifies the
    /// constant-time rewrite preserves value semantics on the boundary.
    #[test]
    fn from_i128_min_extreme() {
        // Z_55 (toy) and the VIA q_1 paper basis (75-bit Q) — the latter
        // confirms the path is well-defined at the realistic scale.
        let b = Z55::default();
        let got = RnsZq::from_i128(b, i128::MIN).to_u128();
        let want = i128::MIN.rem_euclid(55) as u128;
        assert_eq!(got, want);
        // 2^127 mod 55 = 18, so -2^127 mod 55 = 37.
        assert_eq!(got, 37);

        let b = paper::ViaQ1Rns::default();
        let q = b.big_q() as i128; // VIA q_1 ≈ 2^57 < 2^126, fits in i128.
        let got = RnsZq::from_i128(b, i128::MIN).to_u128();
        let want = i128::MIN.rem_euclid(q) as u128;
        assert_eq!(got, want);
    }

    #[test]
    fn zero_one_const() {
        let b = paper::ViaQ1Rns::default();
        assert_eq!(RnsZq::zero(b).to_u128(), 0);
        assert_eq!(RnsZq::one(b).to_u128(), 1);
    }

    #[test]
    fn conditional_select_picks_b_when_set() {
        let b = paper::ViaQ1Rns::default();
        let a = RnsZq::from_u128(b, 3);
        let bb = RnsZq::from_u128(b, 1_000_000);
        let pick_a = RnsZq::conditional_select(&a, &bb, Choice::from(0));
        let pick_b = RnsZq::conditional_select(&a, &bb, Choice::from(1));
        assert_eq!(pick_a.to_u128(), 3);
        assert_eq!(pick_b.to_u128(), 1_000_000);
    }

    #[test]
    fn ct_eq_matches_eq_when_bases_agree() {
        let b = paper::ViaQ1Rns::default();
        let a = RnsZq::from_u128(b, 42);
        let c = RnsZq::from_u128(b, 42);
        let d = RnsZq::from_u128(b, 43);
        assert!(bool::from(a.ct_eq(&c)));
        assert!(!bool::from(a.ct_eq(&d)));
    }

    #[test]
    fn zeroize_clears_both_components() {
        let b = paper::ViaQ1Rns::default();
        let mut z = RnsZq::from_u128(b, 12_345_678_901_234);
        z.zeroize();
        assert_eq!(z.value0(), 0);
        assert_eq!(z.value1(), 0);
        assert_eq!(z.to_u128(), 0);
    }

    #[test]
    fn random_in_range() {
        // Reuse the Counter RNG from zq's element tests.
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
        let b = paper::ViaQ1Rns::default();
        let mut rng = Counter(0);
        for _ in 0..256 {
            let z = RnsZq::random(b, &mut rng);
            assert!(z.value0() < b.m0().q());
            assert!(z.value1() < b.m1().q());
            assert!(z.to_u128() < b.big_q());
        }
    }

    #[test]
    fn paper_aliases_compile() {
        let _: RnsZq<paper::ViaQ1Rns> = RnsZq::zero(paper::ViaQ1Rns::default());
        let _: RnsZq<paper::ViaCQ1Rns> = RnsZq::zero(paper::ViaCQ1Rns::default());
    }

    /// Distributivity `(a + b) * c = a*c + b*c` on `RnsZq` at the paper
    /// basis. Locks the ring axioms against future componentwise kernel
    /// regressions. Closes review item 23 (RnsZq side).
    #[test]
    fn rnszq_ring_axiom_distributivity() {
        let b = paper::ViaQ1Rns::default();
        let a = RnsZq::from_u128(b, 12345678901234567890);
        let b_v = RnsZq::from_u128(b, 1357913579135791357);
        let c = RnsZq::from_u128(b, 9999999999999999999);
        assert_eq!((a + b_v) * c, (a * c) + (b_v * c));
    }

    /// `RnsZq::add` between two `DynRnsBasis` instances with different
    /// primes must panic via the basis-equality assert. Locks the
    /// cross-basis guardrail. Closes review item 22 (RnsZq side).
    #[test]
    #[should_panic(expected = "basis mismatch")]
    fn rnszq_add_panics_on_basis_mismatch() {
        let b1 = DynRnsBasis::new(DynModulus::new(5), DynModulus::new(11));
        let b2 = DynRnsBasis::new(DynModulus::new(7), DynModulus::new(13));
        let a = RnsZq::from_u128(b1, 10);
        let bb = RnsZq::from_u128(b2, 10);
        let _ = a + bb;
    }

    /// `RnsZq::random` distribution-uniformity smoke test on the per-prime
    /// components. Runs a Pearson chi-squared on each prime independently
    /// (since CRT preserves uniformity, this is sufficient). Closes review
    /// item 18 (RnsZq side).
    #[test]
    fn rnszq_random_uniformity_chi_squared() {
        type Z77 = crate::primitives::rns::basis::ConstRnsBasis<7, 11>;
        let b = Z77::default();
        let mut rng = SplitMix64::new(0xDEADBEEF_BAADF00D);
        let mut counts0 = [0u64; 7];
        let mut counts1 = [0u64; 11];
        const N: u64 = 10_000;
        for _ in 0..N {
            let z = RnsZq::random(b, &mut rng);
            counts0[z.value0() as usize] += 1;
            counts1[z.value1() as usize] += 1;
        }
        // χ² at the chosen threshold (50): well above the 99% critical
        // values for 6 d.f. (~16.8) and 10 d.f. (~23.2), but tight enough
        // to catch gross bias.
        let chi2 = |counts: &[u64], q: u64| -> f64 {
            let expected = N as f64 / q as f64;
            counts
                .iter()
                .map(|&o| {
                    let d = o as f64 - expected;
                    d * d / expected
                })
                .sum()
        };
        let c0 = chi2(&counts0, 7);
        let c1 = chi2(&counts1, 11);
        assert!(c0 < 50.0, "m0 chi^2 = {c0}, counts = {counts0:?}");
        assert!(c1 < 50.0, "m1 chi^2 = {c1}, counts = {counts1:?}");
    }

    /// SplitMix64 — duplicate of the helper in `zq::element::tests` so the
    /// uniformity tests don't need a cross-module dependency. Lift to a
    /// shared `test_util` module if a third caller appears.
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
