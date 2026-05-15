//! RNS polynomial wrapper [`PolyRns<N, B, F>`].
//!
//! [`PolyRns`] is the §0.3 analogue of [`crate::primitives::rns::element::RnsZq`]
//! at the polynomial scale: two parallel `[u64; N]` lane buffers — one per
//! RNS slot of the basis $B$ — paired with the basis context and the
//! typestate marker `F: Form`. Used when the modulus is composite, which
//! at paper params means **only $q_1$** for VIA / VIA-C / VIA-B.
//!
//! Storage is SoA (struct-of-arrays): one contiguous `[u64; N]` per prime
//! slot, side-by-side. This matches the layout `crate::primitives::rns::ops`
//! and `super::rns_ops` consume — each kernel calls into its single-prime
//! counterpart twice — and lines up naturally with per-prime NTT (§0.4).
//!
//! All invariants (and the form-neutral `values0` / `values1` field naming)
//! mirror [`super::element::Poly`].

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand_core::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use zeroize::Zeroize;

use crate::primitives::rns::basis::RnsBasis;
use crate::primitives::rns::element::RnsZq;
use crate::primitives::rns::ops as rns_ops;
use crate::primitives::zq::element::Zq;
use crate::primitives::zq::modulus::Modulus;

use super::form::{Coefficient, Evaluation, Form};
use super::ntt::{self, NttFriendly};
use super::rns_ops as ring_rns_ops;
use super::rns_reshape as ring_rns_reshape;

/// A polynomial in $R_{n, Q}$ under the two-prime RNS decomposition
/// $Q = q^{(0)} \cdot q^{(1)}$, in the form indicated by `F`.
///
/// `PolyRns<N, B, Coefficient>` carries the monomial-basis coefficients
/// per RNS slot; `PolyRns<N, B, Evaluation>` carries the negacyclic-NTT
/// evaluations per slot. Mixing forms is a compile error.
#[repr(C, align(32))]
pub struct PolyRns<const N: usize, B: RnsBasis, F: Form> {
    /// Canonical-reduced `u64` values for the first RNS slot
    /// (each in $[0, q^{(0)})$).
    values0: [u64; N],
    /// Canonical-reduced `u64` values for the second RNS slot
    /// (each in $[0, q^{(1)})$).
    values1: [u64; N],
    basis: B,
    _form: PhantomData<F>,
}

// ---------------------------------------------------------------------------
// `_CHECK` shared across all (`N`, `B`, `F`).
// ---------------------------------------------------------------------------

impl<const N: usize, B: RnsBasis, F: Form> PolyRns<N, B, F> {
    const _CHECK: () = {
        assert!(N >= 2, "PolyRns: N >= 2");
        assert!(N.is_power_of_two(), "PolyRns: N must be a power of two");
    };

    /// The ring degree. Touches [`Self::_CHECK`] at monomorphisation.
    pub const N: usize = {
        let () = Self::_CHECK;
        N
    };

    /// The RNS basis this polynomial is associated with.
    #[inline(always)]
    pub const fn basis(&self) -> B
    where
        B: Copy,
    {
        self.basis
    }

    /// Borrow the first slot's value buffer.
    #[inline(always)]
    pub const fn values0(&self) -> &[u64; N] {
        &self.values0
    }

    /// Borrow the second slot's value buffer.
    #[inline(always)]
    pub const fn values1(&self) -> &[u64; N] {
        &self.values1
    }

    /// Construct from per-slot `u64` arrays that are **already in canonical
    /// reduced form** for their respective slot moduli.
    ///
    /// # Safety
    ///
    /// Caller must guarantee `values0[i] < basis.m0().q()` and
    /// `values1[i] < basis.m1().q()` for every `i`. Misuse does not cause
    /// memory-safety UB but silently corrupts downstream cryptographic
    /// arithmetic.
    #[inline(always)]
    pub const unsafe fn from_reduced_unchecked(
        basis: B,
        values0: [u64; N],
        values1: [u64; N],
    ) -> Self {
        Self {
            values0,
            values1,
            basis,
            _form: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Form-neutral constructors
// ---------------------------------------------------------------------------

impl<const N: usize, B: RnsBasis, F: Form> PolyRns<N, B, F> {
    /// The zero polynomial. The all-zeros buffer encodes zero in either form.
    #[inline(always)]
    pub fn zero(basis: B) -> Self {
        let () = Self::_CHECK;
        // SAFETY: 0 < q for every component modulus (both q >= 2).
        unsafe { Self::from_reduced_unchecked(basis, [0u64; N], [0u64; N]) }
    }

    /// Construct from arbitrary per-slot `u64` arrays, reducing each lane.
    ///
    /// Under `F = Evaluation` this trusts that the supplied values are
    /// already valid negacyclic-NTT evaluations — they are not re-NTT'd.
    ///
    /// # Evaluation form on non-NTT-friendly bases
    ///
    /// Form-neutral: accepts any `B: RnsBasis`, not just one where both
    /// `B::M0` and `B::M1` are `NttFriendly<N>`. Constructing
    /// `PolyRns<N, B, Evaluation>` over a basis whose component primes
    /// aren't NTT-friendly produces a per-slot raw $\mathbb{Z}_{q^{(i)}}^N$
    /// pair with **no underlying $R_{n, Q}$ polynomial** — same
    /// caveat as the single-prime [`super::element::Poly::new`].
    /// Production call sites should always go through
    /// [`PolyRns::into_eval`], which statically requires NTT-friendly
    /// component primes.
    pub fn new(basis: B, values0: [u64; N], values1: [u64; N]) -> Self {
        let () = Self::_CHECK;
        let m0 = basis.m0();
        let m1 = basis.m1();
        let mut r0 = [0u64; N];
        let mut r1 = [0u64; N];
        for i in 0..N {
            r0[i] = m0.reduce_u64(values0[i]);
            r1[i] = m1.reduce_u64(values1[i]);
        }
        // SAFETY: every lane is reduced via `Modulus::reduce_u64`.
        unsafe { Self::from_reduced_unchecked(basis, r0, r1) }
    }

    /// Construct from a single `u128` array, RNS-decomposing each lane via
    /// the basis. Convenience for callers who hold the "full $\mathbb{Z}_Q$"
    /// representation.
    pub fn from_u128_array(basis: B, values: &[u128; N]) -> Self {
        let () = Self::_CHECK;
        let mut r0 = [0u64; N];
        let mut r1 = [0u64; N];
        for i in 0..N {
            let (a0, a1) = basis.decompose_u128(values[i]);
            r0[i] = a0;
            r1[i] = a1;
        }
        // SAFETY: `decompose_u128` returns components in canonical range.
        unsafe { Self::from_reduced_unchecked(basis, r0, r1) }
    }

    /// Sample a uniformly random polynomial by drawing each lane
    /// independently via [`Zq::random`] on each slot.
    ///
    /// # Per-slot sampling order
    ///
    /// The RNG is consumed in **interleaved per-lane** order:
    /// `m0[0], m1[0], m0[1], m1[1], …, m0[N-1], m1[N-1]`. Equivalently,
    /// for each coefficient index $i$ in turn we draw from slot 0 then
    /// slot 1. The alternative — slot-major (`m0[0..N]` then
    /// `m1[0..N]`) — would produce a different SHAKE-256 byte
    /// alignment, so this order is part of the cross-language
    /// reproducibility contract (`.docs/primitives.md` §1.1). Pin
    /// this convention when the §1.1 SHAKE-256 PRG lands; deviating
    /// from it will silently desynchronise test vectors against the
    /// Python reference.
    ///
    /// # Evaluation form on non-NTT-friendly bases
    ///
    /// As with [`PolyRns::new`], constructing the `Evaluation` form
    /// over a basis whose component primes aren't `NttFriendly<N>`
    /// produces a uniform per-slot $\mathbb{Z}_{q^{(i)}}^N$ pair with
    /// no associated $R_{n, Q}$ polynomial. The "uniform in $R_{n, Q}$
    /// via NTT bijection" guarantee is conditional on NTT-friendliness
    /// of both component primes.
    pub fn random<R: RngCore + ?Sized>(basis: B, rng: &mut R) -> Self {
        let () = Self::_CHECK;
        let m0 = basis.m0();
        let m1 = basis.m1();
        let mut r0 = [0u64; N];
        let mut r1 = [0u64; N];
        for i in 0..N {
            r0[i] = Zq::random(m0, rng).to_u64();
            r1[i] = Zq::random(m1, rng).to_u64();
        }
        // SAFETY: `Zq::random` always returns a value in [0, q).
        unsafe { Self::from_reduced_unchecked(basis, r0, r1) }
    }
}

// ---------------------------------------------------------------------------
// Coefficient-form impl
// ---------------------------------------------------------------------------

impl<const N: usize, B: RnsBasis> PolyRns<N, B, Coefficient> {
    /// The constant polynomial $1$: `(1, 0, 0, …)` per slot.
    #[inline(always)]
    pub fn one(basis: B) -> Self {
        let () = Self::_CHECK;
        debug_assert!(basis.m0().q() >= 2, "PolyRns::one requires q0 >= 2");
        debug_assert!(basis.m1().q() >= 2, "PolyRns::one requires q1 >= 2");
        let mut r0 = [0u64; N];
        let mut r1 = [0u64; N];
        r0[0] = 1;
        r1[0] = 1;
        // SAFETY: 1 < q for both slots when each q >= 2; rest are zero.
        unsafe { Self::from_reduced_unchecked(basis, r0, r1) }
    }

    /// The coefficient at position $i$ as an [`RnsZq`].
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    #[inline(always)]
    pub fn coeff(&self, i: usize) -> RnsZq<B> {
        assert!(i < N, "PolyRns::coeff: index {i} out of range (N = {N})",);
        // SAFETY: stored values are in their slot range by invariant.
        unsafe { RnsZq::from_reduced_unchecked(self.basis, self.values0[i], self.values1[i]) }
    }

    /// Write the coefficient at position $i$. Asserts basis equality.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N` or if `v.basis() != self.basis()`.
    #[inline(always)]
    pub fn set_coeff(&mut self, i: usize, v: RnsZq<B>) {
        assert!(
            i < N,
            "PolyRns::set_coeff: index {i} out of range (N = {N})",
        );
        assert!(
            v.basis() == self.basis,
            "PolyRns::set_coeff: basis mismatch",
        );
        self.values0[i] = v.value0();
        self.values1[i] = v.value1();
    }

    /// Deterministic rotation: returns $X^k \cdot \mathrm{self}$ in
    /// $R_{n, Q}$. Calls [`super::ops::rotate_slice`] once per slot.
    ///
    /// # Secret-$k$: do not use this method
    ///
    /// Inherits the same constraint as the single-prime
    /// [`super::element::Poly::mul_x_pow`]: `k` is a **public**
    /// parameter and the implementation branches on it. Encrypted-
    /// exponent rotation belongs to the §4.4 `CRot` composite; this
    /// kernel is the building block, not the protocol-facing entry
    /// point.
    pub fn mul_x_pow(&self, k: usize) -> Self {
        let mut d0 = [0u64; N];
        let mut d1 = [0u64; N];
        ring_rns_ops::rotate_slice(
            self.basis,
            &mut d0,
            &mut d1,
            &self.values0,
            &self.values1,
            k,
        );
        // SAFETY: rotate_slice only permutes / negates canonical lanes.
        unsafe { Self::from_reduced_unchecked(self.basis, d0, d1) }
    }

    /// Lift every coefficient to its centred representation
    /// $\tilde c_i \in (-\lfloor Q/2 \rfloor, \lfloor Q/2 \rfloor]$ in
    /// `i128` — see §0.6 RNS variant. **Not constant-time** over the
    /// input values; for secret-data inputs use
    /// [`Self::to_centered_coeffs_ct`].
    pub fn to_centered_coeffs(&self, dst: &mut [i128; N]) {
        rns_ops::to_centered_i128_slice(self.basis, dst, &self.values0, &self.values1);
    }

    /// Constant-time variant of [`Self::to_centered_coeffs`]. Use when
    /// the polynomial is a secret (e.g. §3.4 secret-key rekeying, RNS
    /// variant).
    pub fn to_centered_coeffs_ct(&self, dst: &mut [i128; N]) {
        rns_ops::to_centered_i128_ct_slice(self.basis, dst, &self.values0, &self.values1);
    }
}

/// Evaluation-form conversion — only available when both RNS slot moduli
/// are [`NttFriendly<N>`].
impl<const N: usize, B: RnsBasis> PolyRns<N, B, Coefficient>
where
    B::M0: NttFriendly<N>,
    B::M1: NttFriendly<N>,
{
    /// Convert to evaluation form via per-slot forward negacyclic NTT
    /// (§0.4). Each slot is transformed independently. The eval-form
    /// buffer is in **bit-reversed** order — see
    /// [`super::element::Poly::into_eval`].
    ///
    /// # Secret-bearing inputs
    ///
    /// Inherits the same trust boundary as
    /// [`super::element::Poly::into_eval`]: `self` is consumed by
    /// value, and Rust does not guarantee either slot's source
    /// buffer is zeroed after the move. RNS secrets (e.g. a $q_1$
    /// ring-switch key sample, or any intermediate carrying
    /// coefficient statistics of $S_1$) must be routed through a
    /// `_zeroizing` wrapper rather than this method directly.
    /// Current call sites are non-secret; the wrapper lands with
    /// §2.1 / §3.3.
    #[inline]
    pub fn into_eval(self) -> PolyRns<N, B, Evaluation> {
        let mut b0 = self.values0;
        let mut b1 = self.values1;
        ntt::ntt_inplace::<N, B::M0>(self.basis.m0(), &mut b0);
        ntt::ntt_inplace::<N, B::M1>(self.basis.m1(), &mut b1);
        // SAFETY: ntt_inplace preserves canonical reduction per slot.
        unsafe { PolyRns::<N, B, Evaluation>::from_reduced_unchecked(self.basis, b0, b1) }
    }
}

// ---------------------------------------------------------------------------
// §0.5 ring embedding / projection — RNS, coefficient-form only.
// ---------------------------------------------------------------------------

/// Single-slot embed, single-slot project, $d$-fold pack, $d$-fold
/// unpack — see [`super::element::Poly`] for the single-prime analogues
/// and [`super::reshape`] for the underlying kernels.
impl<const N: usize, B: RnsBasis> PolyRns<N, B, Coefficient> {
    /// $\iota_j^{N \to N_\text{large}}$ per RNS slot. Place `self` into
    /// slot `slot` of a polynomial in $R_{N_\text{large}, Q}$.
    ///
    /// # Panics
    ///
    /// Same compile-time vs runtime split as the single-prime
    /// [`super::element::Poly::embed_at`]: the const-generic
    /// $N$ / $N_\text{large}$ relationship is checked at
    /// monomorphisation, but the runtime `slot < d` bound is a
    /// runtime `assert!`, **not** a compile error.
    pub fn embed_at<const N_LARGE: usize>(&self, slot: usize) -> PolyRns<N_LARGE, B, Coefficient> {
        const {
            assert!(N_LARGE >= N, "embed_at: N_LARGE >= N");
            assert!(N_LARGE.is_multiple_of(N), "embed_at: N must divide N_LARGE");
            assert!(
                N_LARGE.is_power_of_two(),
                "embed_at: N_LARGE must be a power of two",
            );
        }
        let mut b0 = [0u64; N_LARGE];
        let mut b1 = [0u64; N_LARGE];
        ring_rns_reshape::embed_at_slice(
            self.basis,
            &self.values0,
            &self.values1,
            &mut b0,
            &mut b1,
            slot,
        );
        // SAFETY: embed_at_slice writes only zeros or copies of canonical lanes.
        unsafe { PolyRns::<N_LARGE, B, Coefficient>::from_reduced_unchecked(self.basis, b0, b1) }
    }

    /// $\pi_j^{N \to N_\text{small}}$ per RNS slot. Extract slot `slot`
    /// of `self` into a polynomial in $R_{N_\text{small}, Q}$.
    ///
    /// # Panics
    ///
    /// Same compile-time vs runtime split as the single-prime
    /// [`super::element::Poly::project_at`].
    pub fn project_at<const N_SMALL: usize>(
        &self,
        slot: usize,
    ) -> PolyRns<N_SMALL, B, Coefficient> {
        const {
            assert!(N_SMALL <= N, "project_at: N_SMALL <= N");
            assert!(
                N.is_multiple_of(N_SMALL),
                "project_at: N_SMALL must divide N"
            );
            assert!(
                N_SMALL.is_power_of_two(),
                "project_at: N_SMALL must be a power of two",
            );
        }
        let mut b0 = [0u64; N_SMALL];
        let mut b1 = [0u64; N_SMALL];
        ring_rns_reshape::project_at_slice(
            self.basis,
            &self.values0,
            &self.values1,
            &mut b0,
            &mut b1,
            slot,
        );
        // SAFETY: project_at_slice copies canonical lanes per slot.
        unsafe { PolyRns::<N_SMALL, B, Coefficient>::from_reduced_unchecked(self.basis, b0, b1) }
    }

    /// $d$-fold packing per RNS slot. Pack `slots[0..d]` into one
    /// polynomial in $R_{N_\text{large}, Q}$. All slot polys must share
    /// the basis.
    ///
    /// # Panics
    ///
    /// Same compile-time vs runtime split as the single-prime
    /// [`super::element::Poly::pack_slots`]: the const-generic
    /// $N$ / $N_\text{large}$ checks fire at monomorphisation; the
    /// `slots.len()` and per-slot `basis` checks are runtime `assert!`s.
    pub fn pack_slots<const N_LARGE: usize>(
        basis: B,
        slots: &[PolyRns<N, B, Coefficient>],
    ) -> PolyRns<N_LARGE, B, Coefficient> {
        const {
            assert!(N_LARGE >= N, "pack_slots: N_LARGE >= N");
            assert!(
                N_LARGE.is_multiple_of(N),
                "pack_slots: N must divide N_LARGE"
            );
            assert!(
                N_LARGE.is_power_of_two(),
                "pack_slots: N_LARGE must be a power of two",
            );
        }
        let d = N_LARGE / N;
        assert_eq!(
            slots.len(),
            d,
            "pack_slots: expected {d} slot polys, got {}",
            slots.len(),
        );
        // Direct scatter onto both RNS rails: `packed_k[d*i + j] =
        // slots[j].values_k[i]` for $k \in \{0, 1\}$. Inlining the
        // permutation removes the two `[u64; N_LARGE]` concatenation
        // scratch buffers (32 KiB at paper $N_\mathrm{large} = 2048$).
        let mut packed0 = [0u64; N_LARGE];
        let mut packed1 = [0u64; N_LARGE];
        for (j, slot_poly) in slots.iter().enumerate() {
            assert!(
                slot_poly.basis == basis,
                "pack_slots: basis mismatch at slot {j}",
            );
            for i in 0..N {
                packed0[d * i + j] = slot_poly.values0[i];
                packed1[d * i + j] = slot_poly.values1[i];
            }
        }
        // SAFETY: every written lane on each rail is an exact copy of
        // a canonical-form lane from some `slot_poly`; unwritten lanes
        // are 0. Equivalent to the kernel-based permutation.
        unsafe {
            PolyRns::<N_LARGE, B, Coefficient>::from_reduced_unchecked(basis, packed0, packed1)
        }
    }

    /// $d$-fold unpacking per RNS slot. Split `self` into `d` slot
    /// polynomials in $R_{N_\text{small}, Q}$.
    ///
    /// # Panics
    ///
    /// Same compile-time vs runtime split as the single-prime
    /// [`super::element::Poly::unpack_slots`].
    pub fn unpack_slots<const N_SMALL: usize>(
        &self,
        dsts: &mut [PolyRns<N_SMALL, B, Coefficient>],
    ) {
        const {
            assert!(N_SMALL <= N, "unpack_slots: N_SMALL <= N");
            assert!(
                N.is_multiple_of(N_SMALL),
                "unpack_slots: N_SMALL must divide N"
            );
            assert!(
                N_SMALL.is_power_of_two(),
                "unpack_slots: N_SMALL must be a power of two",
            );
        }
        let d = N / N_SMALL;
        assert_eq!(
            dsts.len(),
            d,
            "unpack_slots: expected {d} slot polys, got {}",
            dsts.len(),
        );
        // Direct gather on both rails: `dsts[j].values_k[i] =
        // self.values_k[d*i + j]` for $k \in \{0, 1\}$. Removes the
        // two `[u64; N]` concatenation scratch buffers used by the
        // earlier kernel-based shape.
        for (j, slot_poly) in dsts.iter_mut().enumerate() {
            assert!(
                slot_poly.basis == self.basis,
                "unpack_slots: basis mismatch at slot {j}",
            );
            for i in 0..N_SMALL {
                slot_poly.values0[i] = self.values0[d * i + j];
                slot_poly.values1[i] = self.values1[d * i + j];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation-form impl
// ---------------------------------------------------------------------------

impl<const N: usize, B: RnsBasis> PolyRns<N, B, Evaluation> {
    /// The evaluation at the $i$-th negacyclic-NTT point as an [`RnsZq`].
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    #[inline(always)]
    pub fn eval(&self, i: usize) -> RnsZq<B> {
        assert!(i < N, "PolyRns::eval: index {i} out of range (N = {N})",);
        // SAFETY: stored values are in their slot range by invariant.
        unsafe { RnsZq::from_reduced_unchecked(self.basis, self.values0[i], self.values1[i]) }
    }

    /// Write the evaluation at the $i$-th NTT point. Asserts basis equality.
    #[inline(always)]
    pub fn set_eval(&mut self, i: usize, v: RnsZq<B>) {
        assert!(i < N, "PolyRns::set_eval: index {i} out of range (N = {N})",);
        assert!(v.basis() == self.basis, "PolyRns::set_eval: basis mismatch",);
        self.values0[i] = v.value0();
        self.values1[i] = v.value1();
    }
}

/// Coefficient-form conversion — only available when both RNS slot moduli
/// are [`NttFriendly<N>`].
impl<const N: usize, B: RnsBasis> PolyRns<N, B, Evaluation>
where
    B::M0: NttFriendly<N>,
    B::M1: NttFriendly<N>,
{
    /// Convert back to coefficient form via per-slot inverse NTT (§0.4).
    /// Consumes bit-reversed eval-form input per slot; produces
    /// natural-coefficient output per slot.
    ///
    /// # Secret-bearing inputs
    ///
    /// Same trust boundary as
    /// [`PolyRns::<N, B, Coefficient>::into_eval`]: `self` is
    /// consumed by value and neither slot's source buffer is
    /// guaranteed zeroed after the move. Wrap secret-bearing
    /// inputs through a `_zeroizing` variant when those types land
    /// at §2.1 / §3.3; non-secret current call sites use this
    /// method directly.
    #[inline]
    pub fn into_coeff(self) -> PolyRns<N, B, Coefficient> {
        let mut b0 = self.values0;
        let mut b1 = self.values1;
        ntt::intt_inplace::<N, B::M0>(self.basis.m0(), &mut b0);
        ntt::intt_inplace::<N, B::M1>(self.basis.m1(), &mut b1);
        // SAFETY: intt_inplace preserves canonical reduction per slot.
        unsafe { PolyRns::<N, B, Coefficient>::from_reduced_unchecked(self.basis, b0, b1) }
    }
}

// ---------------------------------------------------------------------------
// Operator overloads — Coefficient form
// ---------------------------------------------------------------------------

impl<const N: usize, B: RnsBasis> Add for PolyRns<N, B, Coefficient> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "PolyRns::add: basis mismatch");
        let lhs0 = self.values0;
        let lhs1 = self.values1;
        rns_ops::add_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &lhs0,
            &lhs1,
            &rhs.values0,
            &rhs.values1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> Sub for PolyRns<N, B, Coefficient> {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "PolyRns::sub: basis mismatch");
        let lhs0 = self.values0;
        let lhs1 = self.values1;
        rns_ops::sub_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &lhs0,
            &lhs1,
            &rhs.values0,
            &rhs.values1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> Neg for PolyRns<N, B, Coefficient> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        let src0 = self.values0;
        let src1 = self.values1;
        rns_ops::neg_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &src0,
            &src1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> Mul for PolyRns<N, B, Coefficient> {
    type Output = Self;
    /// Schoolbook negacyclic multiplication per RNS slot — $O(N^2)$.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "PolyRns::mul: basis mismatch");
        let mut d0 = [0u64; N];
        let mut d1 = [0u64; N];
        ring_rns_ops::negacyclic_mul_slice(
            self.basis,
            &mut d0,
            &mut d1,
            &self.values0,
            &self.values1,
            &rhs.values0,
            &rhs.values1,
        );
        // SAFETY: negacyclic_mul_slice writes canonical-reduced lanes per slot.
        unsafe { Self::from_reduced_unchecked(self.basis, d0, d1) }
    }
}

impl<const N: usize, B: RnsBasis> Mul<u64> for PolyRns<N, B, Coefficient> {
    type Output = Self;
    /// Scalar multiplication. The scalar is reduced into each slot's
    /// prime range before per-lane multiply.
    #[inline]
    fn mul(mut self, scalar: u64) -> Self {
        let s0 = self.basis.m0().reduce_u64(scalar);
        let s1 = self.basis.m1().reduce_u64(scalar);
        let src0 = self.values0;
        let src1 = self.values1;
        rns_ops::scalar_mul_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &src0,
            &src1,
            s0,
            s1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> AddAssign for PolyRns<N, B, Coefficient> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize, B: RnsBasis> SubAssign for PolyRns<N, B, Coefficient> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize, B: RnsBasis> MulAssign for PolyRns<N, B, Coefficient> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize, B: RnsBasis> MulAssign<u64> for PolyRns<N, B, Coefficient> {
    #[inline]
    fn mul_assign(&mut self, scalar: u64) {
        *self = *self * scalar;
    }
}

// ---------------------------------------------------------------------------
// Operator overloads — Evaluation form
// ---------------------------------------------------------------------------

impl<const N: usize, B: RnsBasis> Add for PolyRns<N, B, Evaluation> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "PolyRns::add: basis mismatch");
        let lhs0 = self.values0;
        let lhs1 = self.values1;
        rns_ops::add_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &lhs0,
            &lhs1,
            &rhs.values0,
            &rhs.values1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> Sub for PolyRns<N, B, Evaluation> {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "PolyRns::sub: basis mismatch");
        let lhs0 = self.values0;
        let lhs1 = self.values1;
        rns_ops::sub_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &lhs0,
            &lhs1,
            &rhs.values0,
            &rhs.values1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> Neg for PolyRns<N, B, Evaluation> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        let src0 = self.values0;
        let src1 = self.values1;
        rns_ops::neg_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &src0,
            &src1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> Mul for PolyRns<N, B, Evaluation> {
    type Output = Self;
    /// Pointwise (Hadamard) multiplication per RNS slot — $O(N)$.
    #[inline]
    fn mul(mut self, rhs: Self) -> Self {
        assert!(self.basis == rhs.basis, "PolyRns::mul: basis mismatch");
        let lhs0 = self.values0;
        let lhs1 = self.values1;
        rns_ops::mul_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &lhs0,
            &lhs1,
            &rhs.values0,
            &rhs.values1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> Mul<u64> for PolyRns<N, B, Evaluation> {
    type Output = Self;
    #[inline]
    fn mul(mut self, scalar: u64) -> Self {
        let s0 = self.basis.m0().reduce_u64(scalar);
        let s1 = self.basis.m1().reduce_u64(scalar);
        let src0 = self.values0;
        let src1 = self.values1;
        rns_ops::scalar_mul_slice(
            self.basis,
            &mut self.values0,
            &mut self.values1,
            &src0,
            &src1,
            s0,
            s1,
        );
        self
    }
}

impl<const N: usize, B: RnsBasis> AddAssign for PolyRns<N, B, Evaluation> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize, B: RnsBasis> SubAssign for PolyRns<N, B, Evaluation> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize, B: RnsBasis> MulAssign for PolyRns<N, B, Evaluation> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize, B: RnsBasis> MulAssign<u64> for PolyRns<N, B, Evaluation> {
    #[inline]
    fn mul_assign(&mut self, scalar: u64) {
        *self = *self * scalar;
    }
}

// ---------------------------------------------------------------------------
// Cross-form trait impls
// ---------------------------------------------------------------------------

impl<const N: usize, B: RnsBasis, F: Form> Copy for PolyRns<N, B, F> {}

impl<const N: usize, B: RnsBasis, F: Form> Clone for PolyRns<N, B, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const N: usize, B: RnsBasis, F: Form> PartialEq for PolyRns<N, B, F> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.basis == other.basis && self.values0 == other.values0 && self.values1 == other.values1
    }
}

impl<const N: usize, B: RnsBasis, F: Form> Eq for PolyRns<N, B, F> {}

impl<const N: usize, B: RnsBasis, F: Form> ConstantTimeEq for PolyRns<N, B, F> {
    /// Constant-time equality on both slot value lanes. Caller must
    /// already know both operands share a basis.
    #[inline]
    fn ct_eq(&self, other: &Self) -> Choice {
        let mut acc = Choice::from(1u8);
        for i in 0..N {
            acc &= self.values0[i].ct_eq(&other.values0[i]);
            acc &= self.values1[i].ct_eq(&other.values1[i]);
        }
        acc
    }
}

impl<const N: usize, B: RnsBasis, F: Form> ConditionallySelectable for PolyRns<N, B, F> {
    #[inline]
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        assert!(
            a.basis == b.basis,
            "PolyRns::conditional_select: basis mismatch",
        );
        let mut o0 = [0u64; N];
        let mut o1 = [0u64; N];
        for i in 0..N {
            o0[i] = u64::conditional_select(&a.values0[i], &b.values0[i], choice);
            o1[i] = u64::conditional_select(&a.values1[i], &b.values1[i], choice);
        }
        // SAFETY: each output lane is one of `a` / `b`, already canonical.
        unsafe { Self::from_reduced_unchecked(a.basis, o0, o1) }
    }
}

impl<const N: usize, B: RnsBasis, F: Form> Zeroize for PolyRns<N, B, F> {
    #[inline]
    fn zeroize(&mut self) {
        for v in &mut self.values0 {
            v.zeroize();
        }
        for v in &mut self.values1 {
            v.zeroize();
        }
    }
}

impl<const N: usize, B: RnsBasis, F: Form> Hash for PolyRns<N, B, F> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.values0.hash(state);
        self.values1.hash(state);
        self.basis.m0().q().hash(state);
        self.basis.m1().q().hash(state);
        F::HASH_TAG.hash(state);
    }
}

impl<const N: usize, B: RnsBasis, F: Form> fmt::Debug for PolyRns<N, B, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "PolyRns<{}, q0={}, q1={}, {:?}>([…])",
            N,
            self.basis.m0().q(),
            self.basis.m1().q(),
            F::default(),
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::rns::basis::{ConstRnsBasis, DynRnsBasis, paper};
    use crate::primitives::zq::modulus::DynModulus;

    type Z55 = ConstRnsBasis<5, 11>;

    #[test]
    fn zero_one_at_tiny_basis() {
        let b = Z55::default();
        let z: PolyRns<4, _, Coefficient> = PolyRns::zero(b);
        let o: PolyRns<4, _, Coefficient> = PolyRns::one(b);
        for i in 0..4 {
            assert_eq!(z.coeff(i).value0(), 0);
            assert_eq!(z.coeff(i).value1(), 0);
        }
        assert_eq!(o.coeff(0).to_u128(), 1);
        for i in 1..4 {
            assert_eq!(o.coeff(i).to_u128(), 0);
        }
    }

    #[test]
    fn new_reduces_each_lane_per_slot() {
        let b = Z55::default();
        let p: PolyRns<4, _, Coefficient> = PolyRns::new(b, [6, 5, 0, 7], [12, 0, 11, 13]);
        assert_eq!(p.values0(), &[1u64, 0, 0, 2]); // mod 5
        assert_eq!(p.values1(), &[1u64, 0, 0, 2]); // mod 11
    }

    #[test]
    fn from_u128_array_decomposes_per_lane() {
        let b = Z55::default();
        let xs = [0u128, 1, 23, 54];
        let p: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        for (i, &x) in xs.iter().enumerate() {
            let (a0, a1) = b.decompose_u128(x);
            assert_eq!(p.values0()[i], a0);
            assert_eq!(p.values1()[i], a1);
        }
    }

    #[test]
    fn rns_poly_add_matches_per_slot() {
        let b = paper::ViaQ1Rns::default();
        let a_u: [u128; 4] = [12345, (1u128 << 50) + 7, b.big_q() - 1, 999_999_999_999_999];
        let r_u: [u128; 4] = [54321, 99, b.big_q() / 3, 1];
        let a: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &a_u);
        let r: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &r_u);
        let sum = a + r;
        for i in 0..4 {
            let want = (a_u[i] + r_u[i]) % b.big_q();
            assert_eq!(sum.coeff(i).to_u128(), want, "i={i}");
        }
    }

    #[test]
    fn rns_poly_negacyclic_mul_matches_reconstruct_then_mul() {
        // Tiny basis (Z_55) at small N so we can compare to a u128
        // reference schoolbook mod (X^4 + 1, 55).
        let b = Z55::default();
        let f_u: [u128; 4] = [1, 2, 3, 4];
        let g_u: [u128; 4] = [5, 6, 7, 8];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &f_u);
        let g: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &g_u);
        let got = f * g;
        // Reference: schoolbook in u128 with explicit -1 wrap, then mod Q.
        let mut acc = [0i128; 4];
        for i in 0..4 {
            for j in 0..4 {
                let p = (f_u[i] as i128) * (g_u[j] as i128);
                if i + j < 4 {
                    acc[i + j] += p;
                } else {
                    acc[i + j - 4] -= p;
                }
            }
        }
        let q = b.big_q() as i128;
        for (i, &a) in acc.iter().enumerate() {
            let want = a.rem_euclid(q) as u128;
            assert_eq!(got.coeff(i).to_u128(), want, "i={i}");
        }
    }

    #[test]
    fn rns_poly_negacyclic_mul_at_paper_basis() {
        // Same shape, but at the paper q_1 RNS (≈ 2^57). Reference is u128
        // schoolbook; inputs are small to keep products well below 2^126.
        let b = paper::ViaQ1Rns::default();
        let f_u: [u128; 4] = [1234, 5678, 9012, 3456];
        let g_u: [u128; 4] = [7890, 1, b.big_q() - 1, 1000];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &f_u);
        let g: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &g_u);
        let got = f * g;
        let q = b.big_q() as i128;
        let mut acc = [0i128; 4];
        for i in 0..4 {
            for j in 0..4 {
                let p = (f_u[i] as i128) * (g_u[j] as i128);
                if i + j < 4 {
                    acc[i + j] += p;
                } else {
                    acc[i + j - 4] -= p;
                }
            }
        }
        for (i, &a) in acc.iter().enumerate() {
            let want = a.rem_euclid(q) as u128;
            assert_eq!(got.coeff(i).to_u128(), want, "i={i}");
        }
    }

    #[test]
    fn rns_mul_x_pow_matches_single_prime_per_slot() {
        // Verify PolyRns::mul_x_pow agrees with running the single-prime
        // mul_x_pow on each slot independently.
        use crate::primitives::ring::element::Poly;
        let b = paper::ViaQ1Rns::default();
        let xs: [u128; 4] = [12345, 67890, b.big_q() - 1, 999];
        let p: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        for k in [0usize, 1, 2, 3, 4, 5, 8, 17] {
            let rot = p.mul_x_pow(k);
            // Decompose into single-prime polys, rotate each independently.
            let p0: Poly<4, _, Coefficient> = Poly::new(b.m0(), *p.values0());
            let p1: Poly<4, _, Coefficient> = Poly::new(b.m1(), *p.values1());
            let r0 = p0.mul_x_pow(k);
            let r1 = p1.mul_x_pow(k);
            assert_eq!(rot.values0(), r0.values(), "k={k} slot0");
            assert_eq!(rot.values1(), r1.values(), "k={k} slot1");
        }
    }

    #[test]
    #[should_panic(expected = "basis mismatch")]
    fn rns_poly_add_panics_on_basis_mismatch() {
        let b1 = DynRnsBasis::new(DynModulus::new(5), DynModulus::new(11));
        let b2 = DynRnsBasis::new(DynModulus::new(7), DynModulus::new(13));
        let a: PolyRns<4, _, Coefficient> = PolyRns::new(b1, [0; 4], [0; 4]);
        let r: PolyRns<4, _, Coefficient> = PolyRns::new(b2, [0; 4], [0; 4]);
        let _ = a + r;
    }

    #[test]
    fn rns_const_vs_dyn_paper_q1() {
        let c = paper::ViaQ1Rns::default();
        let d = DynRnsBasis::new(DynModulus::new(268369921), DynModulus::new(536608769));
        let xs: [u128; 4] = [
            0,
            1,
            (1u128 << 50) + 7,
            12345678901234567890u128 % c.big_q(),
        ];
        let ys: [u128; 4] = [c.big_q() - 1, 42, 1u128 << 40, 1];
        let f_c: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(c, &xs);
        let g_c: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(c, &ys);
        let f_d: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(d, &xs);
        let g_d: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(d, &ys);
        // Add
        let s_c = f_c + g_c;
        let s_d = f_d + g_d;
        for i in 0..4 {
            assert_eq!(s_c.coeff(i).to_u128(), s_d.coeff(i).to_u128(), "add i={i}");
        }
        // Mul
        let p_c = f_c * g_c;
        let p_d = f_d * g_d;
        for i in 0..4 {
            assert_eq!(p_c.coeff(i).to_u128(), p_d.coeff(i).to_u128(), "mul i={i}");
        }
        // Rotate
        let r_c = f_c.mul_x_pow(3);
        let r_d = f_d.mul_x_pow(3);
        for i in 0..4 {
            assert_eq!(r_c.coeff(i).to_u128(), r_d.coeff(i).to_u128(), "rot i={i}");
        }
    }

    #[test]
    fn eval_form_add_sub_on_zero() {
        let b = Z55::default();
        let z: PolyRns<4, _, Evaluation> = PolyRns::zero(b);
        let s: PolyRns<4, _, Evaluation> = PolyRns::new(b, [1, 2, 3, 4], [4, 3, 2, 1]);
        assert_eq!(z + s, s);
        assert_eq!(s - z, s);
        assert_eq!(-z, z);
    }

    #[test]
    fn eval_pointwise_mul_zero_yields_zero() {
        let b = Z55::default();
        let z: PolyRns<4, _, Evaluation> = PolyRns::zero(b);
        let s: PolyRns<4, _, Evaluation> = PolyRns::new(b, [1, 2, 3, 4], [4, 3, 2, 1]);
        assert_eq!(z * s, z);
    }

    #[test]
    fn rns_poly_zeroize_clears_both_slots() {
        let b = paper::ViaQ1Rns::default();
        let mut p: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &[1u128, 2, 3, 4]);
        p.zeroize();
        assert_eq!(p.values0(), &[0u64; 4]);
        assert_eq!(p.values1(), &[0u64; 4]);
    }

    #[test]
    fn rns_poly_scalar_mul() {
        let b = Z55::default();
        let xs: [u128; 4] = [1, 23, 30, 54];
        let p: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        let got = p * 7u64;
        for (i, &x) in xs.iter().enumerate() {
            let want = (x * 7) % 55;
            assert_eq!(got.coeff(i).to_u128(), want, "i={i}");
        }
    }

    #[test]
    fn rns_poly_random_in_range() {
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
        let p: PolyRns<8, _, Coefficient> = PolyRns::random(b, &mut rng);
        for i in 0..8 {
            assert!(p.values0()[i] < b.m0().q());
            assert!(p.values1()[i] < b.m1().q());
        }
    }

    /// NTT round-trip identity on `PolyRns` at the VIA q_1 paper basis,
    /// small $N = 4$. Both RNS slot primes are NTT-friendly at $N = 2048$
    /// per paper §A.1 (and hence at any $N$ dividing that), so $N = 4$
    /// works.
    #[test]
    fn rns_ntt_roundtrip_at_paper_q1_small_n() {
        let b = paper::ViaQ1Rns::default();
        let xs: [u128; 4] = [12345, b.big_q() - 1, (1u128 << 50) + 7, 999_999_999_999];
        let p: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        let back = p.into_eval().into_coeff();
        assert_eq!(back, p);
    }

    /// RNS NTT homomorphism: per-slot schoolbook negacyclic mul matches
    /// NTT-mediated pointwise mul + INTT round-trip.
    #[test]
    fn rns_ntt_homomorphism_at_paper_q1_small_n() {
        let b = paper::ViaQ1Rns::default();
        let f_u: [u128; 4] = [1234, 5678, 9012, 3456];
        let g_u: [u128; 4] = [7890, 1, b.big_q() - 1, 1000];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &f_u);
        let g: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &g_u);
        let schoolbook = f * g;
        let via_ntt = (f.into_eval() * g.into_eval()).into_coeff();
        assert_eq!(via_ntt, schoolbook);
    }

    /// `PolyRns::set_eval(i, v)` then `eval(i)` round-trips the value
    /// at a chosen NTT point — exercises the typed eval-form
    /// accessor that the existing unit tests never touch (they all
    /// construct via `PolyRns::new` on raw `[u64; N]` arrays).
    /// Closes review item 18 (RNS side).
    #[test]
    fn rns_poly_eval_set_eval_roundtrip() {
        let b = paper::ViaQ1Rns::default();
        let mut p: PolyRns<8, _, Evaluation> = PolyRns::zero(b);
        let v = RnsZq::from_u128(b, 12_345_678_901u128);
        p.set_eval(3, v);
        // Untouched lanes stay zero; the written lane reads back exactly.
        assert_eq!(p.eval(3), v);
        assert_eq!(p.eval(0), RnsZq::zero(b));
        assert_eq!(p.eval(7), RnsZq::zero(b));
    }

    /// Per-slot bit-reversed-index convention dispatches correctly
    /// inside `PolyRns::into_eval`. The single-prime kernel-level
    /// convention is already pinned by
    /// `ntt::tests::ntt_forward_known_input_q17_n8` and
    /// `poly_eval_pins_bit_reversed_index`; this test confirms that
    /// the RNS wrapper just dispatches per-slot without any extra
    /// permutation. Closes review item 17/18 (RNS side).
    #[test]
    fn rns_poly_eval_dispatches_per_slot_bit_reversed_layout() {
        use crate::primitives::ring::element::Poly;
        use crate::primitives::ring::form::Coefficient as SingleCoefficient;
        let b = paper::ViaQ1Rns::default();
        let q0 = b.m0().q();
        let q1 = b.m1().q();
        // Distinct per-slot inputs so the test fails if the wrapper
        // accidentally evaluates the same slot twice or swaps them.
        let v0: [u64; 8] = core::array::from_fn(|i| (i as u64 * 13 + 1) % q0);
        let v1: [u64; 8] = core::array::from_fn(|i| (i as u64 * 7 + 11) % q1);
        let p: PolyRns<8, _, Coefficient> = PolyRns::new(b, v0, v1);
        let e = p.into_eval();
        // Reference: run the single-prime NTT on each slot directly.
        let s0: Poly<8, _, SingleCoefficient> = Poly::new(b.m0(), v0);
        let s1: Poly<8, _, SingleCoefficient> = Poly::new(b.m1(), v1);
        let e0 = s0.into_eval();
        let e1 = s1.into_eval();
        for i in 0..8 {
            assert_eq!(e.eval(i).value0(), e0.eval(i).to_u64(), "slot0 lane {i}");
            assert_eq!(e.eval(i).value1(), e1.eval(i).to_u64(), "slot1 lane {i}");
        }
    }

    /// Round-trip at paper $N = 2048$, VIA-C q_1 RNS basis. Locks the
    /// realistic-size code path through both RNS slots.
    #[test]
    fn rns_ntt_roundtrip_at_paper_via_c_q1_n2048() {
        let b = paper::ViaCQ1Rns::default();
        // Deterministic non-zero pattern at every lane.
        let mut p: PolyRns<2048, _, Coefficient> = PolyRns::zero(b);
        for i in 0..2048 {
            let v = RnsZq::from_u128(b, (i as u128) * 1234567 + 1);
            p.set_coeff(i, v);
        }
        let back = p.into_eval().into_coeff();
        assert_eq!(back, p);
    }

    // ----- §0.5 RNS ring embedding / projection tests -----

    /// Round-trip $\pi_j \circ \iota_j = \mathrm{id}$ on `PolyRns`, every slot.
    #[test]
    fn rns_poly_embed_project_roundtrip_n4_into_n16() {
        let b = paper::ViaQ1Rns::default();
        let xs: [u128; 4] = [12345, b.big_q() - 1, (1u128 << 40) + 7, 999];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        for j in 0..4usize {
            let big: PolyRns<16, _, Coefficient> = f.embed_at::<16>(j);
            let back: PolyRns<4, _, Coefficient> = big.project_at::<4>(j);
            assert_eq!(back, f, "j={j}");
        }
    }

    /// Slot disjointness for `PolyRns`.
    #[test]
    fn rns_poly_project_at_other_slot_is_zero() {
        let b = Z55::default();
        let xs: [u128; 4] = [1, 23, 30, 54];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        let zero: PolyRns<4, _, Coefficient> = PolyRns::zero(b);
        for j in 0..4usize {
            let big: PolyRns<16, _, Coefficient> = f.embed_at::<16>(j);
            for jp in 0..4usize {
                if jp == j {
                    continue;
                }
                let back = big.project_at::<4>(jp);
                assert_eq!(back, zero, "embed {j}, project {jp}");
            }
        }
    }

    /// $d$-fold pack/unpack identity on `PolyRns`.
    #[test]
    fn rns_poly_pack_unpack_identity_n4_into_n16() {
        let b = paper::ViaQ1Rns::default();
        let slots: [PolyRns<4, _, Coefficient>; 4] = [
            PolyRns::from_u128_array(b, &[1, 2, 3, 4]),
            PolyRns::from_u128_array(b, &[5, 6, 7, 8]),
            PolyRns::from_u128_array(b, &[9, 10, 11, 12]),
            PolyRns::from_u128_array(b, &[13, 14, 15, 16]),
        ];
        let packed: PolyRns<16, _, Coefficient> = PolyRns::pack_slots::<16>(b, &slots);
        let mut back: [PolyRns<4, _, Coefficient>; 4] = [PolyRns::zero(b); 4];
        packed.unpack_slots::<4>(&mut back);
        for j in 0..4 {
            assert_eq!(back[j], slots[j], "slot j={j}");
        }
    }

    /// `PolyRns::embed_at` delegates correctly to per-slot
    /// `Poly::embed_at`. Constructs the same embedding via single-prime
    /// path and compares per-slot value buffers.
    #[test]
    fn rns_poly_embed_at_matches_per_slot() {
        use crate::primitives::ring::element::Poly;
        let b = paper::ViaQ1Rns::default();
        let xs: [u128; 4] = [12345, 67890, b.big_q() - 1, 11];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        // Single-prime per-slot polys, embedded independently.
        let p0: Poly<4, _, Coefficient> = Poly::new(b.m0(), *f.values0());
        let p1: Poly<4, _, Coefficient> = Poly::new(b.m1(), *f.values1());
        for j in 0..4usize {
            let big_rns = f.embed_at::<16>(j);
            let big_p0: Poly<16, _, Coefficient> = p0.embed_at::<16>(j);
            let big_p1: Poly<16, _, Coefficient> = p1.embed_at::<16>(j);
            assert_eq!(big_rns.values0(), big_p0.values(), "j={j} slot0");
            assert_eq!(big_rns.values1(), big_p1.values(), "j={j} slot1");
        }
    }

    // ----- §0.6 RNS centred-coeffs tests -----

    /// Zero polynomial centred-lifts to all-zeros.
    #[test]
    fn polyrns_to_centered_coeffs_zero_is_zero() {
        let b = paper::ViaQ1Rns::default();
        let z: PolyRns<4, _, Coefficient> = PolyRns::zero(b);
        let mut dst = [0i128; 4];
        z.to_centered_coeffs(&mut dst);
        assert_eq!(dst, [0i128, 0, 0, 0]);
        let mut dst_ct = [0i128; 4];
        z.to_centered_coeffs_ct(&mut dst_ct);
        assert_eq!(dst_ct, [0i128, 0, 0, 0]);
    }

    /// `to_centered_coeffs` at the toy `Z55` basis matches a hand
    /// reference: coefficient $v \in [0, 55)$ centres to $v$ if
    /// $v \le 27$ else $v - 55$.
    #[test]
    fn polyrns_to_centered_coeffs_z55_hand_reference() {
        let b = Z55::default();
        let xs: [u128; 4] = [0, 27, 28, 54];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        let mut dst = [0i128; 4];
        f.to_centered_coeffs(&mut dst);
        assert_eq!(dst, [0i128, 27, -27, -1]);
    }

    /// CT centred-coeffs match the non-CT version pointwise at paper
    /// VIA q_1 basis ($Q \approx 2^{57}$).
    #[test]
    fn polyrns_to_centered_coeffs_ct_matches_non_ct() {
        let b = paper::ViaQ1Rns::default();
        let q = b.big_q();
        let half = q / 2;
        let xs: [u128; 4] = [0, half, half + 1, q - 1];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        let mut non_ct = [0i128; 4];
        let mut ct = [0i128; 4];
        f.to_centered_coeffs(&mut non_ct);
        f.to_centered_coeffs_ct(&mut ct);
        assert_eq!(non_ct, ct);
    }

    /// Round-trip identity at paper VIA-C q_1 basis ($Q \approx 2^{75}$):
    /// `to_centered_coeffs` then `from_u128_array((c + Q) as u128 % Q)`
    /// recovers the original polynomial.
    #[test]
    fn polyrns_to_centered_coeffs_roundtrip_paper_via_c_q1() {
        let b = paper::ViaCQ1Rns::default();
        let q = b.big_q();
        let xs: [u128; 4] = [12345, q - 1, q / 3, q / 2 + 7];
        let f: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &xs);
        let mut centred = [0i128; 4];
        f.to_centered_coeffs(&mut centred);
        // Re-lift: signed i128 back to u128 in [0, Q) via modular addition.
        let q_i = q as i128;
        let mut back_u128 = [0u128; 4];
        for (slot, &c) in back_u128.iter_mut().zip(centred.iter()) {
            let r = c.rem_euclid(q_i);
            *slot = r as u128;
        }
        let back: PolyRns<4, _, Coefficient> = PolyRns::from_u128_array(b, &back_u128);
        assert_eq!(back, f);
    }
}
