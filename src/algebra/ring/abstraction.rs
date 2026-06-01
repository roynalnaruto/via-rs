//! Backend-agnostic interface to a coefficient-form negacyclic ring polynomial.
//!
//! The [`RingPoly`] trait abstracts over the two concrete polynomial backends
//! shipped by Layer 0:
//!
//! - [`super::element::Poly<N, M, Coefficient>`] — single-prime, used at
//!   $q_2$, $q_3$, $q_4$, $p$ across every paper parameter set.
//! - [`super::rns_element::PolyRns<N, B, Coefficient>`] — RNS, used at the
//!   composite $q_1$ in realistic VIA-C / VIA-B parameters.
//!
//! Layer 2 (encryption types and primitive operations, see
//! `.docs/primitives.md` §2) is generic over `R: RingPoly<N>`, so the same
//! ciphertext-level code instantiates against either backend. The trait is
//! **sealed** via a private supertrait: only the two crate-internal
//! polynomial types may implement it.
//!
//! ## Why coefficient-form only
//!
//! Both polynomial backends carry a [`super::form::Form`] type-state
//! parameter ([`Coefficient`] / [`super::form::Evaluation`]). The methods
//! exposed by `RingPoly` (centred-coefficient extraction, monomial-basis
//! scalar access, `from_centered_i64s`) are only meaningful in the
//! coefficient basis; placing them on a form-parametric trait would be a
//! lie at every `Evaluation` call site. Layer 2 today lives entirely in
//! coefficient form (the Python reference does too, and the §0.4 NTT body
//! is currently `unimplemented!()`). When eval-form ciphertexts become
//! useful — e.g. once the NTT body lands and we want to keep ciphertexts
//! in eval form across many homomorphic operations — a sibling
//! `RingPolyEval<N>` trait will be introduced for the form-neutral subset.
//!
//! ## Associated-type story
//!
//! Single-prime and RNS polynomials disagree on the natural width of two
//! scalar types:
//!
//! - The polynomial's per-coefficient scalar is [`super::super::zq::element::Zq<M>`]
//!   in the single-prime case and [`super::super::rns::element::RnsZq<B>`]
//!   in the RNS case. Surfaced as `RingPoly::Scalar`.
//! - The *centred* lift of a coefficient is `i64` for single-prime moduli
//!   (which fit in 63 bits) and `i128` for RNS moduli (which can reach
//!   $\sim 2^{75}$ at paper VIA-C parameters; see `.docs/primitives.md` §0.6
//!   "RNS variant"). Surfaced as `RingPoly::CenteredScalar`.
//!
//! Layer-2 algorithms that read centred coefficients (`encode`, gadget
//! decomposition, key-switch noise inspection) parameterise over
//! `R::CenteredScalar`; arithmetic on this scalar is delegated to either
//! existing `i64` helpers (single-prime path) or the upcoming `i128` /
//! wider helpers (RNS path).

use core::ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign};

use rand_core::RngCore;
use zeroize::Zeroize;

use super::element::Poly;
use super::form::Coefficient;
use super::rns_element::PolyRns;
use crate::algebra::rns::basis::RnsBasis;
use crate::algebra::rns::element::RnsZq;
use crate::algebra::zq::element::Zq;
use crate::algebra::zq::modulus::Modulus;

mod sealed {
    pub trait Sealed {}
}

/// Coefficient-form negacyclic ring polynomial.
///
/// Sealed; implemented by [`Poly<N, M, Coefficient>`] (single-prime) and
/// [`PolyRns<N, B, Coefficient>`] (RNS). See the module docs for the
/// rationale behind the coefficient-only restriction.
pub trait RingPoly<const N: usize>:
    Sized
    + Copy
    + Clone
    + Eq
    + PartialEq
    + Add<Self, Output = Self>
    + Sub<Self, Output = Self>
    + Neg<Output = Self>
    + Mul<Self, Output = Self>
    + Mul<u64, Output = Self>
    + AddAssign<Self>
    + SubAssign<Self>
    + Zeroize
    + sealed::Sealed
{
    /// The modulus type: a [`Modulus`] impl for single-prime polynomials,
    /// or an [`RnsBasis`] impl for RNS polynomials.
    type Modulus: Copy + Eq;

    /// The per-coefficient scalar type: [`Zq<M>`] for single-prime,
    /// [`RnsZq<B>`] for RNS.
    type Scalar: Copy + Zeroize;

    /// The width of the centred lift of one coefficient: `i64` for
    /// single-prime, `i128` for RNS (composite `Q` can exceed `i64::MAX`).
    type CenteredScalar: Copy + Default;

    /// The result type of [`Self::project_at`]: the same backend at the
    /// smaller ring degree `N_SMALL`, over the *same* modulus, scalar, and
    /// centred-scalar types. The equality bounds let §3.3 RingSwitch combine
    /// projected masks and RSK samples generically across both backends —
    /// `Poly<N_SMALL, M, _>` for single-prime, `PolyRns<N_SMALL, B, _>` for
    /// RNS.
    type Projected<const N_SMALL: usize>: RingPoly<
            N_SMALL,
            Modulus = Self::Modulus,
            Scalar = Self::Scalar,
            CenteredScalar = Self::CenteredScalar,
        >;

    /// The modulus this polynomial is associated with.
    fn modulus(&self) -> Self::Modulus;

    /// The all-zeros polynomial.
    fn zero(modulus: Self::Modulus) -> Self;

    /// Sample a uniformly random polynomial in $R_{n, q}$ by drawing each
    /// coefficient independently via the underlying $\mathbb{Z}_q$ /
    /// RNS-`Z_Q` uniform sampler. Used to produce the RLWE mask $A$.
    fn random_uniform<R: RngCore + ?Sized>(modulus: Self::Modulus, rng: &mut R) -> Self;

    /// Lift a length-$N$ vector of centred `i64` samples (e.g. the output
    /// of [`crate::sampling::distribution::Distribution::sample_into`]) into
    /// a coefficient-form polynomial in $\mathbb{Z}_q$. Used to convert
    /// freshly sampled ternary / bounded-uniform / Gaussian error vectors
    /// into ring elements during encryption.
    fn from_centered_i64s(modulus: Self::Modulus, samples: &[i64; N]) -> Self;

    /// Lift a length-$N$ vector of centred `i128` samples into a
    /// coefficient-form polynomial in $\mathbb{Z}_q$. The `i128`-wide
    /// counterpart of [`Self::from_centered_i64s`]: it is the natural
    /// constructor for the §3.4 secret-key rekeying path when the *source*
    /// key lives at the RNS-composite $q_1$ (whose centred coefficients are
    /// `i128`, see §0.6 "RNS variant").
    ///
    /// For single-prime targets each sample must fit in `i64` (the centred
    /// lift of a small key is bounded by $\|S\|_\infty \ll 2^{63}$); the
    /// single-prime impl `debug_assert!`s this before narrowing.
    ///
    /// ```rust
    /// use via_rs::algebra::ring::abstraction::RingPoly;
    /// use via_rs::algebra::ring::element::Poly;
    /// use via_rs::algebra::ring::form::Coefficient;
    /// use via_rs::algebra::zq::modulus::ConstModulus;
    ///
    /// type P = Poly<4, ConstModulus<17>, Coefficient>;
    /// let samples = [-8i128, 0, 3, 7];
    /// let p = <P as RingPoly<4>>::from_centered_i128s(ConstModulus, &samples);
    /// let mut out = [0i64; 4];
    /// RingPoly::to_centered_coeffs(&p, &mut out);
    /// assert_eq!(out, [-8, 0, 3, 7]);
    /// ```
    fn from_centered_i128s(modulus: Self::Modulus, samples: &[i128; N]) -> Self;

    /// Write the centred lift of each coefficient into `dst`. **Not
    /// constant-time** over the input values; use for plaintext-side paths
    /// (decode, gadget decomposition, noise measurement).
    fn to_centered_coeffs(&self, dst: &mut [Self::CenteredScalar; N]);

    /// Constant-time variant of [`Self::to_centered_coeffs`]. Use when the
    /// polynomial coefficients depend on a secret key (e.g. §3.4 rekeying).
    fn to_centered_coeffs_ct(&self, dst: &mut [Self::CenteredScalar; N]);

    /// Read the scalar at index `i` (coefficient of $X^i$).
    fn coeff(&self, i: usize) -> Self::Scalar;

    /// Overwrite the scalar at index `i` (coefficient of $X^i$).
    fn set_coeff(&mut self, i: usize, value: Self::Scalar);

    /// The polynomial's modulus value as a `u128`. For single-prime moduli
    /// $q$ this is `q` zero-extended into the lower 64 bits; for an RNS
    /// basis $B$ this is the composite $Q = q^{(0)} \cdot q^{(1)}$ — see
    /// [`crate::algebra::rns::basis::RnsBasis::big_q`].
    ///
    /// Used by Layer-2 `encode`/`decode` (§2.2) to compute $\Delta =
    /// \lceil q / p \rceil$ and to express the centred lift uniformly across
    /// backends.
    fn modulus_value(modulus: Self::Modulus) -> u128;

    /// Construct a polynomial from per-coefficient `u128` values. Each
    /// input may exceed the modulus; the implementation reduces into
    /// $[0, q)$ (single-prime) or RNS-decomposes into $\bigl([0,
    /// q^{(0)}), [0, q^{(1)})\bigr)$ (RNS).
    fn from_u128_coeffs(modulus: Self::Modulus, values: &[u128; N]) -> Self;

    /// Lift each coefficient to its canonical `u128` representation in
    /// $[0, Q)$. For single-prime polynomials the upper 64 bits are
    /// zero; for RNS polynomials this is the Garner reconstruction.
    fn to_u128_coeffs(&self, dst: &mut [u128; N]);

    /// Lift each coefficient to its centred representation as `i128` in
    /// $(-\lfloor Q/2 \rfloor, \lfloor Q/2 \rfloor]$. For single-prime
    /// polynomials this is the `i64` centred lift up-cast to `i128`.
    ///
    /// **Not constant-time** over input values; used at decoding /
    /// noise-measurement boundaries (§0.6).
    fn to_centered_i128_coeffs(&self, dst: &mut [i128; N]);

    /// Deterministic rotation: return $X^k \cdot \mathrm{self}$ in
    /// $R_{N, q}$ (§4.5). Coefficient at position $i$ moves to
    /// $(i + k) \bmod N$ with a negacyclic sign flip when $i + k \ge N$
    /// ($X^N \equiv -1$). Used by §3.3 RingSwitch / §4.4 CRot.
    ///
    /// `k` is a **public** parameter (a loop induction variable); the
    /// implementation may branch on it. Do **not** pass a secret-derived
    /// `k` — route encrypted-exponent rotation through §4.4 `CRot`.
    ///
    /// ```rust
    /// use via_rs::algebra::ring::abstraction::RingPoly;
    /// use via_rs::algebra::ring::element::Poly;
    /// use via_rs::algebra::ring::form::Coefficient;
    /// use via_rs::algebra::zq::modulus::ConstModulus;
    ///
    /// // In R_{4, 17}: X * (1 + 2X + 3X^2 + 4X^3) = -4 + X + 2X^2 + 3X^3.
    /// // The top coefficient wraps negacyclically: 4*X^4 = -4 = 13 mod 17.
    /// type P = Poly<4, ConstModulus<17>, Coefficient>;
    /// let f = <P as RingPoly<4>>::from_u128_coeffs(ConstModulus, &[1, 2, 3, 4]);
    /// let g = f.mul_x_pow(1);
    /// let mut out = [0u128; 4];
    /// RingPoly::to_u128_coeffs(&g, &mut out);
    /// assert_eq!(out, [13, 1, 2, 3]);
    /// ```
    fn mul_x_pow(&self, k: usize) -> Self;

    /// Single-slot projection $\pi_0^{N \to N_\text{small}}$ extended to an
    /// arbitrary slot — extract slot `slot` of `self` into the smaller ring
    /// $R_{N_\text{small}, q}$. Coefficient at position $d \cdot i + slot$
    /// (with $d = N / N_\text{small}$) becomes coefficient $i$ of the
    /// output. Pure index manipulation (§0.5); no algebra dependency.
    ///
    /// # Panics
    ///
    /// - **Compile-time** (const block in the backend): `N_SMALL > N`,
    ///   `N_SMALL ∤ N`, or `N_SMALL` not a power of two.
    /// - **Runtime**: `slot >= N / N_SMALL`.
    fn project_at<const N_SMALL: usize>(&self, slot: usize) -> Self::Projected<N_SMALL>;
}

// ---------------------------------------------------------------------------
// Impl for the single-prime `Poly<N, M, Coefficient>` backend.
// ---------------------------------------------------------------------------

impl<const N: usize, M: Modulus> sealed::Sealed for Poly<N, M, Coefficient> {}

impl<const N: usize, M: Modulus> RingPoly<N> for Poly<N, M, Coefficient> {
    type Modulus = M;
    type Scalar = Zq<M>;
    type CenteredScalar = i64;
    type Projected<const N_SMALL: usize> = Poly<N_SMALL, M, Coefficient>;

    #[inline(always)]
    fn modulus(&self) -> M {
        Poly::modulus(self)
    }

    #[inline(always)]
    fn zero(modulus: M) -> Self {
        Poly::zero(modulus)
    }

    #[inline(always)]
    fn random_uniform<R: RngCore + ?Sized>(modulus: M, rng: &mut R) -> Self {
        Poly::random(modulus, rng)
    }

    fn from_centered_i64s(modulus: M, samples: &[i64; N]) -> Self {
        let mut buf = [0u64; N];
        for i in 0..N {
            buf[i] = modulus.reduce_i64(samples[i]);
        }
        // SAFETY: every lane is reduced via `Modulus::reduce_i64`, which
        // returns a value in `[0, q)`.
        unsafe { Poly::from_reduced_unchecked(modulus, buf) }
    }

    fn from_centered_i128s(modulus: M, samples: &[i128; N]) -> Self {
        let mut buf = [0u64; N];
        for i in 0..N {
            debug_assert!(
                samples[i] >= i64::MIN as i128 && samples[i] <= i64::MAX as i128,
                "from_centered_i128s: coefficient does not fit in i64 for single-prime modulus",
            );
            buf[i] = modulus.reduce_i64(samples[i] as i64);
        }
        // SAFETY: every lane is reduced via `Modulus::reduce_i64`, which
        // returns a value in `[0, q)`.
        unsafe { Poly::from_reduced_unchecked(modulus, buf) }
    }

    #[inline(always)]
    fn to_centered_coeffs(&self, dst: &mut [i64; N]) {
        Poly::to_centered_coeffs(self, dst);
    }

    #[inline(always)]
    fn to_centered_coeffs_ct(&self, dst: &mut [i64; N]) {
        Poly::to_centered_coeffs_ct(self, dst);
    }

    #[inline(always)]
    fn coeff(&self, i: usize) -> Zq<M> {
        Poly::coeff(self, i)
    }

    #[inline(always)]
    fn set_coeff(&mut self, i: usize, value: Zq<M>) {
        Poly::set_coeff(self, i, value);
    }

    #[inline(always)]
    fn modulus_value(modulus: M) -> u128 {
        u128::from(modulus.q())
    }

    fn from_u128_coeffs(modulus: M, values: &[u128; N]) -> Self {
        let mut buf = [0u64; N];
        for i in 0..N {
            buf[i] = modulus.reduce_u128(values[i]);
        }
        // SAFETY: every lane reduced via `Modulus::reduce_u128`.
        unsafe { Poly::from_reduced_unchecked(modulus, buf) }
    }

    fn to_u128_coeffs(&self, dst: &mut [u128; N]) {
        let values = self.values();
        for i in 0..N {
            dst[i] = u128::from(values[i]);
        }
    }

    fn to_centered_i128_coeffs(&self, dst: &mut [i128; N]) {
        let mut tmp = [0i64; N];
        Poly::to_centered_coeffs(self, &mut tmp);
        for i in 0..N {
            dst[i] = i128::from(tmp[i]);
        }
    }

    #[inline(always)]
    fn mul_x_pow(&self, k: usize) -> Self {
        Poly::mul_x_pow(self, k)
    }

    #[inline(always)]
    fn project_at<const N_SMALL: usize>(&self, slot: usize) -> Poly<N_SMALL, M, Coefficient> {
        Poly::project_at::<N_SMALL>(self, slot)
    }
}

// ---------------------------------------------------------------------------
// Impl for the RNS `PolyRns<N, B, Coefficient>` backend.
// ---------------------------------------------------------------------------

impl<const N: usize, B: RnsBasis> sealed::Sealed for PolyRns<N, B, Coefficient> {}

impl<const N: usize, B: RnsBasis> RingPoly<N> for PolyRns<N, B, Coefficient> {
    type Modulus = B;
    type Scalar = RnsZq<B>;
    type CenteredScalar = i128;
    type Projected<const N_SMALL: usize> = PolyRns<N_SMALL, B, Coefficient>;

    #[inline(always)]
    fn modulus(&self) -> B {
        PolyRns::basis(self)
    }

    #[inline(always)]
    fn zero(basis: B) -> Self {
        PolyRns::zero(basis)
    }

    #[inline(always)]
    fn random_uniform<R: RngCore + ?Sized>(basis: B, rng: &mut R) -> Self {
        PolyRns::random(basis, rng)
    }

    fn from_centered_i64s(basis: B, samples: &[i64; N]) -> Self {
        let m0 = basis.m0();
        let m1 = basis.m1();
        let mut buf0 = [0u64; N];
        let mut buf1 = [0u64; N];
        for i in 0..N {
            buf0[i] = m0.reduce_i64(samples[i]);
            buf1[i] = m1.reduce_i64(samples[i]);
        }
        // SAFETY: each lane reduced via `Modulus::reduce_i64`, hence in
        // `[0, q^(j))` for its respective slot modulus.
        unsafe { PolyRns::from_reduced_unchecked(basis, buf0, buf1) }
    }

    fn from_centered_i128s(basis: B, samples: &[i128; N]) -> Self {
        let m0 = basis.m0();
        let m1 = basis.m1();
        let mut buf0 = [0u64; N];
        let mut buf1 = [0u64; N];
        for i in 0..N {
            // No `Modulus::reduce_i128` exists, so reduce the magnitude
            // unsigned then sign-fixup via `neg` (R2 fallback). `unsigned_abs`
            // maps `i128::MIN` correctly; reduce_u128 accepts the full range.
            let mag = samples[i].unsigned_abs();
            let r0 = m0.reduce_u128(mag);
            let r1 = m1.reduce_u128(mag);
            if samples[i] < 0 {
                buf0[i] = m0.neg(r0);
                buf1[i] = m1.neg(r1);
            } else {
                buf0[i] = r0;
                buf1[i] = r1;
            }
        }
        // SAFETY: each lane is `reduce_u128`/`neg` output, hence in
        // `[0, q^(j))` for its respective slot modulus.
        unsafe { PolyRns::from_reduced_unchecked(basis, buf0, buf1) }
    }

    #[inline(always)]
    fn to_centered_coeffs(&self, dst: &mut [i128; N]) {
        PolyRns::to_centered_coeffs(self, dst);
    }

    #[inline(always)]
    fn to_centered_coeffs_ct(&self, dst: &mut [i128; N]) {
        PolyRns::to_centered_coeffs_ct(self, dst);
    }

    #[inline(always)]
    fn coeff(&self, i: usize) -> RnsZq<B> {
        PolyRns::coeff(self, i)
    }

    #[inline(always)]
    fn set_coeff(&mut self, i: usize, value: RnsZq<B>) {
        PolyRns::set_coeff(self, i, value);
    }

    #[inline(always)]
    fn modulus_value(basis: B) -> u128 {
        basis.big_q()
    }

    #[inline(always)]
    fn from_u128_coeffs(basis: B, values: &[u128; N]) -> Self {
        PolyRns::from_u128_array(basis, values)
    }

    fn to_u128_coeffs(&self, dst: &mut [u128; N]) {
        let basis = self.basis();
        let v0 = self.values0();
        let v1 = self.values1();
        for (out, (&a0, &a1)) in dst.iter_mut().zip(v0.iter().zip(v1.iter())) {
            *out = basis.reconstruct(a0, a1);
        }
    }

    #[inline(always)]
    fn to_centered_i128_coeffs(&self, dst: &mut [i128; N]) {
        PolyRns::to_centered_coeffs(self, dst);
    }

    #[inline(always)]
    fn mul_x_pow(&self, k: usize) -> Self {
        PolyRns::mul_x_pow(self, k)
    }

    #[inline(always)]
    fn project_at<const N_SMALL: usize>(&self, slot: usize) -> PolyRns<N_SMALL, B, Coefficient> {
        PolyRns::project_at::<N_SMALL>(self, slot)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::rns::basis::ConstRnsBasis;
    use crate::algebra::zq::modulus::ConstModulus;
    use crate::sampling::prg::Shake256Prg;

    // Toy parameters: q = 17 (prime, > 2N for N=4 NTT-friendliness isn't
    // required since we never call NTT in these tests); RNS basis with
    // small coprime primes (5, 11) for the RnsZq path.

    type SinglePoly = Poly<4, ConstModulus<17>, Coefficient>;
    type RnsPoly = PolyRns<4, ConstRnsBasis<5, 11>, Coefficient>;

    #[test]
    fn zero_compiles_for_both_backends() {
        let _ = <SinglePoly as RingPoly<4>>::zero(ConstModulus);
        let _ = <RnsPoly as RingPoly<4>>::zero(ConstRnsBasis);
    }

    #[test]
    fn random_uniform_compiles_for_both_backends() {
        let mut prg = Shake256Prg::new(b"ring-poly-abstraction-test");
        let _ = <SinglePoly as RingPoly<4>>::random_uniform(ConstModulus, &mut prg);
        let _ = <RnsPoly as RingPoly<4>>::random_uniform(ConstRnsBasis, &mut prg);
    }

    #[test]
    fn modulus_accessor_roundtrips_single_prime() {
        let m = ConstModulus::<17>;
        let p = <SinglePoly as RingPoly<4>>::zero(m);
        assert_eq!(RingPoly::modulus(&p), m);
    }

    #[test]
    fn modulus_accessor_roundtrips_rns() {
        let b = ConstRnsBasis::<5, 11>;
        let p = <RnsPoly as RingPoly<4>>::zero(b);
        assert_eq!(RingPoly::modulus(&p), b);
    }

    #[test]
    fn from_centered_i64s_then_to_centered_coeffs_single_prime() {
        let m = ConstModulus::<17>;
        let samples = [-8i64, 0, 3, 7];
        let p = <SinglePoly as RingPoly<4>>::from_centered_i64s(m, &samples);
        let mut out = [0i64; 4];
        RingPoly::to_centered_coeffs(&p, &mut out);
        assert_eq!(out, samples);
    }

    #[test]
    fn from_centered_i64s_then_to_centered_coeffs_rns() {
        let b = ConstRnsBasis::<5, 11>;
        // RNS Q = 55, centred range is (-27, 27]. Pick values inside.
        let samples = [-20i64, 0, 1, 13];
        let p = <RnsPoly as RingPoly<4>>::from_centered_i64s(b, &samples);
        let mut out = [0i128; 4];
        RingPoly::to_centered_coeffs(&p, &mut out);
        for (got, want) in out.iter().zip(samples.iter()) {
            assert_eq!(*got, i128::from(*want));
        }
    }

    #[test]
    fn ct_centered_matches_variable_time_single_prime() {
        let m = ConstModulus::<17>;
        let samples = [-8i64, -1, 0, 8];
        let p = <SinglePoly as RingPoly<4>>::from_centered_i64s(m, &samples);
        let mut a = [0i64; 4];
        let mut b = [0i64; 4];
        RingPoly::to_centered_coeffs(&p, &mut a);
        RingPoly::to_centered_coeffs_ct(&p, &mut b);
        assert_eq!(a, b);
    }

    #[test]
    fn coeff_set_coeff_roundtrip_single_prime() {
        let m = ConstModulus::<17>;
        let mut p = <SinglePoly as RingPoly<4>>::zero(m);
        let v = Zq::new(m, 9);
        RingPoly::set_coeff(&mut p, 2, v);
        assert_eq!(RingPoly::coeff(&p, 2), v);
        assert_eq!(RingPoly::coeff(&p, 0), Zq::new(m, 0));
    }

    #[test]
    fn coeff_set_coeff_roundtrip_rns() {
        let b = ConstRnsBasis::<5, 11>;
        let mut p = <RnsPoly as RingPoly<4>>::zero(b);
        let v = RnsZq::from_u128(b, 23);
        RingPoly::set_coeff(&mut p, 1, v);
        assert_eq!(RingPoly::coeff(&p, 1), v);
    }

    #[test]
    fn modulus_value_single_prime() {
        assert_eq!(<SinglePoly as RingPoly<4>>::modulus_value(ConstModulus), 17);
    }

    #[test]
    fn modulus_value_rns() {
        // Q = 5 * 11 = 55
        assert_eq!(<RnsPoly as RingPoly<4>>::modulus_value(ConstRnsBasis), 55);
    }

    #[test]
    fn u128_coeffs_roundtrip_single_prime() {
        let m = ConstModulus::<17>;
        // Values larger than q get reduced by `from_u128_coeffs`.
        let inputs = [0u128, 16, 17, 30, 99];
        let mut padded = [0u128; 4];
        padded.copy_from_slice(&inputs[..4]);
        let p = <SinglePoly as RingPoly<4>>::from_u128_coeffs(m, &padded);
        let mut out = [0u128; 4];
        RingPoly::to_u128_coeffs(&p, &mut out);
        // expected = inputs mod 17
        let expected: [u128; 4] = [0, 16, 0, 13];
        assert_eq!(out, expected);
    }

    #[test]
    fn u128_coeffs_roundtrip_rns() {
        let b = ConstRnsBasis::<5, 11>;
        // Q = 55
        let inputs = [0u128, 23, 55, 100];
        let p = <RnsPoly as RingPoly<4>>::from_u128_coeffs(b, &inputs);
        let mut out = [0u128; 4];
        RingPoly::to_u128_coeffs(&p, &mut out);
        let expected: [u128; 4] = [0, 23, 0, 100 - 55];
        assert_eq!(out, expected);
    }

    #[test]
    fn to_centered_i128_matches_to_centered_i64_for_single_prime() {
        let m = ConstModulus::<17>;
        let samples = [-8i64, -1, 0, 8];
        let p = <SinglePoly as RingPoly<4>>::from_centered_i64s(m, &samples);
        let mut i64_out = [0i64; 4];
        let mut i128_out = [0i128; 4];
        RingPoly::to_centered_coeffs(&p, &mut i64_out);
        RingPoly::to_centered_i128_coeffs(&p, &mut i128_out);
        for i in 0..4 {
            assert_eq!(i128::from(i64_out[i]), i128_out[i]);
        }
    }

    #[test]
    fn to_centered_i128_matches_to_centered_for_rns() {
        let b = ConstRnsBasis::<5, 11>;
        let samples = [-20i64, 0, 1, 13];
        let p = <RnsPoly as RingPoly<4>>::from_centered_i64s(b, &samples);
        let mut native_out = [0i128; 4];
        let mut i128_out = [0i128; 4];
        RingPoly::to_centered_coeffs(&p, &mut native_out);
        RingPoly::to_centered_i128_coeffs(&p, &mut i128_out);
        assert_eq!(native_out, i128_out);
    }

    #[test]
    fn from_centered_i128s_single_prime_roundtrip() {
        let m = ConstModulus::<17>;
        let samples = [-8i128, 0, 3, 7];
        let p = <SinglePoly as RingPoly<4>>::from_centered_i128s(m, &samples);
        let mut out = [0i64; 4];
        RingPoly::to_centered_coeffs(&p, &mut out);
        assert_eq!(out, [-8, 0, 3, 7]);
    }

    #[test]
    fn from_centered_i128s_matches_from_centered_i64s_single_prime() {
        let m = ConstModulus::<17>;
        let i64s = [-8i64, -1, 0, 8];
        let i128s = [-8i128, -1, 0, 8];
        let a = <SinglePoly as RingPoly<4>>::from_centered_i64s(m, &i64s);
        let b = <SinglePoly as RingPoly<4>>::from_centered_i128s(m, &i128s);
        assert_eq!(a, b);
    }

    #[test]
    fn from_centered_i128s_rns_roundtrip() {
        let b = ConstRnsBasis::<5, 11>;
        // Q = 55, centred range (-27, 27].
        let samples = [-20i128, 0, 1, 13];
        let p = <RnsPoly as RingPoly<4>>::from_centered_i128s(b, &samples);
        let mut out = [0i128; 4];
        RingPoly::to_centered_coeffs(&p, &mut out);
        assert_eq!(out, samples);
    }

    #[test]
    fn from_centered_i128s_matches_from_centered_i64s_rns() {
        let b = ConstRnsBasis::<5, 11>;
        let i64s = [-20i64, 0, 1, 13];
        let i128s = [-20i128, 0, 1, 13];
        let a = <RnsPoly as RingPoly<4>>::from_centered_i64s(b, &i64s);
        let c = <RnsPoly as RingPoly<4>>::from_centered_i128s(b, &i128s);
        assert_eq!(a, c);
    }

    #[test]
    fn mul_x_pow_single_prime_identity() {
        let m = ConstModulus::<17>;
        let p = <SinglePoly as RingPoly<4>>::from_u128_coeffs(m, &[1, 2, 3, 4]);
        let q = RingPoly::mul_x_pow(&p, 0);
        assert_eq!(p, q);
    }

    #[test]
    fn mul_x_pow_single_prime_negacyclic_wrap() {
        let m = ConstModulus::<17>;
        // X * (1 + 2X + 3X^2 + 4X^3) = -4 + X + 2X^2 + 3X^3; -4 = 13 mod 17.
        let p = <SinglePoly as RingPoly<4>>::from_u128_coeffs(m, &[1, 2, 3, 4]);
        let q = RingPoly::mul_x_pow(&p, 1);
        let mut out = [0u128; 4];
        RingPoly::to_u128_coeffs(&q, &mut out);
        assert_eq!(out, [13, 1, 2, 3]);
    }

    #[test]
    fn mul_x_pow_rns() {
        let b = ConstRnsBasis::<5, 11>;
        // Q = 55. X * (1 + 2X + 3X^2 + 4X^3) = -4 + X + 2X^2 + 3X^3;
        // -4 = 51 mod 55.
        let p = <RnsPoly as RingPoly<4>>::from_u128_coeffs(b, &[1, 2, 3, 4]);
        let q = RingPoly::mul_x_pow(&p, 1);
        let mut out = [0u128; 4];
        RingPoly::to_u128_coeffs(&q, &mut out);
        assert_eq!(out, [51, 1, 2, 3]);
    }

    #[test]
    fn project_at_single_prime_slot0() {
        let m = ConstModulus::<17>;
        // d = 4 / 2 = 2; slot 0 picks coefficients at positions 0, 2.
        let p = <SinglePoly as RingPoly<4>>::from_u128_coeffs(m, &[1, 2, 3, 4]);
        let proj: Poly<2, ConstModulus<17>, Coefficient> = RingPoly::project_at::<2>(&p, 0);
        let mut out = [0u128; 2];
        RingPoly::to_u128_coeffs(&proj, &mut out);
        assert_eq!(out, [1, 3]);
    }

    #[test]
    fn project_at_single_prime_slot1() {
        let m = ConstModulus::<17>;
        // slot 1 picks coefficients at positions 1, 3.
        let p = <SinglePoly as RingPoly<4>>::from_u128_coeffs(m, &[1, 2, 3, 4]);
        let proj: Poly<2, ConstModulus<17>, Coefficient> = RingPoly::project_at::<2>(&p, 1);
        let mut out = [0u128; 2];
        RingPoly::to_u128_coeffs(&proj, &mut out);
        assert_eq!(out, [2, 4]);
    }

    #[test]
    fn project_at_rns_slot0() {
        let b = ConstRnsBasis::<5, 11>;
        let p = <RnsPoly as RingPoly<4>>::from_u128_coeffs(b, &[1, 2, 3, 4]);
        let proj: PolyRns<2, ConstRnsBasis<5, 11>, Coefficient> = RingPoly::project_at::<2>(&p, 0);
        let mut out = [0u128; 2];
        RingPoly::to_u128_coeffs(&proj, &mut out);
        assert_eq!(out, [1, 3]);
    }
}
