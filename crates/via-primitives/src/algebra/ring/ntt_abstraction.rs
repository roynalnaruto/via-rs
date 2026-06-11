//! Evaluation-form multiply interface — `.docs/primitives.md` §0.4.
//!
//! [`RingPolyEval`] is the form-neutral interface a caller needs to run a
//! sequence of ring multiplications through an evaluation form: convert the
//! operands once ([`to_eval`](RingPolyEval::to_eval)), multiply them pointwise
//! and accumulate, then convert the result back once
//! ([`from_eval`](RingPolyEval::from_eval)). It is the sibling trait the
//! [`super::abstraction`] module docs defer: the base [`RingPoly`] stays
//! coefficient-only (its centred-coefficient methods are meaningless in
//! evaluation form), and the eval-form capability lives here so it can be
//! bounded independently exactly where it is wanted (e.g.
//! [`crate::encryption::RLevCiphertext::gadget_product`]).
//!
//! ## Two backings, one interface
//!
//! - **NTT-friendly moduli** (single-prime `M: NttFriendly<N>`, RNS with both
//!   slot moduli friendly): `Eval` is the negacyclic-NTT image. By the NTT
//!   bijection, pointwise `Mul` on it **is** ring multiplication and `AddAssign`
//!   is ring addition, so a length-`L` multiply-accumulate costs `O(N log N)`
//!   transforms + `O(N)` pointwise muls instead of `O(N²)` schoolbook muls. This
//!   is the paper coefficient moduli ($q_1$ RNS, $q_2$, $q_3$).
//! - **Non-NTT moduli** ([`DynModulus`], [`PowerOfTwoModulus`] — the paper
//!   $q_4$/$p$, runtime-parsed params, and toy test params): there is no NTT, so
//!   `Eval = Self` (the coefficient form), `to_eval`/`from_eval` are the
//!   **identity**, and "pointwise `Mul`" is just the existing **schoolbook**
//!   negacyclic `Mul` on the coefficient form. The interface degenerates exactly
//!   to the coefficient-form computation — same result, same cost, no regression.
//!
//! Because every polynomial backend implements `RingPolyEval` (real NTT or
//! identity fallback), callers can bound on it uniformly with a single code
//! path: NTT-friendly instantiations get the speed-up, others transparently run
//! schoolbook. The two backings never conflict — `DynModulus`/`PowerOfTwoModulus`
//! do not implement `NttFriendly`, so the coherence checker proves the
//! `M: NttFriendly` impl and the concrete fallback impls are disjoint.

use core::ops::{AddAssign, Mul};

use super::RingPoly;
use super::element::Poly;
use super::form::{Coefficient, Evaluation};
use super::ntt::NttFriendly;
use super::rns_element::PolyRns;
use crate::algebra::rns::basis::RnsBasis;
use crate::algebra::zq::modulus::{DynModulus, PowerOfTwoModulus};

/// A [`RingPoly`] that exposes an evaluation-form representation supporting
/// pointwise ring multiplication, plus the transforms to and from it.
///
/// The associated [`Eval`](RingPolyEval::Eval) type is the eval-form image;
/// pointwise `Mul` on it is ring multiplication and `AddAssign` is ring addition
/// (both `O(N)` for the NTT backing). See the module docs for the two backings
/// (real NTT vs schoolbook fallback).
pub trait RingPolyEval<const N: usize>: RingPoly<N> {
    /// The evaluation-form image type. Pointwise `Mul` = ring mul; `AddAssign`
    /// = ring add.
    type Eval: Copy + AddAssign + Mul<Output = Self::Eval>;

    /// Coefficient form → evaluation form (forward NTT for NTT-friendly moduli;
    /// the identity otherwise). Consumes `self` (callers `Copy` first, since
    /// [`RingPoly`] requires `Copy`).
    fn to_eval(self) -> Self::Eval;

    /// Evaluation form → coefficient form (inverse NTT for NTT-friendly moduli;
    /// the identity otherwise).
    fn from_eval(eval: Self::Eval) -> Self;

    /// The all-zeros polynomial in evaluation form — the additive identity for
    /// the pointwise ring, used to seed an accumulator.
    fn eval_zero(modulus: Self::Modulus) -> Self::Eval;
}

// ---------------------------------------------------------------------------
// Real NTT backing — single-prime and RNS, gated on NTT-friendliness.
// ---------------------------------------------------------------------------

impl<const N: usize, M: NttFriendly<N>> RingPolyEval<N> for Poly<N, M, Coefficient> {
    type Eval = Poly<N, M, Evaluation>;

    #[inline]
    fn to_eval(self) -> Self::Eval {
        // Inherent `Poly::into_eval` (the forward negacyclic NTT).
        self.into_eval()
    }

    #[inline]
    fn from_eval(eval: Self::Eval) -> Self {
        eval.into_coeff()
    }

    #[inline]
    fn eval_zero(modulus: M) -> Self::Eval {
        Poly::zero(modulus)
    }
}

impl<const N: usize, B: RnsBasis> RingPolyEval<N> for PolyRns<N, B, Coefficient>
where
    B::M0: NttFriendly<N>,
    B::M1: NttFriendly<N>,
{
    type Eval = PolyRns<N, B, Evaluation>;

    #[inline]
    fn to_eval(self) -> Self::Eval {
        self.into_eval()
    }

    #[inline]
    fn from_eval(eval: Self::Eval) -> Self {
        eval.into_coeff()
    }

    #[inline]
    fn eval_zero(basis: B) -> Self::Eval {
        PolyRns::zero(basis)
    }
}

// ---------------------------------------------------------------------------
// Schoolbook fallback — non-NTT moduli. `Eval = Self` (coefficient form),
// transforms are the identity, and the pointwise `Mul` required by the trait
// is the existing schoolbook negacyclic `Mul` on the coefficient form. So a
// caller's eval-form multiply-accumulate degenerates to the exact schoolbook
// computation: identical result, identical cost.
//
// These do not overlap the `M: NttFriendly` impl above because neither
// `DynModulus` nor `PowerOfTwoModulus<_>` implements `NttFriendly` (both are
// local types and `NttFriendly` is a local trait, so coherence sees the full —
// empty — set of such impls and proves disjointness).
// ---------------------------------------------------------------------------

impl<const N: usize> RingPolyEval<N> for Poly<N, DynModulus, Coefficient> {
    type Eval = Poly<N, DynModulus, Coefficient>;

    #[inline]
    fn to_eval(self) -> Self::Eval {
        self
    }

    #[inline]
    fn from_eval(eval: Self::Eval) -> Self {
        eval
    }

    #[inline]
    fn eval_zero(modulus: DynModulus) -> Self::Eval {
        Poly::zero(modulus)
    }
}

impl<const N: usize, const LOG2_Q: u32> RingPolyEval<N>
    for Poly<N, PowerOfTwoModulus<LOG2_Q>, Coefficient>
{
    type Eval = Poly<N, PowerOfTwoModulus<LOG2_Q>, Coefficient>;

    #[inline]
    fn to_eval(self) -> Self::Eval {
        self
    }

    #[inline]
    fn from_eval(eval: Self::Eval) -> Self {
        eval
    }

    #[inline]
    fn eval_zero(modulus: PowerOfTwoModulus<LOG2_Q>) -> Self::Eval {
        Poly::zero(modulus)
    }
}
