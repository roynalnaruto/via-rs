//! NTT-capability extension of [`RingPoly`] — `.docs/primitives.md` §0.4.
//!
//! [`RingPolyNtt`] is the form-neutral interface a caller needs to run a
//! sequence of ring multiplications through the negacyclic NTT: transform the
//! operands once ([`forward_ntt`](RingPolyNtt::forward_ntt)), multiply them
//! pointwise (Hadamard) and accumulate in evaluation form, then transform the
//! result back once ([`inverse_ntt`](RingPolyNtt::inverse_ntt)). By the
//! negacyclic-NTT bijection this is **exactly** ring multiplication, so it is
//! bit-identical to the schoolbook path while turning each `O(N²)` negacyclic
//! mul into an `O(N log N)` transform plus `O(N)` pointwise muls.
//!
//! It is the sibling trait the [`super::abstraction`] module docs defer: the
//! base [`RingPoly`] stays coefficient-only (its centred-coefficient methods are
//! meaningless in evaluation form), and the eval-form capability lives here so
//! it can be bounded *independently*, exactly where the NTT win is wanted (e.g.
//! [`crate::encryption::RLevCiphertext::gadget_product_ntt`]).
//!
//! Implemented **only** for the NTT-friendly instantiations — single-prime
//! [`Poly<N, M, Coefficient>`] with `M: NttFriendly<N>`, and RNS
//! [`PolyRns<N, B, Coefficient>`] with both slot moduli `NttFriendly<N>`. The
//! power-of-two paper moduli ($q_4$, $p$) deliberately do **not** implement it
//! and keep the schoolbook path (they satisfy `q ≡ 0 mod 2N`, not `q ≡ 1`).

use core::ops::{AddAssign, Mul};

use super::RingPoly;
use super::element::Poly;
use super::form::{Coefficient, Evaluation};
use super::ntt::NttFriendly;
use super::rns_element::PolyRns;
use crate::algebra::rns::basis::RnsBasis;

/// A [`RingPoly`] whose modulus admits a negacyclic NTT, exposing the
/// evaluation-form transforms and the additive identity in that form.
///
/// The associated [`Eval`](RingPolyNtt::Eval) type is the NTT image; pointwise
/// `Mul` on it is ring multiplication and `AddAssign` is ring addition (both
/// `O(N)`). See the module docs for the amortisation contract.
pub trait RingPolyNtt<const N: usize>: RingPoly<N> {
    /// The evaluation-form image type. Pointwise `Mul` = ring mul; `AddAssign`
    /// = ring add.
    type Eval: Copy + AddAssign + Mul<Output = Self::Eval>;

    /// Forward negacyclic NTT: coefficient form → evaluation form. Consumes
    /// `self` (callers `Copy` first, since [`RingPoly`] requires `Copy`).
    fn forward_ntt(self) -> Self::Eval;

    /// Inverse negacyclic NTT: evaluation form → coefficient form.
    fn inverse_ntt(eval: Self::Eval) -> Self;

    /// The all-zeros polynomial in evaluation form — the additive identity for
    /// the pointwise ring, used to seed an accumulator.
    fn eval_zero(modulus: Self::Modulus) -> Self::Eval;
}

impl<const N: usize, M: NttFriendly<N>> RingPolyNtt<N> for Poly<N, M, Coefficient> {
    type Eval = Poly<N, M, Evaluation>;

    #[inline]
    fn forward_ntt(self) -> Self::Eval {
        // Inherent `Poly::into_eval`; the trait method is named differently so
        // this does not recurse.
        self.into_eval()
    }

    #[inline]
    fn inverse_ntt(eval: Self::Eval) -> Self {
        eval.into_coeff()
    }

    #[inline]
    fn eval_zero(modulus: M) -> Self::Eval {
        Poly::zero(modulus)
    }
}

impl<const N: usize, B: RnsBasis> RingPolyNtt<N> for PolyRns<N, B, Coefficient>
where
    B::M0: NttFriendly<N>,
    B::M1: NttFriendly<N>,
{
    type Eval = PolyRns<N, B, Evaluation>;

    #[inline]
    fn forward_ntt(self) -> Self::Eval {
        self.into_eval()
    }

    #[inline]
    fn inverse_ntt(eval: Self::Eval) -> Self {
        eval.into_coeff()
    }

    #[inline]
    fn eval_zero(basis: B) -> Self::Eval {
        PolyRns::zero(basis)
    }
}
