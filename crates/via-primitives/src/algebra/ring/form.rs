//! Typestate markers for the polynomial representation form.
//!
//! [`Form`] is a **sealed** trait — only this module can implement it. The
//! two markers [`Coefficient`] and [`Evaluation`] are zero-sized types that
//! parameterise [`super::element::Poly`] and [`super::rns_element::PolyRns`]
//! at the type level. Distinct form markers produce distinct types, so the
//! compiler refuses to mix them:
//!
//! ```compile_fail
//! use via_rs::algebra::ring::element::Poly;
//! use via_rs::algebra::ring::form::{Coefficient, Evaluation};
//! use via_rs::algebra::zq::modulus::ConstModulus;
//! type M = ConstModulus<17>;
//! let c: Poly<4, M, Coefficient> = Poly::zero(M);
//! let e: Poly<4, M, Evaluation> = Poly::zero(M);
//! let _ = c + e; // form mismatch — does not compile
//! ```
//!
//! Sealing prevents downstream crates from inventing a third form (e.g. a
//! Karatsuba-decomposed shape) and accidentally producing a `Poly` whose
//! arithmetic semantics are undefined for VIA.
//!
//! The marker is stored inside the polynomial type as `PhantomData<F>`, so
//! it costs no bytes at runtime; the typestate is purely a compile-time
//! discipline.

use core::fmt::Debug;

mod sealed {
    pub trait Sealed {}
}

/// Marker trait for polynomial representation form. Sealed — only this
/// module's [`Coefficient`] and [`Evaluation`] implement it.
pub trait Form: sealed::Sealed + Copy + Clone + Eq + PartialEq + 'static + Debug + Default {
    /// A single-byte discriminant suitable for inclusion in [`core::hash::Hash`]
    /// implementations so that two polynomials with the same buffer but
    /// different forms hash distinctly.
    const HASH_TAG: u8;
}

/// Coefficient-form marker: `values[i]` is the coefficient of $X^i$ in
/// $\sum_i v_i X^i$.
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Coefficient;

/// Evaluation-form marker: `values[i]` is the evaluation of the polynomial
/// at the $i$-th primitive $2N$-th root of unity (negacyclic NTT layout).
#[derive(Copy, Clone, Eq, PartialEq, Debug, Default)]
pub struct Evaluation;

impl sealed::Sealed for Coefficient {}
impl sealed::Sealed for Evaluation {}

impl Form for Coefficient {
    const HASH_TAG: u8 = b'C';
}

impl Form for Evaluation {
    const HASH_TAG: u8 = b'E';
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markers_are_zero_sized() {
        assert_eq!(core::mem::size_of::<Coefficient>(), 0);
        assert_eq!(core::mem::size_of::<Evaluation>(), 0);
    }

    #[test]
    fn hash_tags_differ() {
        assert_ne!(Coefficient::HASH_TAG, Evaluation::HASH_TAG);
    }

    #[test]
    fn markers_are_default_constructible() {
        let _c = Coefficient;
        let _e = Evaluation;
        let _c2: Coefficient = Default::default();
        let _e2: Evaluation = Default::default();
    }
}
