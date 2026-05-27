//! Ergonomic type aliases for the paper parameter sets.
//!
//! Layer-2 ciphertext types are generic over `R: RingPoly<N>`. At a call
//! site like `RLWECiphertext::<2048, Poly<2048, ViaCQ2, Coefficient>>` the
//! polynomial type is verbose and the `Coefficient` form-marker noise
//! obscures intent. The aliases below collapse each paper modulus into a
//! single short name so users can write
//! `ViaCRLWEQ2::<2048>` instead.
//!
//! Naming convention: `<SchemePrefix><CiphertextKind><ModulusTag>` where:
//!
//! - `SchemePrefix` ∈ `{Via, ViaC}`. VIA-B reuses VIA-C aliases (same
//!   moduli).
//! - `CiphertextKind` ∈ `{Sk, Rlwe, ..., Mlwe}`.
//! - `ModulusTag` is the modulus subscript from `.docs/primitives.md`
//!   Appendix A: `Q1Rns` (RNS at $q_1$), `Q2`, `Q3`, `Q4`, or `P` (the
//!   plaintext modulus).
//!
//! Only the most common pairings ship as named aliases here; bespoke
//! parameter combinations use the raw generic types directly.

#![allow(missing_docs)] // alias names are self-describing.

use crate::algebra::ring::element::Poly;
use crate::algebra::ring::form::Coefficient;
use crate::algebra::ring::rns_element::PolyRns;
use crate::algebra::rns::basis::paper::{ViaCQ1Rns, ViaQ1Rns};
use crate::algebra::zq::modulus::paper::{
    ViaCP, ViaCQ2, ViaCQ3, ViaCQ4, ViaP, ViaQ2, ViaQ3, ViaQ4,
};

use crate::encryption::mlwe::MLWECiphertext;
use crate::encryption::types::{
    ModSwitchedCiphertext, RGSWCiphertext, RLWECiphertext, RLevCiphertext, SecretKey,
};

// ---------------------------------------------------------------------------
// VIA (toy / no-LWE-to-RLWE) parameters
// ---------------------------------------------------------------------------

// q1 ≈ 2^57 (two-prime RNS).
pub type ViaPolyQ1Rns<const N: usize> = PolyRns<N, ViaQ1Rns, Coefficient>;
// q2 ≈ 2^35, q3 ≈ 2^31, q4 = 2^15, p = 256.
pub type ViaPolyQ2<const N: usize> = Poly<N, ViaQ2, Coefficient>;
pub type ViaPolyQ3<const N: usize> = Poly<N, ViaQ3, Coefficient>;
pub type ViaPolyQ4<const N: usize> = Poly<N, ViaQ4, Coefficient>;
pub type ViaPolyP<const N: usize> = Poly<N, ViaP, Coefficient>;

pub type ViaSkQ1Rns<const N: usize> = SecretKey<N, ViaPolyQ1Rns<N>>;
pub type ViaSkQ2<const N: usize> = SecretKey<N, ViaPolyQ2<N>>;

pub type ViaRlweQ1Rns<const N: usize> = RLWECiphertext<N, ViaPolyQ1Rns<N>>;
pub type ViaRlweQ2<const N: usize> = RLWECiphertext<N, ViaPolyQ2<N>>;
pub type ViaRlweQ3<const N: usize> = RLWECiphertext<N, ViaPolyQ3<N>>;

pub type ViaRlevQ1Rns<const N: usize, const L: usize> = RLevCiphertext<N, ViaPolyQ1Rns<N>, L>;
pub type ViaRlevQ2<const N: usize, const L: usize> = RLevCiphertext<N, ViaPolyQ2<N>, L>;

pub type ViaRgswQ1Rns<const N: usize, const L1: usize, const L2: usize> =
    RGSWCiphertext<N, ViaPolyQ1Rns<N>, L1, L2>;
pub type ViaRgswQ2<const N: usize, const L1: usize, const L2: usize> =
    RGSWCiphertext<N, ViaPolyQ2<N>, L1, L2>;

/// VIA's final answer ciphertext: mask at $q_3$, body at $q_4$.
pub type ViaModSwitchedQ3Q4<const N: usize> = ModSwitchedCiphertext<N, ViaPolyQ3<N>, ViaPolyQ4<N>>;

pub type ViaMlweQ1Rns<const RANK: usize, const N: usize> = MLWECiphertext<RANK, N, ViaPolyQ1Rns<N>>;
pub type ViaMlweQ2<const RANK: usize, const N: usize> = MLWECiphertext<RANK, N, ViaPolyQ2<N>>;

// ---------------------------------------------------------------------------
// VIA-C / VIA-B (realistic parameters)
// ---------------------------------------------------------------------------

// q1 ≈ 2^75 (two-prime RNS).
pub type ViaCPolyQ1Rns<const N: usize> = PolyRns<N, ViaCQ1Rns, Coefficient>;
// q2 ≈ 2^34, q3 ≈ 2^23, q4 = 2^12, p = 16.
pub type ViaCPolyQ2<const N: usize> = Poly<N, ViaCQ2, Coefficient>;
pub type ViaCPolyQ3<const N: usize> = Poly<N, ViaCQ3, Coefficient>;
pub type ViaCPolyQ4<const N: usize> = Poly<N, ViaCQ4, Coefficient>;
pub type ViaCPolyP<const N: usize> = Poly<N, ViaCP, Coefficient>;

pub type ViaCSkQ1Rns<const N: usize> = SecretKey<N, ViaCPolyQ1Rns<N>>;
pub type ViaCSkQ2<const N: usize> = SecretKey<N, ViaCPolyQ2<N>>;

pub type ViaCRlweQ1Rns<const N: usize> = RLWECiphertext<N, ViaCPolyQ1Rns<N>>;
pub type ViaCRlweQ2<const N: usize> = RLWECiphertext<N, ViaCPolyQ2<N>>;
pub type ViaCRlweQ3<const N: usize> = RLWECiphertext<N, ViaCPolyQ3<N>>;

pub type ViaCRlevQ1Rns<const N: usize, const L: usize> = RLevCiphertext<N, ViaCPolyQ1Rns<N>, L>;
pub type ViaCRlevQ2<const N: usize, const L: usize> = RLevCiphertext<N, ViaCPolyQ2<N>, L>;

pub type ViaCRgswQ1Rns<const N: usize, const L1: usize, const L2: usize> =
    RGSWCiphertext<N, ViaCPolyQ1Rns<N>, L1, L2>;
pub type ViaCRgswQ2<const N: usize, const L1: usize, const L2: usize> =
    RGSWCiphertext<N, ViaCPolyQ2<N>, L1, L2>;

/// VIA-C's `RespComp` output: mask at $q_3$, body at $q_4$ (paper Figure 7).
pub type ViaCModSwitchedQ3Q4<const N: usize> =
    ModSwitchedCiphertext<N, ViaCPolyQ3<N>, ViaCPolyQ4<N>>;

pub type ViaCMlweQ1Rns<const RANK: usize, const N: usize> =
    MLWECiphertext<RANK, N, ViaCPolyQ1Rns<N>>;
pub type ViaCMlweQ2<const RANK: usize, const N: usize> = MLWECiphertext<RANK, N, ViaCPolyQ2<N>>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::RingPoly;

    // Smoke: every alias must be constructible at the paper ring degree
    // (n_2 = 512 for both VIA and VIA-C, n_1 = 2048).
    //
    // We use `n2 = 512` to avoid blowing test-binary stack on the
    // `[u64; 2048]` allocations; correctness at larger `N` is exercised
    // by the underlying Poly / PolyRns tests.

    #[test]
    fn via_c_rlwe_q2_at_n512_constructs() {
        let m = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        let z = <ViaCPolyQ2<512> as RingPoly<512>>::zero(m);
        let _ct: ViaCRlweQ2<512> = RLWECiphertext::new(z, z);
    }

    #[test]
    fn via_c_rlev_q1rns_at_n512_l2_constructs() {
        // Use n_2 = 512 to keep the test-binary stack footprint small;
        // the underlying PolyRns tests cover n_1 = 2048.
        let b = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let z = <ViaCPolyQ1Rns<512> as RingPoly<512>>::zero(b);
        let rlwe = RLWECiphertext::<512, ViaCPolyQ1Rns<512>>::new(z, z);
        // L = 2 matches the paper VIA-C DMux ctrl gadget depth.
        let _rlev: ViaCRlevQ1Rns<512, 2> = RLevCiphertext::new([rlwe; 2]);
    }
}
