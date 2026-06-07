//! §5.5 — `extr`: general-$d$ RLWE→MLWE coefficient extraction. See
//! `.docs/primitives.md` §5.5.
//!
//! Given $\mathrm{RLWE}_S(M) = (A, B)$ in $R_{n, q}$ and a divisor $D \mid n$,
//! [`extr`] outputs an $(n/D, D)$-MLWE encrypting $\pi_0^{n \to D}(M)$ under the
//! projected key vector $\pi^{n \to D}(S) = (\pi_0(S), \ldots, \pi_{n/D-1}(S))$:
//!
//! $$
//! \mathrm{Extr}_D(A, B) = \bigl(
//!   \pi_0(A),\;
//!   \pi_{n/D-1}(A) \cdot X,\;
//!   \pi_{n/D-2}(A) \cdot X,\;
//!   \ldots,\;
//!   \pi_1(A) \cdot X,\;
//!   \pi_0(B)
//! \bigr),
//! $$
//!
//! all projections being $\pi_\bullet^{n \to D}$. It is pure index manipulation
//! plus $n/D - 1$ multiplications by $X$ in $R_{D, q}$ — **no key material and
//! no noise growth** beyond the host RLWE's.
//!
//! The $D = 1$ case is the classical RLWE→LWE sample extraction (`via.pdf`
//! §2.2): $X \equiv -1$ in $R_{1, q}$, so the $\cdot X$ becomes a negation and
//! the decryption telescopes to $M_0 + e_0$. General $D$ is the first half of
//! VIA-B's `Repack_k` (§7.4); it has **no Python reference** and is validated by
//! the Rust round-trip + property tests below.

use crate::algebra::ring::RingPoly;
use crate::encryption::MLWECiphertext;
use crate::encryption::types::RLWECiphertext;

/// Compile-time relationship for [`extr`]: $D \mid N$ and
/// $\mathrm{RANK\_OUT} = N / D$. Forced at the top of [`extr`].
pub struct ExtrDims<const N: usize, const D: usize, const RANK_OUT: usize>;

impl<const N: usize, const D: usize, const RANK_OUT: usize> ExtrDims<N, D, RANK_OUT> {
    /// Asserts $D \ge 1$, $D \mid N$, and $\mathrm{RANK\_OUT} = N / D$.
    pub const _CHECK: () = {
        assert!(D >= 1, "extr: D must be at least 1");
        assert!(N.is_multiple_of(D), "extr: D must divide N");
        assert!(RANK_OUT == N / D, "extr: RANK_OUT must equal N / D");
    };
}

/// §5.5 — RLWE→MLWE coefficient extraction (general $D$).
///
/// `D` is the **output component degree**; the result is an
/// $(N/D, D)$-MLWE encrypting $\pi_0^{N \to D}(M)$ under $\pi^{N \to D}(S)$. No
/// key material is consumed. `D = 1` is classical sample extraction
/// (`via.pdf` §2.2); general `D` is the first half of VIA-B `Repack_k` (§7.4).
///
/// # Constant-time: No
///
/// Operates on RLWE-uniform ciphertext coefficients (§0.6).
///
/// # Panics
///
/// At compile time if [`ExtrDims::_CHECK`] fails ($D \nmid N$ or
/// $\mathrm{RANK\_OUT} \ne N/D$).
#[allow(non_camel_case_types)]
pub fn extr<const N: usize, const D: usize, const RANK_OUT: usize, R: RingPoly<N>>(
    ct: &RLWECiphertext<N, R>,
) -> MLWECiphertext<RANK_OUT, D, R::Projected<D>> {
    let () = ExtrDims::<N, D, RANK_OUT>::_CHECK;
    let masks = core::array::from_fn(|k| {
        if k == 0 {
            ct.mask.project_at::<D>(0)
        } else {
            // π_{n/D − k}(A) · X  (the · X realises the negacyclic shift; for
            // D = 1 it is a negation since X ≡ −1).
            ct.mask.project_at::<D>(RANK_OUT - k).mul_x_pow(1)
        }
    });
    MLWECiphertext::new(masks, ct.body.project_at::<D>(0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::rns::basis::ConstRnsBasis;
    use crate::algebra::zq::modulus::ConstModulus;
    use crate::encryption::encode;
    use crate::encryption::types::SecretKey;
    use crate::sampling::distribution::Distribution;
    use crate::sampling::prg::Shake256Prg;

    /// Decrypt an $(m, n)$-MLWE under an explicit key vector:
    /// $\mathrm{decode}\bigl(B - \sum_k A_k \cdot S_k\bigr)$.
    fn mlwe_decrypt<const RANK: usize, const N: usize, R, RP>(
        ct: &MLWECiphertext<RANK, N, R>,
        keys: &[R; RANK],
        p_mod: RP::Modulus,
    ) -> RP
    where
        R: RingPoly<N>,
        RP: RingPoly<N>,
    {
        let mut acc = ct.body;
        for (mask, key) in ct.masks.iter().zip(keys.iter()) {
            acc -= *mask * *key;
        }
        crate::encryption::decode::<N, R, RP>(&acc, p_mod)
    }

    /// `D = 1` — classical sample extraction: recover $M_0$ as an $(n, 1)$-MLWE
    /// under the per-coefficient key vector $(S_0, \ldots, S_{n-1})$.
    #[test]
    fn extr_d1_sample_extraction_roundtrip() {
        type R8 = Poly<8, ConstModulus<65537>, Coefficient>;
        type P8 = Poly<8, ConstModulus<16>, Coefficient>;
        type P1 = Poly<1, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        let mut prg = Shake256Prg::new(b"extr-d1");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let m_coeffs = [3u64, 1, 4, 1, 5, 9, 2, 6];
        let pt: P8 = Poly::new(p, m_coeffs);
        let encoded = encode::<8, R8, P8>(&pt, q);
        let rlwe = sk.encrypt(&encoded, Distribution::Ternary, &mut prg);

        let lwe = extr::<8, 1, 8, _>(&rlwe);
        assert_eq!(lwe.masks.len(), 8);
        // Key vector: π_j^{8→1}(S) = S_j, j = 0..7.
        let key_vec: [Poly<1, ConstModulus<65537>, Coefficient>; 8] =
            core::array::from_fn(|j| sk.poly().project_at::<1>(j));
        let recovered: P1 = mlwe_decrypt(&lwe, &key_vec, p);
        assert_eq!(recovered.coeff(0).to_u64(), m_coeffs[0]); // π_0^{8→1}(M) = M_0
    }

    /// General `D = 2` — recover $\pi_0^{8 \to 2}(M) = (M_0, M_4)$ as a
    /// $(4, 2)$-MLWE under $(\pi_0(S), \ldots, \pi_3(S))$.
    #[test]
    fn extr_general_d2_roundtrip() {
        type R8 = Poly<8, ConstModulus<65537>, Coefficient>;
        type P8 = Poly<8, ConstModulus<16>, Coefficient>;
        type P2 = Poly<2, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        let mut prg = Shake256Prg::new(b"extr-d2");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let m_coeffs = [3u64, 1, 4, 1, 5, 9, 2, 6];
        let pt: P8 = Poly::new(p, m_coeffs);
        let encoded = encode::<8, R8, P8>(&pt, q);
        let rlwe = sk.encrypt(&encoded, Distribution::Ternary, &mut prg);

        let mlwe = extr::<8, 2, 4, _>(&rlwe);
        assert_eq!(mlwe.masks.len(), 4);
        // Key vector: π_j^{8→2}(S) for j = 0..3.
        let key_vec: [Poly<2, ConstModulus<65537>, Coefficient>; 4] =
            core::array::from_fn(|j| sk.poly().project_at::<2>(j));
        let recovered: P2 = mlwe_decrypt(&mlwe, &key_vec, p);
        // π_0^{8→2}(M): stride 8/2 = 4 ⇒ (M_0, M_4).
        assert_eq!(recovered.coeff(0).to_u64(), m_coeffs[0]);
        assert_eq!(recovered.coeff(1).to_u64(), m_coeffs[4]);
    }

    /// Paper-class: `extr` (`D = 2`) on the RNS backend.
    #[test]
    fn extr_d2_rns_roundtrip() {
        type Rns8 = PolyRns<8, ConstRnsBasis<7681, 12289>, Coefficient>;
        type P8 = Poly<8, ConstModulus<16>, Coefficient>;
        type P2 = Poly<2, ConstModulus<16>, Coefficient>;
        let basis = ConstRnsBasis::<7681, 12289>;
        let p = ConstModulus::<16>;
        let mut prg = Shake256Prg::new(b"extr-rns");
        let sk = SecretKey::<8, Rns8>::keygen(basis, Distribution::Ternary, &mut prg);
        let m_coeffs = [2u64, 7, 1, 8, 2, 8, 1, 8];
        let pt: P8 = Poly::new(p, m_coeffs);
        let encoded = encode::<8, Rns8, P8>(&pt, basis);
        let rlwe = sk.encrypt(&encoded, Distribution::Ternary, &mut prg);

        let mlwe = extr::<8, 2, 4, _>(&rlwe);
        let key_vec: [PolyRns<2, ConstRnsBasis<7681, 12289>, Coefficient>; 4] =
            core::array::from_fn(|j| sk.poly().project_at::<2>(j));
        let recovered: P2 = mlwe_decrypt(&mlwe, &key_vec, p);
        assert_eq!(recovered.coeff(0).to_u64(), m_coeffs[0]);
        assert_eq!(recovered.coeff(1).to_u64(), m_coeffs[4]);
    }

    /// Shape: `extr` output rank/degree match $(N/D, D)$ at the type level.
    #[test]
    fn extr_output_shape() {
        type R8 = Poly<8, ConstModulus<65537>, Coefficient>;
        let q = ConstModulus::<65537>;
        let mut prg = Shake256Prg::new(b"extr-shape");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let rlwe = sk.encrypt(&Poly::zero(q), Distribution::Ternary, &mut prg);
        let mlwe: MLWECiphertext<2, 4, Poly<4, ConstModulus<65537>, Coefficient>> =
            extr::<8, 4, 2, _>(&rlwe);
        assert_eq!(mlwe.masks.len(), 2);
        let _: Poly<4, ConstModulus<65537>, Coefficient> = mlwe.body;
    }
}
