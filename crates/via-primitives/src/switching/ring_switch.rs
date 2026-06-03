//! §3.3 ring switch + ring-switch-key generation. See
//! `.docs/primitives.md` §3.3.
//!
//! Convert an RLWE ciphertext in $R_{n_1, q}$ under $S_1$ to an RLWE in
//! $R_{n_2, q}$ under $S_2$ encrypting $\pi_0^{n_1 \to n_2}(M)$ — the slot-0
//! projection of the original message into the smaller ring. This is the
//! second half of VIA-C's `RespComp` (§6.2) and the per-column step of VIA's
//! `FirstDim`.
//!
//! ## Algebraic identity
//!
//! In the negacyclic ring $R_{n, q} = \mathbb{Z}_q\lbrack X\rbrack / (X^n + 1)$,
//! $$
//! \pi_0(A \cdot S_1) = \sum_{j=0}^{d-1} \pi_0(X^j \cdot A)
//!     \cdot \pi_0(X^{-j} \cdot S_1), \qquad d = n_1 / n_2,
//! $$
//! where $X^{-j} \equiv -X^{n_1 - j} \pmod{X^{n_1} + 1}$. [`gen_rsk`]
//! encrypts the $d$ *key* halves $\pi_0(X^{-j} S_1)$ under $S_2$;
//! [`ring_switch`] supplies the *public mask* halves $\pi_0(X^j A)$ and
//! combines them via gadget products (§2.4).

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::algebra::ring::abstraction::RingPoly;
use crate::encryption::types::{RLWECiphertext, RLevCiphertext, SecretKey};
use crate::sampling::distribution::Distribution;
use crate::sampling::prg::Shake256Prg;

/// Ring-switch key: $d = N_1 / N_2$ RLev samples that drive conversion of an
/// $R_{N_1, q}$ ciphertext under $S_1$ to an $R_{N_2, q}$ ciphertext under
/// $S_2$ (§3.3). Sample `j` encrypts $\pi_0(X^{-j} \cdot S_1)$ under $S_2$.
///
/// The `D` const-generic is the sample count $d$; the compile-time invariant
/// $N_1 = N_2 \cdot D$ is checked by [`Self::_CHECK`].
///
/// Construct one via [`gen_rsk`] rather than by hand (the samples are RLev
/// encryptions under $S_2$):
///
/// ```rust
/// use via_primitives::algebra::ring::element::Poly;
/// use via_primitives::algebra::ring::form::Coefficient;
/// use via_primitives::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_primitives::encryption::types::SecretKey;
/// use via_primitives::sampling::distribution::Distribution;
/// use via_primitives::sampling::prg::Shake256Prg;
/// use via_primitives::switching::ring_switch::{gen_rsk, RingSwitchKey};
///
/// // Toy params: N1 = 64, N2 = 16, D = 4, L = 2.
/// type Q<const N: usize> = Poly<N, PowerOfTwoModulus<32>, Coefficient>;
/// let qm = PowerOfTwoModulus::<32>;
/// let mut prg = Shake256Prg::new(b"doc-rsk");
/// let s1 = SecretKey::<64, Q<64>>::keygen(qm, Distribution::Ternary, &mut prg);
/// let s2 = SecretKey::<16, Q<16>>::keygen(qm, Distribution::Ternary, &mut prg);
/// let rsk: RingSwitchKey<64, 16, Q<16>, 2, 4> =
///     gen_rsk(&s1, &s2, 4, Distribution::Ternary, &mut prg);
/// assert_eq!(rsk.samples.len(), 4);
/// ```
//
// TODO(generic_const_exprs): once `generic_const_exprs` stabilises, `D` could
// be computed as `N1 / N2` rather than passed and cross-checked by `_CHECK`.
pub struct RingSwitchKey<
    const N1: usize,
    const N2: usize,
    R2: RingPoly<N2>,
    const L: usize,
    const D: usize,
> {
    /// The $D$ RLev samples, indexed by $j \in \lbrack D\rbrack$. Sample `j` encrypts
    /// $\pi_0(X^{-j} S_1)$ under $S_2$ in $R_{N_2, q}$.
    pub samples: [RLevCiphertext<N2, R2, L>; D],
}

impl<const N1: usize, const N2: usize, R2: RingPoly<N2>, const L: usize, const D: usize>
    RingSwitchKey<N1, N2, R2, L, D>
{
    /// Compile-time validation block. Asserts the degree relationship
    /// $N_1 = N_2 \cdot D$, plus $N_2 \ge 2$ and $N_2$ a power of two
    /// (required by the smaller-ring [`RingPoly`] backend). Forced to
    /// evaluate inside [`Self::new`], [`gen_rsk`], and [`ring_switch`] so a
    /// mismatched instantiation fails to compile / panics at first use.
    pub const _CHECK: () = {
        assert!(N2 >= 2, "RingSwitchKey: N2 >= 2");
        assert!(
            N2.is_power_of_two(),
            "RingSwitchKey: N2 must be a power of two",
        );
        assert!(
            N1 == N2 * D,
            "RingSwitchKey: N1 must equal N2 * D (d = n1 / n2 samples)",
        );
    };

    /// Construct a ring-switch key from its $D$ RLev samples. Forces
    /// [`Self::_CHECK`].
    #[inline]
    pub fn new(samples: [RLevCiphertext<N2, R2, L>; D]) -> Self {
        let () = Self::_CHECK;
        Self { samples }
    }
}

impl<const N1: usize, const N2: usize, R2: RingPoly<N2>, const L: usize, const D: usize> Zeroize
    for RingSwitchKey<N1, N2, R2, L, D>
{
    fn zeroize(&mut self) {
        for s in &mut self.samples {
            s.zeroize();
        }
    }
}

impl<const N1: usize, const N2: usize, R2: RingPoly<N2>, const L: usize, const D: usize> Drop
    for RingSwitchKey<N1, N2, R2, L, D>
{
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl<const N1: usize, const N2: usize, R2: RingPoly<N2>, const L: usize, const D: usize>
    ZeroizeOnDrop for RingSwitchKey<N1, N2, R2, L, D>
{
}

impl<const N1: usize, const N2: usize, R2: RingPoly<N2>, const L: usize, const D: usize>
    core::fmt::Debug for RingSwitchKey<N1, N2, R2, L, D>
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("RingSwitchKey")
            .field("N1", &N1)
            .field("N2", &N2)
            .field("L", &L)
            .field("D", &D)
            .finish_non_exhaustive()
    }
}

/// §3.3 — generate a ring-switch key from source key $S_1 \in R_{N_1, q}$ to
/// target key $S_2 \in R_{N_2, q}$.
///
/// Produces $D = N_1 / N_2$ RLev samples; sample `j` encrypts
/// $\pi_0(X^{-j} S_1)$ under $S_2$, using $X^{-j} \equiv -X^{N_1 - j}$ in the
/// negacyclic ring. For $j = 0$ this is simply $\pi_0(S_1)$.
///
/// # Source-key modulus
///
/// The `CenteredScalar = i64` bound on `R1` restricts the source key to a
/// **single-prime** modulus. For VIA-C, where $S_1$ is sampled at the
/// RNS-composite $q_1$, the caller must first rekey $S_1$ to the working
/// single-prime modulus (typically $q_3$) via the §3.4
/// [`rekey_secret_key`](super::rekey::rekey_secret_key) helper before calling
/// `gen_rsk`.
///
/// # PRG order
///
/// `j`-outer; within each `j`, [`SecretKey::encrypt_rlev`] draws
/// `[mask, error]` per gadget level. This exact ordering is the
/// cross-language parity contract (locked by the Part 5 PRG-order KAT).
///
/// ```rust
/// use via_primitives::algebra::ring::element::Poly;
/// use via_primitives::algebra::ring::form::Coefficient;
/// use via_primitives::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_primitives::encryption::types::SecretKey;
/// use via_primitives::sampling::distribution::Distribution;
/// use via_primitives::sampling::prg::Shake256Prg;
/// use via_primitives::switching::ring_switch::{gen_rsk, RingSwitchKey};
///
/// type Q<const N: usize> = Poly<N, PowerOfTwoModulus<32>, Coefficient>;
/// let qm = PowerOfTwoModulus::<32>;
/// let mut prg = Shake256Prg::new(b"doc-gen-rsk");
/// let s1 = SecretKey::<64, Q<64>>::keygen(qm, Distribution::Ternary, &mut prg);
/// let s2 = SecretKey::<16, Q<16>>::keygen(qm, Distribution::Ternary, &mut prg);
/// // D = n1/n2 = 4 samples, gadget depth L = 2, base 4.
/// let _rsk: RingSwitchKey<64, 16, Q<16>, 2, 4> =
///     gen_rsk(&s1, &s2, 4, Distribution::Ternary, &mut prg);
/// ```
#[allow(non_camel_case_types)]
pub fn gen_rsk<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1, Modulus = <R2 as RingPoly<N2>>::Modulus, Projected<N2> = R2, CenteredScalar = i64>,
    R2: RingPoly<N2>,
    const L: usize,
    const D: usize,
>(
    src_sk: &SecretKey<N1, R1>,
    dst_sk: &SecretKey<N2, R2>,
    base: u64,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
) -> RingSwitchKey<N1, N2, R2, L, D> {
    let () = RingSwitchKey::<N1, N2, R2, L, D>::_CHECK;
    let samples = core::array::from_fn(|j| {
        let key_proj: R2 = if j == 0 {
            src_sk.poly().project_at::<N2>(0)
        } else {
            // X^{-j} · S1 ≡ -X^{N1-j} · S1 in the negacyclic ring.
            let rotated = src_sk.poly().mul_x_pow(N1 - j);
            (-rotated).project_at::<N2>(0)
        };
        dst_sk.encrypt_rlev::<L>(&key_proj, base, error_dist, prg)
    });
    RingSwitchKey::new(samples)
}

/// §3.3 — convert an RLWE ciphertext from $R_{N_1, q}$ to $R_{N_2, q}$.
///
/// Homomorphically evaluates the slot-0 projection: the result decrypts under
/// $S_2$ to $\pi_0(M)$ in $R_{N_2, q}$. All arithmetic stays in coefficient
/// form (no NTT round-trip).
///
/// `base` must match the gadget base passed to [`gen_rsk`]; the gadget depth
/// $L$ is carried on the key type.
///
/// ```rust
/// use via_primitives::algebra::ring::abstraction::RingPoly;
/// use via_primitives::algebra::ring::element::Poly;
/// use via_primitives::algebra::ring::form::Coefficient;
/// use via_primitives::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_primitives::encryption::rlwe::encode;
/// use via_primitives::encryption::types::SecretKey;
/// use via_primitives::sampling::distribution::Distribution;
/// use via_primitives::sampling::prg::Shake256Prg;
/// use via_primitives::switching::ring_switch::{gen_rsk, ring_switch, RingSwitchKey};
///
/// type Q<const N: usize> = Poly<N, PowerOfTwoModulus<32>, Coefficient>;
/// type P<const N: usize> = Poly<N, PowerOfTwoModulus<4>, Coefficient>; // p = 16
/// let qm = PowerOfTwoModulus::<32>;
/// let pm = PowerOfTwoModulus::<4>;
/// let mut prg = Shake256Prg::new(b"doc-ring-switch");
/// let s1 = SecretKey::<64, Q<64>>::keygen(qm, Distribution::Ternary, &mut prg);
/// let s2 = SecretKey::<16, Q<16>>::keygen(qm, Distribution::Ternary, &mut prg);
/// let rsk: RingSwitchKey<64, 16, Q<16>, 8, 4> =
///     gen_rsk(&s1, &s2, 8, Distribution::Ternary, &mut prg);
///
/// // Encrypt m in R_{64, q} under S1; coefficient 0 is 7.
/// let m_coeffs: [u64; 64] = core::array::from_fn(|i| (i as u64) % 16);
/// let pt = <P<64>>::new(pm, m_coeffs);
/// let encoded: Q<64> = encode(&pt, qm);
/// let ct = s1.encrypt(&encoded, Distribution::Ternary, &mut prg);
///
/// // Ring-switch to R_{16, q} under S2; decrypt recovers pi_0(m).
/// let switched = ring_switch(&ct, &rsk, 8);
/// let recovered: P<16> = s2.decrypt(&switched, pm);
/// // pi_0 picks coefficients at positions d*i (d = 4): m[0], m[4], ...
/// assert_eq!(recovered.coeff(0).to_u64(), m_coeffs[0]);
/// assert_eq!(recovered.coeff(1).to_u64(), m_coeffs[4]);
/// ```
#[allow(non_camel_case_types)]
pub fn ring_switch<
    const N1: usize,
    const N2: usize,
    R: RingPoly<N1, Projected<N2>: RingPoly<N2, Modulus = R::Modulus>>,
    const L: usize,
    const D: usize,
>(
    ct: &RLWECiphertext<N1, R>,
    rsk: &RingSwitchKey<N1, N2, R::Projected<N2>, L, D>,
    base: u64,
) -> RLWECiphertext<N2, R::Projected<N2>> {
    // Re-evaluate _CHECK at the entry point to defend against an RSK built
    // by struct literal (bypassing `RingSwitchKey::new`, the only other path
    // that forces it).
    let () = RingSwitchKey::<N1, N2, R::Projected<N2>, L, D>::_CHECK;
    let modulus = ct.mask.modulus();
    let body_proj_0 = ct.body.project_at::<N2>(0);
    let mut acc_mask = <R::Projected<N2>>::zero(modulus);
    let mut acc_body = body_proj_0;
    for j in 0..D {
        let mask_proj_j = if j == 0 {
            ct.mask.project_at::<N2>(0)
        } else {
            ct.mask.mul_x_pow(j).project_at::<N2>(0)
        };
        let gp = rsk.samples[j].gadget_product(&mask_proj_j, base);
        acc_mask -= gp.mask;
        acc_body -= gp.body;
    }
    RLWECiphertext::new(acc_mask, acc_body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::zq::modulus::PowerOfTwoModulus;
    use crate::encryption::rlwe::encode;

    // Toy single-prime backend at q = 2^32, with N1 = 64, N2 = 16, D = 4.
    type Q32<const N: usize> = Poly<N, PowerOfTwoModulus<32>, Coefficient>;

    #[test]
    fn module_shell_compiles() {
        // Smoke: the module type-checks and a key can be constructed.
        let _ = core::mem::size_of::<RingSwitchKey<64, 16, Q32<16>, 4, 4>>();
    }

    #[test]
    fn gen_rsk_smoke() {
        let qm = PowerOfTwoModulus::<32>;
        let mut prg = Shake256Prg::new(b"gen-rsk-smoke");
        let s1 = SecretKey::<64, Q32<64>>::keygen(qm, Distribution::Ternary, &mut prg);
        let s2 = SecretKey::<16, Q32<16>>::keygen(qm, Distribution::Ternary, &mut prg);
        let rsk: RingSwitchKey<64, 16, Q32<16>, 4, 4> =
            gen_rsk(&s1, &s2, 4, Distribution::Ternary, &mut prg);
        assert_eq!(rsk.samples.len(), 4);
    }

    #[test]
    fn rsk_zeroize_runs_without_panic() {
        let qm = PowerOfTwoModulus::<32>;
        let mut prg = Shake256Prg::new(b"rsk-zeroize");
        let s1 = SecretKey::<64, Q32<64>>::keygen(qm, Distribution::Ternary, &mut prg);
        let s2 = SecretKey::<16, Q32<16>>::keygen(qm, Distribution::Ternary, &mut prg);
        let mut rsk: RingSwitchKey<64, 16, Q32<16>, 4, 4> =
            gen_rsk(&s1, &s2, 4, Distribution::Ternary, &mut prg);
        rsk.zeroize();
        // Dropping a zeroized key must not panic (Drop re-zeroizes).
    }

    /// Full §3.3 round-trip at toy paper-shaped params: encrypt $m$ in
    /// $R_{N_1, q}$ under $S_1$, ring-switch to $R_{N_2, q}$ under $S_2$, and
    /// recover $\pi_0(m)$ by decrypting under $S_2$.
    #[test]
    fn ring_switch_round_trip_at_paper_params() {
        const N1: usize = 64;
        const N2: usize = 16;
        let qm = PowerOfTwoModulus::<32>;
        let pm = PowerOfTwoModulus::<4>; // p = 16

        let mut prg = Shake256Prg::new(b"ring-switch-roundtrip");
        let s1 = SecretKey::<N1, Q32<N1>>::keygen(qm, Distribution::Ternary, &mut prg);
        let s2 = SecretKey::<N2, Q32<N2>>::keygen(qm, Distribution::Ternary, &mut prg);
        let rsk: RingSwitchKey<N1, N2, Q32<N2>, 8, 4> =
            gen_rsk(&s1, &s2, 8, Distribution::Ternary, &mut prg);

        // Plaintext m with distinct small coefficients in [0, p).
        let m_coeffs: [u64; N1] = core::array::from_fn(|i| (i as u64) % 16);
        let pt: Poly<N1, PowerOfTwoModulus<4>, Coefficient> = Poly::new(pm, m_coeffs);
        let encoded: Q32<N1> = encode(&pt, qm);
        let ct = s1.encrypt(&encoded, Distribution::Ternary, &mut prg);

        let switched = ring_switch(&ct, &rsk, 8);
        let recovered: Poly<N2, PowerOfTwoModulus<4>, Coefficient> = s2.decrypt(&switched, pm);

        // pi_0^{N1 -> N2}(m): coefficient at position d*i (d = N1/N2 = 4).
        let d = N1 / N2;
        let got: [u64; N2] = core::array::from_fn(|i| recovered.coeff(i).to_u64());
        let expected: [u64; N2] = core::array::from_fn(|i| m_coeffs[d * i]);
        assert_eq!(got, expected);
    }

    /// Edge-case coverage: exercise the `mul_x_pow(N1 - j)` branch for the
    /// largest `j = D - 1` at small params (N1=32, N2=8, D=4). Round-trip
    /// plaintext recovery confirms the negacyclic rotation is handled at the
    /// boundary slot.
    #[test]
    fn ring_switch_x_pow_j_branch_coverage() {
        const N1: usize = 32;
        const N2: usize = 8;
        let qm = PowerOfTwoModulus::<32>;
        let pm = PowerOfTwoModulus::<4>;

        let mut prg = Shake256Prg::new(b"ring-switch-edge");
        let s1 = SecretKey::<N1, Q32<N1>>::keygen(qm, Distribution::Ternary, &mut prg);
        let s2 = SecretKey::<N2, Q32<N2>>::keygen(qm, Distribution::Ternary, &mut prg);
        let rsk: RingSwitchKey<N1, N2, Q32<N2>, 2, 4> =
            gen_rsk(&s1, &s2, 256, Distribution::Ternary, &mut prg);

        let m_coeffs: [u64; N1] = core::array::from_fn(|i| (i as u64) % 16);
        let pt: Poly<N1, PowerOfTwoModulus<4>, Coefficient> = Poly::new(pm, m_coeffs);
        let encoded: Q32<N1> = encode(&pt, qm);
        let ct = s1.encrypt(&encoded, Distribution::Ternary, &mut prg);

        let switched = ring_switch(&ct, &rsk, 256);
        let recovered: Poly<N2, PowerOfTwoModulus<4>, Coefficient> = s2.decrypt(&switched, pm);

        let d = N1 / N2;
        let got: [u64; N2] = core::array::from_fn(|i| recovered.coeff(i).to_u64());
        let expected: [u64; N2] = core::array::from_fn(|i| m_coeffs[d * i]);
        assert_eq!(got, expected);
    }

    /// PRG-order KAT anchor: the first RLev sample's mask at VIA-C TOY
    /// params, reproduced from the same seed as `gen_layer3_kats.py`
    /// (`GEN_RSK_J0_L0_MASK`). The full D×L×N2 byte stream is locked by the
    /// integration test `tests/layer3_kats.rs::kat_gen_rsk_prg_order`; this
    /// inline anchor guards the j-outer / level-inner / mask-before-error
    /// ordering without leaving the unit-test crate.
    #[test]
    fn gen_rsk_prg_order_kat() {
        const EXPECTED_J0_L0_MASK: [u64; 16] = [
            7400, 8742, 15653, 8677, 30164, 21318, 53218, 35000, 18439, 57295, 43129, 45238, 44620,
            9098, 15997, 83,
        ];
        type Q3<const N: usize> = Poly<N, PowerOfTwoModulus<16>, Coefficient>;
        let q3 = PowerOfTwoModulus::<16>;
        let mut prg = Shake256Prg::new(b"layer3-kat-gen-rsk-rsk");
        let s1 = SecretKey::<64, Q3<64>>::keygen(q3, Distribution::Ternary, &mut prg);
        let s2 = SecretKey::<16, Q3<16>>::keygen(q3, Distribution::Ternary, &mut prg);
        let rsk: RingSwitchKey<64, 16, Q3<16>, 8, 4> =
            gen_rsk(&s1, &s2, 4, Distribution::Ternary, &mut prg);
        let first = &rsk.samples[0].samples[0];
        let got: [u64; 16] = core::array::from_fn(|i| first.mask.coeff(i).to_u64());
        assert_eq!(got, EXPECTED_J0_L0_MASK);
    }
}
