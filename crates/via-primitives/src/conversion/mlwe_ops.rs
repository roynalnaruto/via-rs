//! §5.1 — MLWE embedding, RLWE↔MLWE conversions, and LWE encrypt/decrypt.
//!
//! See `.docs/primitives.md` §5.1. These compose the §0.5 ring embedding
//! ([`RingPoly::embed_at`]) with the Layer-2 ciphertext types. [`encrypt_lwe`]
//! / [`decrypt_lwe`] are the testing helpers that let the §5.3 cascade be
//! exercised end-to-end (the production query path uses a $\Delta$-free LWE
//! encryption that lands in Layer 6).

use crate::algebra::ring::RingPoly;
use crate::algebra::ring::element::Poly;
use crate::algebra::ring::form::Coefficient;
use crate::algebra::ring::rns_element::PolyRns;
use crate::algebra::rns::basis::RnsBasis;
use crate::algebra::zq::modulus::Modulus;
use crate::conversion::kernels::lwe::dot_residues;
use crate::encryption::MLWECiphertext;
use crate::encryption::types::{RLWECiphertext, SecretKey};
use crate::sampling::distribution::Distribution;
use crate::sampling::prg::Shake256Prg;

// ---------------------------------------------------------------------------
// LweDot — sealed conversion-internal bridge to the per-prime CT kernel.
// ---------------------------------------------------------------------------

/// Conversion-internal bridge (sealed) for the §5.1 LWE body dot product.
///
/// Computes $\bigl(\sum_i a_i \cdot s_i\bigr) \bmod q$ on raw residues via the
/// constant-time flat-slice kernel
/// [`crate::conversion::kernels::lwe::dot_residues`], returning the result as a
/// degree-1 element of this backend
/// ([`RingPoly::Projected<1>`](RingPoly::Projected)). For RNS the kernel is
/// applied once per prime lane; for single-prime, once.
///
/// Auto-implemented for the two [`RingPoly`] backends; you never implement it —
/// the [`RingPoly`] supertrait is sealed, so no foreign type can be an
/// `LweDot`. It exists only so [`encrypt_lwe`] / [`decrypt_lwe`] stay generic
/// over the backend while the numeric core remains a GPU-portable kernel.
pub trait LweDot<const N: usize>: RingPoly<N> {
    /// $\bigl(\sum_i \mathrm{masks}_i \cdot \mathrm{key}_i\bigr) \bmod q$ as a
    /// degree-1 ring element. `masks` is the length-$N$ mask polynomial (its
    /// $N$ coefficients are the LWE mask scalars); `key` is the degree-$N$
    /// secret key. Both are read as canonical residues — exactly as the Python
    /// reference computes the body (`pir/primitives/mlwe.py:149-153`).
    fn lwe_dot(masks: &Self, key: &Self) -> Self::Projected<1>;
}

impl<const N: usize, M: Modulus> LweDot<N> for Poly<N, M, Coefficient> {
    #[inline]
    fn lwe_dot(masks: &Self, key: &Self) -> Poly<1, M, Coefficient> {
        let modulus = RingPoly::modulus(masks);
        let dot = dot_residues(masks.values(), key.values(), modulus.q());
        // `dot ∈ [0, q)` from the kernel's final reduction; `new` re-reduces
        // (a no-op here) so no `unsafe` is needed.
        Poly::<1, M, Coefficient>::new(modulus, [dot])
    }
}

impl<const N: usize, B: RnsBasis> LweDot<N> for PolyRns<N, B, Coefficient> {
    #[inline]
    fn lwe_dot(masks: &Self, key: &Self) -> PolyRns<1, B, Coefficient> {
        let basis = masks.basis();
        let d0 = dot_residues(masks.values0(), key.values0(), basis.m0().q());
        let d1 = dot_residues(masks.values1(), key.values1(), basis.m1().q());
        // SAFETY: `d0 ∈ [0, q0)` and `d1 ∈ [0, q1)` — each kernel call returns
        // a residue reduced modulo its own prime.
        unsafe { PolyRns::<1, B, Coefficient>::from_reduced_unchecked(basis, [d0], [d1]) }
    }
}

// ---------------------------------------------------------------------------
// §5.1 — MLWE embedding and RLWE↔MLWE conversions.
// ---------------------------------------------------------------------------

/// §5.1 — embed an $(m, n)$-MLWE into an $(m, N_\text{large})$-MLWE by applying
/// $\iota_0$ ([`RingPoly::embed_at`] at slot 0) to every mask and the body.
///
/// Because $\iota_0$ is a ring homomorphism ($X \mapsto Y^d$, see
/// `.docs/primitives.md` §0.5), the result preserves the message under the
/// correspondingly embedded key $\iota_0(\mathbf{S})$. `paper:mlwe.py:41-62`.
///
/// VIA-B note: this is the $d = 1$, slot-0 case of §7.1 `Embed_d`; the
/// multi-input `Embed_d` will compose [`RingPoly::embed_at`] over slots
/// $0, \ldots, d-1$ — no change to this function or the trait is needed.
pub fn embed_mlwe<const RANK: usize, const N: usize, const N_LARGE: usize, R, RL>(
    ct: &MLWECiphertext<RANK, N, R>,
) -> MLWECiphertext<RANK, N_LARGE, RL>
where
    R: RingPoly<N, Embedded<N_LARGE> = RL>,
    RL: RingPoly<N_LARGE>,
{
    let masks = core::array::from_fn(|i| ct.masks[i].embed_at::<N_LARGE>(0));
    MLWECiphertext::new(masks, ct.body.embed_at::<N_LARGE>(0))
}

/// Wrap an [`RLWECiphertext`] as a rank-1 [`MLWECiphertext`]. `paper:mlwe.py:65-77`.
pub fn rlwe_to_mlwe<const N: usize, R: RingPoly<N>>(
    ct: &RLWECiphertext<N, R>,
) -> MLWECiphertext<1, N, R> {
    MLWECiphertext::new([ct.mask], ct.body)
}

/// Unwrap a rank-1 [`MLWECiphertext`] to an [`RLWECiphertext`]. The `RANK = 1`
/// const-generic makes "rank must be 1" a compile-time fact — there is no
/// runtime `ValueError` as in `paper:mlwe.py:80-99`.
pub fn mlwe_to_rlwe<const N: usize, R: RingPoly<N>>(
    ct: &MLWECiphertext<1, N, R>,
) -> RLWECiphertext<N, R> {
    RLWECiphertext::new(ct.masks[0], ct.body)
}

// ---------------------------------------------------------------------------
// §5.1 — LWE encryption / decryption (an (n, 1)-MLWE).
// ---------------------------------------------------------------------------

/// §5.1 — encrypt a scalar `message` $\in [0, p)$ as an $(n, 1)$-MLWE (LWE)
/// under the degree-$n$ secret key `sk`.
///
/// The body is $B = \bigl(\sum_i a_i \cdot s_i\bigr) + e + \Delta \cdot m$ with
/// $\Delta = \lceil q / p \rceil$; the dot product runs through the
/// constant-time kernel via [`LweDot`]. The $n$ uniform mask scalars are stored
/// as $n$ degree-1 polynomials. `paper:mlwe.py:102-156`.
///
/// # PRG consumption order
///
/// `n` uniform mask scalars first (one [`RingPoly::random_uniform`] over the
/// degree-$n$ ring draws the same $n$ values, in order, as Python's $n$
/// `randbelow(q)` calls), then **one** error sample — matching `mlwe.py:138-147`.
///
/// # Constant-time
///
/// The secret-dependent step $\sum_i a_i s_i$ is the constant-time
/// [`dot_residues`] kernel.
///
/// ```rust
/// use via_primitives::algebra::ring::element::Poly;
/// use via_primitives::algebra::ring::form::Coefficient;
/// use via_primitives::algebra::zq::modulus::ConstModulus;
/// use via_primitives::conversion::{decrypt_lwe, encrypt_lwe};
/// use via_primitives::encryption::types::SecretKey;
/// use via_primitives::sampling::distribution::Distribution;
/// use via_primitives::sampling::prg::Shake256Prg;
///
/// type R = Poly<8, ConstModulus<65537>, Coefficient>;
/// let q = ConstModulus::<65537>;
/// let mut prg = Shake256Prg::new(b"doc-encrypt-lwe");
/// let sk = SecretKey::<8, R>::keygen(q, Distribution::Ternary, &mut prg);
/// let ct = encrypt_lwe(&sk, 5, 16, Distribution::Ternary, &mut prg);
/// assert_eq!(ct.masks.len(), 8); // (n, 1)-MLWE: n masks
/// assert_eq!(decrypt_lwe(&ct, &sk, 16), 5);
/// ```
pub fn encrypt_lwe<const NLWE: usize, R: LweDot<NLWE>>(
    sk: &SecretKey<NLWE, R>,
    message: u64,
    p: u64,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
) -> MLWECiphertext<NLWE, 1, R::Projected<1>> {
    let q_mod = RingPoly::modulus(sk.poly());
    // 1. n uniform mask scalars (Python: n × randbelow(q)), held as one deg-N poly.
    let masks_poly = R::random_uniform(q_mod, prg);
    // 2. dot = Σ aᵢ·sᵢ mod q (degree 1), via the constant-time kernel.
    let dot = R::lwe_dot(&masks_poly, sk.poly());
    // 3. one error sample (Python: ternary_poly(1) / discrete_gaussian_poly(1)).
    let mut error_sample = [0i64; 1];
    error_dist.sample_into(prg, &mut error_sample);
    let error = <R::Projected<1> as RingPoly<1>>::from_centered_i64s(q_mod, &error_sample);
    // 4. Δ·message mod q, with Δ = ⌈q/p⌉ (integer-only).
    let q_val = <R as RingPoly<NLWE>>::modulus_value(q_mod);
    let delta = q_val.div_ceil(u128::from(p));
    let dm_val = (delta * u128::from(message)) % q_val;
    let encoded = <R::Projected<1> as RingPoly<1>>::from_u128_coeffs(q_mod, &[dm_val]);
    // 5. body = dot + e + Δm ; masks = the n coefficients of `masks_poly`.
    let body = dot + error + encoded;
    let masks = core::array::from_fn(|i| masks_poly.project_at::<1>(i));
    MLWECiphertext::new(masks, body)
}

/// Δ-**free** sibling of [`encrypt_lwe`]: encrypt a raw `u128` value directly
/// into the body, with **no** $\Delta = \lceil q/p \rceil$ encoding.
///
/// VIA-C query compression encrypts each gadget level $b \cdot g_i$ (with
/// $g_i = \lceil q_1 / B^i \rceil$ the gadget value and $b$ the query bit) as a
/// raw LWE sample; the gadget scaling already places the value where the
/// downstream `rlwe_to_rgsw` expects it, so no plaintext $\Delta$ is applied.
/// The message is `u128` because at the paper $q_1 \approx 2^{75}$ the
/// gadget-scaled $b \cdot g_i$ exceeds `u64` for small $i$.
///
/// `body = \sum_i a_i s_i + e + (\text{message} \bmod q)`; the `n` mask scalars
/// are returned as the `(n, 1)`-MLWE masks.
///
/// # PRG consumption order
///
/// Identical to [`encrypt_lwe`]: `n` uniform mask scalars first (one
/// [`RingPoly::random_uniform`] over the degree-$n$ ring), then **one** error
/// sample. This order is the cross-language parity contract (Part-5 QueryComp
/// KAT).
///
/// # Constant-time
///
/// The secret-dependent $\sum_i a_i s_i$ is the constant-time [`LweDot::lwe_dot`]
/// kernel, as in [`encrypt_lwe`].
///
/// ```rust
/// use via_primitives::algebra::ring::element::Poly;
/// use via_primitives::algebra::ring::form::Coefficient;
/// use via_primitives::algebra::zq::modulus::ConstModulus;
/// use via_primitives::conversion::{decrypt_lwe, encrypt_lwe_raw};
/// use via_primitives::encryption::types::SecretKey;
/// use via_primitives::sampling::distribution::Distribution;
/// use via_primitives::sampling::prg::Shake256Prg;
///
/// type R = Poly<8, ConstModulus<65537>, Coefficient>;
/// let q = ConstModulus::<65537>;
/// let mut prg = Shake256Prg::new(b"doc-encrypt-lwe-raw");
/// let sk = SecretKey::<8, R>::keygen(q, Distribution::Ternary, &mut prg);
/// // A raw LWE of the Δ-scaled value Δ·5 (Δ = ⌈q/p⌉, p = 16) decrypts to 5 —
/// // i.e. `encrypt_lwe_raw(sk, Δ·m, …)` ≡ `encrypt_lwe(sk, m, p, …)`.
/// let delta = 65537u128.div_ceil(16);
/// let ct = encrypt_lwe_raw(&sk, delta * 5, Distribution::Ternary, &mut prg);
/// assert_eq!(ct.masks.len(), 8); // (n, 1)-MLWE: n masks
/// assert_eq!(decrypt_lwe(&ct, &sk, 16), 5);
/// ```
pub fn encrypt_lwe_raw<const NLWE: usize, R: LweDot<NLWE>>(
    sk: &SecretKey<NLWE, R>,
    message: u128,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
) -> MLWECiphertext<NLWE, 1, R::Projected<1>> {
    let q_mod = RingPoly::modulus(sk.poly());
    // 1. n uniform mask scalars (same draw as `encrypt_lwe`).
    let masks_poly = R::random_uniform(q_mod, prg);
    // 2. dot = Σ aᵢ·sᵢ mod q (degree 1), constant-time kernel.
    let dot = R::lwe_dot(&masks_poly, sk.poly());
    // 3. one error sample (same draw as `encrypt_lwe`).
    let mut error_sample = [0i64; 1];
    error_dist.sample_into(prg, &mut error_sample);
    let error = <R::Projected<1> as RingPoly<1>>::from_centered_i64s(q_mod, &error_sample);
    // 4. raw message mod q — NO Δ encoding (the gadget scaling is already baked in).
    let q_val = <R as RingPoly<NLWE>>::modulus_value(q_mod);
    let m_val = message % q_val;
    let encoded = <R::Projected<1> as RingPoly<1>>::from_u128_coeffs(q_mod, &[m_val]);
    // 5. body = dot + e + m ; masks = the n coefficients of `masks_poly`.
    let body = dot + error + encoded;
    let masks = core::array::from_fn(|i| masks_poly.project_at::<1>(i));
    MLWECiphertext::new(masks, body)
}

/// §5.1 — decrypt an $(n, 1)$-MLWE to a scalar in $[0, p)$:
/// $\mathrm{decode}\bigl(B - \sum_i a_i \cdot s_i\bigr)$. The decode rounding
/// mirrors [`crate::encryption::decode`] exactly (centre, then
/// `(p·c + q/2).div_euclid(q)`, then `rem_euclid(p)`). `paper:mlwe.py:159-192`.
pub fn decrypt_lwe<const NLWE: usize, R: LweDot<NLWE>>(
    ct: &MLWECiphertext<NLWE, 1, R::Projected<1>>,
    sk: &SecretKey<NLWE, R>,
    p: u64,
) -> u64 {
    let q_mod = RingPoly::modulus(sk.poly());
    // Gather the n degree-1 mask scalars into a degree-N poly for the kernel.
    let mut mask_coeffs = [0u128; NLWE];
    for (slot, mask) in mask_coeffs.iter_mut().zip(ct.masks.iter()) {
        let mut one = [0u128; 1];
        mask.to_u128_coeffs(&mut one);
        *slot = one[0];
    }
    let masks_poly = R::from_u128_coeffs(q_mod, &mask_coeffs);
    let dot = R::lwe_dot(&masks_poly, sk.poly());
    let noisy = ct.body - dot;
    // Decode the single coefficient (integer-only, Python-compatible rounding).
    let q = <R as RingPoly<NLWE>>::modulus_value(q_mod) as i128;
    let p_i = i128::from(p);
    let mut centered = [0i128; 1];
    noisy.to_centered_i128_coeffs(&mut centered);
    let rounded = (p_i * centered[0] + q / 2).div_euclid(q);
    rounded.rem_euclid(p_i) as u64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::rns::basis::ConstRnsBasis;
    use crate::algebra::zq::modulus::ConstModulus;

    type R8 = Poly<8, ConstModulus<65537>, Coefficient>;
    type R4 = Poly<4, ConstModulus<65537>, Coefficient>;

    // -- §5.1 embed_mlwe / conversions ------------------------------------

    #[test]
    fn embed_doubles_dimension_and_preserves_rank() {
        let q = ConstModulus::<65537>;
        let mut prg = Shake256Prg::new(b"embed-shape");
        let sk = SecretKey::<4, R4>::keygen(q, Distribution::Ternary, &mut prg);
        let pt: R4 = Poly::new(q, [3, 5, 7, 9]);
        let rlwe = sk.encrypt(&pt, Distribution::Ternary, &mut prg);
        let mlwe = rlwe_to_mlwe(&rlwe);
        let embedded: MLWECiphertext<1, 8, Poly<8, ConstModulus<65537>, Coefficient>> =
            embed_mlwe(&mlwe);
        assert_eq!(embedded.masks.len(), 1); // rank preserved
        // degree doubled: body is now a degree-8 polynomial (type-level).
        let _: Poly<8, ConstModulus<65537>, Coefficient> = embedded.body;
    }

    #[test]
    fn embed_correctness_decrypt() {
        // ι₀ is a ring homomorphism, so an embedded ciphertext decrypts under
        // the embedded key ι₀(S) to ι₀(M). Use a real plaintext modulus p = 16
        // (Δ = ⌈q/p⌉ large) so the ternary noise rounds away on decode.
        type P4 = Poly<4, ConstModulus<16>, Coefficient>;
        type P8 = Poly<8, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        let mut prg = Shake256Prg::new(b"embed-correct");
        let sk = SecretKey::<4, R4>::keygen(q, Distribution::Ternary, &mut prg);
        let m_coeffs = [1u64, 2, 3, 4];
        let pt: P4 = Poly::new(p, m_coeffs);
        let encoded = crate::encryption::encode::<4, R4, P4>(&pt, q);
        let rlwe = sk.encrypt(&encoded, Distribution::Ternary, &mut prg);

        let embedded = embed_mlwe(&rlwe_to_mlwe(&rlwe));
        let embedded_rlwe = mlwe_to_rlwe(&embedded);
        // ι₀(S): embed the key into the degree-8 ring.
        let sk8 = SecretKey::<8, Poly<8, ConstModulus<65537>, Coefficient>>::from_poly(
            sk.poly().embed_at::<8>(0),
        );
        let recovered: P8 = sk8.decrypt(&embedded_rlwe, p);
        // ι₀ places m[i] at position 2i; coefficient 2i = m[i], odd positions 0.
        for (i, &mi) in m_coeffs.iter().enumerate() {
            assert_eq!(recovered.coeff(2 * i).to_u64(), mi);
            assert_eq!(recovered.coeff(2 * i + 1).to_u64(), 0);
        }
    }

    #[test]
    fn roundtrip_rlwe_mlwe_rlwe_is_identity() {
        let q = ConstModulus::<65537>;
        let mut prg = Shake256Prg::new(b"conv-roundtrip");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let pt: R8 = Poly::new(q, [1, 2, 3, 4, 5, 6, 7, 8]);
        let rlwe = sk.encrypt(&pt, Distribution::Ternary, &mut prg);
        let back = mlwe_to_rlwe(&rlwe_to_mlwe(&rlwe));
        assert_eq!(back.mask, rlwe.mask);
        assert_eq!(back.body, rlwe.body);
    }

    // -- §5.1 encrypt_lwe / decrypt_lwe -----------------------------------

    #[test]
    fn encrypt_lwe_produces_rank_n_degree_1() {
        let q = ConstModulus::<65537>;
        let mut prg = Shake256Prg::new(b"lwe-shape");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let ct = encrypt_lwe(&sk, 7, 16, Distribution::Ternary, &mut prg);
        assert_eq!(ct.masks.len(), 8);
        // each component is degree 1 (type-level): masks are Poly<1,...>.
        let _: Poly<1, ConstModulus<65537>, Coefficient> = ct.masks[0];
        let _: Poly<1, ConstModulus<65537>, Coefficient> = ct.body;
    }

    #[test]
    fn decrypt_lwe_roundtrip_all_messages() {
        let q = ConstModulus::<65537>;
        for message in 0..16u64 {
            let mut prg = Shake256Prg::new(b"lwe-roundtrip");
            let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
            let ct = encrypt_lwe(&sk, message, 16, Distribution::Ternary, &mut prg);
            assert_eq!(decrypt_lwe(&ct, &sk, 16), message, "message {message}");
        }
    }

    #[test]
    fn encrypt_lwe_zero_message() {
        let q = ConstModulus::<65537>;
        let mut prg = Shake256Prg::new(b"lwe-zero");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let ct = encrypt_lwe(&sk, 0, 16, Distribution::Ternary, &mut prg);
        assert_eq!(decrypt_lwe(&ct, &sk, 16), 0);
    }

    /// Paper-class: exercise the RNS + `N = 1` LWE path (query compression lives
    /// at the composite $q_1$). `Q = 5·11 = 55`, `p = 2`.
    #[test]
    fn encrypt_lwe_roundtrip_rns_n1() {
        type Rns8 = PolyRns<8, ConstRnsBasis<5, 11>, Coefficient>;
        let basis = ConstRnsBasis::<5, 11>;
        for message in 0..2u64 {
            let mut prg = Shake256Prg::new(b"lwe-rns");
            let sk = SecretKey::<8, Rns8>::keygen(basis, Distribution::Ternary, &mut prg);
            let ct = encrypt_lwe(&sk, message, 2, Distribution::Ternary, &mut prg);
            let _: PolyRns<1, ConstRnsBasis<5, 11>, Coefficient> = ct.body;
            assert_eq!(decrypt_lwe(&ct, &sk, 2), message, "rns message {message}");
        }
    }

    // -- §5.1 encrypt_lwe_raw (Δ-free) ------------------------------------

    /// `encrypt_lwe_raw(sk, Δ·m mod q, …)` must produce a bit-identical
    /// ciphertext to `encrypt_lwe(sk, m, p, …)` under the same PRG seed — this
    /// pins both correctness (the only difference is the missing Δ encoding) and
    /// the shared PRG consumption order (n masks, then error).
    #[test]
    fn encrypt_lwe_raw_equals_delta_scaled_encrypt_lwe() {
        let q = ConstModulus::<65537>;
        let q_val = 65537u128;
        let delta = q_val.div_ceil(16);
        for m in 0..16u64 {
            let raw = (delta * u128::from(m)) % q_val;

            let mut prg_a = Shake256Prg::new(b"lwe-raw-eq");
            let sk_a = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg_a);
            let ct_scaled = encrypt_lwe(&sk_a, m, 16, Distribution::Ternary, &mut prg_a);

            let mut prg_b = Shake256Prg::new(b"lwe-raw-eq");
            let sk_b = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg_b);
            let ct_raw = encrypt_lwe_raw(&sk_b, raw, Distribution::Ternary, &mut prg_b);

            for i in 0..8 {
                assert_eq!(
                    ct_scaled.masks[i].coeff(0).to_u64(),
                    ct_raw.masks[i].coeff(0).to_u64(),
                    "m {m}, mask {i}"
                );
            }
            assert_eq!(
                ct_scaled.body.coeff(0).to_u64(),
                ct_raw.body.coeff(0).to_u64(),
                "m {m}, body"
            );
        }
    }

    /// Same seed ⇒ identical ciphertext (PRG-order determinism).
    #[test]
    fn encrypt_lwe_raw_deterministic() {
        let q = ConstModulus::<65537>;
        let mk = || {
            let mut prg = Shake256Prg::new(b"lwe-raw-det");
            let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
            encrypt_lwe_raw(&sk, 12345, Distribution::Ternary, &mut prg)
        };
        let a = mk();
        let b = mk();
        for i in 0..8 {
            assert_eq!(a.masks[i].coeff(0).to_u64(), b.masks[i].coeff(0).to_u64());
        }
        assert_eq!(a.body.coeff(0).to_u64(), b.body.coeff(0).to_u64());
    }

    /// Paper-class: at the composite $q_1 \approx 2^{75}$ the Δ-scaled value
    /// $\Delta \cdot m$ exceeds `u64`, so this exercises the `u128` message path
    /// (the whole reason `encrypt_lwe_raw` takes `u128`). `decrypt_lwe` recovers
    /// `m` for all `p = 16` messages.
    #[test]
    fn encrypt_lwe_raw_u128_message_at_q1() {
        type RnsQ1 = PolyRns<8, ConstRnsBasis<137438822401, 274810798081>, Coefficient>;
        let basis = ConstRnsBasis::<137438822401, 274810798081>;
        let q1: u128 = 137438822401u128 * 274810798081u128;
        let delta = q1.div_ceil(16);
        assert!(
            delta > u128::from(u64::MAX),
            "Δ must exceed u64 to genuinely exercise the u128 path"
        );
        for message in 0..16u64 {
            let mut prg = Shake256Prg::new(b"lwe-raw-q1");
            let sk = SecretKey::<8, RnsQ1>::keygen(basis, Distribution::Ternary, &mut prg);
            let raw = delta * u128::from(message);
            let ct = encrypt_lwe_raw(&sk, raw, Distribution::Ternary, &mut prg);
            assert_eq!(
                decrypt_lwe(&ct, &sk, 16),
                message,
                "q1 raw message {message}"
            );
        }
    }
}
