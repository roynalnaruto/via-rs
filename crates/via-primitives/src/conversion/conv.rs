//! §5.2 — single Conv₂ step `conv_step` + per-step key generation
//! `gen_conv_step_key`. See `.docs/primitives.md` §5.2.
//!
//! One step of the LWE→RLWE cascade: convert a $(2m, n)$-MLWE into an
//! $(m, 2n)$-MLWE by embedding every component into the doubled ring
//! ($\iota_0$) and key-switching it from its source key component to the target
//! component, accumulating into the output slots. The two-degree const-generic
//! shape mirrors Layer 3's [`crate::switching::ring_switch::gen_rsk`]: explicit
//! `<N_IN, N_OUT, …>` parameters plus a compile-time [`ConvDims::_CHECK`].

use crate::algebra::ring::{RingPoly, RingPolyEval};
use crate::encryption::MLWECiphertext;
use crate::encryption::types::{RLWECiphertext, RLevCiphertext, SecretKey};
use crate::sampling::distribution::Distribution;
use crate::sampling::prg::Shake256Prg;

/// Compile-time degree/rank relationship for a single Conv₂ step:
/// $\mathrm{RANK\_IN} = 2 \cdot \mathrm{RANK\_OUT}$ and
/// $N_\text{OUT} = 2 \cdot N_\text{IN}$. Forced at the top of [`conv_step`] so a
/// mismatched instantiation fails to compile.
pub struct ConvDims<
    const RANK_IN: usize,
    const N_IN: usize,
    const RANK_OUT: usize,
    const N_OUT: usize,
>;

impl<const RANK_IN: usize, const N_IN: usize, const RANK_OUT: usize, const N_OUT: usize>
    ConvDims<RANK_IN, N_IN, RANK_OUT, N_OUT>
{
    /// Asserts the Conv₂ degree/rank halving/doubling relationship.
    pub const _CHECK: () = {
        assert!(
            RANK_IN == 2 * RANK_OUT,
            "conv_step: RANK_IN must equal 2 · RANK_OUT",
        );
        assert!(N_OUT == 2 * N_IN, "conv_step: N_OUT must equal 2 · N_IN");
    };
}

/// §5.2 — single Conv₂ step: $(2m, n)$-MLWE $\to$ $(m, 2n)$-MLWE.
///
/// Embeds the body and each mask via $\iota_0$ ([`RingPoly::embed_at`] at slot
/// 0) into `R_OUT` (degree $2n$), key-switches each embedded mask from its
/// (embedded) source key component to the target component, and accumulates the
/// result of input mask $\mathrm{RANK\_OUT} \cdot \mathrm{group} + j$ into
/// output slot $j$. `paper:mlwe.py:195-268`.
///
/// # Parallelism (GPU)
///
/// The `RANK_IN` per-mask embed + key-switches are **independent** (the map);
/// only the slot/body accumulation is a reduction. A device backend
/// parallelises the map over `key_idx ∈ [RANK_IN]`.
///
/// # Constant-time: No
///
/// Inputs are RLWE-uniform; the gadget products inside `key_switch` are
/// data-independent (§0.6).
///
/// # Panics
///
/// At compile time if [`ConvDims::_CHECK`] fails ($\mathrm{RANK\_IN} \ne 2
/// \cdot \mathrm{RANK\_OUT}$ or $N_\text{OUT} \ne 2 \cdot N_\text{IN}$).
//
// `needless_range_loop`: `j` indexes the output slot `result_masks[j]` while
// the flat index `idx = RANK_OUT*group + j` selects the input mask / step key —
// the two differ, so a range loop is the clearest form.
#[allow(non_camel_case_types, clippy::needless_range_loop)]
pub fn conv_step<
    const RANK_IN: usize,
    const N_IN: usize,
    const RANK_OUT: usize,
    const N_OUT: usize,
    R_IN: RingPoly<N_IN, Embedded<N_OUT> = R_OUT, Modulus = <R_OUT as RingPoly<N_OUT>>::Modulus>,
    R_OUT: RingPoly<N_OUT> + RingPolyEval<N_OUT>,
    const L: usize,
>(
    ct: &MLWECiphertext<RANK_IN, N_IN, R_IN>,
    step_keys: &[RLevCiphertext<N_OUT, R_OUT, L>; RANK_IN],
    base: u64,
) -> MLWECiphertext<RANK_OUT, N_OUT, R_OUT> {
    let () = ConvDims::<RANK_IN, N_IN, RANK_OUT, N_OUT>::_CHECK;
    let modulus = RingPoly::modulus(&ct.body);
    let mut result_masks: [R_OUT; RANK_OUT] = core::array::from_fn(|_| R_OUT::zero(modulus));
    let mut result_body = ct.body.embed_at::<N_OUT>(0);
    for group in 0..2 {
        for j in 0..RANK_OUT {
            let idx = RANK_OUT * group + j;
            // MAP: embed the mask and key-switch it (independent per idx).
            let embedded = ct.masks[idx].embed_at::<N_OUT>(0);
            let rlwe = RLWECiphertext::new(embedded, R_OUT::zero(modulus));
            let switched = step_keys[idx].key_switch(&rlwe, base);
            // REDUCE: accumulate into output slot j and the body.
            result_masks[j] += switched.mask;
            result_body += switched.body;
        }
    }
    MLWECiphertext::new(result_masks, result_body)
}

/// §5.4 (per-step) — generate the `RANK_IN` RLev key-switching keys for one
/// Conv₂ step, from the single degree-`NLWE` secret key `sk`.
///
/// For key index $\mathrm{group} \cdot m + j$ (with $m = \mathrm{RANK\_IN}/2$):
/// the **source** component is $\iota_0\bigl(\pi_{m \cdot \mathrm{group} +
/// j}^{\,\mathrm{NLWE} \to N_\text{IN}}(S)\bigr)$ and the **target** key is
/// $\pi_j^{\,\mathrm{NLWE} \to N_\text{OUT}}(S)$; the key is
/// $\mathrm{RLev}_{\text{target}}(\text{source})$. `paper:mlwe.py:319-342`.
///
/// # PRG consumption order
///
/// `key_idx`-outer (group-outer, then `j`), each [`SecretKey::encrypt_rlev`]
/// drawing `[mask, error]` per gadget level. This exact order is the
/// cross-language parity contract (Part-5 KAT).
#[allow(non_camel_case_types)]
pub fn gen_conv_step_key<
    const NLWE: usize,
    const N_IN: usize,
    const N_OUT: usize,
    const RANK_IN: usize,
    R: RingPoly<NLWE>,
    const L: usize,
>(
    sk: &SecretKey<NLWE, R>,
    base: u64,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
) -> [RLevCiphertext<N_OUT, R::Projected<N_OUT>, L>; RANK_IN]
where
    R::Projected<N_IN>: RingPoly<N_IN, Embedded<N_OUT> = R::Projected<N_OUT>>,
    R::Projected<N_OUT>: RingPoly<N_OUT>,
{
    const {
        assert!(
            N_OUT == 2 * N_IN,
            "gen_conv_step_key: N_OUT must equal 2 · N_IN"
        );
        assert!(
            RANK_IN.is_multiple_of(2),
            "gen_conv_step_key: RANK_IN must be even"
        );
        assert!(
            NLWE == RANK_IN * N_IN,
            "gen_conv_step_key: NLWE must equal RANK_IN · N_IN",
        );
    }
    core::array::from_fn(|key_idx| {
        gen_conv_step_key_element::<NLWE, N_IN, N_OUT, RANK_IN, R, L>(
            sk, key_idx, base, error_dist, prg,
        )
    })
}

/// §5.4 (per-step, per-key) — generate **one** RLev step key (index `key_idx` in
/// `0..RANK_IN`) for a Conv₂ step. This is the per-element body of
/// [`gen_conv_step_key`]; calling it for `key_idx = 0, 1, …, RANK_IN-1` in order
/// reproduces that function's PRG draw order exactly.
///
/// Exposed so a heap builder can write each step key **directly into its
/// destination slot** (one `RLev` at a time) instead of materialising the whole
/// `[RLev; RANK_IN]` array on the stack — the difference between a peak stack of
/// one `RLev` and one of the entire (~24.75 MB at n=2048) cascade key.
///
/// # PRG consumption order
///
/// One [`SecretKey::encrypt_rlev`] (drawing `[mask, error]` per gadget level).
#[allow(non_camel_case_types)]
pub fn gen_conv_step_key_element<
    const NLWE: usize,
    const N_IN: usize,
    const N_OUT: usize,
    const RANK_IN: usize,
    R: RingPoly<NLWE>,
    const L: usize,
>(
    sk: &SecretKey<NLWE, R>,
    key_idx: usize,
    base: u64,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
) -> RLevCiphertext<N_OUT, R::Projected<N_OUT>, L>
where
    R::Projected<N_IN>: RingPoly<N_IN, Embedded<N_OUT> = R::Projected<N_OUT>>,
    R::Projected<N_OUT>: RingPoly<N_OUT>,
{
    let m = RANK_IN / 2;
    let group = key_idx / m;
    let j = key_idx % m;
    let source = sk
        .poly()
        .project_at::<N_IN>(m * group + j)
        .embed_at::<N_OUT>(0);
    let target = SecretKey::from_poly(sk.poly().project_at::<N_OUT>(j));
    target.encrypt_rlev::<L>(&source, base, error_dist, prg)
}

/// [`gen_conv_step_key_element`] writing the step key **directly into `dst`**
/// instead of returning it by value. Lets the cascade boxed builder avoid
/// assembling the whole step-key RLev (~1.125 MiB at the high-degree n=2048
/// steps) on the stack — each of its `L` RLWE samples is written straight into
/// the heap slot. PRG draws are identical to [`gen_conv_step_key_element`].
///
/// # Safety
///
/// `dst` must point to memory valid for one
/// `RLevCiphertext<N_OUT, R::Projected<N_OUT>, L>`.
#[allow(non_camel_case_types)]
pub unsafe fn gen_conv_step_key_element_into<
    const NLWE: usize,
    const N_IN: usize,
    const N_OUT: usize,
    const RANK_IN: usize,
    R: RingPoly<NLWE>,
    const L: usize,
>(
    sk: &SecretKey<NLWE, R>,
    key_idx: usize,
    base: u64,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
    dst: *mut RLevCiphertext<N_OUT, R::Projected<N_OUT>, L>,
) where
    R::Projected<N_IN>: RingPoly<N_IN, Embedded<N_OUT> = R::Projected<N_OUT>>,
    R::Projected<N_OUT>: RingPoly<N_OUT>,
{
    let m = RANK_IN / 2;
    let group = key_idx / m;
    let j = key_idx % m;
    let source = sk
        .poly()
        .project_at::<N_IN>(m * group + j)
        .embed_at::<N_OUT>(0);
    let target = SecretKey::from_poly(sk.poly().project_at::<N_OUT>(j));
    // SAFETY: `dst` is valid for one RLev (caller contract); `encrypt_rlev_into`
    // initialises every sample, drawing PRG identically to `encrypt_rlev`.
    unsafe { target.encrypt_rlev_into::<L>(dst, &source, base, error_dist, prg) };
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::rns::basis::ConstRnsBasis;
    use crate::algebra::zq::modulus::ConstModulus;
    use crate::conversion::mlwe_ops::encrypt_lwe;
    use crate::encryption::decode;

    const BASE: u64 = 8;
    const L: usize = 6;

    /// Decrypt an $(m, n)$-MLWE under an explicit key vector: $\mathrm{decode}
    /// \bigl(B - \sum_k A_k \cdot S_k\bigr)$.
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
        decode::<N, R, RP>(&acc, p_mod)
    }

    #[test]
    fn conv_step_halves_rank_doubles_degree() {
        type R8 = Poly<8, ConstModulus<65537>, Coefficient>;
        let q = ConstModulus::<65537>;
        let mut prg = Shake256Prg::new(b"conv-shape");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let lwe = encrypt_lwe(&sk, 3, 16, Distribution::Ternary, &mut prg);
        let keys =
            gen_conv_step_key::<8, 1, 2, 8, R8, L>(&sk, BASE, Distribution::Ternary, &mut prg);
        let out: MLWECiphertext<4, 2, Poly<2, ConstModulus<65537>, Coefficient>> =
            conv_step::<8, 1, 4, 2, _, _, L>(&lwe, &keys, BASE);
        assert_eq!(out.masks.len(), 4); // rank halved 8 -> 4
        let _: Poly<2, ConstModulus<65537>, Coefficient> = out.body; // degree doubled 1 -> 2
    }

    #[test]
    fn conv_step_decryption_correctness() {
        type R8 = Poly<8, ConstModulus<65537>, Coefficient>;
        type P2 = Poly<2, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        for message in 0..16u64 {
            let mut prg = Shake256Prg::new(b"conv-decrypt");
            let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
            let lwe = encrypt_lwe(&sk, message, 16, Distribution::Ternary, &mut prg);
            let keys =
                gen_conv_step_key::<8, 1, 2, 8, R8, L>(&sk, BASE, Distribution::Ternary, &mut prg);
            let out = conv_step::<8, 1, 4, 2, _, _, L>(&lwe, &keys, BASE);
            // Target key vector: π_j^{8→2}(S) for j = 0..3.
            let key_vec: [Poly<2, ConstModulus<65537>, Coefficient>; 4] =
                core::array::from_fn(|j| sk.poly().project_at::<2>(j));
            let recovered: P2 = mlwe_decrypt(&out, &key_vec, p);
            // The LWE encrypts the scalar at coefficient 0; after ι₀ it is at
            // coefficient 0 of the degree-2 message, the rest zero.
            assert_eq!(recovered.coeff(0).to_u64(), message, "message {message}");
            assert_eq!(recovered.coeff(1).to_u64(), 0, "message {message}");
        }
    }

    /// Paper-class: a single Conv₂ step on the RNS backend, exercising the
    /// `N = 1 → 2` embed at a composite modulus. `Q = 7681·12289 ≈ 2^26.5`.
    #[test]
    fn conv_step_rns_n1_to_n2_decryption() {
        type Rns8 = PolyRns<8, ConstRnsBasis<7681, 12289>, Coefficient>;
        type P2 = Poly<2, ConstModulus<16>, Coefficient>;
        let basis = ConstRnsBasis::<7681, 12289>;
        let p = ConstModulus::<16>;
        for message in [0u64, 1, 7, 15] {
            let mut prg = Shake256Prg::new(b"conv-rns");
            let sk = SecretKey::<8, Rns8>::keygen(basis, Distribution::Ternary, &mut prg);
            let lwe = encrypt_lwe(&sk, message, 16, Distribution::Ternary, &mut prg);
            let keys = gen_conv_step_key::<8, 1, 2, 8, Rns8, L>(
                &sk,
                BASE,
                Distribution::Ternary,
                &mut prg,
            );
            let out = conv_step::<8, 1, 4, 2, _, _, L>(&lwe, &keys, BASE);
            let key_vec: [PolyRns<2, ConstRnsBasis<7681, 12289>, Coefficient>; 4] =
                core::array::from_fn(|j| sk.poly().project_at::<2>(j));
            let recovered: P2 = mlwe_decrypt(&out, &key_vec, p);
            assert_eq!(
                recovered.coeff(0).to_u64(),
                message,
                "rns message {message}"
            );
        }
    }
}
