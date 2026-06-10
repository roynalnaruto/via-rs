//! Layer 7 — VIA-B homomorphic repacking primitives.
//!
//! `embed_d` (multi-slot MLWEs embedding, §3.1), `mlwes_insert` (multi-input
//! MLWE→RLWE pack), `mlwes_to_mlwe` (§3.2 pair conversion), and — landing in the
//! rest of Part 1/2 — `mlwes_to_rlwe`, `repack_k`, the dedicated-key oracle, and
//! the cascade-key-suffix borrow.
//!
//! Gated `#[cfg(all(feature = "via-b", feature = "alloc"))]` at the
//! [`crate::conversion`] re-export boundary: the repack recursion is
//! runtime-depth (`log2(T·n1/n2)`).
//!
//! ## Key reuse (§3.5)
//!
//! [`mlwes_to_mlwe`] is the single recursion step and REUSES [`super::conv_step`]'s
//! key-switch + slot-accumulation reduction verbatim, substituting the §3.1
//! multi-input [`embed_d`] (`d = 2`, slots 0,1) for `conv_step`'s internal slot-0
//! embed. Its step keys are exactly the cascade's own
//! [`super::gen_conv_step_key`] outputs — which is what lets VIA-B borrow the
//! query-compression cascade keys for repacking (no new offline payload).

use crate::algebra::ring::RingPoly;
use crate::encryption::MLWECiphertext;
use crate::encryption::types::{RLWECiphertext, RLevCiphertext};

/// §3.1 (VIA-B) — MLWEs embedding: `d` many `(RANK, N)`-MLWE → one
/// `(RANK, N_LARGE)`-MLWE (`N_LARGE = N·d`) encrypting
/// `ι^{N→N_LARGE}(M_0, …, M_{d-1})`.
///
/// Output mask coordinate `j` is `ι^{N→N_LARGE}(A^0_j, …, A^{d-1}_j)` and the
/// body is `ι^{N→N_LARGE}(B^0, …, B^{d-1})`, realised as the disjoint sum of the
/// per-slot embeds `Σ_s x_s.embed_at::<N_LARGE>(s)` (each `embed_at(s)` writes
/// coefficient `i` of `x_s` to position `d·i + s`; the slots are disjoint). The
/// `d = 1`, slot-0 case coincides with [`super::embed_mlwe`].
///
/// # Panics
///
/// Runtime: if `cts.len() != N_LARGE / N` (the slot count). Compile-time
/// (inside [`RingPoly::embed_at`]) if `N ∤ N_LARGE` or `N_LARGE < N`.
pub fn embed_d<const RANK: usize, const N: usize, const N_LARGE: usize, R, RL>(
    cts: &[MLWECiphertext<RANK, N, R>],
) -> MLWECiphertext<RANK, N_LARGE, RL>
where
    R: RingPoly<N, Embedded<N_LARGE> = RL>,
    RL: RingPoly<N_LARGE>,
{
    let d = N_LARGE / N;
    assert_eq!(
        cts.len(),
        d,
        "embed_d: input count {} must equal slot count N_LARGE/N = {d}",
        cts.len(),
    );
    // Accumulate `Σ_s x_s.embed_at(s)` seeded by slot 0 (avoids needing `zero`).
    let masks: [RL; RANK] = core::array::from_fn(|j| {
        let mut acc = cts[0].masks[j].embed_at::<N_LARGE>(0);
        for (s, ct) in cts.iter().enumerate().skip(1) {
            acc += ct.masks[j].embed_at::<N_LARGE>(s);
        }
        acc
    });
    let mut body = cts[0].body.embed_at::<N_LARGE>(0);
    for (s, ct) in cts.iter().enumerate().skip(1) {
        body += ct.body.embed_at::<N_LARGE>(s);
    }
    MLWECiphertext::new(masks, body)
}

/// Multi-input MLWE→RLWE pack: `d` rank-1 MLWE inputs of degree `N` → one RLWE
/// of degree `N_LARGE = N·d` interleaving them at slots `0..d-1`.
///
/// The leaf packer the repack recursion bottoms out on; `embed_d` specialised to
/// `RANK = 1` followed by the rank-1 → RLWE unwrap. (`super::mlwe_to_rlwe` is
/// rank-1 *and* same-degree only.)
pub fn mlwes_insert<const N: usize, const N_LARGE: usize, R, RL>(
    cts: &[MLWECiphertext<1, N, R>],
) -> RLWECiphertext<N_LARGE, RL>
where
    R: RingPoly<N, Embedded<N_LARGE> = RL>,
    RL: RingPoly<N_LARGE>,
{
    let embedded = embed_d::<1, N, N_LARGE, R, RL>(cts);
    RLWECiphertext::new(embedded.masks[0], embedded.body)
}

/// §3.2 (VIA-B) — MLWEs-to-MLWE conversion of a **pair**: 2 many
/// `(RANK_IN, N_IN)`-MLWE → one `(RANK_OUT, N_OUT)`-MLWE
/// (`RANK_OUT = RANK_IN/2`, `N_OUT = 2·N_IN`) encrypting `ι^{N_IN→N_OUT}(M)`,
/// under the key `π^{·}(S)` matching `step_keys`.
///
/// The single recursion step of `mlwes_to_rlwe`. REUSES [`super::conv_step`]'s
/// key-switch + slot-accumulation reduction, substituting the §3.1 [`embed_d`]
/// (`d = 2`) of the *pair* for `conv_step`'s internal slot-0 embed (per
/// `.docs/via-b.md` §3.2): (1) interleave the pair into one `(RANK_IN, N_OUT)`-MLWE
/// at slots 0,1; (2) for each `idx ∈ [RANK_IN]` (group-outer, matching
/// `conv_step`'s order) key-switch the already-embedded `masks[idx]` with
/// `step_keys[idx]` and accumulate the switched mask into output slot
/// `idx % RANK_OUT` and the switched body into the body (seeded by the embedded
/// body). `step_keys` is the same `&[RLev<N_OUT, R_OUT, L>; RANK_IN]` slice
/// [`super::conv_step`] consumes.
///
/// Noise: `θ' ≤ θ_c + 2·θ_ks` (Lemma 4.1, `d = 2`).
///
/// # Constant-time: No
///
/// Inputs are RLWE-uniform; the gadget products inside `key_switch` are
/// data-independent (§0.6), exactly as in [`super::conv_step`].
///
/// # Panics
///
/// Compile-time if `RANK_IN ≠ 2·RANK_OUT` or `N_OUT ≠ 2·N_IN`; runtime if
/// `pair.len() != 2`.
#[allow(non_camel_case_types, clippy::needless_range_loop)]
pub fn mlwes_to_mlwe<
    const RANK_IN: usize,
    const N_IN: usize,
    const RANK_OUT: usize,
    const N_OUT: usize,
    R_IN: RingPoly<N_IN, Embedded<N_OUT> = R_OUT, Modulus = <R_OUT as RingPoly<N_OUT>>::Modulus>,
    R_OUT: RingPoly<N_OUT>,
    const L: usize,
>(
    pair: &[MLWECiphertext<RANK_IN, N_IN, R_IN>],
    step_keys: &[RLevCiphertext<N_OUT, R_OUT, L>; RANK_IN],
    base: u64,
) -> MLWECiphertext<RANK_OUT, N_OUT, R_OUT> {
    const {
        assert!(
            RANK_IN == 2 * RANK_OUT,
            "mlwes_to_mlwe: RANK_IN must equal 2 · RANK_OUT",
        );
        assert!(
            N_OUT == 2 * N_IN,
            "mlwes_to_mlwe: N_OUT must equal 2 · N_IN"
        );
    }
    assert_eq!(pair.len(), 2, "mlwes_to_mlwe converts exactly a pair (d=2)");
    // §3.1 — interleave the pair into one (RANK_IN, N_OUT)-MLWE.
    let embedded = embed_d::<RANK_IN, N_IN, N_OUT, R_IN, R_OUT>(pair);
    let modulus = RingPoly::modulus(&embedded.body);
    // §3.2 — conv_step's key-switch + slot reduction (conv.rs:84-98), but the
    // masks are ALREADY embedded (no further slot-0 embed).
    let mut result_masks: [R_OUT; RANK_OUT] = core::array::from_fn(|_| R_OUT::zero(modulus));
    let mut result_body = embedded.body;
    for group in 0..2 {
        for j in 0..RANK_OUT {
            let idx = RANK_OUT * group + j;
            let rlwe = RLWECiphertext::new(embedded.masks[idx], R_OUT::zero(modulus));
            let switched = step_keys[idx].key_switch(&rlwe, base);
            result_masks[j] += switched.mask;
            result_body += switched.body;
        }
    }
    MLWECiphertext::new(result_masks, result_body)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::zq::modulus::ConstModulus;
    use crate::conversion::{extr, gen_conv_step_key};
    use crate::encryption::types::SecretKey;
    use crate::encryption::{decode, encode};
    use crate::sampling::distribution::Distribution;
    use crate::sampling::prg::Shake256Prg;

    type R2 = Poly<2, ConstModulus<65537>, Coefficient>;
    type R4 = Poly<4, ConstModulus<65537>, Coefficient>;
    type R8 = Poly<8, ConstModulus<65537>, Coefficient>;

    /// §3.1 — `embed_d` over `d=2` rank-1 inputs at `(N,N_LARGE)=(2,4)`
    /// interleaves the two bodies/masks at slots 0,1. Hand-checked:
    /// `b0=[10,20]`, `b1=[30,40]` ⇒ `ι^{2→4}(b0,b1)=[10,30,20,40]`.
    #[test]
    fn embed_d_interleaves_two_inputs_at_slots() {
        let q = ConstModulus::<65537>;
        let ct0 = MLWECiphertext::<1, 2, R2>::new(
            [R2::from_u128_coeffs(q, &[1, 2])],
            R2::from_u128_coeffs(q, &[10, 20]),
        );
        let ct1 = MLWECiphertext::<1, 2, R2>::new(
            [R2::from_u128_coeffs(q, &[3, 4])],
            R2::from_u128_coeffs(q, &[30, 40]),
        );
        let out: MLWECiphertext<1, 4, R4> = embed_d::<1, 2, 4, R2, R4>(&[ct0, ct1]);
        let mut body = [0u128; 4];
        out.body.to_u128_coeffs(&mut body);
        assert_eq!(body, [10, 30, 20, 40]); // ι^{2→4}(b0,b1)
        let mut mask = [0u128; 4];
        out.masks[0].to_u128_coeffs(&mut mask);
        assert_eq!(mask, [1, 3, 2, 4]); // ι^{2→4}(a0,a1)
    }

    /// `mlwes_insert` packs two rank-1 degree-2 MLWE into one degree-4 RLWE with
    /// the same slot-0,1 interleave permutation.
    #[test]
    fn mlwes_insert_packs_two_rank1_into_rlwe() {
        let q = ConstModulus::<65537>;
        let ct0 = MLWECiphertext::<1, 2, R2>::new(
            [R2::from_u128_coeffs(q, &[1, 2])],
            R2::from_u128_coeffs(q, &[10, 20]),
        );
        let ct1 = MLWECiphertext::<1, 2, R2>::new(
            [R2::from_u128_coeffs(q, &[3, 4])],
            R2::from_u128_coeffs(q, &[30, 40]),
        );
        let out: RLWECiphertext<4, R4> = mlwes_insert::<2, 4, R2, R4>(&[ct0, ct1]);
        let mut body = [0u128; 4];
        out.body.to_u128_coeffs(&mut body);
        assert_eq!(body, [10, 30, 20, 40]);
        let mut mask = [0u128; 4];
        out.mask.to_u128_coeffs(&mut mask);
        assert_eq!(mask, [1, 3, 2, 4]);
    }

    /// §3.2 — `mlwes_to_mlwe` on a PAIR of `(4,2)`-MLWE → one `(2,4)`-MLWE
    /// (toy level 0). The two inputs are `Extr_2` of two RLWEs of `M0,M1 ∈
    /// R_{8,p}`; the keys are the cascade's own `gen_conv_step_key::<8,2,4,4>`
    /// (NLWE=8=RANK_IN·N_IN=4·2 ✓), proving the repack reuses cascade keys. The
    /// output decrypts under `π^{8→4}(S)` to `ι^{2→4}(π_0^{8→2}(M0), π_0^{8→2}(M1))`.
    #[test]
    fn mlwes_to_mlwe_pair_4x2_to_2x4_decrypts() {
        type P8p = Poly<8, ConstModulus<16>, Coefficient>;
        type P4q = Poly<4, ConstModulus<65537>, Coefficient>;
        type P4p = Poly<4, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        const L: usize = 8;
        const BASE: u64 = 8;
        let mut prg = Shake256Prg::new(b"mlwes-to-mlwe-toy-l0");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let m0 = encode::<8, R8, P8p>(&P8p::new(p, [2, 0, 0, 0, 5, 0, 0, 0]), q);
        let m1 = encode::<8, R8, P8p>(&P8p::new(p, [3, 0, 0, 0, 7, 0, 0, 0]), q);
        let c0 = sk.encrypt(&m0, Distribution::Ternary, &mut prg);
        let c1 = sk.encrypt(&m1, Distribution::Ternary, &mut prg);
        let e0 = extr::<8, 2, 4, _>(&c0); // (4,2)-MLWE
        let e1 = extr::<8, 2, 4, _>(&c1);
        let keys =
            gen_conv_step_key::<8, 2, 4, 4, R8, L>(&sk, BASE, Distribution::Ternary, &mut prg);
        let out: MLWECiphertext<2, 4, P4q> =
            mlwes_to_mlwe::<4, 2, 2, 4, _, _, L>(&[e0, e1], &keys, BASE);
        // Decrypt under π^{8→4}(S) = (π_0(S), π_1(S)).
        let key_vec: [P4q; 2] = core::array::from_fn(|j| sk.poly().project_at::<4>(j));
        let mut acc = out.body;
        for (m, k) in out.masks.iter().zip(key_vec.iter()) {
            acc -= *m * *k;
        }
        let recovered: P4p = decode::<4, P4q, P4p>(&acc, p);
        // π_0^{8→2}(Mi) = (Mi_0, Mi_4) ⇒ ι^{2→4} ⇒ [M0_0, M1_0, M0_4, M1_4] = [2,3,5,7].
        assert_eq!(recovered.coeff(0).to_u64(), 2, "M0_0");
        assert_eq!(recovered.coeff(1).to_u64(), 3, "M1_0");
        assert_eq!(recovered.coeff(2).to_u64(), 5, "M0_4");
        assert_eq!(recovered.coeff(3).to_u64(), 7, "M1_4");
    }

    /// §3.2 — `mlwes_to_mlwe` at the toy level-1 shape `(2,4)→(1,8)`, keys via
    /// `gen_conv_step_key::<8,4,8,2>`; decrypts under the rank-1 `π^{8→8}(S)=S`.
    /// Guards the recursion's second level independently of the first.
    #[test]
    fn mlwes_to_mlwe_pair_2x4_to_1x8_decrypts() {
        type P8p = Poly<8, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        const L: usize = 8;
        const BASE: u64 = 8;
        let mut prg = Shake256Prg::new(b"mlwes-to-mlwe-toy-l1");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let m0 = encode::<8, R8, P8p>(&P8p::new(p, [2, 0, 0, 0, 5, 0, 0, 0]), q);
        let m1 = encode::<8, R8, P8p>(&P8p::new(p, [3, 0, 0, 0, 7, 0, 0, 0]), q);
        let c0 = sk.encrypt(&m0, Distribution::Ternary, &mut prg);
        let c1 = sk.encrypt(&m1, Distribution::Ternary, &mut prg);
        let e0 = extr::<8, 4, 2, _>(&c0); // (2,4)-MLWE
        let e1 = extr::<8, 4, 2, _>(&c1);
        let keys =
            gen_conv_step_key::<8, 4, 8, 2, R8, L>(&sk, BASE, Distribution::Ternary, &mut prg);
        let out: MLWECiphertext<1, 8, R8> =
            mlwes_to_mlwe::<2, 4, 1, 8, _, _, L>(&[e0, e1], &keys, BASE);
        // Decrypt under the rank-1 key S.
        let mut acc = out.body;
        acc -= out.masks[0] * *sk.poly();
        let recovered: P8p = decode::<8, R8, P8p>(&acc, p);
        // π_0^{8→4}(Mi) = (Mi_0, Mi_2, Mi_4, Mi_6); ι^{4→8} interleaves at slots 0,1.
        // M0=(2,0,5,0), M1=(3,0,7,0) ⇒ [2,3,0,0,5,7,0,0].
        assert_eq!(recovered.coeff(0).to_u64(), 2, "M0_0");
        assert_eq!(recovered.coeff(1).to_u64(), 3, "M1_0");
        assert_eq!(recovered.coeff(4).to_u64(), 5, "M0_4");
        assert_eq!(recovered.coeff(5).to_u64(), 7, "M1_4");
    }
}
