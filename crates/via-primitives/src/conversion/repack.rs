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

use alloc::vec::Vec;

use crate::algebra::ring::RingPoly;
use crate::algebra::ring::element::Poly;
use crate::encryption::MLWECiphertext;
use crate::encryption::types::{RLWECiphertext, RLevCiphertext};

// Cascade key structs whose `keys_*` suffix the repack views borrow (zero-copy).
use super::{LweToRlweKeyN8, LweToRlweKeyN64};

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

/// One recursion level of `mlwes_to_rlwe`: pair up `inputs` (length a power of
/// two) and convert each pair via [`mlwes_to_mlwe`] at this level's shape,
/// halving the count and doubling the degree. The level helper the
/// [`repack_cascade!`] macro threads through.
#[allow(non_camel_case_types)]
pub(crate) fn pair_convert<
    const RANK_IN: usize,
    const N_IN: usize,
    const RANK_OUT: usize,
    const N_OUT: usize,
    R_IN: RingPoly<N_IN, Embedded<N_OUT> = R_OUT, Modulus = <R_OUT as RingPoly<N_OUT>>::Modulus>,
    R_OUT: RingPoly<N_OUT>,
    const L: usize,
>(
    inputs: Vec<MLWECiphertext<RANK_IN, N_IN, R_IN>>,
    step_keys: &[RLevCiphertext<N_OUT, R_OUT, L>; RANK_IN],
    base: u64,
) -> Vec<MLWECiphertext<RANK_OUT, N_OUT, R_OUT>> {
    inputs
        .chunks(2)
        .map(|pair| {
            mlwes_to_mlwe::<RANK_IN, N_IN, RANK_OUT, N_OUT, R_IN, R_OUT, L>(pair, step_keys, base)
        })
        .collect()
}

/// Bit-reverse the low `bits` bits of `i`.
///
/// The adjacent-pairing binary-tree recursion in [`mlwes_to_rlwe`] (built by the
/// [`repack_cascade!`] macro) interleaves input `i` into output slot
/// `bit_reverse(i)`; reordering the `T` inputs by this map first makes the
/// output emerge in the paper's natural order `ι^{k/d→k}(M_0,…,M_{T-1})`
/// (`M_t` at slot `t`). For `T = 2` it is the identity.
pub(crate) fn bit_reverse_index(i: usize, bits: u32) -> usize {
    let mut r = 0usize;
    let mut x = i;
    for _ in 0..bits {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    r
}

/// Generate a complete repack preset — the schedule trait, the owned
/// dedicated-key oracle struct, the trait impl, the oracle generator, and the
/// `Repack` function — for a fixed `(n1, K, T)`, mirroring the
/// [`lwe_to_rlwe_cascade!`](crate::lwe_to_rlwe_cascade) pattern.
///
/// `steps` is the contiguous SUFFIX of the cascade's step list the repack
/// reuses, each `(field, RANK_IN, N_IN, RANK_OUT, N_OUT)`; the generated key
/// fields are degree-`N_OUT` RLev arrays whose types match a suffix of the
/// matching `LweToRlweKey*` — so Part 2 borrows them zero-copy. `extr_degree`
/// is `G = K/T` (the `Extr` output degree) and `input_count` is `D = T·n1/K`
/// (the padded MLWE count = `2^depth`).
macro_rules! repack_cascade {
    (
        n1 = $n1:literal,
        t = $t:literal,
        ring = $ring:ident,
        mod_param = $modp:ident,
        mod_bound = $modbound:path,
        levels = $lev:ident,
        extr_degree = $g:literal,
        input_count = $d:literal,
        schedule = $sched:ident,
        key = $key:ident,
        cascade_key = $ckey:ident,
        view = $view:ident,
        from_cascade = $from:ident,
        gen = $gen:ident,
        repack = $repack:ident,
        steps = [ $( ($field:ident, $rin:literal, $nin:literal, $rout:literal, $nout:literal) ),+ $(,)? ],
    ) => {
        #[doc = concat!(
            "Per-level step-key schedule for the `n1=", stringify!($n1), ", T=",
            stringify!($t), "` repack. Both the owned oracle [`", stringify!($key),
            "`] and (Part 2) the borrowing cascade-suffix view implement it, so [`",
            stringify!($repack), "`] is identical for generated and borrowed keys (§3.5)."
        )]
        pub trait $sched<$modp: $modbound, const $lev: usize> {
            $(
                #[doc = concat!("Step keys at degree ", stringify!($nout), ".")]
                fn $field(&self) -> &[$crate::encryption::types::RLevCiphertext<
                    $nout, $ring<$nout, $modp, $crate::algebra::ring::form::Coefficient>, $lev>; $rin];
            )+
        }

        #[doc = concat!(
            "Owned dedicated-key oracle schedule for the `n1=", stringify!($n1),
            ", T=", stringify!($t), "` repack — named-field key struct mirroring a ",
            "suffix of the matching cascade key."
        )]
        pub struct $key<$modp: $modbound, const $lev: usize> {
            $(
                #[doc = concat!("Step keys at degree ", stringify!($nout), ".")]
                pub $field: [$crate::encryption::types::RLevCiphertext<
                    $nout, $ring<$nout, $modp, $crate::algebra::ring::form::Coefficient>, $lev>; $rin],
            )+
        }

        impl<$modp: $modbound, const $lev: usize> $sched<$modp, $lev> for $key<$modp, $lev> {
            $(
                fn $field(&self) -> &[$crate::encryption::types::RLevCiphertext<
                    $nout, $ring<$nout, $modp, $crate::algebra::ring::form::Coefficient>, $lev>; $rin] {
                    &self.$field
                }
            )+
        }

        #[doc = concat!(
            "Generate the `n1=", stringify!($n1), ", T=", stringify!($t),
            "` dedicated-key oracle from the degree-", stringify!($n1),
            " secret key. Each field is the cascade's own `gen_conv_step_key` for ",
            "the matching step, so the borrowed cascade suffix (Part 2) is byte-identical."
        )]
        pub fn $gen<$modp: $modbound, const $lev: usize>(
            sk: &$crate::encryption::types::SecretKey<
                $n1, $ring<$n1, $modp, $crate::algebra::ring::form::Coefficient>>,
            base: u64,
            error_dist: $crate::sampling::distribution::Distribution,
            prg: &mut $crate::sampling::prg::Shake256Prg,
        ) -> $key<$modp, $lev> {
            $key {
                $(
                    $field: $crate::conversion::gen_conv_step_key::<
                        $n1, $nin, $nout, $rin,
                        $ring<$n1, $modp, $crate::algebra::ring::form::Coefficient>, $lev,
                    >(sk, base, error_dist, prg),
                )+
            }
        }

        #[doc = concat!(
            "`Repack` at `n1=", stringify!($n1), ", K`, `T=", stringify!($t),
            "`: pack the designated coefficients of `T` RLWEs over `R_{", stringify!($n1),
            ",q}` into one RLWE over the same ring (paper §3.4) — `Extr` each, pad to ",
            stringify!($d), " with zeros, then ", stringify!($d),
            "-leaf binary-tree `mlwes_to_mlwe` over the schedule, and unwrap."
        )]
        pub fn $repack<$modp: $modbound, const $lev: usize, S: $sched<$modp, $lev>>(
            inputs: &[$crate::encryption::types::RLWECiphertext<
                $n1, $ring<$n1, $modp, $crate::algebra::ring::form::Coefficient>>; $t],
            keys: &S,
            base: u64,
        ) -> $crate::encryption::types::RLWECiphertext<
            $n1, $ring<$n1, $modp, $crate::algebra::ring::form::Coefficient>> {
            // Extr_{K/T=G} each input → (D, G)-MLWE; pad to D with zeros.
            let mut v = Vec::with_capacity($d);
            for i in 0..$t {
                v.push($crate::conversion::extr::<
                    $n1, $g, $d, $ring<$n1, $modp, $crate::algebra::ring::form::Coefficient>,
                >(&inputs[i]));
            }
            // Reorder the T reals by bit-reversal so the binary-tree recursion's
            // interleave emerges in the paper's natural slot order (M_t @ slot t).
            let lt = ($t as usize).trailing_zeros();
            for i in 0..$t {
                let j = $crate::conversion::repack::bit_reverse_index(i, lt);
                if i < j {
                    v.swap(i, j);
                }
            }
            let qm = $crate::algebra::ring::RingPoly::modulus(&inputs[0].body);
            let zg = <$ring<$g, $modp, $crate::algebra::ring::form::Coefficient>
                as $crate::algebra::ring::RingPoly<$g>>::zero(qm);
            let zero = $crate::encryption::MLWECiphertext::<
                $d, $g, $ring<$g, $modp, $crate::algebra::ring::form::Coefficient>>::new([zg; $d], zg);
            for _ in 0..($d - $t) {
                v.push(zero);
            }
            // One shadowing `pair_convert` per recursion level (degrees double, count halves).
            $(
                let v = $crate::conversion::repack::pair_convert::<$rin, $nin, $rout, $nout, _, _, $lev>(
                    v, $sched::$field(keys), base,
                );
            )+
            $crate::conversion::mlwe_to_rlwe(&v.into_iter().next().unwrap())
        }

        #[doc = concat!(
            "Borrowing schedule for the `n1=", stringify!($n1), ", T=", stringify!($t),
            "` repack, backed by a SUFFIX of an existing [`", stringify!($ckey),
            "`] cascade key — zero-copy (the §3.5 key reuse: no new offline payload). ",
            "Built by [`", stringify!($from), "`]."
        )]
        pub struct $view<'a, $modp: $modbound, const $lev: usize> {
            cascade: &'a $ckey<$modp, $lev>,
        }

        impl<'a, $modp: $modbound, const $lev: usize> $sched<$modp, $lev>
            for $view<'a, $modp, $lev>
        {
            $(
                fn $field(&self) -> &[$crate::encryption::types::RLevCiphertext<
                    $nout, $ring<$nout, $modp, $crate::algebra::ring::form::Coefficient>, $lev>; $rin] {
                    // Zero-copy borrow: the step list matches the cascade suffix,
                    // so this field has exactly the schedule's expected type.
                    &self.cascade.$field
                }
            )+
        }

        #[doc = concat!(
            "Borrow the `keys_*` suffix of a [`", stringify!($ckey),
            "`] cascade key as a [`", stringify!($view), "`] repack schedule (",
            stringify!($repack), "'s `pp_qck`-reuse entry point)."
        )]
        pub fn $from<$modp: $modbound, const $lev: usize>(
            cascade: &$ckey<$modp, $lev>,
        ) -> $view<'_, $modp, $lev> {
            $view { cascade }
        }
    };
}

// Toy preset (n1=8, K=4, T=2; depth 2): the reference instantiation, validated
// by `repack_n8_t2_reconstructs`. Its key fields (`keys_4`, `keys_8`) are the
// same types as a suffix of `LweToRlweKeyN8`'s fields — Part 2 borrows them.
repack_cascade! {
    n1 = 8,
    t = 2,
    ring = Poly,
    mod_param = M,
    mod_bound = crate::algebra::zq::modulus::Modulus,
    levels = L,
    extr_degree = 2, // G = K/T = 4/2
    input_count = 4, // D = T·n1/K = 2·8/4
    schedule = RepackScheduleN8T2,
    key = RepackKeysN8T2,
    cascade_key = LweToRlweKeyN8,
    view = RepackViewN8T2,
    from_cascade = repack_keys_n8_t2_from_cascade,
    gen = gen_repack_keys_n8_t2,
    repack = repack_n8_t2,
    steps = [
        (keys_4, 4, 2, 2, 4),
        (keys_8, 2, 4, 1, 8),
    ],
}

// e2e-toy preset (n1=64, K=16, T=8; depth 5): the `ViaBToyParams<64,16,2,8>`
// repack. G = K/T = 2, D = T·n1/K = 32; reuses the `LweToRlweKeyN64` suffix
// `keys_4..keys_64`. Validated by `repack_n64_t8_reconstructs`.
repack_cascade! {
    n1 = 64,
    t = 8,
    ring = Poly,
    mod_param = M,
    mod_bound = crate::algebra::zq::modulus::Modulus,
    levels = L,
    extr_degree = 2,  // G = K/T = 16/8
    input_count = 32, // D = T·n1/K = 8·64/16
    schedule = RepackScheduleN64T8,
    key = RepackKeysN64T8,
    cascade_key = LweToRlweKeyN64,
    view = RepackViewN64T8,
    from_cascade = repack_keys_n64_t8_from_cascade,
    gen = gen_repack_keys_n64_t8,
    repack = repack_n64_t8,
    steps = [
        (keys_4, 32, 2, 16, 4),
        (keys_8, 16, 4, 8, 8),
        (keys_16, 8, 8, 4, 16),
        (keys_32, 4, 16, 2, 32),
        (keys_64, 2, 32, 1, 64),
    ],
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::zq::modulus::ConstModulus;
    use crate::conversion::{extr, gen_conv_step_key, gen_lwe_to_rlwe_key_n8};
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

    /// Full toy repack round-trip: `Repack_4({c0,c1})` decrypts under `S` to the
    /// paper interleave `ι_0^{4→8} ι^{2→4}(π_0(M0), π_0(M1))` — coefficients
    /// 0,2,4,6 carry `M0_0, M1_0, M0_4, M1_4`, odd coefficients zero. Validates
    /// the full Extr→pad→recurse→unwrap path + the dedicated-key oracle (the
    /// §3.4 / #1-risk reconstructability guard).
    #[test]
    fn repack_n8_t2_reconstructs() {
        type P8p = Poly<8, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        const L: usize = 8;
        const BASE: u64 = 8;
        let mut prg = Shake256Prg::new(b"repack-n8-t2-reconstruct");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let m0 = encode::<8, R8, P8p>(&P8p::new(p, [2, 0, 0, 0, 5, 0, 0, 0]), q);
        let m1 = encode::<8, R8, P8p>(&P8p::new(p, [3, 0, 0, 0, 7, 0, 0, 0]), q);
        let c0 = sk.encrypt(&m0, Distribution::Ternary, &mut prg);
        let c1 = sk.encrypt(&m1, Distribution::Ternary, &mut prg);
        let keys = gen_repack_keys_n8_t2::<ConstModulus<65537>, L>(
            &sk,
            BASE,
            Distribution::Ternary,
            &mut prg,
        );
        let out = repack_n8_t2(&[c0, c1], &keys, BASE);
        // Decrypt under S.
        let mut acc = out.body;
        acc -= out.mask * *sk.poly();
        let recovered: P8p = decode::<8, R8, P8p>(&acc, p);
        assert_eq!(recovered.coeff(0).to_u64(), 2, "M0_0 @ slot 0");
        assert_eq!(recovered.coeff(2).to_u64(), 3, "M1_0 @ slot 2");
        assert_eq!(recovered.coeff(4).to_u64(), 5, "M0_4 @ slot 4");
        assert_eq!(recovered.coeff(6).to_u64(), 7, "M1_4 @ slot 6");
        assert_eq!(recovered.coeff(1).to_u64(), 0, "odd slot 1 zero");
        assert_eq!(recovered.coeff(3).to_u64(), 0, "odd slot 3 zero");
    }

    /// e2e-toy depth-5 repack `(n1,K,T)=(64,16,8)`: pack `T=8` RLWEs whose only
    /// live coefficient is `M_t[0] = t+1`. The output interleave
    /// `ι_0^{16→64} ι^{2→16}(π_0^{64→2}(M_t))` places `M_t[0]` at coefficient
    /// `4t`. Validates the macro at depth 5 (5 levels of `mlwes_to_mlwe`).
    #[test]
    fn repack_n64_t8_reconstructs() {
        type R64 = Poly<64, ConstModulus<65537>, Coefficient>;
        type P64p = Poly<64, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        const L: usize = 8;
        const BASE: u64 = 8;
        let mut prg = Shake256Prg::new(b"repack-n64-t8-reconstruct");
        let sk = SecretKey::<64, R64>::keygen(q, Distribution::Ternary, &mut prg);
        // 8 messages: M_t has coeff 0 = t+1, all else 0.
        let cs: [_; 8] = core::array::from_fn(|t| {
            let mut coeffs = [0u64; 64];
            coeffs[0] = (t + 1) as u64;
            let m = encode::<64, R64, P64p>(&P64p::new(p, coeffs), q);
            sk.encrypt(&m, Distribution::Ternary, &mut prg)
        });
        let keys = gen_repack_keys_n64_t8::<ConstModulus<65537>, L>(
            &sk,
            BASE,
            Distribution::Ternary,
            &mut prg,
        );
        let out = repack_n64_t8(&cs, &keys, BASE);
        let mut acc = out.body;
        acc -= out.mask * *sk.poly();
        let recovered: P64p = decode::<64, R64, P64p>(&acc, p);
        for t in 0..8 {
            assert_eq!(
                recovered.coeff(4 * t).to_u64(),
                (t + 1) as u64,
                "M{t}[0] @ slot {}",
                4 * t
            );
        }
        // Non-designated coefficients are zero (spot check).
        assert_eq!(recovered.coeff(1).to_u64(), 0);
        assert_eq!(recovered.coeff(2).to_u64(), 0);
        assert_eq!(recovered.coeff(32).to_u64(), 0); // M_t[32] = 0
    }

    /// §3.5 key reuse: repack with keys BORROWED from a full cascade key
    /// (`gen_lwe_to_rlwe_key_n8`) — exactly the `pp_qck` the server already holds
    /// — reconstructs the paper interleave. Proves the query-compression
    /// cascade's `keys_4`/`keys_8` ARE valid repack keys (no new offline payload).
    #[test]
    fn repack_n8_t2_via_cascade_keys_reconstructs() {
        type P8p = Poly<8, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        const L: usize = 8;
        const BASE: u64 = 8;
        let mut prg = Shake256Prg::new(b"repack-n8-t2-via-cascade");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let m0 = encode::<8, R8, P8p>(&P8p::new(p, [2, 0, 0, 0, 5, 0, 0, 0]), q);
        let m1 = encode::<8, R8, P8p>(&P8p::new(p, [3, 0, 0, 0, 7, 0, 0, 0]), q);
        let c0 = sk.encrypt(&m0, Distribution::Ternary, &mut prg);
        let c1 = sk.encrypt(&m1, Distribution::Ternary, &mut prg);
        // The FULL cascade key (keys_2, keys_4, keys_8), as shipped in pp_qck.
        let cascade = gen_lwe_to_rlwe_key_n8::<ConstModulus<65537>, L>(
            &sk,
            BASE,
            Distribution::Ternary,
            &mut prg,
        );
        let view = repack_keys_n8_t2_from_cascade(&cascade); // borrow keys_4, keys_8
        let inputs = [c0, c1];
        let out = repack_n8_t2(&inputs, &view, BASE);
        let mut acc = out.body;
        acc -= out.mask * *sk.poly();
        let recovered: P8p = decode::<8, R8, P8p>(&acc, p);
        assert_eq!(recovered.coeff(0).to_u64(), 2);
        assert_eq!(recovered.coeff(2).to_u64(), 3);
        assert_eq!(recovered.coeff(4).to_u64(), 5);
        assert_eq!(recovered.coeff(6).to_u64(), 7);
    }

    /// #1-risk byte-equality guard: repack with the BORROWED cascade suffix is
    /// coefficient-for-coefficient identical to repack with a dedicated oracle
    /// whose PRG is aligned to the cascade's (the `keys_2` step consumed first).
    /// Proves the borrowed `keys_4`/`keys_8` ARE the oracle's `keys_4`/`keys_8`.
    #[test]
    fn repack_n8_t2_borrowed_equals_oracle() {
        type P8p = Poly<8, ConstModulus<16>, Coefficient>;
        let q = ConstModulus::<65537>;
        let p = ConstModulus::<16>;
        const L: usize = 8;
        const BASE: u64 = 8;
        let mut prg_kg = Shake256Prg::new(b"repack-n8-t2-equality-kg");
        let sk = SecretKey::<8, R8>::keygen(q, Distribution::Ternary, &mut prg_kg);
        let m0 = encode::<8, R8, P8p>(&P8p::new(p, [2, 0, 0, 0, 5, 0, 0, 0]), q);
        let m1 = encode::<8, R8, P8p>(&P8p::new(p, [3, 0, 0, 0, 7, 0, 0, 0]), q);
        let c0 = sk.encrypt(&m0, Distribution::Ternary, &mut prg_kg);
        let c1 = sk.encrypt(&m1, Distribution::Ternary, &mut prg_kg);
        let inputs = [c0, c1];
        // Cascade key from a fixed key-seed.
        let mut prg_c = Shake256Prg::new(b"repack-equality-keyseed");
        let cascade = gen_lwe_to_rlwe_key_n8::<ConstModulus<65537>, L>(
            &sk,
            BASE,
            Distribution::Ternary,
            &mut prg_c,
        );
        let view = repack_keys_n8_t2_from_cascade(&cascade);
        // PRG-aligned oracle: same key-seed; consume the `keys_2`-shaped step
        // first (as the cascade does), then generate keys_4, keys_8.
        let mut prg_o = Shake256Prg::new(b"repack-equality-keyseed");
        let _keys_2 =
            gen_conv_step_key::<8, 1, 2, 8, R8, L>(&sk, BASE, Distribution::Ternary, &mut prg_o);
        let oracle = gen_repack_keys_n8_t2::<ConstModulus<65537>, L>(
            &sk,
            BASE,
            Distribution::Ternary,
            &mut prg_o,
        );
        let out_view = repack_n8_t2(&inputs, &view, BASE);
        let out_oracle = repack_n8_t2(&inputs, &oracle, BASE);
        assert_eq!(out_view.mask, out_oracle.mask, "mask coeff-for-coeff");
        assert_eq!(out_view.body, out_oracle.body, "body coeff-for-coeff");
    }
}
