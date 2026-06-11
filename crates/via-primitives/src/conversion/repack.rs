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
use crate::algebra::ring::rns_element::PolyRns;
use crate::encryption::MLWECiphertext;
use crate::encryption::types::{RLWECiphertext, RLevCiphertext};
use crate::switching::mod_switch::mod_switch_sym;

// Cascade key structs whose `keys_*` suffix the repack views borrow (zero-copy).
use super::{LweToRlweKeyN8, LweToRlweKeyN64, LweToRlweKeyRnsN2048};

/// Mod-switch every gadget sample of an [`RLevCiphertext`] from its current
/// modulus to `dst_mod`, via per-sample [`mod_switch_sym`].
///
/// The same-modulus-type realization of the §3.5 key reuse across `q1 ≠ q2`: the
/// repack runs at `q2` (the post-CRot modulus, `.docs/via-b.md` §4) but the
/// cascade keys ship at `q1`, so the server mod-switches them internally — no new
/// offline payload. (Single-prime: `R_SRC`/`R_DST` are the same ring type at
/// different modulus *values*. The RNS-`q1` → single-prime-`q2` cross-type switch
/// is a separate path.)
#[allow(non_camel_case_types)] // R_SRC/R_DST mirror `mod_switch_sym`'s convention
pub(crate) fn mod_switch_rlev<const N: usize, R_SRC, R_DST, const L: usize>(
    rlev: &RLevCiphertext<N, R_SRC, L>,
    dst_mod: R_DST::Modulus,
) -> RLevCiphertext<N, R_DST, L>
where
    R_SRC: RingPoly<N>,
    R_DST: RingPoly<N>,
{
    RLevCiphertext {
        samples: core::array::from_fn(|i| {
            mod_switch_sym::<N, R_SRC, R_DST>(&rlev.samples[i], dst_mod)
        }),
    }
}

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

/// Generate the **engine** half of a repack preset — the schedule trait, the
/// owned dedicated-key oracle struct, its trait impl, the by-value oracle
/// generator, and the `Repack` function — for a fixed `(n1, K, T)`, mirroring
/// the [`lwe_to_rlwe_cascade!`](crate::lwe_to_rlwe_cascade) pattern.
///
/// Split out from the borrowing-view half ([`repack_view!`]) so a preset whose
/// cascade key has a *different ring type* (e.g. the single-prime
/// `Poly<2048,q2>` repack derived by cross-type mod-switch from the RNS-`q1`
/// cascade) can take the engine **without** the same-ring view machinery — the
/// view would fail to type-check, since no same-ring cascade key exists for it.
///
/// `steps` is the contiguous SUFFIX of the cascade's step list the repack
/// reuses, each `(field, RANK_IN, N_IN, RANK_OUT, N_OUT)`; the generated key
/// fields are degree-`N_OUT` RLev arrays whose types match a suffix of the
/// matching `LweToRlweKey*` — so [`repack_view!`] borrows them zero-copy.
/// `extr_degree` is `G = K/T` (the `Extr` output degree) and `input_count` is
/// `D = T·n1/K` (the padded MLWE count = `2^depth`).
macro_rules! repack_engine {
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
    };
}

/// Generate the **view** half of a repack preset — the borrowing cascade-suffix
/// schedule, [`from_cascade`](crate::conversion::repack) (zero-copy borrow), and
/// `from_cascade_modswitched` (same-ring `q1 → q2` switch into an owned oracle).
///
/// Pairs with [`repack_engine!`]: the engine defines `$sched` and the owned
/// `$key` oracle, and this macro implements `$sched` for the borrowing `$view`
/// and builds `$key` in `from_cascade_modswitched` (item order between the two
/// macro expansions is irrelevant). Only presets whose cascade key is the
/// **same ring type** as the repack key invoke this — the single-prime
/// `Poly<2048,q2>` repack derives its key by a cross-type mod-switch instead
/// (see `repack_keys_poly_2048_t256_from_rns_cascade_boxed`) and takes the
/// engine only.
macro_rules! repack_view {
    (
        n1 = $n1:literal,
        t = $t:literal,
        ring = $ring:ident,
        mod_param = $modp:ident,
        mod_bound = $modbound:path,
        levels = $lev:ident,
        schedule = $sched:ident,
        key = $key:ident,
        cascade_key = $ckey:ident,
        view = $view:ident,
        from_cascade = $from:ident,
        from_cascade_modswitched = $from_ms:ident,
        repack = $repack:ident,
        steps = [ $( ($field:ident, $rin:literal, $nin:literal, $rout:literal, $nout:literal) ),+ $(,)? ],
    ) => {
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

        #[doc = concat!(
            "Mod-switch the `keys_*` suffix of a [`", stringify!($ckey),
            "`] cascade key (at q1) to `dst_mod` (q2), yielding an OWNED [`",
            stringify!($key), "`] repack schedule. The §3.5 key reuse when `q1 ≠ q2`: ",
            "[`", stringify!($repack), "`] runs at q2 (the post-CRot modulus) but the ",
            "cascade keys ship at q1, so the server mod-switches them per-answer — no ",
            "new offline payload. Same-modulus-type only (single-prime); the RNS-q1 → ",
            "single-prime-q2 cross-type switch is a separate path."
        )]
        pub fn $from_ms<$modp: $modbound, const $lev: usize>(
            cascade: &$ckey<$modp, $lev>,
            dst_mod: <$ring<$n1, $modp, $crate::algebra::ring::form::Coefficient>
                as $crate::algebra::ring::RingPoly<$n1>>::Modulus,
        ) -> $key<$modp, $lev> {
            $key {
                $(
                    $field: core::array::from_fn(|i| {
                        $crate::conversion::repack::mod_switch_rlev::<
                            $nout,
                            $ring<$nout, $modp, $crate::algebra::ring::form::Coefficient>,
                            $ring<$nout, $modp, $crate::algebra::ring::form::Coefficient>,
                            $lev,
                        >(&cascade.$field[i], dst_mod)
                    }),
                )+
            }
        }
    };
}

// Toy preset (n1=8, K=4, T=2; depth 2): the reference instantiation, validated
// by `repack_n8_t2_reconstructs`. Its key fields (`keys_4`, `keys_8`) are the
// same types as a suffix of `LweToRlweKeyN8`'s fields — Part 2 borrows them.
repack_engine! {
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
    gen = gen_repack_keys_n8_t2,
    repack = repack_n8_t2,
    steps = [
        (keys_4, 4, 2, 2, 4),
        (keys_8, 2, 4, 1, 8),
    ],
}
repack_view! {
    n1 = 8,
    t = 2,
    ring = Poly,
    mod_param = M,
    mod_bound = crate::algebra::zq::modulus::Modulus,
    levels = L,
    schedule = RepackScheduleN8T2,
    key = RepackKeysN8T2,
    cascade_key = LweToRlweKeyN8,
    view = RepackViewN8T2,
    from_cascade = repack_keys_n8_t2_from_cascade,
    from_cascade_modswitched = repack_keys_n8_t2_from_cascade_modswitched,
    repack = repack_n8_t2,
    steps = [
        (keys_4, 4, 2, 2, 4),
        (keys_8, 2, 4, 1, 8),
    ],
}

// e2e-toy preset (n1=64, K=16, T=8; depth 5): the `ViaBToyParams<64,16,2,8>`
// repack. G = K/T = 2, D = T·n1/K = 32; reuses the `LweToRlweKeyN64` suffix
// `keys_4..keys_64`. Validated by `repack_n64_t8_reconstructs`.
repack_engine! {
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
repack_view! {
    n1 = 64,
    t = 8,
    ring = Poly,
    mod_param = M,
    mod_bound = crate::algebra::zq::modulus::Modulus,
    levels = L,
    schedule = RepackScheduleN64T8,
    key = RepackKeysN64T8,
    cascade_key = LweToRlweKeyN64,
    view = RepackViewN64T8,
    from_cascade = repack_keys_n64_t8_from_cascade,
    from_cascade_modswitched = repack_keys_n64_t8_from_cascade_modswitched,
    repack = repack_n64_t8,
    steps = [
        (keys_4, 32, 2, 16, 4),
        (keys_8, 16, 4, 8, 8),
        (keys_16, 8, 8, 4, 16),
        (keys_32, 4, 16, 2, 32),
        (keys_64, 2, 32, 1, 64),
    ],
}

// Paper preset (n1=2048, K=512, T=256; depth 10): the `ViaBRealisticParams`
// repack. G = K/T = 2, D = T·n1/K = 1024; reuses the `LweToRlweKeyRnsN2048` (RNS,
// alloc) suffix `keys_4..keys_2048`. The by-value oracle `gen_repack_keys_rns_2048
// _t256` is generated but NOT re-exported — a by-value key at this scale overflows
// the stack (like the cascade's own by-value n2048 gen); the depth-spike borrows
// the heap cascade key via the view. GO/NO-GO: `repack_rns_2048_t256_spike`.
repack_engine! {
    n1 = 2048,
    t = 256,
    ring = PolyRns,
    mod_param = B,
    mod_bound = crate::algebra::rns::basis::RnsBasis,
    levels = L,
    extr_degree = 2,    // G = K/T = 512/256
    input_count = 1024, // D = T·n1/K = 256·2048/512
    schedule = RepackScheduleRns2048T256,
    key = RepackKeysRns2048T256,
    gen = gen_repack_keys_rns_2048_t256,
    repack = repack_rns_2048_t256,
    steps = [
        (keys_4, 1024, 2, 512, 4),
        (keys_8, 512, 4, 256, 8),
        (keys_16, 256, 8, 128, 16),
        (keys_32, 128, 16, 64, 32),
        (keys_64, 64, 32, 32, 64),
        (keys_128, 32, 64, 16, 128),
        (keys_256, 16, 128, 8, 256),
        (keys_512, 8, 256, 4, 512),
        (keys_1024, 4, 512, 2, 1024),
        (keys_2048, 2, 1024, 1, 2048),
    ],
}
repack_view! {
    n1 = 2048,
    t = 256,
    ring = PolyRns,
    mod_param = B,
    mod_bound = crate::algebra::rns::basis::RnsBasis,
    levels = L,
    schedule = RepackScheduleRns2048T256,
    key = RepackKeysRns2048T256,
    cascade_key = LweToRlweKeyRnsN2048,
    view = RepackViewRns2048T256,
    from_cascade = repack_keys_rns_2048_t256_from_cascade,
    from_cascade_modswitched = repack_keys_rns_2048_t256_from_cascade_modswitched,
    repack = repack_rns_2048_t256,
    steps = [
        (keys_4, 1024, 2, 512, 4),
        (keys_8, 512, 4, 256, 8),
        (keys_16, 256, 8, 128, 16),
        (keys_32, 128, 16, 64, 32),
        (keys_64, 64, 32, 32, 64),
        (keys_128, 32, 64, 16, 128),
        (keys_256, 16, 128, 8, 256),
        (keys_512, 8, 256, 4, 512),
        (keys_1024, 4, 512, 2, 1024),
        (keys_2048, 2, 1024, 1, 2048),
    ],
}

// Paper-scale SINGLE-PRIME repack preset (n1=2048, K=512, T=256; depth 10) over
// `Poly<2048, q2>` — the *production* paper repack. The real repack runs at the
// single-prime post-CRot modulus q2 (≈2^34, one u64 prime), NOT the q1-RNS
// `PolyRns` of the `…_rns_2048_t256` preset (a single-modulus noise spike). It
// has no same-ring cascade key — the only n2048 cascade is RNS-`q1` — so it
// takes the engine ONLY; its q2 key is derived by cross-type mod-switch from the
// RNS-`q1` cascade via `repack_keys_poly_2048_t256_from_rns_cascade_boxed` (the
// §3.5 key reuse). The by-value `gen_repack_keys_poly_2048_t256` is generated but
// NOT re-exported — a ~11.25 MiB key by value overflows the stack.
repack_engine! {
    n1 = 2048,
    t = 256,
    ring = Poly,
    mod_param = M,
    mod_bound = crate::algebra::zq::modulus::Modulus,
    levels = L,
    extr_degree = 2,    // G = K/T = 512/256
    input_count = 1024, // D = T·n1/K = 256·2048/512
    schedule = RepackSchedulePoly2048T256,
    key = RepackKeysPoly2048T256,
    gen = gen_repack_keys_poly_2048_t256,
    repack = repack_poly_2048_t256,
    steps = [
        (keys_4, 1024, 2, 512, 4),
        (keys_8, 512, 4, 256, 8),
        (keys_16, 256, 8, 128, 16),
        (keys_32, 128, 16, 64, 32),
        (keys_64, 64, 32, 32, 64),
        (keys_128, 32, 64, 16, 128),
        (keys_256, 16, 128, 8, 256),
        (keys_512, 8, 256, 4, 512),
        (keys_1024, 4, 512, 2, 1024),
        (keys_2048, 2, 1024, 1, 2048),
    ],
}

/// Build the single-prime `Poly<2048,q2>` repack key
/// ([`RepackKeysPoly2048T256`]) from the RNS-`q1` LWE→RLWE cascade key
/// ([`LweToRlweKeyRnsN2048`]) the server already holds, **field-by-field on the
/// heap**, cross-type mod-switching each RNS-`q1` step-key RLev → single-prime
/// `q2`.
///
/// The production paper-scale realization of the §3.5 key reuse: the repack runs
/// at the single-prime post-CRot modulus `q2`, but the cascade ships at the
/// 2-prime RNS `q1`; the server derives the `q2` repack key internally — no new
/// offline payload. The same-ring [`from_cascade_modswitched`](repack_view!) of
/// the macro cannot express this (its source and target are one ring type), and
/// returning the ~11.25 MiB key by value would overflow the stack, so this is
/// hand-written to mirror the cascade's own boxed builder.
///
/// Peak stack is a **single** degree-2048 `RLev` (~589 KiB): each
/// [`mod_switch_rlev`] call returns one switched step key, written straight into
/// its heap slot via `addr_of_mut!`, never assembling a whole field (let alone
/// the whole key) on the stack.
#[allow(clippy::needless_range_loop)] // index drives both src field and heap slot
pub fn repack_keys_poly_2048_t256_from_rns_cascade_boxed<B, M, const L: usize>(
    rns: &LweToRlweKeyRnsN2048<B, L>,
    q2: M,
) -> alloc::boxed::Box<RepackKeysPoly2048T256<M, L>>
where
    B: crate::algebra::rns::basis::RnsBasis,
    M: crate::algebra::zq::modulus::Modulus,
{
    use crate::algebra::ring::form::Coefficient;
    use core::ptr::addr_of_mut;

    let mut boxed = alloc::boxed::Box::<RepackKeysPoly2048T256<M, L>>::new_uninit();
    let ptr = boxed.as_mut_ptr();
    // One `mod_switch_rlev` per (field, index); each writes one switched RLev
    // straight into its heap slot. `$nout` (output degree) and the field's
    // `RANK_IN` mirror the engine `steps` list above exactly.
    macro_rules! switch_field {
        ($field:ident, $nout:literal, $rank_in:literal) => {
            for i in 0..$rank_in {
                // SAFETY: `ptr` is a valid `*mut RepackKeysPoly2048T256`; the
                // slot `(*ptr).$field[i]` is one `RLev` of size matching the
                // value written. Every (field, i) pair below is written exactly
                // once, covering the whole struct before `assume_init`. We form
                // no reference to uninitialised memory (`addr_of_mut!`), and the
                // cross-type `mod_switch_rlev` returns the value by move.
                unsafe {
                    addr_of_mut!((*ptr).$field[i]).write(mod_switch_rlev::<
                        $nout,
                        PolyRns<$nout, B, Coefficient>,
                        Poly<$nout, M, Coefficient>,
                        L,
                    >(&rns.$field[i], q2));
                }
            }
        };
    }
    switch_field!(keys_4, 4, 1024);
    switch_field!(keys_8, 8, 512);
    switch_field!(keys_16, 16, 256);
    switch_field!(keys_32, 32, 128);
    switch_field!(keys_64, 64, 64);
    switch_field!(keys_128, 128, 32);
    switch_field!(keys_256, 256, 16);
    switch_field!(keys_512, 512, 8);
    switch_field!(keys_1024, 1024, 4);
    switch_field!(keys_2048, 2048, 2);
    // SAFETY: every (field, index) slot was written above.
    unsafe { boxed.assume_init() }
}

// Paper-scale SINGLE-PRIME repack preset (n1=2048, K=512, T=8; depth 5) over
// `Poly<2048, q2>`, with N3 = N2/T = 64 — a RUNNABLE-batch variant of the
// production `…_poly_2048_t256` preset (same single-prime q2 ring, same
// cross-type key derivation, just T=8 so a full client↔server paper batch e2e
// runs in minutes, not hours). Engine only; its q2 key comes from
// `repack_keys_poly_2048_t8_from_rns_cascade_boxed`. By-value gen NOT re-exported.
repack_engine! {
    n1 = 2048,
    t = 8,
    ring = Poly,
    mod_param = M,
    mod_bound = crate::algebra::zq::modulus::Modulus,
    levels = L,
    extr_degree = 64, // G = K/T = 512/8
    input_count = 32, // D = T·n1/K = 8·2048/512
    schedule = RepackSchedulePoly2048T8,
    key = RepackKeysPoly2048T8,
    gen = gen_repack_keys_poly_2048_t8,
    repack = repack_poly_2048_t8,
    steps = [
        (keys_128, 32, 64, 16, 128),
        (keys_256, 16, 128, 8, 256),
        (keys_512, 8, 256, 4, 512),
        (keys_1024, 4, 512, 2, 1024),
        (keys_2048, 2, 1024, 1, 2048),
    ],
}

/// Boxed cross-type q2-key derivation for the `T=8` paper preset — the `T=8`
/// analogue of [`repack_keys_poly_2048_t256_from_rns_cascade_boxed`]: builds
/// [`RepackKeysPoly2048T8`] from the RNS-`q1` cascade key field-by-field on the
/// heap (its suffix `keys_128..keys_2048`, 5 fields), cross-type mod-switching
/// each RLev → single-prime `q2`.
#[allow(clippy::needless_range_loop)]
pub fn repack_keys_poly_2048_t8_from_rns_cascade_boxed<B, M, const L: usize>(
    rns: &LweToRlweKeyRnsN2048<B, L>,
    q2: M,
) -> alloc::boxed::Box<RepackKeysPoly2048T8<M, L>>
where
    B: crate::algebra::rns::basis::RnsBasis,
    M: crate::algebra::zq::modulus::Modulus,
{
    use crate::algebra::ring::form::Coefficient;
    use core::ptr::addr_of_mut;

    let mut boxed = alloc::boxed::Box::<RepackKeysPoly2048T8<M, L>>::new_uninit();
    let ptr = boxed.as_mut_ptr();
    macro_rules! switch_field {
        ($field:ident, $nout:literal, $rank_in:literal) => {
            for i in 0..$rank_in {
                // SAFETY: as in the `…_t256_…` builder — one write per (field, i)
                // slot, all covered before `assume_init`; `addr_of_mut!` forms no
                // reference to uninitialised memory.
                unsafe {
                    addr_of_mut!((*ptr).$field[i]).write(mod_switch_rlev::<
                        $nout,
                        PolyRns<$nout, B, Coefficient>,
                        Poly<$nout, M, Coefficient>,
                        L,
                    >(&rns.$field[i], q2));
                }
            }
        };
    }
    switch_field!(keys_128, 128, 32);
    switch_field!(keys_256, 256, 16);
    switch_field!(keys_512, 512, 8);
    switch_field!(keys_1024, 1024, 4);
    switch_field!(keys_2048, 2048, 2);
    // SAFETY: every (field, index) slot was written above.
    unsafe { boxed.assume_init() }
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

    /// §3.5 across q1 ≠ q2 (the server's internal key mod-switch, in isolation):
    /// gen the cascade at q1, encrypt the T inputs at q2 under the SAME S1
    /// (rekeyed q1→q2), mod-switch the cascade suffix q1→q2 via
    /// `repack_keys_n8_t2_from_cascade_modswitched`, repack at q2, decrypt under
    /// S1@q2. The paper interleave reconstructs — the repack reuses the q1 cascade
    /// keys at the q2 post-CRot modulus.
    #[test]
    fn repack_n8_t2_modswitched_q1_to_q2_reconstructs() {
        use crate::algebra::zq::modulus::DynModulus;
        use crate::switching::rekey::rekey_secret_key;
        type R8d = Poly<8, DynModulus, Coefficient>;
        let q1 = DynModulus::new(1 << 36);
        let q2 = DynModulus::new(1 << 28);
        let p = DynModulus::new(16);
        const L: usize = 7;
        const BASE: u64 = 64;
        let mut prg = Shake256Prg::new(b"repack-n8-t2-modswitch-q1q2");
        let sk1 = SecretKey::<8, R8d>::keygen(q1, Distribution::Ternary, &mut prg);
        let sk1_q2 = rekey_secret_key::<8, R8d, R8d>(&sk1, q2);
        // T=2 inputs encrypted at q2 under S1@q2.
        let m0 = encode::<8, R8d, R8d>(&R8d::new(p, [2, 0, 0, 0, 5, 0, 0, 0]), q2);
        let m1 = encode::<8, R8d, R8d>(&R8d::new(p, [3, 0, 0, 0, 7, 0, 0, 0]), q2);
        let c0 = sk1_q2.encrypt(&m0, Distribution::Ternary, &mut prg);
        let c1 = sk1_q2.encrypt(&m1, Distribution::Ternary, &mut prg);
        // Cascade key built at q1; mod-switch its suffix → q2 (server-internal).
        let cascade =
            gen_lwe_to_rlwe_key_n8::<DynModulus, L>(&sk1, BASE, Distribution::Ternary, &mut prg);
        let keys_q2 = repack_keys_n8_t2_from_cascade_modswitched(&cascade, q2);
        let out = repack_n8_t2(&[c0, c1], &keys_q2, BASE);
        // Decrypt under S1@q2.
        let mut acc = out.body;
        acc -= out.mask * *sk1_q2.poly();
        let recovered: R8d = decode::<8, R8d, R8d>(&acc, p);
        assert_eq!(recovered.coeff(0).to_u64(), 2, "M0_0 @ slot 0");
        assert_eq!(recovered.coeff(2).to_u64(), 3, "M1_0 @ slot 2");
        assert_eq!(recovered.coeff(4).to_u64(), 5, "M0_4 @ slot 4");
        assert_eq!(recovered.coeff(6).to_u64(), 7, "M1_4 @ slot 6");
    }

    /// §3.5 CROSS-TYPE (RNS q1 → single-prime q2) — the paper-shape modulus
    /// switch in miniature: gen the **RNS** cascade at q1 (`PolyRns`), cross-type
    /// mod-switch its `keys_4`/`keys_8` suffix → **single-prime** q2 (`Poly`) into
    /// a `RepackKeysN8T2`, encrypt the inputs at q2 under S1@q2, repack at q2, and
    /// decrypt — the interleave reconstructs. Proves the repack reuses the q1-RNS
    /// cascade keys at the single-prime q2 the *paper* repack actually runs at (the
    /// real paper path; the `repack_rns_2048_t256` PolyRns preset is a single-
    /// modulus noise spike, not this).
    #[test]
    fn repack_n8_t2_modswitched_rns_q1_to_single_prime_q2_reconstructs() {
        use crate::algebra::ring::rns_element::PolyRns;
        use crate::algebra::rns::basis::ConstRnsBasis;
        use crate::algebra::zq::modulus::DynModulus;
        use crate::conversion::gen_lwe_to_rlwe_key_rns_n8;
        use crate::switching::rekey::rekey_secret_key;
        type Rns<const N: usize> = PolyRns<N, ConstRnsBasis<7681, 12289>, Coefficient>; // q1≈2^26.6
        type R8d = Poly<8, DynModulus, Coefficient>; // q2 single-prime
        let basis = ConstRnsBasis::<7681, 12289>;
        let q2 = DynModulus::new(1 << 20);
        let p = DynModulus::new(16);
        const L: usize = 7;
        const BASE: u64 = 64;
        let mut prg = Shake256Prg::new(b"repack-n8-t2-rns-q1-to-q2");
        let sk_rns = SecretKey::<8, Rns<8>>::keygen(basis, Distribution::Ternary, &mut prg);
        // Same ternary S1, reinterpreted at the single-prime q2 (cross-type rekey).
        let sk_q2 = rekey_secret_key::<8, Rns<8>, R8d>(&sk_rns, q2);
        let m0 = encode::<8, R8d, R8d>(&R8d::new(p, [2, 0, 0, 0, 5, 0, 0, 0]), q2);
        let m1 = encode::<8, R8d, R8d>(&R8d::new(p, [3, 0, 0, 0, 7, 0, 0, 0]), q2);
        let c0 = sk_q2.encrypt(&m0, Distribution::Ternary, &mut prg);
        let c1 = sk_q2.encrypt(&m1, Distribution::Ternary, &mut prg);
        // RNS cascade at q1; cross-type mod-switch the suffix → single-prime q2.
        let cascade =
            gen_lwe_to_rlwe_key_rns_n8::<_, L>(&sk_rns, BASE, Distribution::Ternary, &mut prg);
        let keys_q2 = RepackKeysN8T2::<DynModulus, L> {
            keys_4: core::array::from_fn(|i| {
                mod_switch_rlev::<4, Rns<4>, Poly<4, DynModulus, Coefficient>, L>(
                    &cascade.keys_4[i],
                    q2,
                )
            }),
            keys_8: core::array::from_fn(|i| {
                mod_switch_rlev::<8, Rns<8>, R8d, L>(&cascade.keys_8[i], q2)
            }),
        };
        let out = repack_n8_t2(&[c0, c1], &keys_q2, BASE);
        let mut acc = out.body;
        acc -= out.mask * *sk_q2.poly();
        let recovered: R8d = decode::<8, R8d, R8d>(&acc, p);
        assert_eq!(recovered.coeff(0).to_u64(), 2, "M0_0 @ slot 0");
        assert_eq!(recovered.coeff(2).to_u64(), 3, "M1_0 @ slot 2");
        assert_eq!(recovered.coeff(4).to_u64(), 5, "M0_4 @ slot 4");
        assert_eq!(recovered.coeff(6).to_u64(), 7, "M1_4 @ slot 6");
    }
}

/// Paper-scale depth-10 repack SPIKE — the P2 GO/NO-GO. Mirrors the cascade's own
/// `spike_n2048_depth18_noise_closes`: builds the ~24.75 MB `LweToRlweKeyRnsN2048`
/// cascade key with the production **boxed** builder on a bounded thread, borrows
/// its `keys_4..keys_2048` suffix as the repack schedule (the §3.5 key reuse — no
/// new offline payload), packs `T = 256` RLWEs over `R_{2048, q1}` through the
/// depth-10 `mlwes_to_mlwe` recursion, and decrypts. PASS ⇒ noise closes at depth
/// 10 ⇒ paper-scale repack GO; panic ⇒ NO-GO.
#[cfg(test)]
mod spike {
    // `via-primitives` is `#![no_std]`; the test harness links `std`, named here
    // explicitly to reach `std::thread` for the bounded-stack spawn.
    extern crate std;

    use super::{
        repack_keys_poly_2048_t256_from_rns_cascade_boxed, repack_keys_rns_2048_t256_from_cascade,
        repack_poly_2048_t256, repack_rns_2048_t256,
    };
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::rns::basis::paper::ViaCQ1Rns;
    use crate::algebra::zq::modulus::ConstModulus;
    use crate::algebra::zq::modulus::paper::ViaCQ2;
    use crate::conversion::gen_lwe_to_rlwe_key_rns_n2048_boxed;
    use crate::encryption::encode;
    use crate::encryption::types::SecretKey;
    use crate::sampling::distribution::Distribution;
    use crate::sampling::prg::Shake256Prg;
    use crate::switching::rekey::rekey_secret_key;
    use alloc::vec::Vec;

    const N1: usize = 2048;
    const T: usize = 256;
    const L_CK: usize = 18; // conversion-key gadget depth (paper Table 6)
    const CK_BASE: u64 = 18; // conversion-key gadget base
    const VAL: u64 = 7; // the single nonzero message coefficient (< p = 16)
    const STACK: usize = 16 << 20; // 16 MB — boxed key is heap; covers builder + scratch

    /// **Depth-10 repack noise GO/NO-GO** at the paper `(n1,K,T) = (2048,512,256)`.
    #[test]
    #[ignore = "very heavy: ~24.75 MB key + depth-10 repack of 256 RLWEs at n=2048; \
                run with --features via-b,alloc --release -- --ignored"]
    fn repack_rns_2048_t256_spike() {
        type R1 = PolyRns<N1, ViaCQ1Rns, Coefficient>;
        type P1 = Poly<N1, ConstModulus<16>, Coefficient>;

        std::thread::Builder::new()
            .stack_size(STACK)
            .spawn(|| {
                let basis = ViaCQ1Rns::default();
                let p = ConstModulus::<16>;
                let mut prg = Shake256Prg::new(b"repack-rns-2048-t256-spike");
                let sk = SecretKey::<N1, R1>::keygen(basis, Distribution::Ternary, &mut prg);

                // The production ~24.75 MB cascade key (heap-built field-by-field, so
                // peak stack ≪ the whole key), borrowed as the repack schedule.
                let cascade = gen_lwe_to_rlwe_key_rns_n2048_boxed::<ViaCQ1Rns, L_CK>(
                    &sk,
                    CK_BASE,
                    Distribution::Ternary,
                    &mut prg,
                );
                let view = repack_keys_rns_2048_t256_from_cascade(&cascade);

                // T = 256 RLWE inputs, each carrying VAL=7 in coeff 0 (else zero).
                // Heap `Vec` (16 MB of inputs) borrowed as `&[_; T]` — a by-value
                // `[RLWE; 256]` would be a ~16 MB stack array.
                let mut v: Vec<_> = Vec::with_capacity(T);
                for _ in 0..T {
                    let mut coeffs = [0u64; N1];
                    coeffs[0] = VAL;
                    let m = encode::<N1, R1, P1>(&Poly::new(p, coeffs), basis);
                    v.push(sk.encrypt(&m, Distribution::Ternary, &mut prg));
                }
                let inputs: &[_; T] = (&v[..]).try_into().expect("T encrypted inputs");

                let out = repack_rns_2048_t256(inputs, &view, CK_BASE);
                let recovered: P1 = sk.decrypt(&out, p);

                // Noise GO/NO-GO (position-agnostic): the interleave is a permutation,
                // so the T designated coefficients each carry VAL and every other
                // coefficient is zero. Any depth-10 noise overflow corrupts a
                // VAL→VAL±1 or a 0→nonzero, failing one of these asserts. (Exact slot
                // positions `4t` are validated by the depth-2/5 reconstructability
                // tests; this spike isolates the depth-10 noise budget.)
                let mut designated = 0usize;
                for i in 0..N1 {
                    let c = recovered.coeff(i).to_u64();
                    assert!(
                        c == 0 || c == VAL,
                        "coeff {i} = {c} ∉ {{0, {VAL}}} — noise!"
                    );
                    if c == VAL {
                        designated += 1;
                    }
                }
                assert_eq!(
                    designated, T,
                    "exactly T={T} designated coefficients carry VAL"
                );
            })
            .expect("spawn repack spike thread")
            .join()
            .expect("repack spike panicked (depth-10 noise did not close?)");
    }

    /// **The PRODUCTION paper repack GO/NO-GO** — depth-10 single-prime repack at
    /// the paper `(n1,K,T) = (2048,512,256)`, the §3.5 key reuse end to end at
    /// real scale. Supersedes [`repack_rns_2048_t256_spike`] (which packs over the
    /// q1-RNS `PolyRns`, a single-modulus noise spike) as the real paper gate:
    /// here the repack runs at the **single-prime** post-CRot modulus q2 the paper
    /// actually uses.
    ///
    /// The pipeline: gen the secret key S1 at q1-RNS, build the ~24.75 MiB RNS
    /// cascade key (boxed), **cross-type mod-switch** its `keys_4..keys_2048`
    /// suffix into the ~11.25 MiB single-prime q2 repack key (boxed), rekey
    /// S1 → q2, encrypt T = 256 inputs at q2 under S1@q2, run
    /// [`repack_poly_2048_t256`], and decrypt under S1@q2. PASS ⇒ the cross-type
    /// key derivation + depth-10 noise budget close at q2 ⇒ production repack GO;
    /// panic ⇒ NO-GO. Both big keys + the inputs live on the heap, so peak stack
    /// stays small (one degree-2048 `RLev` ≈ 589 KiB during the key build).
    #[test]
    #[ignore = "very heavy: ~24.75 MiB RNS cascade + ~11.25 MiB q2 key + depth-10 \
                repack of 256 RLWEs at n=2048; run with --features via-b,alloc --release -- --ignored"]
    fn repack_poly_2048_t256_spike() {
        type Rq1 = PolyRns<N1, ViaCQ1Rns, Coefficient>;
        type Rq2 = Poly<N1, ViaCQ2, Coefficient>;
        type P1 = Poly<N1, ConstModulus<16>, Coefficient>;

        std::thread::Builder::new()
            .stack_size(STACK)
            .spawn(|| {
                let basis = ViaCQ1Rns::default();
                let q2 = ViaCQ2::default();
                let p = ConstModulus::<16>;
                let mut prg = Shake256Prg::new(b"repack-poly-2048-t256-spike");
                let sk1 = SecretKey::<N1, Rq1>::keygen(basis, Distribution::Ternary, &mut prg);

                // The ~24.75 MiB RNS-q1 cascade key (heap-built), then the
                // ~11.25 MiB single-prime q2 repack key derived from it by
                // cross-type mod-switch (heap-built). Both never transit the stack.
                let cascade = gen_lwe_to_rlwe_key_rns_n2048_boxed::<ViaCQ1Rns, L_CK>(
                    &sk1,
                    CK_BASE,
                    Distribution::Ternary,
                    &mut prg,
                );
                let q2_key = repack_keys_poly_2048_t256_from_rns_cascade_boxed(&cascade, q2);

                // S1 reinterpreted at the single-prime q2 (cross-type rekey): the
                // inputs are encrypted at q2 and decrypt under S1@q2.
                let sk_q2 = rekey_secret_key::<N1, Rq1, Rq2>(&sk1, q2);

                // T = 256 RLWE inputs at q2, each carrying VAL in coeff 0 (else
                // zero). Heap `Vec` borrowed as `&[_; T]` — a by-value
                // `[RLWE; 256]` at n=2048 would be a ~16 MiB stack array.
                let mut v: Vec<_> = Vec::with_capacity(T);
                for _ in 0..T {
                    let mut coeffs = [0u64; N1];
                    coeffs[0] = VAL;
                    let m = encode::<N1, Rq2, P1>(&Poly::new(p, coeffs), q2);
                    v.push(sk_q2.encrypt(&m, Distribution::Ternary, &mut prg));
                }
                let inputs: &[_; T] = (&v[..]).try_into().expect("T encrypted inputs");

                let out = repack_poly_2048_t256(inputs, &*q2_key, CK_BASE);
                let recovered: P1 = sk_q2.decrypt(&out, p);

                // Noise GO/NO-GO (position-agnostic, as in the RNS spike): the
                // interleave is a permutation, so exactly T coefficients carry VAL
                // and every other is zero. Any cross-type-key or depth-10 noise
                // overflow corrupts a VAL→VAL±1 or a 0→nonzero.
                let mut designated = 0usize;
                for i in 0..N1 {
                    let c = recovered.coeff(i).to_u64();
                    assert!(
                        c == 0 || c == VAL,
                        "coeff {i} = {c} ∉ {{0, {VAL}}} — noise!"
                    );
                    if c == VAL {
                        designated += 1;
                    }
                }
                assert_eq!(
                    designated, T,
                    "exactly T={T} designated coefficients carry VAL"
                );
            })
            .expect("spawn poly repack spike thread")
            .join()
            .expect("poly repack spike panicked (cross-type key or depth-10 noise did not close?)");
    }
}
