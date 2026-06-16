//! [`ServerScheme`] — the compile-time crypto backend of a [`Server`](crate::Server).
//!
//! A `ServerScheme<N1, N2>` bundles the four ring backends and the cascade-key
//! type that a VIA server instantiation uses, plus the two injected homomorphic
//! leaf operations — the LWE→RLWE **cascade** (paper §4.1) and, under `via-b`,
//! the MLWEs→RLWE **repack** (paper §4). Bundling them behind one trait lets
//! [`Server`](crate::Server) carry a single backend parameter `Rg` instead of
//! five ring/key types plus two `Fn` objects.
//!
//! The ring names mirror the paper's modulus chain $q_1 > q_2 > q_3 > q_4 > p$
//! over degrees $n_2 \mid n_1$:
//!
//! | assoc type | paper | role |
//! |------------|-------|------|
//! | [`R1`](ServerScheme::R1) | $R_{n_1,q_1}$ | cascade output / query-compression |
//! | [`R2`](ServerScheme::R2) | $R_{n_1,q_2}$ | DMux/FirstDim/CMux/CRot + repack |
//! | [`R3`](ServerScheme::R3) | $R_{n_2,q_3}$ | ring-switch output / answer mask |
//! | [`R4`](ServerScheme::R4) | $R_{n_2,q_4}$ | answer body |
//!
//! `cascade`/`repack` are **orchestrator methods** (they compose the per-`n`
//! monomorphic kernels `lwe_to_rlwe_n<N1>_eval` / `repack_<n1>_t<T>`); the
//! data-parallel slice kernels stay free functions, and dispatch is static — so
//! this is *not* a compute-dispatch `Backend` trait.

// The ring-projection types (`MLWECiphertext<…, R1::Projected<1>>`, the
// `PhantomData<fn() -> …>` carrier) are intrinsic here — same `type_complexity`
// the free pipeline functions already allow.
#![allow(clippy::type_complexity)]

use core::marker::PhantomData;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::ring::ntt::NttFriendly;
use via_primitives::algebra::ring::rns_element::PolyRns;
use via_primitives::algebra::ring::{RingPoly, RingPolyEval};
use via_primitives::algebra::rns::basis::RnsBasis;
use via_primitives::algebra::zq::modulus::Modulus;
use via_primitives::conversion::{
    CascadeKey, LweToRlweKeyN8, LweToRlweKeyN64, LweToRlweKeyRnsN2048, lwe_to_rlwe_n8_eval,
    lwe_to_rlwe_n64_eval, lwe_to_rlwe_rns_n2048_eval,
};
use via_primitives::encryption::MLWECiphertext;
use via_primitives::encryption::types::RLWECiphertext;
use zeroize::Zeroize;

// Repack kernels + their per-answer q1→q2 key derivations (via-b only). The
// single-prime families mod-switch the cascade-key suffix to q2 in-type
// (`*_from_cascade_modswitched`); the RNS family cross-types it to single-prime
// q2 on the heap (`*_from_rns_cascade_boxed`).
#[cfg(feature = "via-b")]
use via_primitives::conversion::repack::{
    repack_keys_n8_t2_from_cascade_modswitched, repack_keys_n64_t8_from_cascade_modswitched,
    repack_keys_poly_2048_t8_from_rns_cascade_boxed,
    repack_keys_poly_2048_t256_from_rns_cascade_boxed, repack_n8_t2, repack_n64_t8,
    repack_poly_2048_t8, repack_poly_2048_t256,
};

/// The compile-time crypto backend of a [`Server`](crate::Server): its ring/key
/// types and the cascade/repack leaf operations. One impl per cascade/repack
/// family, emitted by the `server_scheme_poly!` / `server_scheme_rns!` macros;
/// selected at the type level via the [`Scheme`] carrier.
pub trait ServerScheme<const N1: usize, const N2: usize> {
    /// LWE→RLWE cascade-key type (heap-boxed inside `PublicParams`).
    type K: Zeroize + CascadeKey;
    /// $R_{n_1,q_1}$ ($q_1$ at degree $n_1$) — the cascade output / query-compression ring.
    type R1: RingPoly<N1> + RingPolyEval<N1>;
    /// $R_{n_1,q_2}$ ($q_2$ at degree $n_1$) — the DMux/FirstDim/CMux/CRot + repack ring.
    type R2: RingPoly<N1> + RingPolyEval<N1>;
    /// $R_{n_2,q_3}$ ($q_3$ at degree $n_2$) — the ring-switch output / answer **mask** ring.
    type R3: RingPoly<N2> + RingPolyEval<N2>;
    /// $R_{n_2,q_4}$ ($q_4$ at degree $n_2$) — the answer **body** ring. No `RingPolyEval`:
    /// it is only ever the output of RespComp's trailing coefficient-domain `mod_switch_asym`
    /// rescale (paper $\mathsf{ModSwitch}_{q_3,q_4}$), never NTT-multiplied.
    type R4: RingPoly<N2>;

    /// LWE→RLWE conversion (paper §4.1; batch-size-agnostic). Wired per family to
    /// `lwe_to_rlwe_n<N1>_eval`.
    fn cascade(
        arg: &MLWECiphertext<N1, 1, <Self::R1 as RingPoly<N1>>::Projected<1>>,
        key: &<Self::K as CascadeKey>::Eval,
        base: u64,
    ) -> RLWECiphertext<N1, Self::R1>;

    /// MLWEs→RLWE homomorphic repacking (paper §4) of `T` post-CRot ciphertexts into one.
    /// Dispatches to the family's `repack_<n1>_t<T>` via a const-folded `match T`; an
    /// unsupported `T` is a compile error. `q2` and the gadget `base` (= `pp.ck_base`)
    /// are supplied by the server for the per-answer q1→q2 repack-key derivation.
    /// `via-b` only.
    #[cfg(feature = "via-b")]
    fn repack<const T: usize>(
        rotateds: &[RLWECiphertext<N1, Self::R2>],
        k: &Self::K,
        q2: <Self::R2 as RingPoly<N1>>::Modulus,
        base: u64,
    ) -> RLWECiphertext<N1, Self::R2>;
}

/// Zero-sized carrier selecting a [`ServerScheme`] at the type level: the
/// cascade-key type `K` plus the four ring backends `R1`–`R4`. Never
/// constructed — `PhantomData<fn() -> …>` keeps it `Send + Sync` with no drop
/// glue (it must not inherit `K`'s `ZeroizeOnDrop`).
pub struct Scheme<K, R1, R2, R3, R4>(PhantomData<fn() -> (K, R1, R2, R3, R4)>);

/// Emit a [`ServerScheme`] impl for one **single-prime** cascade family — `q1`
/// and `q2` single-prime `Poly` of the same modulus type `M` at degree `$n1` (the
/// toy stacks). Generic over `M`, the cascade depth `L`, and the response rings
/// `R3`/`R4`; `R1 = R2 = Poly<$n1, M>`. `eval_degrees` are the cascade's per-step
/// output degrees (driving the `RingPolyEval` bounds `lwe_to_rlwe_n<$n1>_eval`
/// needs). The optional `repack` arms (via-b) list `(T, repack_fn, keys_fn)` for
/// the supported batch sizes; `keys_fn` mod-switches the cascade-key suffix to
/// q2 (`*_from_cascade_modswitched`).
macro_rules! server_scheme_poly {
    (
        n1 = $n1:literal, n2 = $n2:literal,
        cascade_key = $ckey:ident,
        cascade = $cast:ident,
        eval_degrees = [ $($deg:literal),+ $(,)? ]
        $(, repack = [ $( ($t:literal, $repack:ident, $keys:ident) ),+ $(,)? ] )?
        $(,)?
    ) => {
        impl<M: Modulus, const L: usize, R3, R4> ServerScheme<$n1, $n2>
            for Scheme<$ckey<M, L>, Poly<$n1, M, Coefficient>, Poly<$n1, M, Coefficient>, R3, R4>
        where
            R3: RingPoly<$n2> + RingPolyEval<$n2>,
            R4: RingPoly<$n2>,
            $( Poly<$deg, M, Coefficient>: RingPolyEval<$deg>, )+
        {
            type K = $ckey<M, L>;
            type R1 = Poly<$n1, M, Coefficient>;
            type R2 = Poly<$n1, M, Coefficient>;
            type R3 = R3;
            type R4 = R4;

            fn cascade(
                arg: &MLWECiphertext<$n1, 1, <Poly<$n1, M, Coefficient> as RingPoly<$n1>>::Projected<1>>,
                key: &<$ckey<M, L> as CascadeKey>::Eval,
                base: u64,
            ) -> RLWECiphertext<$n1, Poly<$n1, M, Coefficient>> {
                $cast::<M, L>(arg, key, base)
            }

            $(
            #[cfg(feature = "via-b")]
            #[allow(clippy::match_single_binding)]
            fn repack<const T: usize>(
                rotateds: &[RLWECiphertext<$n1, Poly<$n1, M, Coefficient>>],
                k: &$ckey<M, L>,
                q2: M,
                base: u64,
            ) -> RLWECiphertext<$n1, Poly<$n1, M, Coefficient>> {
                const {
                    assert!(
                        $( T == $t )||+ ,
                        concat!("repack: unsupported batch size T for the n", stringify!($n1), " family")
                    );
                }
                match T {
                    $(
                        $t => {
                            let arr: &[_; $t] =
                                rotateds.try_into().expect("repack: rotateds.len() must equal T");
                            $repack(arr, &$keys(k, q2), base)
                        }
                    )+
                    _ => unreachable!(),
                }
            }
            )?
        }
    };
}

// Toy single-prime cascade families. (n8 repack T=2 is never exercised by a
// `ViaBServer` test — only the via-c n8 server is — but `via-b` requires every
// `ServerScheme` impl to provide `fn repack`.)
server_scheme_poly! {
    n1 = 8, n2 = 4,
    cascade_key = LweToRlweKeyN8, cascade = lwe_to_rlwe_n8_eval,
    eval_degrees = [2, 4, 8],
    repack = [(2, repack_n8_t2, repack_keys_n8_t2_from_cascade_modswitched)],
}
server_scheme_poly! {
    n1 = 64, n2 = 16,
    cascade_key = LweToRlweKeyN64, cascade = lwe_to_rlwe_n64_eval,
    eval_degrees = [2, 4, 8, 16, 32, 64],
    repack = [(8, repack_n64_t8, repack_keys_n64_t8_from_cascade_modswitched)],
}

/// Emit a [`ServerScheme`] impl for one **RNS** cascade family — `q1` a two-prime
/// `PolyRns` at degree `$n1` (the paper stack). Generic over the `q1` RNS basis
/// `B`, the cascade depth `L`, the single-prime `q2` modulus `M2`, and the
/// response rings; `R1 = PolyRns<$n1, B>`, `R2 = Poly<$n1, M2>`. The optional
/// `repack` arms (via-b) cross-type the RNS-q1 cascade key to single-prime q2 on
/// the heap (`*_from_rns_cascade_boxed`).
macro_rules! server_scheme_rns {
    (
        n1 = $n1:literal, n2 = $n2:literal,
        cascade_key = $ckey:ident,
        cascade = $cast:ident,
        eval_degrees = [ $($deg:literal),+ $(,)? ]
        $(,
            repack = [ $( ($t:literal, $repack:ident, $keys:ident) ),+ $(,)? ],
            // q2 NTT degrees the `repack_<n1>_t<T>` trees touch — the generic `M2`
            // must be NTT-friendly there (the paper q2 is, at every power of two ≤ n1).
            repack_ntt = [ $( $nttdeg:literal ),+ $(,)? ]
        )?
        $(,)?
    ) => {
        impl<B: RnsBasis, M2: Modulus, const L: usize, R3, R4> ServerScheme<$n1, $n2>
            for Scheme<$ckey<B, L>, PolyRns<$n1, B, Coefficient>, Poly<$n1, M2, Coefficient>, R3, R4>
        where
            R3: RingPoly<$n2> + RingPolyEval<$n2>,
            R4: RingPoly<$n2>,
            $( PolyRns<$deg, B, Coefficient>: RingPolyEval<$deg>, )+
            Poly<$n1, M2, Coefficient>: RingPolyEval<$n1>,
            $( $( M2: NttFriendly<$nttdeg>, )+ )?
        {
            type K = $ckey<B, L>;
            type R1 = PolyRns<$n1, B, Coefficient>;
            type R2 = Poly<$n1, M2, Coefficient>;
            type R3 = R3;
            type R4 = R4;

            fn cascade(
                arg: &MLWECiphertext<$n1, 1, <PolyRns<$n1, B, Coefficient> as RingPoly<$n1>>::Projected<1>>,
                key: &<$ckey<B, L> as CascadeKey>::Eval,
                base: u64,
            ) -> RLWECiphertext<$n1, PolyRns<$n1, B, Coefficient>> {
                $cast::<B, L>(arg, key, base)
            }

            $(
            #[cfg(feature = "via-b")]
            fn repack<const T: usize>(
                rotateds: &[RLWECiphertext<$n1, Poly<$n1, M2, Coefficient>>],
                k: &$ckey<B, L>,
                q2: M2,
                base: u64,
            ) -> RLWECiphertext<$n1, Poly<$n1, M2, Coefficient>> {
                const {
                    assert!(
                        $( T == $t )||+ ,
                        concat!("repack: unsupported batch size T for the rns-n", stringify!($n1), " family")
                    );
                }
                match T {
                    $(
                        $t => {
                            let arr: &[_; $t] =
                                rotateds.try_into().expect("repack: rotateds.len() must equal T");
                            $repack(arr, &*$keys(k, q2), base)
                        }
                    )+
                    _ => unreachable!(),
                }
            }
            )?
        }
    };
}

// Paper RNS cascade family (q1 = two-prime RNS, n1 = 2048, n2 = 512).
server_scheme_rns! {
    n1 = 2048, n2 = 512,
    cascade_key = LweToRlweKeyRnsN2048, cascade = lwe_to_rlwe_rns_n2048_eval,
    eval_degrees = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048],
    repack = [
        (8, repack_poly_2048_t8, repack_keys_poly_2048_t8_from_rns_cascade_boxed),
        (256, repack_poly_2048_t256, repack_keys_poly_2048_t256_from_rns_cascade_boxed),
    ],
    repack_ntt = [4, 8, 16, 32, 64, 128, 256, 512, 1024],
}
