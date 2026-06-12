//! The VIA-B batch answer pipeline ([`answer_batch`]).
//!
//! VIA-B answers a batch of `T` queries by running the variant-common prefix
//! [`answer_through_crot`] (steps 1–6) once per query, **repacking** the `T`
//! CRot outputs into a single ciphertext (the one new VIA-B step), and
//! then running [`resp_comp`] (step 7) exactly once. So the per-query cost is the
//! VIA-C prefix and only the cheap tail is shared — `T` records for ~one answer.
//!
//! Gated `#[cfg(feature = "via-b")]` at the [`crate`] re-export boundary.

use alloc::vec::Vec;
use via_primitives::algebra::ring::{RingPoly, RingPolyEval};
use via_primitives::conversion::CascadeKey;
use via_primitives::encryption::MLWECiphertext;
use via_primitives::encryption::types::{ModSwitchedCiphertext, RLWECiphertext};
use via_protocol::{BatchedQuery, PublicParams, ViaError};
use zeroize::Zeroize;

use crate::answer::answer_through_crot;
use crate::prepared_db::PreparedDb;
use crate::prepared_keys::PreparedKeys;
use crate::resp_comp::resp_comp;

/// VIA-B answer pipeline for a batch of `T` queries:
///
/// 1. [`answer_through_crot`] × `T` (steps 1–6 at record degree `N3`) → `T`
///    `RLWECiphertext<N1, R2>` @ q2.
/// 2. `Repack_{N2}` the `T` rotateds into one `RLWECiphertext<N1, R2>` (the
///    VIA-B repacking step), via the injected `repack` closure.
/// 3. [`resp_comp`] once → `ModSwitchedCiphertext<N2, R3, R4>`.
///
/// # Why `repack` is injected (not a generic call)
///
/// The repack family is **per-preset** (macro-generated `repack_n8_t2`,
/// `repack_n64_t8`, `repack_rns_2048_t256`, each with its own borrowing
/// schedule view over the cascade key's named-degree `keys_*`), so one generic
/// engine cannot field-access those keys. The caller therefore injects a closure
/// that bakes in the preset + base, e.g.
/// `|rotateds, k| repack_n64_t8(rotateds.try_into().unwrap(), &repack_keys_n64_t8_from_cascade(k), base)`
/// — exactly the `cascade: CascadeFn` injection
/// [`answer_one_query`](crate::answer::answer_one_query) already uses.
/// Its `&K` is `&pp.query_comp_key.lwe_to_rlwe_key` (the cascade-key reuse: the
/// repack borrows the query-compression cascade key — no new offline payload).
///
/// # Type parameters
///
/// As [`answer_one_query`](crate::answer::answer_one_query) plus `N3` (record
/// degree, = the server's `N_REC`) and
/// `T` (batch count). `RepackFn`/`CascadeFn` are the two injected operations.
///
/// # Errors
///
/// - [`ViaError::DimMismatch`] if `batch.len() != T`.
/// - [`ViaError::QueryLengthMismatch`] propagated from any inner
///   [`answer_through_crot`] (each inner query must carry
///   `(log₂I + log₂J + log₂(N1/N3)) · L_QUERY` LWEs).
///
/// # Tracing spans
///
/// `"answer_batch"` (parent) ⊃ `"answer_through_crot"` × `T`, `"step_repack"`,
/// `"step7_resp_comp"`.
///
/// # Constant-time: No
///
/// Branches only on public data (query ciphertexts, the cleartext database); see
/// [`answer_one_query`](crate::answer::answer_one_query).
#[allow(
    non_camel_case_types,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
pub fn answer_batch<
    const N1: usize,
    const N2: usize,
    const N3: usize,
    const T: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R2: RingPoly<N1> + RingPolyEval<N1>,
    R3L: RingPoly<N1, Projected<N2> = R3>,
    R3: RingPoly<N2, Modulus = R3L::Modulus> + RingPolyEval<N2>,
    R4: RingPoly<N2>,
    K: Zeroize + CascadeKey,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
    RepackFn,
    CascadeFn,
>(
    batch: &BatchedQuery<N1, 1, R1::Projected<1>>,
    pp: &PublicParams<K, N1, N2, R1, R3, L_QUERY, L_CK, L_RSK, D>,
    prepared_db: &PreparedDb<N1, R2>,
    prepared_keys: &PreparedKeys<N1, N2, R1, R3, K, L_CK, L_RSK, D>,
    q1_mod: R1::Modulus,
    q2_mod: R2::Modulus,
    q3_mod: R3L::Modulus,
    q4_mod: R4::Modulus,
    repack: RepackFn,
    cascade: CascadeFn,
) -> Result<ModSwitchedCiphertext<N2, R3, R4>, ViaError>
where
    RepackFn: Fn(&[RLWECiphertext<N1, R2>], &K) -> RLWECiphertext<N1, R2>,
    CascadeFn:
        Fn(&MLWECiphertext<N1, 1, R1::Projected<1>>, &K::Eval, u64) -> RLWECiphertext<N1, R1>,
{
    const {
        assert!(N1 >= N3, "answer_batch: N1 must be >= N3");
        assert!(
            N1.is_multiple_of(N3),
            "answer_batch: N1 must be divisible by N3"
        );
        assert!(N3 <= N2, "answer_batch: N3 must be <= N2");
        assert!(T > 0, "answer_batch: T must be > 0");
        // The record-fit invariant (compile-time twin of `ViaBPublicParams::_CHECK`).
        assert!(
            T * N3 <= N2,
            "answer_batch: T * N3 must be <= N2 (record-fit invariant)"
        );
    }

    let _span = tracing::debug_span!("answer_batch", t = T, batch_len = batch.len()).entered();

    if batch.len() != T {
        return Err(ViaError::DimMismatch(
            "answer_batch: batch.len() must equal T",
        ));
    }

    // --- Steps 1–6 × T: the variant-common prefix at record degree N3 -------
    // `&cascade` (a `&F: Fn`) is reused across the T calls; each inner span is
    // `answer_through_crot`'s own, nested under `answer_batch`.
    let mut rotated: Vec<RLWECiphertext<N1, R2>> = Vec::with_capacity(T);
    for query in &batch.queries {
        let ct = answer_through_crot::<N1, N2, N3, R1, R2, R3, K, L_QUERY, L_CK, L_RSK, D, _>(
            query,
            pp,
            prepared_db,
            prepared_keys,
            q1_mod,
            q2_mod,
            &cascade,
        )?;
        rotated.push(ct);
    }

    // --- Repack_{N2}: the one new VIA-B step, reusing the cascade key --
    let cascade_key: &K = &pp.query_comp_key.lwe_to_rlwe_key;
    let repacked = tracing::debug_span!("step_repack").in_scope(|| repack(&rotated, cascade_key));

    // --- Step 7: RespComp once on the repacked ciphertext --------------------
    let answer = tracing::debug_span!("step7_resp_comp").in_scope(|| {
        resp_comp::<N1, N2, R2, R3L, R3, R4, L_RSK, D>(
            &repacked,
            &prepared_keys.rsk,
            q3_mod,
            q4_mod,
            pp.params.gadget_base_rsk,
        )
    });

    Ok(answer)
}
