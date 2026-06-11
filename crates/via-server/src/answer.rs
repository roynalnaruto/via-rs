//! §6 — the 7-step VIA-C answer pipeline ([`answer_one_query`]).
//!
//! The free function is the variant-agnostic engine: it is generic over the
//! ring backend, so the toy single-prime `Poly` and the paper RNS `PolyRns`
//! both instantiate it without duplication. `VIA-C`'s server delegates here;
//! VIA-B (Layer 7) will wrap it directly.
//!
//! ## Pipeline (paper §6, `via_c/server.py:103-236`)
//!
//! | # | Step           | Primitive          | Ring move          |
//! |---|----------------|--------------------|--------------------|
//! | 1 | QueryDecomp    | [`query_decomp`]   | LWE → RGSW @ q1     |
//! | 2 | DMux           | `dmux_tree`        | trivial(Δ) → I×@q1  |
//! | 3 | ModSwitch ×I   | `mod_switch_sym`   | q1 → q2            |
//! | 4 | FirstDim       | [`first_dim`]      | I×@q2 → J×@q2       |
//! | 5 | CMux           | `cmux_tree`        | J×@q2 → 1×@q2       |
//! | 6 | CRot           | `crot`             | 1×@q2 → 1×@q2       |
//! | 7 | RespComp       | [`resp_comp`]      | q2 → q3 → n2 → q4   |
//!
//! **Bit-ordering invariant** (paper `mux.rs`/`rotate.rs`): DMux control bits
//! are **MSB-first**, CMux select bits and CRot rotation bits are **LSB-first**.
//! The client builds the query to honour this; the server just forwards the
//! three RGSW groups [`query_decomp`] sliced out.
//!
//! **Modulus routing.** DMux runs at `q1`; its `I` outputs are mod-switched to
//! `q2` before FirstDim. The CMux/CRot RGSW bits (still at `q1` from
//! QueryDecomp) are `mod_switch_rgsw`'d to `q2` before steps 5/6. The four
//! ring moduli cannot be reconstructed from [`PIRParams`](via_protocol::PIRParams) generically (a
//! `u128`/`u64` has no generic path back to `R::Modulus`), so they are passed
//! explicitly — matching every other composite primitive in this crate.
//!
//! `paper:via_c/server.py:103-236`

use alloc::vec;
use alloc::vec::Vec;
use via_primitives::algebra::ring::{RingPoly, RingPolyEval};
use via_primitives::encryption::MLWECiphertext;
use via_primitives::encryption::types::{ModSwitchedCiphertext, RGSWCiphertext, RLWECiphertext};
use via_primitives::gates::{CRotDir, cmux_tree, crot, dmux_tree, mod_switch_rgsw};
use via_primitives::switching::mod_switch::mod_switch_sym;
use via_protocol::{CompressedQuery, PublicParams, ViaError};
use zeroize::Zeroize;

use crate::first_dim::first_dim;
use crate::query_decomp::query_decomp;
use crate::resp_comp::resp_comp;

/// Steps 1–6 of the VIA-C answer pipeline (QueryDecomp → DMux → ModSwitch →
/// FirstDim → CMux → CRot), extracted as the **variant-common** prefix returning
/// the CRot output `rotated: RLWECiphertext<N1, R2>` @ q2.
///
/// [`answer_one_query`] appends step 7 ([`resp_comp`]) to recover the VIA-C
/// answer; VIA-B's batch answer (Layer 7) instead calls this `T` times — once per
/// batched query — collects the `T` `rotated`s, **repacks** them into one
/// ciphertext, and then runs RespComp exactly once. The seam is precisely here.
///
/// # `N_REC` — record-degree generalization
///
/// `num_crot = log₂(N1 / N_REC)`. At `N_REC = N2` (VIA-C) this is exactly
/// `log₂(N1/N2) = log₂ d`, so the extraction is a behavioural no-op on VIA-C.
/// VIA-B passes the finer `N_REC = N3 ≤ N2`, requesting `log₂(N1/N3)` CRot bits
/// (more rotation resolution for the smaller records).
///
/// # Arguments
///
/// As [`answer_one_query`] minus the RespComp-only `q3_mod`/`q4_mod`: `query`,
/// `pp`, `encoded_db`, `q1_mod`, `q2_mod`, `cascade`. (`R3L`/`R4` likewise drop —
/// they belong to step 7.)
///
/// # Errors
///
/// Same guards as [`answer_one_query`]: [`ViaError::DimMismatch`]
/// (non-power-of-two dims / row-count mismatch) and
/// [`ViaError::QueryLengthMismatch`] (the LWE count must be
/// `(log₂I + log₂J + log₂(N1/N_REC)) · L_QUERY`).
///
/// `paper:via_c/server.py:103-230` (steps 1–6)
#[allow(
    non_camel_case_types,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
pub fn answer_through_crot<
    const N1: usize,
    const N2: usize,
    const N_REC: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R2: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    Rp: RingPoly<N1>,
    K: Zeroize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
    CascadeFn,
>(
    query: &CompressedQuery<N1, 1, R1::Projected<1>>,
    pp: &PublicParams<K, N1, N2, R1, R3, L_QUERY, L_CK, L_RSK, D>,
    encoded_db: &[Vec<Rp>],
    q1_mod: R1::Modulus,
    q2_mod: R2::Modulus,
    cascade: CascadeFn,
) -> Result<RLWECiphertext<N1, R2>, ViaError>
where
    CascadeFn: Fn(&MLWECiphertext<N1, 1, R1::Projected<1>>, &K, u64) -> RLWECiphertext<N1, R1>,
{
    const {
        assert!(N1 >= N_REC, "answer_through_crot: N1 must be >= N_REC");
        assert!(
            N1.is_multiple_of(N_REC),
            "answer_through_crot: N1 must be divisible by N_REC"
        );
    }

    // Parent span; steps 1–6 nest under it (held for the whole call, dropped on
    // every return path including the early error guards).
    let _span = tracing::debug_span!(
        "answer_through_crot",
        num_rows = pp.num_rows,
        num_cols = pp.num_cols
    )
    .entered();

    let params = &pp.params;
    let num_rows = pp.num_rows;
    let num_cols = pp.num_cols;

    // --- Shape guards -----------------------------------------------------
    if !num_rows.is_power_of_two() || !num_cols.is_power_of_two() {
        return Err(ViaError::DimMismatch(
            "num_rows and num_cols must be powers of two",
        ));
    }
    let num_dmux = num_rows.trailing_zeros() as usize; // log₂ I
    let num_cmux = num_cols.trailing_zeros() as usize; // log₂ J
    // R1: num_crot = log₂(N1/N_REC); collapses to log₂(N1/N2) = log₂ d at N_REC=N2.
    let d3 = N1 / N_REC;
    let num_crot = if d3 > 1 {
        d3.trailing_zeros() as usize
    } else {
        0
    };
    if encoded_db.len() != num_rows {
        return Err(ViaError::DimMismatch(
            "encoded_db row count must equal num_rows",
        ));
    }
    let expected_lwes = (num_dmux + num_cmux + num_crot) * L_QUERY;
    if query.ciphertexts.len() != expected_lwes {
        return Err(ViaError::QueryLengthMismatch {
            expected: expected_lwes,
            got: query.ciphertexts.len(),
        });
    }

    // Cascade + conversion-key gadget base (Override 7): the cascade rides the
    // conversion-key gadget, so both legs of `query_decomp` use `pp.ck_base`.
    let ck_base = pp.ck_base;
    let b1 = params.gadget_base_1; // DMux @ q1
    let b2 = params.gadget_base_2; // CMux / CRot @ q2

    // --- Step 1: QueryDecomp — LWEs → 3 RGSW groups @ q1 ------------------
    let dq = tracing::debug_span!("step1_query_decomp").in_scope(|| {
        query_decomp::<N1, R1, K, L_QUERY, L_CK, _>(
            &query.ciphertexts,
            &pp.query_comp_key,
            num_dmux,
            num_cmux,
            num_crot,
            ck_base,
            ck_base,
            cascade,
        )
    });

    // --- Step 2: DMux @ q1 — trivial RLWE(Δ·1) demuxed to I slots ---------
    let dmux_out = tracing::debug_span!("step2_dmux").in_scope(|| {
        let mut delta_coeffs = [0u128; N1];
        delta_coeffs[0] = params.delta();
        let delta_poly = R1::from_u128_coeffs(q1_mod, &delta_coeffs);
        let trivial = RLWECiphertext::trivial(q1_mod, &delta_poly);

        let zero_q1 = RLWECiphertext::new(R1::zero(q1_mod), R1::zero(q1_mod));
        let mut out: Vec<RLWECiphertext<N1, R1>> = vec![zero_q1; num_rows];
        dmux_tree(&dq.dmux_bits, trivial, &mut out, b1, b1);
        out
    });

    // --- Step 3: ModSwitch q1 → q2, ×I -----------------------------------
    let switched = tracing::debug_span!("step3_mod_switch").in_scope(|| {
        dmux_out
            .iter()
            .map(|ct| mod_switch_sym::<N1, R1, R2>(ct, q2_mod))
            .collect::<Vec<RLWECiphertext<N1, R2>>>()
    });

    // --- Step 4: FirstDim — Σ_i c_i · db[i][j] → J columns @ q2 -----------
    let mut fd_results = tracing::debug_span!("step4_first_dim").in_scope(|| {
        // Lift the p-encoded DB to q2 (coefficient reinterpretation: values in
        // [0,p) ⊂ [0,q2), no rescale).
        let db_q2: Vec<Vec<R2>> = encoded_db
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        let mut c = [0u128; N1];
                        cell.to_u128_coeffs(&mut c);
                        R2::from_u128_coeffs(q2_mod, &c)
                    })
                    .collect()
            })
            .collect();
        first_dim::<N1, R2>(&switched, &db_q2, q2_mod)
    });

    // --- Step 5: CMux @ q2 (LSB-first) — select 1 column -----------------
    let selected = tracing::debug_span!("step5_cmux").in_scope(|| {
        let cmux_q2: Vec<RGSWCiphertext<N1, R2, L_QUERY, L_QUERY>> = dq
            .cmux_bits
            .iter()
            .map(|rgsw| mod_switch_rgsw::<N1, R1, R2, L_QUERY, L_QUERY>(rgsw, q2_mod))
            .collect();
        cmux_tree(&cmux_q2, &mut fd_results, b2, b2)
    });

    // --- Step 6: CRot @ q2 (SlotExtract, LSB-first) — rotate target slot → 0
    let rotated = tracing::debug_span!("step6_crot").in_scope(|| {
        let crot_q2: Vec<RGSWCiphertext<N1, R2, L_QUERY, L_QUERY>> = dq
            .crot_bits
            .iter()
            .map(|rgsw| mod_switch_rgsw::<N1, R1, R2, L_QUERY, L_QUERY>(rgsw, q2_mod))
            .collect();
        crot(CRotDir::SlotExtract, &crot_q2, selected, b2, b2)
    });

    Ok(rotated)
}

/// Run the 7-step VIA-C answer pipeline for one compressed query.
///
/// Returns the paper-asymmetric compressed answer
/// `ModSwitchedCiphertext<N2, R3, R4>` (mask @ q3, body @ q4). The caller
/// decrypts it with `SecretKey::decrypt_asymmetric(S2@q3, q3, q4, p)`. The
/// concrete paper instantiation wraps this into `via_protocol::CompressedAnswer`
/// at the wire boundary (that type is locked to the paper q3/q4 rings, so the
/// generic engine stays one rung below it).
///
/// # Type parameters
///
/// - `R1` @ q1 (n1) · `R2` @ q2 (n1) · `R3L` @ q3 (n1, the `mod_switch_sym`
///   intermediate, `Projected<N2> = R3`) · `R3` @ q3 (n2) · `R4` @ q4 (n2).
/// - `Rp` — the **p-encoded** database ring (n1); lifted to `R2` per query.
/// - `K` — the LWE→RLWE cascade-key type (heap-boxed in [`PublicParams`]).
/// - `L_QUERY` / `L_CK` / `L_RSK` — query-RGSW / conversion-key / ring-switch
///   gadget depths. `D = N1 / N2`.
///
/// # Arguments
///
/// `encoded_db` is the **`p`-encoded** `I×J` matrix from [`setup_db`](crate::setup_db())
/// (`Rp` cells); it is lifted `p → q2` per query (coefficients live in
/// `[0,p) ⊂ [0,q2)`, so the lift is a coefficient reinterpretation, no
/// rescale). The cascade & conversion-key gadget base is read from `pp.ck_base`
/// (paper Override 7 — the cascade rides the conversion-key gadget, not
/// `gadget_base_1`); the DMux / CMux+CRot / ring-switch bases come from
/// `pp.params.{gadget_base_1, gadget_base_2, gadget_base_rsk}`.
///
/// # Errors
///
/// - [`ViaError::DimMismatch`] if `num_rows`/`num_cols` are not powers of two,
///   or `encoded_db.len() != num_rows`.
/// - [`ViaError::QueryLengthMismatch`] if the LWE count is not
///   `(log₂I + log₂J + log₂d) · L_QUERY`.
///
/// # Noise
///
/// Toy-param closure (the full cascade→…→ring-switch chain) is verified by the
/// `e2e_toy` gate; paper-scale closure rides the P2 SPIKE budget.
///
/// # Parallelism (GPU)
///
/// Steps 4 (FirstDim) and 5 (CMux) carry the bulk of the work — the FirstDim
/// column MAC is the kernel boundary (via `RLWECiphertext::Mul`'s
/// `negacyclic_mul_slice`), CMux is a log-depth gate tree. Both are sequential
/// here; a batched GPU FirstDim would wrap [`first_dim`]. Each step is wrapped
/// in a `tracing::debug_span!` (`step1_query_decomp` … `step7_resp_comp`) for
/// per-step timing.
///
/// # Constant-time: No
///
/// The query ciphertexts and database are RLWE/RGSW-uniform; no secret data is
/// branched on. `%`/division timing varies only on the public moduli.
///
/// `paper:via_c/server.py:103-236`
#[allow(
    non_camel_case_types,
    clippy::too_many_arguments,
    clippy::type_complexity
)]
pub fn answer_one_query<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R2: RingPoly<N1> + RingPolyEval<N1>,
    R3L: RingPoly<N1, Projected<N2> = R3>,
    R3: RingPoly<N2, Modulus = R3L::Modulus> + RingPolyEval<N2>,
    R4: RingPoly<N2>,
    Rp: RingPoly<N1>,
    K: Zeroize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
    CascadeFn,
>(
    query: &CompressedQuery<N1, 1, R1::Projected<1>>,
    pp: &PublicParams<K, N1, N2, R1, R3, L_QUERY, L_CK, L_RSK, D>,
    encoded_db: &[Vec<Rp>],
    q1_mod: R1::Modulus,
    q2_mod: R2::Modulus,
    q3_mod: R3L::Modulus,
    q4_mod: R4::Modulus,
    cascade: CascadeFn,
) -> Result<ModSwitchedCiphertext<N2, R3, R4>, ViaError>
where
    CascadeFn: Fn(&MLWECiphertext<N1, 1, R1::Projected<1>>, &K, u64) -> RLWECiphertext<N1, R1>,
{
    // Parent span; the prefix's step spans nest under `answer_through_crot` and
    // step 7 nests directly here (held for the whole call, dropped on every
    // return path including the prefix's early error guards).
    let _span = tracing::debug_span!(
        "answer_one_query",
        num_rows = pp.num_rows,
        num_cols = pp.num_cols
    )
    .entered();

    // Steps 1–6 via the variant-common prefix; N_REC = N2 keeps VIA-C semantics
    // (num_crot = log₂(N1/N2) = log₂ d) exactly.
    let rotated = answer_through_crot::<N1, N2, N2, R1, R2, R3, Rp, K, L_QUERY, L_CK, L_RSK, D, _>(
        query, pp, encoded_db, q1_mod, q2_mod, cascade,
    )?;

    // --- Step 7: RespComp — paper-asymmetric q2 → q3 → n2 → q4 ------------
    let answer = tracing::debug_span!("step7_resp_comp").in_scope(|| {
        resp_comp::<N1, N2, R2, R3L, R3, R4, L_RSK, D>(
            &rotated,
            &pp.ring_switch_key,
            q3_mod,
            q4_mod,
            pp.params.gadget_base_rsk,
        )
    });

    Ok(answer)
}

/// A VIA-C server: public parameters + the pre-encoded database + the four ring
/// moduli, bundled so the caller issues `server.answer(&query, cascade)` instead
/// of the 9-argument [`answer_one_query`].
///
/// The engine is generic over the ring backend (toy single-prime `Poly` or
/// paper RNS `PolyRns`); [`Server::answer`] returns the raw paper-asymmetric
/// [`ModSwitchedCiphertext`]. The wire type `via_protocol::CompressedAnswer` is
/// **locked to the paper q3/q4 rings** (the VIA-C answer is always at paper
/// moduli), so the paper instantiation wraps the result —
/// `CompressedAnswer::new(server.answer(&q, cascade)?)` — at the wire boundary;
/// the generic `Server` stays one rung below it (and so is testable at the toy
/// rings). VIA-B (Layer 7) will likewise wrap [`answer_one_query`] directly.
///
/// The four moduli are stored (they cannot be reconstructed from
/// [`PIRParams`](via_protocol::PIRParams) generically); the cascade function is
/// supplied per `answer` call.
///
/// # Constant-time: No
///
/// Answering branches only on public data (query ciphertexts, the cleartext
/// database); see [`answer_one_query`].
pub struct Server<
    K: Zeroize,
    const N1: usize,
    const N2: usize,
    // Record degree: VIA-C packs `N_REC = N2`; VIA-B the finer `N_REC = N3 ≤ N2`
    // (more records per cell, more CRot bits). Gates only `setup_db`'s packing.
    const N_REC: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R2: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    R4: RingPoly<N2>,
    Rp: RingPoly<N1>,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> {
    public_params: PublicParams<K, N1, N2, R1, R3, L_QUERY, L_CK, L_RSK, D>,
    encoded_db: Vec<Vec<Rp>>,
    q1_mod: R1::Modulus,
    q2_mod: R2::Modulus,
    q3_mod: R3::Modulus,
    q4_mod: R4::Modulus,
}

impl<
    K: Zeroize,
    const N1: usize,
    const N2: usize,
    const N_REC: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R2: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    R4: RingPoly<N2>,
    Rp: RingPoly<N1>,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> Server<K, N1, N2, N_REC, R1, R2, R3, R4, Rp, L_QUERY, L_CK, L_RSK, D>
{
    /// Compile-time `N_REC` consistency: the record ring must be a non-trivial
    /// power-of-two divisor of `N1` that fits within the query-compression ring.
    ///
    /// # Panics (compile-time)
    ///
    /// `N_REC < 1`, `N1 % N_REC != 0`, or `N_REC > N2`.
    pub const _CHECK: () = {
        assert!(N_REC >= 1, "Server: N_REC must be >= 1");
        assert!(
            N1.is_multiple_of(N_REC),
            "Server: N1 must be divisible by N_REC"
        );
        assert!(
            N_REC <= N2,
            "Server: N_REC must be <= N2 (record ring fits within query-compression ring)"
        );
    };

    /// Build a server from raw `p`-encoded records and public parameters.
    ///
    /// Encodes the `I×J` database via [`setup_db`](crate::setup_db()) (packing
    /// `d3 = N1/N_REC` records per cell at modulus `p`); the per-query lift
    /// `p → q2` happens inside [`answer`](Server::answer). `RRec` is the record
    /// ring at `(p, n_rec)`; `p_mod` is its modulus.
    #[allow(clippy::too_many_arguments)]
    pub fn setup<RRec>(
        records: &[RRec],
        public_params: PublicParams<K, N1, N2, R1, R3, L_QUERY, L_CK, L_RSK, D>,
        q1_mod: R1::Modulus,
        q2_mod: R2::Modulus,
        q3_mod: R3::Modulus,
        q4_mod: R4::Modulus,
        p_mod: Rp::Modulus,
    ) -> Self
    where
        RRec: RingPoly<N_REC, Modulus = Rp::Modulus, Embedded<N1> = Rp>,
    {
        let () = Self::_CHECK;
        let encoded_db = crate::setup_db::setup_db::<N1, N_REC, Rp, RRec>(
            records,
            public_params.num_rows,
            public_params.num_cols,
            p_mod,
        );
        Self {
            public_params,
            encoded_db,
            q1_mod,
            q2_mod,
            q3_mod,
            q4_mod,
        }
    }

    /// Run the 7-step answer pipeline for one query (delegates to
    /// [`answer_one_query`]; see it for the error conditions).
    ///
    /// `R3L` is the `q3`-at-`n1` `mod_switch_sym` intermediate
    /// (`Projected<N2> = R3`); supply it at the call site (toy: the `q3` ring at
    /// `n1`; paper: `ViaCPolyQ3` at `n1`).
    pub fn answer<R3L, CascadeFn>(
        &self,
        query: &CompressedQuery<N1, 1, R1::Projected<1>>,
        cascade: CascadeFn,
    ) -> Result<ModSwitchedCiphertext<N2, R3, R4>, ViaError>
    where
        R3L: RingPoly<N1, Projected<N2> = R3, Modulus = R3::Modulus>,
        CascadeFn: Fn(&MLWECiphertext<N1, 1, R1::Projected<1>>, &K, u64) -> RLWECiphertext<N1, R1>,
    {
        answer_one_query::<N1, N2, R1, R2, R3L, R3, R4, Rp, K, L_QUERY, L_CK, L_RSK, D, CascadeFn>(
            query,
            &self.public_params,
            &self.encoded_db,
            self.q1_mod,
            self.q2_mod,
            self.q3_mod,
            self.q4_mod,
            cascade,
        )
    }

    /// VIA-B (M1): run the batch answer pipeline for `T` queries — delegates to
    /// [`answer_batch`](crate::batch::answer_batch) with the record degree fixed
    /// to the server's `N_REC` (= `N3` for a [`ViaBServer`]). `repack` and
    /// `cascade` are injected per call (same pattern as [`Server::answer`]'s
    /// `cascade`); see the free function for the pipeline and error conditions.
    ///
    /// `paper:via.pdf §4.5–4.7 (VIA-B answer)`
    #[cfg(feature = "via-b")]
    pub fn answer_batch<R3L, const T: usize, RepackFn, CascadeFn>(
        &self,
        batch: &via_protocol::BatchedQuery<N1, 1, R1::Projected<1>>,
        repack: RepackFn,
        cascade: CascadeFn,
    ) -> Result<ModSwitchedCiphertext<N2, R3, R4>, ViaError>
    where
        R3L: RingPoly<N1, Projected<N2> = R3, Modulus = R3::Modulus>,
        RepackFn: Fn(&[RLWECiphertext<N1, R2>], &K) -> RLWECiphertext<N1, R2>,
        CascadeFn: Fn(&MLWECiphertext<N1, 1, R1::Projected<1>>, &K, u64) -> RLWECiphertext<N1, R1>,
    {
        crate::batch::answer_batch::<
            N1,
            N2,
            N_REC,
            T,
            R1,
            R2,
            R3L,
            R3,
            R4,
            Rp,
            K,
            L_QUERY,
            L_CK,
            L_RSK,
            D,
            RepackFn,
            CascadeFn,
        >(
            batch,
            &self.public_params,
            &self.encoded_db,
            self.q1_mod,
            self.q2_mod,
            self.q3_mod,
            self.q4_mod,
            repack,
            cascade,
        )
    }
}

/// VIA-C server (M1): [`Server`] with the record degree fixed to `N_REC = N2`
/// (one record-ring per query-compression slot — the VIA-C packing). The clean
/// name for the VIA-C instantiation, hiding the `N_REC` slot the variant never
/// varies; `ViaCServer<K, N1, N2, …>` ≡ `Server<K, N1, N2, N2, …>`.
#[allow(type_alias_bounds)]
pub type ViaCServer<
    K: Zeroize,
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R2: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    R4: RingPoly<N2>,
    Rp: RingPoly<N1>,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> = Server<K, N1, N2, N2, R1, R2, R3, R4, Rp, L_QUERY, L_CK, L_RSK, D>;

/// VIA-B server (M1): [`Server`] with the record degree exposed as the finer
/// `N_REC = N3 ≤ N2` (the VIA-B record ring). Gated on `via-b`; VIA-B's batch
/// answer (Layer 7, P4) packs `T` of these finer records per query.
/// `ViaBServer<K, N1, N2, N3, …>` ≡ `Server<K, N1, N2, N3, …>`.
#[cfg(feature = "via-b")]
#[allow(type_alias_bounds)]
pub type ViaBServer<
    K: Zeroize,
    const N1: usize,
    const N2: usize,
    const N3: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R2: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    R4: RingPoly<N2>,
    Rp: RingPoly<N1>,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> = Server<K, N1, N2, N3, R1, R2, R3, R4, Rp, L_QUERY, L_CK, L_RSK, D>;

#[cfg(test)]
mod tests {
    use super::*;
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::zq::modulus::DynModulus;
    use via_primitives::conversion::{
        LweToRlweKeyN8, encrypt_lwe_raw, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8,
    };
    use via_primitives::encryption::types::SecretKey;
    use via_primitives::gates::gen_rlwe_to_rgsw_key;
    use via_primitives::sampling::distribution::Distribution;
    use via_primitives::sampling::prg::Shake256Prg;
    use via_primitives::switching::gen_rsk;
    use via_primitives::switching::rekey::rekey_secret_key;
    use via_protocol::{PIRParams, QueryCompressionKey};

    // Toy single-prime dims: N1=8, N2=4, d=D=2, I=J=2, L_QUERY=2, L_CK=6, L_RSK=8.
    const N1: usize = 8;
    const N2: usize = 4;
    const D: usize = 2;
    const L_QUERY: usize = 2;
    const L_CK: usize = 6;
    const L_RSK: usize = 8;
    type R8 = Poly<N1, DynModulus, Coefficient>;
    type R4 = Poly<N2, DynModulus, Coefficient>;
    type Cascade = LweToRlweKeyN8<DynModulus, L_CK>;

    /// Build a complete toy [`PublicParams`] + encoded DB at single-prime params.
    /// Descending moduli q1 > q2 > q3 > q4 with the cascade's proven base 8.
    #[allow(clippy::type_complexity)]
    fn toy_setup(
        num_rows: usize,
        num_cols: usize,
        records: &[R4],
        prg: &mut Shake256Prg,
    ) -> (
        PublicParams<Cascade, N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>,
        Vec<Vec<R8>>,
        SecretKey<N1, R8>,
        SecretKey<N2, R4>,
        DynModulus,
        DynModulus,
        DynModulus,
        DynModulus,
        DynModulus,
    ) {
        let q1 = DynModulus::new(65537);
        let q2 = DynModulus::new(1 << 14);
        let q3 = DynModulus::new(1 << 12);
        let q4 = DynModulus::new(1 << 10);
        let p = DynModulus::new(16);
        let ck_base = 8u64;
        let b_rsk = 8u64;

        let s1 = SecretKey::<N1, R8>::keygen(q1, Distribution::Ternary, prg);
        let s2 = SecretKey::<N2, R4>::keygen(q3, Distribution::Ternary, prg);

        // Query-compression key: cascade key (base 8, depth L_CK) + RLev(S1²).
        let cascade_key =
            gen_lwe_to_rlwe_key_n8::<_, L_CK>(&s1, ck_base, Distribution::Ternary, prg);
        let conv_key =
            gen_rlwe_to_rgsw_key::<N1, R8, L_CK>(&s1, ck_base, Distribution::Ternary, prg);
        let qck = QueryCompressionKey::new(
            alloc::boxed::Box::new(cascade_key),
            alloc::boxed::Box::new(conv_key),
        );

        // Ring-switch key: rekey S1 q1→q3, then RSK_{S1→S2} at q3.
        let s1_q3 = rekey_secret_key::<N1, R8, R8>(&s1, q3);
        let rsk =
            gen_rsk::<N1, N2, R8, R4, L_RSK, D>(&s1_q3, &s2, b_rsk, Distribution::Ternary, prg);

        let params = PIRParams::new(
            N1,
            N2,
            65537,
            1 << 14,
            1 << 12,
            1 << 10,
            16, // moduli
            256,
            L_QUERY, // gadget_base_1 (DMux @ q1), depth
            128,
            L_QUERY, // gadget_base_2 (CMux/CRot @ q2), depth
            b_rsk,
            L_RSK, // gadget_base_rsk, depth
            via_protocol::KeyDist::Ternary,
            via_protocol::KeyDist::Ternary,
            1,
            None,
            None,
            None,
            40,
        );
        let pp = PublicParams::new(
            qck,
            alloc::boxed::Box::new(rsk),
            params,
            num_rows,
            num_cols,
            ck_base,
            L_CK,
        );

        let db = crate::setup_db::setup_db::<N1, N2, R8, R4>(records, num_rows, num_cols, p);
        (pp, db, s1, s2, q1, q2, q3, q4, p)
    }

    /// Build a length-`n` all-zero query (gadget-level LWEs of bit 0).
    fn zero_query(
        s1: &SecretKey<N1, R8>,
        n: usize,
        prg: &mut Shake256Prg,
    ) -> CompressedQuery<N1, 1, <R8 as RingPoly<N1>>::Projected<1>> {
        let lwes = (0..n)
            .map(|_| encrypt_lwe_raw(s1, 0u128, Distribution::Ternary, prg))
            .collect();
        CompressedQuery::new(lwes)
    }

    /// Structural / wiring gate: build real keys, a real (index-0) query and a
    /// real DB, then run all 7 steps end-to-end. Asserts the 13-generic
    /// signature monomorphizes, every step's buffer sizes/types line up, and
    /// the pipeline returns a `ModSwitchedCiphertext<N2,…>` without panicking.
    /// (Decrypt-correctness — the noise-tuned q1≫q2≫q3≫q4 budget — is the
    /// Task-31 e2e gate.)
    #[test]
    fn answer_one_query_runs_full_pipeline_toy() {
        let mut prg = Shake256Prg::new(b"answer-pipeline-toy");
        // 8 records (d·I·J = 2·2·2) — record[0] non-trivial, rest zero.
        let mut records = alloc::vec![R4::zero(DynModulus::new(16)); 8];
        records[0] = Poly::new(DynModulus::new(16), [1, 2, 3, 4]);

        let (pp, db, s1, s2, q1, q2, q3, q4, p) = toy_setup(2, 2, &records, &mut prg);
        let query = zero_query(&s1, 3 * L_QUERY, &mut prg);

        let answer = answer_one_query::<
            N1,
            N2,
            R8,
            R8,
            R8,
            R4,
            R4,
            R8,
            Cascade,
            L_QUERY,
            L_CK,
            L_RSK,
            D,
            _,
        >(
            &query,
            &pp,
            &db,
            q1,
            q2,
            q3,
            q4,
            lwe_to_rlwe_n8::<DynModulus, L_CK>,
        )
        .expect("pipeline must produce an answer");

        // The recover path (decrypt_asymmetric under S2@q3) must also run.
        // Value is noise-dominated at these un-tuned moduli, so we assert shape
        // (an N2-degree plaintext), not equality — that is Task 31's job.
        let recovered: R4 = s2.decrypt_asymmetric(&answer, q3, q4, p);
        let _ = recovered.coeff(0);
    }

    /// Wrong LWE count → `QueryLengthMismatch` (caught before QueryDecomp).
    #[test]
    fn answer_one_query_rejects_wrong_query_length() {
        let mut prg = Shake256Prg::new(b"answer-bad-len");
        let records = alloc::vec![R4::zero(DynModulus::new(16)); 8];
        let (pp, db, s1, _s2, q1, q2, q3, q4, _p) = toy_setup(2, 2, &records, &mut prg);
        let query = zero_query(&s1, 5, &mut prg); // expected 3·2 = 6

        let err = answer_one_query::<
            N1,
            N2,
            R8,
            R8,
            R8,
            R4,
            R4,
            R8,
            Cascade,
            L_QUERY,
            L_CK,
            L_RSK,
            D,
            _,
        >(
            &query,
            &pp,
            &db,
            q1,
            q2,
            q3,
            q4,
            lwe_to_rlwe_n8::<DynModulus, L_CK>,
        )
        .unwrap_err();
        assert_eq!(
            err,
            ViaError::QueryLengthMismatch {
                expected: 6,
                got: 5
            }
        );
    }

    /// Non-power-of-two `num_rows` → `DimMismatch` (CMux/DMux trees need 2^k).
    #[test]
    fn answer_one_query_rejects_non_power_of_two_dims() {
        let mut prg = Shake256Prg::new(b"answer-bad-dims");
        let records = alloc::vec![R4::zero(DynModulus::new(16)); 12];
        // 3 rows × 2 cols — 3 is not a power of two.
        let (pp, db, s1, _s2, q1, q2, q3, q4, _p) = toy_setup(3, 2, &records, &mut prg);
        let query = zero_query(&s1, 6, &mut prg);

        let err = answer_one_query::<
            N1,
            N2,
            R8,
            R8,
            R8,
            R4,
            R4,
            R8,
            Cascade,
            L_QUERY,
            L_CK,
            L_RSK,
            D,
            _,
        >(
            &query,
            &pp,
            &db,
            q1,
            q2,
            q3,
            q4,
            lwe_to_rlwe_n8::<DynModulus, L_CK>,
        )
        .unwrap_err();
        assert!(matches!(err, ViaError::DimMismatch(_)));
    }

    /// `answer_through_crot` at `N_REC = N2` (VIA-C) returns the CRot-output RLWE
    /// directly — the prefix [`answer_one_query`] delegates to, and the entry
    /// point P4's `answer_batch` will call once per batched query.
    #[test]
    fn answer_through_crot_nrec_n2_returns_rlwe() {
        let mut prg = Shake256Prg::new(b"atc-nrec-n2");
        let mut records = alloc::vec![R4::zero(DynModulus::new(16)); 8];
        records[0] = Poly::new(DynModulus::new(16), [1, 2, 3, 4]);
        let (pp, db, s1, _s2, q1, q2, _q3, _q4, _p) = toy_setup(2, 2, &records, &mut prg);
        let query = zero_query(&s1, 3 * L_QUERY, &mut prg);

        let rotated =
            answer_through_crot::<N1, N2, N2, R8, R8, R4, R8, Cascade, L_QUERY, L_CK, L_RSK, D, _>(
                &query,
                &pp,
                &db,
                q1,
                q2,
                lwe_to_rlwe_n8::<DynModulus, L_CK>,
            )
            .expect("answer_through_crot must return Ok(RLWECiphertext)");
        // Returns an RLWE @ q2 over R8 (n1); shape only — value is the e2e gate's job.
        let _: RLWECiphertext<N1, R8> = rotated;
    }

    /// `N_REC` generalization (R1): `answer_through_crot` with `N_REC = 2 < N2 = 4`
    /// requests `num_crot = log₂(N1/N_REC) = log₂(8/2) = 2` CRot bits (vs VIA-C's
    /// 1), so it expects `(1+1+2)·L_QUERY` LWEs. Feeding the VIA-C-sized
    /// `3·L_QUERY` query surfaces `QueryLengthMismatch` — proving the bit count
    /// tracks `N_REC`, not `N2`.
    #[test]
    fn answer_through_crot_nrec_smaller_requests_more_crot_bits() {
        let mut prg = Shake256Prg::new(b"atc-nrec-smaller");
        let records = alloc::vec![R4::zero(DynModulus::new(16)); 8];
        let (pp, db, s1, _s2, q1, q2, _q3, _q4, _p) = toy_setup(2, 2, &records, &mut prg);
        let query = zero_query(&s1, 3 * L_QUERY, &mut prg);

        let err =
            answer_through_crot::<N1, N2, 2, R8, R8, R4, R8, Cascade, L_QUERY, L_CK, L_RSK, D, _>(
                &query,
                &pp,
                &db,
                q1,
                q2,
                lwe_to_rlwe_n8::<DynModulus, L_CK>,
            )
            .unwrap_err();

        assert_eq!(
            err,
            ViaError::QueryLengthMismatch {
                expected: (1 + 1 + 2) * L_QUERY,
                got: 3 * L_QUERY,
            },
        );
    }

    /// VIA-B `answer_batch` propagates `QueryLengthMismatch` from the inner
    /// `answer_through_crot` — the T-loop runs the prefix per query at `N_REC=N3`.
    /// At `N3=2` each inner query needs `(1+1+log₂(8/2))·L_QUERY = 8` LWEs; feeding
    /// empty inner queries surfaces the mismatch (the repack closure type-checks
    /// but is never reached). Full batch e2e is P5.
    #[cfg(feature = "via-b")]
    #[test]
    fn answer_batch_propagates_inner_query_length_mismatch() {
        use via_primitives::conversion::repack::{repack_keys_n8_t2_from_cascade, repack_n8_t2};
        use via_protocol::BatchedQuery;

        const N3: usize = 2;
        const T: usize = 2;
        let mut prg = Shake256Prg::new(b"answer-batch-len");
        let records = alloc::vec![R4::zero(DynModulus::new(16)); 8];
        let (pp, db, _s1, _s2, q1, q2, q3, q4, _p) = toy_setup(2, 2, &records, &mut prg);

        // T=2 inner queries, each empty (wrong length → QueryLengthMismatch).
        let batch = BatchedQuery::new(alloc::vec![
            CompressedQuery::new(alloc::vec![]),
            CompressedQuery::new(alloc::vec![]),
        ]);

        let err = crate::batch::answer_batch::<
            N1,
            N2,
            N3,
            T,
            R8,
            R8,
            R8,
            R4,
            R4,
            R8,
            Cascade,
            L_QUERY,
            L_CK,
            L_RSK,
            D,
            _,
            _,
        >(
            &batch,
            &pp,
            &db,
            q1,
            q2,
            q3,
            q4,
            |rotateds: &[RLWECiphertext<N1, R8>], k: &Cascade| {
                let arr: &[_; T] = rotateds.try_into().unwrap();
                repack_n8_t2(arr, &repack_keys_n8_t2_from_cascade(k), 8u64)
            },
            lwe_to_rlwe_n8::<DynModulus, L_CK>,
        )
        .unwrap_err();
        assert!(
            matches!(err, ViaError::QueryLengthMismatch { .. }),
            "answer_batch must propagate the inner QueryLengthMismatch; got {err:?}"
        );
    }
}
