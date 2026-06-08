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
use via_primitives::algebra::ring::RingPoly;
use via_primitives::encryption::MLWECiphertext;
use via_primitives::encryption::types::{ModSwitchedCiphertext, RGSWCiphertext, RLWECiphertext};
use via_primitives::gates::{CRotDir, cmux_tree, crot, dmux_tree, mod_switch_rgsw};
use via_primitives::switching::mod_switch::mod_switch_sym;
use via_protocol::{CompressedQuery, PublicParams, ViaError};
use zeroize::Zeroize;

use crate::first_dim::first_dim;
use crate::query_decomp::query_decomp;
use crate::resp_comp::resp_comp;

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
/// Toy-param closure (the full cascade→…→ring-switch chain) is the job of the
/// Task-31 e2e gate; paper-scale closure rides the P2 SPIKE budget.
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
    R1: RingPoly<N1>,
    R2: RingPoly<N1>,
    R3L: RingPoly<N1, Projected<N2> = R3>,
    R3: RingPoly<N2, Modulus = R3L::Modulus>,
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
    let d = N1 / N2;
    let num_crot = if d > 1 {
        d.trailing_zeros() as usize
    } else {
        0
    }; // log₂ d
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
    let b_rsk = params.gadget_base_rsk; // ring-switch @ q3

    // --- Step 1: QueryDecomp — LWEs → 3 RGSW groups @ q1 ------------------
    let dq = query_decomp::<N1, R1, K, L_QUERY, L_CK, _>(
        &query.ciphertexts,
        &pp.query_comp_key,
        num_dmux,
        num_cmux,
        num_crot,
        ck_base,
        ck_base,
        cascade,
    );

    // --- Step 2: DMux @ q1 — trivial RLWE(Δ·1) demuxed to I slots ---------
    let mut delta_coeffs = [0u128; N1];
    delta_coeffs[0] = params.delta();
    let delta_poly = R1::from_u128_coeffs(q1_mod, &delta_coeffs);
    let trivial = RLWECiphertext::trivial(q1_mod, &delta_poly);

    let zero_q1 = RLWECiphertext::new(R1::zero(q1_mod), R1::zero(q1_mod));
    let mut dmux_out: Vec<RLWECiphertext<N1, R1>> = vec![zero_q1; num_rows];
    dmux_tree(&dq.dmux_bits, trivial, &mut dmux_out, b1, b1);

    // --- Step 3: ModSwitch q1 → q2, ×I -----------------------------------
    let switched: Vec<RLWECiphertext<N1, R2>> = dmux_out
        .iter()
        .map(|ct| mod_switch_sym::<N1, R1, R2>(ct, q2_mod))
        .collect();

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

    // --- Step 4: FirstDim — Σ_i c_i · db[i][j] → J columns @ q2 -----------
    let mut fd_results = first_dim::<N1, R2>(&switched, &db_q2, q2_mod);

    // --- Step 5: CMux @ q2 (LSB-first) — select 1 column -----------------
    let cmux_q2: Vec<RGSWCiphertext<N1, R2, L_QUERY, L_QUERY>> = dq
        .cmux_bits
        .iter()
        .map(|rgsw| mod_switch_rgsw::<N1, R1, R2, L_QUERY, L_QUERY>(rgsw, q2_mod))
        .collect();
    let selected = cmux_tree(&cmux_q2, &mut fd_results, b2, b2);

    // --- Step 6: CRot @ q2 (SlotExtract, LSB-first) — rotate target slot → 0
    let crot_q2: Vec<RGSWCiphertext<N1, R2, L_QUERY, L_QUERY>> = dq
        .crot_bits
        .iter()
        .map(|rgsw| mod_switch_rgsw::<N1, R1, R2, L_QUERY, L_QUERY>(rgsw, q2_mod))
        .collect();
    let rotated = crot(CRotDir::SlotExtract, &crot_q2, selected, b2, b2);

    // --- Step 7: RespComp — paper-asymmetric q2 → q3 → n2 → q4 ------------
    let answer = resp_comp::<N1, N2, R2, R3L, R3, R4, L_RSK, D>(
        &rotated,
        &pp.ring_switch_key,
        q3_mod,
        q4_mod,
        b_rsk,
    );

    Ok(answer)
}

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
        let qck = QueryCompressionKey::new(alloc::boxed::Box::new(cascade_key), conv_key);

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
        let pp = PublicParams::new(qck, rsk, params, num_rows, num_cols, ck_base, L_CK);

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
}
