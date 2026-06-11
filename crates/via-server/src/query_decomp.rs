//! §6.1 QueryDecomp — decompress a compressed query into the RGSW groups that
//! drive DMux / CMux / CRot.
//!
//! For each of `total_bits = log₂I + log₂J + log₂d` query bits, the `L_QUERY`
//! LWE ciphertexts for that bit are converted:
//!   LWE ─cascade `lwe_to_rlwe_*`→ RLWE  (×L_QUERY, assembled into an RLev)
//!   then `rlwe_to_rgsw` → one RGSW @ q1.
//! The RGSW are sliced into the dmux / cmux / crot groups. All bits stay at q1;
//! the caller (`answer_one_query`) mod-switches the cmux/crot bits to q2.
//!
//! `paper:primitives/query_comp.py:263-353`, `via_c/server.py:136-142`

use alloc::vec::Vec;
use via_primitives::algebra::ring::RingPolyEval;
use via_primitives::encryption::MLWECiphertext;
use via_primitives::encryption::types::{RGSWCiphertext, RLWECiphertext, RLevCiphertext};
use via_primitives::gates::rlwe_to_rgsw;
use via_protocol::{DecompressedQuery, QueryCompressionKey};
use zeroize::Zeroize;

/// Decompress `total_bits · L_QUERY` LWE ciphertexts into a [`DecompressedQuery`]
/// of three RGSW groups (dmux / cmux / crot), all at `q1`.
///
/// # Type parameters
///
/// - `N1` — large ring degree; `R1` — its ring backend at `q1`.
/// - `K` — the cascade-key type (P2's `LweToRlweKey…`); reached via the `Box<K>`
///   in [`QueryCompressionKey`].
/// - `L_QUERY` — per-bit LWE count = output RGSW gadget depth.
/// - `L_CK` — conversion-key depth (`rlwe_to_rgsw_key`).
/// - `CascadeFn` — the bound `lwe_to_rlwe_*` for this ring (returns RLWE).
///
/// # Panics
///
/// if `lwe_query.len() != (num_dmux + num_cmux + num_crot) * L_QUERY`.
///
/// # Constant-time: No
///
/// Operates on the (public) compressed query and conversion keys; no
/// secret-dependent branching. Timing varies only on the public modulus/depth.
///
/// `paper:primitives/query_comp.py:263-353`
#[allow(clippy::too_many_arguments)]
pub fn query_decomp<
    const N1: usize,
    R1: RingPolyEval<N1>,
    K: Zeroize,
    const L_QUERY: usize,
    const L_CK: usize,
    CascadeFn,
>(
    lwe_query: &[MLWECiphertext<N1, 1, R1::Projected<1>>],
    comp_key: &QueryCompressionKey<K, N1, R1, L_CK>,
    num_dmux: usize,
    num_cmux: usize,
    num_crot: usize,
    cascade_base: u64,
    ck_base: u64,
    cascade: CascadeFn,
) -> DecompressedQuery<N1, R1, L_QUERY>
where
    CascadeFn: Fn(&MLWECiphertext<N1, 1, R1::Projected<1>>, &K, u64) -> RLWECiphertext<N1, R1>,
{
    let total_bits = num_dmux + num_cmux + num_crot;
    assert_eq!(
        lwe_query.len(),
        total_bits * L_QUERY,
        "query_decomp: expected {} LWEs, got {}",
        total_bits * L_QUERY,
        lwe_query.len(),
    );

    let mut rgsw_bits: Vec<RGSWCiphertext<N1, R1, L_QUERY, L_QUERY>> =
        Vec::with_capacity(total_bits);

    for bit_idx in 0..total_bits {
        let lwe_levels = &lwe_query[bit_idx * L_QUERY..(bit_idx + 1) * L_QUERY];

        // LWE → RLWE via the cascade (P2 returns RLWE directly — no unwrap).
        let rlwe_levels: [RLWECiphertext<N1, R1>; L_QUERY] = core::array::from_fn(|i| {
            cascade(&lwe_levels[i], &comp_key.lwe_to_rlwe_key, cascade_base)
        });

        // Assemble the `m` RLev (RLWECiphertext is `Copy`, so reuse `rlwe_levels`).
        let m_rlev = RLevCiphertext::new(rlwe_levels);

        // RLWE → RGSW @ q1 using RLev_{S1}(S1²).
        let rgsw: RGSWCiphertext<N1, R1, L_QUERY, L_QUERY> =
            rlwe_to_rgsw(rlwe_levels, &comp_key.rlwe_to_rgsw_key, m_rlev, ck_base);

        rgsw_bits.push(rgsw);
    }

    DecompressedQuery::new(
        rgsw_bits[..num_dmux].to_vec(),
        rgsw_bits[num_dmux..num_dmux + num_cmux].to_vec(),
        rgsw_bits[num_dmux + num_cmux..].to_vec(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::zq::modulus::DynModulus;
    use via_primitives::conversion::{
        LweToRlweKeyN8, encrypt_lwe, gen_lwe_to_rlwe_key_n8, lwe_to_rlwe_n8,
    };
    use via_primitives::encryption::types::SecretKey;
    use via_primitives::gates::gen_rlwe_to_rgsw_key;
    use via_primitives::sampling::distribution::Distribution;
    use via_primitives::sampling::prg::Shake256Prg;

    // Toy: N1=8, L_QUERY=2, L_CK=6, q=65537, p=16, base=2.
    const N1: usize = 8;
    const L_QUERY: usize = 2;
    const L_CK: usize = 6;
    const BASE: u64 = 2;
    type R8 = Poly<N1, DynModulus, Coefficient>;

    /// 2 dmux + 2 cmux + 1 crot = 5 bits × 2 levels = 10 LWEs →
    /// DecompressedQuery{dmux:2, cmux:2, crot:1}.
    #[test]
    fn query_decomp_shape_correct() {
        let q = DynModulus::new(65537);
        let mut prg = Shake256Prg::new(b"qd-shape-sk");
        let sk = SecretKey::<N1, R8>::keygen(q, Distribution::Ternary, &mut prg);

        let cascade_key: LweToRlweKeyN8<DynModulus, L_CK> =
            gen_lwe_to_rlwe_key_n8(&sk, BASE, Distribution::Ternary, &mut prg);
        let conv_key =
            gen_rlwe_to_rgsw_key::<N1, R8, L_CK>(&sk, BASE, Distribution::Ternary, &mut prg);
        let comp_key = QueryCompressionKey::new(
            alloc::boxed::Box::new(cascade_key),
            alloc::boxed::Box::new(conv_key),
        );

        // 10 trivial-zero LWEs (shape only — decrypt-correctness is the e2e test).
        let zero_lwe = encrypt_lwe(&sk, 0u64, 16, Distribution::Ternary, &mut prg);
        let lwe_query = alloc::vec![zero_lwe; 5 * L_QUERY];

        let dq = query_decomp::<N1, R8, _, L_QUERY, L_CK, _>(
            &lwe_query,
            &comp_key,
            2, // num_dmux
            2, // num_cmux
            1, // num_crot
            BASE,
            BASE,
            lwe_to_rlwe_n8,
        );
        assert_eq!(dq.dmux_bits.len(), 2);
        assert_eq!(dq.cmux_bits.len(), 2);
        assert_eq!(dq.crot_bits.len(), 1);
    }

    /// Length-mismatch panics with a clear message.
    #[test]
    #[should_panic(expected = "expected 10 LWEs, got 9")]
    fn query_decomp_panics_on_length_mismatch() {
        let q = DynModulus::new(65537);
        let mut prg = Shake256Prg::new(b"qd-len-sk");
        let sk = SecretKey::<N1, R8>::keygen(q, Distribution::Ternary, &mut prg);
        let cascade_key: LweToRlweKeyN8<DynModulus, L_CK> =
            gen_lwe_to_rlwe_key_n8(&sk, BASE, Distribution::Ternary, &mut prg);
        let conv_key =
            gen_rlwe_to_rgsw_key::<N1, R8, L_CK>(&sk, BASE, Distribution::Ternary, &mut prg);
        let comp_key = QueryCompressionKey::new(
            alloc::boxed::Box::new(cascade_key),
            alloc::boxed::Box::new(conv_key),
        );
        let zero_lwe = encrypt_lwe(&sk, 0u64, 16, Distribution::Ternary, &mut prg);
        let lwe_query = alloc::vec![zero_lwe; 9]; // not 10

        let _ = query_decomp::<N1, R8, _, L_QUERY, L_CK, _>(
            &lwe_query,
            &comp_key,
            2,
            2,
            1,
            BASE,
            BASE,
            lwe_to_rlwe_n8,
        );
    }
}
