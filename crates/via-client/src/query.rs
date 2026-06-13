//! `Client::query` — compress a flat database index into gadget-scaled LWEs.
//!
//! # PRG consumption order (KAT contract)
//!
//! For a fixed `prg`, `query` emits exactly
//! `L_QUERY · (log₂I + log₂J + log₂d)` LWE ciphertexts in this order:
//!
//! ```text
//! bits = dmux_bits(α, log₂I)   // MSB-first
//!      ++ cmux_bits(β, log₂J)  // LSB-first
//!      ++ crot_bits(γ, log₂d)  // LSB-first
//! for b in bits:                       // bit-major (outer)
//!     for i in 0..L_QUERY:             // gadget-level (inner)
//!         encrypt_lwe_raw(sk1, b·g[i]) // draws n1 mask scalars, then 1 error
//! ```
//!
//! Reversing the outer/inner loop, swapping a bit ordering, or swapping the
//! mask/error draws inside `encrypt_lwe_raw` silently breaks byte-parity with
//! the KAT vectors.

use alloc::vec::Vec;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::conversion::encrypt_lwe_raw;
use via_primitives::conversion::mlwe_ops::LweDot;
use via_primitives::encryption::MLWECiphertext;
use via_primitives::encryption::gadget_vector_values;
use via_primitives::encryption::types::SecretKey;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_protocol::CompressedQuery;

use crate::decompose::{cmux_bits, crot_bits, decompose_index, dmux_bits};

/// `⌈log₂ n⌉` (0 for `n ≤ 1`).
fn ceil_log2(n: usize) -> usize {
    if n <= 1 {
        0
    } else {
        (usize::BITS - (n - 1).leading_zeros()) as usize
    }
}

/// Encrypt one bit group: for each bit, for each gadget level, push an LWE of
/// `b·g[i] mod q1`. Bit-major / level-minor (the PRG-order contract).
fn push_bit_group<const N1: usize, R1, const L_QUERY: usize>(
    bits: &[u8],
    g: &[u128; L_QUERY],
    sk1: &SecretKey<N1, R1>,
    q1_val: u128,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
    out: &mut Vec<MLWECiphertext<N1, 1, R1::Projected<1>>>,
) where
    R1: RingPoly<N1> + LweDot<N1>,
{
    for &b in bits {
        for &gi in g.iter() {
            let msg = if b == 0 { 0u128 } else { gi % q1_val };
            out.push(encrypt_lwe_raw(sk1, msg, error_dist, prg));
        }
    }
}

/// Decompose `index` and encrypt each query bit at each gadget level into an
/// LWE, assembling a [`CompressedQuery`] of `L_QUERY · total_bits` ciphertexts.
///
/// The gadget vector `g[i] = ⌈q1 / Bⁱ⁺¹⌉` scales each bit so the server's
/// cascade + `rlwe_to_rgsw` reconstruct `RGSW_{S1}(b)`. **DMux** bits use base
/// `dmux_base` (`gadget_base_1`); **CMux** and **CRot** bits use `cmux_base`
/// (`gadget_base_2`) — these differ at paper params (55879 vs 81). The gadget
/// base must match the base the server applies to that RGSW's external product
/// (DMux @ q1 with b1; CMux/CRot @ q2 with b2, the q1→q2 mod-switch preserving
/// the base). `q1` is read from `sk1`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_compressed_query<const N1: usize, R1, const L_QUERY: usize>(
    index: usize,
    sk1: &SecretKey<N1, R1>,
    num_rows: usize,
    num_cols: usize,
    d: usize,
    dmux_base: u64,
    cmux_base: u64,
    error_dist: Distribution,
    prg: &mut Shake256Prg,
) -> CompressedQuery<N1, 1, R1::Projected<1>>
where
    R1: RingPoly<N1> + LweDot<N1>,
{
    let (alpha, beta, gamma) = decompose_index(index, num_rows, num_cols);

    let q1_mod = RingPoly::modulus(sk1.poly());
    let q1_val = <R1 as RingPoly<N1>>::modulus_value(q1_mod);
    let g_dmux = gadget_vector_values::<N1, R1, L_QUERY>(q1_mod, dmux_base);
    let g_cmux = gadget_vector_values::<N1, R1, L_QUERY>(q1_mod, cmux_base);

    let total = (ceil_log2(num_rows) + ceil_log2(num_cols) + ceil_log2(d)) * L_QUERY;
    let mut cts: Vec<MLWECiphertext<N1, 1, R1::Projected<1>>> = Vec::with_capacity(total);

    // Order is [DMux | CMux | CRot] (query_decomp slices them in that order).
    // DMux MSB-first @ b1; CMux + CRot LSB-first @ b2.
    push_bit_group(
        &dmux_bits(alpha, ceil_log2(num_rows)),
        &g_dmux,
        sk1,
        q1_val,
        error_dist,
        prg,
        &mut cts,
    );
    push_bit_group(
        &cmux_bits(beta, ceil_log2(num_cols)),
        &g_cmux,
        sk1,
        q1_val,
        error_dist,
        prg,
        &mut cts,
    );
    push_bit_group(
        &crot_bits(gamma, ceil_log2(d)),
        &g_cmux,
        sk1,
        q1_val,
        error_dist,
        prg,
        &mut cts,
    );

    CompressedQuery::new(cts)
}
