//! Query wire types: [`CompressedQuery`] (LWE form) and [`DecompressedQuery`]
//! (RGSW form, grouped by Answer-pipeline role).

use alloc::vec::Vec;
use core::fmt;

use via_primitives::algebra::ring::RingPoly;
use via_primitives::encryption::{MLWECiphertext, RGSWCiphertext};
use zeroize::Zeroize;

// ---------------------------------------------------------------------------
// CompressedQuery
// ---------------------------------------------------------------------------

/// A compressed PIR query: $\ell_\mathrm{query} \cdot (\log I + \log J + \log d)$
/// LWE ciphertexts, each encrypting one index-bit gadget-decomposition level at
/// modulus $q_1$.
///
/// The length is dynamic (determined by the database layout at query-generation
/// time), so this is the first protocol type that cannot be a fixed-size array.
/// Each element is an `MLWECiphertext<RANK, N, R>`; in LWE form `RANK = n1`,
/// `N = 1` (`via-client` builds these via `encrypt_lwe_raw`).
///
/// Length mismatch against the server's expectation is reported as
/// [`ViaError::QueryLengthMismatch`](crate::ViaError::QueryLengthMismatch).
pub struct CompressedQuery<const RANK: usize, const N: usize, R: RingPoly<N>> {
    /// The LWE ciphertexts encoding the compressed query.
    pub ciphertexts: Vec<MLWECiphertext<RANK, N, R>>,
}

impl<const RANK: usize, const N: usize, R: RingPoly<N>> CompressedQuery<RANK, N, R> {
    /// Construct a `CompressedQuery` from a pre-built ciphertext vector.
    #[inline]
    pub fn new(ciphertexts: Vec<MLWECiphertext<RANK, N, R>>) -> Self {
        Self { ciphertexts }
    }

    /// Number of LWE ciphertexts.
    #[inline]
    pub fn len(&self) -> usize {
        self.ciphertexts.len()
    }

    /// Returns `true` if the query carries no ciphertexts.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.ciphertexts.is_empty()
    }
}

impl<const RANK: usize, const N: usize, R: RingPoly<N>> Zeroize for CompressedQuery<RANK, N, R> {
    fn zeroize(&mut self) {
        for ct in &mut self.ciphertexts {
            ct.zeroize();
        }
    }
}

impl<const RANK: usize, const N: usize, R: RingPoly<N>> Drop for CompressedQuery<RANK, N, R> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

/// `Debug` omits ciphertext data; prints only shape.
impl<const RANK: usize, const N: usize, R: RingPoly<N>> fmt::Debug for CompressedQuery<RANK, N, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompressedQuery")
            .field("RANK", &RANK)
            .field("N", &N)
            .field("len", &self.ciphertexts.len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// DecompressedQuery
// ---------------------------------------------------------------------------

/// A decompressed PIR query: RGSW ciphertexts grouped by Answer-pipeline role.
///
/// Every group's RGSW has the **same** gadget length `L_QUERY` (both halves) —
/// query compression builds all bits identically at `gadget_depth_1` levels, then
/// slices them into the three groups. The DMux tree uses all `L_QUERY` rows; the
/// CMux/CRot trees decompose into the first `gadget_depth_2 ≤ L_QUERY` rows.
///
/// # Group sizes
///
/// - `dmux_bits.len() == log₂(I)` — DMux control, MSB-first.
/// - `cmux_bits.len() == log₂(J)` — CMux selection, LSB-first.
/// - `crot_bits.len() == log₂(d) == log₂(n1/n2)` — CRot, LSB-first.
pub struct DecompressedQuery<const N: usize, R: RingPoly<N>, const L_QUERY: usize> {
    /// RGSW ciphertexts for DMux control ($\log_2 I$ bits, MSB-first).
    pub dmux_bits: Vec<RGSWCiphertext<N, R, L_QUERY, L_QUERY>>,
    /// RGSW ciphertexts for CMux selection ($\log_2 J$ bits, LSB-first).
    pub cmux_bits: Vec<RGSWCiphertext<N, R, L_QUERY, L_QUERY>>,
    /// RGSW ciphertexts for CRot ($\log_2 d$ bits, LSB-first).
    pub crot_bits: Vec<RGSWCiphertext<N, R, L_QUERY, L_QUERY>>,
}

impl<const N: usize, R: RingPoly<N>, const L_QUERY: usize> DecompressedQuery<N, R, L_QUERY> {
    /// Construct a `DecompressedQuery` from its three RGSW groups.
    #[inline]
    pub fn new(
        dmux_bits: Vec<RGSWCiphertext<N, R, L_QUERY, L_QUERY>>,
        cmux_bits: Vec<RGSWCiphertext<N, R, L_QUERY, L_QUERY>>,
        crot_bits: Vec<RGSWCiphertext<N, R, L_QUERY, L_QUERY>>,
    ) -> Self {
        Self {
            dmux_bits,
            cmux_bits,
            crot_bits,
        }
    }
}

impl<const N: usize, R: RingPoly<N>, const L_QUERY: usize> Zeroize
    for DecompressedQuery<N, R, L_QUERY>
{
    fn zeroize(&mut self) {
        for ct in &mut self.dmux_bits {
            ct.zeroize();
        }
        for ct in &mut self.cmux_bits {
            ct.zeroize();
        }
        for ct in &mut self.crot_bits {
            ct.zeroize();
        }
    }
}

impl<const N: usize, R: RingPoly<N>, const L_QUERY: usize> Drop
    for DecompressedQuery<N, R, L_QUERY>
{
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl<const N: usize, R: RingPoly<N>, const L_QUERY: usize> fmt::Debug
    for DecompressedQuery<N, R, L_QUERY>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DecompressedQuery")
            .field("N", &N)
            .field("L_QUERY", &L_QUERY)
            .field("dmux_bits.len", &self.dmux_bits.len())
            .field("cmux_bits.len", &self.cmux_bits.len())
            .field("crot_bits.len", &self.crot_bits.len())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// BatchedQuery (VIA-B)
// ---------------------------------------------------------------------------

/// A VIA-B batch query: the `T` independently-compressed VIA-C queries produced
/// by `Client::batch_query`. Length is the batch size `T`; the server runs the
/// VIA-C answer prefix on each, then a single `Repack_{n2}` + `RespComp`.
#[cfg(feature = "via-b")]
pub struct BatchedQuery<const RANK: usize, const N: usize, R: RingPoly<N>> {
    /// The `T` per-index compressed queries.
    pub queries: Vec<CompressedQuery<RANK, N, R>>,
}

#[cfg(feature = "via-b")]
impl<const RANK: usize, const N: usize, R: RingPoly<N>> BatchedQuery<RANK, N, R> {
    /// Construct a `BatchedQuery` from its `T` per-index compressed queries.
    #[inline]
    pub fn new(queries: Vec<CompressedQuery<RANK, N, R>>) -> Self {
        Self { queries }
    }

    /// Batch size `T` (number of queries).
    #[inline]
    pub fn len(&self) -> usize {
        self.queries.len()
    }

    /// Returns `true` if the batch carries no queries.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.queries.is_empty()
    }
}

#[cfg(feature = "via-b")]
impl<const RANK: usize, const N: usize, R: RingPoly<N>> Zeroize for BatchedQuery<RANK, N, R> {
    fn zeroize(&mut self) {
        for q in &mut self.queries {
            q.zeroize();
        }
    }
}

#[cfg(feature = "via-b")]
impl<const RANK: usize, const N: usize, R: RingPoly<N>> Drop for BatchedQuery<RANK, N, R> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

#[cfg(feature = "via-b")]
impl<const RANK: usize, const N: usize, R: RingPoly<N>> fmt::Debug for BatchedQuery<RANK, N, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("BatchedQuery")
            .field("RANK", &RANK)
            .field("N", &N)
            .field("T", &self.queries.len())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::zq::modulus::paper::ViaCQ2;
    use via_primitives::encryption::types::{RLWECiphertext, RLevCiphertext};

    type R = Poly<8, ViaCQ2, Coefficient>;

    fn zero_rlwe() -> RLWECiphertext<8, R> {
        let m = ViaCQ2::default();
        let z = <R as RingPoly<8>>::zero(m);
        RLWECiphertext::new(z, z)
    }

    fn zero_rlev<const L: usize>() -> RLevCiphertext<8, R, L> {
        RLevCiphertext::new([zero_rlwe(); L])
    }

    fn zero_rgsw<const L: usize>() -> RGSWCiphertext<8, R, L, L> {
        RGSWCiphertext::new(zero_rlev::<L>(), zero_rlev::<L>())
    }

    #[test]
    fn compressed_query_construct_and_len() {
        // RLWE-form stand-in (RANK=1, N=8) is enough to exercise the Vec wrapper.
        let z = zero_rlwe();
        let mlwe = MLWECiphertext::<1, 8, R>::new([z.mask], z.body);
        let cq = CompressedQuery::new(alloc::vec![mlwe; 3]);
        assert_eq!(cq.len(), 3);
        assert!(!cq.is_empty());
    }

    #[test]
    fn decompressed_query_construct() {
        let rgsw = zero_rgsw::<2>();
        let dq = DecompressedQuery::new(
            alloc::vec![rgsw; 2], // log2(I) = 2
            alloc::vec![rgsw; 4], // log2(J) = 4
            alloc::vec![rgsw; 2], // log2(d) = 2
        );
        assert_eq!(dq.dmux_bits.len(), 2);
        assert_eq!(dq.cmux_bits.len(), 4);
        assert_eq!(dq.crot_bits.len(), 2);
    }

    #[test]
    fn compressed_query_debug_redacts_data() {
        let z = zero_rlwe();
        let mlwe = MLWECiphertext::<1, 8, R>::new([z.mask], z.body);
        let cq = CompressedQuery::new(alloc::vec![mlwe]);
        let dbg = alloc::format!("{cq:?}");
        assert!(dbg.contains("CompressedQuery"));
        assert!(dbg.contains("len"));
    }

    #[test]
    #[cfg(feature = "via-b")]
    fn batched_query_construct_and_len() {
        let z = zero_rlwe();
        let mlwe = MLWECiphertext::<1, 8, R>::new([z.mask], z.body);
        let cq = CompressedQuery::new(alloc::vec![mlwe]);
        let bq = BatchedQuery::new(alloc::vec![cq]);
        assert_eq!(bq.len(), 1);
        assert!(!bq.is_empty());
    }
}
