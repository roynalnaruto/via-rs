//! Pre-transformed database for FirstDim — the `p→q2` lift + forward NTT, once.
//!
//! [`setup_db`](crate::setup_db()) emits the database in **coefficient form at the
//! plaintext modulus `p`** (the `I×J` packing is a protocol contract with the client's
//! `decompose_index`). [`PreparedDb`] is the *crypto* representation FirstDim consumes:
//! each cell lifted `p→q2` and forward-NTT'd to **evaluation form**, computed **once at
//! setup** so [`first_dim`](crate::first_dim) needs no per-query transform of the DB.
//!
//! This is the storage half of the FirstDim eval-form optimisation; the multiply half is
//! [`first_dim`](crate::first_dim). For non-NTT moduli (`DynModulus`/`PowerOfTwoModulus` —
//! toy/test params) `R2::Eval = R2` and the transform is the identity, so a `PreparedDb`
//! degenerates to the coefficient-form matrix and FirstDim runs schoolbook (same result).

use alloc::vec::Vec;
use via_primitives::algebra::ring::{RingPoly, RingPolyEval};

/// An `I×J` database matrix stored in **evaluation form at `q2`**.
///
/// Each cell is `to_eval(lift_{p→q2}(db[i][j]))` — the forward negacyclic NTT of the
/// `q2`-lifted coefficients (lift = coefficient reinterpretation, coeffs in
/// `[0,p) ⊂ [0,q2)`, no rescale). Built once via [`PreparedDb::from_encoded`] and consumed
/// by [`first_dim`](crate::first_dim) with a pointwise multiply-accumulate.
///
/// # Memory
///
/// Identical footprint to the `p`-encoded matrix it replaces: one `R2::Eval` per cell
/// (`[u64; N1]` for single-prime `q2`), the same bytes the coefficient form occupies, just
/// transformed. No extra storage versus the prior `Server.encoded_db`.
pub struct PreparedDb<const N1: usize, R2>
where
    R2: RingPoly<N1> + RingPolyEval<N1>,
{
    /// `[I][J]` eval-form cells.
    cells: Vec<Vec<R2::Eval>>,
    num_rows: usize,
    num_cols: usize,
}

impl<const N1: usize, R2> PreparedDb<N1, R2>
where
    R2: RingPoly<N1> + RingPolyEval<N1>,
{
    /// Pre-transform a `p`-encoded `I×J` matrix (from [`setup_db`](crate::setup_db())):
    /// lift each cell `p→q2` (coefficient reinterpretation, no rescale) and forward-NTT it.
    ///
    /// `Rp` is the `p`-encoded cell ring. One-time, offline; cost ≈ `I·J` forward NTTs.
    /// (For non-NTT `q2` the NTT is the identity and this is just the lift.)
    ///
    /// # Panics
    ///
    /// If `encoded_db`'s rows are ragged (any row length `!= J`).
    ///
    /// # Constant-time: No
    ///
    /// The database is public; coefficient arithmetic is data-independent apart from the
    /// public moduli.
    pub fn from_encoded<Rp: RingPoly<N1>>(encoded_db: &[Vec<Rp>], q2_mod: R2::Modulus) -> Self {
        let num_rows = encoded_db.len();
        let num_cols = encoded_db.first().map_or(0, |r| r.len());
        for (i, row) in encoded_db.iter().enumerate() {
            assert_eq!(
                row.len(),
                num_cols,
                "PreparedDb: row {i} has J={} != {num_cols}",
                row.len()
            );
        }
        let cells = encoded_db
            .iter()
            .map(|row| {
                row.iter()
                    .map(|cell| {
                        // Lift p→q2 (coeff reinterpretation), then forward NTT.
                        let mut coeffs = [0u128; N1];
                        cell.to_u128_coeffs(&mut coeffs);
                        R2::to_eval(R2::from_u128_coeffs(q2_mod, &coeffs))
                    })
                    .collect()
            })
            .collect();
        Self {
            cells,
            num_rows,
            num_cols,
        }
    }

    /// The first-dimension extent `I` (number of rows).
    #[inline]
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The second-dimension extent `J` (number of columns).
    #[inline]
    pub fn num_cols(&self) -> usize {
        self.num_cols
    }

    /// The eval-form cell `[i][j]` (copied out — `R2::Eval: Copy`).
    #[inline]
    pub(crate) fn cell(&self, i: usize, j: usize) -> R2::Eval {
        self.cells[i][j]
    }
}
