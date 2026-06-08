//! §6.3 Database encoding — [`setup_db`].
//!
//! Packs `d = N1 / N_REC` records per `I×J` matrix cell via the interleaved ring
//! embedding [`RingPoly::embed_at`] at slots `0..d-1`: record `k·I·J + i·J + j`
//! (0-indexed) lands at slot `k` of cell `[i][j]`. This `(idx → (i,j,k))` layout
//! is a hard protocol contract with the client's `decompose_index`.
//!
//! `N_REC`-generic: VIA-C uses `N_REC = n2`; VIA-B a smaller `N_REC = n3`. The
//! records are degree-`N_REC` polynomials over the **plaintext** modulus `p`;
//! the encoded cells are degree-`N1` over the same `p` (FirstDim lifts `p→q2`
//! per query — see `via-server::answer`).
//!
//! `paper:via_c/params.py:157-208`

use alloc::vec::Vec;
use via_primitives::algebra::ring::RingPoly;

/// Encode a flat record database into an `I × J` matrix of polynomials in
/// `R_{N1, p}`, packing `d = N1 / N_REC` records per cell.
///
/// # Layout
///
/// Cell `[i][j]` = `Σ_{k∈[d]} embed_at(records[k·I·J + i·J + j], slot = k)`.
/// Records past the end of `records` are treated as zero (zero-padding).
///
/// # Type parameters
///
/// - `N1` — large-ring (server) degree.
/// - `N_REC` — record degree (`n2` for VIA-C); `d = N1 / N_REC` records per cell.
/// - `R1` — cell ring `R_{N1, p}` (the `Embedded<N1>` of `RRec`).
/// - `RRec` — record ring `R_{N_REC, p}` with `RRec::Embedded<N1> = R1`.
///
/// # Panics
///
/// Compile-time (`_CHECK`): `N1 < N_REC` or `N1 % N_REC != 0`.
///
/// `paper:via_c/params.py:157-208`
pub fn setup_db<
    const N1: usize,
    const N_REC: usize,
    R1: RingPoly<N1>,
    RRec: RingPoly<N_REC, Modulus = R1::Modulus, Embedded<N1> = R1>,
>(
    records: &[RRec],
    num_rows: usize,
    num_cols: usize,
    p_mod: R1::Modulus,
) -> Vec<Vec<R1>> {
    const {
        assert!(N1 >= N_REC, "setup_db: N1 must be >= N_REC");
        assert!(
            N1.is_multiple_of(N_REC),
            "setup_db: N1 must be divisible by N_REC"
        );
    }
    let d = N1 / N_REC;
    let zero_rec = RRec::zero(p_mod);
    let mut matrix: Vec<Vec<R1>> = Vec::with_capacity(num_rows);
    for i in 0..num_rows {
        let mut row: Vec<R1> = Vec::with_capacity(num_cols);
        for j in 0..num_cols {
            let mut cell = R1::zero(p_mod);
            for k in 0..d {
                let flat_idx = k * num_rows * num_cols + i * num_cols + j;
                let rec: &RRec = records.get(flat_idx).unwrap_or(&zero_rec);
                // embed_at slot k: coeff c of `rec` → coeff d·c + k of `cell`.
                cell += rec.embed_at::<N1>(k);
            }
            row.push(cell);
        }
        matrix.push(row);
    }
    matrix
}

#[cfg(test)]
mod tests {
    use super::*;
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::zq::modulus::PowerOfTwoModulus;

    // Toy: N1=4, N_REC=2, d=2, p=2^4=16.
    type P2 = Poly<2, PowerOfTwoModulus<4>, Coefficient>;
    type P4 = Poly<4, PowerOfTwoModulus<4>, Coefficient>;

    fn rec(coeffs: [u64; 2]) -> P2 {
        P2::new(PowerOfTwoModulus::<4>, coeffs)
    }

    #[test]
    fn setup_db_shape_correct() {
        let records: Vec<P2> = (0u64..8).map(|k| rec([k, 0])).collect();
        let db = setup_db::<4, 2, P4, P2>(&records, 2, 2, PowerOfTwoModulus::<4>);
        assert_eq!(db.len(), 2);
        assert_eq!(db[0].len(), 2);
        assert_eq!(db[1].len(), 2);
    }

    #[test]
    fn setup_db_zero_records_produces_zero_cells() {
        let db = setup_db::<4, 2, P4, P2>(&[], 1, 1, PowerOfTwoModulus::<4>);
        let mut coeffs = [0u128; 4];
        db[0][0].to_u128_coeffs(&mut coeffs);
        assert_eq!(coeffs, [0; 4]);
    }

    /// Slot packing: record at `flat_idx=0` (k=0) lands at coeffs `d·c+0`;
    /// record at `flat_idx=4` (k=1) at coeffs `d·c+1`.
    #[test]
    fn setup_db_slot_packing_parity() {
        // N1=4, N_REC=2, d=2, I=2, J=2.
        let mut records: Vec<P2> = alloc::vec![rec([0, 0]); 8];
        records[0] = rec([3, 5]); // k=0,i=0,j=0 → slot 0: coeff 0→0, coeff 1→2.
        records[4] = rec([7, 11]); // k=1,i=0,j=0 → slot 1: coeff 0→1, coeff 1→3.
        let db = setup_db::<4, 2, P4, P2>(&records, 2, 2, PowerOfTwoModulus::<4>);
        let mut c = [0u128; 4];
        db[0][0].to_u128_coeffs(&mut c);
        // slot 0: c[0]=3, c[2]=5; slot 1: c[1]=7, c[3]=11 (all < p=16, no wrap).
        assert_eq!(c, [3, 7, 5, 11]);
    }
}
