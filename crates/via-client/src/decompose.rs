//! Index decomposition and query-bit extraction for VIA-C `QueryComp`.
//!
//! A flat database index addresses one of `d·I·J` records under the layout
//! `index = γ·(I·J) + α·J + β` (slot `γ`, row `α`, column `β`) — the same
//! `flat_idx = k·I·J + i·J + j` the server's `setup_db` packs against. The
//! three coordinates drive the three gate groups:
//!
//! - `α` (row) → **DMux**, bits emitted **MSB-first**;
//! - `β` (column) → **CMux**, bits emitted **LSB-first**;
//! - `γ` (slot) → **CRot**, bits emitted **LSB-first**.
//!
//! Swapping an MSB-first / LSB-first order is a *silent* logic error (same bit
//! count, different content → the server selects the wrong record), so the
//! orderings are pinned by exhaustive reconstructability tests below and, later,
//! by the cross-language KAT.

use alloc::vec::Vec;

/// Decompose a flat database index into `(α, β, γ)` = `(row, column, slot)`.
///
/// Inverts `index = γ·(I·J) + α·J + β`:
/// - `γ = index / (I·J)` — rotation slot (CRot),
/// - `α = (index % (I·J)) / J` — row (DMux),
/// - `β = index % J` — column (CMux).
pub fn decompose_index(index: usize, num_rows: usize, num_cols: usize) -> (usize, usize, usize) {
    let group = num_rows * num_cols;
    let gamma = index / group;
    let alpha = (index % group) / num_cols;
    let beta = index % num_cols;
    (alpha, beta, gamma)
}

/// DMux control bits for `α`, **MSB-first** (`bits[0]` is the most significant,
/// the first dmux split). Length `num_dmux = log₂ I`.
pub fn dmux_bits(alpha: usize, num_dmux: usize) -> Vec<u8> {
    (0..num_dmux)
        .map(|i| ((alpha >> (num_dmux - 1 - i)) & 1) as u8)
        .collect()
}

/// CMux select bits for `β`, **LSB-first** (`bits[i]` controls tree depth `i`).
/// Length `num_cmux = log₂ J`.
pub fn cmux_bits(beta: usize, num_cmux: usize) -> Vec<u8> {
    (0..num_cmux).map(|i| ((beta >> i) & 1) as u8).collect()
}

/// CRot rotation bits for `γ`, **LSB-first** (`bits[i]` drives a `2ⁱ`-slot
/// rotation, matching `crot`'s bit-index convention). Length `num_crot = log₂ d`.
pub fn crot_bits(gamma: usize, num_crot: usize) -> Vec<u8> {
    (0..num_crot).map(|i| ((gamma >> i) & 1) as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    // --- decompose_index -------------------------------------------------

    /// I=4, J=4, d=1: index 6 = 0·16 + 1·4 + 2 → (α,β,γ) = (1, 2, 0).
    #[test]
    fn decompose_i4_j4_known_index() {
        assert_eq!(decompose_index(6, 4, 4), (1, 2, 0));
    }

    /// index 0 → origin.
    #[test]
    fn decompose_zero_is_origin() {
        assert_eq!(decompose_index(0, 4, 8), (0, 0, 0));
    }

    /// I=4, J=8, d=4: index 42 = 1·32 + 1·8 + 2 → (α,β,γ) = (1, 2, 1).
    #[test]
    fn decompose_i4_j8_d4_index42() {
        assert_eq!(decompose_index(42, 4, 8), (1, 2, 1));
    }

    /// Roundtrip: `γ·(I·J) + α·J + β` reconstructs every index in range.
    #[test]
    fn decompose_roundtrip_several() {
        let (i, j) = (4usize, 8usize);
        for idx in 0..(2 * i * j) {
            let (alpha, beta, gamma) = decompose_index(idx, i, j);
            assert_eq!(gamma * (i * j) + alpha * j + beta, idx, "roundtrip {idx}");
        }
    }

    // --- bit extraction (the silent-failure surface) ---------------------

    /// α=3 (0b11), width 4: MSB-first → [0,0,1,1].
    #[test]
    fn dmux_bits_alpha3_width4_msb_first() {
        assert_eq!(dmux_bits(3, 4), vec![0, 0, 1, 1]);
    }

    /// α=4 (0b100), width 3: MSB-first → [1,0,0].
    #[test]
    fn dmux_bits_alpha4_width3() {
        assert_eq!(dmux_bits(4, 3), vec![1, 0, 0]);
    }

    /// β=6 (0b110), width 3: LSB-first → [0,1,1].
    #[test]
    fn cmux_bits_beta6_width3_lsb_first() {
        assert_eq!(cmux_bits(6, 3), vec![0, 1, 1]);
    }

    /// γ=5 (0b101), width 3: LSB-first → [1,0,1].
    #[test]
    fn crot_bits_gamma5_width3_lsb_first() {
        assert_eq!(crot_bits(5, 3), vec![1, 0, 1]);
    }

    /// Width 0 → empty (e.g. d=1 ⇒ no CRot bits).
    #[test]
    fn empty_bits_when_width_zero() {
        assert_eq!(dmux_bits(7, 0), Vec::<u8>::new());
        assert_eq!(cmux_bits(7, 0), Vec::<u8>::new());
        assert_eq!(crot_bits(7, 0), Vec::<u8>::new());
    }

    /// `dmux_bits` reconstructs α when read MSB-first — pins the ordering.
    #[test]
    fn dmux_bits_reconstruct_alpha_msb_first() {
        for alpha in 0usize..16 {
            let bits = dmux_bits(alpha, 4);
            let got: usize = bits
                .iter()
                .enumerate()
                .map(|(i, &b)| (b as usize) << (3 - i))
                .sum();
            assert_eq!(got, alpha, "α={alpha}");
        }
    }

    /// `cmux_bits` reconstructs β when read LSB-first — pins the ordering.
    #[test]
    fn cmux_bits_reconstruct_beta_lsb_first() {
        for beta in 0usize..16 {
            let bits = cmux_bits(beta, 4);
            let got: usize = bits
                .iter()
                .enumerate()
                .map(|(i, &b)| (b as usize) << i)
                .sum();
            assert_eq!(got, beta, "β={beta}");
        }
    }
}
