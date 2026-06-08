//! §6.4 FirstDim — the plaintext×ciphertext inner product (Answer step 4).
//!
//! For each column `j ∈ [J]`:  `c_j = Σ_{i∈[I]} c_i · db[i][j]`, where `c_i` is
//! the mod-switched RLWE of row `i` (already at `q2`) and `db[i][j]` is the
//! encoded database cell **already lifted to `R_{N1,q2}`** (the lift `p→q2` is a
//! coefficient reinterpretation done by `answer_one_query` before this call —
//! the coefficients are in `[0,p) ⊂ [0,q2)`, so no rescale is needed).
//!
//! The `c_i · db[i][j]` product is `RLWECiphertext::Mul` — `(mask·db, body·db)`,
//! a **negacyclic** plaintext×ciphertext multiply whose hot loop is
//! `via_primitives`' `negacyclic_mul_slice` (the layer-0 GPU-portable kernel).
//! A future *batched* GPU FirstDim would wrap this whole function.
//!
//! `paper:via_c/server.py:174-186`

use alloc::vec::Vec;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::encryption::types::RLWECiphertext;

/// Compute the `J` FirstDim output ciphertexts at `q2`.
///
/// # Arguments
///
/// - `switched` — the `I` RLWE ciphertexts after `mod_switch_sym` q1→q2.
/// - `encoded_db` — the `I×J` database matrix, **lifted to `R_{N1,q2}`**.
/// - `q2_mod` — the `q2` modulus value (for the zero accumulator).
///
/// # Panics
///
/// - `switched.len() != encoded_db.len()` (I mismatch).
/// - `J == 0`, or any row's length `!= J`.
///
/// `paper:via_c/server.py:174-186`
pub fn first_dim<const N1: usize, Q2: RingPoly<N1>>(
    switched: &[RLWECiphertext<N1, Q2>],
    encoded_db: &[Vec<Q2>],
    q2_mod: Q2::Modulus,
) -> Vec<RLWECiphertext<N1, Q2>> {
    assert_eq!(
        switched.len(),
        encoded_db.len(),
        "first_dim: I mismatch: switched={}, encoded_db rows={}",
        switched.len(),
        encoded_db.len()
    );
    let num_cols = encoded_db.first().map_or(0, |r| r.len());
    assert!(num_cols > 0, "first_dim: J must be > 0");
    for (i, row) in encoded_db.iter().enumerate() {
        assert_eq!(
            row.len(),
            num_cols,
            "first_dim: encoded_db[{i}].len() = {} != J = {num_cols}",
            row.len()
        );
    }

    let zero_ct = RLWECiphertext::new(Q2::zero(q2_mod), Q2::zero(q2_mod));
    let mut results = Vec::with_capacity(num_cols);
    for j in 0..num_cols {
        let mut acc = zero_ct;
        for (ct, db_row) in switched.iter().zip(encoded_db.iter()) {
            // c_i · db[i][j]: negacyclic plaintext×ciphertext multiply.
            acc += *ct * db_row[j];
        }
        results.push(acc);
    }
    results
}

#[cfg(test)]
mod tests {
    use super::*;
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::zq::modulus::DynModulus;
    use via_primitives::encryption::encode;
    use via_primitives::encryption::types::SecretKey;
    use via_primitives::sampling::distribution::Distribution;
    use via_primitives::sampling::prg::Shake256Prg;

    type Rq<const N: usize> = Poly<N, DynModulus, Coefficient>;

    /// Zero DB → every output decrypts to 0.
    #[test]
    fn first_dim_zero_db_produces_zero_output() {
        const N: usize = 4;
        let q = DynModulus::new(65537);
        let p = DynModulus::new(16);
        let mut prg = Shake256Prg::new(b"fd-zero");
        let sk = SecretKey::<N, Rq<N>>::keygen(q, Distribution::Ternary, &mut prg);
        let msg: Rq<N> = Poly::new(p, [1, 0, 0, 0]);
        let ct = sk.encrypt(&encode(&msg, q), Distribution::Ternary, &mut prg);

        // I=1, J=2, db all-zero.
        let zero = Rq::<N>::zero(q);
        let db = alloc::vec![alloc::vec![zero, zero]];
        let results = first_dim::<N, Rq<N>>(&[ct], &db, q);
        assert_eq!(results.len(), 2);
        let zero_msg: Rq<N> = Poly::new(p, [0; 4]);
        for c in &results {
            let dec: Rq<N> = sk.decrypt(c, p);
            assert_eq!(dec, zero_msg);
        }
    }

    /// I=2, J=2: db = [[1, 0], [0, 1]] selects column 0 = RLWE(3), column 1 = RLWE(0).
    #[test]
    fn first_dim_accumulation_correctness() {
        const N: usize = 4;
        let q = DynModulus::new(65537);
        let p = DynModulus::new(16);
        let mut prg = Shake256Prg::new(b"fd-acc");
        let sk = SecretKey::<N, Rq<N>>::keygen(q, Distribution::Ternary, &mut prg);

        let msg: Rq<N> = Poly::new(p, [3, 0, 0, 0]);
        let ct_enc = sk.encrypt(&encode(&msg, q), Distribution::Ternary, &mut prg);
        let zero_msg: Rq<N> = Poly::new(p, [0; 4]);
        let zero_ct = sk.encrypt(&encode(&zero_msg, q), Distribution::Ternary, &mut prg);

        // db[0][0]=1, db[1][0]=0, db[0][1]=0, db[1][1]=1.
        let one = Rq::<N>::from_u128_coeffs(q, &[1, 0, 0, 0]);
        let zero = Rq::<N>::zero(q);
        let db = alloc::vec![alloc::vec![one, zero], alloc::vec![zero, one]];

        let results = first_dim::<N, Rq<N>>(&[ct_enc, zero_ct], &db, q);
        assert_eq!(results.len(), 2);
        // Column 0: 1·RLWE(3) + 0·RLWE(0) = RLWE(3).
        assert_eq!(sk.decrypt::<Rq<N>>(&results[0], p), msg);
        // Column 1: 0·RLWE(3) + 1·RLWE(0) = RLWE(0).
        assert_eq!(sk.decrypt::<Rq<N>>(&results[1], p), zero_msg);
    }

    /// Shape: I=3, J=5 → 5 outputs.
    #[test]
    fn first_dim_shape() {
        const N: usize = 4;
        let q = DynModulus::new(65537);
        let p = DynModulus::new(16);
        let mut prg = Shake256Prg::new(b"fd-shape");
        let sk = SecretKey::<N, Rq<N>>::keygen(q, Distribution::Ternary, &mut prg);
        let m: Rq<N> = Poly::new(p, [0; 4]);
        let ct = sk.encrypt(&encode(&m, q), Distribution::Ternary, &mut prg);
        let zero = Rq::<N>::zero(q);
        let db = alloc::vec![alloc::vec![zero; 5]; 3];
        let results = first_dim::<N, Rq<N>>(&[ct; 3], &db, q);
        assert_eq!(results.len(), 5);
    }

    /// Confirms FirstDim is a **polynomial (negacyclic)** multiply, not pointwise:
    /// with `db = X` and a message at coefficient 3, `3·X³ · X = 3·X⁴ ≡ −3` in
    /// `R_4 = Z[X]/(X⁴+1)` → coeff 0 = `p−3`. A pointwise multiply would give 0.
    #[test]
    fn first_dim_is_negacyclic_poly_mul() {
        const N: usize = 4;
        let q = DynModulus::new(65537);
        let p = DynModulus::new(16);
        let mut prg = Shake256Prg::new(b"fd-nega");
        let sk = SecretKey::<N, Rq<N>>::keygen(q, Distribution::Ternary, &mut prg);
        let msg: Rq<N> = Poly::new(p, [0, 0, 0, 3]); // 3·X³
        let ct = sk.encrypt(&encode(&msg, q), Distribution::Ternary, &mut prg);
        let x = Rq::<N>::from_u128_coeffs(q, &[0, 1, 0, 0]); // X
        let db = alloc::vec![alloc::vec![x]];
        let results = first_dim::<N, Rq<N>>(&[ct], &db, q);
        // 3X³·X = 3X⁴ ≡ −3 ⇒ coeff 0 = 16−3 = 13, rest 0.
        let expected: Rq<N> = Poly::new(p, [13, 0, 0, 0]);
        assert_eq!(sk.decrypt::<Rq<N>>(&results[0], p), expected);
    }
}
