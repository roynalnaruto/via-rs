//! FirstDim — the plaintext×ciphertext inner product (Answer step 4).
//!
//! For each column `j ∈ [J]`:  `c_j = Σ_{i∈[I]} c_i · db[i][j]`, where `c_i` is
//! the mod-switched RLWE of row `i` (already at `q2`) and `db[i][j]` is the
//! database cell. The cells are supplied **pre-transformed** as a [`PreparedDb`]
//! (lifted `p→q2` and forward-NTT'd to evaluation form once at setup), so this
//! step does no per-query transform of the database.
//!
//! The product is an **evaluation-form multiply-accumulate** over
//! [`RingPolyEval`]: transform the `I` selection ciphertexts to eval form once,
//! then for each column accumulate `db_eval[i][j] · sel_eval[i]` pointwise (the
//! same seed-from-first shape as
//! [`RLevCiphertext::gadget_product`](via_primitives::encryption::RLevCiphertext)).
//! For NTT-friendly `q2` this is `O(N)` pointwise muls (the speed-up); for the
//! non-NTT fallback (`DynModulus`/`PowerOfTwoModulus`) `Eval = Self`, the
//! transforms are the identity, and the pointwise `Mul` degenerates to the
//! existing schoolbook `negacyclic_mul_slice` — same result, same cost.
//!
//! The columns are independent, so under `feature = "alloc"` (where `std` is
//! linked) the `J`-loop is split across `available_parallelism()` worker
//! threads via [`std::thread::scope`] (no extra dependency); the `no_std` build
//! keeps the serial loop. Results are identical and assembled in column order.

use alloc::vec::Vec;
use via_primitives::algebra::ring::{RingPoly, RingPolyEval};
use via_primitives::encryption::types::RLWECiphertext;

use crate::prepared_db::PreparedDb;

/// Compute the `J` FirstDim output ciphertexts at `q2`.
///
/// # Arguments
///
/// - `switched` — the `I` RLWE ciphertexts after `mod_switch_sym` q1→q2.
/// - `prepared` — the `I×J` database, pre-transformed to eval form at `q2`
///   ([`PreparedDb::from_encoded`]).
///
/// # Panics
///
/// - `switched.len() != prepared.num_rows()` (I mismatch).
/// - `prepared.num_cols() == 0` (J must be > 0).
///
/// # Noise
///
/// Each `c_i · db[i][j]` scales the ciphertext noise by `‖db[i][j]‖` (cell
/// coefficients are in `[0,p)`); the `I` products sum, so the output noise is
/// roughly `I·p·` the input noise. The downstream `mod_switch`/`ring_switch`
/// budget (RespComp) must absorb it. (The eval form is exact, so the noise is
/// identical to the schoolbook computation.)
///
/// # Parallelism (GPU)
///
/// Each column `j` is independent — the `j`-loop and the inner pointwise MAC map
/// onto a 2-D grid `(thread_j, thread_coeff)`. The per-coefficient kernel
/// boundary already exists in `via-primitives` (the eval-form `Mul`, ultimately
/// `negacyclic_mul_slice` for the schoolbook backing); a batched GPU FirstDim
/// would wrap this whole function. The CPU path is sequential.
///
/// # Constant-time: No
///
/// Operates on RLWE-uniform ciphertext and public database coefficients; no
/// secret data is branched on.
pub fn first_dim<const N1: usize, R2>(
    switched: &[RLWECiphertext<N1, R2>],
    prepared: &PreparedDb<N1, R2>,
) -> Vec<RLWECiphertext<N1, R2>>
where
    R2: RingPoly<N1> + RingPolyEval<N1> + Send + Sync,
    R2::Eval: Send + Sync,
{
    let (i_len, j_len) = (prepared.num_rows(), prepared.num_cols());
    assert_eq!(
        switched.len(),
        i_len,
        "first_dim: I mismatch: switched={}, prepared rows={i_len}",
        switched.len()
    );
    assert!(j_len > 0, "first_dim: J must be > 0");

    // Transform the I selection ciphertexts to eval form ONCE (reused across all J).
    let sel: Vec<(R2::Eval, R2::Eval)> = switched
        .iter()
        .map(|ct| (R2::to_eval(ct.mask), R2::to_eval(ct.body)))
        .collect();

    // One output column: MAC the I cells against the selection, pointwise in
    // eval form. Seed from i=0 (I ≥ 1 by the asserts above), then MAC i=1..I.
    let column = |j: usize| -> RLWECiphertext<N1, R2> {
        let mut acc_mask = prepared.cell(0, j) * sel[0].0;
        let mut acc_body = prepared.cell(0, j) * sel[0].1;
        for (i, &(sel_mask, sel_body)) in sel.iter().enumerate().skip(1) {
            acc_mask += prepared.cell(i, j) * sel_mask;
            acc_body += prepared.cell(i, j) * sel_body;
        }
        RLWECiphertext::new(R2::from_eval(acc_mask), R2::from_eval(acc_body))
    };

    // Columns are independent — fan the J-loop across worker threads under
    // `alloc` (= std linked); fall back to a serial loop for `no_std`.
    #[cfg(feature = "alloc")]
    {
        let workers = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1)
            .min(j_len);
        if workers > 1 {
            let column = &column;
            let chunk = j_len.div_ceil(workers);
            // Workers don't inherit the caller's (large) stack, and the eval-form
            // ring ops put sizeable temporaries on the stack at paper/secure n —
            // give each a generous stack (overridable via VIA_FIRSTDIM_STACK_MB).
            let stack = std::env::var("VIA_FIRSTDIM_STACK_MB")
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
                .unwrap_or(64)
                << 20;
            let mut results: Vec<RLWECiphertext<N1, R2>> = Vec::with_capacity(j_len);
            std::thread::scope(|s| {
                let handles: Vec<_> = (0..j_len)
                    .step_by(chunk)
                    .map(|lo| {
                        let hi = (lo + chunk).min(j_len);
                        std::thread::Builder::new()
                            .stack_size(stack)
                            .spawn_scoped(s, move || (lo..hi).map(column).collect::<Vec<_>>())
                            .expect("spawn first_dim worker")
                    })
                    .collect();
                for h in handles {
                    results.extend(h.join().expect("first_dim worker thread panicked"));
                }
            });
            return results;
        }
    }
    (0..j_len).map(column).collect()
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
        let prepared = PreparedDb::<N, Rq<N>>::from_encoded(&db, q);
        let results = first_dim::<N, Rq<N>>(&[ct], &prepared);
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

        let prepared = PreparedDb::<N, Rq<N>>::from_encoded(&db, q);
        let results = first_dim::<N, Rq<N>>(&[ct_enc, zero_ct], &prepared);
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
        let prepared = PreparedDb::<N, Rq<N>>::from_encoded(&db, q);
        let results = first_dim::<N, Rq<N>>(&[ct; 3], &prepared);
        assert_eq!(results.len(), 5);
    }
}
