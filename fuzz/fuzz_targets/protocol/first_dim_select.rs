//! Fuzz: §6.4 FirstDim one-hot selection.
//!
//! With a one-hot database column (`db[i][j] = 1` iff `i == t`, else `0`), the
//! plaintext×ciphertext MAC `Σ_i c_i·db[i][j]` collapses to `c_t`, so every one
//! of the `J` outputs must decrypt to row `t`'s message. Catches an accumulator
//! bug, a row/column transposition, or a non-negacyclic multiply.
//!
//! Run with `cargo +nightly fuzz run protocol_first_dim_select`.

#![no_main]

use arbitrary::{Arbitrary, Unstructured};
use libfuzzer_sys::fuzz_target;

use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::zq::modulus::DynModulus;
use via_primitives::encryption::SecretKey;
use via_primitives::encryption::rlwe::encode;
use via_primitives::sampling::{Distribution, Shake256Prg};
use via_server::first_dim;

const N: usize = 8;
type R = Poly<N, DynModulus, Coefficient>;

const Q: u64 = 8_380_417; // ≫ p, so a single product stays well inside Δ/2
const P: u64 = 16;

#[derive(Debug)]
struct Input {
    sk_seed: Vec<u8>,
    enc_seed: Vec<u8>,
    num_rows: usize,
    num_cols: usize,
    target: usize,
    values: Vec<u64>,
}

impl<'a> Arbitrary<'a> for Input {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let seed = |u: &mut Unstructured<'a>| -> arbitrary::Result<Vec<u8>> {
            let l = u.int_in_range::<usize>(1..=32)?;
            let mut s = vec![0u8; l];
            u.fill_buffer(&mut s)?;
            Ok(s)
        };
        let sk_seed = seed(u)?;
        let enc_seed = seed(u)?;
        let num_rows = u.int_in_range::<usize>(1..=4)?;
        let num_cols = u.int_in_range::<usize>(1..=4)?;
        let target = u.int_in_range::<usize>(0..=num_rows - 1)?;
        let mut values = vec![0u64; num_rows];
        for v in &mut values {
            *v = u.int_in_range::<u64>(0..=P - 1)?;
        }
        Ok(Input { sk_seed, enc_seed, num_rows, num_cols, target, values })
    }
}

fuzz_target!(|input: Input| {
    let q = DynModulus::new(Q);
    let p = DynModulus::new(P);
    let i = input.num_rows;
    let j = input.num_cols;
    let t = input.target;

    let mut sk_prg = Shake256Prg::new(&input.sk_seed);
    let sk = SecretKey::<N, R>::keygen(q, Distribution::Ternary, &mut sk_prg);
    let mut enc_prg = Shake256Prg::new(&input.enc_seed);

    // Encrypt one RLWE per row, message = constant poly [v_i, 0, …].
    let switched: Vec<_> = (0..i)
        .map(|row| {
            let mut coeffs = [0u64; N];
            coeffs[0] = input.values[row];
            let msg = R::new(p, coeffs);
            sk.encrypt(&encode::<N, R, R>(&msg, q), Distribution::Ternary, &mut enc_prg)
        })
        .collect();

    // One-hot db: row t selected, all others zero. Cells live in R_{n,q}.
    let one = R::new(q, {
        let mut c = [0u64; N];
        c[0] = 1;
        c
    });
    let zero = R::zero(q);
    let db: Vec<Vec<R>> = (0..i)
        .map(|row| (0..j).map(|_| if row == t { one } else { zero }).collect())
        .collect();

    let out = first_dim::<N, R>(&switched, &db, q);
    assert_eq!(out.len(), j, "first_dim must emit J ciphertexts");

    let mut expected = [0u64; N];
    expected[0] = input.values[t];
    let expected = R::new(p, expected);
    for (col, ct) in out.iter().enumerate() {
        let got: R = sk.decrypt::<R>(ct, p);
        assert_eq!(got, expected, "first_dim col {col} must select row {t}");
    }
});
