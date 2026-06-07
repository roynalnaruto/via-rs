//! §4.5 deterministic rotation (Part 1) + §4.4 controlled rotation (Part 2).
//!
//! See `.docs/primitives.md` §4.4-§4.5.
//!
//! `rotate(ct, k)` multiplies both RLWE components by $X^k$ via
//! [`crate::algebra::ring::RingPoly::mul_x_pow`]. `crot` (Part 2) layers
//! RGSW-controlled rotation on top, parameterised by `CRotDir`.

use crate::algebra::ring::RingPoly;
use crate::encryption::types::{RGSWCiphertext, RLWECiphertext};

use super::mux::cmux;

/// §4.5 — Rotate an RLWE ciphertext by $X^k$: returns $\mathrm{RLWE}(M \cdot
/// X^k)$.
///
/// Both the mask $A$ and body $B$ are multiplied by $X^k$ via
/// [`RingPoly::mul_x_pow`], which normalises `k` modulo `N` and applies the
/// negacyclic sign (coefficient at position $i$ maps to $(i+k) \bmod N$,
/// negated when $i+k \ge N$, since $X^N \equiv -1$). The result decrypts to
/// $M \cdot X^k$ under the same key.
///
/// `paper:gates.py:63-81`
///
/// # Constant-time
///
/// `mul_x_pow` is data-independent and `k` is a **public** loop/tree index.
/// Encrypted-exponent rotation goes through §4.4 `crot`, not here.
///
/// # Example
///
/// ```rust
/// use via_primitives::algebra::ring::RingPoly;
/// use via_primitives::algebra::ring::element::Poly;
/// use via_primitives::algebra::ring::form::Coefficient;
/// use via_primitives::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_primitives::encryption::types::RLWECiphertext;
/// use via_primitives::gates::rotate;
///
/// type R = Poly<4, PowerOfTwoModulus<10>, Coefficient>;
/// let q = PowerOfTwoModulus::<10>;
///
/// // Trivial (noiseless) ciphertext with body [1, 0, 0, 0].
/// let body = <R as RingPoly<4>>::from_u128_coeffs(q, &[1, 0, 0, 0]);
/// let ct = RLWECiphertext::<4, R>::new(R::zero(q), body);
///
/// // rotate(·, 1): [1,0,0,0] -> [0,1,0,0].
/// let mut out = [0u128; 4];
/// RingPoly::to_u128_coeffs(&rotate(&ct, 1).body, &mut out);
/// assert_eq!(out, [0, 1, 0, 0]);
///
/// // rotate(·, 4) = rotate(·, N): X^N = -1, so [1,0,0,0] -> [-1,0,0,0].
/// RingPoly::to_u128_coeffs(&rotate(&ct, 4).body, &mut out);
/// assert_eq!(out[0], 1023); // -1 mod 2^10
/// ```
pub fn rotate<const N: usize, R: RingPoly<N>>(
    ct: &RLWECiphertext<N, R>,
    k: usize,
) -> RLWECiphertext<N, R> {
    RLWECiphertext::new(ct.mask.mul_x_pow(k), ct.body.mul_x_pow(k))
}

/// §4.4 — Per-bit rotation direction for [`crot`].
///
/// - `Forward` — rotate by $+2^i$: `rotate(result, 1 << i)` then
///   `cmux(bit, result, rotated)`. Matches `gates.py:251-253`.
/// - `SlotExtract` — rotate by $-2^i$ via the negacyclic identity
///   $X^{-2^i} \equiv -X^{N-2^i}$: `rotate(result, N - (1 << i))`, negate, then
///   CMux. Brings ring-slot $\gamma$ to slot 0 for downstream `ring_switch` /
///   `project_at(0)` extraction (VIA-C Figure 8 Step 6). Matches
///   `server.py:218-225`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CRotDir {
    /// Positive rotation: result ← CMux(bit, result, $X^{2^i} \cdot$ result).
    Forward,
    /// Negative (slot-extraction) rotation: result ← CMux(bit, result,
    /// $-X^{N-2^i} \cdot$ result) = CMux(bit, result, $X^{-2^i} \cdot$ result).
    SlotExtract,
}

/// §4.4 — Controlled rotation: apply a sequence of RGSW-controlled rotations to
/// `ct`, one per `rotation_bits[i]`.
///
/// With `dir = CRotDir::Forward` the result encrypts $M \cdot X^{\gamma}$,
/// $\gamma = \sum_i b_i \cdot 2^i$. With `dir = CRotDir::SlotExtract` it
/// encrypts $M \cdot X^{-\gamma}$ via $X^{-2^i} \equiv -X^{N-2^i} \pmod{X^N+1}$
/// (negate the rotated ciphertext before the CMux — `Neg` is `Copy`-cheap).
///
/// `paper:gates.py:227-254` (Forward), `paper:server.py:218-225` (SlotExtract).
///
/// # Two-base convention
///
/// Like [`cmux`], `(base_neg_s_m, base_m)` feed the two RGSW halves; pass equal
/// values to reproduce the Python single-`base` behaviour.
///
/// # Constant-time
///
/// Each step runs one `cmux` (always both gadget products); no secret branch.
/// The shifts $2^i$ / $N - 2^i$ are public.
///
/// # Example
///
/// ```rust
/// use via_primitives::algebra::ring::RingPoly;
/// use via_primitives::algebra::ring::element::Poly;
/// use via_primitives::algebra::ring::form::Coefficient;
/// use via_primitives::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_primitives::encryption::rlwe::encode;
/// use via_primitives::encryption::types::SecretKey;
/// use via_primitives::sampling::distribution::Distribution;
/// use via_primitives::sampling::prg::Shake256Prg;
/// use via_primitives::gates::{crot, CRotDir};
///
/// type R = Poly<4, PowerOfTwoModulus<10>, Coefficient>;
/// type RP = Poly<4, PowerOfTwoModulus<1>, Coefficient>;
/// let q = PowerOfTwoModulus::<10>;
/// let p = PowerOfTwoModulus::<1>;
///
/// let mut sk_prg = Shake256Prg::new(b"crot-doc-sk");
/// let sk = SecretKey::<4, R>::keygen(q, Distribution::Ternary, &mut sk_prg);
/// let mut one = [0u128; 4];
/// one[0] = 1;
/// let one_poly = <R as RingPoly<4>>::from_u128_coeffs(q, &one);
/// let mut rgsw_prg = Shake256Prg::new(b"crot-doc-rgsw");
/// let bit = sk.encrypt_rgsw::<10, 10>(&one_poly, 2, 2, Distribution::Ternary, &mut rgsw_prg);
///
/// let msg: RP = Poly::new(p, [1, 0, 0, 0]);
/// let mut enc_prg = Shake256Prg::new(b"crot-doc-enc");
/// let ct = sk.encrypt(&encode(&msg, q), Distribution::Ternary, &mut enc_prg);
///
/// // Forward by 2^0 = 1: [1,0,0,0] -> [0,1,0,0].
/// let out = crot(CRotDir::Forward, &[bit], ct, 2, 2);
/// assert_eq!(sk.decrypt::<RP>(&out, p), Poly::new(p, [0, 1, 0, 0]));
/// ```
pub fn crot<const N: usize, R: RingPoly<N>, const L1: usize, const L2: usize>(
    dir: CRotDir,
    rotation_bits: &[RGSWCiphertext<N, R, L1, L2>],
    ct: RLWECiphertext<N, R>,
    base_neg_s_m: u64,
    base_m: u64,
) -> RLWECiphertext<N, R> {
    let mut result = ct;
    for (i, bit) in rotation_bits.iter().enumerate() {
        result = match dir {
            CRotDir::Forward => {
                let rotated = rotate(&result, 1 << i);
                cmux(bit, &result, &rotated, base_neg_s_m, base_m)
            }
            CRotDir::SlotExtract => {
                // X^{-2^i} = -X^{N-2^i} in R_{N,q} = Z_q[X]/(X^N+1).
                let neg_rot = -rotate(&result, N - (1 << i));
                cmux(bit, &result, &neg_rot, base_neg_s_m, base_m)
            }
        };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::zq::modulus::PowerOfTwoModulus;
    use crate::encryption::rlwe::encode;
    use crate::encryption::types::SecretKey;
    use crate::sampling::distribution::Distribution;
    use crate::sampling::prg::Shake256Prg;

    type RQ<const N: usize> = Poly<N, PowerOfTwoModulus<10>, Coefficient>;
    type RP<const N: usize> = Poly<N, PowerOfTwoModulus<1>, Coefficient>;

    fn sk_and_ct(
        sk_seed: &[u8],
        enc_seed: &[u8],
        coeffs: [u64; 4],
    ) -> (SecretKey<4, RQ<4>>, RLWECiphertext<4, RQ<4>>, RP<4>) {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        let mut sk_prg = Shake256Prg::new(sk_seed);
        let sk = SecretKey::<4, RQ<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let msg: RP<4> = Poly::new(p, coeffs);
        let encoded: RQ<4> = encode(&msg, q);
        let mut enc_prg = Shake256Prg::new(enc_seed);
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
        (sk, ct, msg)
    }

    #[test]
    fn rotate_identity_k_zero() {
        let p = PowerOfTwoModulus::<1>;
        let (sk, ct, msg) = sk_and_ct(b"rot-zero-key", b"rot-zero-enc", [1, 0, 1, 0]);
        let recovered: RP<4> = sk.decrypt(&rotate(&ct, 0), p);
        assert_eq!(recovered, msg);
    }

    #[test]
    fn rotate_by_one_shifts_coefficients() {
        let p = PowerOfTwoModulus::<1>;
        let (sk, ct, _) = sk_and_ct(b"rot-one-key", b"rot-one-enc", [1, 0, 0, 0]);
        let recovered: RP<4> = sk.decrypt(&rotate(&ct, 1), p);
        assert_eq!(recovered, Poly::new(p, [0, 1, 0, 0]));
    }

    #[test]
    fn rotate_by_n_negates_coefficients() {
        let p = PowerOfTwoModulus::<1>;
        let (sk, ct, msg) = sk_and_ct(b"rot-full-key", b"rot-full-enc", [1, 0, 1, 0]);
        // p=2: -m == m, so X^N rotation is plaintext-identity.
        let recovered: RP<4> = sk.decrypt(&rotate(&ct, 4), p);
        assert_eq!(recovered, msg);
    }

    #[test]
    fn rotate_normalizes_k_mod_n() {
        let p = PowerOfTwoModulus::<1>;
        let (sk, ct, msg) = sk_and_ct(b"rot-norm-key", b"rot-norm-enc", [1, 1, 0, 0]);
        // X^(2N) = (X^N)^2 = 1 -> identity.
        let recovered: RP<4> = sk.decrypt(&rotate(&ct, 8), p);
        assert_eq!(recovered, msg);
    }

    // ----- §4.4 crot (Part 2) -----

    fn toy_rgsw(
        sk: &SecretKey<4, RQ<4>>,
        bit: u64,
        seed: &[u8],
    ) -> RGSWCiphertext<4, RQ<4>, 10, 10> {
        let q = PowerOfTwoModulus::<10>;
        let mut c = [0u128; 4];
        c[0] = bit as u128;
        let m: RQ<4> = <RQ<4> as RingPoly<4>>::from_u128_coeffs(q, &c);
        let mut prg = Shake256Prg::new(seed);
        sk.encrypt_rgsw::<10, 10>(&m, 2, 2, Distribution::Ternary, &mut prg)
    }

    #[test]
    fn crot_forward_by_1() {
        let p = PowerOfTwoModulus::<1>;
        let (sk, ct, _) = sk_and_ct(b"crot-f1-key", b"crot-f1-enc", [1, 0, 0, 0]);
        let bits = [toy_rgsw(&sk, 1, b"crot-f1-b0")];
        let out = crot(CRotDir::Forward, &bits, ct, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [0, 1, 0, 0]));
    }

    #[test]
    fn crot_forward_by_2() {
        let p = PowerOfTwoModulus::<1>;
        let (sk, ct, _) = sk_and_ct(b"crot-f2-key", b"crot-f2-enc", [1, 0, 0, 0]);
        let bits = [
            toy_rgsw(&sk, 0, b"crot-f2-b0"),
            toy_rgsw(&sk, 1, b"crot-f2-b1"),
        ];
        let out = crot(CRotDir::Forward, &bits, ct, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [0, 0, 1, 0]));
    }

    #[test]
    fn crot_forward_by_3() {
        let p = PowerOfTwoModulus::<1>;
        let (sk, ct, _) = sk_and_ct(b"crot-f3-key", b"crot-f3-enc", [1, 0, 0, 0]);
        let bits = [
            toy_rgsw(&sk, 1, b"crot-f3-b0"),
            toy_rgsw(&sk, 1, b"crot-f3-b1"),
        ];
        let out = crot(CRotDir::Forward, &bits, ct, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [0, 0, 0, 1]));
    }

    #[test]
    fn crot_slot_extract_correctness() {
        // bits [1,1] => gamma = 3, SlotExtract applies X^{-3}.
        // Reference: -rotate(ct, N-3). At p=2 negation is a noop, so the
        // decrypted plaintexts compare equal. The result must differ from the
        // Forward(+3) result [0,0,0,1].
        let p = PowerOfTwoModulus::<1>;
        let (sk, ct, _) = sk_and_ct(b"crot-se-key", b"crot-se-enc", [1, 0, 0, 0]);
        let bits = [
            toy_rgsw(&sk, 1, b"crot-se-b0"),
            toy_rgsw(&sk, 1, b"crot-se-b1"),
        ];
        let out = crot(CRotDir::SlotExtract, &bits, ct, 2, 2);
        let manual = -rotate(&ct, 4 - 3); // -X^{N-3}, N = 4
        assert_eq!(
            sk.decrypt::<RP<4>>(&out, p),
            sk.decrypt::<RP<4>>(&manual, p)
        );
        assert_ne!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [0, 0, 0, 1]));
    }
}
