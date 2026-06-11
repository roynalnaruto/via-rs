//! §4.1 CMux + §4.2 DMux (Part 1) and §4.3 CMux/DMux trees (Part 2).
//!
//! See `.docs/primitives.md` §4.1-§4.3 and `.docs/via.pdf` Algorithms 1-2.
//!
//! All gates are thin wrappers over
//! [`crate::encryption::RGSWCiphertext::external_product`]. The trees are
//! slice-based (no allocation): `cmux_tree` reduces in place, `dmux_tree`
//! writes into a caller-provided buffer.
//!
//! ## Two-base API vs. Python single-base
//!
//! The Python reference passes a single `(base, depth)` pair. The Rust API
//! mirrors `external_product` and accepts **two** bases `(base_neg_s_m,
//! base_m)` — one per RGSW RLev half (paper Tables 5-6 list distinct `(L,B)`
//! per half). Passing `base_neg_s_m == base_m` reproduces Python exactly.

use crate::algebra::ring::RingPolyEval;
use crate::encryption::types::{RGSWCiphertext, RLWECiphertext};

/// §4.1 — Homomorphic 1-of-2 multiplexer: returns `ct0` when the encrypted bit
/// is 0, `ct1` when it is 1.
///
/// Computes $c_0 + \mathrm{RGSW}(b) \boxtimes (c_1 - c_0)$: when $b = 0$ the
/// external product is $\approx 0$ (leaving $c_0$); when $b = 1$ it is
/// $\approx c_1 - c_0$ (yielding $c_1$).
///
/// `paper:gates.py:84-116`
///
/// # Two-base divergence from Python
///
/// Python passes one `(base, depth)`; this function threads `(base_neg_s_m,
/// base_m)` to match `external_product`. Pass equal values to reproduce
/// Python's symmetric behaviour.
///
/// # Noise
///
/// One external product's growth per call (`~‖M₁‖₁·σ_e + L·B/4` per half).
/// In a depth-`m` `cmux_tree` this accumulates over `m` levels.
///
/// # Constant-time
///
/// The external product always runs both gadget products and the final add;
/// no branch depends on secret data.
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
/// use via_primitives::gates::cmux;
///
/// type R = Poly<4, PowerOfTwoModulus<10>, Coefficient>;
/// type RP = Poly<4, PowerOfTwoModulus<1>, Coefficient>;
/// let q = PowerOfTwoModulus::<10>;
/// let p = PowerOfTwoModulus::<1>;
///
/// let mut sk_prg = Shake256Prg::new(b"cmux-doc-sk");
/// let sk = SecretKey::<4, R>::keygen(q, Distribution::Ternary, &mut sk_prg);
///
/// // RGSW(0) selects ct0.
/// let mut rgsw_prg = Shake256Prg::new(b"cmux-doc-rgsw");
/// let rgsw0 = sk.encrypt_rgsw::<10, 10>(&R::zero(q), 2, 2, Distribution::Ternary, &mut rgsw_prg);
///
/// let msg0: RP = Poly::new(p, [1, 0, 0, 0]);
/// let msg1: RP = Poly::new(p, [0, 1, 0, 0]);
/// let mut enc_prg = Shake256Prg::new(b"cmux-doc-enc");
/// let ct0 = sk.encrypt(&encode(&msg0, q), Distribution::Ternary, &mut enc_prg);
/// let ct1 = sk.encrypt(&encode(&msg1, q), Distribution::Ternary, &mut enc_prg);
///
/// let out = cmux(&rgsw0, &ct0, &ct1, 2, 2);
/// let recovered: RP = sk.decrypt(&out, p);
/// assert_eq!(recovered, msg0);
/// ```
pub fn cmux<const N: usize, R: RingPolyEval<N>, const L1: usize, const L2: usize>(
    select_bit: &RGSWCiphertext<N, R, L1, L2>,
    ct0: &RLWECiphertext<N, R>,
    ct1: &RLWECiphertext<N, R>,
    base_neg_s_m: u64,
    base_m: u64,
) -> RLWECiphertext<N, R> {
    let diff = *ct1 - *ct0;
    *ct0 + select_bit.external_product(&diff, base_neg_s_m, base_m)
}

/// §4.2 — Homomorphic 1-to-2 demultiplexer: routes `ct` to position `b`.
///
/// Returns `(result0, result1)` where `result0` decrypts to $M \cdot (1 - b)$
/// and `result1` to $M \cdot b$. Computed as `product = RGSW(b) ⊠ ct`,
/// `result0 = ct - product`, `result1 = product`.
///
/// `paper:gates.py:119-149`
///
/// # Two-base divergence / Constant-time
///
/// Same as [`cmux`]: `(base_neg_s_m, base_m)` per half; the external product
/// always runs; no secret branch.
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
/// use via_primitives::gates::dmux;
///
/// type R = Poly<4, PowerOfTwoModulus<10>, Coefficient>;
/// type RP = Poly<4, PowerOfTwoModulus<1>, Coefficient>;
/// let q = PowerOfTwoModulus::<10>;
/// let p = PowerOfTwoModulus::<1>;
///
/// let mut sk_prg = Shake256Prg::new(b"dmux-doc-sk");
/// let sk = SecretKey::<4, R>::keygen(q, Distribution::Ternary, &mut sk_prg);
/// let mut rgsw_prg = Shake256Prg::new(b"dmux-doc-rgsw");
/// let rgsw0 = sk.encrypt_rgsw::<10, 10>(&R::zero(q), 2, 2, Distribution::Ternary, &mut rgsw_prg);
///
/// let msg: RP = Poly::new(p, [1, 0, 1, 0]);
/// let mut enc_prg = Shake256Prg::new(b"dmux-doc-enc");
/// let ct = sk.encrypt(&encode(&msg, q), Distribution::Ternary, &mut enc_prg);
///
/// // RGSW(0): result0 carries M, result1 is zero.
/// let (r0, r1) = dmux(&rgsw0, &ct, 2, 2);
/// assert_eq!(sk.decrypt::<RP>(&r0, p), msg);
/// assert_eq!(sk.decrypt::<RP>(&r1, p), Poly::new(p, [0, 0, 0, 0]));
/// ```
pub fn dmux<const N: usize, R: RingPolyEval<N>, const L1: usize, const L2: usize>(
    control_bit: &RGSWCiphertext<N, R, L1, L2>,
    ct: &RLWECiphertext<N, R>,
    base_neg_s_m: u64,
    base_m: u64,
) -> (RLWECiphertext<N, R>, RLWECiphertext<N, R>) {
    let product = control_bit.external_product(ct, base_neg_s_m, base_m);
    (*ct - product, product)
}

/// §4.3 — Homomorphic 1-of-$2^m$ selection via a binary CMux tree (VIA
/// Algorithm 1). Returns the input at index $\sum_i b_i \cdot 2^i$
/// (`select_bits[0]` = LSB; the *reverse* of [`dmux_tree`]'s order). Reduces
/// `inputs` **in place**; the returned ciphertext is `inputs[0]`.
///
/// `paper:gates.py:152-190`
///
/// # Panics
///
/// if `inputs.len() != 1 << select_bits.len()`.
///
/// # In-place safety
///
/// Each result is **copied out of the slice before the write**
/// (`RLWECiphertext: Copy`): a one-liner `inputs[j] = cmux(&inputs[2j],
/// &inputs[2j+1], …)` would fail the borrow checker, which borrows the whole
/// slice (not individual elements). With the copy, the write `inputs[j]`
/// (`j < half`) can never alias an unread source `inputs[2j']` (`j' > j`)
/// because `j ≤ 2j`.
///
/// # Noise
///
/// One external product per level; after `m` levels the decryption bound is
/// `m·(‖S‖₁·σ_e + L·B/4) < q/(2p)` (holds at VIA-C paper params and trivially
/// at toy params).
///
/// # Constant-time
///
/// No — see [`cmux`]; the gadget-decomposition path is data-dependent only on
/// public ciphertext coefficients.
pub fn cmux_tree<const N: usize, R: RingPolyEval<N>, const L1: usize, const L2: usize>(
    select_bits: &[RGSWCiphertext<N, R, L1, L2>],
    inputs: &mut [RLWECiphertext<N, R>],
    base_neg_s_m: u64,
    base_m: u64,
) -> RLWECiphertext<N, R> {
    let m = select_bits.len();
    assert!(
        inputs.len() == 1 << m,
        "cmux_tree: inputs.len() must equal 2^m = {}; got {}",
        1usize << m,
        inputs.len(),
    );
    let mut active = inputs.len();
    for bit in select_bits {
        let half = active >> 1;
        for j in 0..half {
            // Copy out before the mutable write — Rust borrows the whole slice,
            // not individual elements, so a one-liner would not compile.
            let a = inputs[2 * j];
            let b = inputs[2 * j + 1];
            inputs[j] = cmux(bit, &a, &b, base_neg_s_m, base_m);
        }
        active = half;
    }
    inputs[0]
}

/// §4.3 — Homomorphic 1-to-$2^m$ distribution via a binary DMux tree (VIA
/// Algorithm 2). Fills `out` so `out[k]` carries $M$ when
/// **$k = \sum_i b_i \cdot 2^{m-1-i}$** — i.e. `control_bits[0]` is the **MSB**
/// of the output index (the *reverse* of [`cmux_tree`]'s LSB-first order); all
/// other slots decrypt to 0. For `control_bits = [RGSW(1), RGSW(0)]`, $m=2$:
/// $k = 1·2 + 0·1 = 2$. Verified against `test_gates.py:454-491`.
///
/// `paper:gates.py:193-224`
///
/// # Panics
///
/// if `out.len() != 1 << control_bits.len()`.
///
/// # In-place expansion safety
///
/// At level $i$ (active = $2^i$), process $j$ from $2^i-1$ down to 0: copy
/// `src = out[j]`, then write `out[2j+1]`, `out[2j]`. All earlier writes this
/// level landed at indices $\ge 2(j+1) = 2j+2 > j$, so `out[j]` still holds its
/// pre-expansion value when read.
///
/// # Noise
///
/// Output at depth `m` carries `m` chained external products:
/// `m·(‖S‖₁·σ_e + L·B/4) < q/(2p)`, same budget as [`cmux_tree`].
pub fn dmux_tree<const N: usize, R: RingPolyEval<N>, const L1: usize, const L2: usize>(
    control_bits: &[RGSWCiphertext<N, R, L1, L2>],
    input: RLWECiphertext<N, R>,
    out: &mut [RLWECiphertext<N, R>],
    base_neg_s_m: u64,
    base_m: u64,
) {
    let m = control_bits.len();
    assert!(
        out.len() == 1 << m,
        "dmux_tree: out.len() must equal 2^m = {}; got {}",
        1usize << m,
        out.len(),
    );
    out[0] = input;
    for (i, bit) in control_bits.iter().enumerate() {
        let active = 1usize << i;
        for j in (0..active).rev() {
            let src = out[j]; // read before any write to out[j]
            let (r0, r1) = dmux(bit, &src, base_neg_s_m, base_m);
            out[2 * j + 1] = r1;
            out[2 * j] = r0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::RingPoly;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::zq::modulus::PowerOfTwoModulus;
    use crate::encryption::rlwe::encode;
    use crate::encryption::types::SecretKey;
    use crate::sampling::distribution::Distribution;
    use crate::sampling::prg::Shake256Prg;

    type RQ<const N: usize> = Poly<N, PowerOfTwoModulus<10>, Coefficient>;
    type RP<const N: usize> = Poly<N, PowerOfTwoModulus<1>, Coefficient>;

    fn keygen(seed: &[u8]) -> SecretKey<4, RQ<4>> {
        let mut prg = Shake256Prg::new(seed);
        SecretKey::<4, RQ<4>>::keygen(PowerOfTwoModulus::<10>, Distribution::Ternary, &mut prg)
    }

    fn enc(sk: &SecretKey<4, RQ<4>>, coeffs: [u64; 4], seed: &[u8]) -> RLWECiphertext<4, RQ<4>> {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        let msg: RP<4> = Poly::new(p, coeffs);
        let mut prg = Shake256Prg::new(seed);
        sk.encrypt(&encode(&msg, q), Distribution::Ternary, &mut prg)
    }

    fn rgsw_bit(
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
    fn cmux_selects_ct0_when_bit_zero() {
        let p = PowerOfTwoModulus::<1>;
        let sk = keygen(b"cmux-zero-key");
        let rgsw0 = rgsw_bit(&sk, 0, b"cmux-zero-rgsw");
        let ct0 = enc(&sk, [1, 0, 0, 0], b"cmux-z-ct0");
        let ct1 = enc(&sk, [0, 1, 0, 0], b"cmux-z-ct1");
        let out = cmux(&rgsw0, &ct0, &ct1, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [1, 0, 0, 0]));
    }

    #[test]
    fn cmux_selects_ct1_when_bit_one() {
        let p = PowerOfTwoModulus::<1>;
        let sk = keygen(b"cmux-one-key");
        let rgsw1 = rgsw_bit(&sk, 1, b"cmux-one-rgsw");
        let ct0 = enc(&sk, [1, 0, 0, 0], b"cmux-o-ct0");
        let ct1 = enc(&sk, [0, 1, 0, 0], b"cmux-o-ct1");
        let out = cmux(&rgsw1, &ct0, &ct1, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [0, 1, 0, 0]));
    }

    #[test]
    fn dmux_routes_to_position_0_when_bit_zero() {
        let p = PowerOfTwoModulus::<1>;
        let sk = keygen(b"dmux-zero-key");
        let rgsw0 = rgsw_bit(&sk, 0, b"dmux-zero-rgsw");
        let ct = enc(&sk, [1, 0, 1, 0], b"dmux-z-ct");
        let (r0, r1) = dmux(&rgsw0, &ct, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&r0, p), Poly::new(p, [1, 0, 1, 0]));
        assert_eq!(sk.decrypt::<RP<4>>(&r1, p), Poly::new(p, [0, 0, 0, 0]));
    }

    #[test]
    fn dmux_routes_to_position_1_when_bit_one() {
        let p = PowerOfTwoModulus::<1>;
        let sk = keygen(b"dmux-one-key");
        let rgsw1 = rgsw_bit(&sk, 1, b"dmux-one-rgsw");
        let ct = enc(&sk, [1, 0, 1, 0], b"dmux-o-ct");
        let (r0, r1) = dmux(&rgsw1, &ct, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&r0, p), Poly::new(p, [0, 0, 0, 0]));
        assert_eq!(sk.decrypt::<RP<4>>(&r1, p), Poly::new(p, [1, 0, 1, 0]));
    }

    /// Paper-class noise sanity: CMux at ViaCQ2, p=256, L=2, B=81, σ=4.
    #[test]
    fn cmux_noise_sanity_at_via_c_q2() {
        use crate::algebra::zq::modulus::paper::ViaCQ2;
        type Q<const N: usize> = Poly<N, ViaCQ2, Coefficient>;
        type P256<const N: usize> = Poly<N, PowerOfTwoModulus<8>, Coefficient>;
        let q = ViaCQ2::default();
        let p = PowerOfTwoModulus::<8>;
        let base = 81u64;
        let dist = Distribution::Gaussian { sigma: 4.0 };

        let mut sk_prg = Shake256Prg::new(b"cmux-vc-q2-sk");
        let sk = SecretKey::<16, Q<16>>::keygen(q, dist, &mut sk_prg);
        let mut one = [0u128; 16];
        one[0] = 1;
        let one_poly = <Q<16> as RingPoly<16>>::from_u128_coeffs(q, &one);
        let mut rgsw_prg = Shake256Prg::new(b"cmux-vc-q2-rgsw");
        let rgsw1 = sk.encrypt_rgsw::<2, 2>(&one_poly, base, base, dist, &mut rgsw_prg);

        let msg0: P256<16> = Poly::new(p, [3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let msg1: P256<16> = Poly::new(p, [0, 7, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
        let mut enc_prg = Shake256Prg::new(b"cmux-vc-q2-enc");
        let ct0 = sk.encrypt(&encode(&msg0, q), dist, &mut enc_prg);
        let ct1 = sk.encrypt(&encode(&msg1, q), dist, &mut enc_prg);
        let out = cmux(&rgsw1, &ct0, &ct1, base, base);
        assert_eq!(sk.decrypt::<P256<16>>(&out, p), msg1);
    }

    /// Asymmetric depths + bases must still select correctly (catches a base
    /// copy-paste bug).
    #[test]
    fn cmux_asymmetric_bases_selects_correctly() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        let mut sk_prg = Shake256Prg::new(b"cmux-asym-sk");
        let sk = SecretKey::<4, RQ<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut c = [0u128; 4];
        c[0] = 1;
        let one: RQ<4> = <RQ<4> as RingPoly<4>>::from_u128_coeffs(q, &c);
        let mut rgsw_prg = Shake256Prg::new(b"cmux-asym-rgsw");
        let rgsw1 = sk.encrypt_rgsw::<5, 10>(&one, 4, 2, Distribution::Ternary, &mut rgsw_prg);
        let ct0 = enc(&sk, [1, 0, 0, 0], b"cmux-asym-ct0");
        let ct1 = enc(&sk, [0, 0, 1, 0], b"cmux-asym-ct1");
        let out = cmux(&rgsw1, &ct0, &ct1, 4, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [0, 0, 1, 0]));
    }

    // ----- §4.3 trees (Part 2) -----

    // Four messages, each a 1 at a distinct position.
    fn four_inputs(sk: &SecretKey<4, RQ<4>>) -> [RLWECiphertext<4, RQ<4>>; 4] {
        [
            enc(sk, [1, 0, 0, 0], b"tree-in0"),
            enc(sk, [0, 1, 0, 0], b"tree-in1"),
            enc(sk, [0, 0, 1, 0], b"tree-in2"),
            enc(sk, [0, 0, 0, 1], b"tree-in3"),
        ]
    }

    #[test]
    fn cmux_tree_selects_index_0() {
        let p = PowerOfTwoModulus::<1>;
        let sk = keygen(b"ctree-i0-key");
        let bits = [
            rgsw_bit(&sk, 0, b"ctree-i0-b0"),
            rgsw_bit(&sk, 0, b"ctree-i0-b1"),
        ];
        let mut inputs = four_inputs(&sk);
        let out = cmux_tree(&bits, &mut inputs, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [1, 0, 0, 0]));
    }

    #[test]
    fn cmux_tree_selects_index_3() {
        let p = PowerOfTwoModulus::<1>;
        let sk = keygen(b"ctree-i3-key");
        let bits = [
            rgsw_bit(&sk, 1, b"ctree-i3-b0"),
            rgsw_bit(&sk, 1, b"ctree-i3-b1"),
        ];
        let mut inputs = four_inputs(&sk);
        let out = cmux_tree(&bits, &mut inputs, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out, p), Poly::new(p, [0, 0, 0, 1]));
    }

    #[test]
    #[should_panic(expected = "cmux_tree: inputs.len() must equal 2^m")]
    fn cmux_tree_length_mismatch_panics() {
        let sk = keygen(b"ctree-bad-key");
        let bits = [
            rgsw_bit(&sk, 0, b"ctree-bad-b0"),
            rgsw_bit(&sk, 0, b"ctree-bad-b1"),
        ];
        // 2 bits require 4 inputs; pass 3.
        let mut inputs = [
            enc(&sk, [0, 0, 0, 0], b"ctree-bad-0"),
            enc(&sk, [0, 0, 0, 0], b"ctree-bad-1"),
            enc(&sk, [0, 0, 0, 0], b"ctree-bad-2"),
        ];
        let _ = cmux_tree(&bits, &mut inputs, 2, 2);
    }

    #[test]
    fn dmux_tree_distributes_to_index_0() {
        let p = PowerOfTwoModulus::<1>;
        let sk = keygen(b"dtree-i0-key");
        let bits = [
            rgsw_bit(&sk, 0, b"dtree-i0-b0"),
            rgsw_bit(&sk, 0, b"dtree-i0-b1"),
        ];
        let ct = enc(&sk, [1, 0, 1, 0], b"dtree-i0-ct");
        let zero = enc(&sk, [0, 0, 0, 0], b"dtree-i0-zero");
        let mut out = [zero; 4];
        dmux_tree(&bits, ct, &mut out, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out[0], p), Poly::new(p, [1, 0, 1, 0]));
        for slot in &out[1..] {
            assert_eq!(sk.decrypt::<RP<4>>(slot, p), Poly::new(p, [0, 0, 0, 0]));
        }
    }

    #[test]
    fn dmux_tree_distributes_to_index_2() {
        // bits [1,0]: control_bits[0]=MSB ⇒ k = 1·2 + 0·1 = 2. Regression guard
        // for the MSB-first index convention.
        let p = PowerOfTwoModulus::<1>;
        let sk = keygen(b"dtree-i2-key");
        let bits = [
            rgsw_bit(&sk, 1, b"dtree-i2-b0"),
            rgsw_bit(&sk, 0, b"dtree-i2-b1"),
        ];
        let ct = enc(&sk, [1, 0, 1, 0], b"dtree-i2-ct");
        let zero = enc(&sk, [0, 0, 0, 0], b"dtree-i2-zero");
        let mut out = [zero; 4];
        dmux_tree(&bits, ct, &mut out, 2, 2);
        assert_eq!(sk.decrypt::<RP<4>>(&out[2], p), Poly::new(p, [1, 0, 1, 0]));
        for k in [0usize, 1, 3] {
            assert_eq!(sk.decrypt::<RP<4>>(&out[k], p), Poly::new(p, [0, 0, 0, 0]));
        }
    }

    #[test]
    #[should_panic(expected = "dmux_tree: out.len() must equal 2^m")]
    fn dmux_tree_length_mismatch_panics() {
        let sk = keygen(b"dtree-bad-key");
        let bits = [
            rgsw_bit(&sk, 0, b"dtree-bad-b0"),
            rgsw_bit(&sk, 0, b"dtree-bad-b1"),
        ];
        let ct = enc(&sk, [0, 0, 0, 0], b"dtree-bad-ct");
        let zero = enc(&sk, [0, 0, 0, 0], b"dtree-bad-zero");
        let mut out = [zero; 3]; // needs 4
        dmux_tree(&bits, ct, &mut out, 2, 2);
    }
}
