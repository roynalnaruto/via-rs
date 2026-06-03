//! §3.1 symmetric + §3.2 asymmetric modulus switching. See
//! `.docs/primitives.md` §3.1–§3.2.
//!
//! Rescale an RLWE ciphertext from one modulus to another by rounding each
//! coefficient: $c'_i = \mathrm{round}(c_i \cdot q' / q) \bmod q'$, integer
//! arithmetic only (float drift would corrupt 75-bit-class moduli). The
//! coefficient hot loop lives in [`super::kernels::mod_switch`]; the
//! orchestrators here own the ring-type plumbing.
//!
//! - [`mod_switch_sym`] (§3.1) — both mask and body to the same target $q'$.
//! - [`mod_switch_asym`] (§3.2) — mask and body to **different** targets.
//!   The §3.2 "body-only rescale" flavour is the call convention
//!   `R_MASK = R_SRC` with `mask_mod = src_mod` (the mask passes through
//!   unchanged because $q'_A = q$).
//!
//! # Constant-time: No
//!
//! Both orchestrators rescale RLWE-uniform ciphertext coefficients, which
//! leak nothing about secrets under the RLWE assumption (§0.6). Not for
//! secret-key material — see [`super::rekey`].

use crate::algebra::ring::abstraction::RingPoly;
use crate::encryption::types::{ModSwitchedCiphertext, RLWECiphertext};

use super::kernels::RescaleConsts;

/// §3.1 — symmetric modulus switch $\mathrm{ModSwitch}_{q \to q'}$.
///
/// Rescale both mask and body of `ct` from its source modulus $q$ to the
/// destination modulus $q'$ carried by `dst_mod`, returning an RLWE
/// ciphertext over the destination ring backend `R_DST`. Used for noise
/// reduction between pipeline steps and as the first step of VIA-C's
/// `RespComp` (compressing the answer from $q_2$ to $q_3$).
///
/// ```rust
/// use via_rs::algebra::ring::abstraction::RingPoly;
/// use via_rs::algebra::ring::element::Poly;
/// use via_rs::algebra::ring::form::Coefficient;
/// use via_rs::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_rs::encryption::types::RLWECiphertext;
/// use via_rs::switching::mod_switch::mod_switch_sym;
///
/// // Rescale a trivial ciphertext from q = 2^8 to q' = 2^4.
/// type Src = Poly<4, PowerOfTwoModulus<8>, Coefficient>;
/// type Dst = Poly<4, PowerOfTwoModulus<4>, Coefficient>;
/// let mask = <Src as RingPoly<4>>::from_u128_coeffs(PowerOfTwoModulus, &[0; 4]);
/// let body = <Src as RingPoly<4>>::from_u128_coeffs(PowerOfTwoModulus, &[128, 0, 0, 0]);
/// let ct = RLWECiphertext::new(mask, body);
/// let out: RLWECiphertext<4, Dst> = mod_switch_sym(&ct, PowerOfTwoModulus);
/// let mut b = [0u128; 4];
/// RingPoly::to_u128_coeffs(&out.body, &mut b);
/// // 128 * 16 / 256 = 8.
/// assert_eq!(b, [8, 0, 0, 0]);
/// ```
// `R_SRC` / `R_DST` mirror the paper's $q \to q'$ source/destination naming;
// the snake_case reads far clearer than `RSrc` / `RDst` at every call site.
#[allow(non_camel_case_types)]
pub fn mod_switch_sym<const N: usize, R_SRC: RingPoly<N>, R_DST: RingPoly<N>>(
    ct: &RLWECiphertext<N, R_SRC>,
    dst_mod: R_DST::Modulus,
) -> RLWECiphertext<N, R_DST> {
    let q_src = R_SRC::modulus_value(ct.mask.modulus());
    let q_dst = R_DST::modulus_value(dst_mod);
    let consts = RescaleConsts::new(q_src, q_dst);

    let mut mask_u128 = [0u128; N];
    let mut body_u128 = [0u128; N];
    ct.mask.to_u128_coeffs(&mut mask_u128);
    ct.body.to_u128_coeffs(&mut body_u128);

    // In-place rescale: each lane is read once then overwritten before the
    // next iteration. `RescaleConsts::scale` yields a value in [0, q_dst];
    // `from_u128_coeffs` reduces it into [0, q_dst) below.
    for v in mask_u128.iter_mut() {
        *v = consts.scale(*v);
    }
    for v in body_u128.iter_mut() {
        *v = consts.scale(*v);
    }

    RLWECiphertext::new(
        R_DST::from_u128_coeffs(dst_mod, &mask_u128),
        R_DST::from_u128_coeffs(dst_mod, &body_u128),
    )
}

/// §3.2 — asymmetric modulus switch.
///
/// Rescale the mask of `ct` to `mask_mod` and the body to `body_mod`, which
/// may be **different** moduli, returning a [`ModSwitchedCiphertext`]. Two
/// flavours collapse onto this single function:
///
/// - **Full asymmetric** (VIA's final Answer step): both targets differ from
///   the source, e.g. mask at $q_3$ and body at $q_4$.
/// - **Body-only rescale** (VIA-C's `RespComp` trailing op): call with
///   `R_MASK = R_SRC` and `mask_mod = ct.mask.modulus()`, so the mask
///   rescales $q \to q$ (identity, since $q'_A = q$) while only the body
///   shrinks to $q_4$.
///
/// Decryption uses [`crate::encryption::SecretKey::decrypt_asymmetric`].
///
/// ```rust
/// use via_rs::algebra::ring::abstraction::RingPoly;
/// use via_rs::algebra::ring::element::Poly;
/// use via_rs::algebra::ring::form::Coefficient;
/// use via_rs::algebra::zq::modulus::PowerOfTwoModulus;
/// use via_rs::encryption::types::RLWECiphertext;
/// use via_rs::switching::mod_switch::mod_switch_asym;
///
/// // Body-only rescale: mask stays at q = 2^8, body shrinks to 2^4.
/// type Q8 = Poly<4, PowerOfTwoModulus<8>, Coefficient>;
/// type Q4 = Poly<4, PowerOfTwoModulus<4>, Coefficient>;
/// let mask = <Q8 as RingPoly<4>>::from_u128_coeffs(PowerOfTwoModulus, &[7, 0, 0, 0]);
/// let body = <Q8 as RingPoly<4>>::from_u128_coeffs(PowerOfTwoModulus, &[128, 0, 0, 0]);
/// let ct = RLWECiphertext::new(mask, body);
/// let out = mod_switch_asym::<4, _, Q8, Q4>(&ct, PowerOfTwoModulus, PowerOfTwoModulus);
/// let mut m = [0u128; 4];
/// let mut b = [0u128; 4];
/// RingPoly::to_u128_coeffs(&out.mask, &mut m);
/// RingPoly::to_u128_coeffs(&out.body, &mut b);
/// assert_eq!(m, [7, 0, 0, 0]); // mask unchanged (q → q)
/// assert_eq!(b, [8, 0, 0, 0]); // 128 * 16 / 256 = 8
/// ```
#[allow(non_camel_case_types)]
pub fn mod_switch_asym<
    const N: usize,
    R_SRC: RingPoly<N>,
    R_MASK: RingPoly<N>,
    R_BODY: RingPoly<N>,
>(
    ct: &RLWECiphertext<N, R_SRC>,
    mask_mod: R_MASK::Modulus,
    body_mod: R_BODY::Modulus,
) -> ModSwitchedCiphertext<N, R_MASK, R_BODY> {
    let q_src = R_SRC::modulus_value(ct.mask.modulus());
    let mask_consts = RescaleConsts::new(q_src, R_MASK::modulus_value(mask_mod));
    let body_consts = RescaleConsts::new(q_src, R_BODY::modulus_value(body_mod));

    let mut mask_u128 = [0u128; N];
    let mut body_u128 = [0u128; N];
    ct.mask.to_u128_coeffs(&mut mask_u128);
    ct.body.to_u128_coeffs(&mut body_u128);

    for v in mask_u128.iter_mut() {
        *v = mask_consts.scale(*v);
    }
    for v in body_u128.iter_mut() {
        *v = body_consts.scale(*v);
    }

    ModSwitchedCiphertext::new(
        R_MASK::from_u128_coeffs(mask_mod, &mask_u128),
        R_BODY::from_u128_coeffs(body_mod, &body_u128),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::zq::modulus::PowerOfTwoModulus;

    type Q8 = Poly<4, PowerOfTwoModulus<8>, Coefficient>;
    type Q4 = Poly<4, PowerOfTwoModulus<4>, Coefficient>;

    fn q8(coeffs: &[u128; 4]) -> Q8 {
        <Q8 as RingPoly<4>>::from_u128_coeffs(PowerOfTwoModulus, coeffs)
    }

    #[test]
    fn mod_switch_sym_round_trip_at_pow2_moduli() {
        // q = 256, q' = 16. Scale factor 1/16.
        let ct = RLWECiphertext::new(q8(&[16, 32, 0, 240]), q8(&[128, 0, 8, 255]));
        let out: RLWECiphertext<4, Q4> = mod_switch_sym(&ct, PowerOfTwoModulus);
        let mut m = [0u128; 4];
        let mut b = [0u128; 4];
        RingPoly::to_u128_coeffs(&out.mask, &mut m);
        RingPoly::to_u128_coeffs(&out.body, &mut b);
        // round(c * 16 / 256) = round(c / 16):
        // mask: 16→1, 32→2, 0→0, 240→15.
        assert_eq!(m, [1, 2, 0, 15]);
        // body: 128→8, 0→0, 8→round(0.5)=1, 255→round(15.9)=16→0 mod 16.
        assert_eq!(b, [8, 0, 1, 0]);
    }

    #[test]
    fn mod_switch_asym_returns_correct_moduli() {
        // Mask → q3 = 16, body → q4 = 16 here (toy), but distinct types.
        let ct = RLWECiphertext::new(q8(&[16, 0, 0, 0]), q8(&[128, 0, 0, 0]));
        let out = mod_switch_asym::<4, _, Q4, Q4>(&ct, PowerOfTwoModulus, PowerOfTwoModulus);
        // Mask values < q_mask = 16, body values < q_body = 16.
        let mut m = [0u128; 4];
        let mut b = [0u128; 4];
        RingPoly::to_u128_coeffs(&out.mask, &mut m);
        RingPoly::to_u128_coeffs(&out.body, &mut b);
        assert!(m.iter().all(|&x| x < 16));
        assert!(b.iter().all(|&x| x < 16));
        assert_eq!(m, [1, 0, 0, 0]);
        assert_eq!(b, [8, 0, 0, 0]);
    }

    #[test]
    fn mod_switch_asym_body_only_call_convention() {
        // R_MASK = R_SRC, mask_mod = src_mod: mask passes through unchanged,
        // only body shrinks.
        let ct = RLWECiphertext::new(q8(&[7, 200, 0, 0]), q8(&[128, 0, 0, 0]));
        let out = mod_switch_asym::<4, _, Q8, Q4>(&ct, PowerOfTwoModulus, PowerOfTwoModulus);
        let mut m = [0u128; 4];
        let mut b = [0u128; 4];
        RingPoly::to_u128_coeffs(&out.mask, &mut m);
        RingPoly::to_u128_coeffs(&out.body, &mut b);
        // Mask q → q is the identity rescale.
        assert_eq!(m, [7, 200, 0, 0]);
        assert_eq!(b, [8, 0, 0, 0]);
    }

    #[test]
    fn mod_switch_sym_kat_viac_q1p0_to_q2() {
        // Lock Python `_round_scale` parity at hardcoded inputs. Use
        // single-prime moduli q_src and q_dst with the in-test reference
        // formula round(c * q_dst / q_src) = (c * q_dst + q_src/2) / q_src.
        use crate::algebra::zq::modulus::DynModulus;
        const Q_SRC: u64 = 1 << 20;
        const Q_DST: u64 = 1 << 12;
        type Dyn = Poly<4, DynModulus, Coefficient>;
        let src_mod = DynModulus::new(Q_SRC);
        let dst_mod = DynModulus::new(Q_DST);

        let inputs: [u128; 4] = [0, (Q_SRC / 2) as u128, (Q_SRC - 1) as u128, 1];
        let ct = RLWECiphertext::new(
            <Dyn as RingPoly<4>>::from_u128_coeffs(src_mod, &inputs),
            <Dyn as RingPoly<4>>::from_u128_coeffs(src_mod, &inputs),
        );
        let out: RLWECiphertext<4, Dyn> = mod_switch_sym(&ct, dst_mod);

        // Reference `_round_scale` formula, evaluated in-test.
        let round_scale = |c: u128| -> u128 {
            ((c * Q_DST as u128 + (Q_SRC as u128 / 2)) / Q_SRC as u128) % Q_DST as u128
        };
        let expected: [u128; 4] = core::array::from_fn(|i| round_scale(inputs[i]));

        let mut got = [0u128; 4];
        RingPoly::to_u128_coeffs(&out.body, &mut got);
        assert_eq!(got, expected);
        let mut got_mask = [0u128; 4];
        RingPoly::to_u128_coeffs(&out.mask, &mut got_mask);
        assert_eq!(got_mask, expected);
    }
}
