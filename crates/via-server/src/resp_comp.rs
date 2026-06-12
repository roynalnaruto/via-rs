//! RespComp — **asymmetric** response compression (Answer step 7).
//!
//! Three steps:
//!   1. `mod_switch_sym` q2 → q3  (symmetric; both mask and body, stays at n1)
//!   2. `ring_switch`    n1 → n2  @ q3  (input is **uniform q3** after step 1)
//!   3. `mod_switch_asym(_, q3, q4)`  trailing **body-only** rescale → `(A@q3, ⌊q4·B/q3⌉)`
//!
//! Doing the asymmetric rescale **last** (after ring-switch, at degree n2)
//! sidesteps the limitation of going symmetric — a mask@q3/body@q4 mismatch that
//! `ring_switch` can't handle. `Client::recover` decrypts with
//! `SecretKey::decrypt_asymmetric(S2@q3, q3, q4, p)`.

use via_primitives::algebra::ring::{RingPoly, RingPolyEval};
use via_primitives::encryption::types::{ModSwitchedCiphertext, RLWECiphertext};
use via_primitives::switching::RingSwitchKeyEval;
use via_primitives::switching::mod_switch::{mod_switch_asym, mod_switch_sym};
use via_primitives::switching::ring_switch::ring_switch_eval;

/// Paper-asymmetric RespComp: `RLWE_{S1}(M) @ (q2, n1)` → `ModSwitchedCT<n2, q3-mask, q4-body>`.
///
/// # Type parameters
///
/// - `R2` — input ring at `(q2, n1)`.
/// - `R3L` — intermediate ring at `(q3, n1)` (the `mod_switch_sym` output);
///   `R3L::Projected<N2> = R3`.
/// - `R3` — ring at `(q3, n2)` (ring-switch output and answer **mask**).
/// - `R4` — ring at `(q4, n2)` (answer **body** after the trailing rescale).
/// - `L`, `D` — ring-switch key depth and `D = N1 / N2`.
///
/// # Noise
///
/// Three rescales + one ring-switch; correctness needs the total under
/// `q4 / (2p)`. Toy params close (see tests); paper-scale closure follows from
/// the noise analysis in the paper.
///
/// # Panics
///
/// The `ring_switch` step requires `N1 = N2·D` — enforced at compile time by
/// `RingSwitchKey`'s `_CHECK` const when the key is built.
///
/// # Constant-time: No
///
/// Operates on RLWE-uniform ciphertext coefficients; no secret data is branched
/// on. `%`/division timing varies only on the public moduli.
#[allow(non_camel_case_types)]
pub fn resp_comp<
    const N1: usize,
    const N2: usize,
    R2: RingPoly<N1>,
    R3L: RingPoly<N1, Projected<N2> = R3>,
    R3: RingPoly<N2, Modulus = R3L::Modulus> + RingPolyEval<N2>,
    R4: RingPoly<N2>,
    const L: usize,
    const D: usize,
>(
    ct: &RLWECiphertext<N1, R2>,
    rsk: &RingSwitchKeyEval<N1, N2, R3, L, D>,
    q3_mod: R3L::Modulus,
    q4_mod: R4::Modulus,
    base_rsk: u64,
) -> ModSwitchedCiphertext<N2, R3, R4> {
    // Step 1: symmetric q2 → q3, still at degree n1.
    let ct_q3_n1: RLWECiphertext<N1, R3L> = mod_switch_sym::<N1, R2, R3L>(ct, q3_mod);

    // Step 2: ring-switch n1 → n2 @ q3 (uniform-q3 input — no asymmetry yet).
    let ct_q3_n2: RLWECiphertext<N2, R3> =
        ring_switch_eval::<N1, N2, R3L, L, D>(&ct_q3_n1, rsk, base_rsk);

    // Step 3: trailing asymmetric rescale — mask stays @ q3, body → q4.
    mod_switch_asym::<N2, R3, R3, R4>(&ct_q3_n2, q3_mod, q4_mod)
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
    use via_primitives::switching::rekey::rekey_secret_key;
    use via_primitives::switching::{RingSwitchKey, gen_rsk};

    type Rq<const N: usize> = Poly<N, DynModulus, Coefficient>;

    // Toy: N1=8, N2=4, D=2; ring-switch gadget (base 32, depth 6 → 32^6 = 2^30 = q3).
    const N1: usize = 8;
    const N2: usize = 4;
    const D: usize = 2;
    const L: usize = 6;
    const BASE: u64 = 32;

    /// Decrypt-correctness through the whole q2 → q3 → q4 chain: encrypt `M`
    /// under `S1@q2`, RespComp, then `decrypt_asymmetric` under `S2@q3` recovers
    /// the slot-0 projection of `M`.
    #[test]
    fn resp_comp_decrypts_to_slot0_projection() {
        let q2 = DynModulus::new(1 << 40);
        let q3 = DynModulus::new(1 << 30);
        let q4 = DynModulus::new(1 << 20);
        let p = DynModulus::new(16);

        let mut prg = Shake256Prg::new(b"resp-comp-toy");
        // S1 @ q2 (for encryption); S2 @ q3 (answer key). S1 @ q3 (for the RSK).
        let s1_q2 = SecretKey::<N1, Rq<N1>>::keygen(q2, Distribution::Ternary, &mut prg);
        let s2_q3 = SecretKey::<N2, Rq<N2>>::keygen(q3, Distribution::Ternary, &mut prg);
        let s1_q3 = rekey_secret_key::<N1, Rq<N1>, Rq<N1>>(&s1_q2, q3);
        let rsk: RingSwitchKey<N1, N2, Rq<N2>, L, D> =
            gen_rsk(&s1_q3, &s2_q3, BASE, Distribution::Ternary, &mut prg);
        let rsk_eval = rsk.to_eval();

        // Encrypt M (small coeffs in [0,p)) under S1 @ q2.
        let m_coeffs: [u64; N1] = core::array::from_fn(|i| (i as u64) % 16);
        let pt: Rq<N1> = Poly::new(p, m_coeffs);
        let ct = s1_q2.encrypt(&encode(&pt, q2), Distribution::Ternary, &mut prg);

        let answer =
            resp_comp::<N1, N2, Rq<N1>, Rq<N1>, Rq<N2>, Rq<N2>, L, D>(&ct, &rsk_eval, q3, q4, BASE);

        // decrypt_asymmetric(S2@q3, q3, q4, p) → slot-0 projection π_0(M).
        let recovered: Rq<N2> = s2_q3.decrypt_asymmetric(&answer, q3, q4, p);
        let d = N1 / N2; // = D = 2
        let expected: [u64; N2] = core::array::from_fn(|i| m_coeffs[d * i]);
        let got: [u64; N2] = core::array::from_fn(|i| recovered.coeff(i).to_u64());
        assert_eq!(got, expected, "RespComp must recover π_0(M)");
    }
}
