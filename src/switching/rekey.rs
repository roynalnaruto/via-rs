//! §3.4 constant-time secret-key rekeying. See `.docs/primitives.md` §3.4.
//!
//! Re-interpret a small-coefficient secret key $S \in R_{n, q_\text{src}}$ at
//! a new modulus $q_\text{dst}$, producing $S' \in R_{n, q_\text{dst}}$ with
//! the *same* integer coefficient vector. The coefficients are centred to
//! $(-q_\text{src}/2, q_\text{src}/2]$ and reduced mod $q_\text{dst}$;
//! well-defined when $\|S\|_\infty < q_\text{dst}/2$ (automatic for ternary /
//! bounded-uniform / narrow-Gaussian keys).
//!
//! Used so that $S_1$ (sampled at $q_1$) and $S_2$ (sampled at $q_2$) can be
//! used at the smaller moduli the compression pipeline needs — e.g. $S_1$ is
//! rekeyed to $q_3$ during `RespCompSetup` so the ring-switch key
//! ([`super::ring_switch::gen_rsk`]) can be generated under matching moduli.

use crate::algebra::ring::abstraction::RingPoly;
use crate::encryption::types::SecretKey;

/// Private dispatch over the source key's centred-scalar width. A single-prime
/// source has `CenteredScalar = i64` and lifts via
/// [`RingPoly::from_centered_i64s`]; an RNS source has `CenteredScalar = i128`
/// and lifts via [`RingPoly::from_centered_i128s`]. Keeping this internal lets
/// [`rekey_secret_key`] stay a single generic function over both backends.
#[allow(non_camel_case_types)]
trait RekeySource<const N: usize, R_DST: RingPoly<N>>: Sized {
    /// Build the destination polynomial at `dst_mod` from this scalar type's
    /// centred coefficient array.
    fn build_dst_poly(dst_mod: R_DST::Modulus, centered: &[Self; N]) -> R_DST;
}

#[allow(non_camel_case_types)]
impl<const N: usize, R_DST: RingPoly<N>> RekeySource<N, R_DST> for i64 {
    #[inline]
    fn build_dst_poly(dst_mod: R_DST::Modulus, centered: &[i64; N]) -> R_DST {
        R_DST::from_centered_i64s(dst_mod, centered)
    }
}

#[allow(non_camel_case_types)]
impl<const N: usize, R_DST: RingPoly<N>> RekeySource<N, R_DST> for i128 {
    #[inline]
    fn build_dst_poly(dst_mod: R_DST::Modulus, centered: &[i128; N]) -> R_DST {
        R_DST::from_centered_i128s(dst_mod, centered)
    }
}

/// §3.4 — rekey a secret key from its source modulus to `dst_mod`.
///
/// Centres `src_sk`'s coefficients in constant time, then reduces them mod the
/// destination modulus. Works across backends: a single-prime source
/// (`CenteredScalar = i64`) and an RNS source (`CenteredScalar = i128`) both
/// dispatch through the private `RekeySource` trait.
///
/// # Constant-time
///
/// The centring uses [`RingPoly::to_centered_coeffs_ct`] (the secret-data
/// variant), and the per-coefficient reduction is branchless. The whole path
/// is constant-time over the key, as §3.4 requires (a variable-time centring
/// would leak the key's Hamming weight through timing).
///
/// ```rust
/// use via_rs::algebra::ring::abstraction::RingPoly;
/// use via_rs::algebra::ring::element::Poly;
/// use via_rs::algebra::ring::form::Coefficient;
/// use via_rs::algebra::zq::modulus::{ConstModulus, PowerOfTwoModulus};
/// use via_rs::encryption::types::SecretKey;
/// use via_rs::sampling::distribution::Distribution;
/// use via_rs::sampling::prg::Shake256Prg;
/// use via_rs::switching::rekey::rekey_secret_key;
///
/// // Sample a ternary key at q2 = 97, rekey to q3 = 2^16.
/// type Q2 = Poly<8, ConstModulus<97>, Coefficient>;
/// type Q3 = Poly<8, PowerOfTwoModulus<16>, Coefficient>;
/// let mut prg = Shake256Prg::new(b"doc-rekey");
/// let sk_q2 = SecretKey::<8, Q2>::keygen(ConstModulus, Distribution::Ternary, &mut prg);
/// let sk_q3: SecretKey<8, Q3> = rekey_secret_key(&sk_q2, PowerOfTwoModulus);
///
/// // Centred coefficients are identical across the two moduli.
/// let mut a = [0i64; 8];
/// let mut b = [0i64; 8];
/// sk_q2.poly().to_centered_coeffs(&mut a);
/// sk_q3.poly().to_centered_coeffs(&mut b);
/// assert_eq!(a, b);
/// ```
// `RekeySource` is a deliberately private dispatch trait; every concrete
// `CenteredScalar` (i64 / i128) implements it, so external callers can still
// invoke this function — they just can't name the bound. The `private_bounds`
// lint flags the API-surface asymmetry, which is intended here.
#[allow(non_camel_case_types, private_bounds)]
pub fn rekey_secret_key<const N: usize, R_SRC: RingPoly<N>, R_DST: RingPoly<N>>(
    src_sk: &SecretKey<N, R_SRC>,
    dst_mod: R_DST::Modulus,
) -> SecretKey<N, R_DST>
where
    R_SRC::CenteredScalar: RekeySource<N, R_DST>,
{
    let mut centered = [R_SRC::CenteredScalar::default(); N];
    src_sk.poly().to_centered_coeffs_ct(&mut centered);
    let new_poly = R_SRC::CenteredScalar::build_dst_poly(dst_mod, &centered);
    SecretKey::from_poly(new_poly)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::rns::basis::paper::ViaCQ1Rns;
    use crate::algebra::zq::modulus::{ConstModulus, PowerOfTwoModulus};
    use crate::sampling::distribution::Distribution;
    use crate::sampling::prg::Shake256Prg;

    type Q2<const N: usize> = Poly<N, ConstModulus<97>, Coefficient>;
    type Q3<const N: usize> = Poly<N, PowerOfTwoModulus<16>, Coefficient>;

    /// Centred coefficients are preserved across the rekey (single-prime →
    /// single-prime).
    #[test]
    fn rekey_single_prime_coefficients_preserved() {
        let mut prg = Shake256Prg::new(b"rekey-coeffs-preserved");
        let sk_q2 = SecretKey::<4, Q2<4>>::keygen(ConstModulus, Distribution::Ternary, &mut prg);
        let sk_q3: SecretKey<4, Q3<4>> = rekey_secret_key(&sk_q2, PowerOfTwoModulus);

        let mut a = [0i64; 4];
        let mut b = [0i64; 4];
        sk_q2.poly().to_centered_coeffs_ct(&mut a);
        sk_q3.poly().to_centered_coeffs_ct(&mut b);
        assert_eq!(a, b);
    }

    /// A rekeyed key still decrypts what it encrypts (single-prime).
    #[test]
    fn rekey_single_prime_round_trip_encrypt_decrypt() {
        use crate::encryption::rlwe::encode;
        let q3m = PowerOfTwoModulus::<16>;
        let pm = PowerOfTwoModulus::<4>; // p = 16
        let mut prg = Shake256Prg::new(b"rekey-roundtrip");

        // Sample at q2 = 97, rekey to q3 = 2^16, then use at q3.
        let sk_q2 = SecretKey::<8, Q2<8>>::keygen(ConstModulus, Distribution::Ternary, &mut prg);
        let sk_q3: SecretKey<8, Q3<8>> = rekey_secret_key(&sk_q2, q3m);

        let m: [u64; 8] = [1, 2, 3, 4, 5, 6, 7, 8];
        let pt: Poly<8, PowerOfTwoModulus<4>, Coefficient> = Poly::new(pm, m);
        let encoded: Q3<8> = encode(&pt, q3m);
        let ct = sk_q3.encrypt(&encoded, Distribution::Ternary, &mut prg);
        let recovered: Poly<8, PowerOfTwoModulus<4>, Coefficient> = sk_q3.decrypt(&ct, pm);
        let got: [u64; 8] = core::array::from_fn(|i| recovered.coeff(i).to_u64());
        assert_eq!(got, m);
    }

    /// RNS source (i128 centred) → single-prime target. Verifies the i128
    /// dispatch path and that centred coefficients are preserved.
    #[test]
    fn rekey_rns_to_single_prime() {
        type RnsQ1<const N: usize> = PolyRns<N, ViaCQ1Rns, Coefficient>;
        let b = ViaCQ1Rns::default();
        let q3m = PowerOfTwoModulus::<16>;
        let mut prg = Shake256Prg::new(b"rekey-rns");

        let sk_q1 = SecretKey::<8, RnsQ1<8>>::keygen(b, Distribution::Ternary, &mut prg);
        let sk_q3: SecretKey<8, Q3<8>> = rekey_secret_key(&sk_q1, q3m);

        let mut src_centered = [0i128; 8];
        sk_q1.poly().to_centered_coeffs_ct(&mut src_centered);
        let mut dst_centered = [0i64; 8];
        sk_q3.poly().to_centered_coeffs_ct(&mut dst_centered);
        for (s, d) in src_centered.iter().zip(dst_centered.iter()) {
            assert_eq!(*s, i128::from(*d));
        }
    }

    /// The i128 kernel debug-asserts on oversize coefficients (§3.4 keys are
    /// always small; an oversize value is a caller bug).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "does not fit in i64")]
    fn rekey_ct_panics_on_oversize_coefficient() {
        use super::super::kernels::rekey::rekey_centered_i128_to_modulus_slice;
        let mut dst = [0u64; 1];
        let oversize = [i64::MAX as i128 + 1];
        rekey_centered_i128_to_modulus_slice(PowerOfTwoModulus::<16>, &mut dst, &oversize);
    }
}
