//! RLWE keygen, encode, decode, encrypt, decrypt — `.docs/primitives.md` §2.2.
//!
//! The full RLWE-symmetric primitive set, generic over a [`RingPoly`]
//! backend so the same code runs at single-prime `q` and at the RNS
//! composite `Q ≈ 2^{75}`.
//!
//! ## API shape
//!
//! Key-bearing operations are inherent methods on [`SecretKey`]:
//!
//! - [`SecretKey::keygen`] — sample from one of the §1.3-§1.5
//!   distributions.
//! - [`SecretKey::encrypt`] — encrypt an already-$\Delta$-encoded message.
//! - [`SecretKey::decrypt_raw`] — return $B - A \cdot S$ (the encoded
//!   message plus noise); used for noise inspection and the Layer-3
//!   [`SecretKey::decrypt_asymmetric`] path.
//! - [`SecretKey::decrypt`] — wrap `decrypt_raw` + [`decode`].
//! - [`SecretKey::decrypt_asymmetric`] — §3.2 `RespCompRecover`: decrypt a
//!   [`ModSwitchedCiphertext`] whose mask (at $q_3$) and body (at $q_4$)
//!   live at different moduli.
//!
//! [`encode`] and [`decode`] remain free functions: they don't touch
//! secret material (they're polynomial-side scaling / rounding operations
//! between the plaintext and ciphertext moduli).
//!
//! ## Why symmetric encryption uses `&self` on `SecretKey`
//!
//! The paper specifies symmetric RLWE: $B = A \cdot S + e + M'$ literally
//! requires $S$, so there is no public/private split. Putting `encrypt`
//! on `SecretKey` is the right architectural mirror — the client holds
//! `S` and uses it for both ends. A genuine public-key variant would be a
//! separate algorithm with different noise analysis and is not part of
//! the VIA protocol.
//!
//! ## Python parity
//!
//! Every primitive in this module mirrors `pir/primitives/rlwe.py`
//! line-for-line. The non-obvious bits:
//!
//! - `Δ = -(-q // p)` (`encode`) becomes `q.div_ceil(p)` — same ceil-div
//!   semantics, integer-only.
//! - The decode rounding step `(p * c_centered + q // 2) // q` uses
//!   [`i128::div_euclid`], **not** `/` — Python's `//` floors toward
//!   $-\infty$; Rust's `/` truncates toward zero. They disagree at exactly
//!   the negative-numerator boundary cases noise correction provokes.
//! - [`SecretKey::encrypt`] consumes PRG bytes in the order **mask `A`
//!   first, error `e` second**, matching `rlwe.py:137-173`. Reversing the
//!   order would silently desynchronise every encrypt test vector.

use crate::algebra::ring::RingPoly;
use crate::sampling::distribution::Distribution;
use crate::sampling::prg::Shake256Prg;

use super::types::{ModSwitchedCiphertext, RLWECiphertext, SecretKey};

// ---------------------------------------------------------------------------
// Key-bearing primitives — methods on `SecretKey<N, R>` (§2.2).
// ---------------------------------------------------------------------------

impl<const N: usize, R: RingPoly<N>> SecretKey<N, R> {
    /// Sample a secret key from one of the §1.3-§1.5 distributions
    /// ([`Distribution::Ternary`], [`Distribution::BoundedUniform`], or
    /// [`Distribution::Gaussian`]) and lift the signed samples into the
    /// ring $R_{n, q}$ via [`RingPoly::from_centered_i64s`].
    ///
    /// PRG consumption matches the Layer-1 sampler exactly, so seeding
    /// the [`Shake256Prg`] identically to the Python reference yields
    /// byte-identical secret keys.
    pub fn keygen(modulus: R::Modulus, dist: Distribution, prg: &mut Shake256Prg) -> Self {
        let mut samples = [0i64; N];
        dist.sample_into(prg, &mut samples);
        let poly = R::from_centered_i64s(modulus, &samples);
        SecretKey::from_poly(poly)
    }

    /// §2.2 — encrypt an already-$\Delta$-encoded message $M' \in R_{n, q}$
    /// under this secret key.
    ///
    /// Algorithm:
    ///
    /// 1. Sample $A \leftarrow \mathcal{U}(R_{n, q})$ via
    ///    [`RingPoly::random_uniform`].
    /// 2. Sample $e \leftarrow \chi_e$ via the supplied
    ///    [`Distribution`], lifted into the ring.
    /// 3. Return $(A, \; A \cdot S + e + M')$.
    ///
    /// # PRG consumption order
    ///
    /// `A` is sampled **before** `e`. The Python reference does the same;
    /// reversing this order would silently break every encrypt parity
    /// test vector.
    ///
    /// # Constant-time
    ///
    /// The secret-dependent step $A \cdot S$ is constant-time over $S$:
    /// it forwards to the schoolbook `negacyclic_mul_slice`, whose scalar
    /// kernel is Barrett-reduced (data-independent). Addition of $e$ and
    /// $M'$ is component-wise and equally CT.
    ///
    /// # Argument
    ///
    /// `encoded` is the **$\Delta$-scaled** message, not the raw
    /// plaintext: call [`encode`] first if you have an unscaled
    /// plaintext. The bare API matches the convention from §2.1 ("RLev
    /// re-uses `encrypt` with gadget-scaled messages"); folding `encode`
    /// in would break the Phase-6 RLev path.
    pub fn encrypt(
        &self,
        encoded: &R,
        error_dist: Distribution,
        prg: &mut Shake256Prg,
    ) -> RLWECiphertext<N, R> {
        let modulus = encoded.modulus();
        debug_assert!(
            self.poly.modulus() == modulus,
            "encrypt: secret-key modulus must match encoded-message modulus"
        );
        // Step 1 — mask `A` first (order matters for Python parity).
        let mask = R::random_uniform(modulus, prg);
        // Step 2 — error `e` second.
        let mut error_samples = [0i64; N];
        error_dist.sample_into(prg, &mut error_samples);
        let error = R::from_centered_i64s(modulus, &error_samples);
        // Step 3 — body `B = A·S + e + M'`. `Mul` on Coefficient form is
        // schoolbook negacyclic; `Add` is componentwise.
        let body = mask * self.poly + error + *encoded;
        RLWECiphertext::new(mask, body)
    }

    /// §2.2 — return $B - A \cdot S$, the encoded message plus
    /// decryption noise. Useful for direct noise inspection and as the
    /// inner half of the Layer-3 `decrypt_asymmetric` path; ordinary
    /// callers want [`SecretKey::decrypt`] instead.
    ///
    /// Constant-time over $S$ via the same argument as
    /// [`SecretKey::encrypt`].
    pub fn decrypt_raw(&self, ct: &RLWECiphertext<N, R>) -> R {
        debug_assert!(
            self.poly.modulus() == ct.mask.modulus() && self.poly.modulus() == ct.body.modulus(),
            "decrypt_raw: secret-key modulus must match ciphertext modulus"
        );
        ct.body - ct.mask * self.poly
    }

    /// §2.2 — full decryption: recover the plaintext at modulus $p$.
    ///
    /// Composition of [`SecretKey::decrypt_raw`] (which yields the
    /// noisy encoded message at $q$) and the free function [`decode`]
    /// (which centres, rounds to the nearest multiple of $\Delta$, and
    /// reduces mod $p$).
    ///
    /// `RP` is the plaintext-side polynomial backend; the [`decode`]
    /// step's variable-time centring is acceptable here because the
    /// decrypted value is RLWE-uniform under the §0.6 / §2.2 security
    /// argument.
    pub fn decrypt<RP: RingPoly<N>>(&self, ct: &RLWECiphertext<N, R>, p_mod: RP::Modulus) -> RP {
        let raw = self.decrypt_raw(ct);
        decode::<N, R, RP>(&raw, p_mod)
    }

    /// §3.2 — decrypt a [`ModSwitchedCiphertext`] whose mask and body live
    /// at **different** moduli (paper Figure 7, `RespCompRecover`). The mask
    /// is at $q_3$, the body at $q_4$; the secret key (this `SecretKey`,
    /// originally sampled at $q_2$ / $q_3$) is centred and re-interpreted at
    /// $q_3$ to compute $A' \cdot S$, which is then rescaled $q_3 \to q_4$:
    ///
    /// $$
    /// \hat m = \Bigl\lfloor p \cdot \bigl(B' - \lfloor q_4 \cdot A' \cdot S
    ///   / q_3 \rceil\bigr) / q_4 \Bigr\rceil \bmod p.
    /// $$
    ///
    /// # Single-prime secret key
    ///
    /// The `where R: RingPoly<N, CenteredScalar = i64>` bound restricts this
    /// method to single-prime secret keys, which matches every paper-spec
    /// VIA / VIA-C / VIA-B $S_2$ distribution (they live at the single-prime
    /// $q_2$ / $q_3$, never at an RNS composite). A future RNS-source $S_2$
    /// would have to dispatch through §3.4 `RekeySource` before the centred
    /// lift.
    ///
    /// # Constant-time
    ///
    /// The centred lift of the secret key uses the constant-time
    /// [`RingPoly::to_centered_coeffs_ct`]; the subsequent rescale operates
    /// on ciphertext (RLWE-uniform) coefficients and the final [`decode`] is
    /// variable-time on the about-to-be-revealed plaintext, both acceptable
    /// under the §0.6 timing argument.
    pub fn decrypt_asymmetric<RM: RingPoly<N>, RB: RingPoly<N>, RP: RingPoly<N>>(
        &self,
        ct: &ModSwitchedCiphertext<N, RM, RB>,
        q3_mod: RM::Modulus,
        q4_mod: RB::Modulus,
        p_mod: RP::Modulus,
    ) -> RP
    where
        R: RingPoly<N, CenteredScalar = i64>,
    {
        // Step 1: centre S to (-q_src/2, q_src/2] in constant time (secret).
        let mut centered = [0i64; N];
        self.poly.to_centered_coeffs_ct(&mut centered);
        // Step 2: re-interpret S at q3 (the mask modulus).
        let sk_q3: RM = RM::from_centered_i64s(q3_mod, &centered);
        // Step 3: product = A' · S in R_{n, q3}.
        let product = ct.mask * sk_q3;
        // Step 4: rescale product q3 → q4 per-coefficient. Inline integer
        // arithmetic keeps `encryption/` free of any `switching/` import
        // (the `RescaleConsts` helper lives in Layer 3).
        let q3 = RM::modulus_value(q3_mod);
        let q4 = RB::modulus_value(q4_mod);
        let q3_half = q3 / 2;
        let mut product_u128 = [0u128; N];
        product.to_u128_coeffs(&mut product_u128);
        for v in product_u128.iter_mut() {
            *v = (*v * q4 + q3_half) / q3;
        }
        let switched: RB = RB::from_u128_coeffs(q4_mod, &product_u128);
        // Step 5: noisy = B' - switched in R_{n, q4}.
        let noisy = ct.body - switched;
        // Step 6: decode to plaintext at p.
        decode::<N, RB, RP>(&noisy, p_mod)
    }
}

// ---------------------------------------------------------------------------
// encode — §2.2.1
// ---------------------------------------------------------------------------

/// Lift a plaintext polynomial $m \in R_{n, p}$ to its $\Delta$-scaled
/// encoded form $M' = \Delta \cdot m \in R_{n, q}$, where
/// $\Delta = \lceil q / p \rceil$.
///
/// Both source ring `RP` and target ring `RQ` are generic [`RingPoly`]
/// instantiations. In paper-parameter use `RP` is single-prime
/// `Poly<N, MP, Coefficient>` (with $p \in \{16, 256\}$); the RNS form is
/// supported but never exercised at paper parameters.
///
/// # Range bound
///
/// Computation runs in `u128`. With paper parameters $Q \le 2^{75}$ and
/// $p \le 256$, every intermediate fits comfortably (no overflow
/// possible).
///
/// # Panics
///
/// Panics in debug builds if $p > Q$ — that would make $\Delta < 1$ and
/// the encoding non-injective. Release builds silently produce
/// $\Delta = 1$ via [`u128::div_ceil`] semantics.
pub fn encode<const N: usize, RQ: RingPoly<N>, RP: RingPoly<N>>(
    plaintext: &RP,
    q_mod: RQ::Modulus,
) -> RQ {
    let p = RP::modulus_value(plaintext.modulus());
    let q = RQ::modulus_value(q_mod);
    debug_assert!(p >= 2, "encode: plaintext modulus must be at least 2");
    debug_assert!(
        p <= q,
        "encode: plaintext modulus exceeds ciphertext modulus"
    );

    let delta = q.div_ceil(p);

    let mut p_coeffs = [0u128; N];
    plaintext.to_u128_coeffs(&mut p_coeffs);

    let mut scaled = [0u128; N];
    for i in 0..N {
        // c * delta is bounded by `p * delta = p * ceil(q/p) <= q + p`,
        // which fits in u128 trivially for any realistic parameters.
        scaled[i] = (p_coeffs[i] * delta) % q;
    }

    RQ::from_u128_coeffs(q_mod, &scaled)
}

// ---------------------------------------------------------------------------
// decode — §2.2.2
// ---------------------------------------------------------------------------

/// Recover a plaintext polynomial in $R_{n, p}$ from a (possibly noisy)
/// encoded message in $R_{n, q}$.
///
/// For each coefficient:
///
/// 1. Centre to $\tilde c \in (-\lfloor Q/2 \rfloor, \lfloor Q/2 \rfloor]$
///    via [`RingPoly::to_centered_i128_coeffs`].
/// 2. Round $\hat m = \big\lfloor \dfrac{p \cdot \tilde c}{q} \big\rceil$
///    via the Python-compatible flooring formula
///    `(p * c_centered + q / 2).div_euclid(q)`. Using `i128::div_euclid`
///    (rather than `/`) is load-bearing: Python's `//` floors toward
///    $-\infty$, so for negative numerators the two operators disagree at
///    exactly the boundary cases that arise during noise correction.
/// 3. Reduce $\hat m \bmod p$ via [`i128::rem_euclid`].
///
/// All arithmetic is integer-only. With paper parameters
/// ($p \le 256$, $|c_\text{centered}| \le Q/2 \le 2^{74}$) the product
/// $p \cdot c$ fits in `i128` with ~45 bits of headroom.
pub fn decode<const N: usize, RQ: RingPoly<N>, RP: RingPoly<N>>(
    scaled: &RQ,
    p_mod: RP::Modulus,
) -> RP {
    let q = RQ::modulus_value(scaled.modulus()) as i128;
    let p = RP::modulus_value(p_mod) as i128;
    debug_assert!(p >= 2, "decode: plaintext modulus must be at least 2");
    debug_assert!(
        p <= q,
        "decode: plaintext modulus exceeds ciphertext modulus"
    );

    let half_q = q / 2;

    let mut centered = [0i128; N];
    scaled.to_centered_i128_coeffs(&mut centered);

    let mut decoded = [0u128; N];
    for i in 0..N {
        let c = centered[i];
        debug_assert!(
            p.checked_mul(c.abs()).is_some(),
            "decode: p * |c_centered| overflows i128"
        );
        let numerator = p * c;
        // Python: `(p * c_centered + q // 2) // q`. Rust's `/` truncates
        // toward zero; we need Python's floor-toward-minus-infinity, hence
        // `div_euclid`.
        let rounded = (numerator + half_q).div_euclid(q);
        // Python: `% p` always returns non-negative for positive p.
        decoded[i] = rounded.rem_euclid(p) as u128;
    }

    RP::from_u128_coeffs(p_mod, &decoded)
}

// ---------------------------------------------------------------------------
// Auxiliary RLWE primitives — §2.2.5.
//
// These are coordinate-level operations on `RLWECiphertext` that the
// protocol uses repeatedly (DMux's `c - C ⊠ c`, CMux's `c₀ + C ⊠ (c₁ -
// c₀)`, the negation half of VIA-C's CRot, `FirstDim`'s plaintext-times-
// ciphertext mul, and the trivial RLWE root value of VIA-C's DMux tree).
// Every one forwards componentwise to the underlying `RingPoly` arithmetic;
// no key material involved.
// ---------------------------------------------------------------------------

impl<const N: usize, R: RingPoly<N>> core::ops::Add for RLWECiphertext<N, R> {
    type Output = Self;

    /// Component-wise sum: $(A_1, B_1) + (A_2, B_2) = (A_1 + A_2,
    /// B_1 + B_2)$. Decrypts under the **same key** to $m_1 + m_2 \bmod p$
    /// — the noise terms sum too, so the caller must budget for noise
    /// growth across repeated additions.
    fn add(self, rhs: Self) -> Self {
        Self::new(self.mask + rhs.mask, self.body + rhs.body)
    }
}

impl<const N: usize, R: RingPoly<N>> core::ops::Sub for RLWECiphertext<N, R> {
    type Output = Self;

    /// Component-wise difference. Decrypts to $m_1 - m_2 \bmod p$.
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.mask - rhs.mask, self.body - rhs.body)
    }
}

impl<const N: usize, R: RingPoly<N>> core::ops::Neg for RLWECiphertext<N, R> {
    type Output = Self;

    /// Component-wise negation. Decrypts to $-m \bmod p$.
    fn neg(self) -> Self {
        Self::new(-self.mask, -self.body)
    }
}

impl<const N: usize, R: RingPoly<N>> core::ops::AddAssign for RLWECiphertext<N, R> {
    fn add_assign(&mut self, rhs: Self) {
        self.mask += rhs.mask;
        self.body += rhs.body;
    }
}

impl<const N: usize, R: RingPoly<N>> core::ops::SubAssign for RLWECiphertext<N, R> {
    fn sub_assign(&mut self, rhs: Self) {
        self.mask -= rhs.mask;
        self.body -= rhs.body;
    }
}

impl<const N: usize, R: RingPoly<N>> core::ops::Mul<R> for RLWECiphertext<N, R> {
    type Output = Self;

    /// $\text{ct} \cdot f = (f \cdot A, \; f \cdot B)$. Decrypts to
    /// $f \cdot m \bmod p$.
    ///
    /// The plaintext factor $f$ must have **small infinity-norm** — the
    /// noise $f \cdot e$ in the result grows by $\|f\|_1$ in the worst
    /// case. With paper-class parameters and a single-monomial $f$
    /// (FirstDim's typical pattern) noise grows by 1; high-weight $f$
    /// requires larger $q$ headroom.
    ///
    /// `R` is `Copy` (via the [`RingPoly`] supertrait bounds), so the
    /// reuse of `f` for both factor multiplications is free.
    fn mul(self, f: R) -> Self {
        Self::new(self.mask * f, self.body * f)
    }
}

impl<const N: usize, R: RingPoly<N>> RLWECiphertext<N, R> {
    /// Construct a **trivial RLWE ciphertext** $(0, M')$ — a noiseless
    /// encryption that decrypts to $M' \cdot \Delta^{-1}$ (i.e., the
    /// plaintext recovered by `decode(M', p)`) under **any** secret key.
    ///
    /// The body carries the message; the zero mask contributes nothing
    /// to decryption regardless of `S`. VIA-C feeds
    /// `trivial(q_1, ⌊q₁ / p⌉ · 1)` into the DMux tree as the root value
    /// (`.docs/primitives.md` §2.2.5), so the server doesn't need a fresh
    /// encryption of $\Delta$ from the client.
    ///
    /// `encoded` is the **$\Delta$-scaled** message, matching the
    /// `SecretKey::encrypt` convention. Use [`encode`] first if you have
    /// an unscaled plaintext.
    #[inline]
    pub fn trivial(modulus: R::Modulus, encoded: &R) -> Self {
        Self::new(R::zero(modulus), *encoded)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::rns::basis::ConstRnsBasis;
    use crate::algebra::zq::modulus::{ConstModulus, PowerOfTwoModulus};
    // Toy backends used across the test suite.
    //
    // For encode/decode to round-trip correctly we need q sufficiently
    // larger than p — specifically the ceiling-induced slack in
    // `delta = ceil(q/p)` must not exceed half a `delta`-step in decoded
    // space. With `q = 1024 = 2^10` and `p in {2, 16, 256}` we get
    // exact-integer `delta = q/p`, eliminating that slack.
    type SinglePolyQ17<const N: usize> = Poly<N, ConstModulus<17>, Coefficient>;
    type SinglePolyQ1024<const N: usize> = Poly<N, PowerOfTwoModulus<10>, Coefficient>;
    type SinglePolyP2<const N: usize> = Poly<N, PowerOfTwoModulus<1>, Coefficient>;
    type SinglePolyP16<const N: usize> = Poly<N, PowerOfTwoModulus<4>, Coefficient>;
    type RnsPoly5x11<const N: usize> = PolyRns<N, ConstRnsBasis<5, 11>, Coefficient>;
    // For p=16 round-trips, both primes large enough so Q >> p^2.
    type RnsPoly127x251<const N: usize> = PolyRns<N, ConstRnsBasis<127, 251>, Coefficient>;

    // -----------------------------------------------------------------------
    // keygen
    // -----------------------------------------------------------------------

    /// Keygen wires through to the §1.3 ternary sampler. The Layer-1
    /// parity test (`src/sampling/ternary.rs:67`) fixes the byte-level
    /// output of `Shake256Prg::new(b"test")` + ternary sampling at N=16
    /// to be `[1, -1, 1, -1, -1, 0, -1, -1, 1, -1, -1, 0, -1, 0, 1, -1]`.
    /// The SecretKey poly's centred coefficients must match.
    #[test]
    fn keygen_ternary_parity_with_layer1_at_n16_single_prime() {
        let mut prg = Shake256Prg::new(b"test");
        let sk = SecretKey::<16, SinglePolyQ17<16>>::keygen(
            ConstModulus,
            Distribution::Ternary,
            &mut prg,
        );
        let mut centered = [0i64; 16];
        sk.poly().to_centered_coeffs(&mut centered);
        let expected: [i64; 16] = [1, -1, 1, -1, -1, 0, -1, -1, 1, -1, -1, 0, -1, 0, 1, -1];
        assert_eq!(centered, expected);
    }

    /// Same parity guarantee on the RNS backend — the centred lift
    /// reduces the same i64 vector against each prime, so the centred
    /// coefficients of the resulting PolyRns must match.
    #[test]
    fn keygen_ternary_parity_with_layer1_at_n16_rns() {
        let mut prg = Shake256Prg::new(b"test");
        let sk = SecretKey::<16, RnsPoly5x11<16>>::keygen(
            ConstRnsBasis,
            Distribution::Ternary,
            &mut prg,
        );
        let mut centered = [0i128; 16];
        sk.poly().to_centered_coeffs(&mut centered);
        let expected: [i128; 16] = [1, -1, 1, -1, -1, 0, -1, -1, 1, -1, -1, 0, -1, 0, 1, -1];
        assert_eq!(centered, expected);
    }

    /// Independent keygen invocations with distinct seeds should diverge.
    #[test]
    fn keygen_with_distinct_seeds_diverges() {
        let mut prg_a = Shake256Prg::new(b"keygen-A");
        let mut prg_b = Shake256Prg::new(b"keygen-B");
        let sk_a = SecretKey::<32, SinglePolyQ17<32>>::keygen(
            ConstModulus,
            Distribution::Ternary,
            &mut prg_a,
        );
        let sk_b = SecretKey::<32, SinglePolyQ17<32>>::keygen(
            ConstModulus,
            Distribution::Ternary,
            &mut prg_b,
        );
        let mut a = [0i64; 32];
        let mut b = [0i64; 32];
        sk_a.poly().to_centered_coeffs(&mut a);
        sk_b.poly().to_centered_coeffs(&mut b);
        assert_ne!(a, b);
    }

    // -----------------------------------------------------------------------
    // encode / decode — round-trip
    // -----------------------------------------------------------------------

    /// Build a plaintext polynomial from raw integer coefficients in `[0, p)`.
    fn pt_from_vec<const N: usize, MP: crate::algebra::zq::modulus::Modulus>(
        p_mod: MP,
        coeffs: [u64; N],
    ) -> Poly<N, MP, Coefficient> {
        Poly::new(p_mod, coeffs)
    }

    /// All possible plaintexts in $R_{n=4, p=2}$ must round-trip through
    /// encode/decode at q = 17 single-prime.
    #[test]
    fn encode_decode_roundtrip_p2_q17_n4_exhaustive() {
        let p = PowerOfTwoModulus::<1>; // p = 2
        let q = ConstModulus::<17>;
        for bits in 0u8..16 {
            let coeffs = [
                u64::from(bits & 1),
                u64::from((bits >> 1) & 1),
                u64::from((bits >> 2) & 1),
                u64::from((bits >> 3) & 1),
            ];
            let pt: SinglePolyP2<4> = pt_from_vec(p, coeffs);
            let enc: SinglePolyQ17<4> = encode(&pt, q);
            let dec: SinglePolyP2<4> = decode(&enc, p);
            for (i, &expected) in coeffs.iter().enumerate() {
                assert_eq!(dec.coeff(i).to_u64(), expected, "bits={bits} i={i}");
            }
        }
    }

    /// $p = 16$, $q = 1024$ — clean integer $\Delta = 64$.
    #[test]
    fn encode_decode_roundtrip_p16_q1024_n4() {
        let p = PowerOfTwoModulus::<4>; // p = 16
        let q = PowerOfTwoModulus::<10>; // q = 1024
        let coeffs = [0u64, 1, 7, 15];
        let pt: SinglePolyP16<4> = pt_from_vec(p, coeffs);
        let enc: SinglePolyQ1024<4> = encode(&pt, q);
        let dec: SinglePolyP16<4> = decode(&enc, p);
        for (i, &expected) in coeffs.iter().enumerate() {
            assert_eq!(dec.coeff(i).to_u64(), expected);
        }
    }

    /// $p = 16$, RNS $Q = 127 \cdot 251 = 31{,}877$ — exercise the RNS
    /// encode path with $Q \gg p^2$.
    #[test]
    fn encode_decode_roundtrip_p16_rns31877_n4() {
        let p = PowerOfTwoModulus::<4>;
        let basis = ConstRnsBasis::<127, 251>;
        let coeffs = [0u64, 3, 8, 15];
        let pt: SinglePolyP16<4> = pt_from_vec(p, coeffs);
        let enc: RnsPoly127x251<4> = encode(&pt, basis);
        let dec: SinglePolyP16<4> = decode(&enc, p);
        for (i, &expected) in coeffs.iter().enumerate() {
            assert_eq!(dec.coeff(i).to_u64(), expected);
        }
    }

    /// Exhaustive $p = 16$ round-trip at $q = 1024$ — every $m \in [0,16)$
    /// in every slot of an $N = 1$-ish position must round-trip.
    #[test]
    fn encode_decode_roundtrip_p16_q1024_exhaustive() {
        let p = PowerOfTwoModulus::<4>;
        let q = PowerOfTwoModulus::<10>;
        for m in 0u64..16 {
            let pt: SinglePolyP16<4> = pt_from_vec(p, [m, m, m, m]);
            let enc: SinglePolyQ1024<4> = encode(&pt, q);
            let dec: SinglePolyP16<4> = decode(&enc, p);
            for i in 0..4 {
                assert_eq!(dec.coeff(i).to_u64(), m, "m={m} i={i}");
            }
        }
    }

    /// VIA paper plaintext modulus $p = 256$, paired with the VIA-C
    /// ciphertext modulus $q_2 = 17175674881 \approx 2^{34}$.
    #[test]
    fn encode_decode_roundtrip_p256_via_c_q2_n4() {
        use crate::algebra::zq::modulus::paper::ViaCQ2;
        type SinglePolyP256<const N: usize> = Poly<N, PowerOfTwoModulus<8>, Coefficient>;
        type SinglePolyViaCQ2<const N: usize> = Poly<N, ViaCQ2, Coefficient>;
        let p = PowerOfTwoModulus::<8>; // p = 256
        let q = ViaCQ2::default();
        let coeffs = [0u64, 1, 127, 255];
        let pt: SinglePolyP256<4> = pt_from_vec(p, coeffs);
        let enc: SinglePolyViaCQ2<4> = encode(&pt, q);
        let dec: SinglePolyP256<4> = decode(&enc, p);
        for (i, &expected) in coeffs.iter().enumerate() {
            assert_eq!(dec.coeff(i).to_u64(), expected);
        }
    }

    /// Encode produces $\Delta \cdot c$ at every coefficient. Verify the
    /// scale factor exactly for $p = 2$, $q = 17$ (so $\Delta = 9$).
    #[test]
    fn encode_scales_each_coefficient_by_delta() {
        let p = PowerOfTwoModulus::<1>;
        let q = ConstModulus::<17>;
        // m = 1 + X + 0*X^2 + 1*X^3
        let pt: SinglePolyP2<4> = pt_from_vec(p, [1, 1, 0, 1]);
        let enc: SinglePolyQ17<4> = encode(&pt, q);
        // delta = ceil(17 / 2) = 9
        for i in 0..4 {
            let expected = if i == 2 { 0 } else { 9 };
            assert_eq!(enc.coeff(i).to_u64(), expected, "i={i}");
        }
    }

    // -----------------------------------------------------------------------
    // decode rounding matches Python's flooring `//` for negative numerator
    // -----------------------------------------------------------------------

    /// The decode rounding step at a value just past the half-bin must
    /// pick the right neighbour. With q = 17 and p = 2:
    /// - `delta = 9`, `half_q = 8`.
    /// - For a noisy coefficient at `Δ * 1 + δ` where `δ ∈ {-4, ..., 4}`,
    ///   decoding must still recover `1`.
    /// - For `δ` such that the centred value is exactly `-q/2`, Python's
    ///   `(p * (-q/2) + q/2) // q = (p * -q/2 + q/2) // q`. With p=2,
    ///   q=17, half_q=8: `(2 * -8 + 8) // 17 = -8 // 17 = -1`, so
    ///   `(-1) % 2 = 1`. Test that we agree.
    #[test]
    fn decode_rounding_matches_python_at_negative_boundary() {
        let p = PowerOfTwoModulus::<1>;
        // Build a ciphertext whose centred coefficients are exactly the
        // tricky boundary values [-8, -4, 4, 8] at modulus q = 17.
        let centred = [-8i64, -4, 4, 8];
        let ct = <SinglePolyQ17<4>>::from_centered_i64s(ConstModulus, &centred);
        let dec: SinglePolyP2<4> = decode(&ct, p);
        // Hand-compute Python's `(p * c + half_q) // q  %  p`:
        //   c=-8 : (-16 + 8) // 17 = -1 // 17 = -1 -> -1 % 2 = 1
        //   c=-4 : ( -8 + 8) // 17 =  0 // 17 =  0 ->  0 % 2 = 0
        //   c= 4 : (  8 + 8) // 17 = 16 // 17 =  0 ->  0 % 2 = 0
        //   c= 8 : ( 16 + 8) // 17 = 24 // 17 =  1 ->  1 % 2 = 1
        let expected = [1u64, 0, 0, 1];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(dec.coeff(i).to_u64(), exp, "i={i}");
        }
    }

    // -----------------------------------------------------------------------
    // keygen at Gaussian + bounded uniform exercises the dispatcher
    // -----------------------------------------------------------------------

    #[test]
    fn keygen_gaussian_runs_without_panic() {
        let mut prg = Shake256Prg::new(b"keygen-gaussian");
        let _sk = SecretKey::<16, SinglePolyQ17<16>>::keygen(
            ConstModulus,
            Distribution::Gaussian { sigma: 1.5 },
            &mut prg,
        );
    }

    #[test]
    fn keygen_bounded_uniform_runs_without_panic() {
        let mut prg = Shake256Prg::new(b"keygen-bounded");
        let _sk = SecretKey::<16, SinglePolyQ17<16>>::keygen(
            ConstModulus,
            Distribution::BoundedUniform { bound: 2 },
            &mut prg,
        );
    }

    // =======================================================================
    // Phase 3 — encrypt / decrypt_raw / decrypt
    // =======================================================================

    // For the round-trip grid we need both backends + a paper-class modulus.
    type SinglePolyViaCQ2<const N: usize> =
        Poly<N, crate::algebra::zq::modulus::paper::ViaCQ2, Coefficient>;
    type RnsPolyViaCQ1<const N: usize> =
        PolyRns<N, crate::algebra::rns::basis::paper::ViaCQ1Rns, Coefficient>;
    type SinglePolyP256<const N: usize> = Poly<N, PowerOfTwoModulus<8>, Coefficient>;

    // -----------------------------------------------------------------------
    // PRG-order pin: encrypt samples mask first, then error.
    // -----------------------------------------------------------------------

    /// Re-runs encrypt's internal sampling sequence by hand against a
    /// freshly-seeded PRG and asserts the produced (mask, body) match
    /// the corresponding output of `sk.encrypt(...)` byte-for-byte.
    ///
    /// If the order ever flips to "error first, then mask", `A_enc` and
    /// `A_manual` diverge and this test fails — protecting Python byte
    /// parity for every encrypt-flavoured test vector.
    #[test]
    fn encrypt_consumes_prg_in_order_mask_then_error() {
        let q = ConstModulus::<17>;
        // Build the secret key from its own PRG so the key-bytes don't
        // contaminate either side of the encrypt-vs-manual comparison.
        let mut sk_prg = Shake256Prg::new(b"sk-seed");
        let sk = SecretKey::<8, SinglePolyQ17<8>>::keygen(q, Distribution::Ternary, &mut sk_prg);

        let encoded = <SinglePolyQ17<8>>::zero(q); // M' = 0 so B == A·S + e.

        // Side A: encrypt against `enc_prg`.
        let mut enc_prg = Shake256Prg::new(b"encrypt-order");
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);

        // Side B: manually drive a separately-seeded `manual_prg` in the
        // documented order — mask first, then error.
        let mut manual_prg = Shake256Prg::new(b"encrypt-order");
        let mask_manual = <SinglePolyQ17<8> as RingPoly<8>>::random_uniform(q, &mut manual_prg);
        let mut error_samples = [0i64; 8];
        Distribution::Ternary.sample_into(&mut manual_prg, &mut error_samples);
        let error_manual = <SinglePolyQ17<8>>::from_centered_i64s(q, &error_samples);
        let body_manual = mask_manual * sk.poly + error_manual + encoded;

        assert_eq!(ct.mask, mask_manual, "encrypt's mask diverged from manual");
        assert_eq!(ct.body, body_manual, "encrypt's body diverged from manual");
    }

    #[test]
    fn encrypt_with_distinct_prgs_diverges() {
        let q = PowerOfTwoModulus::<10>;
        let mut sk_prg = Shake256Prg::new(b"sk");
        let sk = SecretKey::<8, SinglePolyQ1024<8>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let m: SinglePolyP2<8> = Poly::new(PowerOfTwoModulus, [1, 0, 1, 0, 1, 1, 0, 0]);
        let encoded: SinglePolyQ1024<8> = encode(&m, q);

        let mut prg_a = Shake256Prg::new(b"enc-A");
        let mut prg_b = Shake256Prg::new(b"enc-B");
        let ct_a = sk.encrypt(&encoded, Distribution::Ternary, &mut prg_a);
        let ct_b = sk.encrypt(&encoded, Distribution::Ternary, &mut prg_b);
        assert!(ct_a.mask != ct_b.mask || ct_a.body != ct_b.body);
    }

    // -----------------------------------------------------------------------
    // Round-trip grid: encrypt → decrypt → decode == original plaintext.
    // -----------------------------------------------------------------------

    /// Toy at the smallest possible parameter set: q = 17, p = 2, N = 4.
    /// Δ = 9 is on the edge of the noise budget; this test pins that the
    /// ternary-error path actually fits.
    #[test]
    fn encrypt_decrypt_roundtrip_q17_p2_n4_ternary() {
        let q = ConstModulus::<17>;
        let p = PowerOfTwoModulus::<1>;
        let mut sk_prg = Shake256Prg::new(b"sk-q17-p2");
        let sk = SecretKey::<4, SinglePolyQ17<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-q17-p2");
        for bits in 0u8..16 {
            let coeffs = [
                u64::from(bits & 1),
                u64::from((bits >> 1) & 1),
                u64::from((bits >> 2) & 1),
                u64::from((bits >> 3) & 1),
            ];
            let m: SinglePolyP2<4> = Poly::new(p, coeffs);
            let encoded: SinglePolyQ17<4> = encode(&m, q);
            let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
            let recovered: SinglePolyP2<4> = sk.decrypt(&ct, p);
            for (i, &expected) in coeffs.iter().enumerate() {
                assert_eq!(recovered.coeff(i).to_u64(), expected, "bits={bits} i={i}");
            }
        }
    }

    /// p = 2, q = 1024 — generous Δ = 512, plenty of headroom.
    #[test]
    fn encrypt_decrypt_roundtrip_q1024_p2_n4_ternary() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        let mut sk_prg = Shake256Prg::new(b"sk-q1024-p2");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-q1024-p2");
        let m: SinglePolyP2<4> = Poly::new(p, [1, 0, 1, 1]);
        let encoded: SinglePolyQ1024<4> = encode(&m, q);
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
        let recovered: SinglePolyP2<4> = sk.decrypt(&ct, p);
        for i in 0..4 {
            assert_eq!(recovered.coeff(i).to_u64(), m.coeff(i).to_u64());
        }
    }

    /// p = 16, q = 1024, exhaustive across `m ∈ [0, 16)` per slot.
    #[test]
    fn encrypt_decrypt_roundtrip_q1024_p16_n4_ternary_exhaustive() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-q1024-p16");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-q1024-p16");
        for m_val in 0u64..16 {
            let m: SinglePolyP16<4> = Poly::new(p, [m_val, m_val, m_val, m_val]);
            let encoded: SinglePolyQ1024<4> = encode(&m, q);
            let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
            let recovered: SinglePolyP16<4> = sk.decrypt(&ct, p);
            for i in 0..4 {
                assert_eq!(recovered.coeff(i).to_u64(), m_val, "m_val={m_val} i={i}");
            }
        }
    }

    /// Paper-class single-prime: q = VIA-C q_2 (≈2^34), p = 256, Gaussian
    /// error. Uses N=16 to stretch the schoolbook poly mul a bit while
    /// keeping the test fast.
    #[test]
    fn encrypt_decrypt_roundtrip_via_c_q2_p256_n16_gaussian() {
        let q = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        let p = PowerOfTwoModulus::<8>;
        let mut sk_prg = Shake256Prg::new(b"sk-via-c-q2");
        let sk =
            SecretKey::<16, SinglePolyViaCQ2<16>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-via-c-q2");
        let coeffs: [u64; 16] = [
            0, 1, 13, 31, 63, 127, 200, 255, 7, 42, 99, 137, 200, 250, 5, 17,
        ];
        let m: SinglePolyP256<16> = Poly::new(p, coeffs);
        let encoded: SinglePolyViaCQ2<16> = encode(&m, q);
        let ct = sk.encrypt(
            &encoded,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let recovered: SinglePolyP256<16> = sk.decrypt(&ct, p);
        for (i, &expected) in coeffs.iter().enumerate() {
            assert_eq!(recovered.coeff(i).to_u64(), expected, "i={i}");
        }
    }

    /// RNS round-trip: Q = 127·251 = 31877, p = 16, N = 4, Ternary error.
    /// Exercises the RNS `A·S`, `from_centered_i64s`, and decode's
    /// always-i128 centred lift through the abstraction.
    #[test]
    fn encrypt_decrypt_roundtrip_rns31877_p16_n4_ternary() {
        let basis = ConstRnsBasis::<127, 251>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-rns31877");
        let sk =
            SecretKey::<4, RnsPoly127x251<4>>::keygen(basis, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-rns31877");
        let coeffs = [0u64, 3, 8, 15];
        let m: SinglePolyP16<4> = Poly::new(p, coeffs);
        let encoded: RnsPoly127x251<4> = encode(&m, basis);
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
        let recovered: SinglePolyP16<4> = sk.decrypt(&ct, p);
        for (i, &expected) in coeffs.iter().enumerate() {
            assert_eq!(recovered.coeff(i).to_u64(), expected);
        }
    }

    /// Paper-class RNS: Q = VIA-C q_1 ≈ 2^75, p = 16, Gaussian error.
    /// The flagship test — proves that the trait abstraction's `u128`
    /// machinery actually works at the modulus sizes the protocol uses.
    #[test]
    fn encrypt_decrypt_roundtrip_via_c_q1_rns_p16_n16_gaussian() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-via-c-q1rns");
        let sk =
            SecretKey::<16, RnsPolyViaCQ1<16>>::keygen(basis, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-via-c-q1rns");
        let coeffs: [u64; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let m: SinglePolyP16<16> = Poly::new(p, coeffs);
        let encoded: RnsPolyViaCQ1<16> = encode(&m, basis);
        let ct = sk.encrypt(
            &encoded,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let recovered: SinglePolyP16<16> = sk.decrypt(&ct, p);
        for (i, &expected) in coeffs.iter().enumerate() {
            assert_eq!(recovered.coeff(i).to_u64(), expected, "i={i}");
        }
    }

    // -----------------------------------------------------------------------
    // decrypt_raw recovers exactly `e + M'` (mod q).
    // -----------------------------------------------------------------------

    /// `decrypt_raw(encrypt(M', sk), sk) − M'` should equal the sampled
    /// error `e` (mod q), centred to its expected tail. With Ternary
    /// errors `|e_i| ≤ 1` per coefficient.
    #[test]
    fn decrypt_raw_recovers_encoded_plus_ternary_noise_in_budget() {
        let q = PowerOfTwoModulus::<10>;
        let mut sk_prg = Shake256Prg::new(b"sk-noise-ternary");
        let sk =
            SecretKey::<16, SinglePolyQ1024<16>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let encoded = <SinglePolyQ1024<16>>::zero(q); // pick M'=0 so raw == e
        let mut enc_prg = Shake256Prg::new(b"enc-noise-ternary");
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
        let raw = sk.decrypt_raw(&ct);
        let mut centered = [0i64; 16];
        raw.to_centered_coeffs(&mut centered);
        for (i, &c) in centered.iter().enumerate() {
            assert!(c.abs() <= 1, "ternary noise out of budget at i={i}: {c}");
        }
    }

    /// Gaussian σ = 4: virtually all samples should land within ±30
    /// (≈7.5σ). Generous bound; the cryptographic correctness of the
    /// Gaussian sampler itself is tested in `src/sampling/gaussian.rs`.
    #[test]
    fn decrypt_raw_gaussian_noise_within_seven_sigma() {
        let q = PowerOfTwoModulus::<10>;
        let mut sk_prg = Shake256Prg::new(b"sk-noise-gauss");
        let sk =
            SecretKey::<64, SinglePolyQ1024<64>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let encoded = <SinglePolyQ1024<64>>::zero(q);
        let mut enc_prg = Shake256Prg::new(b"enc-noise-gauss");
        let ct = sk.encrypt(
            &encoded,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let raw = sk.decrypt_raw(&ct);
        let mut centered = [0i64; 64];
        raw.to_centered_coeffs(&mut centered);
        for (i, &c) in centered.iter().enumerate() {
            assert!(
                c.abs() <= 30,
                "gaussian noise at i={i} exceeded 7.5σ bound: {c}"
            );
        }
    }

    // =======================================================================
    // Phase 4 — auxiliary RLWE primitives (§2.2.5)
    // =======================================================================

    // -----------------------------------------------------------------------
    // Add / Sub / Neg homomorphism
    // -----------------------------------------------------------------------

    /// `decrypt(E(m₁) + E(m₂)) == m₁ + m₂ mod p`. Generous q=1024 / p=2
    /// budget makes ternary-error sum trivially safe.
    #[test]
    fn add_decrypts_to_sum_at_q1024_p2() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        let mut sk_prg = Shake256Prg::new(b"sk-add");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-add");
        let m1: SinglePolyP2<4> = Poly::new(p, [1, 0, 1, 0]);
        let m2: SinglePolyP2<4> = Poly::new(p, [0, 1, 0, 1]);
        let enc1: SinglePolyQ1024<4> = encode(&m1, q);
        let enc2: SinglePolyQ1024<4> = encode(&m2, q);
        let ct1 = sk.encrypt(&enc1, Distribution::Ternary, &mut enc_prg);
        let ct2 = sk.encrypt(&enc2, Distribution::Ternary, &mut enc_prg);
        let ct_sum = ct1 + ct2;
        let recovered: SinglePolyP2<4> = sk.decrypt(&ct_sum, p);
        for i in 0..4 {
            let expected = (m1.coeff(i).to_u64() + m2.coeff(i).to_u64()) % 2;
            assert_eq!(recovered.coeff(i).to_u64(), expected, "i={i}");
        }
    }

    /// `decrypt(E(m₁) − E(m₂)) == m₁ − m₂ mod p`. Use p=16 so the
    /// difference is distinguishable from the sum (at p=2 both yield
    /// `m₁ ⊕ m₂`).
    #[test]
    fn sub_decrypts_to_difference_at_q1024_p16() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-sub");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-sub");
        let m1: SinglePolyP16<4> = Poly::new(p, [5, 10, 3, 7]);
        let m2: SinglePolyP16<4> = Poly::new(p, [2, 4, 11, 6]);
        let enc1: SinglePolyQ1024<4> = encode(&m1, q);
        let enc2: SinglePolyQ1024<4> = encode(&m2, q);
        let ct1 = sk.encrypt(&enc1, Distribution::Ternary, &mut enc_prg);
        let ct2 = sk.encrypt(&enc2, Distribution::Ternary, &mut enc_prg);
        let ct_diff = ct1 - ct2;
        let recovered: SinglePolyP16<4> = sk.decrypt(&ct_diff, p);
        for i in 0..4 {
            let expected = (m1.coeff(i).to_u64() + 16 - m2.coeff(i).to_u64()) % 16;
            assert_eq!(recovered.coeff(i).to_u64(), expected, "i={i}");
        }
    }

    /// `decrypt(−E(m)) == −m mod p`. p=16 keeps the negation
    /// distinguishable.
    #[test]
    fn neg_decrypts_to_negation_at_q1024_p16() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-neg");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-neg");
        let m: SinglePolyP16<4> = Poly::new(p, [3, 5, 7, 11]);
        let enc: SinglePolyQ1024<4> = encode(&m, q);
        let ct = sk.encrypt(&enc, Distribution::Ternary, &mut enc_prg);
        let recovered: SinglePolyP16<4> = sk.decrypt(&(-ct), p);
        for i in 0..4 {
            let expected = (16 - m.coeff(i).to_u64()) % 16;
            assert_eq!(recovered.coeff(i).to_u64(), expected, "i={i}");
        }
    }

    /// `let mut ct = ct1; ct += ct2;` is observationally identical to
    /// `ct1 + ct2`.
    #[test]
    fn add_assign_matches_add() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        let mut sk_prg = Shake256Prg::new(b"sk-add-assign");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-add-assign");
        let m1: SinglePolyP2<4> = Poly::new(p, [1, 1, 0, 0]);
        let m2: SinglePolyP2<4> = Poly::new(p, [0, 1, 1, 0]);
        let enc1: SinglePolyQ1024<4> = encode(&m1, q);
        let enc2: SinglePolyQ1024<4> = encode(&m2, q);
        let ct1 = sk.encrypt(&enc1, Distribution::Ternary, &mut enc_prg);
        let ct2 = sk.encrypt(&enc2, Distribution::Ternary, &mut enc_prg);
        let by_operator = ct1 + ct2;
        let mut by_assign = ct1;
        by_assign += ct2;
        assert_eq!(by_operator, by_assign);
    }

    /// `let mut ct = ct1; ct -= ct2;` is observationally identical to
    /// `ct1 - ct2`.
    #[test]
    fn sub_assign_matches_sub() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-sub-assign");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-sub-assign");
        let m1: SinglePolyP16<4> = Poly::new(p, [9, 4, 2, 13]);
        let m2: SinglePolyP16<4> = Poly::new(p, [1, 2, 3, 4]);
        let enc1: SinglePolyQ1024<4> = encode(&m1, q);
        let enc2: SinglePolyQ1024<4> = encode(&m2, q);
        let ct1 = sk.encrypt(&enc1, Distribution::Ternary, &mut enc_prg);
        let ct2 = sk.encrypt(&enc2, Distribution::Ternary, &mut enc_prg);
        let by_operator = ct1 - ct2;
        let mut by_assign = ct1;
        by_assign -= ct2;
        assert_eq!(by_operator, by_assign);
    }

    /// A 3-element chain `E(m₁) + E(m₂) + E(m₃)` decrypts to
    /// `m₁ + m₂ + m₃`. Confirms operator associativity and that noise
    /// accumulation stays inside budget for short chains.
    #[test]
    fn add_three_ciphertexts_in_chain() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-chain");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-chain");
        let m1: SinglePolyP16<4> = Poly::new(p, [1, 2, 3, 4]);
        let m2: SinglePolyP16<4> = Poly::new(p, [5, 6, 7, 8]);
        let m3: SinglePolyP16<4> = Poly::new(p, [0, 1, 2, 3]);
        let ct1 = sk.encrypt(
            &encode::<4, SinglePolyQ1024<4>, _>(&m1, q),
            Distribution::Ternary,
            &mut enc_prg,
        );
        let ct2 = sk.encrypt(
            &encode::<4, SinglePolyQ1024<4>, _>(&m2, q),
            Distribution::Ternary,
            &mut enc_prg,
        );
        let ct3 = sk.encrypt(
            &encode::<4, SinglePolyQ1024<4>, _>(&m3, q),
            Distribution::Ternary,
            &mut enc_prg,
        );
        let ct_sum = ct1 + ct2 + ct3;
        let recovered: SinglePolyP16<4> = sk.decrypt(&ct_sum, p);
        for i in 0..4 {
            let expected =
                (m1.coeff(i).to_u64() + m2.coeff(i).to_u64() + m3.coeff(i).to_u64()) % 16;
            assert_eq!(recovered.coeff(i).to_u64(), expected, "i={i}");
        }
    }

    // -----------------------------------------------------------------------
    // Trivial ciphertext
    // -----------------------------------------------------------------------

    /// `(0, M')` decrypts to `m` regardless of which key is used —
    /// `B − A·S = M' − 0·S = M'`, so `S` does not appear in the result.
    #[test]
    fn trivial_decrypts_to_message_under_any_key() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_a_prg = Shake256Prg::new(b"sk-A");
        let mut sk_b_prg = Shake256Prg::new(b"sk-B");
        let sk_a =
            SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_a_prg);
        let sk_b =
            SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_b_prg);
        let m: SinglePolyP16<4> = Poly::new(p, [7, 1, 14, 0]);
        let encoded: SinglePolyQ1024<4> = encode(&m, q);
        let ct = <RLWECiphertext<4, SinglePolyQ1024<4>>>::trivial(q, &encoded);
        let recovered_a: SinglePolyP16<4> = sk_a.decrypt(&ct, p);
        let recovered_b: SinglePolyP16<4> = sk_b.decrypt(&ct, p);
        for i in 0..4 {
            let expected = m.coeff(i).to_u64();
            assert_eq!(recovered_a.coeff(i).to_u64(), expected, "sk_a, i={i}");
            assert_eq!(recovered_b.coeff(i).to_u64(), expected, "sk_b, i={i}");
        }
    }

    // -----------------------------------------------------------------------
    // Polynomial × ciphertext
    // -----------------------------------------------------------------------

    /// `ct * X` rotates the message via negacyclic mul:
    /// $X \cdot (a_0 + a_1 X + a_2 X^2 + a_3 X^3) \equiv -a_3 + a_0 X + a_1 X^2 + a_2 X^3 \pmod{X^4 + 1}$.
    #[test]
    fn polynomial_mul_by_monomial_x_rotates_message_with_negacyclic_wrap() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-mul-x");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-mul-x");
        let m: SinglePolyP16<4> = Poly::new(p, [3, 5, 7, 9]);
        let encoded: SinglePolyQ1024<4> = encode(&m, q);
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
        // f = X (in the ciphertext modulus, since `Mul<R>` is at the
        // ciphertext ring).
        let f: SinglePolyQ1024<4> = Poly::new(q, [0, 1, 0, 0]);
        let recovered: SinglePolyP16<4> = sk.decrypt(&(ct * f), p);
        // Expected: [-9, 3, 5, 7] mod 16 = [7, 3, 5, 7].
        let expected: [u64; 4] = [7, 3, 5, 7];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(recovered.coeff(i).to_u64(), exp, "i={i}");
        }
    }

    /// `ct * (1 + X)` produces `m + X·m`. Exercises the small-norm `f`
    /// case where noise grows by `||f||₁ = 2`.
    #[test]
    fn polynomial_mul_by_weight_2_polynomial() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-mul-1px");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-mul-1px");
        let m: SinglePolyP16<4> = Poly::new(p, [2, 3, 5, 7]);
        let encoded: SinglePolyQ1024<4> = encode(&m, q);
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
        // f = 1 + X.
        let f: SinglePolyQ1024<4> = Poly::new(q, [1, 1, 0, 0]);
        let recovered: SinglePolyP16<4> = sk.decrypt(&(ct * f), p);
        // (1 + X) · (2 + 3X + 5X^2 + 7X^3)
        //   = 2 + 3X + 5X^2 + 7X^3
        //   + 2X + 3X^2 + 5X^3 + 7X^4
        // mod (X^4 + 1):  X^4 = -1, so 7X^4 = -7.
        //   = (2 - 7) + (3 + 2)X + (5 + 3)X^2 + (7 + 5)X^3
        //   = -5 + 5X + 8X^2 + 12X^3
        // mod 16: [11, 5, 8, 12].
        let expected: [u64; 4] = [11, 5, 8, 12];
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(recovered.coeff(i).to_u64(), exp, "i={i}");
        }
    }

    /// Multiplying by the unit polynomial `1` is the identity on the
    /// ciphertext bit-for-bit (operator `PartialEq` derived in Phase 1).
    #[test]
    fn polynomial_mul_by_unit_is_identity() {
        let q = PowerOfTwoModulus::<10>;
        let p = PowerOfTwoModulus::<1>;
        let mut sk_prg = Shake256Prg::new(b"sk-mul-one");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-mul-one");
        let m: SinglePolyP2<4> = Poly::new(p, [1, 0, 1, 1]);
        let encoded: SinglePolyQ1024<4> = encode(&m, q);
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut enc_prg);
        let one: SinglePolyQ1024<4> = Poly::new(q, [1, 0, 0, 0]);
        assert_eq!(ct * one, ct);
    }

    // -----------------------------------------------------------------------
    // Paper-class + RNS coverage
    // -----------------------------------------------------------------------

    /// Sum at VIA-C `q₂` ≈ 2³⁴ with Gaussian σ=4 errors. Validates that
    /// the operator works on a paper-class single-prime modulus.
    #[test]
    fn add_decrypts_to_sum_at_via_c_q2_gaussian() {
        let q = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        let p = PowerOfTwoModulus::<8>;
        let mut sk_prg = Shake256Prg::new(b"sk-add-via-c");
        let sk =
            SecretKey::<16, SinglePolyViaCQ2<16>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-add-via-c");
        let m1_coeffs: [u64; 16] = [0, 1, 3, 7, 15, 31, 63, 127, 200, 250, 99, 42, 5, 17, 33, 64];
        let m2_coeffs: [u64; 16] = [
            1, 2, 5, 11, 23, 47, 95, 191, 50, 100, 150, 200, 250, 0, 1, 2,
        ];
        let m1: SinglePolyP256<16> = Poly::new(p, m1_coeffs);
        let m2: SinglePolyP256<16> = Poly::new(p, m2_coeffs);
        let ct1 = sk.encrypt(
            &encode::<16, SinglePolyViaCQ2<16>, _>(&m1, q),
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let ct2 = sk.encrypt(
            &encode::<16, SinglePolyViaCQ2<16>, _>(&m2, q),
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let recovered: SinglePolyP256<16> = sk.decrypt(&(ct1 + ct2), p);
        for i in 0..16 {
            let expected = (m1_coeffs[i] + m2_coeffs[i]) % 256;
            assert_eq!(recovered.coeff(i).to_u64(), expected, "i={i}");
        }
    }

    /// `ct * X` at VIA-C `q₁` ≈ 2⁷⁵ RNS. Validates that polynomial
    /// multiplication threads correctly through the `PolyRns` schoolbook.
    #[test]
    fn polynomial_mul_at_via_c_q1_rns() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        let p = PowerOfTwoModulus::<4>;
        let mut sk_prg = Shake256Prg::new(b"sk-mul-via-c-rns");
        let sk =
            SecretKey::<16, RnsPolyViaCQ1<16>>::keygen(basis, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-mul-via-c-rns");
        let m_coeffs: [u64; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let m: SinglePolyP16<16> = Poly::new(p, m_coeffs);
        let encoded: RnsPolyViaCQ1<16> = encode(&m, basis);
        let ct = sk.encrypt(
            &encoded,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let f = <RnsPolyViaCQ1<16> as RingPoly<16>>::from_centered_i64s(basis, &{
            let mut v = [0i64; 16];
            v[1] = 1; // f = X
            v
        });
        let recovered: SinglePolyP16<16> = sk.decrypt(&(ct * f), p);
        // (X) · (0 + X + 2X² + … + 15X¹⁵)
        //   = X + X² + 2X³ + … + 14X¹⁵ + 15X¹⁶
        // mod (X¹⁶ + 1):  X¹⁶ = -1, so 15X¹⁶ = -15.
        //   = -15 + 0·X + 1·X² + 2·X³ + … + 14·X¹⁵.
        //
        // mod p=16: slot[0] = 16 - 15 = 1; slot[i+1] = m_coeffs[i] for i in 0..15.
        let expected: [u64; 16] =
            core::array::from_fn(|i| if i == 0 { 1 } else { m_coeffs[i - 1] % 16 });
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(recovered.coeff(i).to_u64(), exp, "i={i}");
        }
    }

    /// Composition: `(E(m₁) + E(m₂)) * X`. Catches noise-blow-up
    /// regressions and operator-precedence surprises (would mis-parse if
    /// `Mul<R>` and `Add` had wrong associativity).
    #[test]
    fn add_then_polynomial_mul_at_via_c_q2() {
        let q = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        let p = PowerOfTwoModulus::<8>;
        let mut sk_prg = Shake256Prg::new(b"sk-comp-via-c");
        let sk =
            SecretKey::<16, SinglePolyViaCQ2<16>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-comp-via-c");
        let m1_coeffs: [u64; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let m2_coeffs: [u64; 16] = [16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1];
        let m1: SinglePolyP256<16> = Poly::new(p, m1_coeffs);
        let m2: SinglePolyP256<16> = Poly::new(p, m2_coeffs);
        let ct1 = sk.encrypt(
            &encode::<16, SinglePolyViaCQ2<16>, _>(&m1, q),
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let ct2 = sk.encrypt(
            &encode::<16, SinglePolyViaCQ2<16>, _>(&m2, q),
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );
        let f = <SinglePolyViaCQ2<16> as RingPoly<16>>::from_centered_i64s(q, &{
            let mut v = [0i64; 16];
            v[1] = 1; // f = X
            v
        });
        let recovered: SinglePolyP256<16> = sk.decrypt(&((ct1 + ct2) * f), p);
        // sum_coeffs[i] = (m1[i] + m2[i]) % 256
        let sum: [u64; 16] = core::array::from_fn(|i| (m1_coeffs[i] + m2_coeffs[i]) % 256);
        // After * X (rotation with negacyclic wrap at index 15 -> 0):
        let expected: [u64; 16] = core::array::from_fn(|i| {
            if i == 0 {
                (256 - sum[15]) % 256
            } else {
                sum[i - 1]
            }
        });
        for (i, &exp) in expected.iter().enumerate() {
            assert_eq!(recovered.coeff(i).to_u64(), exp, "i={i}");
        }
    }

    // -----------------------------------------------------------------------
    // decrypt_asymmetric — §3.2 RespCompRecover
    // -----------------------------------------------------------------------

    /// End-to-end §3.2 round-trip: encrypt at $q_3$, body-rescale to $q_4$
    /// (the asymmetric "body-only" ModSwitch), then recover the plaintext
    /// via [`SecretKey::decrypt_asymmetric`]. Mask stays at $q_3$, body at
    /// $q_4$ — matching the `ModSwitchedCiphertext` shape Figure 7 produces.
    #[test]
    fn decrypt_asymmetric_round_trip_pow2() {
        type Q3 = Poly<8, PowerOfTwoModulus<12>, Coefficient>; // q3 = 4096
        type Q4 = Poly<8, PowerOfTwoModulus<8>, Coefficient>; // q4 = 256
        type P2 = Poly<8, PowerOfTwoModulus<1>, Coefficient>; // p = 2
        let q3m = PowerOfTwoModulus::<12>;
        let q4m = PowerOfTwoModulus::<8>;
        let pm = PowerOfTwoModulus::<1>;

        let mut prg = Shake256Prg::new(b"decrypt-asym-roundtrip");
        let sk = SecretKey::<8, Q3>::keygen(q3m, Distribution::Ternary, &mut prg);

        let coeffs = [1u64, 0, 1, 1, 0, 1, 0, 0];
        let pt: P2 = pt_from_vec(pm, coeffs);
        let encoded: Q3 = encode(&pt, q3m);
        let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut prg);

        // Body-only rescale B: q3 → q4 (the §3.2 trailing op of RespComp).
        let q3: u128 = 1 << 12;
        let q4: u128 = 1 << 8;
        let mut b = [0u128; 8];
        ct.body.to_u128_coeffs(&mut b);
        for v in b.iter_mut() {
            *v = (*v * q4 + q3 / 2) / q3;
        }
        let body_q4 = <Q4 as RingPoly<8>>::from_u128_coeffs(q4m, &b);
        let msct = ModSwitchedCiphertext::<8, Q3, Q4>::new(ct.mask, body_q4);

        let recovered: P2 = sk.decrypt_asymmetric(&msct, q3m, q4m, pm);
        let mut got = [0u128; 8];
        recovered.to_u128_coeffs(&mut got);
        let expected: [u128; 8] = core::array::from_fn(|i| u128::from(coeffs[i]));
        assert_eq!(got, expected);
    }
}
