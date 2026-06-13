//! RLev encryption.
//!
//! An RLev ciphertext encrypts a message $M$ as an array of $L$ RLWE
//! ciphertexts, the $i$-th encrypting $g_i \cdot M$ where $g_i$ is the
//! $i$-th MSB-first entry of the gadget vector
//! (`gadget_vector_values::<N, R, L>(modulus, base)`).
//!
//! Unlike ordinary [`SecretKey::encrypt`], **the message is not
//! $\Delta$-encoded** — the gadget entries serve as the scaling.
//! Callers pass `message` as a raw ring element and rely on
//! `gadget_product` to recover semantic meaning from the
//! per-level encryptions.
//!
//! `gadget_product` (RLev × plaintext → RLWE) lives in
//! this same file.

use crate::algebra::ring::{RingPoly, RingPolyEval};
use crate::sampling::distribution::Distribution;
use crate::sampling::prg::Shake256Prg;

use super::gadget::{gadget_extract_lsb_into, gadget_scale_into, gadget_vector_values};
use super::types::{RLWECiphertext, RLevCiphertext, RLevEval, SecretKey};

impl<const N: usize, R: RingPoly<N>> SecretKey<N, R> {
    /// Encrypt `message` as an RLev: array of `L` RLWE
    /// ciphertexts where `samples[i]` encrypts `g_i · message` and
    /// `g_i` is the `i`-th MSB-first entry of the gadget vector.
    ///
    /// `base` is the gadget base `B`; the depth `L` is the const
    /// generic on the return type. **The message is not `Δ`-encoded.**
    ///
    /// # PRG consumption order
    ///
    /// Sequential by sample: `samples[0]` is fully sampled (mask, then
    /// error — matching [`SecretKey::encrypt`]'s order) before
    /// `samples[1]`, and so on. Reversing the iteration would
    /// silently break every RLev test vector.
    ///
    /// # Argument convention
    ///
    /// `message` is the **raw** ring element: the gadget scaling is
    /// applied internally per level. RLev does not go through `encode`,
    /// which is what the gadget product relies on.
    pub fn encrypt_rlev<const L: usize>(
        &self,
        message: &R,
        base: u64,
        error_dist: Distribution,
        prg: &mut Shake256Prg,
    ) -> RLevCiphertext<N, R, L> {
        let modulus = self.poly.modulus();
        let g_values = gadget_vector_values::<N, R, L>(modulus, base);

        // Per level: build `g_i · message`, then encrypt it. We carry
        // `g_i` through a constant polynomial `(g_i, 0, …, 0)` so the
        // multiplication uses the trait's `Mul<R>` — works uniformly
        // for both backends without needing a separate `Mul<u128>`
        // impl on the polynomial type.
        let samples = core::array::from_fn(|i| {
            let mut g_const = [0u128; N];
            g_const[0] = g_values[i];
            let g_poly = R::from_u128_coeffs(modulus, &g_const);
            let scaled: R = *message * g_poly;
            self.encrypt(&scaled, error_dist, prg)
        });
        RLevCiphertext::new(samples)
    }

    /// Heap-building sibling of [`encrypt_rlev`](Self::encrypt_rlev): writes the
    /// `L` samples **one at a time straight into a `Box`**, so the full
    /// `[RLWECiphertext; L]` array (≈ `L · 2N · word` bytes — ~1.125 MiB at the
    /// depth-18 n=2048 conversion key) is never assembled on the stack.
    ///
    /// PRG draws are **byte-identical** to [`encrypt_rlev`](Self::encrypt_rlev)
    /// (sample `i` ascending: `g_i·message`, then `encrypt` = mask-then-error),
    /// so the two produce the same key — guarded by an equivalence unit test.
    /// Mirrors the cascade `gen_..._boxed` builder pattern.
    /// Write an RLev encryption of `message` **directly into `dst`**, one RLWE
    /// sample at a time — the peak stack is one RLWE (≈ `2N·word`) instead of the
    /// whole `[RLWECiphertext; L]` array. PRG draws are byte-identical to
    /// [`encrypt_rlev`](Self::encrypt_rlev). This is the building block for both
    /// [`encrypt_rlev_boxed`](Self::encrypt_rlev_boxed) and the cascade key's
    /// per-element heap builder.
    ///
    /// # Safety
    ///
    /// `dst` must point to memory valid for writes of one
    /// `RLevCiphertext<N, R, L>` (typically uninitialised, e.g. a `Box`
    /// allocation or a field slot). Every `samples[i]` for `i ∈ 0..L` is
    /// initialised exactly once; the caller must not read `*dst` before this
    /// returns and is responsible for treating `*dst` as initialised afterwards.
    pub(crate) unsafe fn encrypt_rlev_into<const L: usize>(
        &self,
        dst: *mut RLevCiphertext<N, R, L>,
        message: &R,
        base: u64,
        error_dist: Distribution,
        prg: &mut Shake256Prg,
    ) {
        let modulus = self.poly.modulus();
        let g_values = gadget_vector_values::<N, R, L>(modulus, base);
        for (i, &g_i) in g_values.iter().enumerate() {
            let mut g_const = [0u128; N];
            g_const[0] = g_i;
            let g_poly = R::from_u128_coeffs(modulus, &g_const);
            let scaled: R = *message * g_poly;
            let sample = self.encrypt(&scaled, error_dist, prg);
            // SAFETY: `dst` is valid for one RLev (caller contract); `samples[i]`
            // with `i ∈ 0..L` is written exactly once, in ascending order.
            unsafe { core::ptr::addr_of_mut!((*dst).samples[i]).write(sample) };
        }
    }

    /// Heap-allocating wrapper over the crate-private `encrypt_rlev_into`:
    /// the full `[RLWECiphertext; L]` array (≈ 1.125 MiB at the depth-18
    /// n=2048 conversion key) is built straight into the `Box`, never on the
    /// stack. Byte-identical to [`encrypt_rlev`](Self::encrypt_rlev).
    #[cfg(feature = "alloc")]
    pub fn encrypt_rlev_boxed<const L: usize>(
        &self,
        message: &R,
        base: u64,
        error_dist: Distribution,
        prg: &mut Shake256Prg,
    ) -> alloc::boxed::Box<RLevCiphertext<N, R, L>> {
        let mut boxed = alloc::boxed::Box::<RLevCiphertext<N, R, L>>::new_uninit();
        // SAFETY: `as_mut_ptr` is valid for one (uninit) RLev; `encrypt_rlev_into`
        // initialises every `samples[i]`, so `assume_init` is sound.
        unsafe {
            self.encrypt_rlev_into(boxed.as_mut_ptr(), message, base, error_dist, prg);
            boxed.assume_init()
        }
    }
}

// ---------------------------------------------------------------------------
// Gadget product.
//
// `plaintext ⊡ RLev_S(message) → RLWE_S(plaintext · message)` — the
// multiplicative core of every homomorphic gate, plus the inner step
// of key switching and ring switching.
// ---------------------------------------------------------------------------

impl<const N: usize, R: RingPoly<N> + RingPolyEval<N>, const L: usize> RLevCiphertext<N, R, L> {
    /// Gadget product: $\text{plaintext} \boxdot \mathrm{RLev}_S(M)
    /// \to \mathrm{RLWE}_S(\text{plaintext} \cdot M)$.
    ///
    /// # Algorithm
    ///
    /// 1. Scale `plaintext` into an `[i128; N]` scratch via
    ///    [`super::gadget::gadget_scale_into`].
    /// 2. For each level `k ∈ 0..L`, extract one LSB digit per
    ///    coefficient via [`super::gadget::gadget_extract_lsb_into`],
    ///    lift to a polynomial via [`RingPoly::from_centered_i64s`],
    ///    and accumulate `digit_poly · samples[L-1-k]` into the result.
    ///
    /// The LSB-first extraction step `k` pairs with the MSB-first
    /// sample index `L-1-k`: `samples[0]` encrypts `g_0 · M` (the
    /// largest gadget entry), which is the highest-place-value digit
    /// `d_{L-1}` — extracted at LSB step `k = L-1`, i.e., `L-1-k = 0`.
    /// **Off-by-one or wrong direction silently produces wrong
    /// products; the dedicated `msb_first_pairing_lock` test pins this.**
    ///
    /// # Memory
    ///
    /// `O(N)` scratch — `[i128; N]` for the scaled value plus
    /// `[i64; N]` for the digit buffer. At `N=2048`, ~48 KiB
    /// total, vs ~288 KiB if we materialised the full
    /// `[[i64; N]; L]` decomposition.
    ///
    /// # Noise growth
    ///
    /// Per-coefficient noise of the result is bounded by
    /// `||plaintext||_1 · σ_e + round(q / B^L) / 2 + L · B / 4` — the
    /// plaintext-1-norm-times-σ term from the per-level encryption
    /// noises summing, plus the gadget reconstruction error.
    ///
    /// `plaintext` should have small infinity-norm for the noise term
    /// to stay inside the decryption budget.
    ///
    /// # Backing & cost
    ///
    /// The per-level multiplies run through the [`RingPolyEval`] evaluation
    /// form. For **NTT-friendly** moduli ($q_1$ RNS / $q_2$ / $q_3$) that
    /// is the negacyclic NTT, so this is `3L` forward + `2` inverse
    /// `O(N \log N)` transforms and `2L` `O(N)` pointwise muls — ~100× fewer
    /// scalar multiplications than schoolbook at `N = 2048`. For **non-NTT**
    /// moduli the eval form is the coefficient form and `to_eval`/`from_eval`
    /// are identities, so it degenerates to the schoolbook `O(N²)` path. The
    /// result is **bit-identical** either way (the NTT is an exact ring
    /// isomorphism over $\mathbb{Z}_q$); pinned by
    /// `gadget_product_matches_schoolbook_*`.
    pub fn gadget_product(&self, plaintext: &R, base: u64) -> RLWECiphertext<N, R> {
        const { assert!(L >= 1, "gadget_product: L must be >= 1") };
        let modulus = plaintext.modulus();
        let mut scratch = [0i128; N];
        gadget_scale_into::<N, R>(plaintext, base, L as u8, &mut scratch);
        let mut digit_buf = [0i64; N];

        // LSB-first extraction step `k` pairs with the MSB-first sample index
        // `L - 1 - k` (the `msb_first_pairing_lock` test guards it). Each product
        // is accumulated in evaluation form, then transformed back once. For
        // non-NTT moduli `to_eval`/`from_eval` are identities and the pointwise
        // `*` is the schoolbook negacyclic mul, so this is exactly the old path.
        // The `k = 0` product seeds the accumulators (so no eval-form zero is
        // needed — `RingPolyEval` is standalone and carries no modulus).
        gadget_extract_lsb_into::<N>(base, &mut scratch, &mut digit_buf);
        let digit0 = R::to_eval(R::from_centered_i64s(modulus, &digit_buf));
        let s0 = &self.samples[L - 1];
        // `digit0` is `Copy`, so each `*` copies it rather than moving.
        let mut acc_mask = digit0 * R::to_eval(s0.mask);
        let mut acc_body = digit0 * R::to_eval(s0.body);

        for k in 1..L {
            gadget_extract_lsb_into::<N>(base, &mut scratch, &mut digit_buf);
            let digit_eval = R::to_eval(R::from_centered_i64s(modulus, &digit_buf));
            let sample = &self.samples[L - 1 - k];
            acc_mask += digit_eval * R::to_eval(sample.mask);
            acc_body += digit_eval * R::to_eval(sample.body);
        }
        RLWECiphertext::new(R::from_eval(acc_mask), R::from_eval(acc_body))
    }
}

// ---------------------------------------------------------------------------
// Eval-key storage (T7) — pre-transform a static RLev key once, then run
// `gadget_product` without re-transforming the samples on every call.
// ---------------------------------------------------------------------------

impl<const N: usize, R: RingPoly<N> + RingPolyEval<N>, const L: usize> RLevCiphertext<N, R, L> {
    /// Transform this RLev's `L` samples to evaluation form **once**
    /// (deterministic negacyclic NTT — no PRG), yielding an [`RLevEval`] whose
    /// [`gadget_product`](RLevEval::gadget_product) skips the per-call sample
    /// transforms. For STATIC keys (reused every query) this moves the `2L`
    /// per-call sample transforms out of the hot path into a one-time setup cost.
    ///
    /// For non-NTT moduli `to_eval` is the identity, so `RLevEval` is just the
    /// coefficient form and `gadget_product` degenerates to the schoolbook path.
    pub fn to_eval(&self) -> RLevEval<N, R, L> {
        RLevEval {
            samples: core::array::from_fn(|k| {
                (
                    R::to_eval(self.samples[k].mask),
                    R::to_eval(self.samples[k].body),
                )
            }),
        }
    }
}

impl<const N: usize, R: RingPoly<N> + RingPolyEval<N>, const L: usize> RLevEval<N, R, L> {
    /// Eval-form gadget product — **identical math** to
    /// [`RLevCiphertext::gadget_product`], minus the per-call `to_eval` of the
    /// (already pre-transformed) samples. Only the `L` dynamic digits are
    /// transformed per call (`L+2` transforms instead of `3L+2`).
    ///
    /// Bit-identical to the coefficient path (the NTT is an exact ring
    /// isomorphism), so KAT / e2e parity holds with no regeneration.
    pub fn gadget_product(&self, plaintext: &R, base: u64) -> RLWECiphertext<N, R> {
        const { assert!(L >= 1, "gadget_product: L must be >= 1") };
        let modulus = plaintext.modulus();
        let mut scratch = [0i128; N];
        gadget_scale_into::<N, R>(plaintext, base, L as u8, &mut scratch);
        let mut digit_buf = [0i64; N];

        // Mirrors `RLevCiphertext::gadget_product`: LSB-first digit `k` pairs
        // with MSB-first sample `L-1-k`; the `k=0` product seeds the accumulators.
        // The samples are read pre-transformed (no `R::to_eval` here).
        gadget_extract_lsb_into::<N>(base, &mut scratch, &mut digit_buf);
        let digit0 = R::to_eval(R::from_centered_i64s(modulus, &digit_buf));
        let (s0_mask, s0_body) = self.samples[L - 1];
        let mut acc_mask = digit0 * s0_mask;
        let mut acc_body = digit0 * s0_body;

        for k in 1..L {
            gadget_extract_lsb_into::<N>(base, &mut scratch, &mut digit_buf);
            let digit_eval = R::to_eval(R::from_centered_i64s(modulus, &digit_buf));
            let (s_mask, s_body) = self.samples[L - 1 - k];
            acc_mask += digit_eval * s_mask;
            acc_body += digit_eval * s_body;
        }
        RLWECiphertext::new(R::from_eval(acc_mask), R::from_eval(acc_body))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::zq::modulus::PowerOfTwoModulus;
    use crate::encryption::gadget::gadget_vector_values;

    type SinglePolyQ1024<const N: usize> = Poly<N, PowerOfTwoModulus<10>, Coefficient>;
    type SinglePolyViaCQ2<const N: usize> =
        Poly<N, crate::algebra::zq::modulus::paper::ViaCQ2, Coefficient>;
    type RnsPolyViaCQ1<const N: usize> =
        PolyRns<N, crate::algebra::rns::basis::paper::ViaCQ1Rns, Coefficient>;

    /// Build a polynomial whose constant term holds `value` (as u128)
    /// and all other coefficients are zero. Used as the "g_i" or "scalar"
    /// polynomial when constructing test expectations of the form
    /// `g_i · message` via the trait's `Mul<R>`.
    fn const_term_poly<const N: usize, R: RingPoly<N>>(modulus: R::Modulus, value: u128) -> R {
        let mut coeffs = [0u128; N];
        coeffs[0] = value;
        R::from_u128_coeffs(modulus, &coeffs)
    }

    // -----------------------------------------------------------------------
    // encrypt_rlev — per-sample decryption + ordering
    // -----------------------------------------------------------------------

    /// Each `samples[i]` decrypts to `g_i · M` plus small noise. At
    /// toy `(q=1024, B=2, L=10)` with ternary error, the noise per
    /// coefficient is at most 1 (single ternary draw per encryption).
    #[test]
    fn encrypt_rlev_sample_i_decrypts_to_g_i_times_m_q1024_b2() {
        let q = PowerOfTwoModulus::<10>;
        const L: usize = 10;
        let mut sk_prg = Shake256Prg::new(b"sk-rlev");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-rlev");

        // M = [3, 0, 1, 0] — small enough that g_i · M stays inside q.
        let message: SinglePolyQ1024<4> = Poly::new(q, [3, 0, 1, 0]);
        let rlev = sk.encrypt_rlev::<L>(&message, 2, Distribution::Ternary, &mut enc_prg);

        let g_values = gadget_vector_values::<4, SinglePolyQ1024<4>, L>(q, 2);
        for (i, sample) in rlev.samples.iter().enumerate() {
            // Expected: g_i · M as a polynomial.
            let g_poly = const_term_poly::<4, SinglePolyQ1024<4>>(q, g_values[i]);
            let expected = message * g_poly;
            let recovered = sk.decrypt_raw(sample);
            let diff = recovered - expected;
            let mut centred = [0i64; 4];
            diff.to_centered_coeffs(&mut centred);
            for (j, &c) in centred.iter().enumerate() {
                assert!(
                    c.abs() <= 4,
                    "rlev sample {i}: noise at coeff {j} = {c}, expected ≤ 4 (ternary tail)"
                );
            }
        }
    }

    /// `samples[0]` pairs with the **largest** gadget entry. Verify by
    /// asserting `decrypt_raw(samples[0])` is closer to `g_0 · M` than
    /// to `g_{L-1} · M` — locks in the MSB-first ordering convention.
    #[test]
    fn encrypt_rlev_sample_ordering_matches_gadget_vector() {
        let q = PowerOfTwoModulus::<10>;
        const L: usize = 4;
        let mut sk_prg = Shake256Prg::new(b"sk-order");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-order");

        let message: SinglePolyQ1024<4> = Poly::new(q, [1, 0, 0, 0]);
        let rlev = sk.encrypt_rlev::<L>(&message, 2, Distribution::Ternary, &mut enc_prg);

        let g_values = gadget_vector_values::<4, SinglePolyQ1024<4>, L>(q, 2);
        // q=1024, B=2, L=4: g = [512, 256, 128, 64].
        assert_eq!(g_values[0], 512);
        assert_eq!(g_values[L - 1], 64);

        // samples[0] should decrypt to ≈ 512 in coefficient 0.
        let raw0 = sk.decrypt_raw(&rlev.samples[0]);
        let mut c0 = [0i64; 4];
        raw0.to_centered_coeffs(&mut c0);
        // Expect c0[0] ≈ 512 — well past the small noise.
        assert!(
            (c0[0] - 512).abs() <= 4,
            "samples[0] coeff 0 = {} expected near 512",
            c0[0]
        );

        // samples[L-1] should decrypt to ≈ 64.
        let raw_last = sk.decrypt_raw(&rlev.samples[L - 1]);
        let mut clast = [0i64; 4];
        raw_last.to_centered_coeffs(&mut clast);
        assert!(
            (clast[0] - 64).abs() <= 4,
            "samples[L-1] coeff 0 = {} expected near 64",
            clast[0]
        );
    }

    /// Realistic single-prime: VIA-C `q₂` with CMux-sel gadget
    /// `(L=2, B=81)` and σ=4 Gaussian errors.
    #[test]
    fn encrypt_rlev_at_via_c_q2_b81_l2_gaussian() {
        let q = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        const L: usize = 2;
        let mut sk_prg = Shake256Prg::new(b"sk-rlev-via-c-q2");
        let sk =
            SecretKey::<16, SinglePolyViaCQ2<16>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-rlev-via-c-q2");

        let coeffs: [u64; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let message: SinglePolyViaCQ2<16> = Poly::new(q, coeffs);
        let rlev = sk.encrypt_rlev::<L>(
            &message,
            81,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );

        let g_values = gadget_vector_values::<16, SinglePolyViaCQ2<16>, L>(q, 81);
        for (i, sample) in rlev.samples.iter().enumerate() {
            let g_poly = const_term_poly::<16, SinglePolyViaCQ2<16>>(q, g_values[i]);
            let expected = message * g_poly;
            let recovered = sk.decrypt_raw(sample);
            let diff = recovered - expected;
            let mut centred = [0i64; 16];
            diff.to_centered_coeffs(&mut centred);
            for (j, &c) in centred.iter().enumerate() {
                assert!(
                    c.abs() <= 32,
                    "rlev sample {i} coeff {j}: noise {c} > 8σ bound 32"
                );
            }
        }
    }

    /// Realistic RNS: VIA-C `q₁` (`Q ≈ 2⁷⁵`) with ring-switch-key
    /// gadget `(L=8, B=8)`. Exercises the RNS poly mul + encrypt path
    /// through every per-level RLWE in the RLev.
    #[test]
    fn encrypt_rlev_at_via_c_q1_rns_b8_l8_gaussian() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        const L: usize = 8;
        let mut sk_prg = Shake256Prg::new(b"sk-rlev-via-c-q1rns");
        let sk =
            SecretKey::<16, RnsPolyViaCQ1<16>>::keygen(basis, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-rlev-via-c-q1rns");

        // Pick a small-magnitude message so g_i · M stays interpretable.
        let message_coeffs: [i64; 16] = [1, -1, 0, 2, -2, 1, 0, 0, 1, -1, 0, 0, 1, 0, -1, 1];
        let message =
            <RnsPolyViaCQ1<16> as RingPoly<16>>::from_centered_i64s(basis, &message_coeffs);
        let rlev = sk.encrypt_rlev::<L>(
            &message,
            8,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );

        let g_values = gadget_vector_values::<16, RnsPolyViaCQ1<16>, L>(basis, 8);
        for (i, sample) in rlev.samples.iter().enumerate() {
            let g_poly = const_term_poly::<16, RnsPolyViaCQ1<16>>(basis, g_values[i]);
            let expected = message * g_poly;
            let recovered = sk.decrypt_raw(sample);
            let diff = recovered - expected;
            let mut centred = [0i128; 16];
            diff.to_centered_coeffs(&mut centred);
            for (j, &c) in centred.iter().enumerate() {
                assert!(
                    c.abs() <= 32,
                    "rlev RNS sample {i} coeff {j}: noise {c} > 8σ bound 32"
                );
            }
        }
    }

    /// Encrypt a known plaintext into an RLev, then walk every sample
    /// and assert the post-decrypt noise stays inside the 8σ tail of
    /// the configured Gaussian. Catches accidental noise amplification
    /// in the per-level RLWE encrypt path.
    #[test]
    fn encrypt_rlev_noise_within_eight_sigma_at_paper_sigma() {
        let q = PowerOfTwoModulus::<10>;
        const L: usize = 4;
        let mut sk_prg = Shake256Prg::new(b"sk-rlev-noise");
        let sk =
            SecretKey::<32, SinglePolyQ1024<32>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-rlev-noise");

        // Use zero message so `decrypt_raw - 0 = pure noise + g_i · 0 = noise`.
        let zero = <SinglePolyQ1024<32> as RingPoly<32>>::zero(q);
        let rlev = sk.encrypt_rlev::<L>(
            &zero,
            4,
            Distribution::Gaussian { sigma: 4.0 },
            &mut enc_prg,
        );

        for sample in rlev.samples.iter() {
            let raw = sk.decrypt_raw(sample);
            let mut centred = [0i64; 32];
            raw.to_centered_coeffs(&mut centred);
            for (j, &c) in centred.iter().enumerate() {
                assert!(c.abs() <= 32, "rlev noise sample coeff {j} = {c} > 32");
            }
        }
    }

    // -----------------------------------------------------------------------
    // gadget_product
    // -----------------------------------------------------------------------

    /// `decrypt_raw(rlev.gadget_product(&m1, B), sk) ≈ m1 · m2 + noise`.
    /// Toy `(q=1024, B=2, L=10)` with ternary errors and small-norm `m1`.
    #[test]
    fn gadget_product_recovers_m1_times_m2_at_q1024_b2() {
        let q = PowerOfTwoModulus::<10>;
        const L: usize = 10;
        let mut sk_prg = Shake256Prg::new(b"sk-gp-toy");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-gp-toy");

        let m2: SinglePolyQ1024<4> = Poly::new(q, [5, 3, 0, 7]);
        let rlev = sk.encrypt_rlev::<L>(&m2, 2, Distribution::Ternary, &mut enc_prg);

        // Small-norm m1 to keep noise inside budget.
        let m1: SinglePolyQ1024<4> = Poly::new(q, [1, 0, 1, 0]);
        let ct = rlev.gadget_product(&m1, 2);
        let recovered = sk.decrypt_raw(&ct);

        // Expected: m1 · m2 (negacyclic mul at q).
        let expected = m1 * m2;
        let diff = recovered - expected;
        let mut centred = [0i64; 4];
        diff.to_centered_coeffs(&mut centred);
        // Noise bound: ||m1||_1 · σ_e + reconstruction error ≤ 2 · 1 + 1 = 3,
        // plus per-level slack. Use 16 for headroom.
        for (j, &c) in centred.iter().enumerate() {
            assert!(c.abs() <= 16, "gadget product diff at coeff {j}: {c}");
        }
    }

    /// **MSB-first pairing lock.** Construct an RLev encrypting a
    /// distinguishable `M`, then call `gadget_product` with plaintext
    /// `c = 512` (at q=1024). Its base-2 decomposition is
    /// `[1, 0, 0, …, 0]` MSB-first (the leading 1 lands in `digit[0]`).
    ///
    /// **`digit[0]` pairs with `samples[0]` = enc(g_0 · M) = enc(512 · M)**, so the
    /// result should decrypt to ≈ `512 · M`. If the pairing were
    /// inverted (`digit[0]` paired with `samples[L-1]` = enc(M)), the
    /// result would decrypt to ≈ M — visibly different at the
    /// per-coefficient noise scale we tolerate.
    #[test]
    fn gadget_product_msb_first_pairing_lock() {
        let q = PowerOfTwoModulus::<10>;
        const L: usize = 10;
        let mut sk_prg = Shake256Prg::new(b"sk-gp-pair");
        let sk = SecretKey::<4, SinglePolyQ1024<4>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-gp-pair");

        let m: SinglePolyQ1024<4> = Poly::new(q, [3, 1, 0, 2]);
        let rlev = sk.encrypt_rlev::<L>(&m, 2, Distribution::Ternary, &mut enc_prg);

        let plaintext: SinglePolyQ1024<4> = Poly::new(q, [512, 0, 0, 0]);
        let ct = rlev.gadget_product(&plaintext, 2);
        let recovered = sk.decrypt_raw(&ct);

        // Expected: 512 · M = 512 · (3 + X + 2X³) mod (1024, X⁴+1).
        // = 1536 + 512X + 1024X³ mod 1024
        // = 512 + 512X + 0X² + 0X³.
        let expected = plaintext * m;
        let diff = recovered - expected;
        let mut centred = [0i64; 4];
        diff.to_centered_coeffs(&mut centred);
        // Noise bound: ||plaintext||_1 = 512 — but only the MSB digit
        // is nonzero, so effective noise is ~σ_e per level. Use 16.
        for (j, &c) in centred.iter().enumerate() {
            assert!(
                c.abs() <= 16,
                "MSB-first pairing-lock failed at coeff {j}: diff={c}, recovered={:?}",
                recovered
            );
        }
    }

    /// Realistic single-prime: VIA-C `q₂` ≈ 2³⁴ with CMux-sel gadget
    /// `(L=2, B=81)`, σ=4 Gaussian.
    #[test]
    fn gadget_product_at_via_c_q2_b81_l2_gaussian() {
        let q = crate::algebra::zq::modulus::paper::ViaCQ2::default();
        const L: usize = 2;
        let mut sk_prg = Shake256Prg::new(b"sk-gp-vc-q2");
        let sk =
            SecretKey::<16, SinglePolyViaCQ2<16>>::keygen(q, Distribution::Ternary, &mut sk_prg);
        let mut enc_prg = Shake256Prg::new(b"enc-gp-vc-q2");

        let m2_coeffs: [u64; 16] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16];
        let m2: SinglePolyViaCQ2<16> = Poly::new(q, m2_coeffs);
        let rlev =
            sk.encrypt_rlev::<L>(&m2, 81, Distribution::Gaussian { sigma: 4.0 }, &mut enc_prg);

        // Small-norm m1 (binary).
        let m1: SinglePolyViaCQ2<16> =
            Poly::new(q, [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1]);
        let ct = rlev.gadget_product(&m1, 81);
        let recovered = sk.decrypt_raw(&ct);

        let expected = m1 * m2;
        let diff = recovered - expected;
        let mut centred = [0i64; 16];
        diff.to_centered_coeffs(&mut centred);
        // Bound: ||m1||_1 ≤ 6, σ=4 → ~6 · 8σ = 192, plus per-level
        // reconstruction error (L·B/4 = 40). Use 512 generously.
        for (j, &c) in centred.iter().enumerate() {
            assert!(
                c.abs() <= 512,
                "VIA-C q₂ gadget product noise at coeff {j}: {c}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // gadget_product — eval-backed; bit-identical to the schoolbook reference
    // -----------------------------------------------------------------------

    /// Schoolbook reference: the pre-NTT coefficient-form gadget-product loop.
    /// Pins that the eval-backed [`RLevCiphertext::gadget_product`] is
    /// **bit-identical** at NTT-friendly moduli (where it takes the real NTT
    /// path) — the NTT is an exact ring isomorphism, so they must agree exactly,
    /// not merely within noise.
    fn schoolbook_gadget_product<const N: usize, R: RingPoly<N>, const L: usize>(
        rlev: &RLevCiphertext<N, R, L>,
        plaintext: &R,
        base: u64,
    ) -> RLWECiphertext<N, R> {
        let modulus = plaintext.modulus();
        let mut scratch = [0i128; N];
        gadget_scale_into::<N, R>(plaintext, base, L as u8, &mut scratch);
        let mut result_mask = R::zero(modulus);
        let mut result_body = R::zero(modulus);
        let mut digit_buf = [0i64; N];
        for k in 0..L {
            gadget_extract_lsb_into::<N>(base, &mut scratch, &mut digit_buf);
            let digit_poly = R::from_centered_i64s(modulus, &digit_buf);
            let sample = &rlev.samples[L - 1 - k];
            result_mask += digit_poly * sample.mask;
            result_body += digit_poly * sample.body;
        }
        RLWECiphertext::new(result_mask, result_body)
    }

    /// Single-prime NTT-friendly modulus `q = 17` at `N = 4`: the eval-backed
    /// (real NTT) `gadget_product` equals the schoolbook reference exactly.
    #[test]
    fn gadget_product_matches_schoolbook_single_prime() {
        use crate::algebra::zq::modulus::ConstModulus;
        type P = Poly<4, ConstModulus<17>, Coefficient>;
        const L: usize = 4;
        let q = ConstModulus::<17>;
        let samples: [RLWECiphertext<4, P>; L] = core::array::from_fn(|i| {
            let i = i as u128;
            let mask = <P as RingPoly<4>>::from_u128_coeffs(
                q,
                &[(3 * i + 1) % 17, (i + 4) % 17, (5 * i) % 17, (i * i) % 17],
            );
            let body = <P as RingPoly<4>>::from_u128_coeffs(
                q,
                &[(i + 7) % 17, (2 * i + 1) % 17, (4 * i) % 17, (i + 9) % 17],
            );
            RLWECiphertext::new(mask, body)
        });
        let rlev = RLevCiphertext::<4, P, L>::new(samples);
        let plaintext = <P as RingPoly<4>>::from_u128_coeffs(q, &[3, 0, 5, 1]);
        let base = 2u64;
        let got = rlev.gadget_product(&plaintext, base);
        let want = schoolbook_gadget_product(&rlev, &plaintext, base);
        assert_eq!(got.mask, want.mask, "single-prime mask mismatch");
        assert_eq!(got.body, want.body, "single-prime body mismatch");
    }

    /// RNS counterpart on the VIA `q₁` basis (NTT-friendly at `N = 4`):
    /// the per-slot NTT path equals the schoolbook reference exactly.
    #[test]
    fn gadget_product_matches_schoolbook_rns() {
        use crate::algebra::rns::basis::paper::ViaQ1Rns;
        type P = PolyRns<4, ViaQ1Rns, Coefficient>;
        const L: usize = 5;
        let basis = ViaQ1Rns::default();
        let samples: [RLWECiphertext<4, P>; L] = core::array::from_fn(|i| {
            let i = i as u128;
            let mask = <P as RingPoly<4>>::from_u128_coeffs(
                basis,
                &[1000 * i + 1, 2 * i + 3, 7 * i, i * i + 5],
            );
            let body =
                <P as RingPoly<4>>::from_u128_coeffs(basis, &[i + 11, 3 * i + 2, 9 * i, i + 13]);
            RLWECiphertext::new(mask, body)
        });
        let rlev = RLevCiphertext::<4, P, L>::new(samples);
        let plaintext = <P as RingPoly<4>>::from_u128_coeffs(basis, &[3, 1, 5, 2]);
        let base = 8u64;
        let got = rlev.gadget_product(&plaintext, base);
        let want = schoolbook_gadget_product(&rlev, &plaintext, base);
        assert_eq!(got.mask, want.mask, "RNS mask mismatch");
        assert_eq!(got.body, want.body, "RNS body mismatch");
    }

    /// T7 eval-key path: `to_eval().gadget_product(...)` (samples pre-transformed
    /// once) equals the coefficient-form `gadget_product(...)` **exactly** — so
    /// storing a static key in eval form is bit-identical on the result. Covers
    /// single-prime (`q=17`) and RNS (`q₁`) NTT-friendly moduli; since the
    /// coeff path is itself pinned to schoolbook above, this transitively pins the
    /// eval path to schoolbook.
    #[test]
    fn eval_gadget_product_matches_coeff() {
        use crate::algebra::rns::basis::paper::ViaQ1Rns;
        use crate::algebra::zq::modulus::ConstModulus;

        // Single-prime q=17 @ N=4 (same vectors as the schoolbook test).
        {
            type P = Poly<4, ConstModulus<17>, Coefficient>;
            const L: usize = 4;
            let q = ConstModulus::<17>;
            let samples: [RLWECiphertext<4, P>; L] = core::array::from_fn(|i| {
                let i = i as u128;
                let mask = <P as RingPoly<4>>::from_u128_coeffs(
                    q,
                    &[(3 * i + 1) % 17, (i + 4) % 17, (5 * i) % 17, (i * i) % 17],
                );
                let body = <P as RingPoly<4>>::from_u128_coeffs(
                    q,
                    &[(i + 7) % 17, (2 * i + 1) % 17, (4 * i) % 17, (i + 9) % 17],
                );
                RLWECiphertext::new(mask, body)
            });
            let rlev = RLevCiphertext::<4, P, L>::new(samples);
            let plaintext = <P as RingPoly<4>>::from_u128_coeffs(q, &[3, 0, 5, 1]);
            let base = 2u64;
            let coeff = rlev.gadget_product(&plaintext, base);
            let eval = rlev.to_eval().gadget_product(&plaintext, base);
            assert_eq!(coeff.mask, eval.mask, "single-prime eval mask mismatch");
            assert_eq!(coeff.body, eval.body, "single-prime eval body mismatch");
        }

        // RNS q₁ @ N=4 (same vectors as the schoolbook RNS test).
        {
            type P = PolyRns<4, ViaQ1Rns, Coefficient>;
            const L: usize = 5;
            let basis = ViaQ1Rns::default();
            let samples: [RLWECiphertext<4, P>; L] = core::array::from_fn(|i| {
                let i = i as u128;
                let mask = <P as RingPoly<4>>::from_u128_coeffs(
                    basis,
                    &[1000 * i + 1, 2 * i + 3, 7 * i, i * i + 5],
                );
                let body = <P as RingPoly<4>>::from_u128_coeffs(
                    basis,
                    &[i + 11, 3 * i + 2, 9 * i, i + 13],
                );
                RLWECiphertext::new(mask, body)
            });
            let rlev = RLevCiphertext::<4, P, L>::new(samples);
            let plaintext = <P as RingPoly<4>>::from_u128_coeffs(basis, &[3, 1, 5, 2]);
            let base = 8u64;
            let coeff = rlev.gadget_product(&plaintext, base);
            let eval = rlev.to_eval().gadget_product(&plaintext, base);
            assert_eq!(coeff.mask, eval.mask, "RNS eval mask mismatch");
            assert_eq!(coeff.body, eval.body, "RNS eval body mismatch");
        }
    }

    /// Non-NTT runtime modulus (`DynModulus`, power-of-two): `gadget_product`
    /// takes the schoolbook identity-fallback and equals the reference —
    /// confirming the dummy `RingPolyEval` impl drives `gadget_product`.
    #[test]
    fn gadget_product_dynmodulus_fallback_matches_schoolbook() {
        use crate::algebra::zq::modulus::DynModulus;
        type P = Poly<4, DynModulus, Coefficient>;
        const L: usize = 4;
        let q = DynModulus::new(1024);
        let samples: [RLWECiphertext<4, P>; L] = core::array::from_fn(|i| {
            let i = i as u128;
            let mask = <P as RingPoly<4>>::from_u128_coeffs(
                q,
                &[
                    (3 * i + 1) % 1024,
                    (i + 4) % 1024,
                    (5 * i) % 1024,
                    (i * i) % 1024,
                ],
            );
            let body = <P as RingPoly<4>>::from_u128_coeffs(
                q,
                &[
                    (i + 7) % 1024,
                    (2 * i + 1) % 1024,
                    (4 * i) % 1024,
                    (i + 9) % 1024,
                ],
            );
            RLWECiphertext::new(mask, body)
        });
        let rlev = RLevCiphertext::<4, P, L>::new(samples);
        let plaintext = <P as RingPoly<4>>::from_u128_coeffs(q, &[3, 0, 5, 1]);
        let base = 2u64;
        let got = rlev.gadget_product(&plaintext, base);
        let want = schoolbook_gadget_product(&rlev, &plaintext, base);
        assert_eq!(got.mask, want.mask);
        assert_eq!(got.body, want.body);
    }

    /// `encrypt_rlev_boxed` must draw PRG identically to `encrypt_rlev` and so
    /// produce a byte-identical RLev (the heap-building must not perturb the
    /// key). Exercised at the RNS q₁ — the case it exists for.
    #[cfg(feature = "alloc")]
    #[test]
    fn encrypt_rlev_boxed_matches_by_value_rns_q1() {
        let basis = crate::algebra::rns::basis::paper::ViaCQ1Rns::default();
        const L: usize = 8;
        let mut sk_prg = Shake256Prg::new(b"sk-rlev-boxed-parity");
        let sk =
            SecretKey::<16, RnsPolyViaCQ1<16>>::keygen(basis, Distribution::Ternary, &mut sk_prg);
        let message_coeffs: [i64; 16] = [1, -1, 0, 2, -2, 1, 0, 0, 1, -1, 0, 0, 1, 0, -1, 1];
        let message =
            <RnsPolyViaCQ1<16> as RingPoly<16>>::from_centered_i64s(basis, &message_coeffs);

        let mut prg_a = Shake256Prg::new(b"rlev-boxed-parity-enc");
        let by_value = sk.encrypt_rlev::<L>(
            &message,
            8,
            Distribution::Gaussian { sigma: 4.0 },
            &mut prg_a,
        );
        let mut prg_b = Shake256Prg::new(b"rlev-boxed-parity-enc");
        let boxed = sk.encrypt_rlev_boxed::<L>(
            &message,
            8,
            Distribution::Gaussian { sigma: 4.0 },
            &mut prg_b,
        );

        assert_eq!(
            by_value.samples, boxed.samples,
            "boxed RLev must be byte-identical to the by-value RLev"
        );
    }
}
