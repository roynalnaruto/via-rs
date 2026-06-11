//! Layer 2 — Encryption types and primitive operations.
//!
//! This module hosts the ciphertext datatypes and the basic cryptographic
//! operations on them, as specified in `.docs/primitives.md §2`:
//!
//! - §2.1 — ciphertext types: [`SecretKey`], [`RLWECiphertext`],
//!   [`RLevCiphertext`], [`RGSWCiphertext`], [`MLWECiphertext`],
//!   [`ModSwitchedCiphertext`].
//! - §2.2 — `keygen`, `encode`, `decode`, `encrypt`, `decrypt` (Phase 2-3, TBD).
//! - §2.2.5 — auxiliary RLWE primitives (Phase 4, TBD).
//! - §2.3 — gadget vector and decomposition (VIA convention, Phase 5, TBD).
//! - §2.4 — gadget product, external product, key switch (Phase 7-8, TBD).
//!
//! ## Backend abstraction
//!
//! Every ciphertext type is generic over `R: RingPoly<N>`
//! ([`crate::algebra::ring::RingPoly`]). The trait is implemented by both
//! the single-prime [`crate::algebra::ring::element::Poly<N, M, Coefficient>`]
//! and the RNS [`crate::algebra::ring::rns_element::PolyRns<N, B, Coefficient>`].
//! Layer 2 algorithms are written once and instantiate against either
//! backend.
//!
//! Convenience type aliases for the paper parameter sets live in
//! [`aliases`].
//!
//! ## Coefficient form only (for now)
//!
//! All Layer-2 ciphertexts hold coefficient-form polynomials, and every
//! polynomial multiply here takes the schoolbook negacyclic path. This stays
//! faithful to the Python reference at `.references/via-spec/pir/`, which is
//! coefficient-form throughout, so KAT parity is unaffected.
//!
//! This is a *wiring* gap, not a missing primitive: the negacyclic NTT in
//! `algebra::ring::ntt` is fully implemented and tested on both backends
//! (forward + inverse; single-prime `Poly` and RNS `PolyRns`), and the
//! `into_eval()` / `into_coeff()` form conversions are live. The pending
//! optimisation is to keep ciphertexts in evaluation form across the
//! gadget-product / external-product hot loops so the `O(N log N)` transform
//! cost is amortised over many multiplies.
//!
//! ## Example — full encrypt/decrypt round-trip
//!
//! ```
//! use via_primitives::algebra::ring::element::Poly;
//! use via_primitives::algebra::ring::form::Coefficient;
//! use via_primitives::algebra::zq::modulus::PowerOfTwoModulus;
//! use via_primitives::encryption::{SecretKey, decode, encode};
//! use via_primitives::sampling::distribution::Distribution;
//! use via_primitives::sampling::prg::Shake256Prg;
//!
//! type Plaintext = Poly<4, PowerOfTwoModulus<4>, Coefficient>;     // p = 16
//! type Ciphertext = Poly<4, PowerOfTwoModulus<10>, Coefficient>;   // q = 1024
//!
//! // Sample a ternary secret key.
//! let mut prg = Shake256Prg::new(b"phase3-example");
//! let sk = SecretKey::<4, Ciphertext>::keygen(
//!     PowerOfTwoModulus,
//!     Distribution::Ternary,
//!     &mut prg,
//! );
//!
//! // Encode → encrypt → decrypt → check.
//! let m = Plaintext::new(PowerOfTwoModulus, [0, 1, 7, 15]);
//! let encoded: Ciphertext = encode(&m, PowerOfTwoModulus);
//! let ct = sk.encrypt(&encoded, Distribution::Ternary, &mut prg);
//! let recovered: Plaintext = sk.decrypt(&ct, PowerOfTwoModulus);
//! for i in 0..4 {
//!     assert_eq!(m.coeff(i), recovered.coeff(i));
//! }
//!
//! // The bare `encode` / `decode` round-trip (no encryption) still works.
//! let recovered_via_decode: Plaintext = decode(&encoded, PowerOfTwoModulus);
//! for i in 0..4 {
//!     assert_eq!(m.coeff(i), recovered_via_decode.coeff(i));
//! }
//!
//! // Phase-6 primitive: encrypt a raw message as an RLev (no Δ encoding).
//! // `samples[i]` decrypts to `g_i · m`; the full meaning is recovered by
//! // Phase-7's `gadget_product`.
//! let m_raw = Ciphertext::new(PowerOfTwoModulus, [0, 1, 7, 15]);
//! let _rlev = sk.encrypt_rlev::<4>(&m_raw, 2, Distribution::Ternary, &mut prg);
//!
//! // Phase-8 primitive: generate a key-switching key and convert a
//! // ciphertext from one secret key to another.
//! let dst_sk = SecretKey::<4, Ciphertext>::keygen(
//!     PowerOfTwoModulus,
//!     Distribution::Ternary,
//!     &mut prg,
//! );
//! let ksk = via_primitives::encryption::gen_ksk::<4, Ciphertext, 4>(
//!     &sk, &dst_sk, 2, Distribution::Ternary, &mut prg,
//! );
//! let _switched = ksk.key_switch(&ct, 2);
//! ```

pub mod gadget;
pub mod keyswitch;
pub mod mlwe;
pub mod rgsw;
pub mod rlev;
pub mod rlwe;
pub mod types;

pub use keyswitch::gen_ksk;

pub use gadget::{
    gadget_decompose_into, gadget_extract_lsb_into, gadget_scale_into, gadget_vector_values,
    reconstruct,
};
pub use mlwe::MLWECiphertext;
pub use rlwe::{decode, encode};
pub use types::{ModSwitchedCiphertext, RGSWCiphertext, RLWECiphertext, RLevCiphertext, SecretKey};

/// Backward-compatible re-export: `encryption::aliases` still resolves to the
/// top-level [`crate::params`] module (relocated there so the paper-parameter
/// aliases can name Layer-3 types without an upward-layer import).
pub use crate::params as aliases;
