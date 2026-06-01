//! `via-rs` — a pure-no-std Rust implementation of the VIA, VIA-C, and VIA-B
//! single-server PIR schemes (Liu, Wang, Zhang, 2025).
//!
//! Each layer of the paper's primitive stack lives in its own top-level module:
//!
//! - [`algebra`] — Layer 0: arithmetic substrate (§0.1–§0.6).
//! - [`sampling`] — Layer 1: SHAKE-256 PRG and the four sampling distributions
//!   (§1.1–§1.6) consumed by every higher layer.
//! - [`encryption`] — Layer 2: ciphertext types (SecretKey, RLWE, RLev,
//!   RGSW, MLWE, ModSwitched) and the primitive operations on them
//!   (§2.1–§2.4). Generic over a polynomial backend via
//!   [`algebra::ring::RingPoly`], so the same code instantiates against
//!   either the single-prime [`algebra::ring::element::Poly`] or the RNS
//!   [`algebra::ring::rns_element::PolyRns`] carrier.
//! - [`switching`] — Layer 3: modulus switching (§3.1–§3.2), ring switching
//!   (§3.3), and secret-key rekeying (§3.4) — the reshaping primitives that
//!   move ciphertexts between moduli and ring degrees.
//!
//! Further layers (homomorphic gates, MLWE cascade, protocol composites) will
//! land as further top-level modules.
//!
//! See `.docs/primitives.md` for the layered primitive overview and
//! `.docs/via.pdf` for the original paper.

#![no_std]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

pub mod algebra;
pub mod encryption;
pub mod sampling;
pub mod switching;
