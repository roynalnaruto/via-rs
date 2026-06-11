//! `via-primitives` — a pure-no-std Rust implementation of the VIA, VIA-C, and VIA-B
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
//! - [`gates`] — Layer 4: homomorphic gates (§4.1–§4.7) — CMux/DMux and their
//!   recursive trees, controlled rotation (CRot), and RLWE→RGSW conversion.
//! - [`conversion`] — Layer 5: the MLWE LWE→RLWE conversion cascade (§5.1–§5.5)
//!   — MLWE embedding, single Conv₂ step, the full $\log_2 n$-deep cascade,
//!   its key generation, and RLWE→MLWE coefficient extraction (`Extr_d`).
//! - [`params`] — Ergonomic type aliases for the paper parameter sets (formerly
//!   `encryption::aliases`). Re-exported from `encryption` for backward compat.
//!
//! Further layers (protocol composites) live in `via-protocol`, `via-client`,
//! and `via-server`.
//!
//! See `.docs/primitives.md` for the layered primitive overview and
//! `.docs/via.pdf` for the original paper.

#![no_std]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

// Optional heap support (the `alloc` feature). Used only by the paper-scale
// LWE→RLWE `…_boxed` cascade builders; the crate is otherwise no-alloc.
#[cfg(feature = "alloc")]
extern crate alloc;

pub mod algebra;
pub mod conversion;
pub mod encryption;
pub mod gates;
pub mod params;
pub mod sampling;
pub mod switching;

/// Test-only helpers shared across unit tests (e.g. the `SplitMix64` PRG). Gated
/// out of every non-test build, so it has no effect on the `no_std` surface.
#[cfg(test)]
mod test_util;
