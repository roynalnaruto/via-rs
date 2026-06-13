//! `via-primitives` — a pure-no-std Rust implementation of the VIA, VIA-C, and VIA-B
//! single-server PIR schemes (Liu, Wang, Zhang, 2025).
//!
//! Each layer of the primitive stack lives in its own top-level module:
//!
//! - [`algebra`] — arithmetic substrate.
//! - [`sampling`] — SHAKE-256 PRG and the four sampling distributions
//!   consumed by every higher layer.
//! - [`encryption`] — ciphertext types (SecretKey, RLWE, RLev,
//!   RGSW, MLWE, ModSwitched) and the primitive operations on them.
//!   Generic over a polynomial backend via
//!   [`algebra::ring::RingPoly`], so the same code instantiates against
//!   either the single-prime [`algebra::ring::element::Poly`] or the RNS
//!   [`algebra::ring::rns_element::PolyRns`] carrier.
//! - [`switching`] — modulus switching, ring switching,
//!   and secret-key rekeying — the reshaping primitives that
//!   move ciphertexts between moduli and ring degrees.
//! - [`gates`] — homomorphic gates — CMux/DMux and their
//!   recursive trees, controlled rotation (CRot), and RLWE→RGSW conversion.
//! - [`conversion`] — the MLWE LWE→RLWE conversion cascade
//!   — MLWE embedding, single Conv₂ step, the full $\log_2 n$-deep cascade,
//!   its key generation, and RLWE→MLWE coefficient extraction (`Extr_d`).
//! - [`params`] — Ergonomic type aliases for the paper parameter sets (formerly
//!   `encryption::aliases`). Re-exported from `encryption` for backward compat.
//!
//! Further layers (protocol composites) live in `via-protocol`, `via-client`,
//! and `via-server`.

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
