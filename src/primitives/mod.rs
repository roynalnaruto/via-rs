//! VIA primitives — the layered building blocks of the protocol.
//!
//! Each sub-module implements one section of `.docs/primitives.md`. The layers
//! are stacked so that every higher layer only depends on lower layers:
//!
//! - **Layer 0** — arithmetic substrate ([`zq`] for §0.1, [`rns`] for §0.2;
//!   NTT, ring embedding, centred repr to follow).
//! - **Layer 1** — sampling (PRG, uniform, ternary, bounded-uniform, discrete
//!   Gaussian).
//! - **Layer 2** — RLWE ciphertext types and primitive homomorphic operations.
//! - **Layer 3+** — ring switching, homomorphic gates, MLWE cascade, protocol
//!   composites.

pub mod ring;
pub mod rns;
pub mod zq;
