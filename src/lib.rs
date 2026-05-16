//! `via-rs` — a pure-no-std Rust implementation of the VIA, VIA-C, and VIA-B
//! single-server PIR schemes (Liu, Wang, Zhang, 2025).
//!
//! Each layer of the paper's primitive stack lives in its own top-level module:
//!
//! - [`algebra`] — Layer 0: arithmetic substrate (§0.1–§0.6).
//!
//! Further layers (Sampling, RLWE, ring/key switching, homomorphic gates, MLWE
//! cascade, protocol composites) will land as further top-level modules.
//!
//! See `.docs/primitives.md` for the layered primitive overview and
//! `.docs/via.pdf` for the original paper.

#![no_std]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

pub mod algebra;
