//! `via-rs` — a pure-no-std Rust implementation of the VIA, VIA-C, and VIA-B
//! single-server PIR schemes (Liu, Wang, Zhang, 2025).
//!
//! See `.docs/primitives.md` for the layered primitive overview and
//! `.docs/via.pdf` for the original paper.

#![no_std]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

pub mod primitives;
