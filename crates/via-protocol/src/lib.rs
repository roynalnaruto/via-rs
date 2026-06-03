//! `via-protocol` — shared wire types, `PIRParams` presets, and `ViaError`
//! for the VIA PIR family.
//!
//! `#![no_std]` when the `alloc` feature is disabled; `alloc` is on by default.
//! `tracing` spans live at this boundary; `via-primitives` keeps `_CHECK`+panic.

#![cfg_attr(not(feature = "alloc"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

// Re-export primitives so protocol consumers only need one dep declaration.
pub use via_primitives as primitives;
