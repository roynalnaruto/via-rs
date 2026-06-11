//! `via-client` — client-side keygen, query compression, and answer recovery
//! for the VIA PIR family.
//!
//! Depends on `via-protocol` (wire types) and `via-primitives` (crypto).
//! **Must not depend on `via-server`** — enforced by a CI dep-tree check.

#![cfg_attr(not(feature = "alloc"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

extern crate alloc;

#[cfg(feature = "via-b")]
pub mod batch;
pub mod client;
pub mod decompose;
pub mod query;

#[cfg(feature = "via-b")]
pub use batch::deinterleave_batch;
pub use client::Client;
pub use decompose::decompose_index;
