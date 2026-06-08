//! `via-server` — server-side DB encoding, query decompression, and answer
//! generation for the VIA PIR family.
//!
//! Depends on `via-protocol` (wire types) and `via-primitives` (crypto).
//! **Must not depend on `via-client`** — enforced by a CI dep-tree check.

#![cfg_attr(not(feature = "alloc"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

extern crate alloc;

pub mod query_decomp;
pub mod setup_db;

pub use query_decomp::query_decomp;
pub use setup_db::setup_db;
