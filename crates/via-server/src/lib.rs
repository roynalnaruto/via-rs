//! `via-server` — server-side DB encoding, query decompression, and answer
//! generation for the VIA PIR family.
//!
//! Depends on `via-protocol` (wire types) and `via-primitives` (crypto).
//! **Must not depend on `via-client`** — enforced by a CI dep-tree check.

#![cfg_attr(not(feature = "alloc"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

extern crate alloc;

pub mod first_dim;
pub mod query_decomp;
pub mod resp_comp;
pub mod setup_db;

pub use first_dim::first_dim;
pub use query_decomp::query_decomp;
pub use resp_comp::resp_comp;
pub use setup_db::setup_db;
