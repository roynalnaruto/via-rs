//! `via-server` — server-side DB encoding, query decompression, and answer
//! generation for the VIA PIR family.
//!
//! Depends on `via-protocol` (wire types) and `via-primitives` (crypto).
//! **Must not depend on `via-client`** — enforced by a CI dep-tree check.

#![cfg_attr(not(feature = "alloc"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

extern crate alloc;

pub mod answer;
#[cfg(feature = "via-b")]
pub mod batch;
pub mod first_dim;
pub mod prepared_db;
pub mod prepared_keys;
pub mod query_decomp;
pub mod resp_comp;
pub mod scheme;
pub mod setup_db;

#[cfg(feature = "via-b")]
pub use answer::ViaBServer;
pub use answer::{Server, ServerConfig, ViaCServer, answer_one_query, answer_through_crot};
#[cfg(feature = "via-b")]
pub use batch::answer_batch;
pub use first_dim::first_dim;
pub use prepared_db::PreparedDb;
pub use prepared_keys::PreparedKeys;
pub use query_decomp::query_decomp;
pub use resp_comp::resp_comp;
pub use scheme::{Scheme, ServerScheme};
pub use setup_db::setup_db;
