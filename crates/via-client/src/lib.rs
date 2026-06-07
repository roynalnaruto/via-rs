//! `via-client` — client-side keygen, query compression, and answer recovery
//! for the VIA PIR family.
//!
//! Depends on `via-protocol` (wire types) and `via-primitives` (crypto).
//! **Must not depend on `via-server`** — enforced by a CI dep-tree check.

#![cfg_attr(not(feature = "alloc"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]
