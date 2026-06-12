//! Homomorphic gates.
//!
//! Encrypted control flow built on the external product:
//!
//! - [`cmux`] / [`dmux`] — 1-of-2 multiplexer / 1-to-2 demultiplexer.
//! - [`cmux_tree`] / [`dmux_tree`] — recursive 1-of-2^m / 1-to-2^m
//!   selection tree; slice-based, no allocation.
//! - [`crot`] / [`CRotDir`] — controlled rotation (forward + slot-extract).
//! - [`rotate()`] — deterministic ciphertext rotation by $X^k$.
//! - [`rlwe_to_rgsw`] / [`gen_rlwe_to_rgsw_key`] — RLWE→RGSW conversion.
//! - [`mod_switch_rgsw`] — RGSW modulus-switch (consumes the switching-layer
//!   [`crate::switching::mod_switch::mod_switch_sym`]).
//!
//! ## File → primitive map
//!
//! | File         | Primitives                                  |
//! |--------------|---------------------------------------------|
//! | `rotate.rs`  | `rotate`, `crot`, `CRotDir`                 |
//! | `mux.rs`     | `cmux`, `dmux`, `cmux_tree`/`dmux_tree`     |
//! | `convert.rs` | `rlwe_to_rgsw` (+key), `mod_switch_rgsw`    |
//!
//! These gates add **no** new [`crate::algebra::ring::RingPoly`] trait items —
//! `RingPoly::mul_x_pow` and `RingPoly::project_at` suffice.

pub mod rotate;
pub mod mux;
pub mod convert;

// Atomic gates.
pub use mux::{cmux, dmux};
pub use rotate::rotate;

// Composites (trees, crot).
pub use mux::{cmux_tree, dmux_tree};
pub use rotate::{CRotDir, crot};

// Conversion (rlwe_to_rgsw, mod_switch_rgsw).
#[cfg(feature = "alloc")]
pub use convert::gen_rlwe_to_rgsw_key_boxed;
pub use convert::{gen_rlwe_to_rgsw_key, mod_switch_rgsw, rlwe_to_rgsw, rlwe_to_rgsw_eval};
