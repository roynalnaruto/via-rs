//! Layer 4 — homomorphic gates. See `.docs/primitives.md` §4.
//!
//! Encrypted control flow built on the external product (§2.4):
//!
//! - §4.1 [`cmux`] / §4.2 [`dmux`] — 1-of-2 multiplexer / 1-to-2 demultiplexer.
//! - §4.3 [`cmux_tree`] / [`dmux_tree`] — recursive 1-of-2^m / 1-to-2^m
//!   (Algorithms 1-2); slice-based, no allocation.
//! - §4.4 [`crot`] / [`CRotDir`] — controlled rotation (forward + slot-extract).
//! - §4.5 [`rotate()`] — deterministic ciphertext rotation by $X^k$.
//! - §4.6 [`rlwe_to_rgsw`] / [`gen_rlwe_to_rgsw_key`] — RLWE→RGSW conversion.
//! - §4.7 [`mod_switch_rgsw`] — RGSW modulus-switch (consumes the Layer-3
//!   [`crate::switching::mod_switch::mod_switch_sym`]).
//!
//! ## File → primitive map
//!
//! | File         | Primitives                                            |
//! |--------------|-------------------------------------------------------|
//! | `rotate.rs`  | §4.5 `rotate`, §4.4 `crot`, `CRotDir`                  |
//! | `mux.rs`     | §4.1 `cmux`, §4.2 `dmux`, §4.3 `cmux_tree`/`dmux_tree` |
//! | `convert.rs` | §4.6 `rlwe_to_rgsw` (+key), §4.7 `mod_switch_rgsw`     |
//!
//! Layer 4 adds **no** new [`crate::algebra::ring::RingPoly`] trait items —
//! `RingPoly::mul_x_pow` and `RingPoly::project_at` from Layer 3 suffice.

// §4.5 rotate + §4.4 crot — Parts 1-2.
pub mod rotate;
// §4.1 cmux + §4.2 dmux + §4.3 trees — Parts 1-2.
pub mod mux;
// §4.6 rlwe_to_rgsw (+key) + §4.7 mod_switch_rgsw — Part 3.
pub mod convert;

// Parts 1-3 APPEND their `pub use <submod>::{...};` lines below. Part 0 owns
// the `pub mod` lines only; do not add `pub mod` declarations elsewhere.

// Part 1 — atomic gates (§4.5, §4.1, §4.2).
pub use mux::{cmux, dmux};
pub use rotate::rotate;

// Part 2 — composites (§4.3 trees, §4.4 crot).
pub use mux::{cmux_tree, dmux_tree};
pub use rotate::{CRotDir, crot};

// Part 3 — conversion (§4.6 rlwe_to_rgsw, §4.7 mod_switch_rgsw).
#[cfg(feature = "alloc")]
pub use convert::gen_rlwe_to_rgsw_key_boxed;
pub use convert::{gen_rlwe_to_rgsw_key, mod_switch_rgsw, rlwe_to_rgsw};
