//! Layer 1 — sampling: the deterministic PRG and the per-distribution
//! coefficient samplers used by every higher layer.
//!
//! Each sub-module implements one section of `.docs/primitives.md` §1.x:
//!
//! - [`prg`] — §1.1 SHAKE-256 counter-mode PRG. Drives every randomized
//!   operation downstream; preserves byte-exact cross-language reproducibility.
//! - [`uniform`] — §1.2 uniform sampler over $\mathbb{Z}_q$.
//! - [`ternary`] — §1.3 ternary sampler over $\{-1, 0, 1\}$.
//! - [`bounded`] — §1.4 bounded-uniform sampler over $[-B, B]$.
//! - [`gaussian`] — §1.5 discrete Gaussian sampler via Box-Muller. The
//!   only floating-point primitive in the crate; routed through `libm` for
//!   cross-platform determinism.
//! - [`distribution`] — §1.6 [`Distribution`] dispatcher enum: a typed
//!   bundling of `(which sampler, what parameter)` used at every key- and
//!   error-sampling call site. Sampling-only; does **not** know about
//!   ciphertexts or secret keys (that's Layer 2).
//! - [`lift`] — Layer-1 → Layer-0 bridge:
//!   [`lift_centered_i8_into_zq`] / [`lift_centered_i32_into_zq`] /
//!   [`lift_centered_i64_into_zq`] reduce signed sampler outputs into
//!   canonical $[0, q)$ under a [`Modulus`](crate::algebra::zq::modulus::Modulus).
//!
//! ## Cross-language reproducibility contract
//!
//! The whole point of this layer is that two implementations seeded the same
//! way must produce **byte-identical** key material, ciphertexts, and answers
//! at every later layer. That contract bottoms out here: the PRG framing
//! (SHAKE-256, 136-byte blocks, little-endian `u64` counter) and every
//! sampler's per-coefficient byte budget must match the Python reference at
//! `.references/via-spec/pir/primitives/sampling.py` exactly. Unit tests in
//! each sub-module pin a handful of seed → output vectors lifted from the
//! reference.
//!
//! ## Floating-point carve-out
//!
//! §1.5 (the discrete Gaussian) is the **only** floating-point primitive in
//! via-rs. Every other primitive — at every other layer — is integer-only.
//! The Gaussian's Box-Muller path routes through [`libm`] (`log`, `sqrt`,
//! `cos`, `rint`) so the f64 output is deterministic across platforms.
//! Rounding uses round-half-to-even (banker's) via `libm::rint`, matching
//! Python's built-in `round()`.
//!
//! ## Dependency direction
//!
//! Layer 1 imports Layer 0 — notably the
//! [`Modulus`](crate::algebra::zq::modulus::Modulus) trait, used by `lift` to
//! reduce signed coefficients via `reduce_i64` (constant-time over the input
//! sign, which matters for secret-key coefficients). Nothing in Layer 0
//! depends on Layer 1.
//!
//! ## Public surface
//!
//! The canonical entry points are re-exported at the module root for ergonomic
//! `use crate::sampling::{...};` access; the per-sub-module paths remain
//! available for code that wants to be explicit about provenance.

pub mod bounded;
pub mod distribution;
pub mod gaussian;
pub mod lift;
pub mod prg;
pub mod ternary;
pub mod uniform;

pub use distribution::Distribution;
pub use lift::{lift_centered_i8_into_zq, lift_centered_i32_into_zq, lift_centered_i64_into_zq};
pub use prg::Shake256Prg;
