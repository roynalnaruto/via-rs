//! Layer 1 — sampling: the deterministic PRG and the per-distribution
//! coefficient samplers used by every higher layer.
//!
//! Each sub-module implements one section of `.docs/primitives.md` §1.x:
//!
//! - [`prg`] — §1.1 SHAKE-256 counter-mode PRG. Drives every randomized
//!   operation downstream; preserves byte-exact cross-language reproducibility.
//!
//! Sub-modules for §1.2–§1.6 (uniform, ternary, bounded-uniform, discrete
//! Gaussian, and the `ErrorDist` dispatcher) land in subsequent phases.
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
//! ## Dependency direction
//!
//! Layer 1 imports Layer 0 (notably the [`Modulus`](crate::algebra::zq::modulus::Modulus)
//! trait for the lift helpers in a later phase); nothing in Layer 0 depends on
//! Layer 1.

pub mod prg;
