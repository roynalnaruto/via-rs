//! Layer 0 — the algebraic substrate underpinning VIA / VIA-C / VIA-B.
//!
//! Each sub-module implements one section of `.docs/primitives.md` §0.x:
//!
//! - [`zq`] — §0.1 integers modulo $q$.
//! - [`rns`] — §0.2 RNS / double-CRT representation.
//! - [`ring`] — §0.3 polynomial ring $R_{n, q}$, plus §0.4 negacyclic NTT,
//!   §0.5 ring embedding / projection, and §0.6 centred / balanced representation.
//!
//! Higher layers (sampling, RLWE, ring/key switching, homomorphic gates, the MLWE
//! cascade, and the protocol composites) live in sibling top-level modules; see
//! the crate-level documentation for the full layer map.
//!
//! The dependency direction is strictly bottom-up: nothing in this module
//! imports from higher layers.

pub mod ring;
pub mod rns;
pub(crate) mod wide;
pub mod zq;
