//! The algebraic substrate underpinning VIA / VIA-C / VIA-B.
//!
//! Each sub-module implements one concept:
//!
//! - [`zq`] — integers modulo $q$.
//! - [`rns`] — RNS / double-CRT representation.
//! - [`ring`] — polynomial ring $R_{n, q}$, plus negacyclic NTT,
//!   ring embedding / projection, and centred / balanced representation.
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
