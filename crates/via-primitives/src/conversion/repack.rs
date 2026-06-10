//! Layer 7 — VIA-B homomorphic repacking primitives.
//!
//! `Embed_d` (multi-slot MLWEs embedding), MLWEs-to-MLWE / MLWEs-to-RLWE
//! conversion, `Repack_k` (= `Extr_{k/d}` ∘ MLWEs-to-RLWE), and the
//! cascade-key-suffix borrow that realises the paper's *no-new-offline-payload*
//! identity (§4.6).
//!
//! Filled by Layer-7 **Part 1** (engine + dedicated-key oracle) and **Part 2**
//! (key-reuse adapter + depth spike). Gated
//! `#[cfg(all(feature = "via-b", feature = "alloc"))]` at the
//! [`crate::conversion`] re-export boundary: the repack recursion is
//! runtime-depth (`log2(T·n1/n2)`), so it holds a heap `Vec` of MLWE
//! ciphertexts and a heterogeneous-degree key schedule.
//!
//! This module is intentionally empty until Part 1.
