//! Layer-7 (VIA-B) Rust-golden KAT fixtures.
//!
//! **No Python reference exists for VIA-B** (the spec implements only VIA/VIA-C).
//! Unlike `data/layer6_kats.rs` (cross-language), these fixtures are produced by
//! a `via-b,kat-regen`-gated Rust test (`just regen-kats-layer7`, Layer-7
//! Part 5) and checked in. They are asserted at the decrypt/plaintext boundary
//! (the recovered records / the paper interleave formula), not as ciphertext
//! byte-parity against an oracle.
//!
//! Placeholder until Part 5 — `#![allow(dead_code)]` so the orphan compiles if
//! pulled in early.
#![allow(dead_code)]

/// Deterministic seed for the Layer-7 KAT round-trips (pinned in Part 5).
pub const SEED: [u8; 32] = *b"via-b-layer7-kat-seed-0000000000";
