//! `via-integration` — cross-crate integration tests for the VIA PIR family.
//!
//! This crate carries the **client ↔ server** end-to-end tests that cannot live
//! in `via-client` or `via-server`: the CI dependency-isolation gate
//! (`cargo tree --package via-client | grep via-server`, and vice-versa) forbids
//! either crate from depending on the other, even as a dev-dependency. A neutral
//! third crate that depends on both is the only place a full round-trip test can
//! live.
//!
//! There is no library code here — see `tests/`.

#![no_std]
