//! Layer 3 — switching primitives. See `.docs/primitives.md` §3.
//!
//! Reshape ciphertexts between moduli (§3.1 symmetric / §3.2 asymmetric
//! ModSwitch), between rings of different degree (§3.3 RingSwitch), and
//! re-interpret a small-coefficient secret key at a new modulus (§3.4
//! rekeying).
//!
//! ## GPU-portability convention
//!
//! Coefficient-level arithmetic lives in [`kernels`] as POD-by-value + flat
//! slice functions (the Layer-0 kernel shape), while the orchestrators in the
//! sibling submodules handle ring-type plumbing and PRG draws. This keeps the
//! numeric hot loops trait-free so they can later lower to CUDA / Metal.
pub mod kernels;
// §3.1 + §3.2 — filled in by Part 1.
pub mod mod_switch;
// §3.4 — filled in by Part 4.
pub mod rekey;
// §3.3 — filled in by Part 3.
pub mod ring_switch;

pub use kernels::RescaleConsts;
pub use mod_switch::{mod_switch_asym, mod_switch_sym};
pub use rekey::rekey_secret_key;
pub use ring_switch::{RingSwitchKey, RingSwitchKeyEval, gen_rsk, ring_switch, ring_switch_eval};
// Kernels are reachable via `switching::kernels::rekey::*` but intentionally
// not re-exported here (the orchestrator is the public entry point).
