//! Switching primitives.
//!
//! Reshape ciphertexts between moduli (symmetric / asymmetric
//! ModSwitch), between rings of different degree (RingSwitch), and
//! re-interpret a small-coefficient secret key at a new modulus
//! (rekeying).
//!
//! ## GPU-portability convention
//!
//! Coefficient-level arithmetic lives in [`kernels`] as POD-by-value + flat
//! slice functions (the flat-slice kernel shape), while the orchestrators in
//! the sibling submodules handle ring-type plumbing and PRG draws. This keeps
//! the numeric hot loops trait-free so they can later lower to CUDA / Metal.
pub mod kernels;
pub mod mod_switch;
pub mod rekey;
pub mod ring_switch;

pub use kernels::RescaleConsts;
pub use mod_switch::{mod_switch_asym, mod_switch_sym};
pub use rekey::rekey_secret_key;
pub use ring_switch::{RingSwitchKey, RingSwitchKeyEval, gen_rsk, ring_switch, ring_switch_eval};
// Kernels are reachable via `switching::kernels::rekey::*` but intentionally
// not re-exported here (the orchestrator is the public entry point).
