//! GPU-portable coefficient-level kernels for switching primitives.
//!
//! Each kernel follows the low-level shape convention (POD constants by value +
//! flat slices; see [`crate::algebra::zq::ops`]) so the same body lowers to a
//! CUDA / Metal launch with no trait indirection on the device.
pub mod mod_switch;
pub mod rekey;

pub use mod_switch::RescaleConsts;
