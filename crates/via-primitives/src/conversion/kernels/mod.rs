//! GPU-portable scalar-level kernels for conversion.
//!
//! POD constants by value + flat slices (see [`crate::algebra::zq::ops`]); the
//! same bodies lower to a CUDA / Metal launch with no trait indirection on the
//! device. The orchestrators in the sibling submodules do ring-type plumbing
//! and PRG draws.
pub mod lwe;
