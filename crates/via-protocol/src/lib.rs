//! `via-protocol` — shared wire types, `PIRParams` presets, and `ViaError`
//! for the VIA PIR family.
//!
//! `#![no_std]` when the `alloc` feature is disabled; `alloc` is on by default.
//! `tracing` spans live at this boundary; `via-primitives` keeps `_CHECK`+panic.

#![cfg_attr(not(feature = "alloc"), no_std)]
#![deny(rustdoc::broken_intra_doc_links)]
#![warn(missing_docs)]

extern crate alloc;

// Internal modules. The public API is re-exported at the crate root below —
// consumers import `via_protocol::{…}`, not submodule paths.
mod error;
mod params;
mod presets;

// Re-export primitives so protocol consumers only need one dep declaration.
pub use via_primitives as primitives;

pub use error::{Result, ViaError};
pub use params::{KeyDist, PIRParams};
pub use presets::{
    REALISTIC_PARAMS, TOY_PARAMS, ViaCPublicParams, ViaCRealisticParams, ViaCToyParams,
    pir_params_matches_preset,
};

#[cfg(test)]
mod smoke {
    //! Cross-crate `$crate::` macro-hygiene smoke test.
    //!
    //! The `lwe_to_rlwe_cascade!` macro is `#[macro_export]`ed with
    //! `$crate::`-hygienic paths and instantiated inside `via-primitives`
    //! (e.g. `LweToRlweKeyN8`). Naming the generated type + generator from a
    //! *dependent* crate forces the compiler to resolve those macro-expanded
    //! paths (and every field type, e.g. `RLevCiphertext`) across the crate
    //! boundary — catching any `pub(crate)` that should have been `pub`.
    use via_primitives::algebra::zq::modulus::ConstModulus;
    use via_primitives::conversion::{LweToRlweKeyN8, gen_lwe_to_rlwe_key_n8};

    #[test]
    fn cascade_key_type_importable_from_dependent_crate() {
        // `size_of` forces the struct layout — hence every field type — to
        // resolve from via-protocol's vantage point.
        let _ = core::mem::size_of::<LweToRlweKeyN8<ConstModulus<65537>, 2>>();
        // The generator fn must also be nameable here (not necessarily callable
        // without a full SecretKey).
        let _gen = gen_lwe_to_rlwe_key_n8::<ConstModulus<65537>, 2>;
    }
}
