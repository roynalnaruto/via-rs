//! Const-generic VIA-C parameter markers + the `PIRParams` preset constants.
//!
//! [`ViaCPublicParams`] is a zero-sized marker carrying the compile-time
//! dimensions (`N1, N2, L_QUERY, L_CK, L_RSK, D`) that monomorphise the wire
//! types; [`TOY_PARAMS`] / [`REALISTIC_PARAMS`] are the matching runtime
//! [`PIRParams`] sidecars. A `_CHECK` const block ties them together.
//!
//! # The single `L_QUERY` gadget depth
//!
//! The query is compressed at `gadget_depth_1` LWE levels per bit,
//! so **every** DMux/CMux/CRot RGSW in a
//! `DecompressedQuery` has the *same* gadget length — that one value is
//! `L_QUERY = gadget_depth_1`. The DMux tree consumes all `L_QUERY` rows; the
//! CMux/CRot trees decompose into the first `gadget_depth_2 ≤ L_QUERY` rows.
//! Hence a single `L_QUERY` const is faithful (realistic parameters set
//! DMux = CMux = 2), while `gadget_depth_2` stays a
//! runtime [`PIRParams`] field for the CMux tree.

use crate::params::{KeyDist, PIRParams};

// ---------------------------------------------------------------------------
// ViaCPublicParams — const-generic ZST marker
// ---------------------------------------------------------------------------

/// Zero-sized marker carrying the VIA-C compile-time dimensions.
///
/// Const params: `N1` (large ring), `N2` (small ring), `L_QUERY` (query RGSW
/// gadget length = `gadget_depth_1`), `L_CK` (LWE→RLWE / RLWE→RGSW conversion
/// key depth), `L_RSK` (ring-switch key depth), `D = N1 / N2` (records per cell
/// / CRot slot count).
///
/// Use the concrete aliases [`ViaCToyParams`] / [`ViaCRealisticParams`] at call
/// sites; bespoke parameters use `ViaCPublicParams<N1, N2, L_QUERY, L_CK, L_RSK, D>`
/// directly. The matching runtime sidecars are [`TOY_PARAMS`] / [`REALISTIC_PARAMS`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViaCPublicParams<
    const N1: usize,
    const N2: usize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
>;

impl<
    const N1: usize,
    const N2: usize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> ViaCPublicParams<N1, N2, L_QUERY, L_CK, L_RSK, D>
{
    /// Compile-time dimension consistency check.
    ///
    /// Force its evaluation (`let () = ViaCToyParams::_CHECK;`) to turn a
    /// mis-typed preset alias into a compile error.
    ///
    /// # Panics (compile-time)
    ///
    /// - `N2 < 2` or `N2` not a power of two
    /// - `N1` not a power of two, or `N1 <= N2`
    /// - `N1 != N2 * D`
    /// - any of `L_QUERY` / `L_CK` / `L_RSK` is zero
    pub const _CHECK: () = {
        assert!(N2 >= 2, "ViaCPublicParams: N2 must be >= 2");
        assert!(
            N2.is_power_of_two(),
            "ViaCPublicParams: N2 must be a power of two"
        );
        assert!(
            N1.is_power_of_two(),
            "ViaCPublicParams: N1 must be a power of two"
        );
        assert!(N1 > N2, "ViaCPublicParams: N1 must be > N2");
        assert!(N1 == N2 * D, "ViaCPublicParams: N1 must equal N2 * D");
        assert!(L_QUERY > 0, "ViaCPublicParams: L_QUERY must be positive");
        assert!(L_CK > 0, "ViaCPublicParams: L_CK must be positive");
        assert!(L_RSK > 0, "ViaCPublicParams: L_RSK must be positive");
    };
}

// ---------------------------------------------------------------------------
// Concrete preset aliases
// ---------------------------------------------------------------------------

/// Toy VIA-C preset: `n1=64, n2=16, L_QUERY=20, L_CK=40, L_RSK=8, D=4`.
///
/// `gadget_depth_1=20` → `L_QUERY`, conversion-key depth 40. Security
/// parameter is 0 — no security claim. Runtime sidecar: [`TOY_PARAMS`].
pub type ViaCToyParams = ViaCPublicParams<64, 16, 20, 40, 8, 4>;

/// Realistic VIA-C preset: `n1=2048, n2=512, L_QUERY=2, L_CK=18, L_RSK=8, D=4`.
///
/// DMux/CMux length 2, Conversion Key 18, Ring-Switching Key 8. Runtime
/// sidecar: [`REALISTIC_PARAMS`].
pub type ViaCRealisticParams = ViaCPublicParams<2048, 512, 2, 18, 8, 4>;

// ---------------------------------------------------------------------------
// ViaBPublicParams — const-generic ZST marker (VIA-B)
// ---------------------------------------------------------------------------

/// Zero-sized marker carrying the VIA-B compile-time dimensions.
///
/// Extends [`ViaCPublicParams`] with the record-ring degree `N3` (`N3 | N2 | N1`)
/// and the batch size `T`. VIA-C is the degenerate case `N3 = N2`, `T = 1`.
/// `D = N1 / N2` is the ring-switch fold; records-per-cell / CRot range is
/// `N1 / N3`.
///
/// Use the aliases [`ViaBToyParams`] / [`ViaBRealisticParams`]; the runtime
/// sidecars are [`TOY_B_PARAMS`] / [`REALISTIC_B_PARAMS`].
#[cfg(feature = "via-b")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ViaBPublicParams<
    const N1: usize,
    const N2: usize,
    const N3: usize,
    const T: usize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
>;

#[cfg(feature = "via-b")]
impl<
    const N1: usize,
    const N2: usize,
    const N3: usize,
    const T: usize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> ViaBPublicParams<N1, N2, N3, T, L_QUERY, L_CK, L_RSK, D>
{
    /// Compile-time dimension consistency check.
    ///
    /// Force its evaluation (`let () = ViaBToyParams::_CHECK;`) to turn a
    /// mis-typed preset alias into a compile error.
    ///
    /// # Panics (compile-time)
    ///
    /// All [`ViaCPublicParams::_CHECK`] conditions, plus: `N3 < 1` or `N3` not a
    /// power of two; `N3 ∤ N2`; `T < 1` or `T` not a power of two;
    /// `T * N3 > N2` (single-repack record-fit).
    pub const _CHECK: () = {
        // Inherit the VIA-C invariants (N1 = N2·D, power-of-two, L_* > 0).
        let () = ViaCPublicParams::<N1, N2, L_QUERY, L_CK, L_RSK, D>::_CHECK;
        assert!(N3 >= 1, "ViaBPublicParams: N3 must be >= 1");
        assert!(
            N3.is_power_of_two(),
            "ViaBPublicParams: N3 must be a power of two"
        );
        // N3, N2 both powers of two ⇒ N3 | N2 ⟺ N3 ≤ N2.
        assert!(N3 <= N2, "ViaBPublicParams: N3 must divide N2");
        assert!(T >= 1, "ViaBPublicParams: T must be >= 1");
        assert!(
            T.is_power_of_two(),
            "ViaBPublicParams: T must be a power of two"
        );
        assert!(
            T * N3 <= N2,
            "ViaBPublicParams: T * N3 must be <= N2 (single-repack record-fit)"
        );
    };
}

/// Toy VIA-B preset: `n1=64, n2=16, n3=2, T=8, L_QUERY=20, L_CK=40, L_RSK=8, D=4`.
///
/// Reuses the VIA-C n64 toy stack; `T·N3 = 16 = N2` is the single-repack
/// boundary. Runtime sidecar: [`TOY_B_PARAMS`].
#[cfg(feature = "via-b")]
pub type ViaBToyParams = ViaBPublicParams<64, 16, 2, 8, 20, 40, 8, 4>;

/// Realistic VIA-B preset: `n1=2048, n2=512, n3=2, T=256, L_QUERY=2, L_CK=18, L_RSK=8, D=4`.
///
/// Database dims and gadget params match VIA-C; `n3 = 2`
/// (1-byte records at `p=16`), `T = 256`. Runtime sidecar:
/// [`REALISTIC_B_PARAMS`].
#[cfg(feature = "via-b")]
pub type ViaBRealisticParams = ViaBPublicParams<2048, 512, 2, 256, 2, 18, 8, 4>;

// ---------------------------------------------------------------------------
// PIRParams preset constants
// ---------------------------------------------------------------------------

/// Toy `PIRParams` sidecar.
///
/// `gadget_depth_1 = 20 = L_QUERY`; `gadget_depth_2 = 16` is the (smaller)
/// CMux/CRot tree depth. Validated at compile time by `PIRParams::new`'s
/// `const`-evaluated `debug_assert`s.
pub const TOY_PARAMS: PIRParams = PIRParams::new(
    64,
    16,
    // q1 = 2^40 (single-prime toy), q2 = 2^32, q3 = 2^16, q4 = 2^12, p = 16
    1u128 << 40,
    1u64 << 32,
    1u64 << 16,
    1u64 << 12,
    16,
    // Large-ring (DMux) gadget: base 4, depth 20 (= L_QUERY)
    4,
    20,
    // Small-ring (CMux/CRot) tree gadget: base 4, depth 16
    4,
    16,
    // Ring-switch gadget: base 4, depth 8
    4,
    8,
    KeyDist::Ternary,
    KeyDist::Ternary,
    1,
    None,
    None,
    None,
    0,
);

/// Realistic `PIRParams` sidecar.
pub const REALISTIC_PARAMS: PIRParams = PIRParams::new(
    2048,
    512,
    // q1 = 137_438_822_401 * 274_810_798_081 (two NTT-friendly primes, ≈ 2^75)
    137_438_822_401u128 * 274_810_798_081u128,
    // q2 ≈ 2^34, q3 ≈ 2^23, q4 = 2^12, p = 16
    17_175_674_881,
    8_380_417,
    4096,
    16,
    // Large-ring (DMux) gadget: base 55879, depth 2 (= L_QUERY)
    55879,
    2,
    // Small-ring (CMux/CRot) tree gadget: base 81, depth 2
    81,
    2,
    // Ring-switch gadget: base 8, depth 8
    8,
    8,
    KeyDist::Gaussian,
    KeyDist::Gaussian,
    26,
    Some(1.0),
    Some(1.0),
    Some(1.0),
    128,
);

/// Toy VIA-B `PIRParams` sidecar: [`TOY_PARAMS`] values + `n3 = 2`, `t = 8`.
#[cfg(feature = "via-b")]
pub const TOY_B_PARAMS: PIRParams = PIRParams::new_b(
    64,
    16,
    1u128 << 40,
    1u64 << 32,
    1u64 << 16,
    1u64 << 12,
    16,
    4,
    20,
    4,
    16,
    4,
    8,
    KeyDist::Ternary,
    KeyDist::Ternary,
    1,
    None,
    None,
    None,
    0,
    // VIA-B: record ring n3 = 2, batch T = 8.
    2,
    8,
);

/// Realistic VIA-B `PIRParams` sidecar: [`REALISTIC_PARAMS`] values + `n3 = 2`,
/// `t = 256` (1-byte records).
#[cfg(feature = "via-b")]
pub const REALISTIC_B_PARAMS: PIRParams = PIRParams::new_b(
    2048,
    512,
    137_438_822_401u128 * 274_810_798_081u128,
    17_175_674_881,
    8_380_417,
    4096,
    16,
    55879,
    2,
    81,
    2,
    8,
    8,
    KeyDist::Gaussian,
    KeyDist::Gaussian,
    26,
    Some(1.0),
    Some(1.0),
    Some(1.0),
    128,
    // VIA-B: record ring n3 = 2, batch T = 256.
    2,
    256,
);

// ---------------------------------------------------------------------------
// Runtime cross-assert helper
// ---------------------------------------------------------------------------

/// Debug-build cross-assert: verify a [`PIRParams`] instance agrees with the
/// const dims of a `ViaCPublicParams<N1, N2, …>` preset.
///
/// Called from `PublicParams::new` (Task 17) and the preset tests.
///
/// # Panics (debug only)
///
/// Panics if any of `n1`/`n2`/`d()` mismatch, if `gadget_depth_1 != L_QUERY`,
/// if `gadget_depth_2 > L_QUERY`, or if `gadget_depth_rsk != L_RSK`.
#[inline]
pub fn pir_params_matches_preset<
    const N1: usize,
    const N2: usize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
>(
    p: &PIRParams,
) {
    let () = ViaCPublicParams::<N1, N2, L_QUERY, L_CK, L_RSK, D>::_CHECK;
    debug_assert_eq!(p.n1, N1, "PIRParams.n1 must equal preset N1");
    debug_assert_eq!(p.n2, N2, "PIRParams.n2 must equal preset N2");
    debug_assert_eq!(p.d(), D, "PIRParams.d() must equal preset D");
    debug_assert_eq!(
        p.gadget_depth_1, L_QUERY,
        "PIRParams.gadget_depth_1 must equal L_QUERY (query RGSW gadget length)"
    );
    debug_assert!(
        p.gadget_depth_2 <= L_QUERY,
        "PIRParams.gadget_depth_2 must be <= L_QUERY"
    );
    debug_assert_eq!(
        p.gadget_depth_rsk, L_RSK,
        "PIRParams.gadget_depth_rsk must equal L_RSK"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Force `_CHECK` evaluation for both presets. A mis-typed alias fails to
    /// compile.
    #[test]
    fn toy_check_evaluates() {
        let () = ViaCToyParams::_CHECK;
    }

    #[test]
    fn realistic_check_evaluates() {
        let () = ViaCRealisticParams::_CHECK;
    }

    #[test]
    fn presets_are_zsts() {
        assert_eq!(core::mem::size_of::<ViaCToyParams>(), 0);
        assert_eq!(core::mem::size_of::<ViaCRealisticParams>(), 0);
    }

    /// The toy const sidecar agrees with its const-generic marker.
    #[test]
    fn toy_params_matches_preset() {
        pir_params_matches_preset::<64, 16, 20, 40, 8, 4>(&TOY_PARAMS);
        assert_eq!(TOY_PARAMS.n1, 64);
        assert_eq!(TOY_PARAMS.n2, 16);
        assert_eq!(TOY_PARAMS.d(), 4);
        assert_eq!(TOY_PARAMS.gadget_depth_1, 20); // = L_QUERY
        assert_eq!(TOY_PARAMS.gadget_depth_2, 16); // CMux tree depth
    }

    /// The realistic const sidecar agrees with its const-generic marker.
    #[test]
    fn realistic_params_matches_preset() {
        pir_params_matches_preset::<2048, 512, 2, 18, 8, 4>(&REALISTIC_PARAMS);
        assert_eq!(REALISTIC_PARAMS.n1, 2048);
        assert_eq!(REALISTIC_PARAMS.n2, 512);
        assert_eq!(REALISTIC_PARAMS.d(), 4);
        // delta = ceil(q1 / 16); q1 ≈ 2^75 so delta is a u128 > u64::MAX.
        assert!(REALISTIC_PARAMS.delta() > u128::from(u64::MAX));
    }

    /// Cross-assert panics in debug if n1 mismatches the preset.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn pir_params_matches_preset_panics_on_n1_mismatch() {
        let mut p = TOY_PARAMS;
        p.n1 = 128; // wrong n1 (no longer 64)
        pir_params_matches_preset::<64, 16, 20, 40, 8, 4>(&p);
    }
}
