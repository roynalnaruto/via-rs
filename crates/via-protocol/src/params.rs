//! Runtime parameter sidecar: [`PIRParams`] + [`KeyDist`].
//!
//! Mirrors `.references/via-spec/pir/primitives/params.py`. Const-generic ring
//! dimensions / gadget depths live on [`crate::ViaCPublicParams`]; this struct
//! carries the runtime `u64`/`u128` moduli, gadget bases, key distributions,
//! and sigmas needed for sampling and serialization.

use core::fmt;

// ---------------------------------------------------------------------------
// KeyDist
// ---------------------------------------------------------------------------

/// Key distribution for secret-key sampling.
///
/// Mirrors `pir/primitives/params.py` `key_dist_1`/`key_dist_2` string tags
/// but as a typed enum.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KeyDist {
    /// Ternary: coefficients in $\{-1, 0, 1\}$ with equal probability.
    Ternary,
    /// Bounded uniform: coefficients sampled uniformly in
    /// $[{-}\texttt{bound}, \texttt{bound}]$.
    BoundedUniform,
    /// Discrete Gaussian with standard deviation `sigma` (carried separately
    /// as an `Option<f64>` on [`PIRParams`]).
    Gaussian,
}

impl fmt::Display for KeyDist {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ternary => f.write_str("ternary"),
            Self::BoundedUniform => f.write_str("bounded_uniform"),
            Self::Gaussian => f.write_str("gaussian"),
        }
    }
}

// ---------------------------------------------------------------------------
// PIRParams
// ---------------------------------------------------------------------------

/// Runtime VIA-C scheme parameters.
///
/// All fields are public for direct inspection by client and server; mutation
/// is intentionally prevented by the lack of `&mut self` setters (construct a
/// new `PIRParams` for different parameters).
///
/// # Panics
///
/// [`PIRParams::new`] panics in debug builds when:
/// - `n1` or `n2` is zero or not a power of two.
/// - `n1 <= n2`.
/// - The moduli chain `q1 > q2 > q3 > q4 > p` is violated.
/// - Any gadget depth is zero.
/// - `key_dist_1 == KeyDist::Gaussian` but `key_sigma_1.is_none()`.
/// - `key_dist_2 == KeyDist::Gaussian` but `key_sigma_2.is_none()`.
#[derive(Clone, PartialEq)]
pub struct PIRParams {
    // ── Ring dimensions ──────────────────────────────────────────────────────
    /// Large ring degree $n_1$ (e.g. 64 toy / 2048 realistic).
    pub n1: usize,
    /// Small ring degree $n_2$ (e.g. 16 toy / 512 realistic).
    pub n2: usize,

    // ── Moduli chain: q1 > q2 > q3 > q4 > p ────────────────────────────────
    /// Largest ciphertext modulus $q_1$ (RNS product for realistic params;
    /// a single prime for toy params). Stored as `u128` because the realistic
    /// value `137_438_822_401 × 274_810_798_081 ≈ 2^{75}` overflows `u64`.
    pub q1: u128,
    /// Ciphertext modulus $q_2$ after DMux mod-switch.
    pub q2: u64,
    /// Ciphertext modulus $q_3$ after CRot / ring-switch input.
    pub q3: u64,
    /// Final small modulus $q_4$ (power-of-two for body rescale).
    pub q4: u64,
    /// Plaintext modulus $p$.
    pub p: u64,

    // ── Gadget decomposition ─────────────────────────────────────────────────
    /// Gadget base $B_1$ for the large-ring (DMux ctrl) RGSW external product.
    /// Its depth `gadget_depth_1` also equals the query RGSW gadget length
    /// `L_QUERY` (the query is compressed at `gadget_depth_1` — `client.py:117`).
    pub gadget_base_1: u64,
    /// Gadget depth $\ell_1$ for the large-ring (DMux) RGSW = `L_QUERY`.
    pub gadget_depth_1: usize,
    /// Gadget base $B_2$ for the small-ring (CMux/CRot sel) tree decomposition.
    pub gadget_base_2: u64,
    /// Gadget depth $\ell_2$ for the small-ring (CMux/CRot) tree decomposition.
    /// `≤ gadget_depth_1`: the CMux tree decomposes into the first $\ell_2$ rows
    /// of the `L_QUERY`-row query RGSW (`server.py:196`).
    pub gadget_depth_2: usize,
    /// Gadget base $B_\mathrm{rsk}$ for the ring-switch key.
    pub gadget_base_rsk: u64,
    /// Gadget depth $\ell_\mathrm{rsk}$ for the ring-switch key.
    pub gadget_depth_rsk: usize,

    // ── Key distributions ────────────────────────────────────────────────────
    /// Distribution for $S_1$ (the large-ring key at $q_1$).
    pub key_dist_1: KeyDist,
    /// Distribution for $S_2$ (the small-ring key at $q_3$).
    pub key_dist_2: KeyDist,
    /// Bound for $S_2$ when `key_dist_2 == BoundedUniform`.
    pub key_bound_2: u64,
    /// $\sigma$ for $S_1$ when `key_dist_1 == Gaussian`; `None` otherwise.
    pub key_sigma_1: Option<f64>,
    /// $\sigma$ for $S_2$ when `key_dist_2 == Gaussian`; `None` otherwise.
    pub key_sigma_2: Option<f64>,
    /// Error distribution $\sigma$; `None` means ternary error.
    pub error_sigma: Option<f64>,

    // ── Security ─────────────────────────────────────────────────────────────
    /// Security parameter $\lambda$ in bits. `0` means no security claim
    /// (e.g. toy params).
    pub security_param: u32,
}

impl PIRParams {
    /// Construct `PIRParams`, validating all invariants.
    ///
    /// # Panics
    ///
    /// See struct-level documentation.
    #[allow(clippy::too_many_arguments)]
    pub const fn new(
        n1: usize,
        n2: usize,
        q1: u128,
        q2: u64,
        q3: u64,
        q4: u64,
        p: u64,
        gadget_base_1: u64,
        gadget_depth_1: usize,
        gadget_base_2: u64,
        gadget_depth_2: usize,
        gadget_base_rsk: u64,
        gadget_depth_rsk: usize,
        key_dist_1: KeyDist,
        key_dist_2: KeyDist,
        key_bound_2: u64,
        key_sigma_1: Option<f64>,
        key_sigma_2: Option<f64>,
        error_sigma: Option<f64>,
        security_param: u32,
    ) -> Self {
        // Ring dimension invariants.
        debug_assert!(
            n1 > 0 && n1.is_power_of_two(),
            "n1 must be a positive power of two"
        );
        debug_assert!(
            n2 > 0 && n2.is_power_of_two(),
            "n2 must be a positive power of two"
        );
        debug_assert!(n1 > n2, "n1 must be > n2 for ring switching");
        // Moduli chain.
        debug_assert!(q1 > q2 as u128, "moduli chain: q1 > q2 violated");
        debug_assert!(q2 > q3, "moduli chain: q2 > q3 violated");
        debug_assert!(q3 > q4, "moduli chain: q3 > q4 violated");
        debug_assert!(q4 > p, "moduli chain: q4 > p violated");
        // Gadget depths.
        debug_assert!(gadget_depth_1 > 0, "gadget_depth_1 must be positive");
        debug_assert!(gadget_depth_2 > 0, "gadget_depth_2 must be positive");
        debug_assert!(gadget_depth_rsk > 0, "gadget_depth_rsk must be positive");
        // The CMux tree decomposes into the first `gadget_depth_2` rows of the
        // `gadget_depth_1`-row (= L_QUERY) query RGSW, so depth_2 must not exceed it.
        debug_assert!(
            gadget_depth_2 <= gadget_depth_1,
            "gadget_depth_2 must be <= gadget_depth_1 (= L_QUERY query RGSW rows)"
        );
        // Gaussian sigma presence. `matches!` keeps this `const`-evaluable
        // (derived `PartialEq`/`!=` is not usable in a `const fn`).
        debug_assert!(
            !matches!(key_dist_1, KeyDist::Gaussian) || key_sigma_1.is_some(),
            "key_sigma_1 required when key_dist_1 = Gaussian"
        );
        debug_assert!(
            !matches!(key_dist_2, KeyDist::Gaussian) || key_sigma_2.is_some(),
            "key_sigma_2 required when key_dist_2 = Gaussian"
        );
        Self {
            n1,
            n2,
            q1,
            q2,
            q3,
            q4,
            p,
            gadget_base_1,
            gadget_depth_1,
            gadget_base_2,
            gadget_depth_2,
            gadget_base_rsk,
            gadget_depth_rsk,
            key_dist_1,
            key_dist_2,
            key_bound_2,
            key_sigma_1,
            key_sigma_2,
            error_sigma,
            security_param,
        }
    }

    /// Dimension ratio $d = n_1 / n_2$.
    #[inline]
    pub const fn d(&self) -> usize {
        self.n1 / self.n2
    }

    /// Encoding scale $\Delta = \lceil q_1 / p \rceil$.
    ///
    /// Returned as `u128` to handle the `q1 ≈ 2^{75}` realistic value; this is
    /// the boundary P2's `encrypt_lwe_raw(message: u128)` consumes.
    #[inline]
    pub const fn delta(&self) -> u128 {
        self.q1.div_ceil(self.p as u128)
    }

    /// $\log_2 n_1$ — the number of DMux bits.
    #[inline]
    pub const fn log_n1(&self) -> u32 {
        self.n1.trailing_zeros()
    }

    /// $\log_2 n_2$ — the number of CMux + CRot bits combined.
    #[inline]
    pub const fn log_n2(&self) -> u32 {
        self.n2.trailing_zeros()
    }
}

/// `Debug` omits the sigma fields to avoid accidental leakage in logs.
impl fmt::Debug for PIRParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PIRParams")
            .field("n1", &self.n1)
            .field("n2", &self.n2)
            .field("q1", &self.q1)
            .field("q2", &self.q2)
            .field("q3", &self.q3)
            .field("q4", &self.q4)
            .field("p", &self.p)
            .field("key_dist_1", &self.key_dist_1)
            .field("key_dist_2", &self.key_dist_2)
            .field("security_param", &self.security_param)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::format;

    fn toy() -> PIRParams {
        PIRParams::new(
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
        )
    }

    #[test]
    fn toy_params_construct() {
        let p = toy();
        assert_eq!(p.n1, 64);
        assert_eq!(p.n2, 16);
        assert_eq!(p.d(), 4);
        // gadget_depth_2 (CMux) <= gadget_depth_1 (= L_QUERY).
        assert!(p.gadget_depth_2 <= p.gadget_depth_1);
    }

    #[test]
    fn toy_params_delta() {
        let p = toy();
        // delta = ceil(2^40 / 16) = 2^40 / 16 = 2^36.
        assert_eq!(p.delta(), 1u128 << 36);
    }

    #[test]
    fn toy_params_log_n() {
        let p = toy();
        assert_eq!(p.log_n1(), 6); // 2^6 = 64
        assert_eq!(p.log_n2(), 4); // 2^4 = 16
    }

    #[test]
    fn realistic_params_q1_u128() {
        // The realistic q1 overflows u64; it must fit in u128.
        let q1: u128 = 137_438_822_401u128 * 274_810_798_081u128;
        let p = PIRParams::new(
            2048,
            512,
            q1,
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
        );
        assert_eq!(p.n1, 2048);
        assert_eq!(p.d(), 4);
        assert_eq!(p.security_param, 128);
    }

    #[test]
    fn key_dist_display() {
        assert_eq!(format!("{}", KeyDist::Ternary), "ternary");
        assert_eq!(format!("{}", KeyDist::Gaussian), "gaussian");
    }

    #[test]
    #[cfg(debug_assertions)]
    #[should_panic]
    fn new_panics_on_n1_not_power_of_two() {
        // n1 = 3 is not a power of two.
        PIRParams::new(
            3,
            2,
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
        );
    }
}
