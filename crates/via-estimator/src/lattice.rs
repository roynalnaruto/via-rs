//! Concrete lattice-security estimator: classical **core-SVP primal-uSVP**
//! (Bai–Galbraith secret-scaled embedding).
//!
//! # Model
//!
//! For an LWE/RLWE instance `(n, q, χ_s, χ_e)` we estimate the cost of the
//! primal unique-SVP attack under the *core-SVP* model: BKZ-β is charged a
//! single SVP call at `0.292·β` classical bits (sieving). The attack succeeds
//! once the block size β is large enough that the unique embedded short vector
//! is recovered, per the standard "2016 estimate" condition
//!
//! ```text
//!   sqrt(β) · σ_e  ≤  δ(β)^(2β − d − 1) · det^(1/d)
//! ```
//!
//! with the Bai–Galbraith rescaling that maps the secret coordinates (width
//! `σ_s`) onto the error width `σ_e` by scaling them by `ν = σ_e/σ_s`; this
//! adds `ν^n` to the lattice volume, so `det = q^m · ν^n` over `d = n + m + 1`
//! dimensions. We minimise β over the number of samples `m`.
//!
//! # Scope / honesty
//!
//! This is a single-attack, conservative estimate — **not** a replacement for
//! the full [malb/lattice-estimator](https://github.com/malb/lattice-estimator)
//! attack suite. For the small/ternary secrets and moduli VIA uses, primal-uSVP
//! is the representative binding attack (dual/hybrid are not tighter), but the
//! absolute number carries a few-bit model uncertainty. The
//! [`calibration`](crate) tests pin it against the Homomorphic Encryption
//! Security Standard anchors so the reader can see its accuracy.

use core::f64::consts::{E, PI};

/// A secret/error distribution, summarised by the second moment that drives
/// lattice hardness.
#[derive(Clone, Copy, Debug)]
pub enum Dist {
    /// Uniform ternary `{-1,0,1}`: `E[s^2] = 2/3`, so `σ = sqrt(2/3)`.
    Ternary,
    /// Centered discrete Gaussian of standard deviation `σ`.
    Gaussian(f64),
}

impl Dist {
    /// Standard deviation (square root of the second moment) of the
    /// distribution.
    pub fn sigma(self) -> f64 {
        match self {
            Dist::Ternary => (2.0_f64 / 3.0).sqrt(),
            Dist::Gaussian(s) => s,
        }
    }
}

/// Root-Hermite factor `δ` delivered by BKZ with block size `β`
/// (the standard self-dual asymptotic, valid for `β ≳ 50`).
fn delta_bkz(beta: f64) -> f64 {
    let inner = (PI * beta).powf(1.0 / beta) * beta / (2.0 * PI * E);
    inner.powf(1.0 / (2.0 * (beta - 1.0)))
}

/// Classical core-SVP cost of one BKZ-β SVP oracle call, in bits (sieving).
fn core_svp_bits(beta: f64) -> f64 {
    0.292 * beta
}

/// The "2016 estimate" feasibility margin (in nats) of recovering the unique
/// short vector with block size `β` using `m` samples. Non-negative ⇒ the
/// attack succeeds.
fn usvp_margin(beta: f64, m: f64, n: f64, ln_q: f64, ln_nu: f64, sigma_e: f64) -> f64 {
    let d = n + m + 1.0;
    let ln_det = m * ln_q + n * ln_nu; // det = q^m · ν^n
    let lhs = 0.5 * beta.ln() + sigma_e.ln(); // ln( sqrt(β) · σ_e )
    let rhs = (2.0 * beta - d - 1.0) * delta_bkz(beta).ln() + ln_det / d;
    rhs - lhs
}

/// Estimate the classical security (bits) of an LWE/RLWE instance via
/// core-SVP primal-uSVP. `log2_q` is `log2` of the ciphertext modulus.
pub fn primal_usvp_bits(n: usize, log2_q: f64, secret: Dist, error: Dist) -> f64 {
    let n = n as f64;
    let ln_q = log2_q * core::f64::consts::LN_2;
    let sigma_s = secret.sigma();
    let sigma_e = error.sigma();
    let ln_nu = (sigma_e / sigma_s).ln();

    // Smallest β (over any sample count m) for which the attack succeeds.
    // m is swept on a grid up to ~2.5·n; β from 50 upward.
    let m_max = (2.5 * n).ceil() as usize + 64;
    let m_step = ((n / 64.0).floor() as usize).max(1);

    let mut beta = 50.0_f64;
    while beta <= 2000.0 {
        let mut feasible = false;
        let mut m = 1usize;
        while m <= m_max {
            if usvp_margin(beta, m as f64, n, ln_q, ln_nu, sigma_e) >= 0.0 {
                feasible = true;
                break;
            }
            m += m_step;
        }
        if feasible {
            return core_svp_bits(beta);
        }
        beta += 1.0;
    }
    core_svp_bits(2000.0) // saturate (astronomically hard)
}
