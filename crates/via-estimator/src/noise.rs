//! Appendix-C noise-budget / correctness calculator for VIA-C.
//!
//! Reimplements the per-stage error-variance recursion of the VIA paper
//! (Appendix C.2, Lemmas C.1–C.5, pp.24–27) and the decryption-failure tail
//! bound, so the `2^-40` correctness claim is *reproducible in-repo* rather than
//! asserted. All `θ` are **variances** (the paper's convention: `θ_S = E[s^2]`,
//! `θ_E = σ_E^2`).
//!
//! ## Modelling choices (made explicit because the paper is ambiguous here)
//!
//! - **`θ_crot` is taken un-squared.** The paper prints `θ_crot²` (p.26) but
//!   consumes `θ_crot` linearly one line later; the squared LHS is a typo (the
//!   only self-consistent reading of the variance recursion).
//! - **Conversion-key gadget `(B,ℓ)` is a scenario input**, not a runtime
//!   `PIRParams` field — the shipped preset carries DMux/CMux/ring-switch bases
//!   but *not* the conversion-key base (itself worth noting). Table 6 draft uses
//!   `(18,18)`, the May-11 erratum uses `(11,18)`.
//! - **Two decode thresholds are reported.** The paper proves against
//!   `⌊(q3−q4)/p⌉/2−1` (variance `θ_ans` at `q3`); Yue Chen (and SpiralPIR
//!   Thm 2.11 / SimplePIR Thm C.1) argue the operative point is the *final*
//!   modulus `q4`, threshold `Δ/2 = ⌊q4/2p⌋` (variance rescaled `q3→q4`, adding
//!   the small-ring rounding `(1+n2·θ_{2,S})/12`). We compute both.
//! - **Per-coefficient vs union bound.** The paper writes a single `erfc(·)`;
//!   we report the per-coefficient tail and the `×n1` union as a band.

use core::f64::consts::{LN_2, PI};

/// One gadget decomposition `(base, length)`. The external product uses two
/// identical sub-bases (the RLWE `a`/`b` components), matching the paper's
/// `(B,B),(ℓ,ℓ)` Table-6 entries.
#[derive(Clone, Copy, Debug)]
pub struct Gadget {
    /// Decomposition base `B`.
    pub base: f64,
    /// Number of digits `ℓ`.
    pub len: f64,
}

impl Gadget {
    /// `B^(2ℓ)` — the denominator of the approximate-decomposition residual.
    fn b_pow_2l(self) -> f64 {
        self.base.powf(2.0 * self.len)
    }
}

/// Everything the recursion needs for one VIA-C instance. Moduli are `f64`
/// (the `q1 ≈ 2^75` product is represented exactly enough for variance ratios).
#[derive(Clone, Copy, Debug)]
pub struct NoiseConfig {
    /// Large ring degree.
    pub n1: f64,
    /// Small ring degree.
    pub n2: f64,
    /// Record ring degree (`n3 = n2` for unbatched VIA-C).
    pub n3: f64,
    /// Plaintext modulus.
    pub p: f64,
    /// First-dimension rows `I`.
    pub i: f64,
    /// CMux-tree columns `J`.
    pub j: f64,
    /// Modulus chain.
    pub q1: f64,
    /// Modulus chain.
    pub q2: f64,
    /// Modulus chain.
    pub q3: f64,
    /// Modulus chain.
    pub q4: f64,
    /// Big-ring secret-key second moment `θ_{1,S}`.
    pub theta_1s: f64,
    /// Big-ring error variance `θ_{1,E}`.
    pub theta_1e: f64,
    /// Small-ring secret-key second moment `θ_{2,S}`.
    pub theta_2s: f64,
    /// Conversion-key gadget (scenario input — not a runtime field).
    pub conv: Gadget,
    /// DMux control RGSW gadget (`gadget_base_1`/`depth_1`).
    pub dmux: Gadget,
    /// CMux/CRot selection RGSW gadget (`gadget_base_2`/`depth_2`).
    pub cmux: Gadget,
    /// Ring-switch key gadget (`gadget_base_rsk`/`depth_rsk`).
    pub rs: Gadget,
    /// Apply Yue's correction: multiply the **DMux/CMux** approximate-decomposition
    /// residual by the secret second moment `θ_{1,S}`. The paper's Lemmas C.1/C.2
    /// omit this factor (valid only for a small/binary secret), yet its *own*
    /// conversion (C.3) and ring-switch ([45]) lemmas include it — an internal
    /// inconsistency Yue flagged. With a large-variance Gaussian secret this term
    /// dominates; with ternary keys it is negligible. `false` reproduces the
    /// paper (and the authors' −43.4); `true` is the corrected bound.
    pub yue_residual_correction: bool,
}

/// The result of running the recursion: the answer-ciphertext error variance
/// and the two decode-threshold failure exponents.
#[derive(Clone, Copy, Debug)]
pub struct NoiseResult {
    /// Answer error variance at `q3` (paper threshold point).
    pub theta_ans_q3: f64,
    /// Answer error variance rescaled to `q4` (Yue / code threshold point).
    pub theta_ans_q4: f64,
    /// Paper threshold `⌊(q3−q4)/p⌉/2 − 1`.
    pub thr_q3: f64,
    /// Final-modulus threshold `Δ/2 = ⌊q4/2p⌋`.
    pub thr_q4: f64,
    /// `log2 P_fail` at the `q3` threshold (per coefficient).
    pub log2_pfail_q3: f64,
    /// `log2 P_fail` at the `q4` threshold (per coefficient).
    pub log2_pfail_q4: f64,
}

impl NoiseResult {
    /// `log2 P_fail` over the whole decoded record: a union bound over the `n2`
    /// message coefficients that must all decode. This is the convention that
    /// reproduces the authors' erratum figure (−52.4 per-coef + log2(512) =
    /// −43.4), so it is the headline number.
    pub fn log2_pfail_q4_union(&self, n2: f64) -> f64 {
        self.log2_pfail_q4 + n2.log2()
    }

    /// As [`Self::log2_pfail_q4_union`] but at the paper's `q3` threshold.
    pub fn log2_pfail_q3_union(&self, n2: f64) -> f64 {
        self.log2_pfail_q3 + n2.log2()
    }
}

fn log2(x: f64) -> f64 {
    x.log2()
}

/// `log2(erfc(x))` for `x ≥ 0`, accurate into the deep tail via the asymptotic
/// expansion (our regime is `x ≳ 4`).
pub fn log2_erfc(x: f64) -> f64 {
    if x < 2.0 {
        // erfc = 1 − erf(x); Abramowitz–Stegun 7.1.26.
        let t = 1.0 / (1.0 + 0.327_591_1 * x);
        let poly = t
            * (0.254_829_592
                + t * (-0.284_496_736
                    + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
        let erf = 1.0 - poly * (-x * x).exp();
        (1.0 - erf).ln() / LN_2
    } else {
        // erfc(x) = exp(−x²)/(x√π) · (1 − 1/2x² + 3/4x⁴ − 15/8x⁶ + …)
        let x2 = x * x;
        let series = 1.0 - 1.0 / (2.0 * x2) + 3.0 / (4.0 * x2 * x2) - 15.0 / (8.0 * x2 * x2 * x2);
        (-x2 - (x * PI.sqrt()).ln() + series.ln()) / LN_2
    }
}

/// Run the full Appendix-C VIA-C recursion for `cfg`.
pub fn run(cfg: NoiseConfig) -> NoiseResult {
    let NoiseConfig {
        n1,
        n2,
        n3,
        p,
        i,
        j,
        q1,
        q2,
        q3,
        q4,
        theta_1s,
        theta_1e,
        theta_2s,
        conv,
        dmux,
        cmux,
        rs,
        yue_residual_correction,
    } = cfg;

    // Yue's correction: the DMux/CMux approximate-decomposition residual scales
    // with the secret second moment E[s²]=θ_{1,S} (as the paper's own C.3/[45]
    // residuals do). 1.0 reproduces the paper's C.1/C.2 (binary-secret) bound.
    let resfac = if yue_residual_correction {
        theta_1s
    } else {
        1.0
    };

    // --- Leaf: conversion key-switch + control-bit RGSW (q1) ----------------
    let theta_ks_conv = n1 * conv.len * conv.base * conv.base * theta_1e / 12.0
        + theta_1s * q1 * q1 / (12.0 * conv.b_pow_2l());
    let theta_ctrl = conv.len * conv.base * conv.base * theta_1e / 12.0
        + q1 * q1 / (12.0 * conv.b_pow_2l())
        + n1 * theta_1s * (theta_1e + 2.0 * theta_ks_conv * log2(n1));

    // --- Leaf gate variances (two equal sub-bases ⇒ factor 2 on digit term,
    //     (n1+1) on residual; residual carries θ_{1,S} under Yue's correction) -
    let theta_dmux = n1 * 2.0 * dmux.len * dmux.base * dmux.base * theta_ctrl / 12.0
        + resfac * (n1 + 1.0) * q1 * q1 / (12.0 * dmux.b_pow_2l());

    // Selection/rotation RGSW are the mod-switched control bits (q1→q2).
    let theta_sel = theta_ctrl * q2 * q2 / (q1 * q1) + (1.0 + n1 * theta_1s) / 12.0;
    let theta_cmux = n1 * 2.0 * cmux.len * cmux.base * cmux.base * theta_sel / 12.0
        + resfac * (n1 + 1.0) * q2 * q2 / (12.0 * cmux.b_pow_2l());
    let theta_crot_leaf = theta_cmux; // CRot reuses the selection gadget at q2

    // --- The 8-step recursion (VIA-C: no repack) ----------------------------
    let theta_dmux_stage = theta_ctrl + theta_dmux * log2(i);
    let theta_ms = theta_dmux_stage * q2 * q2 / (q1 * q1) + (1.0 + n1 * theta_1s) / 12.0;
    let theta_first = i * n1 * theta_ms * p * p / 4.0;
    let theta_cmux_stage = theta_first + theta_cmux * log2(j);
    let theta_crot = theta_cmux_stage + theta_crot_leaf * log2(n1 / n3);

    // Step 8 — response compression: ring-switch (q2→q3) + key-switch.
    let theta_ans_q3 = theta_crot * q3 * q3 / (q2 * q2)
        + (1.0 + n1 * theta_1s) / 12.0
        + rs.len * rs.base * rs.base * theta_1e / 12.0
        + theta_1s * q3 * q3 / (12.0 * rs.b_pow_2l());

    // Final mod-switch q3→q4 (small ring) for the Yue/code decode point.
    let theta_ans_q4 = theta_ans_q3 * q4 * q4 / (q3 * q3) + (1.0 + n2 * theta_2s) / 12.0;

    let thr_q3 = ((q3 - q4) / p).round() / 2.0 - 1.0;
    let thr_q4 = (q4 / (2.0 * p)).floor();

    let log2_pfail_q3 = log2_erfc(thr_q3 / (2.0 * theta_ans_q3).sqrt());
    let log2_pfail_q4 = log2_erfc(thr_q4 / (2.0 * theta_ans_q4).sqrt());

    NoiseResult {
        theta_ans_q3,
        theta_ans_q4,
        thr_q3,
        thr_q4,
        log2_pfail_q3,
        log2_pfail_q4,
    }
}
