//! In-repo security + correctness estimator for the VIA presets.
//!
//! Two reproducible calculators, run on the **shipped** `via_protocol` presets:
//!
//! - [`lattice`] — classical core-SVP primal-uSVP lattice-security estimate per
//!   exposed RLWE instance (large ring, small ring).
//! - [`noise`] — the Appendix-C error-variance recursion → decryption-failure
//!   probability.
//!
//! This replaces the (uncommitted, unreproducible) external-analyst tooling the
//! parameter docs previously referenced: it is the audit trail for the
//! `security_param` and `2^-40` claims. Run `cargo run -p via-estimator` for the
//! full table; the `#[test]`s gate CI.

pub mod lattice;
pub mod noise;

use lattice::Dist;
use noise::{Gadget, NoiseConfig, NoiseResult};
use via_protocol::PIRParams;

/// Standard-deviation of the encryption error used for the security estimate.
/// The Homomorphic Encryption Security Standard / lattice-estimator default;
/// VIA's design error is larger (σ_{1,E}=1024 on the big ring), so this is the
/// conservative choice — security is driven by the secret distribution and the
/// `n / log q` ratio here.
pub const STD_ERROR_SIGMA: f64 = 3.19;

/// Additive calibration (bits) that aligns the bare core-SVP `0.292·β` cost
/// with the **Homomorphic Encryption Security Standard** 128-bit boundaries
/// (`n=1024→log q≈27`, `2048→54`, `4096→109`, ternary secret). Fit as the mean
/// gap to 128 across those three anchors; the residual spread is ≈±3 bits, so
/// treat all reported figures as **±5 bits**. The `calibration_*` tests pin
/// this. Bare sieving cost runs ~29 bits below the standard's tabulated edge,
/// which is why the offset is needed for the numbers to mean what "128-bit"
/// conventionally means.
pub const CALIBRATION_BITS: f64 = 29.6;

/// `log2` of a (possibly `u128`) modulus.
pub fn log2_u128(q: u128) -> f64 {
    let bits = 128 - q.leading_zeros();
    if bits <= 53 {
        (q as f64).log2()
    } else {
        let shift = bits - 52;
        let hi = (q >> shift) as f64;
        hi.log2() + shift as f64
    }
}

// ---------------------------------------------------------------------------
// Security
// ---------------------------------------------------------------------------

/// One exposed lattice instance of a preset.
#[derive(Clone, Copy, Debug)]
pub struct Instance {
    /// Human label.
    pub name: &'static str,
    /// Ring dimension.
    pub n: usize,
    /// `log2` of the instance modulus.
    pub log2_q: f64,
}

/// The two RLWE instances a VIA-C preset exposes: the large ring at `q1` and
/// the ring-switch-key small ring at `q3`.
pub fn instances(p: &PIRParams) -> [Instance; 2] {
    [
        Instance {
            name: "large ring (q1)",
            n: p.n1,
            log2_q: log2_u128(p.q1),
        },
        Instance {
            name: "small ring (q3)",
            n: p.n2,
            log2_q: (p.q3 as f64).log2(),
        },
    ]
}

/// Security (bits) of an instance under a chosen secret distribution and the
/// standard error, on the HE-standard-calibrated scale (so 128 means 128).
pub fn security_bits(inst: Instance, secret: Dist) -> f64 {
    lattice::primal_usvp_bits(inst.n, inst.log2_q, secret, Dist::Gaussian(STD_ERROR_SIGMA))
        + CALIBRATION_BITS
}

/// Minimum security (bits) over both exposed instances — the scheme's level.
pub fn min_security_bits(p: &PIRParams, secret: Dist) -> f64 {
    instances(p)
        .iter()
        .map(|&i| security_bits(i, secret))
        .fold(f64::INFINITY, f64::min)
}

// ---------------------------------------------------------------------------
// Correctness (noise budget)
// ---------------------------------------------------------------------------

/// A full gadget set for one correctness scenario. The conversion-key gadget is
/// not a runtime `PIRParams` field, so all four are specified here.
#[derive(Clone, Copy, Debug)]
pub struct GadgetSet {
    /// LWE→RLWE conversion key gadget.
    pub conv: Gadget,
    /// DMux control RGSW gadget (= the query / `L_QUERY` gadget).
    pub dmux: Gadget,
    /// CMux / CRot selection RGSW gadget.
    pub cmux: Gadget,
    /// Ring-switch key gadget.
    pub rs: Gadget,
}

/// Published-paper **draft** Table 6 gadgets (also what `REALISTIC_PARAMS`
/// ships): DMux 55879, CMux 81, ring-switch 8, conversion 18.
pub const DRAFT_GADGETS: GadgetSet = GadgetSet {
    conv: Gadget {
        base: 18.0,
        len: 18.0,
    },
    dmux: Gadget {
        base: 55879.0,
        len: 2.0,
    },
    cmux: Gadget {
        base: 81.0,
        len: 2.0,
    },
    rs: Gadget {
        base: 8.0,
        len: 8.0,
    },
};

/// Authors' **May-11 erratum** gadgets (also what `SECURE_PARAMS` ships):
/// DMux 18073, CMux 307, ring-switch 4, conversion 11.
pub const ERRATUM_GADGETS: GadgetSet = GadgetSet {
    conv: Gadget {
        base: 11.0,
        len: 18.0,
    },
    dmux: Gadget {
        base: 18073.0,
        len: 2.0,
    },
    cmux: Gadget {
        base: 307.0,
        len: 2.0,
    },
    rs: Gadget {
        base: 4.0,
        len: 8.0,
    },
};

/// Run the Appendix-C recursion for a preset's dims/moduli with an explicit
/// gadget set and key model (`paper_gauss=false` ⇒ ternary keys+error, as the
/// code runs). `yue` applies Yue's DMux/CMux residual correction. `i`,`j` are
/// the database dimensions (32 GiB: `2^11`, `2^14`).
pub fn noise_for(
    p: &PIRParams,
    g: GadgetSet,
    paper_gauss: bool,
    yue: bool,
    i: f64,
    j: f64,
) -> NoiseResult {
    let (theta_1s, theta_1e, theta_2s) = if paper_gauss {
        (32.0 * 32.0, 1024.0 * 1024.0, 26.0 * 26.0)
    } else {
        (2.0 / 3.0, 2.0 / 3.0, 2.0 / 3.0)
    };
    noise::run(NoiseConfig {
        yue_residual_correction: yue,
        n1: p.n1 as f64,
        n2: p.n2 as f64,
        n3: p.n2 as f64, // unbatched VIA-C: n3 = n2
        p: p.p as f64,
        i,
        j,
        q1: 2.0_f64.powf(log2_u128(p.q1)),
        q2: p.q2 as f64,
        q3: p.q3 as f64,
        q4: p.q4 as f64,
        theta_1s,
        theta_1e,
        theta_2s,
        conv: g.conv,
        dmux: g.dmux,
        cmux: g.cmux,
        rs: g.rs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use via_protocol::{REALISTIC_PARAMS, SECURE_PARAMS};

    const I_32GIB: f64 = 2048.0; // 2^11
    const J_32GIB: f64 = 16384.0; // 2^14

    /// The calibrated estimator must land on ~128 at the HE-standard anchors.
    #[test]
    fn calibration_he_anchors() {
        for (n, logq) in [(1024usize, 27.0), (2048, 54.0), (4096, 109.0)] {
            let b =
                lattice::primal_usvp_bits(n, logq, Dist::Ternary, Dist::Gaussian(STD_ERROR_SIGMA))
                    + CALIBRATION_BITS;
            assert!(
                (122.0..=134.0).contains(&b),
                "anchor n={n} logq={logq} -> {b:.1} bits, expected ~128"
            );
        }
    }

    /// Published VIA-C asserts 128 but, run on the ternary keys it actually
    /// uses, falls far short — the A2 gap. (External malb run: ~87; ours is
    /// conservative.)
    #[test]
    fn realistic_security_gap() {
        let min = min_security_bits(&REALISTIC_PARAMS, Dist::Ternary);
        assert!(
            min < REALISTIC_PARAMS.security_param as f64,
            "should miss its asserted 128"
        );
        assert!(
            (65.0..=95.0).contains(&min),
            "ternary min ~77, got {min:.1}"
        );
    }

    /// The SECURE preset genuinely clears its asserted ≥120 bits on BOTH rings.
    #[test]
    fn secure_meets_120_bits() {
        let min = min_security_bits(&SECURE_PARAMS, Dist::Ternary);
        assert!(
            min >= SECURE_PARAMS.security_param as f64,
            "secure min {min:.1} < 120"
        );
    }

    /// Validation anchor: the erratum gadgets at the paper's dimensions
    /// reproduce the authors' stated log2 P_fail ≈ −43.4 (record-union, q3
    /// threshold). This pins the whole noise model.
    #[test]
    fn reproduces_authors_erratum_minus_43() {
        let r = noise_for(
            &REALISTIC_PARAMS,
            ERRATUM_GADGETS,
            true,
            false,
            I_32GIB,
            J_32GIB,
        );
        let u = r.log2_pfail_q3_union(REALISTIC_PARAMS.n2 as f64);
        assert!((-45.0..=-42.0).contains(&u), "expected ~ -43.4, got {u:.1}");
    }

    /// The DRAFT gadgets the code actually ships do NOT meet 2^-40 (A1).
    #[test]
    fn draft_gadgets_fail_correctness() {
        let r = noise_for(
            &REALISTIC_PARAMS,
            DRAFT_GADGETS,
            true,
            false,
            I_32GIB,
            J_32GIB,
        );
        assert!(
            r.log2_pfail_q3_union(REALISTIC_PARAMS.n2 as f64) > -40.0,
            "draft should fail"
        );
    }

    /// FINDING: SECURE_PARAMS as shipped (DMux ℓ=2) does NOT meet 2^-40 at the
    /// operative q4 decode under the record-union convention.
    #[test]
    fn secure_shipped_misses_correctness() {
        let r = noise_for(
            &SECURE_PARAMS,
            ERRATUM_GADGETS,
            false,
            false,
            I_32GIB,
            J_32GIB,
        );
        assert!(
            r.log2_pfail_q4_union(SECURE_PARAMS.n2 as f64) > -40.0,
            "shipped SECURE unexpectedly clears 2^-40"
        );
    }

    /// FINDING: the fix is DMux (L_QUERY) ℓ 2→3 — NOT the doc's CMux ℓ→3.
    #[test]
    fn dmux_len3_fixes_secure_cmux_does_not() {
        let cmux3 = GadgetSet {
            cmux: Gadget {
                base: 307.0,
                len: 3.0,
            },
            ..ERRATUM_GADGETS
        };
        let dmux3 = GadgetSet {
            dmux: Gadget {
                base: 18073.0,
                len: 3.0,
            },
            ..ERRATUM_GADGETS
        };
        let n2 = SECURE_PARAMS.n2 as f64;
        let with_cmux3 = noise_for(&SECURE_PARAMS, cmux3, false, false, I_32GIB, J_32GIB)
            .log2_pfail_q4_union(n2);
        let with_dmux3 = noise_for(&SECURE_PARAMS, dmux3, false, false, I_32GIB, J_32GIB)
            .log2_pfail_q4_union(n2);
        assert!(
            with_cmux3 > -40.0,
            "CMux ℓ→3 should NOT fix it ({with_cmux3:.1})"
        );
        assert!(
            with_dmux3 < -40.0,
            "DMux ℓ→3 SHOULD fix it ({with_dmux3:.1})"
        );
    }

    /// Yue's bug: with the DMux/CMux residual correction and a large-variance
    /// Gaussian secret, the paper's own erratum parameters no longer clear
    /// 2^-40 — corroborating Yue's "the bound doesn't hold for large-variance
    /// keys with approximate decomposition".
    #[test]
    fn yue_correction_breaks_paper_gaussian() {
        let off = noise_for(
            &REALISTIC_PARAMS,
            ERRATUM_GADGETS,
            true,
            false,
            I_32GIB,
            J_32GIB,
        )
        .log2_pfail_q3_union(REALISTIC_PARAMS.n2 as f64);
        let on = noise_for(
            &REALISTIC_PARAMS,
            ERRATUM_GADGETS,
            true,
            true,
            I_32GIB,
            J_32GIB,
        )
        .log2_pfail_q3_union(REALISTIC_PARAMS.n2 as f64);
        assert!(
            off < -40.0,
            "paper-lemma should clear (validates at {off:.1})"
        );
        assert!(
            on > -40.0,
            "Yue-corrected should FAIL for Gaussian keys ({on:.1})"
        );
    }

    /// Yue's correction FLIPS the ternary SECURE verdict: under the paper's
    /// binary-flavoured `×1` residual it reads as failing (≈−29), but with the
    /// physically-correct `×E[s²]=2/3` factor the residual shrinks and it CLEARS
    /// 2^-40 (≈−47). The residual is the dominant term, hence the large swing —
    /// so the budget is real but model-sensitive.
    #[test]
    fn yue_correction_flips_ternary_secure_to_clearing() {
        let paper = noise_for(
            &SECURE_PARAMS,
            ERRATUM_GADGETS,
            false,
            false,
            I_32GIB,
            J_32GIB,
        )
        .log2_pfail_q4_union(SECURE_PARAMS.n2 as f64);
        let correct = noise_for(
            &SECURE_PARAMS,
            ERRATUM_GADGETS,
            false,
            true,
            I_32GIB,
            J_32GIB,
        )
        .log2_pfail_q4_union(SECURE_PARAMS.n2 as f64);
        assert!(
            paper > -40.0,
            "paper ×1 bound reads as failing ({paper:.1})"
        );
        assert!(
            correct < -40.0,
            "physically-correct ×E[s²] bound clears ({correct:.1})"
        );
    }
}
