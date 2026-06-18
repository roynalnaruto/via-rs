//! Driver: prints the full parameter-audit table for the shipped VIA presets.
//! `cargo run -p via-estimator`.

use via_estimator::lattice::{Dist, primal_usvp_bits};
use via_estimator::noise::Gadget;
use via_estimator::{
    CALIBRATION_BITS, DRAFT_GADGETS, ERRATUM_GADGETS, GadgetSet, instances, noise_for,
    security_bits,
};
use via_protocol::{REALISTIC_PARAMS, SECURE_PARAMS};

const I_32GIB: f64 = 2048.0; // 2^11
const J_32GIB: f64 = 16384.0; // 2^14

fn main() {
    println!("\n=== VIA parameter audit — in-repo estimator (core-SVP primal-uSVP) ===\n");

    println!("[calibration] HE-standard 128-bit anchors (ternary secret), calibrated:");
    for (n, logq) in [(1024usize, 27.0), (2048, 54.0), (4096, 109.0)] {
        let raw = primal_usvp_bits(n, logq, Dist::Ternary, Dist::Gaussian(3.19));
        println!(
            "  n={n:5}  log2 q={logq:5.1}  ->  {:6.1} bits  (raw {:.1} + {:.1}; target ~128)",
            raw + CALIBRATION_BITS,
            raw,
            CALIBRATION_BITS
        );
    }

    for (label, p) in [
        ("REALISTIC (published VIA-C)", &REALISTIC_PARAMS),
        ("SECURE (>=120)", &SECURE_PARAMS),
    ] {
        println!(
            "\n[security] {label}  (security_param asserts {})",
            p.security_param
        );
        for inst in instances(p) {
            let tern = security_bits(inst, Dist::Ternary);
            let gs = if inst.n == p.n1 { 32.0 } else { 26.0 };
            let gauss = security_bits(inst, Dist::Gaussian(gs));
            println!(
                "  {:<16} n={:5} log2q={:6.2}  ternary(code)={:6.1}  gaussian-paper(σ={:>4})={:6.1}",
                inst.name, inst.n, inst.log2_q, tern, gs as u32, gauss
            );
        }
        let min_t = instances(p)
            .iter()
            .map(|&i| security_bits(i, Dist::Ternary))
            .fold(f64::INFINITY, f64::min);
        println!("  -> min over instances (ternary, as shipped): {min_t:.1} bits");
    }

    println!("\n[correctness] Appendix-C noise budget @ 32 GiB (I=2^11, J=2^14)\n");
    let fix_cmux3 = GadgetSet {
        cmux: Gadget {
            base: 307.0,
            len: 3.0,
        },
        ..ERRATUM_GADGETS
    };
    let fix_dmux3 = GadgetSet {
        dmux: Gadget {
            base: 18073.0,
            len: 3.0,
        },
        ..ERRATUM_GADGETS
    };
    // (label, preset, gadgets, paper_gauss, yue_correction)
    let scenarios = [
        (
            "REALISTIC, DRAFT gadgets (shipped), paper-Gaussian",
            &REALISTIC_PARAMS,
            DRAFT_GADGETS,
            true,
            false,
        ),
        (
            "REALISTIC, ERRATUM gadgets, paper-Gaussian [validate -> -43.4]",
            &REALISTIC_PARAMS,
            ERRATUM_GADGETS,
            true,
            false,
        ),
        (
            "REALISTIC, ERRATUM gadgets, paper-Gaussian + YUE correction [bug exposed]",
            &REALISTIC_PARAMS,
            ERRATUM_GADGETS,
            true,
            true,
        ),
        (
            "SECURE, ERRATUM gadgets (shipped), ternary keys",
            &SECURE_PARAMS,
            ERRATUM_GADGETS,
            false,
            false,
        ),
        (
            "SECURE, ERRATUM gadgets, ternary + YUE correction [~unchanged]",
            &SECURE_PARAMS,
            ERRATUM_GADGETS,
            false,
            true,
        ),
        (
            "SECURE + CMux len 2->3 (doc's proposed fix), ternary",
            &SECURE_PARAMS,
            fix_cmux3,
            false,
            false,
        ),
        (
            "SECURE + DMux/L_QUERY len 2->3 (real fix), ternary",
            &SECURE_PARAMS,
            fix_dmux3,
            false,
            false,
        ),
    ];
    for (label, p, g, paper_gauss, yue) in scenarios {
        let r = noise_for(p, g, paper_gauss, yue, I_32GIB, J_32GIB);
        let n2 = p.n2 as f64;
        println!("  {label}");
        println!(
            "    q3 thr={:>7.0}: per-coef {:>7.1}, record-union {:>7.1}   [paper threshold]",
            r.thr_q3,
            r.log2_pfail_q3,
            r.log2_pfail_q3_union(n2)
        );
        println!(
            "    q4 thr(Δ/2)={:>5.0}: per-coef {:>7.1}, record-union {:>7.1}   [operative decode]\n",
            r.thr_q4,
            r.log2_pfail_q4,
            r.log2_pfail_q4_union(n2)
        );
    }
}
