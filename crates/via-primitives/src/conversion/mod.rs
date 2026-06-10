//! Layer 5 — MLWE LWE-to-RLWE conversion cascade. See `.docs/primitives.md` §5.
//!
//! Converts an LWE ciphertext (an $(n, 1)$-MLWE) into an RLWE (a $(1, n)$-MLWE)
//! through $\log_2 n$ rank-halving / degree-doubling [`conv::conv_step`]s
//! (§5.2), driven by the [`cascade`] macro (§5.3-§5.4), plus MLWE embedding
//! (§5.1, [`mlwe_ops`]) and RLWE→MLWE extraction (§5.5, [`extract`]).
//!
//! ## File → primitive map
//!
//! | File             | Primitives                                              |
//! |------------------|---------------------------------------------------------|
//! | `kernels/lwe.rs` | LWE body dot product (POD / flat-slice, **CT over key**)|
//! | `mlwe_ops.rs`    | §5.1 `embed_mlwe`, `rlwe_to_mlwe`/`mlwe_to_rlwe`, `encrypt_lwe`/`decrypt_lwe` |
//! | `conv.rs`        | §5.2 `conv_step`, `gen_conv_step_key`                    |
//! | `cascade.rs`     | §5.3/§5.4 `lwe_to_rlwe_cascade!`, `LWEToRLWEKey`, gen    |
//! | `extract.rs`     | §5.5 `extr` (general-$d$; VIA-B `Repack` prerequisite)   |
//!
//! ## The cascade in one picture
//!
//! For $n = 8$, with the MLWE invariant $\mathrm{rank} \cdot \mathrm{degree} =
//! n$ held throughout:
//!
//! ```text
//!  (8, 1) --conv_step--> (4, 2) --conv_step--> (2, 4) --conv_step--> (1, 8)
//!    LWE                                                              RLWE
//! ```
//!
//! Both the rank and the degree change at every step, and both are
//! compile-time const-generics, so the cascade cannot be a runtime loop (as it
//! is in the Python reference `pir/primitives/mlwe.py`). Instead [`conv::conv_step`]
//! is a generic single-step kernel and the [`crate::lwe_to_rlwe_cascade!`] macro emits
//! a monomorphic `lwe_to_rlwe_n<N>` chain — plus the matching
//! heterogeneous-degree `LWEToRLWEKey` struct + generator — per concrete $n$.
//!
//! ## GPU-portability convention
//!
//! Scalar-level arithmetic lives in [`kernels`] as POD-by-value + flat-slice
//! functions (the Layer-0 kernel shape; see [`crate::algebra::zq::ops`]); the
//! orchestrators here do ring-type plumbing and PRG draws.
//! [`conv::conv_step`] is a **map-reduce** — the `RANK_IN` per-mask
//! embed+key-switches are independent (the map), and only the slot/body
//! accumulation is a reduction — so the map lowers to a device launch.
//!
//! ## Layer-0 prerequisites (landed in Part 0)
//!
//! Relies on [`crate::algebra::ring::RingPoly::embed_at`] / `Embedded` (the
//! enlarging dual of `project_at`) and the relaxed $N \ge 1$ backend bound:
//! the LWE-form components live in $R_{1, q} \cong \mathbb{Z}_q$.

pub mod kernels;

// §5.1 — `embed_mlwe`, RLWE↔MLWE conversions, LWE encrypt/decrypt (Part 1).
pub mod mlwe_ops;
// §5.2 — single Conv₂ step `conv_step` + `gen_conv_step_key` (Part 2).
pub mod conv;
// §5.3/§5.4 — `lwe_to_rlwe_cascade!` macro + `LWEToRLWEKey` (Part 3).
pub mod cascade;
// §5.5 — `extr` general-$d$ RLWE→MLWE extraction (Part 4).
pub mod extract;
// §7 — VIA-B homomorphic repacking (Layer 7 Part 1/2). `alloc`-gated: the
// repack recursion holds a runtime `Vec` of MLWE ciphertexts + a
// heterogeneous-degree key schedule. Empty until Part 1 lands.
#[cfg(all(feature = "via-b", feature = "alloc"))]
pub mod repack;

pub use conv::{
    ConvDims, conv_step, gen_conv_step_key, gen_conv_step_key_element,
    gen_conv_step_key_element_into,
};
pub use mlwe_ops::{
    decrypt_lwe, embed_mlwe, encrypt_lwe, encrypt_lwe_raw, mlwe_to_rlwe, rlwe_to_mlwe,
};
// The `lwe_to_rlwe_cascade!` macro is `#[macro_export]`ed at the crate root; the
// toy-parameter instantiations it produces are re-exported here. The degree-64
// instantiation is the VIA-C toy end-to-end query-compression cascade.
pub use cascade::{
    LweToRlweKeyN4, LweToRlweKeyN8, LweToRlweKeyN64, LweToRlweKeyRnsN8, gen_lwe_to_rlwe_key_n4,
    gen_lwe_to_rlwe_key_n8, gen_lwe_to_rlwe_key_n64, gen_lwe_to_rlwe_key_rns_n8, lwe_to_rlwe_n4,
    lwe_to_rlwe_n8, lwe_to_rlwe_n64, lwe_to_rlwe_rns_n8,
};
// Paper-scale n₁ = 2048 cascade (`alloc`-only — its ~24.75 MB key is heap-built).
// The supported constructor is the heap builder `gen_lwe_to_rlwe_key_rns_n2048_boxed`;
// the by-value generator is intentionally NOT re-exported (it would overflow the
// stack). `LweToRlweKeyRnsN2048` is the `K` in `QueryCompressionKey<K>`; the server
// consumes `lwe_to_rlwe_rns_n2048`.
#[cfg(feature = "alloc")]
pub use cascade::{
    LweToRlweKeyRnsN2048, gen_lwe_to_rlwe_key_rns_n2048_boxed, lwe_to_rlwe_rns_n2048,
};
pub use extract::{ExtrDims, extr};
// §7 — VIA-B repacking primitives (Part 1/2). Same gate as the `repack` module.
#[cfg(all(feature = "via-b", feature = "alloc"))]
pub use repack::{embed_d, mlwes_insert, mlwes_to_mlwe};
// Kernels stay reachable via `conversion::kernels::lwe::*` but are intentionally
// not re-exported here (the orchestrator is the public entry point).
