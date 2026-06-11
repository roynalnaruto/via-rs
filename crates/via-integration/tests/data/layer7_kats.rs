//! Layer-7 (VIA-B) KAT fixtures — **intentionally none** (decision record).
//!
//! Unlike layers 3–6, VIA-B ships **no KAT vectors**, by design:
//!
//! - The KATs in this repo assert **byte-for-byte parity between Rust and the
//!   Python reference** (`.references/via-spec`, via `scripts/gen_layerN_kats.py`).
//!   They exist to catch divergence from that reference oracle.
//! - The reference implements **VIA and VIA-C only**: there is no `pir/via_b/`,
//!   no `repack` / `mlwes_to_mlwe` / `Extr_d`, and no `gen_layer7_kats.py`. It even
//!   self-documents the gap — `via_noise.py` notes "No repacking for single-query
//!   VIA-C", and `gen_layer5_kats.py` notes "`Extr_d` is absent from the Python
//!   reference (VIA-B only)".
//! - With **no Python oracle**, a cross-language VIA-B KAT is impossible. A
//!   *Rust-golden* snapshot (asserting Rust against its own pinned-seed output)
//!   would only duplicate the seeded VIA-B regression anchors that already exist:
//!   [`batch_e2e_toy`](../batch_e2e_toy.rs) (toy single-prime) and
//!   [`batch_e2e_paper`](../batch_e2e_paper.rs) (paper RNS) — both pin a
//!   deterministic seed and assert full record recovery end-to-end.
//!
//! So Layer-7 adds no KAT fixtures. This file is retained as the decision record,
//! next to the real cross-language fixtures (`data/layer6_kats.rs`, etc.).
