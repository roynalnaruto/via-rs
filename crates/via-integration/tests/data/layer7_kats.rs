//! VIA-B KAT fixtures — **intentionally none** (decision record).
//!
//! Unlike the other layers, VIA-B ships **no KAT vectors**, by design:
//!
//! - The KATs in this repo assert **byte-for-byte parity between Rust and a
//!   cross-language oracle**. They exist to catch divergence from that
//!   oracle.
//! - That oracle covers **VIA and VIA-C only**: it has no VIA-B module,
//!   no `repack` / `mlwes_to_mlwe` / `Extr_d`, and no VIA-B KAT generator. The
//!   gap is explicit — there is no repacking for single-query VIA-C, and
//!   `Extr_d` is VIA-B only.
//! - With **no cross-language oracle** for VIA-B, such a KAT is impossible. A
//!   *Rust-golden* snapshot (asserting Rust against its own pinned-seed output)
//!   would only duplicate the seeded VIA-B regression anchors that already exist:
//!   [`batch_e2e_toy`](../batch_e2e_toy.rs) (toy single-prime) and
//!   [`batch_e2e_paper`](../batch_e2e_paper.rs) (paper RNS) — both pin a
//!   deterministic seed and assert full record recovery end-to-end.
//!
//! So VIA-B adds no KAT fixtures. This file is retained as the decision record,
//! next to the real cross-language fixtures (`data/layer6_kats.rs`, etc.).
