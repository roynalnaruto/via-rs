//! Primitive §0.3 — the polynomial ring $R_{n, q} = \mathbb{Z}_q\lbrack X\rbrack / (X^n + 1)$.
//!
//! This module is the carrier type for every higher layer of the VIA stack:
//! secret keys, plaintexts, RLWE / RGSW / MLWE ciphertexts, ring-switching
//! key samples, database cells. See `.docs/primitives.md` §0.3 for the
//! mathematical contract.
//!
//! ## Two API tiers
//!
//! - **Single-prime** [`element::Poly<N, M, F>`] — coefficient or evaluation
//!   form polynomial over $\mathbb{Z}_q$ with `M: Modulus` (see §0.1). Used
//!   at every modulus except the composite $q_1$.
//! - **RNS** [`rns_element::PolyRns<N, B, F>`] — paired storage, one
//!   single-prime polynomial per RNS slot, with `B: RnsBasis` (see §0.2).
//!   Used at the composite $q_1$ in realistic VIA-C / VIA-B parameters.
//!
//! Both shapes carry the ring degree $N$ as a `const` generic. $N$ must be a
//! power of two and $\ge 2$ — enforced at monomorphisation by a `_CHECK`
//! block that fires the first time any constructor is reached for a given
//! type instantiation.
//!
//! ## Two forms (Coefficient / Evaluation) via typestate
//!
//! The ring has two natural representations:
//!
//! - **Coefficient form** — the canonical $\sum_i v_i X^i$ basis; natural for
//!   sampling, encode/decode, addition, ring embedding/projection, gadget
//!   decomposition, and the deterministic $X^k$ rotation (paper §4.5).
//! - **Evaluation form** — the negacyclic NTT evaluations at the primitive
//!   $2N$-th roots of unity; natural for $O(N)$ pointwise multiplication
//!   and the external product.
//!
//! Both share one struct name parameterised by a [`form::Form`] marker:
//! [`form::Coefficient`] or [`form::Evaluation`]. Mixing forms (e.g.
//! `coeff_poly + eval_poly`) is a **compile error** — the typestate
//! parameter is part of the type, not a runtime flag. Conversions are
//! explicit: `poly.into_eval()` / `poly.into_coeff()`.
//!
//! ## Multiplication semantics
//!
//! `Mul` on the **coefficient form** is **schoolbook negacyclic** — $O(N^2)$
//! scalar `Modulus::mul`s with the negacyclic-wrap branch. No hidden NTT.
//! This is intentional: the PIR pipeline lives in evaluation form (most
//! ciphertext storage and arithmetic happens after a single `into_eval()`
//! call), so coefficient-form multiplication is reserved for setup paths
//! where cost transparency matters more than wall-clock speed.
//!
//! `Mul` on the **evaluation form** is **pointwise** — a `zq::ops::mul_slice`
//! Hadamard product. After the NTT, ring multiplication is component-wise.
//!
//! Callers that want $O(N \log N)$ multiplication on coefficient-form input
//! must opt in explicitly: convert via `into_eval()`, multiply pointwise,
//! convert back via `into_coeff()`. §0.4 may add a `mul_via_ntt` helper that
//! wraps this round-trip; the plain `Mul` impl will not change semantics.
//!
//! The negacyclic NTT transform body (§0.4) — `element::Poly::into_eval` /
//! `into_coeff` and their `ntt::ntt_inplace` / `ntt::intt_inplace` cores — is
//! implemented and tested on both backends (the round-trip tests run; they are
//! not `#[ignore]`d). See "Multiplication semantics" above. Still pending:
//!
//! ## What is **not** here
//!
//! - **Ring embedding / projection** $\iota_j^{n' \to n}$ and
//!   $\pi_j^{n \to n'}$ — §0.5. Pure coefficient-level index moves; lands
//!   when the ring-switching layer (§3.3) needs it.
//! - **Centred representation** as a free function — §0.6. The
//!   [`element::Poly::to_centered_coeffs`] method on the coefficient form is
//!   the on-ramp; the layer-wide free function joins it later.
//! - **Gadget decomposition** of a polynomial coefficient-wise — §2.3.
//!
//! ## GPU portability
//!
//! Every slice kernel takes flat `&[u64]` slices (matching `zq::ops` and
//! `rns::ops` at §0.1 / §0.2), and the polynomial types carry
//! `#[repr(C, align(32))]` for AVX2 / AVX-512 / GPU NTT downstream. The CPU
//! loop body in [`ops::negacyclic_mul_slice`] is intentionally the same
//! code we will later vectorise and lower to CUDA / Metal.

pub mod abstraction;
pub mod element;
pub mod form;
pub mod ntt;
pub mod ntt_abstraction;
pub mod ops;
pub mod reshape;
pub mod rns_element;
pub mod rns_ops;
pub mod rns_reshape;

pub use abstraction::RingPoly;
pub use ntt_abstraction::RingPolyEval;
