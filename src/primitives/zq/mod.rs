//! Primitive §0.1 — integers modulo $q$.
//!
//! This module is the foundation of the VIA primitive layering: every
//! polynomial coefficient at every layer of the stack lives in some
//! $\mathbb{Z}_q$ described by this module. See `.docs/primitives.md` §0.1
//! for the mathematical contract.
//!
//! ## Three implementations of `Modulus`
//!
//! - [`modulus::ConstModulus<Q>`] — zero-sized, compile-time modulus, used for the
//!   paper's parameter sets (see [`modulus::paper`]).
//! - [`modulus::PowerOfTwoModulus<LOG2_Q>`] — zero-sized, compile-time
//!   $q = 2^{\text{LOG2\\_Q}}$, mask reduction. Used for $q_4$ and $p$.
//! - [`modulus::DynModulus`] — runtime modulus carrying its precomputed Barrett
//!   constants. Used for tests, toy parameters, and JSON-driven test vectors.
//!
//! ## Two API surfaces
//!
//! - [`element::Zq<M>`] — single-value ergonomic wrapper with operator overloads,
//!   [`subtle::ConditionallySelectable`], and [`zeroize::Zeroize`].
//! - [`ops`] — GPU-portable kernels on flat `&[u64]` slices for batched
//!   polynomial coefficient vectors. Same signature shape as a CUDA / AVX2
//!   kernel; later specialisations will land here without disturbing the
//!   call sites.
//!
//! ## Why `u64`?
//!
//! Every modulus that appears at §0.1 fits in `u64` (largest is VIA-C's $q_1$
//! second RNS prime $\approx 2^{38}$). The composite $q_1$ itself is handled
//! at §0.2 via RNS decomposition into u64 primes; §0.1 only ever sees a
//! single prime or a single power-of-two. Multiplication intermediates use
//! `u128`; the Barrett constant $\mu = \lfloor 2^{128} / q \rfloor$ also
//! fits in `u128`. See `.docs/primitives.md` §0.1 "Why u64 is sufficient" for
//! the per-modulus audit.

pub mod element;
pub mod modulus;
pub mod ops;
pub mod reduce;
