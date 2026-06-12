//! RNS / double-CRT representation.
//!
//! For a composite modulus $Q = q^{(0)} \cdot q^{(1)}$ with $\gcd(q^{(0)},
//! q^{(1)}) = 1$, the Chinese Remainder Theorem gives the ring isomorphism
//! $\mathbb{Z}\_{Q} \cong \mathbb{Z}\_{q^{(0)}} \times \mathbb{Z}\_{q^{(1)}}$, so
//! addition, subtraction, multiplication, and negation all act
//! coordinate-wise on the residue pair.
//!
//! ## Why exactly two primes?
//!
//! Every realistic parameter set uses **at most two RNS primes**:
//!
//! - **VIA**: $q_1 = 268\,369\,921 \cdot 536\,608\,769 \approx 2^{57}$.
//! - **VIA-C / VIA-B**: $q_1 = 137\,438\,822\,401 \cdot 274\,810\,798\,081
//!   \approx 2^{75}$.
//!
//! The remaining moduli ($q_2, q_3, q_4, p$) are single primes or powers of
//! two and live entirely in the single-prime $\mathbb{Z}_q$ layer. A generic
//! $K$-prime RNS would force
//! const-generic arrays (and either typelists or runtime `Vec`s) for zero
//! benefit at our parameter scale. Fixing the basis at two primes lets every
//! reconstruction emit a tidy `u128` (since $Q < 2^{126}$ always, and
//! $\le 2^{75}$ for paper params), keeps the operator overloads statically
//! expanded, and avoids any `alloc` dependency under `#![no_std]`.
//!
//! ## What is **not** here
//!
//! Double-CRT — the composition of this scalar RNS with the per-prime
//! negacyclic NTT — lives in the ring layer alongside $R_{n, q}$ and the NTT.
//! **This module is scalar-only**: it provides the [`RnsBasis`] trait,
//! the [`element::RnsZq<B>`] wrapper, and per-prime SoA slice kernels, but the
//! polynomial-shaped per-RNS-slot layout is the ring layer's responsibility.
//!
//! ## API tiers (mirroring the single-prime $\mathbb{Z}_q$ layer)
//!
//! - [`basis::RnsBasis`] — the trait, value-typed and `'static`.
//! - [`basis::ConstRnsBasis<Q0, Q1>`] / [`basis::paper`] — zero-sized,
//!   compile-time CRT inverse, used for the paper-pinned parameter sets.
//! - [`basis::DynRnsBasis`] — runtime, panics on coprimality failure.
//! - [`element::RnsZq<B>`] — single-value ergonomic wrapper with operator
//!   overloads, [`subtle::ConditionallySelectable`], and [`zeroize::Zeroize`].
//! - [`ops`] — GPU-portable kernels on flat per-prime `&[u64]` slices;
//!   internally two parallel calls to the single-prime kernel for each
//!   operation.
//!
//! [`RnsBasis`]: basis::RnsBasis

pub mod basis;
pub mod element;
pub mod ops;
pub mod reduce;
