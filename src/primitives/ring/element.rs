//! Single-prime polynomial wrapper [`Poly<N, M, F>`].
//!
//! [`Poly`] carries an `N`-coefficient vector in canonical reduced
//! `u64` form alongside a [`Modulus`] context and a typestate marker
//! [`F: Form`](Form) that distinguishes coefficient form from evaluation
//! (NTT) form at the type level.
//!
//! # Invariants
//!
//! - $N \ge 2$ and $N$ is a power of two — enforced at monomorphisation
//!   by a `const _CHECK` block (mirrors the §0.1 / §0.2 pattern).
//! - Every stored value is in $[0, q)$. Safe constructors reduce; the
//!   `unsafe` [`Poly::from_reduced_unchecked`] trusts the caller.
//!
//! # Field naming
//!
//! The values field is called `values: [u64; N]` rather than `coeffs`,
//! because the same struct can be in either form. Under
//! `F = Coefficient` they are the polynomial's coefficients in the
//! monomial basis $\sum_i v_i X^i$; under `F = Evaluation` they are
//! evaluations at the negacyclic $2N$-th roots of unity (filled in by
//! §0.4's NTT). Form-aware accessors `coeff(i)` / `eval(i)` exist on
//! the respective impls to read the values with the right name.
//!
//! # Storage rationale (`[u64; N]` vs `Vec`)
//!
//! The crate is `#![no_std]` with no allocator dependency. Const-generic
//! arrays let us validate $N$ at monomorphisation, let LLVM propagate
//! trip counts for unrolling, eliminate allocator-failure code paths in
//! cryptographic hot loops, give type-level "same degree" guarantees on
//! every binary op without runtime asserts, and map cleanly to a CUDA
//! shared-memory buffer or SoA stride layout. Realistic $N$ tops out at
//! $n_1 = 2048$ (paper §A.1), making one `Poly` 16 KiB — well within
//! the default 8 MiB stack. Higher layers that aggregate many polys per
//! ciphertext will introduce `Box<Poly<…>>`; the const-generic shape
//! survives the transition unchanged.
//!
//! # Multiplication semantics
//!
//! `Mul` on `Poly<N, M, Coefficient>` is **schoolbook negacyclic**
//! ($O(N^2)$) — see the module-level docs for the rationale. `Mul` on
//! `Poly<N, M, Evaluation>` is **pointwise** ($O(N)$). To go from
//! coefficient form to $O(N \log N)$ multiplication, call
//! [`Poly::into_eval`] explicitly. **No hidden NTT inside `*`.**

use core::fmt;
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use rand_core::RngCore;
use subtle::{Choice, ConditionallySelectable, ConstantTimeEq};
use zeroize::Zeroize;

use crate::primitives::zq::element::Zq;
use crate::primitives::zq::modulus::Modulus;
use crate::primitives::zq::ops as zq_ops;

use super::form::{Coefficient, Evaluation, Form};
use super::ntt::{self, NttFriendly};
use super::ops as ring_ops;
use super::reshape;

/// A polynomial in $R_{n, q} = \mathbb{Z}_q\lbrack X\rbrack / (X^n + 1)$, represented in
/// the form indicated by the typestate parameter `F`.
///
/// `Poly<N, M, Coefficient>` carries the monomial-basis coefficients;
/// `Poly<N, M, Evaluation>` carries the negacyclic-NTT evaluation values.
/// The two are distinct types — the compiler refuses to mix them, so you
/// cannot accidentally add a coefficient-form poly to an eval-form poly.
///
/// # Memory layout
///
/// `#[repr(C, align(32))]` — the `values` array is 32-byte aligned for
/// AVX2 / AVX-512 SIMD and CUDA shared-memory loads at §0.4+. The
/// `modulus` and `_form` fields sit after the value buffer and pay only
/// tail padding (which is zero bytes for the `ConstModulus` /
/// `PowerOfTwoModulus` zero-sized cases and at most a few bytes for
/// `DynModulus`).
///
/// # Construction
///
/// Use [`Poly::zero`], [`Poly::one`], [`Poly::new`] (which reduces each
/// lane), or [`Poly::random`]. The `unsafe` [`Poly::from_reduced_unchecked`]
/// is for callers who can prove the input is already in canonical form
/// (e.g. when forwarding from another `Poly`'s buffer).
#[repr(C, align(32))]
pub struct Poly<const N: usize, M: Modulus, F: Form> {
    /// Canonical-reduced `u64` values of the polynomial. Coefficients
    /// under `F = Coefficient`; NTT evaluations under `F = Evaluation`.
    values: [u64; N],
    modulus: M,
    _form: PhantomData<F>,
}

// ---------------------------------------------------------------------------
// `_CHECK` block: shared across all (`N`, `M`, `F`) so any constructor
// triggers it at monomorphisation.
// ---------------------------------------------------------------------------

impl<const N: usize, M: Modulus, F: Form> Poly<N, M, F> {
    /// Compile-time validation block. Asserts $N \ge 2$ and $N$ is a
    /// power of two. Touched by [`Poly::N`] and from every constructor
    /// (via `let () = Self::_CHECK;`) so a bad $N$ fails to compile.
    const _CHECK: () = {
        assert!(N >= 2, "Poly: N >= 2");
        assert!(N.is_power_of_two(), "Poly: N must be a power of two");
    };

    /// The ring degree, also reachable as the const-generic parameter.
    ///
    /// Reading this constant forces [`Self::_CHECK`] to evaluate at
    /// monomorphisation; combined with the `let () = Self::_CHECK;` line
    /// in every constructor, this means any invalid `N` fails to compile.
    pub const N: usize = {
        let () = Self::_CHECK;
        N
    };

    /// The modulus this polynomial is associated with.
    #[inline(always)]
    pub const fn modulus(&self) -> M
    where
        M: Copy,
    {
        self.modulus
    }

    /// Borrow the underlying values as a fixed-size slice (in
    /// canonical reduced form). Form-neutral accessor — for a typed read
    /// of a single value use [`Poly::coeff`] on coefficient form or
    /// [`Poly::eval`] on evaluation form.
    #[inline(always)]
    pub const fn values(&self) -> &[u64; N] {
        &self.values
    }

    /// Construct a [`Poly`] from a `u64` array that is **already in
    /// canonical reduced form** (each lane in $[0, q)$).
    ///
    /// # Safety
    ///
    /// Caller must guarantee every `values[i] < modulus.q()`. Misuse does
    /// not cause memory-safety UB (this is a plain `u64`-array wrapper),
    /// but it silently corrupts downstream cryptographic arithmetic — a
    /// class of bug we want every caller to acknowledge explicitly. Use
    /// the safe constructors ([`Poly::zero`], [`Poly::one`], [`Poly::new`],
    /// [`Poly::random`]) when you cannot prove the precondition locally.
    #[inline(always)]
    pub const unsafe fn from_reduced_unchecked(modulus: M, values: [u64; N]) -> Self {
        Self {
            values,
            modulus,
            _form: PhantomData,
        }
    }
}

// ---------------------------------------------------------------------------
// Constructors and accessors common to either form
// (shared because the underlying storage is identical).
// ---------------------------------------------------------------------------

impl<const N: usize, M: Modulus, F: Form> Poly<N, M, F> {
    /// The zero polynomial. The all-zeros buffer encodes zero in either
    /// form (NTT of zero is zero), so this is form-neutral.
    #[inline(always)]
    pub fn zero(modulus: M) -> Self {
        let () = Self::_CHECK;
        // SAFETY: 0 is in [0, q) for every valid modulus (q >= 2).
        unsafe { Self::from_reduced_unchecked(modulus, [0u64; N]) }
    }

    /// Construct from arbitrary `u64` values, reducing each lane into
    /// $[0, q)$.
    ///
    /// On the coefficient form this is `[v_0, v_1, …, v_{N-1}]` interpreted
    /// as $\sum_i v_i X^i$. On the evaluation form this trusts the caller
    /// that the `values` are already valid negacyclic-NTT evaluations of
    /// some polynomial — they are not re-NTT'd.
    ///
    /// # Evaluation form on non-NTT-friendly moduli
    ///
    /// This constructor is form-neutral: it accepts any `M: Modulus`,
    /// not just `NttFriendly<N>`. Constructing
    /// `Poly<N, M, Evaluation>` when `M` is *not* `NttFriendly<N>`
    /// produces a buffer that has **no underlying $R_{n, q}$ polynomial**
    /// — the NTT bijection $R_{n, q} \to \mathbb{Z}_q^N$ requires
    /// $q \equiv 1 \pmod{2N}$. Such an eval-form `Poly` still supports
    /// the pointwise arithmetic ops (`add`, `sub`, `neg`, `mul`,
    /// scalar `mul`) as raw $\mathbb{Z}_q^N$ vectors, which is useful
    /// for testing and for protocol layers that operate componentwise.
    /// But [`Poly::eval`] returns "the value at the $i$-th NTT point"
    /// — meaningless without an NTT — and there is no way to recover
    /// a coefficient-form polynomial without `NttFriendly<N>`.
    ///
    /// Production call sites in `Evaluation` form should always go
    /// through [`Poly::into_eval`] from coefficient form, which
    /// statically requires `M: NttFriendly<N>`.
    pub fn new(modulus: M, values: [u64; N]) -> Self {
        let () = Self::_CHECK;
        let mut reduced = [0u64; N];
        for i in 0..N {
            reduced[i] = modulus.reduce_u64(values[i]);
        }
        // SAFETY: every lane is reduced via `Modulus::reduce_u64`.
        unsafe { Self::from_reduced_unchecked(modulus, reduced) }
    }

    /// Sample a uniformly random polynomial by drawing each lane
    /// independently via [`Zq::random`].
    ///
    /// On the coefficient form this samples a uniform polynomial in
    /// $R_{n, q}$. On the evaluation form, when `M: NttFriendly<N>`,
    /// this samples a uniform vector of NTT evaluations — by the NTT
    /// bijection this is *also* uniform in $R_{n, q}$.
    ///
    /// # Evaluation form on non-NTT-friendly moduli
    ///
    /// As with [`Poly::new`], this constructor is form-neutral: it
    /// accepts any `M: Modulus`. When `M` is *not* `NttFriendly<N>`
    /// the result is a uniform $\mathbb{Z}_q^N$ vector with **no
    /// associated $R_{n, q}$ polynomial** — the "uniform in $R_{n, q}$
    /// via NTT bijection" guarantee in the paragraph above is conditional
    /// on the bijection existing. Use the evaluation form on
    /// non-NTT-friendly moduli only for componentwise testing and
    /// protocol scratch buffers.
    pub fn random<R: RngCore + ?Sized>(modulus: M, rng: &mut R) -> Self {
        let () = Self::_CHECK;
        let mut buf = [0u64; N];
        for slot in &mut buf {
            *slot = Zq::random(modulus, rng).to_u64();
        }
        // SAFETY: `Zq::random` always returns a value in [0, q).
        unsafe { Self::from_reduced_unchecked(modulus, buf) }
    }
}

// ---------------------------------------------------------------------------
// Coefficient-form impl: `one`, ring multiplication via schoolbook,
// `mul_x_pow`, `coeff` accessor, `to_centered_coeffs`, and the
// `into_eval()` transform stub.
// ---------------------------------------------------------------------------

impl<const N: usize, M: Modulus> Poly<N, M, Coefficient> {
    /// The constant polynomial $1$: $[1, 0, 0, \ldots, 0]$.
    ///
    /// # Panics in debug
    ///
    /// Debug-asserts $q \ge 2$ so that the leading `1` is in $[0, q)$.
    #[inline(always)]
    pub fn one(modulus: M) -> Self {
        let () = Self::_CHECK;
        debug_assert!(modulus.q() >= 2, "Poly::one requires q >= 2");
        let mut buf = [0u64; N];
        buf[0] = 1;
        // SAFETY: `1 < q` when `q >= 2`; remaining lanes are zero.
        unsafe { Self::from_reduced_unchecked(modulus, buf) }
    }

    /// The coefficient at position $i$ as a [`Zq`].
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    #[inline(always)]
    pub fn coeff(&self, i: usize) -> Zq<M> {
        assert!(i < N, "Poly::coeff: index {i} out of range (N = {N})");
        // SAFETY: stored values are in [0, q) by invariant.
        unsafe { Zq::from_reduced_unchecked(self.modulus, self.values[i]) }
    }

    /// Write the coefficient at position $i$.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N` or if `v.modulus() != self.modulus()`.
    #[inline(always)]
    pub fn set_coeff(&mut self, i: usize, v: Zq<M>) {
        assert!(i < N, "Poly::set_coeff: index {i} out of range (N = {N})");
        assert!(
            v.modulus() == self.modulus,
            "Poly::set_coeff: modulus mismatch",
        );
        self.values[i] = v.to_u64();
    }

    /// Lift every coefficient to its centred representation
    /// $\tilde{v}_i \in (-\lfloor q/2 \rfloor, \lfloor q/2 \rfloor]$ —
    /// see §0.6. Used at decoding boundaries; **not constant-time**
    /// over the input values. For secret-data inputs, use
    /// [`Self::to_centered_coeffs_ct`].
    pub fn to_centered_coeffs(&self, dst: &mut [i64; N]) {
        zq_ops::to_centered_i64_slice(self.modulus, dst, &self.values);
    }

    /// Constant-time variant of [`Self::to_centered_coeffs`]. Use this
    /// when the polynomial is a secret (e.g. §3.4 secret-key rekeying).
    pub fn to_centered_coeffs_ct(&self, dst: &mut [i64; N]) {
        zq_ops::to_centered_i64_ct_slice(self.modulus, dst, &self.values);
    }

    /// Deterministic rotation: returns $X^k \cdot \mathrm{self}$ in
    /// $R_{n, q}$ — paper §4.5. The rotation exponent $k$ is taken as a
    /// public parameter and the implementation may branch on it; the
    /// coefficient *values* are still handled in constant time.
    ///
    /// # Secret-$k$: do not use this method
    ///
    /// **Do not call this with a $k$ that depends on secret data**
    /// (e.g. a query index or any value derived from a key or
    /// plaintext). The implementation reduces `k` modulo `2 * n` and
    /// branches on the wrap quadrant, which leaks `k` through timing.
    /// At every paper §4.5 call site `k` is a public loop induction
    /// variable (the per-bit shift $2^i$ in `CRot`), and the
    /// *encrypted* control bits feed [`CMux`](../../../) — not this
    /// rotation primitive directly. When the §4.4 `CRot` primitive
    /// lands it will compose `mul_x_pow` with `CMux` to support
    /// encrypted exponents; route secret-$k$ call sites through that
    /// composite rather than calling `mul_x_pow` with secret input.
    pub fn mul_x_pow(&self, k: usize) -> Self {
        let mut dst = [0u64; N];
        ring_ops::rotate_slice(self.modulus, &mut dst, &self.values, k);
        // SAFETY: `rotate_slice` only permutes / negates canonical lanes.
        unsafe { Self::from_reduced_unchecked(self.modulus, dst) }
    }
}

/// Evaluation-form conversion — only available when `M: NttFriendly<N>`.
impl<const N: usize, M: NttFriendly<N>> Poly<N, M, Coefficient> {
    /// Convert to evaluation form via the forward negacyclic NTT
    /// (§0.4).
    ///
    /// The output buffer is in **bit-reversed** order — `Poly::eval(i)`
    /// on the result returns the value at $\psi^{2 \cdot \mathrm{br}(i) + 1}$.
    /// Pointwise multiplication and additive ring ops are order-agnostic,
    /// so consumers that only round-trip + multiply + add do not see the
    /// permutation.
    ///
    /// # Secret-bearing inputs
    ///
    /// This method consumes `self` by value: the source buffer is
    /// moved into a stack-local scratch slot and the in-place NTT
    /// overwrites it. Rust does **not** guarantee the original stack
    /// location is zeroed after the move. If the caller's `self`
    /// carries secret data (e.g. a §2.1 secret-key polynomial, or a
    /// §3.3 ring-switch-key intermediate that exposes coefficient
    /// statistics of $S_1$), the residual stack slot remains
    /// observable to anyone who can read that memory.
    ///
    /// Today the only call sites are tests and non-secret protocol
    /// paths, so direct use is safe. When secret-bearing types land
    /// (§2.1 `SecretKey`, §3.3 ring-switch keys), funnel secret
    /// inputs through a `_zeroizing` wrapper that calls
    /// [`zeroize::Zeroize::zeroize`] on `self.values` immediately
    /// after the NTT. **Do not** call this method directly on a
    /// secret-bearing `Poly`.
    #[inline]
    pub fn into_eval(self) -> Poly<N, M, Evaluation> {
        let mut buf = self.values;
        ntt::ntt_inplace::<N, M>(self.modulus, &mut buf);
        // SAFETY: ntt_inplace preserves the canonical-reduction invariant.
        unsafe { Poly::<N, M, Evaluation>::from_reduced_unchecked(self.modulus, buf) }
    }
}

// ---------------------------------------------------------------------------
// §0.5 ring embedding / projection — coefficient-form only.
// ---------------------------------------------------------------------------

/// Single-slot embedding $\iota_j^{N \to N_\text{large}}$ — see §0.5.
///
/// Compile-time `_CHECK` (via inline `const` block) enforces
/// $N_\text{large} \ge N$, $N \mid N_\text{large}$, and $N_\text{large}$
/// is a power of two.
impl<const N: usize, M: Modulus> Poly<N, M, Coefficient> {
    /// Place `self` into slot `slot` of a polynomial in the larger ring
    /// $R_{N_\text{large}, q}$. Coefficient $f_i$ lands at position
    /// $d \cdot i + \mathrm{slot}$ where $d = N_\text{large} / N$.
    ///
    /// # Panics
    ///
    /// Two distinct failure modes:
    ///
    /// - **Compile-time** (via inline `const` block): `N_LARGE < N`,
    ///   `N_LARGE` not a multiple of `N`, or `N_LARGE` not a power of
    ///   two. These depend only on the const-generic parameters and
    ///   are caught at monomorphisation.
    /// - **Runtime** (via plain `assert!`): `slot >= d = N_LARGE / N`.
    ///   The `slot` argument is a runtime value, so the const block
    ///   *cannot* check it. `f.embed_at::<16>(7)` when $d = 4$
    ///   compiles, then panics when executed. This is by design — the
    ///   slot index in §3.3 / §5.5 / §6.3 is a loop variable bounded
    ///   by $d$ at the call site, not a constant known at the type
    ///   parameter.
    pub fn embed_at<const N_LARGE: usize>(&self, slot: usize) -> Poly<N_LARGE, M, Coefficient> {
        const {
            assert!(N_LARGE >= N, "embed_at: N_LARGE >= N");
            assert!(N_LARGE.is_multiple_of(N), "embed_at: N must divide N_LARGE",);
            assert!(
                N_LARGE.is_power_of_two(),
                "embed_at: N_LARGE must be a power of two",
            );
        }
        let mut buf = [0u64; N_LARGE];
        reshape::embed_at_slice(&self.values, &mut buf, slot);
        // SAFETY: embed_at_slice writes lanes that are either zero or
        // copies of canonical-form source lanes.
        unsafe { Poly::<N_LARGE, M, Coefficient>::from_reduced_unchecked(self.modulus, buf) }
    }
}

/// Single-slot projection $\pi_j^{N \to N_\text{small}}$ and the
/// $d$-fold pack/unpack pair.
impl<const N: usize, M: Modulus> Poly<N, M, Coefficient> {
    /// Extract slot `slot` of `self` into a polynomial in the smaller
    /// ring $R_{N_\text{small}, q}$. The coefficient at position
    /// $(d \cdot i) + \mathrm{slot}$ becomes coefficient $i$ of the
    /// output, where $d = N / N_\text{small}$.
    ///
    /// # Panics
    ///
    /// Same compile-time vs runtime split as [`Self::embed_at`]:
    ///
    /// - **Compile-time**: `N_SMALL > N`, `N` not a multiple of
    ///   `N_SMALL`, or `N_SMALL` not a power of two (when $\ge 2$;
    ///   the degenerate `N_SMALL = 1` is rejected by `Poly`'s own
    ///   `_CHECK` which requires `N >= 2`).
    /// - **Runtime**: `slot >= d`. Runtime panic, not compile error.
    pub fn project_at<const N_SMALL: usize>(&self, slot: usize) -> Poly<N_SMALL, M, Coefficient> {
        const {
            assert!(N_SMALL <= N, "project_at: N_SMALL <= N");
            assert!(
                N.is_multiple_of(N_SMALL),
                "project_at: N_SMALL must divide N",
            );
            assert!(
                N_SMALL.is_power_of_two(),
                "project_at: N_SMALL must be a power of two",
            );
        }
        let mut buf = [0u64; N_SMALL];
        reshape::project_at_slice(&self.values, &mut buf, slot);
        // SAFETY: project_at_slice copies canonical lanes.
        unsafe { Poly::<N_SMALL, M, Coefficient>::from_reduced_unchecked(self.modulus, buf) }
    }

    /// $d$-fold packing: pack `slots[0..d]` into one polynomial in the
    /// larger ring $R_{N_\text{large}, q}$, where `slots[j]` is placed
    /// in slot `j`. Requires `slots.len() * N == N_LARGE`.
    ///
    /// # Panics
    ///
    /// - **Compile-time**: $N$ / $N_\text{large}$ relationship is
    ///   invalid (same checks as [`Self::embed_at`]).
    /// - **Runtime**: `slots.len() != N_LARGE / N` (slot count is a
    ///   slice length, not a const generic), or any `slots[j]` has a
    ///   different modulus than `modulus`.
    pub fn pack_slots<const N_LARGE: usize>(
        modulus: M,
        slots: &[Poly<N, M, Coefficient>],
    ) -> Poly<N_LARGE, M, Coefficient> {
        const {
            assert!(N_LARGE >= N, "pack_slots: N_LARGE >= N");
            assert!(
                N_LARGE.is_multiple_of(N),
                "pack_slots: N must divide N_LARGE",
            );
            assert!(
                N_LARGE.is_power_of_two(),
                "pack_slots: N_LARGE must be a power of two",
            );
        }
        let d = N_LARGE / N;
        assert_eq!(
            slots.len(),
            d,
            "pack_slots: expected {d} slot polys, got {}",
            slots.len(),
        );
        // Build the concatenated source layout `[slot_0 | slot_1 | …]`
        // on the stack and run the $d$-fold permutation kernel. The
        // intermediate scratch is the same size as the output (one
        // `Poly<N_LARGE>` worth) and is dropped after the move.
        let mut concat = [0u64; N_LARGE];
        for (j, slot_poly) in slots.iter().enumerate() {
            assert!(
                slot_poly.modulus == modulus,
                "pack_slots: modulus mismatch at slot {j}",
            );
            concat[j * N..(j + 1) * N].copy_from_slice(&slot_poly.values);
        }
        let mut packed = [0u64; N_LARGE];
        reshape::pack_slots_slice(&concat, &mut packed, N);
        // SAFETY: pack_slots_slice only permutes canonical lanes.
        unsafe { Poly::<N_LARGE, M, Coefficient>::from_reduced_unchecked(modulus, packed) }
    }

    /// $d$-fold unpacking: split `self` into `d` slot polynomials.
    /// `dsts[j]` receives slot `j`. Requires `dsts.len() * N_SMALL == N`.
    ///
    /// # Panics
    ///
    /// - **Compile-time**: invalid $N$ / $N_\text{small}$ relationship
    ///   (same checks as [`Self::project_at`]).
    /// - **Runtime**: `dsts.len() != N / N_SMALL`, or any element of
    ///   `dsts` has a different modulus than `self`.
    pub fn unpack_slots<const N_SMALL: usize>(&self, dsts: &mut [Poly<N_SMALL, M, Coefficient>]) {
        const {
            assert!(N_SMALL <= N, "unpack_slots: N_SMALL <= N");
            assert!(
                N.is_multiple_of(N_SMALL),
                "unpack_slots: N_SMALL must divide N",
            );
            assert!(
                N_SMALL.is_power_of_two(),
                "unpack_slots: N_SMALL must be a power of two",
            );
        }
        let d = N / N_SMALL;
        assert_eq!(
            dsts.len(),
            d,
            "unpack_slots: expected {d} slot polys, got {}",
            dsts.len(),
        );
        // Run the $d$-fold permutation into a scratch buffer in
        // concatenated layout, then copy each slot's lanes into the
        // caller's `Poly`.
        let mut concat = [0u64; N];
        reshape::unpack_slots_slice(&self.values, &mut concat, N_SMALL);
        for (j, slot_poly) in dsts.iter_mut().enumerate() {
            assert!(
                slot_poly.modulus == self.modulus,
                "unpack_slots: modulus mismatch at slot {j}",
            );
            slot_poly
                .values
                .copy_from_slice(&concat[j * N_SMALL..(j + 1) * N_SMALL]);
        }
    }
}

// ---------------------------------------------------------------------------
// Evaluation-form impl: pointwise eval accessor and the `into_coeff`
// transform stub. `Mul` is pointwise (defined further down).
// ---------------------------------------------------------------------------

impl<const N: usize, M: Modulus> Poly<N, M, Evaluation> {
    /// The evaluation at the $i$-th negacyclic-NTT point, as a [`Zq`].
    ///
    /// # Panics
    ///
    /// Panics if `i >= N`.
    #[inline(always)]
    pub fn eval(&self, i: usize) -> Zq<M> {
        assert!(i < N, "Poly::eval: index {i} out of range (N = {N})");
        // SAFETY: stored values are in [0, q) by invariant.
        unsafe { Zq::from_reduced_unchecked(self.modulus, self.values[i]) }
    }

    /// Write the evaluation at the $i$-th NTT point.
    ///
    /// # Panics
    ///
    /// Panics if `i >= N` or if `v.modulus() != self.modulus()`.
    #[inline(always)]
    pub fn set_eval(&mut self, i: usize, v: Zq<M>) {
        assert!(i < N, "Poly::set_eval: index {i} out of range (N = {N})");
        assert!(
            v.modulus() == self.modulus,
            "Poly::set_eval: modulus mismatch",
        );
        self.values[i] = v.to_u64();
    }
}

/// Coefficient-form conversion — only available when `M: NttFriendly<N>`.
impl<const N: usize, M: NttFriendly<N>> Poly<N, M, Evaluation> {
    /// Convert back to coefficient form via the inverse negacyclic NTT
    /// (§0.4).
    ///
    /// Assumes the eval-form buffer is in **bit-reversed** order — i.e.
    /// the output of a prior [`Poly::<N, M, Coefficient>::into_eval`]
    /// call. The result is in natural-coefficient order.
    ///
    /// # Secret-bearing inputs
    ///
    /// Same trust boundary as
    /// [`Poly::<N, M, Coefficient>::into_eval`]: `self` is consumed
    /// by value and Rust does not guarantee the original stack slot
    /// is zeroed after the move. If `self` carries secret data, the
    /// residual eval-form buffer remains observable. Route secret
    /// inputs through a `_zeroizing` wrapper (to land with §2.1
    /// `SecretKey` / §3.3 ring-switch keys); current call sites are
    /// non-secret and use this method directly.
    #[inline]
    pub fn into_coeff(self) -> Poly<N, M, Coefficient> {
        let mut buf = self.values;
        ntt::intt_inplace::<N, M>(self.modulus, &mut buf);
        // SAFETY: intt_inplace preserves the canonical-reduction invariant.
        unsafe { Poly::<N, M, Coefficient>::from_reduced_unchecked(self.modulus, buf) }
    }
}

// ---------------------------------------------------------------------------
// Operator overloads — Coefficient form
// ---------------------------------------------------------------------------

impl<const N: usize, M: Modulus> Add for Poly<N, M, Coefficient> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Poly::add: modulus mismatch");
        let lhs = self.values;
        zq_ops::add_slice(self.modulus, &mut self.values, &lhs, &rhs.values);
        self
    }
}

impl<const N: usize, M: Modulus> Sub for Poly<N, M, Coefficient> {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Poly::sub: modulus mismatch");
        let lhs = self.values;
        zq_ops::sub_slice(self.modulus, &mut self.values, &lhs, &rhs.values);
        self
    }
}

impl<const N: usize, M: Modulus> Neg for Poly<N, M, Coefficient> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        let src = self.values;
        zq_ops::neg_slice(self.modulus, &mut self.values, &src);
        self
    }
}

impl<const N: usize, M: Modulus> Mul for Poly<N, M, Coefficient> {
    type Output = Self;
    /// Schoolbook negacyclic multiplication in $R_{n, q}$ — $O(N^2)$.
    /// **No hidden NTT.** For $O(N \log N)$, call [`Poly::into_eval`]
    /// explicitly.
    #[inline]
    fn mul(self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Poly::mul: modulus mismatch");
        let mut dst = [0u64; N];
        ring_ops::negacyclic_mul_slice(self.modulus, &mut dst, &self.values, &rhs.values);
        // SAFETY: negacyclic_mul_slice writes canonical-reduced lanes.
        unsafe { Self::from_reduced_unchecked(self.modulus, dst) }
    }
}

impl<const N: usize, M: Modulus> Mul<u64> for Poly<N, M, Coefficient> {
    type Output = Self;
    /// Scalar multiplication. The scalar is reduced into $[0, q)$ before
    /// the per-lane multiply.
    #[inline]
    fn mul(mut self, scalar: u64) -> Self {
        let s = self.modulus.reduce_u64(scalar);
        let src = self.values;
        zq_ops::scalar_mul_slice(self.modulus, &mut self.values, &src, s);
        self
    }
}

impl<const N: usize, M: Modulus> AddAssign for Poly<N, M, Coefficient> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize, M: Modulus> SubAssign for Poly<N, M, Coefficient> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize, M: Modulus> MulAssign for Poly<N, M, Coefficient> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize, M: Modulus> MulAssign<u64> for Poly<N, M, Coefficient> {
    #[inline]
    fn mul_assign(&mut self, scalar: u64) {
        *self = *self * scalar;
    }
}

// ---------------------------------------------------------------------------
// Operator overloads — Evaluation form (pointwise mul)
// ---------------------------------------------------------------------------

impl<const N: usize, M: Modulus> Add for Poly<N, M, Evaluation> {
    type Output = Self;
    #[inline]
    fn add(mut self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Poly::add: modulus mismatch");
        let lhs = self.values;
        zq_ops::add_slice(self.modulus, &mut self.values, &lhs, &rhs.values);
        self
    }
}

impl<const N: usize, M: Modulus> Sub for Poly<N, M, Evaluation> {
    type Output = Self;
    #[inline]
    fn sub(mut self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Poly::sub: modulus mismatch");
        let lhs = self.values;
        zq_ops::sub_slice(self.modulus, &mut self.values, &lhs, &rhs.values);
        self
    }
}

impl<const N: usize, M: Modulus> Neg for Poly<N, M, Evaluation> {
    type Output = Self;
    #[inline]
    fn neg(mut self) -> Self {
        let src = self.values;
        zq_ops::neg_slice(self.modulus, &mut self.values, &src);
        self
    }
}

impl<const N: usize, M: Modulus> Mul for Poly<N, M, Evaluation> {
    type Output = Self;
    /// Pointwise (Hadamard) multiplication of NTT evaluations — $O(N)$.
    /// Equals ring multiplication once both operands are in evaluation
    /// form, by the negacyclic NTT bijection.
    #[inline]
    fn mul(mut self, rhs: Self) -> Self {
        assert!(self.modulus == rhs.modulus, "Poly::mul: modulus mismatch");
        let lhs = self.values;
        zq_ops::mul_slice(self.modulus, &mut self.values, &lhs, &rhs.values);
        self
    }
}

impl<const N: usize, M: Modulus> Mul<u64> for Poly<N, M, Evaluation> {
    type Output = Self;
    #[inline]
    fn mul(mut self, scalar: u64) -> Self {
        let s = self.modulus.reduce_u64(scalar);
        let src = self.values;
        zq_ops::scalar_mul_slice(self.modulus, &mut self.values, &src, s);
        self
    }
}

impl<const N: usize, M: Modulus> AddAssign for Poly<N, M, Evaluation> {
    #[inline]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const N: usize, M: Modulus> SubAssign for Poly<N, M, Evaluation> {
    #[inline]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const N: usize, M: Modulus> MulAssign for Poly<N, M, Evaluation> {
    #[inline]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const N: usize, M: Modulus> MulAssign<u64> for Poly<N, M, Evaluation> {
    #[inline]
    fn mul_assign(&mut self, scalar: u64) {
        *self = *self * scalar;
    }
}

// ---------------------------------------------------------------------------
// Cross-form trait impls (apply to either form)
// ---------------------------------------------------------------------------

impl<const N: usize, M: Modulus, F: Form> Copy for Poly<N, M, F> {}

impl<const N: usize, M: Modulus, F: Form> Clone for Poly<N, M, F> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<const N: usize, M: Modulus, F: Form> PartialEq for Poly<N, M, F> {
    /// Equal iff every lane matches **and** the moduli agree. For
    /// zero-sized moduli the modulus check is a no-op. The form is
    /// encoded in the type, so it is not compared dynamically.
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.modulus == other.modulus && self.values == other.values
    }
}

impl<const N: usize, M: Modulus, F: Form> Eq for Poly<N, M, F> {}

impl<const N: usize, M: Modulus, F: Form> ConstantTimeEq for Poly<N, M, F> {
    /// Constant-time equality on the *value* lanes (per-lane `ct_eq`
    /// folded with `&`). The modulus is a public parameter; this
    /// comparison is meaningful only when the caller has already
    /// established both operands share a modulus.
    #[inline]
    fn ct_eq(&self, other: &Self) -> Choice {
        let mut acc = Choice::from(1u8);
        for i in 0..N {
            acc &= self.values[i].ct_eq(&other.values[i]);
        }
        acc
    }
}

impl<const N: usize, M: Modulus, F: Form> ConditionallySelectable for Poly<N, M, F> {
    /// Select `b` when `choice` is set, else `a`. Asserts modulus equality.
    #[inline]
    fn conditional_select(a: &Self, b: &Self, choice: Choice) -> Self {
        assert!(
            a.modulus == b.modulus,
            "Poly::conditional_select: modulus mismatch",
        );
        let mut out = [0u64; N];
        for ((slot, &av), &bv) in out.iter_mut().zip(a.values.iter()).zip(b.values.iter()) {
            *slot = u64::conditional_select(&av, &bv, choice);
        }
        // SAFETY: each lane is one of `a.values[i]` or `b.values[i]`,
        // both in [0, q) by invariant.
        unsafe { Self::from_reduced_unchecked(a.modulus, out) }
    }
}

impl<const N: usize, M: Modulus, F: Form> Zeroize for Poly<N, M, F> {
    /// Wipe every lane. Leaves the modulus intact (public parameter).
    #[inline]
    fn zeroize(&mut self) {
        for v in &mut self.values {
            v.zeroize();
        }
    }
}

impl<const N: usize, M: Modulus, F: Form> Hash for Poly<N, M, F> {
    /// Hash the values, the modulus `q`, **and** a per-form discriminant
    /// byte so the same buffer in different forms hashes distinctly.
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.values.hash(state);
        self.modulus.q().hash(state);
        F::HASH_TAG.hash(state);
    }
}

impl<const N: usize, M: Modulus, F: Form> fmt::Debug for Poly<N, M, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Poly<{}, q={}, {:?}>({:?})",
            N,
            self.modulus.q(),
            F::default(),
            &self.values[..],
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

/// `compile_fail` doctest: mixing coefficient and evaluation forms is a
/// type error.
///
/// ```compile_fail
/// use via_rs::primitives::ring::element::Poly;
/// use via_rs::primitives::ring::form::{Coefficient, Evaluation};
/// use via_rs::primitives::zq::modulus::ConstModulus;
/// type M = ConstModulus<17>;
/// let c: Poly<4, M, Coefficient> = Poly::zero(M);
/// let e: Poly<4, M, Evaluation> = Poly::zero(M);
/// let _ = c + e;
/// ```
///
/// `compile_fail` doctest: `N` must be at least 2.
///
/// ```compile_fail
/// use via_rs::primitives::ring::element::Poly;
/// use via_rs::primitives::ring::form::Coefficient;
/// use via_rs::primitives::zq::modulus::ConstModulus;
/// let _: Poly<1, ConstModulus<17>, Coefficient> = Poly::zero(ConstModulus::<17>);
/// ```
///
/// `compile_fail` doctest: `N` must be a power of two.
///
/// ```compile_fail
/// use via_rs::primitives::ring::element::Poly;
/// use via_rs::primitives::ring::form::Coefficient;
/// use via_rs::primitives::zq::modulus::ConstModulus;
/// let _: Poly<3, ConstModulus<17>, Coefficient> = Poly::zero(ConstModulus::<17>);
/// ```
#[cfg(doctest)]
struct CompileFailDocs;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::primitives::zq::modulus::{ConstModulus, DynModulus, paper};

    type M17 = ConstModulus<17>;

    #[test]
    fn zero_one_at_paper_modulus() {
        let m = paper::ViaCQ3::default();
        let z: Poly<2048, _, Coefficient> = Poly::zero(m);
        let o: Poly<2048, _, Coefficient> = Poly::one(m);
        // zero: every coefficient is 0
        for i in 0..2048 {
            assert_eq!(z.coeff(i).to_u64(), 0);
        }
        // one: coeff(0) == 1, rest are zero
        assert_eq!(o.coeff(0).to_u64(), 1);
        for i in 1..2048 {
            assert_eq!(o.coeff(i).to_u64(), 0);
        }
    }

    #[test]
    fn new_reduces_each_lane() {
        let m = M17::default();
        let p: Poly<4, _, Coefficient> = Poly::new(m, [20, 0, 17, 16]);
        assert_eq!(p.values(), &[3u64, 0, 0, 16]);
    }

    #[test]
    fn ring_axiom_distributivity_small() {
        let m = M17::default();
        let a: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        let b: Poly<4, _, Coefficient> = Poly::new(m, [9, 1, 4, 13]);
        let c: Poly<4, _, Coefficient> = Poly::new(m, [2, 8, 6, 10]);
        assert_eq!((a + b) * c, (a * c) + (b * c));
    }

    /// `f * f * f * f` where `f = X` should give `-1` in coefficient `[0]`.
    /// X^4 = -1 in R_{4, 17} ⇒ [16, 0, 0, 0].
    #[test]
    fn x_pow_n_equals_neg_one() {
        let m = M17::default();
        let x: Poly<4, _, Coefficient> = Poly::new(m, [0, 1, 0, 0]);
        let x2 = x * x; // X^2
        let x4 = x2 * x2; // X^4 = -1
        assert_eq!(x4.values(), &[16u64, 0, 0, 0]);
    }

    /// Hand-computed wrap test:
    /// f = 1 + 2X + 3X^2 + 4X^3, g = 5 + 6X + 7X^2 + 8X^3, in R_{4, q}.
    /// fg = ? mod (X^4 + 1, q). We compute the reference and compare.
    #[test]
    fn wrap_specific_pair() {
        let q = 17u64;
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [1, 2, 3, 4]);
        let g: Poly<4, _, Coefficient> = Poly::new(m, [5, 6, 7, 8]);
        let got = f * g;
        // Reference: schoolbook with explicit -1 wrap at i+j >= 4.
        let lhs = [1i128, 2, 3, 4];
        let rhs = [5i128, 6, 7, 8];
        let mut acc = [0i128; 4];
        for i in 0..4 {
            for j in 0..4 {
                let p = lhs[i] * rhs[j];
                if i + j < 4 {
                    acc[i + j] += p;
                } else {
                    acc[i + j - 4] -= p;
                }
            }
        }
        let want: [u64; 4] = core::array::from_fn(|i| acc[i].rem_euclid(q as i128) as u64);
        assert_eq!(got.values(), &want);
    }

    #[test]
    fn mul_x_pow_k_zero_is_identity() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        assert_eq!(f.mul_x_pow(0), f);
    }

    #[test]
    fn mul_x_pow_k_lt_n() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        // X * f = [-7, 3, 5, 11] in coefficient form.
        let got = f.mul_x_pow(1);
        assert_eq!(got.values(), &[m.q() - 7, 3, 5, 11]);
    }

    #[test]
    fn mul_x_pow_k_eq_n() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        let got = f.mul_x_pow(4);
        // Equals negation: each coefficient -> -coeff mod q.
        for i in 0..4 {
            assert_eq!(got.coeff(i).to_u64(), m.neg(f.coeff(i).to_u64()));
        }
    }

    #[test]
    fn mul_x_pow_k_eq_2n_is_identity() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        assert_eq!(f.mul_x_pow(8), f);
    }

    #[test]
    fn mul_x_pow_via_repeated_x_mul() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        let x: Poly<4, _, Coefficient> = Poly::new(m, [0, 1, 0, 0]);
        for k in [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 13, 51] {
            let mut via_mul = f;
            for _ in 0..k {
                via_mul *= x;
            }
            let via_rot = f.mul_x_pow(k);
            assert_eq!(via_mul, via_rot, "k={k}");
        }
    }

    #[test]
    fn conditional_select_picks_b() {
        let m = M17::default();
        let a: Poly<4, _, Coefficient> = Poly::new(m, [1, 2, 3, 4]);
        let b: Poly<4, _, Coefficient> = Poly::new(m, [10, 11, 12, 13]);
        let pick_a = Poly::conditional_select(&a, &b, Choice::from(0));
        let pick_b = Poly::conditional_select(&a, &b, Choice::from(1));
        assert_eq!(pick_a, a);
        assert_eq!(pick_b, b);
    }

    #[test]
    fn zeroize_clears_values() {
        let m = M17::default();
        let mut p: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        p.zeroize();
        assert_eq!(p.values(), &[0u64; 4]);
    }

    #[test]
    fn hash_distinguishes_modulus_and_form() {
        use core::hash::BuildHasher;
        use core::hash::BuildHasherDefault;
        use core::hash::Hasher;
        // `Hasher` brought into scope so the FxHasher impl below resolves.

        // Use the std-free SipHasher core impl indirectly via a small
        // fnv-like rolling hasher to keep this test no_std-friendly.
        // We just need *some* hasher that depends on input order.
        struct FxHasher(u64);
        impl Default for FxHasher {
            fn default() -> Self {
                FxHasher(0xcbf2_9ce4_8422_2325)
            }
        }
        impl Hasher for FxHasher {
            fn finish(&self) -> u64 {
                self.0
            }
            fn write(&mut self, bytes: &[u8]) {
                for &b in bytes {
                    self.0 = self.0.wrapping_mul(0x0100_0000_01b3) ^ (b as u64);
                }
            }
        }
        let bh = BuildHasherDefault::<FxHasher>::default();

        let m = M17::default();
        let coeff: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        let eval: Poly<4, _, Evaluation> = Poly::new(m, [3, 5, 11, 7]);
        // Same buffer, different form ⇒ different hash.
        assert_ne!(bh.hash_one(coeff), bh.hash_one(eval));

        // Different modulus ⇒ different hash.
        let m2 = DynModulus::new(19);
        let coeff2: Poly<4, _, Coefficient> = Poly::new(m2, [3, 5, 11, 7]);
        assert_ne!(bh.hash_one(coeff), bh.hash_one(coeff2));
    }

    #[test]
    #[should_panic(expected = "modulus mismatch")]
    fn add_panics_on_modulus_mismatch() {
        let m17 = DynModulus::new(17);
        let m19 = DynModulus::new(19);
        let a: Poly<4, _, Coefficient> = Poly::new(m17, [1, 2, 3, 4]);
        let b: Poly<4, _, Coefficient> = Poly::new(m19, [1, 2, 3, 4]);
        let _ = a + b;
    }

    /// Cross-type modulus mismatch is statically prevented (`Poly<_,
    /// M17, _>::set_coeff` only accepts `Zq<M17>`). The runtime assert
    /// fires when both polynomial and value share the *type*
    /// `DynModulus` but carry different `q` values.
    #[test]
    #[should_panic(expected = "modulus mismatch")]
    fn set_coeff_panics_on_runtime_modulus_mismatch() {
        let m17 = DynModulus::new(17);
        let m19 = DynModulus::new(19);
        let mut p: Poly<4, DynModulus, Coefficient> = Poly::new(m17, [0, 0, 0, 0]);
        let mismatch = Zq::new(m19, 5);
        p.set_coeff(0, mismatch);
    }

    #[test]
    fn eval_form_add_sub_neg_on_zero_polys() {
        let m = M17::default();
        let z: Poly<4, _, Evaluation> = Poly::zero(m);
        let s: Poly<4, _, Evaluation> = Poly::new(m, [1, 2, 3, 4]);
        assert_eq!(z + s, s);
        assert_eq!(s - z, s);
        assert_eq!(-z, z);
    }

    #[test]
    fn eval_pointwise_mul_against_zero_yields_zero() {
        let m = M17::default();
        let z: Poly<4, _, Evaluation> = Poly::zero(m);
        let s: Poly<4, _, Evaluation> = Poly::new(m, [1, 2, 3, 4]);
        let p = z * s;
        assert_eq!(p, z);
    }

    #[test]
    fn eval_pointwise_mul_is_lanewise() {
        let m = M17::default();
        let a: Poly<4, _, Evaluation> = Poly::new(m, [1, 2, 3, 4]);
        let b: Poly<4, _, Evaluation> = Poly::new(m, [5, 6, 7, 8]);
        let p = a * b;
        let q = m.q();
        let want: [u64; 4] = core::array::from_fn(|i| (a.values()[i] * b.values()[i]) % q);
        assert_eq!(p.values(), &want);
    }

    #[test]
    fn eval_zero_round_trip_through_coeff_is_zero() {
        // §0.4: with the NTT body wired in, zero round-trips to zero.
        let m = M17::default();
        let z: Poly<4, _, Evaluation> = Poly::zero(m);
        let back = z.into_coeff();
        assert_eq!(back, Poly::<4, _, Coefficient>::zero(m));
    }

    /// NTT round-trip identity: `f.into_eval().into_coeff() == f`.
    #[test]
    fn ntt_roundtrip_identity_n4_q17() {
        let m = M17::default();
        for input in [
            [0u64, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [3, 5, 11, 7],
            [16, 16, 16, 16],
        ] {
            let f: Poly<4, _, Coefficient> = Poly::new(m, input);
            let back = f.into_eval().into_coeff();
            assert_eq!(back, f, "input={input:?}");
        }
    }

    /// NTT homomorphism: schoolbook coefficient-form `f * g` equals
    /// `(f.into_eval() * g.into_eval()).into_coeff()` (pointwise mul in
    /// eval form is ring mul in coefficient form).
    #[test]
    fn ntt_homomorphism_n4_q17() {
        let m = M17::default();
        let cases: &[([u64; 4], [u64; 4])] = &[
            ([1, 0, 0, 0], [3, 5, 11, 7]),
            ([0, 1, 0, 0], [1, 2, 3, 4]),
            ([5, 11, 3, 7], [9, 1, 4, 13]),
            ([16, 16, 16, 16], [1, 1, 1, 1]),
            ([1, 2, 3, 4], [5, 6, 7, 8]),
        ];
        for (a, b) in cases {
            let f: Poly<4, _, Coefficient> = Poly::new(m, *a);
            let g: Poly<4, _, Coefficient> = Poly::new(m, *b);
            let schoolbook = f * g;
            let ntt_mediated = (f.into_eval() * g.into_eval()).into_coeff();
            assert_eq!(ntt_mediated, schoolbook, "a={a:?}, b={b:?}");
        }
    }

    /// Linearity: `(f + g).into_eval() == f.into_eval() + g.into_eval()`.
    #[test]
    fn ntt_linearity_n4_q17() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [1, 2, 3, 4]);
        let g: Poly<4, _, Coefficient> = Poly::new(m, [5, 11, 3, 7]);
        let lhs = (f + g).into_eval();
        let rhs = f.into_eval() + g.into_eval();
        assert_eq!(lhs, rhs);
    }

    /// The constant polynomial $1$ evaluates to $1$ at every NTT point
    /// regardless of order — `into_eval` produces `[1, 1, 1, 1]`.
    #[test]
    fn ntt_one_is_all_ones_in_eval() {
        let m = M17::default();
        let one: Poly<4, _, Coefficient> = Poly::one(m);
        let e = one.into_eval();
        assert_eq!(e.values(), &[1u64, 1, 1, 1]);
    }

    /// `mul_x_pow(k)` (coefficient-form rotation) agrees with
    /// pointwise-NTT multiplication by `X^k`'s NTT.
    #[test]
    fn ntt_mul_x_pow_via_homomorphism() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        for k in [0usize, 1, 2, 3, 4, 5, 7, 12] {
            let mut x_k_coeffs = [0u64; 4];
            let k_eff = k % 8;
            let k_red = k_eff % 4;
            let neg = k_eff >= 4;
            x_k_coeffs[k_red] = if neg { m.q() - 1 } else { 1 };
            let x_k: Poly<4, _, Coefficient> = Poly::new(m, x_k_coeffs);
            let via_rot = f.mul_x_pow(k);
            let via_ntt = (f.into_eval() * x_k.into_eval()).into_coeff();
            assert_eq!(via_rot, via_ntt, "k={k}");
        }
    }

    /// `compile_fail` doctest: `into_eval` on `PowerOfTwoModulus` fails
    /// — there is no `NttFriendly` impl for that modulus type.
    ///
    /// ```compile_fail
    /// use via_rs::primitives::ring::element::Poly;
    /// use via_rs::primitives::ring::form::Coefficient;
    /// use via_rs::primitives::zq::modulus::PowerOfTwoModulus;
    /// type M = PowerOfTwoModulus<4>;
    /// let p: Poly<4, M, Coefficient> = Poly::zero(M);
    /// let _ = p.into_eval();
    /// ```
    #[cfg(doctest)]
    struct IntoEvalRequiresNttFriendlyDocs;

    #[test]
    fn scalar_mul_reduces() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [1, 2, 3, 4]);
        let got = f * 7u64;
        let q = m.q();
        let want: [u64; 4] = core::array::from_fn(|i| (((i as u64) + 1) * 7) % q);
        assert_eq!(got.values(), &want);
    }

    #[test]
    fn from_reduced_unchecked_round_trips_via_new() {
        let m = M17::default();
        // SAFETY: every lane is in [0, q).
        let p: Poly<4, _, Coefficient> = unsafe { Poly::from_reduced_unchecked(m, [3, 5, 11, 7]) };
        let q: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        assert_eq!(p, q);
    }

    /// `Poly::random` uniformity smoke test — chi-squared at q=17 across
    /// `N * 10_000` samples flattened into a single histogram. Reuses the
    /// SplitMix64 helper pattern from `zq::element::tests`.
    #[test]
    fn random_uniformity_chi_squared() {
        let m = M17::default();
        let mut rng = SplitMix64::new(0x9CAFEF00D);
        let mut counts = [0u64; 17];
        const N_POLYS: usize = 1_000;
        for _ in 0..N_POLYS {
            let p: Poly<8, _, Coefficient> = Poly::random(m, &mut rng);
            for &v in p.values() {
                counts[v as usize] += 1;
            }
        }
        let total = (N_POLYS * 8) as f64;
        let expected = total / 17.0;
        let chi2: f64 = counts
            .iter()
            .map(|&o| {
                let d = o as f64 - expected;
                d * d / expected
            })
            .sum();
        // 99% threshold for χ² with 16 d.f. is ~32; allow generous 60 to
        // avoid flakiness while still catching gross bias.
        assert!(chi2 < 60.0, "chi^2 = {chi2}, counts = {counts:?}");
    }

    /// SplitMix64 — small PRG used in uniformity tests. Mirrors the
    /// helper duplicated in `zq/element.rs` / `rns/element.rs`.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
    }

    impl RngCore for SplitMix64 {
        fn next_u32(&mut self) -> u32 {
            self.next_u64() as u32
        }
        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn fill_bytes(&mut self, dst: &mut [u8]) {
            for chunk in dst.chunks_mut(8) {
                let bytes = self.next_u64().to_le_bytes();
                chunk.copy_from_slice(&bytes[..chunk.len()]);
            }
        }
        fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), rand_core::Error> {
            self.fill_bytes(dst);
            Ok(())
        }
    }

    // ----- §0.5 ring embedding / projection tests -----

    /// `f.embed_at::<N_LARGE>(j).project_at::<N>(j) == f` for every
    /// slot, single-prime case at small (N, N_LARGE).
    #[test]
    fn poly_embed_project_roundtrip_n4_into_n16() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        for j in 0..4usize {
            let big: Poly<16, _, Coefficient> = f.embed_at::<16>(j);
            let back: Poly<4, _, Coefficient> = big.project_at::<4>(j);
            assert_eq!(back, f, "j={j}");
        }
    }

    /// Slot disjointness at the `Poly` API.
    #[test]
    fn poly_project_at_other_slot_is_zero() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        let zero: Poly<4, _, Coefficient> = Poly::zero(m);
        for j in 0..4usize {
            let big: Poly<16, _, Coefficient> = f.embed_at::<16>(j);
            for jp in 0..4usize {
                if jp == j {
                    continue;
                }
                let back: Poly<4, _, Coefficient> = big.project_at::<4>(jp);
                assert_eq!(back, zero, "embed j={j}, project jp={jp}");
            }
        }
    }

    /// $\iota_0$ is a ring homomorphism: $\iota_0(f g) = \iota_0(f) \cdot
    /// \iota_0(g)$ where both multiplications are schoolbook negacyclic
    /// in their respective rings. This is the key correctness property
    /// distinguishing $\iota_0$ from $\iota_j$ for $j > 0$.
    #[test]
    fn poly_iota_zero_is_ring_homomorphism_n4_into_n8() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        let g: Poly<4, _, Coefficient> = Poly::new(m, [9, 1, 4, 13]);
        let prod_small = f * g;
        // Embed both factors at slot 0 of N=8 ring, multiply, project back.
        let fe: Poly<8, _, Coefficient> = f.embed_at::<8>(0);
        let ge: Poly<8, _, Coefficient> = g.embed_at::<8>(0);
        let prod_large = fe * ge;
        let projected_back: Poly<4, _, Coefficient> = prod_large.project_at::<4>(0);
        assert_eq!(projected_back, prod_small);
    }

    /// Same homomorphism check at a non-trivial (4, 16) ratio (d = 4).
    #[test]
    fn poly_iota_zero_homomorphism_n4_into_n16() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [3, 5, 11, 7]);
        let g: Poly<4, _, Coefficient> = Poly::new(m, [9, 1, 4, 13]);
        let prod_small = f * g;
        let fe: Poly<16, _, Coefficient> = f.embed_at::<16>(0);
        let ge: Poly<16, _, Coefficient> = g.embed_at::<16>(0);
        let prod_large = fe * ge;
        let projected_back = prod_large.project_at::<4>(0);
        assert_eq!(projected_back, prod_small);
    }

    /// $d$-fold round-trip: pack then unpack recovers the slot polys.
    #[test]
    fn poly_pack_unpack_identity_n4_into_n16() {
        let m = M17::default();
        let slots: [Poly<4, _, Coefficient>; 4] = [
            Poly::new(m, [1, 2, 3, 4]),
            Poly::new(m, [5, 6, 7, 8]),
            Poly::new(m, [9, 10, 11, 12]),
            Poly::new(m, [13, 14, 15, 16]),
        ];
        let packed: Poly<16, _, Coefficient> = Poly::pack_slots::<16>(m, &slots);
        let mut back: [Poly<4, _, Coefficient>; 4] = [Poly::zero(m); 4];
        packed.unpack_slots::<4>(&mut back);
        for j in 0..4 {
            assert_eq!(back[j], slots[j], "slot j={j}");
        }
    }

    /// Pack agrees with the per-slot embed-and-sum reference.
    #[test]
    fn poly_pack_slots_matches_per_slot_embed() {
        let m = M17::default();
        let slots: [Poly<4, _, Coefficient>; 4] = [
            Poly::new(m, [1, 2, 3, 4]),
            Poly::new(m, [5, 6, 7, 8]),
            Poly::new(m, [9, 10, 11, 12]),
            Poly::new(m, [13, 14, 15, 16]),
        ];
        let via_pack: Poly<16, _, Coefficient> = Poly::pack_slots::<16>(m, &slots);
        // Reference: sum slots[j].embed_at::<16>(j) for j in 0..4.
        let mut via_sum: Poly<16, _, Coefficient> = Poly::zero(m);
        for (j, slot_poly) in slots.iter().enumerate() {
            via_sum += slot_poly.embed_at::<16>(j);
        }
        assert_eq!(via_pack, via_sum);
    }

    /// Round-trip at realistic paper sizes $N_\text{small} = 512$,
    /// $N_\text{large} = 2048$, modulus `ViaCQ3`. Locks the realistic
    /// path.
    #[test]
    fn poly_embed_project_paper_n2_into_n1() {
        let m = paper::ViaCQ3::default();
        let q = m.q();
        let mut buf = [0u64; 512];
        for (i, v) in buf.iter_mut().enumerate() {
            *v = (i as u64 * 123_456_789 + 7) % q;
        }
        let f: Poly<512, _, Coefficient> = Poly::new(m, buf);
        let big: Poly<2048, _, Coefficient> = f.embed_at::<2048>(0);
        let back: Poly<512, _, Coefficient> = big.project_at::<512>(0);
        assert_eq!(back, f);
    }

    /// Slot-index runtime panic on `slot >= d`.
    #[test]
    #[should_panic(expected = "slot")]
    fn poly_embed_at_panics_on_out_of_range_slot() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [0, 0, 0, 0]);
        let _ = f.embed_at::<8>(2); // d = 2; slot < 2 required
    }

    /// Modulus-mismatch runtime panic in `pack_slots`.
    #[test]
    #[should_panic(expected = "modulus mismatch")]
    fn poly_pack_slots_panics_on_modulus_mismatch() {
        let m17 = DynModulus::new(17);
        let m19 = DynModulus::new(19);
        let slots: [Poly<4, DynModulus, Coefficient>; 2] =
            [Poly::new(m17, [0, 0, 0, 0]), Poly::new(m19, [0, 0, 0, 0])];
        // N = 4 (source slot degree), N_LARGE = 8 (target ring degree),
        // d = 2 slots. Mismatch between slot 0's m17 and the m19 in slot 1.
        let _ = Poly::<4, DynModulus, Coefficient>::pack_slots::<8>(m17, &slots);
    }

    // ----- §0.6 centred-coeffs tests -----

    /// `to_centered_coeffs` at a small modulus produces the expected
    /// per-coefficient centred values.
    #[test]
    fn poly_to_centered_coeffs_small() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [0, 8, 9, 16]);
        let mut dst = [0i64; 4];
        f.to_centered_coeffs(&mut dst);
        assert_eq!(dst, [0i64, 8, -8, -1]);
    }

    /// CT centred-coeffs match the non-CT version pointwise.
    #[test]
    fn poly_to_centered_coeffs_ct_matches_non_ct() {
        let m = M17::default();
        let f: Poly<4, _, Coefficient> = Poly::new(m, [0, 8, 9, 16]);
        let mut non_ct = [0i64; 4];
        let mut ct = [0i64; 4];
        f.to_centered_coeffs(&mut non_ct);
        f.to_centered_coeffs_ct(&mut ct);
        assert_eq!(non_ct, ct);
    }

    /// Round-trip identity: `to_centered_coeffs` then re-reduce via
    /// `Modulus::reduce_i64` recovers the original `Poly`.
    #[test]
    fn poly_to_centered_coeffs_roundtrip() {
        let m = paper::ViaCQ3::default();
        let f: Poly<8, _, Coefficient> = Poly::new(
            m,
            [
                0,
                1,
                m.q() / 2,
                m.q() / 2 + 1,
                m.q() - 1,
                12345,
                m.q() - 12345,
                7,
            ],
        );
        let mut centred = [0i64; 8];
        f.to_centered_coeffs(&mut centred);
        let mut back = [0u64; 8];
        for (b, &c) in back.iter_mut().zip(centred.iter()) {
            *b = m.reduce_i64(c);
        }
        let back_poly: Poly<8, _, Coefficient> = Poly::new(m, back);
        assert_eq!(back_poly, f);
    }
}

/// `compile_fail` doctests for §0.5 degree-relationship checks.
///
/// `N_LARGE` smaller than `N`:
///
/// ```compile_fail
/// use via_rs::primitives::ring::element::Poly;
/// use via_rs::primitives::ring::form::Coefficient;
/// use via_rs::primitives::zq::modulus::ConstModulus;
/// type M = ConstModulus<17>;
/// let f: Poly<8, M, Coefficient> = Poly::zero(M);
/// let _ = f.embed_at::<4>(0);
/// ```
///
/// `N_LARGE` not a multiple of `N`:
///
/// ```compile_fail
/// use via_rs::primitives::ring::element::Poly;
/// use via_rs::primitives::ring::form::Coefficient;
/// use via_rs::primitives::zq::modulus::ConstModulus;
/// type M = ConstModulus<17>;
/// let f: Poly<4, M, Coefficient> = Poly::zero(M);
/// let _ = f.embed_at::<6>(0);
/// ```
///
/// `N_LARGE` not a power of two:
///
/// ```compile_fail
/// use via_rs::primitives::ring::element::Poly;
/// use via_rs::primitives::ring::form::Coefficient;
/// use via_rs::primitives::zq::modulus::ConstModulus;
/// type M = ConstModulus<17>;
/// let f: Poly<4, M, Coefficient> = Poly::zero(M);
/// let _ = f.embed_at::<12>(0);
/// ```
#[cfg(doctest)]
struct ReshapeCompileFailDocs;
