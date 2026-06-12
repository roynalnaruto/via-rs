//! Negacyclic NTT transforms.
//!
//! The negacyclic NTT for $R_{n, q} = \mathbb{Z}_q\lbrack X\rbrack / (X^n + 1)$
//! maps a polynomial in coefficient form to its evaluations at the $n$
//! primitive $2n$-th roots of unity in $\mathbb{Z}_q$. Pointwise
//! multiplication in evaluation form then equals $R_{n, q}$ multiplication
//! by the NTT bijection. The transform requires $q \equiv 1 \pmod{2n}$;
//! moduli that satisfy this admit a primitive $2n$-th root $\psi$, and the
//! odd powers $\{\psi^{2i + 1}\}_{i = 0}^{n - 1}$ are the negacyclic
//! evaluation points.
//!
//! ## API
//!
//! - The [`NttFriendly`] trait (parameterised by `const N: usize`, extending
//!   [`Modulus`]) gates which moduli
//!   admit an NTT at degree $N$. Const items on the trait carry the
//!   primitive root $\psi$, the inverse scaling factor $N^{-1} \bmod q$,
//!   and the two bit-reversed twiddle tables (forward and inverse).
//! - [`ConstModulus<Q>`] auto-impls `NttFriendly<N>` for every $(Q, N)$
//!   pair satisfying $Q \equiv 1 \pmod{2N}$. A compile-time `_CHECK_NTT`
//!   block forces validation at monomorphisation; invalid pairs fail to
//!   build. **No `DynModulus` impl** — runtime NTT contexts are deferred.
//! - The kernels `ntt_inplace` (forward) and `intt_inplace` (inverse)
//!   are `pub(crate)` adapters consumed by [`super::element::Poly::into_eval`]
//!   / `into_coeff` and the [`super::rns_element::PolyRns`] analogues.
//!
//! ## Algorithm
//!
//! **Iterative Cooley–Tukey, radix-2**, per Longa–Naehrig 2016.
//!
//! - Forward: **decimation-in-time** (DIT), $\log_2 N$ in-place butterfly
//!   stages. Input in natural order; output in **bit-reversed** order.
//! - Inverse: **decimation-in-frequency** (DIF), $\log_2 N$ in-place
//!   butterfly stages. Consumes inputs in bit-reversed order (matching
//!   forward's output) and produces outputs in natural order. A final
//!   pass scales each lane by $N^{-1} \bmod q$.
//!
//! Bit-reversed eval-form storage saves two $O(N)$ permutation passes per
//! round-trip. Pointwise multiplication and additive ring operations are
//! order-agnostic, so the convention is invisible to the protocol layers;
//! only `Poly::eval(i)` semantically returns the value at the bit-reversed
//! index (documented at the call site).
//!
//! ## Constant-time
//!
//! The butterfly inner loop is data-independent over the operand values
//! (every stage runs the same `Modulus::mul` / `add` / `sub` sequence
//! regardless of contents). The twiddle access pattern depends only on
//! the public ring degree $N$. There are no early exits.

use crate::algebra::rns::reduce::mod_inverse_u64;
use crate::algebra::zq::modulus::{ConstModulus, Modulus};

// ---------------------------------------------------------------------------
// Const-fn primitives — used by the `ConstModulus<Q>` trait impl below
// ---------------------------------------------------------------------------

/// Modular exponentiation $\mathrm{base}^{\mathrm{exp}} \bmod q$,
/// `const fn` for use in trait-const-item evaluation.
///
/// Intermediates use `u128`; safe for any `q < 2^64`.
#[inline]
const fn mod_pow(base: u64, mut exp: u64, q: u64) -> u64 {
    let q128 = q as u128;
    let mut result: u128 = 1 % q128;
    let mut b: u128 = (base as u128) % q128;
    while exp > 0 {
        if exp & 1 == 1 {
            result = (result * b) % q128;
        }
        b = (b * b) % q128;
        exp >>= 1;
    }
    result as u64
}

/// $\lfloor \log_2 x \rfloor$ for `x` a positive power of two.
#[inline]
const fn log2_pow2(x: u64) -> u32 {
    debug_assert_pow2_u64(x);
    x.trailing_zeros()
}

/// Const-fn `debug_assert!` proxy — `trailing_zeros` is already well-defined
/// for non-pow2 inputs, but we want a tighter contract.
#[inline]
const fn debug_assert_pow2_u64(x: u64) {
    assert!(
        x > 0 && (x & (x - 1)) == 0,
        "expected positive power of two"
    );
}

/// Bit-reverse `x` over the low `log_n` bits.
#[inline]
const fn bit_reverse(x: usize, log_n: u32) -> usize {
    let mut r: usize = 0;
    let mut i: u32 = 0;
    while i < log_n {
        if (x >> i) & 1 != 0 {
            r |= 1 << (log_n - 1 - i);
        }
        i += 1;
    }
    r
}

/// Find a primitive $2N$-th root of unity in $\mathbb{Z}_q$ — `const fn`.
///
/// Strategy: try each base
/// $g = 2, 3, \ldots$; compute $\mathrm{candidate} = g^{(q - 1) / 2N}
/// \bmod q$; verify the candidate has order **exactly** $2N$ by
/// squaring $\log_2(2N)$ times — every intermediate power must be
/// $\ne 1$ (rules out a proper divisor of $2N$ being the order) and
/// the final power must be $1$ (confirms the order divides $2N$, which
/// is already guaranteed by Lagrange for `candidate = base^((q-1)/2N)`
/// but is asserted as a defence-in-depth).
///
/// # Search cost
///
/// For NTT-friendly $q$ (i.e. $q \equiv 1 \pmod{2N}$), the map
/// $g \mapsto g^{(q-1)/2N}$ sends $\mathbb{Z}_q^*$ onto the $2N$-th
/// roots of unity in $\mathbb{Z}_q$. Among those roots, $\varphi(2N) = N$
/// have order exactly $2N$, so a *uniformly random* base hits a
/// primitive root with probability $\tfrac{1}{2}$. We don't sample
/// uniformly — we try $g = 2, 3, \ldots$ in order — but in practice
/// every realistic NTT-friendly prime succeeds within the
/// first few trials.
///
/// # Search cap
///
/// The loop is capped at `MAX_BASE_TRIES` candidate bases (1024 today,
/// chosen large enough for any realistic NTT-friendly prime but small
/// enough that const-evaluation terminates in milliseconds). If the cap
/// is exhausted, the function panics at monomorphisation. Without this
/// cap, a pathological $q$ where all small bases happen to land in the
/// order-$N$ subgroup would force the search to iterate up to $q$
/// itself — at paper-scale $q \approx 2^{38}$, that is billions of
/// const-eval iterations and would either hang the compiler or hit a
/// `--cap-fn-recursion-limit` analogue.
///
/// If a `ConstModulus<Q>: NttFriendly<N>` instantiation ever trips the
/// cap, the right fix is to inspect $Q$ — either it isn't actually
/// NTT-friendly (the `_CHECK_NTT` precondition is supposed to catch
/// this), or it's a degenerate case that warrants raising
/// `MAX_BASE_TRIES`. Don't silently bump the cap without understanding
/// why the prior bound failed.
///
/// # Panics
///
/// Panics if no primitive $2N$-th root is found within `MAX_BASE_TRIES`
/// trials. Caller is responsible for asserting $q \equiv 1 \pmod{2N}$
/// before invocation (`check_ntt` does this).
#[inline]
const fn find_primitive_2n_th_root(q: u64, two_n: u64) -> u64 {
    /// Upper bound on the number of candidate bases. See the function-
    /// level doc for the rationale.
    const MAX_BASE_TRIES: u64 = 1024;

    assert!(q >= 2, "find_primitive_2n_th_root: q >= 2");
    debug_assert_pow2_u64(two_n);
    assert!(
        (q - 1).is_multiple_of(two_n),
        "find_primitive_2n_th_root: q != 1 (mod 2N)",
    );
    let quotient = (q - 1) / two_n;
    let log_two_n = log2_pow2(two_n);
    let cap = if q < MAX_BASE_TRIES + 2 {
        q
    } else {
        MAX_BASE_TRIES + 2
    };
    let mut base: u64 = 2;
    while base < cap {
        let candidate = mod_pow(base, quotient, q);
        // Verify order is exactly two_n: every intermediate power must
        // be != 1, and candidate^two_n must be 1.
        let mut curr = candidate;
        let mut valid = true;
        let mut i: u32 = 0;
        while i < log_two_n {
            if curr == 1 {
                valid = false;
                break;
            }
            // Square: curr = curr^2 mod q.
            curr = mod_pow(curr, 2, q);
            i += 1;
        }
        if valid && curr == 1 {
            return candidate;
        }
        base += 1;
    }
    panic!(
        "find_primitive_2n_th_root: cap exhausted; check that q ≡ 1 (mod 2N), or raise MAX_BASE_TRIES if this prime is genuinely pathological",
    );
}

/// Build the bit-reversed table of `root^k mod q` for `k in 0..N`.
///
/// `table[bit_reverse(k, log2(N))] = root^k mod q`. Used to precompute
/// both forward (`root = psi`) and inverse (`root = psi^{-1}`) twiddles
/// at compile time.
#[inline]
const fn build_twiddle_table_bit_reversed<const N: usize>(q: u64, root: u64) -> [u64; N] {
    assert!(N >= 2, "twiddle table: N >= 2");
    debug_assert_pow2_u64(N as u64);
    let log_n = log2_pow2(N as u64);
    let q128 = q as u128;
    let r = (root as u128) % q128;
    let mut table = [0u64; N];
    let mut cur: u128 = 1 % q128;
    let mut k: usize = 0;
    while k < N {
        table[bit_reverse(k, log_n)] = cur as u64;
        cur = (cur * r) % q128;
        k += 1;
    }
    table
}

// ---------------------------------------------------------------------------
// `NttFriendly<const N>` trait — gates which moduli admit an NTT at N
// ---------------------------------------------------------------------------

/// A modulus that admits a negacyclic NTT at ring degree $N$.
///
/// Implementations carry the primitive $2N$-th root of unity, the
/// inverse-NTT scaling factor $N^{-1} \bmod q$, and the two bit-reversed
/// twiddle tables. For [`ConstModulus<Q>`] these are `const`-evaluated at
/// monomorphisation; a $(Q, N)$ pair failing the NTT-friendliness check
/// $(Q - 1) \bmod 2N \ne 0$ fails to compile.
///
/// # Storage
///
/// $2N$ `u64`s per implementer ($\approx 32$ KiB at $N = 2048$), placed
/// in the binary's read-only data section. The two scalar constants
/// ($\psi$, $N^{-1}$) round out the impl.
pub trait NttFriendly<const N: usize>: Modulus {
    /// A primitive $2N$-th root of unity in $\mathbb{Z}_q$.
    const PSI: u64;
    /// $N^{-1} \bmod q$ — the inverse-NTT scaling factor.
    const N_INV: u64;
    /// Bit-reversed table of $\psi^k$ powers, $k \in [0, N)$.
    const TWIDDLES_FORWARD: [u64; N];
    /// Bit-reversed table of $\psi^{-k}$ powers, $k \in [0, N)$.
    const TWIDDLES_INVERSE: [u64; N];
}

impl<const Q: u64, const N: usize> NttFriendly<N> for ConstModulus<Q> {
    const PSI: u64 = {
        let () = check_ntt::<Q, N>();
        find_primitive_2n_th_root(Q, 2 * N as u64)
    };
    const N_INV: u64 = {
        let () = check_ntt::<Q, N>();
        mod_inverse_u64(N as u64 % Q, Q)
    };
    const TWIDDLES_FORWARD: [u64; N] = {
        // Re-derive PSI here rather than going through `Self::PSI` to
        // avoid the trait-resolution ambiguity that bites when N is not
        // pinned by the enclosing reference.
        let psi = find_primitive_2n_th_root(Q, 2 * N as u64);
        build_twiddle_table_bit_reversed::<N>(Q, psi)
    };
    const TWIDDLES_INVERSE: [u64; N] = {
        let psi = find_primitive_2n_th_root(Q, 2 * N as u64);
        let psi_inv = mod_inverse_u64(psi, Q);
        build_twiddle_table_bit_reversed::<N>(Q, psi_inv)
    };
}

/// Compile-time validation for [`NttFriendly`] on [`ConstModulus<Q>`].
///
/// Asserts $N \ge 2$, $N$ is a power of two, and $Q \equiv 1 \pmod{2N}$.
/// Touched from every `const` item in the impl block so any violation
/// fails at monomorphisation.
#[inline]
const fn check_ntt<const Q: u64, const N: usize>() {
    assert!(N >= 2, "NttFriendly: N >= 2");
    debug_assert_pow2_u64(N as u64);
    assert!(Q >= 2, "NttFriendly: Q >= 2");
    assert!(
        (Q - 1).is_multiple_of(2 * N as u64),
        "NttFriendly: Q must satisfy Q ≡ 1 (mod 2N)",
    );
}

// ---------------------------------------------------------------------------
// Cooley–Tukey kernels (free functions, generic over `M: Modulus`)
// ---------------------------------------------------------------------------

/// Iterative Cooley–Tukey decimation-in-time forward NTT, in place.
///
/// Consumes `buf` in natural order and overwrites it with the negacyclic
/// NTT evaluations in bit-reversed order. `twiddles[k]` must be
/// $\psi^{\mathrm{br}(k)}$ where $\psi$ is a primitive $2 \cdot \mathrm{buf.len}()$-th
/// root of unity.
///
/// # Panics
///
/// Panics if `buf.len() != twiddles.len()` or `buf.len()` is not a
/// power of two.
pub(crate) fn cooley_tukey_dit_forward<M: Modulus>(m: M, buf: &mut [u64], twiddles: &[u64]) {
    let n = buf.len();
    assert_eq!(n, twiddles.len(), "ntt: buf/twiddles length mismatch");
    assert!(
        n.is_power_of_two() && n >= 2,
        "ntt: buf length power of two"
    );
    // Debug-only invariant sweep: every lane must already be in
    // canonical `[0, q)` form. The wrapper layer (`Poly::into_eval` etc.)
    // enforces this via the `Poly` type invariant, so callers going
    // through the polynomial API never trigger this. The check fires
    // for direct GPU / SIMD adapter callers that may bypass the
    // wrapper. Zero release-build cost — the entire loop is
    // `cfg(debug_assertions)`-gated.
    #[cfg(debug_assertions)]
    for (i, &v) in buf.iter().enumerate() {
        debug_assert!(
            v < m.q(),
            "ntt: input lane {i} = {v} not in [0, q={})",
            m.q(),
        );
    }
    let log_n = n.trailing_zeros();
    for round in 0..log_n {
        let block_count: usize = 1 << round;
        let half_stride: usize = n >> (1 + round);
        let stride = 2 * half_stride;
        for block in 0..block_count {
            let w = twiddles[block_count + block];
            let base = block * stride;
            for j in base..(base + half_stride) {
                let x = buf[j];
                let y = m.mul(buf[j + half_stride], w);
                buf[j] = m.add(x, y);
                buf[j + half_stride] = m.sub(x, y);
            }
        }
    }
}

/// Iterative Cooley–Tukey decimation-in-frequency inverse NTT, in place.
///
/// Consumes `buf` in bit-reversed order (matching the forward NTT's
/// output) and produces outputs in natural order. `twiddles[k]` must be
/// $\psi^{-\mathrm{br}(k)}$. A final pass scales every lane by `n_inv`
/// ($N^{-1} \bmod q$).
///
/// # Panics
///
/// Panics if `buf.len() != twiddles.len()` or `buf.len()` is not a
/// power of two.
pub(crate) fn cooley_tukey_dif_inverse<M: Modulus>(
    m: M,
    buf: &mut [u64],
    twiddles_inv: &[u64],
    n_inv: u64,
) {
    let n = buf.len();
    assert_eq!(n, twiddles_inv.len(), "intt: buf/twiddles length mismatch");
    assert!(
        n.is_power_of_two() && n >= 2,
        "intt: buf length power of two"
    );
    // Debug-only invariant sweep — see the matching note on
    // [`cooley_tukey_dit_forward`].
    #[cfg(debug_assertions)]
    for (i, &v) in buf.iter().enumerate() {
        debug_assert!(
            v < m.q(),
            "intt: input lane {i} = {v} not in [0, q={})",
            m.q(),
        );
    }
    debug_assert!(n_inv < m.q(), "intt: n_inv >= q");
    let log_n = n.trailing_zeros();
    for round in 0..log_n {
        let block_count: usize = n >> (1 + round);
        let half_stride: usize = 1 << round;
        let stride = 2 * half_stride;
        for block in 0..block_count {
            let w = twiddles_inv[block_count + block];
            let base = block * stride;
            for j in base..(base + half_stride) {
                let x = buf[j];
                let y = buf[j + half_stride];
                buf[j] = m.add(x, y);
                buf[j + half_stride] = m.mul(m.sub(x, y), w);
            }
        }
    }
    // Final 1/N scaling pass.
    for v in buf.iter_mut() {
        *v = m.mul(*v, n_inv);
    }
}

// ---------------------------------------------------------------------------
// `pub(crate)` adapters consumed by `Poly` / `PolyRns`
// ---------------------------------------------------------------------------

/// Forward negacyclic NTT — coefficient form → evaluation form, in place.
///
/// Reads the precomputed forward twiddles from `M::TWIDDLES_FORWARD`.
#[inline]
pub(crate) fn ntt_inplace<const N: usize, M: NttFriendly<N>>(m: M, buf: &mut [u64; N]) {
    cooley_tukey_dit_forward(m, buf.as_mut_slice(), M::TWIDDLES_FORWARD.as_slice());
}

/// Inverse negacyclic NTT — evaluation form → coefficient form, in place.
///
/// Reads the precomputed inverse twiddles from `M::TWIDDLES_INVERSE` and
/// the scaling factor from `M::N_INV`.
#[inline]
pub(crate) fn intt_inplace<const N: usize, M: NttFriendly<N>>(m: M, buf: &mut [u64; N]) {
    cooley_tukey_dif_inverse(
        m,
        buf.as_mut_slice(),
        M::TWIDDLES_INVERSE.as_slice(),
        M::N_INV,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::zq::modulus::paper;

    type M17 = ConstModulus<17>;

    #[test]
    fn mod_pow_small() {
        assert_eq!(mod_pow(2, 0, 17), 1);
        assert_eq!(mod_pow(2, 4, 17), 16);
        assert_eq!(mod_pow(2, 8, 17), 1);
        assert_eq!(mod_pow(3, 16, 17), 1); // Fermat's little theorem
        // Cross-check at a realistic prime: any a^(q-1) ≡ 1 by Fermat.
        let q = 8380417u64;
        assert_eq!(mod_pow(7, q - 1, q), 1);
        assert_eq!(mod_pow(12345, q - 1, q), 1);
    }

    #[test]
    fn bit_reverse_small() {
        // log_n = 2: 0b00 -> 0b00, 0b01 -> 0b10, 0b10 -> 0b01, 0b11 -> 0b11.
        assert_eq!(bit_reverse(0, 2), 0);
        assert_eq!(bit_reverse(1, 2), 2);
        assert_eq!(bit_reverse(2, 2), 1);
        assert_eq!(bit_reverse(3, 2), 3);
        // log_n = 3: 0b001 -> 0b100, 0b010 -> 0b010, 0b011 -> 0b110.
        assert_eq!(bit_reverse(1, 3), 4);
        assert_eq!(bit_reverse(2, 3), 2);
        assert_eq!(bit_reverse(3, 3), 6);
    }

    /// `find_primitive_2n_th_root` returns a value of order exactly $2N$
    /// at every realistic prime / $N$ pair. This is the corner the
    /// `MAX_BASE_TRIES` cap protects: a regression that requires more
    /// trials than the cap would fail at compile time, not silently
    /// produce a wrong twiddle factor. Run at the smallest $N$
    /// where every realistic prime is NTT-friendly (N=4) to keep the test
    /// fast while still exercising each prime.
    #[test]
    fn find_primitive_2n_th_root_paper_primes_within_cap() {
        for q in [
            // Single primes.
            paper::ViaCQ3::Q,
            paper::ViaQ3::Q,
            paper::ViaCQ2::Q,
            paper::ViaQ2::Q,
            // RNS slot primes (both halves of $q_1$ for VIA and VIA-C).
            268369921u64,
            536608769,
            137438822401,
            274810798081,
        ] {
            let psi = find_primitive_2n_th_root(q, 8);
            // Order divides 2N=8.
            assert_eq!(mod_pow(psi, 8, q), 1, "q={q}: psi^8 != 1");
            // Order is not a proper divisor of 2N.
            assert_ne!(mod_pow(psi, 4, q), 1, "q={q}: psi^4 == 1 (order divides 4)");
        }
    }

    #[test]
    fn find_primitive_root_q17_2n8() {
        // 2 has order 8 in Z_17 (2^4 = 16 = -1, 2^8 = 1).
        let psi = find_primitive_2n_th_root(17, 8);
        // The const fn picks the smallest base whose `base^((q-1)/(2N))` has
        // order exactly 2N. That candidate is `psi = base^((q-1)/2N)`.
        // For q=17, 2N=8, (q-1)/2N = 2, so candidates are base^2 mod 17.
        // base=2: 2^2=4; order of 4 in Z_17 is 4 (4^4 = 256 mod 17 = 1, 4^2 = 16 != 1).
        //   Order 4, not 8 — rejected.
        // base=3: 3^2=9; 9^2=81=13, 9^4=13^2=169=16=-1, 9^8=1. Order 8. ✓
        assert_eq!(psi, 9);
        // Verify psi^8 = 1 and psi^4 != 1.
        assert_eq!(mod_pow(psi, 8, 17), 1);
        assert_ne!(mod_pow(psi, 4, 17), 1);
    }

    #[test]
    fn twiddle_table_q17_n4() {
        // psi = 9 from above. Table[bit_reverse(k, 2)] = 9^k mod 17.
        // 9^0=1, 9^1=9, 9^2=13, 9^3=15.
        // bit_reverse(0,2)=0; table[0]=1
        // bit_reverse(1,2)=2; table[2]=9
        // bit_reverse(2,2)=1; table[1]=13
        // bit_reverse(3,2)=3; table[3]=15
        let table = build_twiddle_table_bit_reversed::<4>(17, 9);
        assert_eq!(table, [1, 13, 9, 15]);
    }

    #[test]
    fn ntt_friendly_consts_q17_n4() {
        assert_eq!(<M17 as NttFriendly<4>>::PSI, 9);
        // N_INV = 4^{-1} mod 17 = 13 (since 4*13 = 52 = 51 + 1 = 3*17 + 1).
        assert_eq!(<M17 as NttFriendly<4>>::N_INV, 13);
        // Sanity: PSI * PSI_INV mod 17 = 1.
        let psi_inv = mod_inverse_u64(<M17 as NttFriendly<4>>::PSI, 17);
        assert_eq!((9 * psi_inv) % 17, 1);
    }

    #[test]
    fn ntt_friendly_consts_n_inv_paper() {
        // For each modulus we'll use at N=2048, check N * N_INV ≡ 1 (mod q).
        fn check<const Q: u64>(_q_marker: ConstModulus<Q>)
        where
            ConstModulus<Q>: NttFriendly<2048>,
        {
            let n_inv = <ConstModulus<Q> as NttFriendly<2048>>::N_INV;
            assert_eq!(((2048u128 % Q as u128) * n_inv as u128) % (Q as u128), 1);
        }
        check(paper::ViaCQ3::default());
        check(paper::ViaCQ2::default());
        check(paper::ViaQ3::default());
        check(paper::ViaQ2::default());
    }

    /// Every *coefficient* modulus that carries the
    /// `O(N²)` multiply cost is NTT-friendly at the degree `N = 2048`, so
    /// the eval-form `RLevCiphertext::gadget_product_ntt` path applies to it.
    /// This additionally covers the RNS `q₁` slot primes (both VIA and VIA-C)
    /// that `ntt_friendly_consts_n_inv_paper` does not. The power-of-two `q₄`
    /// and `p` are intentionally excluded — they keep the schoolbook path and do
    /// not implement `NttFriendly` (they satisfy `q ≡ 0 mod 2N`, not `q ≡ 1`).
    #[test]
    fn paper_coefficient_moduli_are_ntt_friendly_at_n2048() {
        use crate::algebra::rns::basis::RnsBasis;
        use crate::algebra::rns::basis::paper as rns_paper;

        // Compile-time witness: touching `N_INV` forces `check_ntt::<Q, 2048>()`
        // (the `(Q-1) % 4096 == 0` assertion), so this fails to compile if any
        // listed modulus stops being NTT-friendly. `N_INV` (not `PSI`) is used
        // to avoid the expensive const-eval primitive-root search.
        fn witness<M: NttFriendly<2048>>() {
            assert!(M::N_INV != 0);
        }
        // RNS q₁ slot primes — tied to the actual basis definitions.
        witness::<<rns_paper::ViaCQ1Rns as RnsBasis>::M0>();
        witness::<<rns_paper::ViaCQ1Rns as RnsBasis>::M1>();
        witness::<<rns_paper::ViaQ1Rns as RnsBasis>::M0>();
        witness::<<rns_paper::ViaQ1Rns as RnsBasis>::M1>();
        // Single-prime q₂ / q₃.
        witness::<paper::ViaCQ2>();
        witness::<paper::ViaCQ3>();
        witness::<paper::ViaQ2>();
        witness::<paper::ViaQ3>();

        // Human-readable companion: every eval-path modulus has q ≡ 1 (mod 4096).
        for q in [
            137_438_822_401u64, // ViaCQ1 slot 0
            274_810_798_081,    // ViaCQ1 slot 1
            268_369_921,        // ViaQ1 slot 0
            536_608_769,        // ViaQ1 slot 1
            17_175_674_881,     // ViaCQ2
            8_380_417,          // ViaCQ3
            34_359_214_081,     // ViaQ2
            2_147_352_577,      // ViaQ3
        ] {
            assert_eq!(q % 4096, 1, "coeff modulus {q} must be ≡ 1 mod 2N=4096");
        }

        // q₄ (2¹² / 2¹⁵) and p (16 / 256) are power-of-two ⇒ q ≡ 0 mod 2N, so
        // they are NOT NttFriendly and stay on the schoolbook path by design.
        for pow2 in [4096u64, 32768, 16, 256] {
            assert_ne!(
                pow2 % 4096,
                1,
                "power-of-two modulus {pow2} must not be NTT-friendly"
            );
        }
    }

    /// Forward NTT on a known input at q=17, N=4 produces the evaluations
    /// at psi^1, psi^3, psi^5, psi^7 in bit-reversed order.
    #[test]
    fn ntt_forward_known_input_q17_n4() {
        // f = a + bX + cX^2 + dX^3 with [a,b,c,d] = [1,2,3,4].
        // psi = 9. psi^1 = 9, psi^3 = 9^3 mod 17 = 15, psi^5 = 9^5 mod 17 = 8,
        // psi^7 = 9^7 mod 17 = 2.
        // f(9) = 1 + 18 + 243 + 2916 = ... mod 17.
        // Just compute via the kernel and verify against direct evaluation.
        let mut buf = [1u64, 2, 3, 4];
        let m = M17::default();
        cooley_tukey_dit_forward(
            m,
            &mut buf,
            <M17 as NttFriendly<4>>::TWIDDLES_FORWARD.as_slice(),
        );
        // Bit-reversed output:
        //   buf[0] = f(psi^{2*0+1}) = f(psi^1) = f(9)
        //   buf[1] = f(psi^{2*2+1}) = f(psi^5) = f(8)
        //   buf[2] = f(psi^{2*1+1}) = f(psi^3) = f(15)
        //   buf[3] = f(psi^{2*3+1}) = f(psi^7) = f(2)
        let eval_at = |x: u64| -> u64 {
            let mut acc = 0u128;
            let mut xp = 1u128;
            let coeffs = [1u128, 2, 3, 4];
            for c in coeffs {
                acc = (acc + c * xp) % 17;
                xp = (xp * x as u128) % 17;
            }
            acc as u64
        };
        assert_eq!(buf[0], eval_at(9));
        assert_eq!(buf[1], eval_at(8));
        assert_eq!(buf[2], eval_at(15));
        assert_eq!(buf[3], eval_at(2));
    }

    /// Bit-reversed convention check, parametric over $(M, N)$. Asserts
    /// that after `cooley_tukey_dit_forward`, lane $i$ holds the
    /// evaluation at $\psi^{2 \cdot \mathrm{br}(i, \log_2 N) + 1}$, where
    /// $\mathrm{br}$ is the bit-reversal permutation of width $\log_2 N$.
    /// This is the on-the-wire definition that `Poly::eval(i)` relies on
    /// to mean "the value at the $i$-th NTT point" — see the module
    /// doc.
    ///
    /// Returns the forward-NTT buffer so callers can chain a downstream
    /// check against `Poly::eval(...)` (see
    /// `poly_eval_pins_bit_reversed_index`).
    fn assert_bit_reversed_layout<const N: usize, M: NttFriendly<N>>(
        m: M,
        input: [u64; N],
    ) -> [u64; N] {
        let mut buf = input;
        cooley_tukey_dit_forward(m, &mut buf, M::TWIDDLES_FORWARD.as_slice());
        let psi = M::PSI;
        let log_n = (N as u64).trailing_zeros();
        let q = m.q() as u128;
        for (i, &lane) in buf.iter().enumerate() {
            let br_i = bit_reverse(i, log_n);
            let exp = (2 * br_i + 1) as u64;
            let psi_exp = mod_pow(psi, exp, m.q());
            // Reference: directly evaluate the polynomial at psi_exp.
            let mut acc = 0u128;
            let mut xp = 1u128;
            for &c in input.iter() {
                acc = (acc + (c as u128) * xp) % q;
                xp = (xp * (psi_exp as u128)) % q;
            }
            assert_eq!(
                lane as u128,
                acc,
                "q={} N={N} lane {i}: bit_reverse({i}, {log_n})={br_i}, exp={exp}",
                m.q(),
            );
        }
        buf
    }

    /// Bit-reversed NTT output convention at $(q, N) = (17, 8)$. Pairs
    /// with `ntt_forward_known_input_q17_n4` to lock the convention at a
    /// non-trivial $\log_2 N = 3$ (where bit-reversal is no longer
    /// "swap pairs"). Closes review item 6 (single-prime side).
    #[test]
    fn ntt_forward_known_input_q17_n8() {
        let m = M17::default();
        let _ = assert_bit_reversed_layout::<8, _>(m, [1u64, 2, 3, 4, 5, 6, 7, 8]);
        // Also exercise a less-uniform input so a regression that
        // happens to give a symmetric output on a monotone input still
        // surfaces.
        let _ = assert_bit_reversed_layout::<8, _>(m, [16u64, 0, 3, 11, 0, 9, 5, 7]);
    }

    /// Same check at a realistic prime ($q_3$ for VIA-C, $N = 8$). Locks
    /// the convention at the realistic-modulus / non-trivial-log-N
    /// regime that the protocol actually consumes.
    #[test]
    fn ntt_forward_known_input_paper_via_c_q3_n8() {
        let m = paper::ViaCQ3::default();
        let q = m.q();
        let input: [u64; 8] = core::array::from_fn(|i| (i as u64 * 12345 + 7) % q);
        let _ = assert_bit_reversed_layout::<8, _>(m, input);
    }

    /// `Poly::eval(i)` returns the value at the bit-reversed-position
    /// NTT point. Independent of `ntt_one_is_all_ones_in_eval` (which
    /// can't distinguish bit-reversed from natural order on a constant
    /// input). Closes review items 17 / 18 (single-prime side) by
    /// chaining through the `Poly` accessor onto the kernel-level
    /// `assert_bit_reversed_layout` reference.
    #[test]
    fn poly_eval_pins_bit_reversed_index() {
        use crate::algebra::ring::element::Poly;
        use crate::algebra::ring::form::Coefficient;
        let m = M17::default();
        let input = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let expected = assert_bit_reversed_layout::<8, _>(m, input);
        let p: Poly<8, _, Coefficient> = Poly::new(m, input);
        let e = p.into_eval();
        for (i, &want) in expected.iter().enumerate() {
            assert_eq!(e.eval(i).to_u64(), want, "lane {i}");
        }
    }

    /// `intt(ntt(f)) == f` at small (N, q).
    #[test]
    fn ntt_intt_roundtrip_small() {
        for input in [
            [0u64, 0, 0, 0],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [1, 2, 3, 4],
            [16, 16, 16, 16],
            [5, 11, 3, 7],
        ] {
            let mut buf = input;
            let m = M17::default();
            cooley_tukey_dit_forward(
                m,
                &mut buf,
                <M17 as NttFriendly<4>>::TWIDDLES_FORWARD.as_slice(),
            );
            cooley_tukey_dif_inverse(
                m,
                &mut buf,
                <M17 as NttFriendly<4>>::TWIDDLES_INVERSE.as_slice(),
                <M17 as NttFriendly<4>>::N_INV,
            );
            assert_eq!(buf, input, "input={input:?}");
        }
    }

    /// Adapter API `ntt_inplace` / `intt_inplace` round-trips correctly.
    #[test]
    fn adapters_roundtrip_q17_n8() {
        let input = [1u64, 2, 3, 4, 5, 6, 7, 8];
        let mut buf = input;
        let m = M17::default();
        ntt_inplace::<8, _>(m, &mut buf);
        intt_inplace::<8, _>(m, &mut buf);
        assert_eq!(buf, input);
    }

    /// Round-trip at $N = 2048$, $q = q_3$ (VIA-C). Exercises the
    /// realistic-size code path. Slow but locks the contract.
    #[test]
    fn ntt_roundtrip_at_paper_via_c_q3() {
        let m = paper::ViaCQ3::default();
        let mut buf = [0u64; 2048];
        // Fill with a deterministic non-zero pattern.
        for (i, slot) in buf.iter_mut().enumerate() {
            *slot = (i as u64) % m.q();
        }
        let original = buf;
        ntt_inplace::<2048, _>(m, &mut buf);
        intt_inplace::<2048, _>(m, &mut buf);
        assert_eq!(buf, original);
    }

    /// `cooley_tukey_dit_forward` debug-asserts every input lane is in
    /// `[0, q)`. Pins the canonical-form contract for direct kernel
    /// callers (the `Poly` wrapper enforces it via its type invariant,
    /// but a future GPU adapter that bypasses the wrapper would slip
    /// past without this assert).
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "ntt: input lane")]
    fn cooley_tukey_dit_forward_debug_asserts_canonical_input() {
        let m = M17::default();
        // Lane 0 holds q, which is exactly out of `[0, q)`.
        let mut buf = [17u64, 0, 0, 0];
        cooley_tukey_dit_forward(
            m,
            &mut buf,
            <M17 as NttFriendly<4>>::TWIDDLES_FORWARD.as_slice(),
        );
    }

    /// Same contract on the inverse NTT.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "intt: input lane")]
    fn cooley_tukey_dif_inverse_debug_asserts_canonical_input() {
        let m = M17::default();
        let mut buf = [0u64, 17, 0, 0];
        cooley_tukey_dif_inverse(
            m,
            &mut buf,
            <M17 as NttFriendly<4>>::TWIDDLES_INVERSE.as_slice(),
            <M17 as NttFriendly<4>>::N_INV,
        );
    }

    /// Homomorphism: `(f · g)_schoolbook == NTT^{-1}( NTT(f) ⊙ NTT(g) )`
    /// where `⊙` is pointwise multiplication. Confirms the negacyclic
    /// NTT is the right transform for $R_{n, q}$ multiplication.
    #[test]
    fn ntt_homomorphism_small() {
        use crate::algebra::ring::ops::negacyclic_mul_slice;
        let m = M17::default();
        let cases: &[([u64; 4], [u64; 4])] = &[
            ([1, 0, 0, 0], [3, 5, 11, 7]),
            ([0, 1, 0, 0], [1, 2, 3, 4]),
            ([5, 11, 3, 7], [9, 1, 4, 13]),
            ([16, 16, 16, 16], [1, 1, 1, 1]),
        ];
        for (f, g) in cases {
            // Schoolbook reference.
            let mut want = [0u64; 4];
            negacyclic_mul_slice(m, &mut want, f, g);
            // NTT-mediated.
            let mut fe = *f;
            let mut ge = *g;
            ntt_inplace::<4, _>(m, &mut fe);
            ntt_inplace::<4, _>(m, &mut ge);
            let mut prod = [0u64; 4];
            for i in 0..4 {
                prod[i] = m.mul(fe[i], ge[i]);
            }
            intt_inplace::<4, _>(m, &mut prod);
            assert_eq!(prod, want, "f={f:?}, g={g:?}");
        }
    }
}

/// `compile_fail` doctest: `ConstModulus<5>: NttFriendly<4>` violates
/// `5 ≡ 1 (mod 8)` and must fail to compile when any const item is
/// touched.
///
/// ```compile_fail
/// use via_primitives::algebra::ring::ntt::NttFriendly;
/// use via_primitives::algebra::zq::modulus::ConstModulus;
/// const _: u64 = <ConstModulus<5> as NttFriendly<4>>::PSI;
/// ```
///
/// Positive companion: `ConstModulus<17>: NttFriendly<4>` satisfies
/// `17 ≡ 1 (mod 8)` and instantiates with `PSI = 9` (the primitive
/// $2N$-th root reached by [`find_primitive_2n_th_root`]: $\mathrm{base}
/// = 3$, $(q - 1) / (2N) = 2$, $3^2 \bmod 17 = 9$, whose order is
/// exactly $8 = 2N$). Locks the happy path alongside the rejection.
///
/// ```
/// use via_primitives::algebra::ring::ntt::NttFriendly;
/// use via_primitives::algebra::zq::modulus::ConstModulus;
/// const PSI: u64 = <ConstModulus<17> as NttFriendly<4>>::PSI;
/// const N_INV: u64 = <ConstModulus<17> as NttFriendly<4>>::N_INV;
/// assert_eq!(PSI, 9);
/// // 4 * 13 = 52 ≡ 1 (mod 17).
/// assert_eq!(N_INV, 13);
/// ```
#[cfg(doctest)]
struct NttFriendlyCompileFailDocs;
