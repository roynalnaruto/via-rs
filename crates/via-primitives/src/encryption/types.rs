//! Ciphertext datatypes.
//!
//! Every type is generic over a polynomial backend
//! `R: RingPoly<N>` so the same struct works for both the single-prime
//! [`Poly`](crate::algebra::ring::element::Poly) and the RNS
//! [`PolyRns`](crate::algebra::ring::rns_element::PolyRns) carriers. See
//! [`crate::algebra::ring::abstraction`] for the trait, and
//! [`super::aliases`] for ergonomic typedefs at the standard parameter sets.

use core::fmt;
use core::marker::PhantomData;

use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::algebra::ring::{RingPoly, RingPolyEval};

// ---------------------------------------------------------------------------
// SecretKey
// ---------------------------------------------------------------------------

/// A secret key — a single polynomial $S \in R_{n, q}$ sampled from one of
/// the key distributions (ternary, bounded-uniform, or discrete
/// Gaussian).
///
/// VIA-C / VIA-B carry two keys $S_1 \in R_{n_1, q_1}$ and $S_2 \in R_{n_2,
/// q_2}$ at distinct moduli; either is represented by a distinct
/// `SecretKey<N, R>` instantiation.
///
/// `SecretKey` deliberately does **not** implement `Copy`: it implements
/// [`ZeroizeOnDrop`] so the key material is cleared when the value is
/// dropped, and `Copy + Drop` would conflict. Manual [`Clone`] is provided
/// for the legitimate paths that need duplicates (e.g. handing the same
/// key to two algorithms in succession).
pub struct SecretKey<const N: usize, R: RingPoly<N>> {
    /// The secret polynomial. Stored in coefficient form.
    pub poly: R,
}

impl<const N: usize, R: RingPoly<N>> SecretKey<N, R> {
    /// Wrap an existing polynomial as a secret key. Most callers should
    /// instead use the `keygen` free function which samples a
    /// fresh polynomial from the chosen key distribution.
    #[inline(always)]
    pub fn from_poly(poly: R) -> Self {
        Self { poly }
    }

    /// Borrow the underlying polynomial.
    #[inline(always)]
    pub fn poly(&self) -> &R {
        &self.poly
    }
}

impl<const N: usize, R: RingPoly<N>> Clone for SecretKey<N, R> {
    fn clone(&self) -> Self {
        Self { poly: self.poly }
    }
}

impl<const N: usize, R: RingPoly<N>> Zeroize for SecretKey<N, R> {
    fn zeroize(&mut self) {
        self.poly.zeroize();
    }
}

impl<const N: usize, R: RingPoly<N>> Drop for SecretKey<N, R> {
    fn drop(&mut self) {
        self.zeroize();
    }
}

// Marker derived from `Drop + Zeroize` impls above. `zeroize::ZeroizeOnDrop`
// is a marker trait, not a derive trigger; implementing it is the contract
// that we both `zeroize()` and `Drop`. Manual impl keeps this honest.
impl<const N: usize, R: RingPoly<N>> ZeroizeOnDrop for SecretKey<N, R> {}

/// `Debug` deliberately redacts the secret polynomial — printing key
/// material to a log would be a security bug. The output shape mirrors
/// the public-API rule we expect for every secret-bearing type in this
/// crate.
impl<const N: usize, R: RingPoly<N>> fmt::Debug for SecretKey<N, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SecretKey")
            .field("N", &N)
            .field("poly", &"<redacted>")
            .finish()
    }
}

// ---------------------------------------------------------------------------
// RLWECiphertext
// ---------------------------------------------------------------------------

/// An RLWE ciphertext $(A, B)$ where $A \in R_{n, q}$ is uniform and
/// $B = A \cdot S + e + M'$ for an *already-encoded* message $M' = \Delta
/// \cdot m$ with $\Delta = \lceil q / p \rceil$. The
/// `encrypt` primitive is agnostic to $\Delta$ because RLev re-uses it
/// with gadget-scaled messages rather than $\Delta$-encoded ones.
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct RLWECiphertext<const N: usize, R: RingPoly<N>> {
    /// Mask $A$ — uniformly random in $R_{n, q}$.
    pub mask: R,
    /// Body $B = A \cdot S + e + M'$ — the only component that decryption
    /// "subtracts the mask from" to recover the encoded message plus noise.
    pub body: R,
}

impl<const N: usize, R: RingPoly<N>> RLWECiphertext<N, R> {
    /// Construct an RLWE ciphertext from its `mask` / `body` components.
    /// Used by `encrypt` and by the trivial-RLWE constructor.
    #[inline(always)]
    pub fn new(mask: R, body: R) -> Self {
        Self { mask, body }
    }
}

impl<const N: usize, R: RingPoly<N>> Zeroize for RLWECiphertext<N, R> {
    fn zeroize(&mut self) {
        self.mask.zeroize();
        self.body.zeroize();
    }
}

impl<const N: usize, R: RingPoly<N>> fmt::Debug for RLWECiphertext<N, R> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RLWECiphertext")
            .field("N", &N)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// RLevCiphertext
// ---------------------------------------------------------------------------

/// A tuple of $L$ RLWE ciphertexts encrypting the gadget-scaled messages
/// $\bigl(g_i \cdot M\bigr)_{i = 1}^{L}$, where $g_i = \lceil q / B^i
/// \rceil$ are the entries of the VIA-convention gadget vector.
///
/// **Note**: the gadget entries replace the $\Delta$ scaling — RLev does
/// **not** go through `encode`.
///
/// The gadget depth $L$ and base $B$ are tuned per call site (DMux ctrl,
/// CMux sel, ring-switch, LWE-to-RLWE conv, RLWE-to-RGSW conv). $L$ is a
/// const generic parameter; $B$ is threaded through function calls.
#[derive(Clone, Copy)]
pub struct RLevCiphertext<const N: usize, R: RingPoly<N>, const L: usize> {
    /// The $L$ RLWE samples, ordered MSB-first so that `samples[0]` pairs
    /// with $g_0 = \lceil q / B \rceil$ (algorithm step 4).
    pub samples: [RLWECiphertext<N, R>; L],
}

impl<const N: usize, R: RingPoly<N>, const L: usize> RLevCiphertext<N, R, L> {
    /// Construct an RLev ciphertext from an array of $L$ RLWE samples.
    #[inline(always)]
    pub fn new(samples: [RLWECiphertext<N, R>; L]) -> Self {
        Self { samples }
    }
}

impl<const N: usize, R: RingPoly<N>, const L: usize> Zeroize for RLevCiphertext<N, R, L> {
    fn zeroize(&mut self) {
        for s in &mut self.samples {
            s.zeroize();
        }
    }
}

impl<const N: usize, R: RingPoly<N>, const L: usize> fmt::Debug for RLevCiphertext<N, R, L> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RLevCiphertext")
            .field("N", &N)
            .field("L", &L)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// RLevEval — evaluation-form RLev key (T7 eval-key storage)
// ---------------------------------------------------------------------------

/// An RLev whose `L` samples are stored in **evaluation form** — each
/// `(mask, body)` pre-transformed via the negacyclic NTT.
///
/// Derived once from a coefficient-form [`RLevCiphertext`] via
/// [`RLevCiphertext::to_eval`](crate::encryption::RLevCiphertext::to_eval) so
/// [`RLevEval::gadget_product`](crate::encryption::RLevEval::gadget_product) can
/// skip the per-call `to_eval` of the (already-transformed) samples. This is the
/// storage form for **static** keys — the conversion key, ring-switch key, and
/// (T7 Phase B) the LWE→RLWE cascade keys — reused on every query.
///
/// `Copy` like [`RLevCiphertext`]. The NTT image of a secret key is itself
/// secret, so it is `Zeroize`; owners that store it (e.g. `PreparedKeys`,
/// [`RingSwitchKeyEval`](crate::switching::RingSwitchKeyEval)) are
/// `ZeroizeOnDrop`.
#[derive(Clone, Copy)]
pub struct RLevEval<const N: usize, R: RingPoly<N> + RingPolyEval<N>, const L: usize> {
    /// The `L` eval-form `(mask, body)` samples, MSB-first (same ordering as
    /// [`RLevCiphertext::samples`]).
    pub(crate) samples: [(R::Eval, R::Eval); L],
}

impl<const N: usize, R: RingPoly<N> + RingPolyEval<N>, const L: usize> Zeroize
    for RLevEval<N, R, L>
{
    fn zeroize(&mut self) {
        for (mask, body) in &mut self.samples {
            mask.zeroize();
            body.zeroize();
        }
    }
}

impl<const N: usize, R: RingPoly<N> + RingPolyEval<N>, const L: usize> fmt::Debug
    for RLevEval<N, R, L>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RLevEval")
            .field("N", &N)
            .field("L", &L)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// RGSWCiphertext
// ---------------------------------------------------------------------------

/// A pair $\bigl(\mathrm{RLev}(-S \cdot M), \, \mathrm{RLev}(M)\bigr)$ —
/// the homomorphic encoding consumed by the external product.
///
/// The two RLev halves can use **different** gadget parameters: separate
/// $(\ell, B)$ for `ctrl,1` vs `ctrl,2`, `sel,1`
/// vs `sel,2`, etc. So each half carries its own const-generic depth `L1`
/// / `L2`; the bases are threaded through `encrypt_rgsw` and
/// `external_product`.
#[derive(Clone, Copy)]
pub struct RGSWCiphertext<const N: usize, R: RingPoly<N>, const L1: usize, const L2: usize> {
    /// $\mathrm{RLev}_S(-S \cdot M)$ — depth $L_1$.
    pub neg_s_m: RLevCiphertext<N, R, L1>,
    /// $\mathrm{RLev}_S(M)$ — depth $L_2$.
    pub m: RLevCiphertext<N, R, L2>,
}

impl<const N: usize, R: RingPoly<N>, const L1: usize, const L2: usize>
    RGSWCiphertext<N, R, L1, L2>
{
    /// Construct an RGSW ciphertext from its two RLev halves.
    #[inline(always)]
    pub fn new(neg_s_m: RLevCiphertext<N, R, L1>, m: RLevCiphertext<N, R, L2>) -> Self {
        Self { neg_s_m, m }
    }
}

impl<const N: usize, R: RingPoly<N>, const L1: usize, const L2: usize> Zeroize
    for RGSWCiphertext<N, R, L1, L2>
{
    fn zeroize(&mut self) {
        self.neg_s_m.zeroize();
        self.m.zeroize();
    }
}

impl<const N: usize, R: RingPoly<N>, const L1: usize, const L2: usize> fmt::Debug
    for RGSWCiphertext<N, R, L1, L2>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RGSWCiphertext")
            .field("N", &N)
            .field("L1", &L1)
            .field("L2", &L2)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// ModSwitchedCiphertext
// ---------------------------------------------------------------------------

/// An $(A, B)$ pair where $A$ and $B$ may live at **different** moduli.
///
/// Used in two places:
///
/// - VIA's final answer (mask at $q_3$, body at $q_4$), returned by the
///   asymmetric `ModSwitch` of the Answer pipeline.
/// - VIA-C's `RespComp` output `(A, ⌊q_4 B / q_3⌋)` (mask at $q_3$, body
///   at $q_4$ after the final body rescale).
///
/// The two backing polynomial types `RA` and `RB` are independent;
/// nothing prevents them from being the same type at the same modulus
/// (which would degenerate to a symmetric ModSwitch).
#[derive(Clone, Copy)]
pub struct ModSwitchedCiphertext<const N: usize, RA: RingPoly<N>, RB: RingPoly<N>> {
    /// Mask, at modulus `RA::Modulus`.
    pub mask: RA,
    /// Body, at modulus `RB::Modulus`.
    pub body: RB,
    /// PhantomData over `N` — needed so the type is generic in `N` even
    /// when both `RA` and `RB` happen to share it (the compiler infers `N`
    /// from the trait bounds, but the type itself doesn't reference `N`
    /// outside those bounds without this).
    _marker: PhantomData<[(); N]>,
}

impl<const N: usize, RA: RingPoly<N>, RB: RingPoly<N>> ModSwitchedCiphertext<N, RA, RB> {
    /// Construct a `ModSwitchedCiphertext` from its mask / body components.
    #[inline(always)]
    pub fn new(mask: RA, body: RB) -> Self {
        Self {
            mask,
            body,
            _marker: PhantomData,
        }
    }
}

impl<const N: usize, RA: RingPoly<N>, RB: RingPoly<N>> Zeroize
    for ModSwitchedCiphertext<N, RA, RB>
{
    fn zeroize(&mut self) {
        self.mask.zeroize();
        self.body.zeroize();
    }
}

impl<const N: usize, RA: RingPoly<N>, RB: RingPoly<N>> fmt::Debug
    for ModSwitchedCiphertext<N, RA, RB>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ModSwitchedCiphertext")
            .field("N", &N)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algebra::ring::element::Poly;
    use crate::algebra::ring::form::Coefficient;
    use crate::algebra::ring::rns_element::PolyRns;
    use crate::algebra::rns::basis::ConstRnsBasis;
    use crate::algebra::zq::modulus::ConstModulus;

    type SingleR = Poly<4, ConstModulus<17>, Coefficient>;
    type RnsR = PolyRns<4, ConstRnsBasis<5, 11>, Coefficient>;

    // The smoke tests below only need to compile (and run zeroize). They
    // don't assert correctness — that's the job of the dedicated KAT tests.

    #[test]
    fn secret_key_constructs_with_both_backends() {
        let m = ConstModulus::<17>;
        let b = ConstRnsBasis::<5, 11>;
        let _sk_single = SecretKey::<4, SingleR>::from_poly(<SingleR as RingPoly<4>>::zero(m));
        let _sk_rns = SecretKey::<4, RnsR>::from_poly(<RnsR as RingPoly<4>>::zero(b));
    }

    #[test]
    fn rlwe_ciphertext_constructs_with_both_backends() {
        let m = ConstModulus::<17>;
        let b = ConstRnsBasis::<5, 11>;
        let single_zero = <SingleR as RingPoly<4>>::zero(m);
        let rns_zero = <RnsR as RingPoly<4>>::zero(b);
        let _single = RLWECiphertext::<4, SingleR>::new(single_zero, single_zero);
        let _rns = RLWECiphertext::<4, RnsR>::new(rns_zero, rns_zero);
    }

    #[test]
    fn rlev_ciphertext_constructs_with_both_backends() {
        let m = ConstModulus::<17>;
        let b = ConstRnsBasis::<5, 11>;
        let single_ct = RLWECiphertext::<4, SingleR>::new(
            <SingleR as RingPoly<4>>::zero(m),
            <SingleR as RingPoly<4>>::zero(m),
        );
        let rns_ct = RLWECiphertext::<4, RnsR>::new(
            <RnsR as RingPoly<4>>::zero(b),
            <RnsR as RingPoly<4>>::zero(b),
        );
        let _single = RLevCiphertext::<4, SingleR, 2>::new([single_ct; 2]);
        let _rns = RLevCiphertext::<4, RnsR, 3>::new([rns_ct; 3]);
    }

    #[test]
    fn rgsw_ciphertext_constructs_with_distinct_l1_l2() {
        let m = ConstModulus::<17>;
        let single_ct = RLWECiphertext::<4, SingleR>::new(
            <SingleR as RingPoly<4>>::zero(m),
            <SingleR as RingPoly<4>>::zero(m),
        );
        let rlev2 = RLevCiphertext::<4, SingleR, 2>::new([single_ct; 2]);
        let rlev4 = RLevCiphertext::<4, SingleR, 4>::new([single_ct; 4]);
        let _rgsw = RGSWCiphertext::<4, SingleR, 2, 4>::new(rlev2, rlev4);
    }

    #[test]
    fn mod_switched_ciphertext_allows_different_backends() {
        let m = ConstModulus::<17>;
        let b = ConstRnsBasis::<5, 11>;
        let single = <SingleR as RingPoly<4>>::zero(m);
        let rns = <RnsR as RingPoly<4>>::zero(b);
        // Mask in single-prime form, body in RNS form — the type accepts
        // any pairing.
        let _ct = ModSwitchedCiphertext::<4, SingleR, RnsR>::new(single, rns);
    }

    /// Verify `SecretKey`'s `Debug` impl never spills the underlying
    /// polynomial. We render into a fixed 256-byte stack buffer (the
    /// crate is `no_std + no alloc`) and check for the redaction marker.
    #[test]
    fn secret_key_redacts_in_debug() {
        use core::fmt::Write as _;
        let m = ConstModulus::<17>;
        let sk = SecretKey::<4, SingleR>::from_poly(<SingleR as RingPoly<4>>::zero(m));
        struct W {
            buf: [u8; 256],
            cursor: usize,
        }
        impl core::fmt::Write for W {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                let bytes = s.as_bytes();
                let end = self.cursor + bytes.len();
                if end > self.buf.len() {
                    return Err(core::fmt::Error);
                }
                self.buf[self.cursor..end].copy_from_slice(bytes);
                self.cursor = end;
                Ok(())
            }
        }
        let mut w = W {
            buf: [0u8; 256],
            cursor: 0,
        };
        write!(w, "{sk:?}").expect("Debug output must fit in 256B");
        let s = core::str::from_utf8(&w.buf[..w.cursor]).unwrap();
        assert!(
            s.contains("redacted"),
            "SecretKey Debug must redact its poly: got {s:?}"
        );
    }
}
