//! Pre-transformed static answer keys — T7 eval-key storage.
//!
//! The three **static** answer keys — the conversion key
//! (`pp.query_comp_key.rlwe_to_rgsw_key`), the ring-switch key
//! (`pp.ring_switch_key`), and the LWE→RLWE **cascade** key
//! (`pp.query_comp_key.lwe_to_rlwe_key`) — are generated once and reused on every
//! query, and each is consumed via `gadget_product`. This struct holds their
//! **evaluation-form** images, derived **once at setup**
//! ([`PreparedKeys::from_public_params`]), so the per-query consumers
//! (`query_decomp` → `lwe_to_rlwe_*_eval` + `rlwe_to_rgsw_eval`, `resp_comp` →
//! `ring_switch_eval`) skip the per-call `to_eval` of the key samples.
//!
//! Mirrors [`PreparedDb`](crate::prepared_db::PreparedDb): the canonical
//! **coefficient** keys stay in `PublicParams` (the cross-language KAT-parity
//! contract is on the coeff keygen), and the eval form is **derived** —
//! deterministic NTT, no PRG. `ZeroizeOnDrop`: the NTT image of a secret key is
//! itself secret.

use alloc::boxed::Box;

use via_primitives::algebra::ring::{RingPoly, RingPolyEval};
use via_primitives::conversion::CascadeKey;
use via_primitives::encryption::RLevEval;
use via_primitives::switching::RingSwitchKeyEval;
use via_protocol::PublicParams;
use zeroize::{Zeroize, ZeroizeOnDrop};

/// Evaluation-form images of the three static answer keys (T7).
///
/// - `R1` — the `q1` ring (conversion key `RLev_{S1}(S1²)` at `n1`).
/// - `R3` — the `q3@n2` ring (ring-switch key samples).
/// - `K` — the coefficient cascade key (`pp.query_comp_key.lwe_to_rlwe_key`); its
///   [`CascadeKey::Eval`] mirror is held heap-boxed in [`Self::cascade`].
pub struct PreparedKeys<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    K: CascadeKey,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> {
    /// Eval-form conversion key — drives `rlwe_to_rgsw_eval` in `query_decomp`.
    pub(crate) conv_key: RLevEval<N1, R1, L_CK>,
    /// Eval-form ring-switch key — drives `ring_switch_eval` in `resp_comp`.
    pub(crate) rsk: RingSwitchKeyEval<N1, N2, R3, L_RSK, D>,
    /// Eval-form LWE→RLWE cascade key (heap-boxed, ~24.75 MB at paper scale) —
    /// drives `lwe_to_rlwe_*_eval` via the injected cascade closure in
    /// `query_decomp`.
    pub(crate) cascade: Box<K::Eval>,
}

impl<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    K: CascadeKey + Zeroize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> PreparedKeys<N1, N2, R1, R3, K, L_CK, L_RSK, D>
{
    /// Derive the eval-form static keys once from the (coefficient) public
    /// parameters at setup. Deterministic (forward NTT, no PRG) — the coeff keys
    /// in `pp` stay canonical for the KAT-parity contract. The cascade key is
    /// heap-built ([`CascadeKey::to_eval_boxed`]) so it never transits the stack.
    pub fn from_public_params<const L_QUERY: usize>(
        pp: &PublicParams<K, N1, N2, R1, R3, L_QUERY, L_CK, L_RSK, D>,
    ) -> Self {
        Self {
            conv_key: pp.query_comp_key.rlwe_to_rgsw_key.to_eval(),
            rsk: pp.ring_switch_key.to_eval(),
            cascade: pp.query_comp_key.lwe_to_rlwe_key.to_eval_boxed(),
        }
    }
}

impl<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    K: CascadeKey,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> Zeroize for PreparedKeys<N1, N2, R1, R3, K, L_CK, L_RSK, D>
{
    fn zeroize(&mut self) {
        self.conv_key.zeroize();
        self.rsk.zeroize();
        self.cascade.zeroize();
    }
}

impl<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    K: CascadeKey,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> Drop for PreparedKeys<N1, N2, R1, R3, K, L_CK, L_RSK, D>
{
    fn drop(&mut self) {
        self.zeroize();
    }
}

impl<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1> + RingPolyEval<N1>,
    R3: RingPoly<N2> + RingPolyEval<N2>,
    K: CascadeKey,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> ZeroizeOnDrop for PreparedKeys<N1, N2, R1, R3, K, L_CK, L_RSK, D>
{
}
