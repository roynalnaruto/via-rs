//! Compressed answer wire type.
//!
//! [`CompressedAnswer`] is the paper-path `RespComp` output: an **asymmetric**
//! `ModSwitchedCiphertext` with mask at $q_3$ and body at $q_4$ (paper Figure 7:
//! $\mathrm{ans} \leftarrow (A, \lfloor q_4 \cdot B / q_3 \rceil)$).
//!
//! # Paper vs Python divergence
//!
//! The Python reference `via_c/resp_comp.py` uses a **symmetric** mod-switch
//! (both mask and body at $q_3$) "for implementation simplicity". The paper
//! (Figure 7) and this implementation use the asymmetric variant: mask stays at
//! $q_3$, body is rescaled to $q_4$. Per "paper wins", the asymmetric q3/q4
//! split is the target; `decrypt_asymmetric(S2 @ q3, q3, q4, p)` is the matching
//! recovery primitive. Cross-language KATs must therefore compare at the
//! *recover output*, not the raw ciphertext modulus.

use core::fmt;

use via_primitives::params::ViaCModSwitchedQ3Q4;

/// VIA-C compressed answer: the `RespComp` output ciphertext
/// (`via_c/params.py:CompressedAnswer`).
///
/// Wraps `ViaCModSwitchedQ3Q4<N2>` =
/// `ModSwitchedCiphertext<N2, Poly<N2, q3>, Poly<N2, q4>>` — mask at $q_3$, body
/// at $q_4$ — where `N2` is the small ring degree after ring-switching.
/// Decryption uses `decrypt_asymmetric(S2 @ q3, q3, q4, p)`.
#[derive(Clone, Copy)]
pub struct CompressedAnswer<const N2: usize> {
    /// The asymmetric mod-switched ciphertext encoding the selected record.
    pub ciphertext: ViaCModSwitchedQ3Q4<N2>,
}

impl<const N2: usize> CompressedAnswer<N2> {
    /// Construct a `CompressedAnswer` from the `RespComp` output ciphertext.
    #[inline]
    pub fn new(ciphertext: ViaCModSwitchedQ3Q4<N2>) -> Self {
        Self { ciphertext }
    }
}

impl<const N2: usize> fmt::Debug for CompressedAnswer<N2> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("CompressedAnswer")
            .field("N2", &N2)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use via_primitives::algebra::ring::RingPoly;
    use via_primitives::algebra::zq::modulus::paper::{ViaCQ3, ViaCQ4};
    use via_primitives::params::{ViaCPolyQ3, ViaCPolyQ4};

    #[test]
    fn compressed_answer_n16_constructs() {
        let q3 = ViaCQ3::default();
        let q4 = ViaCQ4::default();
        let mask = <ViaCPolyQ3<16> as RingPoly<16>>::zero(q3);
        let body = <ViaCPolyQ4<16> as RingPoly<16>>::zero(q4);
        let ct = ViaCModSwitchedQ3Q4::<16>::new(mask, body);
        let ans = CompressedAnswer::new(ct);
        let dbg = alloc::format!("{ans:?}");
        assert!(dbg.contains("CompressedAnswer"));
    }
}
