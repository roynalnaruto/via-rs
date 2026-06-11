//! The common [`VariantParams`] trait — one read surface for the scheme
//! dimensions shared by VIA-C and VIA-B.
//!
//! Implemented by the runtime [`PIRParams`] sidecar and by the compile-time ZST
//! markers ([`ViaCPublicParams`], and under `via-b` `ViaBPublicParams`). VIA-C
//! is the degenerate case `n3 = n2`, `t = 1`, where `d3() == d()`.

use crate::params::PIRParams;
#[cfg(feature = "via-b")]
use crate::presets::ViaBPublicParams;
use crate::presets::ViaCPublicParams;

/// Common accessor surface for the VIA scheme dimensions.
///
/// `d() = n1/n2` is the ring-switch fold; `d3() = n1/n3` is the records-per-cell
/// / CRot range. For VIA-C, `n3 == n2` and `t == 1`, so `d3() == d()`.
pub trait VariantParams {
    /// Large ring degree $n_1$.
    fn n1(&self) -> usize;
    /// Small ring degree $n_2$.
    fn n2(&self) -> usize;
    /// Record-ring degree $n_3$ ($= n_2$ for VIA-C).
    fn n3(&self) -> usize;
    /// Batch size $T$ ($= 1$ for VIA-C).
    fn t(&self) -> usize;
    /// Ring-switch fold $d = n_1 / n_2$.
    fn d(&self) -> usize {
        self.n1() / self.n2()
    }
    /// Records per cell / CRot range $d_3 = n_1 / n_3$ ($= d$ for VIA-C).
    fn d3(&self) -> usize {
        self.n1() / self.n3()
    }
}

impl VariantParams for PIRParams {
    fn n1(&self) -> usize {
        self.n1
    }
    fn n2(&self) -> usize {
        self.n2
    }
    fn n3(&self) -> usize {
        self.n3
    }
    fn t(&self) -> usize {
        self.t
    }
}

impl<
    const N1: usize,
    const N2: usize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> VariantParams for ViaCPublicParams<N1, N2, L_QUERY, L_CK, L_RSK, D>
{
    fn n1(&self) -> usize {
        N1
    }
    fn n2(&self) -> usize {
        N2
    }
    fn n3(&self) -> usize {
        N2 // VIA-C: record ring = small ring
    }
    fn t(&self) -> usize {
        1
    }
}

#[cfg(feature = "via-b")]
impl<
    const N1: usize,
    const N2: usize,
    const N3: usize,
    const T: usize,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> VariantParams for ViaBPublicParams<N1, N2, N3, T, L_QUERY, L_CK, L_RSK, D>
{
    fn n1(&self) -> usize {
        N1
    }
    fn n2(&self) -> usize {
        N2
    }
    fn n3(&self) -> usize {
        N3
    }
    fn t(&self) -> usize {
        T
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::presets::TOY_PARAMS;

    #[test]
    fn via_c_marker_is_degenerate() {
        // Explicit unit value of the ZST marker (guaranteed value position).
        let m = ViaCPublicParams::<64, 16, 20, 40, 8, 4>;
        assert_eq!(m.n1(), 64);
        assert_eq!(m.n2(), 16);
        assert_eq!(m.n3(), 16); // = n2 for VIA-C
        assert_eq!(m.t(), 1);
        assert_eq!(m.d(), 4);
        assert_eq!(m.d3(), 4); // = d for VIA-C
    }

    #[test]
    fn pir_params_variant_view() {
        assert_eq!(TOY_PARAMS.n3(), TOY_PARAMS.n2()); // VIA-C: n3 = n2
        assert_eq!(TOY_PARAMS.t(), 1);
        assert_eq!(TOY_PARAMS.d3(), TOY_PARAMS.d());
    }
}
