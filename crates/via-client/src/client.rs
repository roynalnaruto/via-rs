//! [`Client`] — the VIA-C client: keygen + public-parameter assembly
//! (`setup`), index compression (`query`), and answer recovery (`recover`),
//! exactly the three actions of VIA-C Figure 8.
//!
//! `paper:via_c/client.py`

use via_primitives::algebra::ring::RingPoly;
use via_primitives::conversion::mlwe_ops::LweDot;
use via_primitives::encryption::types::{ModSwitchedCiphertext, SecretKey};
use via_primitives::gates::gen_rlwe_to_rgsw_key_boxed;
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::RingSwitchKey;
use via_protocol::{CompressedQuery, PIRParams, PublicParams, QueryCompressionKey};
use zeroize::ZeroizeOnDrop;

use crate::query::build_compressed_query;

/// VIA-C client state: the two secret keys plus the query configuration needed
/// by `query`. Generic over the same const-generics as [`PublicParams`] so the
/// compiler enforces dimensional consistency between client and server.
///
/// `R1` is `S1`'s ring at `(q1, n1)`; `R2` is `S2`'s ring at `(q3, n2)`.
///
/// # Zeroize
///
/// `sk1`/`sk2` are scrubbed on drop (the derive zeroizes them; the public
/// config fields are `#[zeroize(skip)]`-ed). `SecretKey` is itself
/// `ZeroizeOnDrop`, so the key material is cleared even without the derive.
#[derive(ZeroizeOnDrop)]
pub struct Client<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1>,
    R2: RingPoly<N2>,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> {
    sk1: SecretKey<N1, R1>,
    sk2: SecretKey<N2, R2>,
    #[zeroize(skip)]
    num_rows: usize,
    #[zeroize(skip)]
    num_cols: usize,
    #[zeroize(skip)]
    dmux_base: u64,
    #[zeroize(skip)]
    cmux_base: u64,
    #[zeroize(skip)]
    error_dist: Distribution,
}

impl<
    const N1: usize,
    const N2: usize,
    R1: RingPoly<N1>,
    R2: RingPoly<N2>,
    const L_QUERY: usize,
    const L_CK: usize,
    const L_RSK: usize,
    const D: usize,
> Client<N1, N2, R1, R2, L_QUERY, L_CK, L_RSK, D>
{
    /// VIA-C `Setup` — generate `S1 @ q1` and `S2 @ q3`, assemble the
    /// query-compression key and the ring-switch key, and return the
    /// `(Client, PublicParams)` pair (the server gets the [`PublicParams`]).
    ///
    /// Two key generators are **injected** as closures because they are
    /// ring-specific and cannot be expressed generically here:
    /// - `gen_cascade_key` picks the right `lwe_to_rlwe_n*` cascade for `(N1, R1)`;
    /// - `gen_ring_switch_key` performs `rekey_secret_key(S1 → q3)` **then**
    ///   `gen_rsk` (the rekey's `RekeySource` bound is private to
    ///   `via-primitives`, so it must run at a concrete call site). Generating
    ///   `gen_rsk` on the original `S1` without the rekey is a compile error;
    ///   rekeying to the wrong modulus is a silent error — the closure must use
    ///   `S2`'s modulus (`q3`) as the rekey target.
    ///
    /// # PRG draw order (KAT-pinned)
    ///
    /// keygen `S1`, keygen `S2`, cascade key, `gen_rlwe_to_rgsw_key`
    /// (`RLev_{S1}(S1²)`), then the ring-switch key (`D` RLev samples,
    /// mask-then-error per level).
    ///
    /// `paper:via_c/client.py:58-144`
    #[allow(clippy::too_many_arguments)]
    pub fn setup<K, GenCascade, GenRsk>(
        q1_mod: R1::Modulus,
        q3_mod: R2::Modulus,
        params: PIRParams,
        num_rows: usize,
        num_cols: usize,
        ck_base: u64,
        key_dist_1: Distribution,
        key_dist_2: Distribution,
        error_dist: Distribution,
        prg: &mut Shake256Prg,
        gen_cascade_key: GenCascade,
        gen_ring_switch_key: GenRsk,
    ) -> (
        Self,
        PublicParams<K, N1, N2, R1, R2, L_QUERY, L_CK, L_RSK, D>,
    )
    where
        K: zeroize::Zeroize,
        // Returns `Box<K>` (not `K`): the paper cascade key is ~24.75 MB and is
        // built directly on the heap (`gen_..._boxed`); moving it by value would
        // overflow the stack. The toy path just wraps its small key in `Box::new`.
        GenCascade:
            FnOnce(&SecretKey<N1, R1>, u64, Distribution, &mut Shake256Prg) -> alloc::boxed::Box<K>,
        GenRsk: FnOnce(
            &SecretKey<N1, R1>,
            &SecretKey<N2, R2>,
            Distribution,
            &mut Shake256Prg,
        ) -> RingSwitchKey<N1, N2, R2, L_RSK, D>,
    {
        // 1–2. Secret keys.
        let sk1 = SecretKey::<N1, R1>::keygen(q1_mod, key_dist_1, prg);
        let sk2 = SecretKey::<N2, R2>::keygen(q3_mod, key_dist_2, prg);

        // 3–4. Query-compression key: cascade (PRG first), then RLev_{S1}(S1²).
        // Both keys are built straight onto the heap so neither (the ~24.75 MiB
        // cascade key, the ~1.125 MiB conv-key RLev at paper scale) transits the
        // stack.
        let cascade_key = gen_cascade_key(&sk1, ck_base, error_dist, prg);
        let rlwe_to_rgsw_key =
            gen_rlwe_to_rgsw_key_boxed::<N1, R1, L_CK>(&sk1, ck_base, error_dist, prg);
        let qck = QueryCompressionKey::new(cascade_key, rlwe_to_rgsw_key);

        // 5. Ring-switch key (rekey S1→q3 then gen_rsk, inside the closure).
        let rsk = gen_ring_switch_key(&sk1, &sk2, error_dist, prg);

        // Query gadget bases: DMux uses b1, CMux/CRot use b2 (differ at paper
        // params). Read before `params` is moved into PublicParams.
        let dmux_base = params.gadget_base_1;
        let cmux_base = params.gadget_base_2;
        let pp = PublicParams::new(qck, rsk, params, num_rows, num_cols, ck_base, L_CK);

        let client = Self {
            sk1,
            sk2,
            num_rows,
            num_cols,
            dmux_base,
            cmux_base,
            error_dist,
        };
        (client, pp)
    }

    /// VIA-C `Query` — compress a flat database `index` into a
    /// [`CompressedQuery`]. See [`crate::query`] for the PRG-order contract.
    ///
    /// `paper:via_c/client.py:146-176`
    pub fn query(
        &self,
        index: usize,
        prg: &mut Shake256Prg,
    ) -> CompressedQuery<N1, 1, R1::Projected<1>>
    where
        R1: LweDot<N1>,
    {
        build_compressed_query::<N1, R1, L_QUERY>(
            index,
            &self.sk1,
            self.num_rows,
            self.num_cols,
            D,
            self.dmux_base,
            self.cmux_base,
            self.error_dist,
            prg,
        )
    }

    /// VIA-C `Recover` — decrypt the (paper-asymmetric) server answer with `S2`,
    /// returning the recovered plaintext record polynomial.
    ///
    /// Takes the raw `ModSwitchedCiphertext<N2, RM, RB>` (mask @ q3, body @ q4);
    /// the paper wire type `via_protocol::CompressedAnswer` is the same value
    /// unwrapped at the boundary. Mirrors the generic server, which returns the
    /// raw ciphertext.
    ///
    /// `paper:via_c/client.py:178-203`
    pub fn recover<RM, RB, RP>(
        &self,
        answer: &ModSwitchedCiphertext<N2, RM, RB>,
        q3_mod: RM::Modulus,
        q4_mod: RB::Modulus,
        p_mod: RP::Modulus,
    ) -> RP
    where
        R2: RingPoly<N2, CenteredScalar = i64>,
        RM: RingPoly<N2>,
        RB: RingPoly<N2>,
        RP: RingPoly<N2>,
    {
        self.sk2
            .decrypt_asymmetric::<RM, RB, RP>(answer, q3_mod, q4_mod, p_mod)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use via_primitives::algebra::ring::element::Poly;
    use via_primitives::algebra::ring::form::Coefficient;
    use via_primitives::algebra::zq::modulus::DynModulus;
    use via_primitives::conversion::{LweToRlweKeyN8, gen_lwe_to_rlwe_key_n8};
    use via_primitives::switching::gen_rsk;
    use via_primitives::switching::rekey::rekey_secret_key;
    use via_protocol::KeyDist;

    // Toy single-prime client params (N1=8, N2=4, d=D=2).
    const N1: usize = 8;
    const N2: usize = 4;
    const D: usize = 2;
    const L_QUERY: usize = 2;
    const L_CK: usize = 6;
    const L_RSK: usize = 8;
    const CK_BASE: u64 = 8;
    const RSK_BASE: u64 = 8;
    const QUERY_BASE: u64 = 64;
    type R8 = Poly<N1, DynModulus, Coefficient>;
    type R4 = Poly<N2, DynModulus, Coefficient>;
    type ToyClient = Client<N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>;

    fn toy_params() -> PIRParams {
        PIRParams::new(
            N1,
            N2,
            1u128 << 36,
            1 << 28,
            1 << 20,
            1 << 12,
            16, //
            QUERY_BASE,
            L_QUERY,
            QUERY_BASE,
            L_QUERY,
            RSK_BASE,
            L_RSK, //
            KeyDist::Ternary,
            KeyDist::Ternary,
            1,
            None,
            None,
            None,
            40,
        )
    }

    /// Drive `setup` with the toy n8 cascade + a concrete rekey→gen_rsk closure.
    #[allow(clippy::type_complexity)]
    fn toy_setup(
        num_rows: usize,
        num_cols: usize,
        prg: &mut Shake256Prg,
    ) -> (
        ToyClient,
        PublicParams<LweToRlweKeyN8<DynModulus, L_CK>, N1, N2, R8, R4, L_QUERY, L_CK, L_RSK, D>,
    ) {
        let q1 = DynModulus::new(1 << 36);
        let q3 = DynModulus::new(1 << 20);
        ToyClient::setup(
            q1,
            q3,
            toy_params(),
            num_rows,
            num_cols,
            CK_BASE,
            Distribution::Ternary,
            Distribution::Ternary,
            Distribution::Ternary,
            prg,
            |sk, base, dist, p| {
                alloc::boxed::Box::new(gen_lwe_to_rlwe_key_n8::<DynModulus, L_CK>(
                    sk, base, dist, p,
                ))
            },
            |sk1, sk2, dist, p| {
                let q3 = RingPoly::modulus(sk2.poly());
                let s1_q3 = rekey_secret_key::<N1, R8, R8>(sk1, q3);
                gen_rsk::<N1, N2, R8, R4, L_RSK, D>(&s1_q3, sk2, RSK_BASE, dist, p)
            },
        )
    }

    /// `setup` returns a `PublicParams` carrying the dimensions/bases it was given.
    #[test]
    fn setup_produces_public_params_with_dims() {
        let mut prg = Shake256Prg::new(b"client-setup");
        let (_client, pp) = toy_setup(2, 2, &mut prg);
        assert_eq!(pp.num_rows, 2);
        assert_eq!(pp.num_cols, 2);
        assert_eq!(pp.ck_base, CK_BASE);
        assert_eq!(pp.ck_depth, L_CK);
    }

    /// `query` emits exactly `L_QUERY · (log₂I + log₂J + log₂d)` LWEs.
    /// For I=J=d=2: total_bits = 1+1+1 = 3 → 2·3 = 6 ciphertexts.
    #[test]
    fn query_length_is_l_query_times_total_bits() {
        let mut prg = Shake256Prg::new(b"client-query-len");
        let (client, _pp) = toy_setup(2, 2, &mut prg);
        let cq = client.query(5, &mut prg);
        assert_eq!(cq.ciphertexts.len(), L_QUERY * (1 + 1 + 1));
    }
}
