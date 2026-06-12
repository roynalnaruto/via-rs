//! Part-1 integration test: exercise every `via-protocol` crate-root export
//! together — the green milestone for the param + wire-type layer.
//!
//! Everything is imported from the crate root (`via_protocol::{…}`), confirming
//! the public API surface; primitive ciphertext types come through the
//! re-exported `via_protocol::primitives`.

use via_protocol::primitives::algebra::ring::RingPoly;
use via_protocol::primitives::algebra::ring::element::Poly;
use via_protocol::primitives::algebra::ring::form::Coefficient;
use via_protocol::primitives::algebra::rns::basis::paper::ViaCQ1Rns;
use via_protocol::primitives::algebra::zq::modulus::paper::{ViaCQ2, ViaCQ3, ViaCQ4};
use via_protocol::primitives::encryption::types::{RLWECiphertext, RLevCiphertext};
use via_protocol::primitives::encryption::{MLWECiphertext, RGSWCiphertext};
use via_protocol::primitives::params::{
    ViaCModSwitchedQ3Q4, ViaCPolyQ1Rns, ViaCPolyQ3, ViaCPolyQ4,
};
use via_protocol::primitives::switching::RingSwitchKey;

use via_protocol::{
    CompressedAnswer, CompressedQuery, DecompressedQuery, PrgCompressed, PublicParams,
    QueryCompressionKey, REALISTIC_PARAMS, TOY_PARAMS, Uncompressed, ViaCRealisticParams,
    ViaCToyParams, WireFormat, pir_params_matches_preset,
};

use zeroize::Zeroize;

// Toy dims, used throughout.
const N1: usize = 64;
const N2: usize = 16;
const L_QUERY: usize = 20;
const L_CK: usize = 40;
const L_RSK: usize = 8;
const D: usize = 4;

type Q2Ring = Poly<8, ViaCQ2, Coefficient>;

fn zero_q2_rlwe() -> RLWECiphertext<8, Q2Ring> {
    let m = ViaCQ2::default();
    let z = <Q2Ring as RingPoly<8>>::zero(m);
    RLWECiphertext::new(z, z)
}

fn zero_q2_rgsw<const L: usize>() -> RGSWCiphertext<8, Q2Ring, L, L> {
    let rlev = RLevCiphertext::new([zero_q2_rlwe(); L]);
    RGSWCiphertext::new(rlev, rlev)
}

/// `_CHECK` evaluates for both presets, and the preset `PIRParams` consts agree
/// with their const-generic markers.
#[test]
fn presets_check_and_cross_assert() {
    let () = ViaCToyParams::_CHECK;
    let () = ViaCRealisticParams::_CHECK;
    pir_params_matches_preset::<64, 16, 20, 40, 8, 4>(&TOY_PARAMS);
    pir_params_matches_preset::<2048, 512, 2, 18, 8, 4>(&REALISTIC_PARAMS);
    // delta is a u128 > u64 at realistic q1 ≈ 2^75.
    assert!(REALISTIC_PARAMS.delta() > u128::from(u64::MAX));
    assert_eq!(TOY_PARAMS.d(), 4);
}

/// `CompressedQuery` and `DecompressedQuery` construct and debug-print shape.
#[test]
fn query_types_construct() {
    let z = zero_q2_rlwe();
    let lwe = MLWECiphertext::<1, 8, Q2Ring>::new([z.mask], z.body);
    let cq = CompressedQuery::new(alloc_vec(lwe, 3));
    assert_eq!(cq.len(), 3);
    assert!(alloc::format!("{cq:?}").contains("CompressedQuery"));

    let rgsw = zero_q2_rgsw::<2>();
    let dq = DecompressedQuery::new(
        alloc_vec(rgsw, 2), // log2(I)
        alloc_vec(rgsw, 4), // log2(J)
        alloc_vec(rgsw, 2), // log2(d)
    );
    assert_eq!(
        dq.dmux_bits.len() + dq.cmux_bits.len() + dq.crot_bits.len(),
        8
    );
    assert!(alloc::format!("{dq:?}").contains("DecompressedQuery"));
}

/// `CompressedAnswer` wraps the asymmetric q3/q4 mod-switched ciphertext.
#[test]
fn compressed_answer_constructs() {
    let mask = <ViaCPolyQ3<16> as RingPoly<16>>::zero(ViaCQ3::default());
    let body = <ViaCPolyQ4<16> as RingPoly<16>>::zero(ViaCQ4::default());
    let ct = ViaCModSwitchedQ3Q4::<16>::new(mask, body);
    let ans = CompressedAnswer::new(ct);
    assert!(alloc::format!("{ans:?}").contains("CompressedAnswer"));
}

/// `QueryCompressionKey<K>` + `PublicParams` construct, redact secrets in
/// `Debug`, and zeroize.
#[test]
fn key_bundle_constructs_redacts_zeroizes() {
    #[derive(Debug)]
    struct StubKey(u64);
    impl Zeroize for StubKey {
        fn zeroize(&mut self) {
            self.0 = 0;
        }
    }

    let b = ViaCQ1Rns::default();
    let z = <ViaCPolyQ1Rns<N1> as RingPoly<N1>>::zero(b);
    let rlev = RLevCiphertext::new([RLWECiphertext::new(z, z); L_CK]);
    let mut qck = QueryCompressionKey::<StubKey, N1, ViaCPolyQ1Rns<N1>, L_CK>::new(
        alloc::boxed::Box::new(StubKey(0xABCD)),
        alloc::boxed::Box::new(rlev),
    );
    let dbg = alloc::format!("{qck:?}");
    assert!(dbg.contains("<redacted>") && !dbg.contains("ABCD"));

    // Ring-switch key at q3 (degree N2).
    let q3z = <ViaCPolyQ3<N2> as RingPoly<N2>>::zero(ViaCQ3::default());
    let rsk_rlev = RLevCiphertext::new([RLWECiphertext::new(q3z, q3z); L_RSK]);
    let rsk = RingSwitchKey::<N1, N2, ViaCPolyQ3<N2>, L_RSK, D>::new([rsk_rlev; D]);

    let qck2 = QueryCompressionKey::<StubKey, N1, ViaCPolyQ1Rns<N1>, L_CK>::new(
        alloc::boxed::Box::new(StubKey(1)),
        alloc::boxed::Box::new(RLevCiphertext::new([RLWECiphertext::new(z, z); L_CK])),
    );
    let pp = PublicParams::<
        StubKey,
        N1,
        N2,
        ViaCPolyQ1Rns<N1>,
        ViaCPolyQ3<N2>,
        L_QUERY,
        L_CK,
        L_RSK,
        D,
    >::new(qck2, alloc::boxed::Box::new(rsk), TOY_PARAMS, 2, 2, 2, L_CK);
    assert!(alloc::format!("{pp:?}").contains("<redacted>"));
    assert_eq!(pp.num_rows, 2);

    qck.zeroize();
    assert_eq!(qck.lwe_to_rlwe_key.0, 0);
}

/// `Uncompressed` round-trips a `Vec<u64>`.
#[test]
fn uncompressed_wire_format_round_trip() {
    let orig: alloc::vec::Vec<u64> = alloc::vec![0, 1, 137_438_822_401, u64::MAX];
    let bytes = Uncompressed::serialize(&orig).unwrap();
    let decoded = Uncompressed::deserialize(&bytes).unwrap();
    assert_eq!(orig, decoded);
}

/// `PrgCompressed::regenerate_masks` matches a direct `Shake256Prg` draw — the
/// pinned cross-language KAT contract.
#[test]
fn prg_mask_regen_matches_direct_prg_draw() {
    use via_protocol::primitives::sampling::Shake256Prg;
    let q: u64 = 8_380_417; // VIA-C q3
    let seed = b"protocol-int-mask-seed";
    let regen = PrgCompressed::<32>::regenerate_masks(seed, q, 16);
    let mut prg = Shake256Prg::new(seed);
    let expected: alloc::vec::Vec<u64> = (0..16).map(|_| prg.uniform_below(q)).collect();
    assert_eq!(regen, expected);
}

// Tiny `alloc::vec!`-of-clones helper (the integration crate is std, but we keep
// the alloc-path explicit to mirror the no_std lib).
extern crate alloc;
fn alloc_vec<T: Clone>(value: T, n: usize) -> alloc::vec::Vec<T> {
    alloc::vec![value; n]
}
