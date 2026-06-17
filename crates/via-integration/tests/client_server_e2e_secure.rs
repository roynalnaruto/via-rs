//! **Secure-scale** (≥120-bit) in-memory client ↔ server e2e.
//!
//! The n1=4096 analogue of `client_server_e2e_paper`: it runs the full VIA-C
//! protocol round-trip at the `SECURE_PARAMS` instantiation
//! (`docs/params-concrete.html`): `n1 = 4096`, `n2 = 1024`, `d = 4`,
//! `q1 ≈ 2^74.7` (two-prime RNS, ≡1 mod 8192) `≫ q2 ≈ 2^34 ≫ q3 ≈ 2^23 ≫
//! q4 = 2^15`, `p = 16`, erratum gadget bases (DMux 18073, CMux/CRot 307,
//! ring-switch 4), ternary keys.
//!
//! It is an empirical correctness witness for the n1=4096 instantiation — it
//! exercises the new `LweToRlweKeyRnsN4096` cascade and the whole gate pipeline
//! at the secure dimensions/moduli, on a small (2×2) database (so the noise
//! budget here is far slacker than the 32 GiB analysis the parameters target —
//! this proves the *mechanics* close, not the worst-case budget).
//!
//! `#[ignore]` — heavy: the n4096 RNS cascade key is ~54 MB and the schoolbook
//! O(n²) pipeline at n=4096 runs for minutes. Run with:
//!
//! ```text
//! cargo test -p via-integration --release --test client_server_e2e_secure -- --ignored
//! ```

use via_client::Client;
use via_primitives::algebra::ring::RingPoly;
use via_primitives::algebra::ring::element::Poly;
use via_primitives::algebra::ring::form::Coefficient;
use via_primitives::algebra::ring::rns_element::PolyRns;
use via_primitives::algebra::rns::basis::paper::ViaSecQ1Rns;
use via_primitives::algebra::zq::modulus::paper::{ViaSecP, ViaSecQ2, ViaSecQ3, ViaSecQ4};
use via_primitives::conversion::{LweToRlweKeyRnsN4096, gen_lwe_to_rlwe_key_rns_n4096_boxed};
use via_primitives::params::{ViaCPolyP, ViaCPolyQ2, ViaCPolyQ3};
use via_primitives::sampling::distribution::Distribution;
use via_primitives::sampling::prg::Shake256Prg;
use via_primitives::switching::gen_rsk;
use via_primitives::switching::rekey::rekey_secret_key;
use via_protocol::{PublicParams, SECURE_PARAMS};
use via_server::{ServerConfig, ViaCServer};

const N1: usize = 4096;
const N2: usize = 1024;
const D: usize = 4; // d = N1 / N2
const L_QUERY: usize = 2;
const L_CK: usize = 18;
const L_RSK: usize = 8;
// Conversion-key gadget base: 18^18 ≈ 2^75 fully covers q1 ≈ 2^74.7 (the
// proven-correct coverage; the erratum's noise-optimal B=11 is approximate and
// unnecessary at this 2×2 database). Security is independent of this base.
const CK_BASE: u64 = 18;
// Ring-switch gadget base (erratum / SECURE_PARAMS.gadget_base_rsk).
const RSK_BASE: u64 = 4;

const NUM_ROWS: usize = 2; // I
const NUM_COLS: usize = 2; // J

// Secure ring instantiation (q1 = ViaSecQ1Rns, q4 = ViaSecQ4 = 2^15; q2/q3/p
// are the same ConstModulus types as VIA-C and reuse the ViaCPoly* aliases).
type R1 = PolyRns<N1, ViaSecQ1Rns, Coefficient>; // S1 @ q1-RNS, n1
type R2N1 = ViaCPolyQ2<N1>; // q2 @ n1 (DMux output / FirstDim)
type R3N1 = ViaCPolyQ3<N1>; // q3 @ n1 (mod_switch_sym intermediate + rekey target)
type R3N2 = ViaCPolyQ3<N2>; // q3 @ n2 (S2 ring + answer mask)
type R4N2 = Poly<N2, ViaSecQ4, Coefficient>; // q4 @ n2 (answer body, q4 = 2^15)
type RpN1 = ViaCPolyP<N1>; // p @ n1 (DB cell embed target)
type Rec = ViaCPolyP<N2>; // p @ n2 (records)
type K = LweToRlweKeyRnsN4096<ViaSecQ1Rns, L_CK>;

type SecClient = Client<N1, N2, R1, R3N2, L_QUERY, L_CK, L_RSK, D>;
type SecServer = ViaCServer<K, N1, N2, R1, R2N1, R3N2, R4N2, L_QUERY, L_CK, L_RSK, D>;
type SecPp = PublicParams<K, N1, N2, R1, R3N2, L_QUERY, L_CK, L_RSK, D>;

fn secure_setup(prg: &mut Shake256Prg) -> (SecClient, SecPp) {
    SecClient::setup(
        ViaSecQ1Rns::default(),
        ViaSecQ3::default(),
        SECURE_PARAMS,
        NUM_ROWS,
        NUM_COLS,
        CK_BASE,
        Distribution::Ternary,
        Distribution::Ternary,
        Distribution::Ternary,
        prg,
        gen_lwe_to_rlwe_key_rns_n4096_boxed::<ViaSecQ1Rns, L_CK>,
        |sk1, sk2, dist, prg| {
            let q3_mod = RingPoly::modulus(sk2.poly());
            let s1_q3 = rekey_secret_key::<N1, R1, R3N1>(sk1, q3_mod);
            gen_rsk::<N1, N2, R3N1, R3N2, L_RSK, D>(&s1_q3, sk2, RSK_BASE, dist, prg)
        },
    )
    .expect("client setup")
}

fn record(m: usize, p: ViaSecP) -> Rec {
    let coeffs: [u64; N2] =
        core::array::from_fn(|j| if j < 4 { ((m + 1 + j) % 16) as u64 } else { 0 });
    Poly::new(p, coeffs)
}

fn round_trip(index: usize) -> (Rec, Rec) {
    let q1 = ViaSecQ1Rns::default();
    let q2 = ViaSecQ2::default();
    let q3 = ViaSecQ3::default();
    let q4 = ViaSecQ4::default();
    let p = ViaSecP::default();
    let mut prg = Shake256Prg::new(b"via-c-secure-120-e2e");

    let (client, pp) = secure_setup(&mut prg);

    let records: Vec<Rec> = (0..D * NUM_ROWS * NUM_COLS).map(|m| record(m, p)).collect();
    let server = SecServer::setup::<RpN1, Rec>(ServerConfig::new(pp, q1, q2, q3, q4), &records, p)
        .expect("server setup");

    let query = client.query(index, &mut prg).expect("client query");
    let answer = server.answer(&query).expect("server answer");
    let recovered: Rec = client
        .recover::<R3N2, R4N2, Rec>(&answer, q3, q4, p)
        .expect("client recover");

    (recovered, records[index])
}

/// Index 15 = the full gate path (DMux + CMux + both CRot bits all selecting)
/// at the secure n1=4096 moduli/depths.
#[test]
#[ignore = "secure-scale n4096 RNS pipeline — heavy; run with --release -- --ignored"]
fn client_server_e2e_secure_index_15() {
    let stack_mb: usize = std::env::var("VIA_E2E_STACK_MB")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    std::thread::Builder::new()
        .stack_size(stack_mb << 20)
        .spawn(|| {
            let (got, want) = round_trip(15);
            assert_eq!(
                got, want,
                "secure-scale recover(query(15)) must equal record[15]"
            );
        })
        .expect("spawn secure-scale thread")
        .join()
        .expect("secure-scale e2e thread panicked");
}
