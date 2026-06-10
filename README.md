# via-rs

A pure-Rust, `no_std` implementation of the **VIA** family of single-server
Private Information Retrieval (PIR) schemes — [Liu, Wang & Zhang (2025)](https://eprint.iacr.org/2025/2074).

> 📖 **[Full documentation →](https://0xalizk.github.io/via-rs/)**
> A from-first-principles intro to VIA (no crypto background needed) plus an
> implementation/architecture analysis and audit. This README is the quick
> path for developers who want to clone, build, and test.

## What this is

PIR lets a client fetch the `i`-th record of a server-held database while
keeping the index `i` hidden from the server — single-server, with no
non-collusion assumption. VIA's distinctive moves are a logarithmic-depth
**DMux tree** for query expansion and (in VIA-C) a low-noise **LWE→RLWE
conversion** for query compression. See the [docs](https://0xalizk.github.io/via-rs/#intro)
for how and why it works.

**Status:** the **VIA-C** variant is implemented end-to-end; plain VIA and the
batch variant VIA-B are not yet implemented. The crypto core is the focus (no
HTTP/transport layer). See the [implementation analysis](https://0xalizk.github.io/via-rs/#impl)
for what is and isn't covered, and known parameter/correctness caveats.

## Codebase layout

A Cargo workspace of five crates; `via-primitives` is organised as a layered
primitive stack mirroring the paper.

| Crate | Role |
| --- | --- |
| [`crates/via-primitives`](crates/via-primitives) | Crypto primitives, in layers: `algebra` (Zq / RNS / polynomial ring + NTT) → `sampling` (SHAKE-256 PRG, ternary/Gaussian) → `encryption` (RLWE/RLev/RGSW/MLWE, gadget) → `switching` (mod-/ring-/key-switch) → `gates` (CMux, DMux, CRot) → `conversion` (LWE→RLWE cascade). |
| [`crates/via-protocol`](crates/via-protocol) | Parameters, presets (toy / realistic), and wire types (`Query`, `Answer`, keys). |
| [`crates/via-client`](crates/via-client) | `setup` → `query(idx)` → `recover`. |
| [`crates/via-server`](crates/via-server) | `setup_db` → query decompression → first-dimension → answer → response compression. |
| [`crates/via-integration`](crates/via-integration) | End-to-end tests, cross-language KATs, and Criterion benchmarks. |

## Build & test

**Prerequisites:** a Rust toolchain (stable, tested on 1.96). Fuzzing
additionally needs a nightly toolchain plus `cargo install cargo-fuzz`.
Recipes are wired up in the [`Justfile`](Justfile) (`cargo install just`), but
plain `cargo` works too.

```sh
# Clone
git clone https://github.com/0xalizk/via-rs && cd via-rs

# Test (toy parameters — runs by default, fast)
cargo test --workspace            # or: just test
```

The heavy **paper-scale** end-to-end test (n=2048 RNS pipeline) is `#[ignore]`d
because it is slow in debug; run it in release:

```sh
cargo test --release -p via-integration --test client_server_e2e_paper -- --ignored
```

Other common workflows:

```sh
just lint        # cargo fmt + clippy -D warnings   (just lint-check in CI)
just doc         # rustdoc with KaTeX math
just fuzz <T>    # run a cargo-fuzz target (needs nightly + cargo-fuzz)
just             # list every recipe
```

## License

See repository.
