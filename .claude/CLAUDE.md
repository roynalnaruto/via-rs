# via-rs

A pure-Rust, `no_std` implementation of the **VIA** family of single-server
Private Information Retrieval (PIR) schemes.

## Background

**Private Information Retrieval (PIR)** lets a client fetch the `i`-th entry
of a server-held database while keeping the index `i` hidden from the server.
A single-server PIR scheme must achieve this without trusting the server and
without relying on non-collusion assumptions across multiple servers.

**VIA** (Liu, Wang, Zhang) is a recent lattice-based single-server PIR scheme
that eliminates the offline / preprocessing phase while still achieving
`O_λ(log N)` online communication. Its design rests on two ingredients:

- a **DMux–CMux** structure that replaces coefficient expansion with a
  logarithmic-depth tree of homomorphic muxes, generating an encrypted
  one-hot selection vector with no precomputed hints; and
- a novel **LWE-to-RLWE conversion** with logarithmic-sized public
  parameters and only `O(n log n)` noise growth, used for query compression.

The paper defines three variants, all of which this project targets:

- **VIA** — no offline communication; `Õ(1)` online communication.
- **VIA-C** — permits offline communication for further query/response
  compression via the new LWE-to-RLWE conversion.
- **VIA-B** — batch-query extension of VIA-C using homomorphic repacking
  (MLWEs-to-RLWE), optimised for many small-record queries.

Reference: [`.docs/via.pdf`](.docs/via.pdf).

## Scope

This crate aims to implement all three variants (VIA, VIA-C, VIA-B) in
idiomatic, `no_std`-compatible Rust, with constant-time arithmetic where
secret-dependent, and with the cryptographic primitives (RLWE, key-switching,
DMux, CMux, ring-switching, LWE-to-RLWE conversion) exposed as reusable
building blocks.

## Build / lint / test / doc / fuzz

Common workflows are wired up in the [`Justfile`](Justfile). Run `just` with
no arguments to list every recipe; the most-used ones:

| Command            | What it does                                              |
| ------------------ | --------------------------------------------------------- |
| `just check`       | `cargo check --all-targets` (fast type-check, no codegen) |
| `just build`       | `cargo build` (append `--release` for optimised)          |
| `just test`        | Full test suite — unit tests + doctests                   |
| `just lint`        | `cargo fmt` in place, then `cargo clippy -D warnings`     |
| `just lint-check`  | CI variant — format check only, deny clippy warnings      |
| `just doc`         | Build rustdoc with KaTeX math and open in a browser       |
| `just doc-build`   | Same as `doc` without opening (for CI)                    |
| `just fuzz-list`   | List available `cargo-fuzz` targets                       |
| `just fuzz-build`  | Compile every fuzz target                                 |
| `just fuzz T [S]`  | Run fuzz target `T` for `S` seconds (default 60)          |

Fuzzing requires a nightly toolchain and `cargo install cargo-fuzz`.
