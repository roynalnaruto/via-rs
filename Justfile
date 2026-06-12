# Justfile — common project commands.
#
# Run `just` (no args) to list all recipes. Every recipe runs from the repo root.

# Show the list of recipes.
default:
    @just --list

# ─── check / build / test ────────────────────────────────────────────────

# Fast type-check across all targets — no codegen.
check:
    cargo check --all-targets

# Build the crate. Pass `--release` (e.g. `just build --release`) for optimised.
build *FLAGS:
    cargo build {{FLAGS}}

# Run the full test suite (unit tests + doctests). Extra flags forwarded.
test *FLAGS:
    cargo test {{FLAGS}}

# Type-check under every variant feature set — guards against via-c regression
# when via-b is introduced, and asserts the via-b surface compiles.
check-variants:
    cargo check --workspace --all-targets
    cargo check --workspace --all-targets --features via-c
    cargo check --workspace --all-targets --features via-b
    # via-primitives stays no-alloc-clean under via-c (the repack module is
    # `cfg(all(via-b, alloc))`, so it is fully absent here — the zero-cost gate).
    cargo check --package via-primitives --features via-c

# Run the test suite under the default build and the via-b feature (the new
# gated surface) — unit tests + doctests for both.
test-variants:
    cargo test --workspace
    cargo test --workspace --features via-b

# ─── bench ───────────────────────────────────────────────────────────────
#
# Per-step + full-pipeline perf benchmarks (criterion). Use the save/compare
# pair to justify a change: `just bench-save before`, make the change, then
# `just bench-cmp before` — criterion reports each step's % delta + whether it
# is a statistically significant regression.

# Run the fast (toy-scale) per-step suite — VIA-C + VIA-B batch (seconds).
bench:
    cargo bench --bench pipeline_toy
    cargo bench --bench pipeline_batch_toy --features via-b

# Run the paper-scale suite (n2048 RNS; reduced sampling; minutes) — on demand.
bench-paper:
    cargo bench --bench pipeline_paper
    cargo bench --bench pipeline_batch_paper --features via-b

# Save the current fast-suite timings as a named baseline (run BEFORE a change).
bench-save NAME="main":
    cargo bench --bench pipeline_toy -- --save-baseline {{NAME}}

# Compare current fast-suite timings against a saved baseline (run AFTER a change).
bench-cmp NAME="main":
    cargo bench --bench pipeline_toy -- --baseline {{NAME}}

# Run the via-primitives kernel micro-benchmarks (NTT-mediated vs schoolbook
# gadget product, raw NTT round-trip) — isolates the primitives the pipeline
# benches reach only transitively.
bench-primitives *FLAGS:
    cargo bench --package via-primitives --bench kernels {{FLAGS}}

# ─── docs ────────────────────────────────────────────────────────────────

# Build rustdoc for the full workspace (KaTeX math rendering) and open in a browser.
# `--features alloc,via-b` mirrors the docs.rs feature sets: alloc unlocks the
# paper-scale boxed paths in via-primitives; via-b documents the full variant
# surface in via-client / via-server / via-protocol.
doc:
    cargo doc --no-deps --document-private-items --features alloc,via-b --open

# Same as `doc` but without opening a browser — for CI.
doc-build:
    cargo doc --no-deps --document-private-items --features alloc,via-b

# ─── lint ────────────────────────────────────────────────────────────────

# Format in-place + run clippy (warnings denied). Use before pushing.
lint:
    cargo fmt --all
    cargo clippy --all-targets -- -D warnings
    cd fuzz && cargo fmt --all
    cd fuzz && cargo clippy --all-targets -- -D warnings

# CI-friendly variant: check formatting without modifying files; deny clippy warnings.
lint-check:
    cargo fmt --all -- --check
    cargo clippy --all-targets -- -D warnings
    cd fuzz && cargo fmt --all -- --check
    cd fuzz && cargo clippy --all-targets -- -D warnings

# ─── no_std ────────────────────────────────────────────────────────────────

# Verify via-primitives builds for a bare-metal target (no std, no alloc).
no-std-check:
    cargo build --target thumbv7em-none-eabihf --package via-primitives

# ─── client ⊥ server structural check ────────────────────────────────────

# NB: detect PRESENCE with `grep -q` + `if` — `grep -qv` is not a correct test
# (on GNU grep it exits 0 whenever any line lacks the pattern).
# Assert that via-client and via-server have no transitive dep on each other.
client-server-check:
    @if cargo tree --package via-client 2>&1 | grep -q "via-server"; then \
        echo "FAIL: via-client depends on via-server"; exit 1; fi
    @if cargo tree --package via-server 2>&1 | grep -q "via-client"; then \
        echo "FAIL: via-server depends on via-client"; exit 1; fi
    @echo "OK: client ⊥ server isolation confirmed"

# ─── fuzz ────────────────────────────────────────────────────────────────
#
# Requires `cargo install cargo-fuzz` and a nightly toolchain.

# List all available fuzz targets.
fuzz-list:
    cd fuzz && cargo +nightly fuzz list

# Build every fuzz target (compile-only) — default + the VIA-B (§7) targets.
fuzz-build:
    cd fuzz && cargo +nightly fuzz build
    cd fuzz && cargo +nightly fuzz build --features via-b

# Run a single fuzz target for SECS seconds (default 60). Example: `just fuzz zq_reduce 300`.
fuzz TARGET SECS="60":
    cd fuzz && cargo +nightly fuzz run {{TARGET}} -- -max_total_time={{SECS}}

# Run a single VIA-B (§7) fuzz target for SECS seconds (e.g. `just fuzz-b conversion_repack_roundtrip`).
fuzz-b TARGET SECS="60":
    cd fuzz && cargo +nightly fuzz run --features via-b {{TARGET}} -- -max_total_time={{SECS}}

# ─── KAT vectors ───────────────────────────────────────────────────────────
#
# Requires Python >= 3.11 with the `.references/via-spec` reference on the
# path (the script inserts it itself).

# Regenerate the Layer-3 cross-language KAT constants in crates/via-primitives/tests/data/.
regen-kats-layer3:
    cd .references/via-spec && python3 scripts/gen_layer3_kats.py

# Regenerate the Layer-4 cross-language KAT constants in crates/via-primitives/tests/data/.
regen-kats-layer4:
    cd .references/via-spec && python3 scripts/gen_layer4_kats.py

# Regenerate the Layer-5 cross-language KAT constants in crates/via-primitives/tests/data/.
regen-kats-layer5:
    cd .references/via-spec && python3 scripts/gen_layer5_kats.py

# Regenerate the Layer-6 cross-language KAT constants in crates/via-primitives/tests/data/.
regen-kats-layer6:
    cd .references/via-spec && python3 scripts/gen_layer6_kats.py

# Layer-7 (VIA-B) has NO KATs by design: the Python reference (.references/via-spec)
# implements VIA/VIA-C only — no via_b, no repack/Extr_d, no gen_layer7. With no
# oracle, cross-language KATs are impossible, and a Rust-golden snapshot would just
# duplicate the seeded batch_e2e_{toy,paper} regression anchors. See the decision
# record at crates/via-integration/tests/data/layer7_kats.rs.
regen-kats-layer7:
    @echo "Layer-7 (VIA-B) has no KATs by design — no Python VIA-B oracle; the seeded batch_e2e_{toy,paper} tests are the anchors. See tests/data/layer7_kats.rs."
