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

# ─── bench ───────────────────────────────────────────────────────────────
#
# Per-step + full-pipeline perf benchmarks (criterion). Use the save/compare
# pair to justify a change: `just bench-save before`, make the change, then
# `just bench-cmp before` — criterion reports each step's % delta + whether it
# is a statistically significant regression.

# Run the fast (toy-scale) per-step benchmark suite (seconds).
bench:
    cargo bench --bench pipeline_toy

# Run the paper-scale suite (n2048 RNS; reduced sampling; minutes) — on demand.
bench-paper:
    cargo bench --bench pipeline_paper

# Save the current fast-suite timings as a named baseline (run BEFORE a change).
bench-save NAME="main":
    cargo bench --bench pipeline_toy -- --save-baseline {{NAME}}

# Compare current fast-suite timings against a saved baseline (run AFTER a change).
bench-cmp NAME="main":
    cargo bench --bench pipeline_toy -- --baseline {{NAME}}

# ─── docs ────────────────────────────────────────────────────────────────

# Build rustdoc for via-primitives (KaTeX math rendering) and open in a browser.
# `--features alloc` so the paper-scale `…_boxed` builders + n2048 path are
# documented and their intra-doc links resolve.
doc:
    cargo doc --no-deps --document-private-items --package via-primitives --features alloc --open

# Same as `doc` but without opening a browser — for CI.
doc-build:
    cargo doc --no-deps --document-private-items --package via-primitives --features alloc

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

# Build every fuzz target (compile-only).
fuzz-build:
    cd fuzz && cargo +nightly fuzz build

# Run a single fuzz target for SECS seconds (default 60). Example: `just fuzz zq_reduce 300`.
fuzz TARGET SECS="60":
    cd fuzz && cargo +nightly fuzz run {{TARGET}} -- -max_total_time={{SECS}}

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
