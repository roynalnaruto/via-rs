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

# ─── docs ────────────────────────────────────────────────────────────────

# Build rustdoc with KaTeX math rendering and open in a browser.
doc:
    cargo doc --no-deps --document-private-items --open

# Same as `doc` but without opening a browser — for CI.
doc-build:
    cargo doc --no-deps --document-private-items

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
