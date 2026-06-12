#!/usr/bin/env python3
"""Build a green/red base-vs-branch per-phase comparison table for `/benchmark`.

After benching the PR base with `cargo bench … -- --save-baseline base` and the
PR head with `cargo bench … -- --baseline base`, criterion leaves, per bench id,
three files under `target/criterion/<id>/`:

    base/estimates.json    mean.point_estimate  -> base-branch time (ns)
    new/estimates.json     mean.point_estimate  -> PR-head time (ns)
    change/estimates.json  mean.point_estimate  -> relative change (fraction; + = slower)

This reads them for every paper-scale phase and writes a sticky PR-comment body
(`target/bench_compare_comment.md`) with two grouped tables — VIA-C (`paper/…`)
and VIA-B (`paper_batch/…`) — each row `| Phase | Base | Branch | Δ |` flagged
🟢 faster / 🔴 slower / ⚪ within noise.

stdlib only (Python 3 on ubuntu-latest). Wall-clock CI timings are noisy, so the
table is advisory — hence the noise band and the verify-locally note. Distinct
marker from `bench_regression_comment.py` so the two stickies never collide.
"""

from __future__ import annotations

import json
import os
import pathlib

MARKER = "<!-- via-bench-compare -->"
# |Δ| within this band reads as noise (⚪), not an improvement/regression.
NOISE_PCT = float(os.environ.get("BENCH_NOISE_PCT", "3"))
CRITERION = pathlib.Path("target/criterion")
ROWS = os.environ.get("VIA_BENCH_ROWS", "?")
COLS = os.environ.get("VIA_BENCH_COLS", "?")


def read_mean_ns(path: pathlib.Path) -> float | None:
    """`mean.point_estimate` (nanoseconds) from a criterion estimates.json, or None."""
    try:
        return json.loads(path.read_text())["mean"]["point_estimate"]
    except (OSError, ValueError, KeyError):
        return None


def full_id(bench_dir: pathlib.Path) -> str:
    """Human bench id (e.g. `paper/04_first_dim`) from benchmark.json, else dir name."""
    for candidate in (bench_dir / "new" / "benchmark.json", bench_dir / "benchmark.json"):
        try:
            meta = json.loads(candidate.read_text())
            return meta.get("full_id") or meta.get("title") or bench_dir.name
        except (OSError, ValueError):
            continue
    return bench_dir.name


def human_ns(ns: float | None) -> str:
    """Render a nanosecond duration as ns/µs/ms/s with 3 significant figures."""
    if ns is None:
        return "—"
    for unit, scale in (("s", 1e9), ("ms", 1e6), ("µs", 1e3), ("ns", 1.0)):
        if ns >= scale:
            return f"{ns / scale:.3g} {unit}"
    return f"{ns:.3g} ns"


def collect() -> list[tuple[str, float | None, float | None, float]]:
    """(full_id, base_ns, branch_ns, pct_change) for every paper phase with a baseline."""
    rows: list[tuple[str, float | None, float | None, float]] = []
    for change in CRITERION.glob("**/change/estimates.json"):  # recursive (defensive)
        bench_dir = change.parent.parent
        name = full_id(bench_dir)
        # Only the paper-scale suites belong in this table (ignore stale toy/etc.).
        if not (name.startswith("paper/") or name.startswith("paper_batch/")):
            continue
        pct = read_mean_ns(change)  # change point_estimate is already a fraction
        if pct is None:
            continue
        base = read_mean_ns(bench_dir / "base" / "estimates.json")
        branch = read_mean_ns(bench_dir / "new" / "estimates.json")
        rows.append((name, base, branch, pct * 100.0))
    return rows


def emoji(pct: float) -> str:
    if pct < -NOISE_PCT:
        return "🟢"  # faster = improvement
    if pct > NOISE_PCT:
        return "🔴"  # slower = regression
    return "⚪"


def table(title: str, rows: list[tuple[str, float | None, float | None, float]]) -> list[str]:
    if not rows:
        return []
    rows = sorted(rows, key=lambda r: r[0])  # by phase id → 01_, 02_, … order
    out = [f"#### {title}", "", "| Phase | Base | Branch | Δ |", "| --- | --- | --- | --- |"]
    for name, base, branch, pct in rows:
        sign = "+" if pct >= 0 else "−"
        out.append(
            f"| `{name}` | {human_ns(base)} | {human_ns(branch)} | "
            f"{emoji(pct)} {sign}{abs(pct):.1f}% |"
        )
    out.append("")
    return out


def main() -> None:
    rows = collect()
    via_c = [r for r in rows if r[0].startswith("paper/")]
    via_b = [r for r in rows if r[0].startswith("paper_batch/")]

    lines = [
        MARKER,
        "### Paper-scale benchmark comparison — base vs branch",
        "",
        f"Grid **I×J = {ROWS}×{COLS}**, same runner, criterion `--baseline base`. "
        f"🟢 faster · 🔴 slower · ⚪ within ±{NOISE_PCT:.0f}% noise.",
        "",
    ]
    lines += table("VIA-C (per-phase)", via_c)
    lines += table("VIA-B (batch)", via_b)
    if not via_c and not via_b:
        lines.append(
            "_No comparison data — the base branch lacked the bench targets "
            "(first-introduction PR). Re-run `/benchmark` once this lands on the default branch._"
        )
    lines += [
        "",
        "_Wall-clock CI timings are noisy, so this is advisory. Reproduce locally at the "
        "same grid with_ `VIA_BENCH_ROWS=… VIA_BENCH_COLS=… cargo bench … -- --save-baseline base` "
        "_(on base), then_ `… -- --baseline base` _(on this branch)._",
    ]

    body = "\n".join(lines) + "\n"
    out = pathlib.Path("target/bench_compare_comment.md")  # under the gitignored target/
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(body)


if __name__ == "__main__":
    main()
