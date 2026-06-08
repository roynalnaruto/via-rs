#!/usr/bin/env python3
"""Turn criterion `--baseline` comparison output into a PR-comment body.

After `cargo bench --bench pipeline_toy -- --baseline base`, criterion writes
the per-step relative change to `target/criterion/<id>/change/estimates.json`
(`mean.point_estimate` is a fraction; positive = slower). This reads them,
flags every step slower than BENCH_REGRESSION_THRESHOLD_PCT (default 10),
writes `comment.md`, and emits `regressed=true|false` to $GITHUB_OUTPUT.

stdlib only (Python 3 on ubuntu-latest). Wall-clock CI timings are noisy, so
this is advisory — hence the conservative threshold and the verify-locally note.
"""

from __future__ import annotations

import json
import os
import pathlib

MARKER = "<!-- via-bench-regression -->"
THRESHOLD = float(os.environ.get("BENCH_REGRESSION_THRESHOLD_PCT", "10"))
CRITERION = pathlib.Path("target/criterion")


def bench_title(bench_dir: pathlib.Path) -> str:
    """Human bench id (e.g. `toy/04_first_dim`) from benchmark.json, else dir name."""
    for candidate in (bench_dir / "new" / "benchmark.json", bench_dir / "benchmark.json"):
        try:
            meta = json.loads(candidate.read_text())
            return meta.get("full_id") or meta.get("title") or bench_dir.name
        except (OSError, ValueError):
            continue
    return bench_dir.name


def collect() -> list[tuple[str, float]]:
    """(bench id, percent change) for every step with a baseline comparison."""
    rows: list[tuple[str, float]] = []
    for change in CRITERION.glob("*/change/estimates.json"):
        bench_dir = change.parent.parent
        try:
            pct = json.loads(change.read_text())["mean"]["point_estimate"] * 100.0
        except (OSError, ValueError, KeyError):
            continue
        rows.append((bench_title(bench_dir), pct))
    return rows


def main() -> None:
    rows = collect()
    regressed = sorted((r for r in rows if r[1] > THRESHOLD), key=lambda r: -r[1])

    lines = [MARKER]
    if regressed:
        lines += [
            f"### ⚠️ Wall-clock benchmark regression vs `main` (> {THRESHOLD:.0f}%)",
            "",
            "| Step | Slower by |",
            "| --- | --- |",
            *(f"| `{name}` | **+{pct:.1f}%** |" for name, pct in regressed),
            "",
            "_Wall-clock CI timings are noisy, so this is advisory. Reproduce locally with_ "
            "`just bench-save before` _(on `main`), then_ `just bench-cmp before` _(on this branch)._",
        ]
    else:
        lines.append(f"### ✅ No wall-clock benchmark regression > {THRESHOLD:.0f}% vs `main`.")

    body = "\n".join(lines) + "\n"
    out = pathlib.Path("target/bench_regression_comment.md")  # under the gitignored target/
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(body)
    print(body)

    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a", encoding="utf-8") as fh:
            fh.write(f"regressed={'true' if regressed else 'false'}\n")


if __name__ == "__main__":
    main()
