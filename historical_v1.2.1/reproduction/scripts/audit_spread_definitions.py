#!/usr/bin/env python3
"""Compare CRSP spread definitions and timing against the GKX baspread series."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_research_overlap.parquet"
DEFAULT_SPREADS = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V009/corwin_schultz_monthly.parquet"
CANDIDATES = [
    "high_low_spread_monthly",
    "high_low_spread_trailing3m",
    "cs_spread_monthly",
    "cs_spread_trailing3m",
    "quoted_spread_monthly",
    "quoted_spread_month_end",
    "quoted_spread_trailing3m",
]
PERIODS = {
    "full_1957_2021": (date(1957, 1, 1), date(2021, 12, 1)),
    "early_1957_1979": (date(1957, 1, 1), date(1979, 12, 1)),
    "middle_1980_1999": (date(1980, 1, 1), date(1999, 12, 1)),
    "recent_2000_2021": (date(2000, 1, 1), date(2021, 12, 1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gkx", type=Path, default=DEFAULT_GKX)
    parser.add_argument("--spreads", type=Path, default=DEFAULT_SPREADS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    metrics_path = output / "spread_definition_audit.csv"
    report_path = output / "comparison_report.json"
    manifest_path = output / "comparison_manifest.json"
    products = [metrics_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    gkx = pl.scan_parquet(args.gkx).select("permno", "month", "baspread")
    spreads = pl.scan_parquet(args.spreads).select(["permno", "month"] + CANDIDATES)
    rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        target = gkx.filter((pl.col("month") >= start) & (pl.col("month") <= end))
        for candidate in CANDIDATES:
            for lag in [0, 1, 2]:
                values = (
                    spreads.select(
                        "permno",
                        pl.col("month").dt.offset_by(f"{lag}mo"),
                        pl.col(candidate).alias("candidate"),
                    )
                    .join(target, on=["permno", "month"], how="inner")
                    .drop_nulls(["baspread", "candidate"])
                    .collect()
                )
                ranked = values.with_columns(
                    pl.col("baspread").rank(method="average").over("month").alias("target_rank"),
                    pl.col("candidate").rank(method="average").over("month").alias("candidate_rank"),
                    pl.len().over("month").alias("month_n"),
                ).with_columns(
                    ((pl.col("target_rank") - pl.col("candidate_rank")).abs() / (pl.col("month_n") - 1))
                    .alias("absolute_percentile_error")
                )
                monthly = ranked.group_by("month").agg(
                    pl.corr("target_rank", "candidate_rank").alias("rank_correlation"),
                    pl.col("absolute_percentile_error").mean().alias("percentile_error"),
                )
                correlations = monthly["rank_correlation"].drop_nulls()
                rows.append(
                    {
                        "candidate": candidate,
                        "lag_months": lag,
                        "period": period,
                        "rows": values.height,
                        "months": monthly.height,
                        "mean_monthly_rank_correlation": correlations.mean(),
                        "median_monthly_rank_correlation": correlations.median(),
                        "p05_monthly_rank_correlation": correlations.quantile(0.05, interpolation="linear"),
                        "mean_absolute_percentile_error": monthly["percentile_error"].mean(),
                    }
                )
    metrics = pl.DataFrame(rows)
    metrics.write_csv(metrics_path)
    full = metrics.filter(pl.col("period") == "full_1957_2021").sort(
        "mean_monthly_rank_correlation", descending=True
    )
    best = full.row(0, named=True)
    report = {
        "schema_version": 1,
        "experiment_id": "P1-G0-V009",
        "best_definition": best,
        "decision": {
            "primary_baspread": "high_low_spread_monthly lagged one month",
            "corwin_schultz_role": "robustness only",
            "quoted_spread_role": "distinct transaction-cost measure; not a GKX baspread substitute",
            "version_bridge": "Use the self-constructed CIZ definition for the full 1950-2025 model panel; use GKX only to audit historical ordering.",
        },
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V009",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [metrics_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
