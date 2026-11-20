#!/usr/bin/env python3
"""Build and audit the four GKX return-history characteristics through 2025."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V006/us_equity_research_master.parquet"
DEFAULT_GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_research_overlap.parquet"
FEATURE_WINDOWS = {
    "mom6m": {"positive": [2, 3, 4, 5, 6]},
    "mom12m": {"positive": list(range(2, 13))},
    "mom36m": {"positive": list(range(13, 37))},
    "chmom": {"positive": list(range(1, 7)), "negative": list(range(7, 13))},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--gkx", type=Path, default=DEFAULT_GKX)
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


def compounded_return(lags: list[int]) -> pl.Expr:
    value = pl.lit(1.0)
    for lag in lags:
        value = value * (1.0 + pl.col("ret").shift(lag).over("permno"))
    return value - 1.0


def continuous_window(lags: list[int]) -> pl.Expr:
    first, last = min(lags), max(lags)
    return (
        (pl.col("month").shift(first).over("permno") == pl.col("month").dt.offset_by(f"-{first}mo"))
        & (pl.col("month").shift(last).over("permno") == pl.col("month").dt.offset_by(f"-{last}mo"))
    )


def feature_expression(feature: str) -> pl.Expr:
    spec = FEATURE_WINDOWS[feature]
    lags = spec["positive"] + spec.get("negative", [])
    positive = compounded_return(spec["positive"])
    value = positive
    if "negative" in spec:
        value = positive - compounded_return(spec["negative"])
    return pl.when(continuous_window(lags)).then(value).otherwise(None).alias(feature)


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    panel_path = output / "momentum_characteristics.parquet"
    audit_path = output / "momentum_replication_audit.csv"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [panel_path, audit_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    panel = (
        pl.scan_parquet(args.master)
        .select("permno", "month", "ret")
        .sort(["permno", "month"])
        .with_columns(*[feature_expression(feature) for feature in FEATURE_WINDOWS])
        .select(["permno", "month"] + list(FEATURE_WINDOWS))
        .collect()
    )
    panel.write_parquet(panel_path, compression="zstd", row_group_size=100_000)
    gkx = pl.scan_parquet(args.gkx).select(["permno", "month"] + list(FEATURE_WINDOWS)).collect()
    overlap = gkx.join(panel, on=["permno", "month"], how="inner", suffix="_self")
    audit_rows = []
    for feature in FEATURE_WINDOWS:
        candidate = f"{feature}_self"
        values = overlap.select(["month", feature, candidate]).drop_nulls()
        ranked = values.with_columns(
            pl.col(feature).rank(method="average").over("month").alias("target_rank"),
            pl.col(candidate).rank(method="average").over("month").alias("candidate_rank"),
            pl.len().over("month").alias("month_n"),
        ).with_columns(
            ((pl.col("target_rank") - pl.col("candidate_rank")).abs() / (pl.col("month_n") - 1))
            .alias("percentile_error")
        )
        monthly = ranked.group_by("month").agg(
            pl.corr("target_rank", "candidate_rank").alias("rank_correlation"),
            pl.col("percentile_error").mean().alias("percentile_error"),
        )
        correlations = monthly["rank_correlation"].drop_nulls()
        audit_rows.append(
            {
                "feature": feature,
                "formation_window": json.dumps(FEATURE_WINDOWS[feature]),
                "rows": values.height,
                "months": monthly.height,
                "mean_monthly_rank_correlation": correlations.mean(),
                "median_monthly_rank_correlation": correlations.median(),
                "p05_monthly_rank_correlation": correlations.quantile(0.05, interpolation="linear"),
                "mean_absolute_percentile_error": monthly["percentile_error"].mean(),
            }
        )
    audit = pl.DataFrame(audit_rows)
    audit.write_csv(audit_path)
    report = {
        "schema_version": 1,
        "experiment_id": "P1-G0-V010",
        "panel_rows": panel.height,
        "panel_unique_keys": panel.select(pl.struct("permno", "month").n_unique()).item(),
        "month_min": str(panel["month"].min()),
        "month_max": str(panel["month"].max()),
        "feature_nonmissing_rows": {
            feature: int(panel[feature].is_not_null().sum()) for feature in FEATURE_WINDOWS
        },
        "feature_windows": FEATURE_WINDOWS,
        "replication": {row["feature"]: row for row in audit_rows},
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V010",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [panel_path, audit_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
