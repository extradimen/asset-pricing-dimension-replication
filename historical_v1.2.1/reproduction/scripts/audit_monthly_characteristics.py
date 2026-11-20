#!/usr/bin/env python3
"""Audit independently reconstructed CRSP monthly characteristics against GKX."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
DEFAULT_MASTER = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V006/us_equity_research_master.parquet"

FEATURE_SPECS = {
    "mvel1": {"sources": ["market_cap"], "formula": "market_cap"},
    "mom1m": {"sources": ["ret"], "formula": "ret"},
    "maxret": {"sources": ["max_daily_return"], "formula": "max_daily_return"},
    "retvol": {"sources": ["daily_return_std"], "formula": "daily_return_std"},
    "ill": {"sources": ["amihud_million"], "formula": "amihud_million / 1e6"},
    "turn": {"sources": ["turnover"], "formula": "three-month mean turnover / 100"},
    "baspread": {"sources": ["mean_quoted_spread"], "formula": "mean_quoted_spread"},
    "dolvol": {"sources": ["volume", "prc"], "formula": "log(volume * abs(prc))"},
}
PERIODS = {
    "full_1957_2021": (date(1957, 1, 1), date(2021, 12, 1)),
    "early_1957_1979": (date(1957, 1, 1), date(1979, 12, 1)),
    "middle_1980_1999": (date(1980, 1, 1), date(1999, 12, 1)),
    "recent_2000_2021": (date(2000, 1, 1), date(2021, 12, 1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gkx", type=Path, default=DEFAULT_GKX)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
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


def candidate_expression(feature: str) -> pl.Expr:
    if feature == "ill":
        return pl.col("amihud_million") / 1_000_000.0
    if feature == "turn":
        return pl.col("turnover_3m") / 100.0
    if feature == "dolvol":
        value = pl.col("volume") * pl.col("prc").abs()
        return pl.when(value > 0).then(value.log()).otherwise(None)
    return pl.col(FEATURE_SPECS[feature]["sources"][0])


def build_comparison(gkx_path: Path, master_path: Path) -> pl.DataFrame:
    features = list(FEATURE_SPECS)
    sources = sorted({source for item in FEATURE_SPECS.values() for source in item["sources"]})
    gkx = pl.scan_parquet(gkx_path).select(["permno", "month"] + features)
    master = (
        pl.scan_parquet(master_path)
        .filter((pl.col("month") >= date(1956, 11, 1)) & (pl.col("month") <= date(2021, 12, 1)))
        .select(["permno", "month", "primaryexch"] + sources)
        .sort(["permno", "month"])
        .with_columns(
            pl.col("turnover")
            .rolling_mean(window_size=3, min_samples=3)
            .over("permno")
            .alias("turnover_3m")
        )
    )
    same = master.select(
        ["permno", "month", "primaryexch"]
        + [
            candidate_expression(feature).alias(f"candidate_{feature}_same")
            for feature in FEATURE_SPECS
        ]
    )
    joined = gkx.join(same, on=["permno", "month"], how="inner")
    for lag in [1, 2]:
        lagged = master.select(
            pl.col("permno"),
            pl.col("month").dt.offset_by(f"{lag}mo"),
            pl.col("primaryexch").alias(f"primaryexch_lag{lag}"),
            *[
                candidate_expression(feature).alias(f"candidate_{feature}_lag{lag}")
                for feature in FEATURE_SPECS
            ],
        )
        joined = joined.join(lagged, on=["permno", "month"], how="left")
    return joined.sort(["month", "permno"]).collect()


def metric_row(frame: pl.DataFrame, feature: str, timing: str, period: str) -> dict[str, object]:
    candidate = f"candidate_{feature}_{timing}"
    clean = frame.select(["month", feature, candidate]).drop_nulls()
    if clean.is_empty():
        return {
            "feature": feature,
            "timing": timing,
            "period": period,
            "rows": 0,
            "months": 0,
        }
    ranked = clean.with_columns(
        pl.col(feature).rank(method="average").over("month").alias("_target_rank"),
        pl.col(candidate).rank(method="average").over("month").alias("_candidate_rank"),
        pl.len().over("month").alias("_month_n"),
    ).with_columns(
        pl.when(pl.col("_month_n") > 1)
        .then((pl.col("_target_rank") - pl.col("_candidate_rank")).abs() / (pl.col("_month_n") - 1))
        .otherwise(0.0)
        .alias("_absolute_percentile_error")
    )
    monthly = ranked.group_by("month").agg(
        pl.corr("_target_rank", "_candidate_rank").alias("rank_correlation"),
        pl.col("_absolute_percentile_error").mean().alias("mean_absolute_percentile_error"),
        pl.len().alias("rows"),
    )
    correlations = monthly["rank_correlation"].drop_nulls()
    source = FEATURE_SPECS[feature]
    return {
        "feature": feature,
        "source": "+".join(source["sources"]),
        "transform": source["formula"],
        "timing": timing,
        "period": period,
        "rows": clean.height,
        "months": monthly.height,
        "pearson_level_correlation": clean.select(pl.corr(feature, candidate)).item(),
        "mean_monthly_rank_correlation": correlations.mean(),
        "median_monthly_rank_correlation": correlations.median(),
        "p05_monthly_rank_correlation": correlations.quantile(0.05, interpolation="linear"),
        "mean_absolute_percentile_error": monthly["mean_absolute_percentile_error"].mean(),
        "median_monthly_rows": monthly["rows"].median(),
    }


def finite_or_none(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    metrics_path = output / "monthly_characteristic_audit.csv"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [metrics_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    comparison = build_comparison(args.gkx, args.master)
    rows: list[dict[str, object]] = []
    for period, (start, end) in PERIODS.items():
        period_frame = comparison.filter((pl.col("month") >= start) & (pl.col("month") <= end))
        for feature in FEATURE_SPECS:
            for timing in ["same", "lag1", "lag2"]:
                rows.append(metric_row(period_frame, feature, timing, period))
    metrics = pl.DataFrame(rows).with_columns(pl.all().map_elements(finite_or_none, return_dtype=pl.self_dtype()))
    metrics.write_csv(metrics_path)

    full = metrics.filter(pl.col("period") == "full_1957_2021")
    timing_decisions = []
    for feature in FEATURE_SPECS:
        candidates = full.filter(pl.col("feature") == feature).sort(
            "mean_monthly_rank_correlation", descending=True, nulls_last=True
        )
        best = candidates.row(0, named=True)
        timing_decisions.append(
            {
                "feature": feature,
                "selected_timing": best["timing"],
                "mean_monthly_rank_correlation": best["mean_monthly_rank_correlation"],
                "mean_absolute_percentile_error": best["mean_absolute_percentile_error"],
            }
        )
    report = {
        "schema_version": 1,
        "experiment_id": "P1-G0-V008",
        "data_snapshot_ids": ["wrds-us-equity-2025-12-v1", "gkx-datashare-2021-v1"],
        "comparison_rows": comparison.height,
        "comparison_unique_keys": comparison.select(pl.struct("permno", "month").n_unique()).item(),
        "month_min": str(comparison["month"].min()),
        "month_max": str(comparison["month"].max()),
        "feature_specs": FEATURE_SPECS,
        "timing_decisions": timing_decisions,
        "interpretation_rule": "Timing is selected only for the replication audit. Formula acceptance and 2022-2025 extension require feature-specific validation, especially for turnover and the Corwin-Schultz bid-ask spread.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V008",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size}
            for path in [args.gkx, args.master]
        ],
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [metrics_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
