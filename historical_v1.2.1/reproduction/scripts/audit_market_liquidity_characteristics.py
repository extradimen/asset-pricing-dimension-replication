#!/usr/bin/env python3
"""Audit V011 weekly risk and daily liquidity definitions against GKX."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V011"
DEFAULT_GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
PERIODS = {
    "full_1957_2021": (date(1957, 1, 1), date(2021, 12, 1)),
    "early_1957_1979": (date(1957, 1, 1), date(1979, 12, 1)),
    "middle_1980_1999": (date(1980, 1, 1), date(1999, 12, 1)),
    "recent_2000_2021": (date(2000, 1, 1), date(2021, 12, 1)),
}
SPECS = {
    "weekly_beta": ("beta", "beta_weekly_3y", [-1, 0, 1]),
    "weekly_betasq": ("betasq", "betasq_weekly_3y", [-1, 0, 1]),
    "weekly_idiovol": ("idiovol", "idiovol_weekly_3y", [-1, 0, 1]),
    "daily_beta": ("beta", "beta_daily_3m", [0, 1, 2]),
    "daily_idiovol": ("idiovol", "idiovol_daily_capm_3m", [0, 1, 2]),
    "daily_std_dolvol": ("std_dolvol", "std_dolvol_3m", [0, 1, 2]),
    "daily_std_turn": ("std_turn", "std_turn_3m", [0, 1, 2]),
    "daily_zerotrade": ("zerotrade", "zerotrade_3m", [0, 1, 2]),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--gkx", type=Path, default=DEFAULT_GKX)
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


def metric(frame: pl.DataFrame, spec: str, target: str, candidate: str, shift: int, period: str) -> dict:
    clean = frame.select("month", target, candidate).drop_nulls()
    ranked = clean.with_columns(
        pl.col(target).rank(method="average").over("month").alias("target_rank"),
        pl.col(candidate).rank(method="average").over("month").alias("candidate_rank"),
        pl.len().over("month").alias("month_n"),
    ).with_columns(
        ((pl.col("target_rank") - pl.col("candidate_rank")).abs() / (pl.col("month_n") - 1))
        .alias("absolute_percentile_error")
    )
    monthly = ranked.group_by("month").agg(
        pl.corr("target_rank", "candidate_rank").alias("rank_correlation"),
        pl.col("absolute_percentile_error").mean().alias("mean_absolute_percentile_error"),
        pl.len().alias("rows"),
    )
    correlations = monthly["rank_correlation"].drop_nulls()
    return {
        "spec": spec, "target": target, "candidate": candidate, "shift_months": shift,
        "period": period, "rows": clean.height, "months": monthly.height,
        "level_correlation": clean.select(pl.corr(target, candidate)).item(),
        "mean_monthly_rank_correlation": correlations.mean(),
        "median_monthly_rank_correlation": correlations.median(),
        "p05_monthly_rank_correlation": correlations.quantile(0.05, interpolation="linear"),
        "mean_absolute_percentile_error": monthly["mean_absolute_percentile_error"].mean(),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    metrics_path = output / "market_liquidity_definition_audit.csv"
    report_path = output / "comparison_report.json"
    manifest_path = output / "comparison_manifest.json"
    products = [metrics_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Audit output exists; use --overwrite")

    weekly_path = output / "weekly_market_characteristics.parquet"
    daily_path = output / "daily_rolling_characteristics.parquet"
    targets = sorted({item[0] for item in SPECS.values()})
    gkx = pl.read_parquet(args.gkx, columns=["permno", "month"] + targets)
    weekly = pl.read_parquet(weekly_path)
    daily = pl.read_parquet(daily_path)
    rows: list[dict] = []
    for spec, (target, candidate, shifts) in SPECS.items():
        source = weekly if spec.startswith("weekly_") else daily
        for shift in shifts:
            shifted = source.select(
                "permno", pl.col("month").dt.offset_by(f"{shift}mo"), candidate
            )
            joined = gkx.select("permno", "month", target).join(
                shifted, on=["permno", "month"], how="inner"
            )
            for period, (start, end) in PERIODS.items():
                subset = joined.filter((pl.col("month") >= start) & (pl.col("month") <= end))
                rows.append(metric(subset, spec, target, candidate, shift, period))
    metrics = pl.DataFrame(rows).with_columns(
        pl.all().map_elements(
            lambda value: None if isinstance(value, float) and not math.isfinite(value) else value,
            return_dtype=pl.self_dtype(),
        )
    )
    metrics.write_csv(metrics_path)
    full = metrics.filter(pl.col("period") == "full_1957_2021")
    decisions = []
    for spec in SPECS:
        best = full.filter(pl.col("spec") == spec).sort(
            "mean_monthly_rank_correlation", descending=True
        ).row(0, named=True)
        decisions.append({
            "spec": spec, "selected_shift_months": best["shift_months"],
            "rows": best["rows"],
            "mean_monthly_rank_correlation": best["mean_monthly_rank_correlation"],
            "p05_monthly_rank_correlation": best["p05_monthly_rank_correlation"],
            "mean_absolute_percentile_error": best["mean_absolute_percentile_error"],
        })
    report = {
        "schema_version": 1, "experiment_id": "P1-G0-V011",
        "decisions": decisions,
        "primary_definition": ["weekly_beta", "weekly_betasq", "weekly_idiovol"],
        "robustness_definitions": [spec for spec in SPECS if spec.startswith("daily_")],
        "interpretation": "The weekly three-year measures reconstruct GKX. Daily three-month measures are retained as economically distinct short-horizon robustness variables. Liquidity measures are CIZ extensions with a documented SIZ/CIZ version bridge.",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V011", "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [args.gkx, weekly_path, daily_path]
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
