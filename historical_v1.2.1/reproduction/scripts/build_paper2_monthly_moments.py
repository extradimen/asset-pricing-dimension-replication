#!/usr/bin/env python3
"""Aggregate the Paper 2 stock-day panel into auditable stock-month moments."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import pyarrow.parquet as pq


FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def moment_expressions() -> list[pl.Expr]:
    expressions: list[pl.Expr] = [
        pl.len().cast(pl.Int16).alias("n_days"),
        pl.col("date").min().alias("first_date"),
        pl.col("date").max().alias("last_date"),
        pl.col("delist_flag").any().alias("has_delist_day"),
        pl.col("stock_excess_return").sum().alias("sum_y"),
        pl.col("stock_excess_return").pow(2).sum().alias("sum_y2"),
    ]
    for factor in FACTORS:
        expressions.extend([
            pl.col(factor).sum().alias(f"sum_{factor}"),
            (pl.col(factor) * pl.col("stock_excess_return")).sum().alias(f"sum_{factor}_y"),
        ])
    for left_index, left in enumerate(FACTORS):
        for right in FACTORS[left_index:]:
            expressions.append((pl.col(left) * pl.col(right)).sum().alias(f"sum_{left}_{right}"))
    return expressions


def monthly_moments(frame: pl.LazyFrame, minimum_days: int) -> pl.LazyFrame:
    valid = frame.filter(
        pl.col("stock_excess_return").is_not_null()
        & pl.all_horizontal([pl.col(factor).is_not_null() for factor in FACTORS])
    )
    return (
        valid.group_by("permno", "feature_month", "target_month")
        .agg(moment_expressions())
        .with_columns((pl.col("n_days") >= minimum_days).alias("future_validation_eligible"))
        .sort("permno", "target_month")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--minimum-days", type=int, default=15)
    args = parser.parse_args()
    if args.minimum_days < 2:
        raise ValueError("minimum-days must be at least two")

    started = time.time()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    product = output / "stock_month_moments.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"

    source = pl.scan_parquet(args.input)
    source_stats = source.select(
        pl.len().alias("input_rows"),
        pl.col("stock_excess_return").is_null().sum().alias("missing_stock_returns"),
        pl.any_horizontal([pl.col(factor).is_null() for factor in FACTORS]).sum().alias("missing_factor_rows"),
    ).collect().row(0, named=True)
    if source_stats["missing_factor_rows"]:
        raise ValueError("Factor missingness is not permitted")
    monthly_moments(source, args.minimum_days).sink_parquet(product, compression="zstd", mkdir=True)

    result = pl.scan_parquet(product).select(
        pl.len().alias("stock_months"),
        pl.struct("permno", "target_month").n_unique().alias("unique_keys"),
        pl.col("permno").n_unique().alias("unique_permnos"),
        pl.col("target_month").min().alias("minimum_target_month"),
        pl.col("target_month").max().alias("maximum_target_month"),
        pl.col("future_validation_eligible").sum().alias("eligible_stock_months"),
        pl.col("n_days").min().alias("minimum_days_observed"),
        pl.col("n_days").median().alias("median_days_observed"),
        pl.col("n_days").max().alias("maximum_days_observed"),
    ).collect().row(0, named=True)
    serializable = {key: str(value) if hasattr(value, "isoformat") else value for key, value in result.items()}
    report = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "status": "completed",
        "input_sha256": args.input_sha256,
        "minimum_days_rule": args.minimum_days,
        **source_stats,
        **serializable,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if report["stock_months"] != report["unique_keys"]:
        raise ValueError("Duplicate stock-month keys in output")
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "command": " ".join(os.sys.argv),
        "outputs": [
            {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [product, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
