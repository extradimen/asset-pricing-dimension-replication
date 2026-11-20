#!/usr/bin/env python3
"""Build the frozen common real-data input for Paper 2 exposure models."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

try:
    from audit_paper2_g0 import FF5, read_factor_file
except ModuleNotFoundError:  # Support package-style imports in tests.
    from scripts.audit_paper2_g0 import FF5, read_factor_file


FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def month_date(key: str) -> date:
    return date(int(key[:4]), int(key[4:6]), 1)


def factor_state_rows(
    ff5_monthly: dict[str, np.ndarray],
    momentum_monthly: dict[str, np.ndarray],
    ff5_daily: dict[str, np.ndarray],
) -> list[dict[str, object]]:
    common = sorted(set(ff5_monthly) & set(momentum_monthly))
    monthly = {
        key: np.concatenate([ff5_monthly[key][:5], momentum_monthly[key][:1]])
        for key in common
    }
    daily_dates = np.asarray([
        np.datetime64(f"{key[:4]}-{key[4:6]}-{key[6:]}") for key in sorted(ff5_daily)
    ])
    daily_values = np.vstack([ff5_daily[key] for key in sorted(ff5_daily)])
    rows: list[dict[str, object]] = []
    for index, key in enumerate(common):
        row: dict[str, object] = {"feature_month": month_date(key)}
        for factor_index, factor in enumerate(FACTORS):
            for window in [1, 3, 12]:
                values = [monthly[item][factor_index] for item in common[max(0, index - window + 1) : index + 1]]
                row[f"z_{factor}_return_{window}m"] = (
                    float(np.prod(1.0 + np.asarray(values)) - 1.0) if len(values) == window else None
                )
            values12 = [monthly[item][factor_index] for item in common[max(0, index - 11) : index + 1]]
            row[f"z_{factor}_volatility_12m"] = (
                float(np.std(values12, ddof=1)) if len(values12) == 12 else None
            )
        current_month = np.datetime64(f"{key[:4]}-{key[4:6]}", "M")
        current = current_month.astype("datetime64[D]")
        next_month = (current_month + 1).astype("datetime64[D]")
        trailing_start = (current_month - 11).astype("datetime64[D]")
        current_mask = (daily_dates >= current) & (daily_dates < next_month)
        trailing_mask = (daily_dates >= trailing_start) & (daily_dates < next_month)
        row["z_market_realized_volatility_1m"] = (
            float(np.sqrt(np.square(daily_values[current_mask, 0]).sum())) if current_mask.any() else None
        )
        if trailing_mask.sum() and index >= 11:
            total_market = daily_values[trailing_mask, 0] + daily_values[trailing_mask, 5]
            wealth = np.cumprod(1.0 + total_market)
            peaks = np.maximum.accumulate(np.concatenate([[1.0], wealth]))[1:]
            row["z_market_drawdown_12m"] = float(np.min(wealth / peaks - 1.0))
        else:
            row["z_market_drawdown_12m"] = None
        row["state_history_complete"] = index >= 11 and all(
            row[f"z_{factor}_return_12m"] is not None for factor in FACTORS
        )
        rows.append(row)
    return rows


def weighted_mean(column: str) -> pl.Expr:
    valid_weight = pl.when(pl.col(column).is_not_null()).then(pl.col("market_cap")).otherwise(0.0)
    return (
        (pl.col(column).fill_null(0.0) * pl.col("market_cap")).sum()
        / valid_weight.sum()
    ).alias(f"z_value_weighted_{column.removeprefix('x_')}")


def build_states(
    core: pl.LazyFrame,
    daily_characteristics: pl.LazyFrame,
    factor_states: pl.DataFrame,
) -> pl.LazyFrame:
    realized_cross_section = (
        core.select("permno", pl.col("target_month").alias("feature_month"), pl.col("ret_fwd1").alias("known_return"))
        .filter(pl.col("known_return").is_not_null())
        .group_by("feature_month")
        .agg(
            pl.col("known_return").std().alias("z_cross_section_return_dispersion_1m"),
            (pl.col("known_return") > 0).mean().alias("z_cross_section_positive_breadth_1m"),
            pl.col("known_return").mean().alias("z_cross_section_equal_weight_return_1m"),
        )
    )
    characteristic_aggregates = core.group_by(pl.col("month").alias("feature_month")).agg(
        weighted_mean("x_bm"), weighted_mean("x_roaq"), weighted_mean("x_agr")
    )
    median_idiovol = daily_characteristics.group_by(pl.col("month").alias("feature_month")).agg(
        pl.col("idiovol_daily_capm_3m").median().alias("z_cross_section_median_idiovol_3m")
    )
    return (
        factor_states.lazy()
        .join(realized_cross_section, on="feature_month", how="left")
        .join(characteristic_aggregates, on="feature_month", how="left")
        .join(median_idiovol, on="feature_month", how="left")
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--core86", type=Path, required=True)
    parser.add_argument("--daily-characteristics", type=Path, required=True)
    parser.add_argument("--moments", type=Path, required=True)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--ff5-monthly", type=Path, required=True)
    parser.add_argument("--momentum-monthly", type=Path, required=True)
    parser.add_argument("--ff5-daily", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    product = output / "exposure_common_input.parquet"

    core = pl.scan_parquet(args.core86)
    schema = core.collect_schema()
    company_features = [name for name in schema.names() if name.startswith("x_")]
    missing_features = [name for name in schema.names() if name.startswith("missing_")]
    if len(company_features) != 86 or len(missing_features) != 86:
        raise ValueError("Core-86 feature or missing-indicator count changed")
    _, ff5_monthly = read_factor_file(args.ff5_monthly, FF5, 6)
    _, momentum_monthly = read_factor_file(args.momentum_monthly, ["Mom"], 6)
    _, ff5_daily = read_factor_file(args.ff5_daily, FF5, 8)
    factor_states = pl.DataFrame(factor_state_rows(ff5_monthly, momentum_monthly, ff5_daily))
    states = build_states(core, pl.scan_parquet(args.daily_characteristics), factor_states)
    state_columns = [name for name in states.collect_schema().names() if name.startswith("z_")]

    core_selected = core.select(
        "permno", pl.col("month").alias("feature_month"), "target_month",
        pl.col("market_cap").log1p().cast(pl.Float32).alias("log_market_cap"),
        *company_features, *missing_features,
    )
    moments = pl.scan_parquet(args.moments)
    b0 = pl.scan_parquet(args.b0).select(
        "permno", "feature_month", "target_month", "b0_eligible", "history_days", "history_months",
        "alpha", *[f"beta_{factor}" for factor in FACTORS], "target_rmse", "target_r2",
    )
    panel = (
        core_selected
        .filter(pl.col("feature_month").is_between(date(1963, 7, 1), date(2019, 11, 1)))
        .join(moments, on=["permno", "feature_month", "target_month"], how="inner")
        .join(b0, on=["permno", "feature_month", "target_month"], how="left")
        .join(states, on="feature_month", how="left")
        .filter(pl.col("future_validation_eligible") & pl.col("state_history_complete"))
        .with_columns(
            pl.when(pl.col("feature_month") <= date(1999, 12, 1)).then(pl.lit("train"))
            .when(pl.col("feature_month") <= date(2009, 12, 1)).then(pl.lit("validation"))
            .otherwise(pl.lit("development_oos")).alias("split")
        )
        .sort("feature_month", "permno")
    )
    panel.sink_parquet(product, compression="zstd", mkdir=True)
    result = pl.scan_parquet(product)
    output_schema = result.collect_schema().names()
    if "ret_fwd1" in output_schema:
        raise ValueError("Forbidden forward return leaked into exposure input")
    stats = result.select(
        pl.len().alias("rows"),
        pl.struct("permno", "feature_month").n_unique().alias("unique_keys"),
        pl.col("permno").n_unique().alias("permnos"),
        pl.col("feature_month").min().alias("minimum_feature_month"),
        pl.col("feature_month").max().alias("maximum_feature_month"),
        (pl.col("target_month") != pl.col("feature_month").dt.offset_by("1mo")).sum().alias("timing_mismatches"),
        pl.col("split").eq("train").sum().alias("train_rows"),
        pl.col("split").eq("validation").sum().alias("validation_rows"),
        pl.col("split").eq("development_oos").sum().alias("development_oos_rows"),
        pl.col("b0_eligible").sum().alias("b0_eligible_rows"),
        pl.all_horizontal([pl.col(name).is_not_null() for name in state_columns]).sum().alias("complete_state_rows"),
    ).collect().row(0, named=True)
    serializable = {key: str(value) if hasattr(value, "isoformat") else value for key, value in stats.items()}
    report = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "status": "completed",
        "company_features": len(company_features),
        "missing_indicators": len(missing_features),
        "state_features": len(state_columns),
        "forbidden_columns_present": [name for name in ["ret_fwd1"] if name in output_schema],
        **serializable,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    if report["rows"] != report["unique_keys"] or report["timing_mismatches"] != 0:
        raise ValueError("Exposure input key or timing audit failed")
    if report["complete_state_rows"] != report["rows"]:
        raise ValueError("Incomplete state row survived eligibility filter")
    report_path = output / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest_path = output / "output_manifest.json"
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "command": " ".join(os.sys.argv),
        "outputs": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in [product, report_path]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
