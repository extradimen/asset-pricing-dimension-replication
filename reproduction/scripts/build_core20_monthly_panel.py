#!/usr/bin/env python3
"""Assemble the audited Core-20 monthly characteristics and their GKX bridge."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/wrds-us-equity-2025-12-v1"
GKX_DIR = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007"
FEATURES = [
    "baspread", "beta", "betasq", "chmom", "dolvol", "idiovol", "ill", "indmom",
    "maxret", "mom12m", "mom1m", "mom36m", "mom6m", "mvel1", "pricedelay",
    "retvol", "std_dolvol", "std_turn", "turn", "zerotrade",
]
FEATURE_LINEAGE = {
    "baspread": "V009 monthly mean daily high-low range, source month t-1",
    "beta": "V011 three-year weekly equal-weighted-market beta, audited target-month label",
    "betasq": "V011 squared three-year weekly beta, audited target-month label",
    "chmom": "V010 months t-1..t-6 momentum minus t-7..t-12 momentum",
    "dolvol": "log(monthly volume times absolute month-end price), source month t-2",
    "idiovol": "V011 three-year weekly market-model residual volatility, audited target-month label",
    "ill": "monthly Amihud illiquidity divided by one million, source month t-1",
    "indmom": "V012 equal-weighted two-digit-SIC industry mom12m",
    "maxret": "maximum daily return, source month t-1",
    "mom12m": "V010 compounded returns t-2..t-12",
    "mom1m": "monthly return, source month t-1",
    "mom36m": "V010 compounded returns t-13..t-36",
    "mom6m": "V010 compounded returns t-2..t-6",
    "mvel1": "month-end market capitalization, source month t-1",
    "pricedelay": "V013 Hou-Moskowitz D1, formation month shifted two months",
    "retvol": "monthly daily-return standard deviation, source month t-1",
    "std_dolvol": "V011 three-month daily log-dollar-volume volatility",
    "std_turn": "V011 three-month daily turnover volatility",
    "turn": "three-month mean monthly turnover divided by 100, window ending t-1",
    "zerotrade": "V011 three-month zero-trading measure",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def shifted(frame: pl.LazyFrame, months: int) -> pl.LazyFrame:
    return frame.with_columns(pl.col("month").dt.offset_by(f"{months}mo"))


def rank_expression(feature: str) -> pl.Expr:
    count = pl.col(feature).count().over("month")
    rank = pl.col(feature).rank(method="average").over("month")
    return (
        pl.when(pl.col(feature).is_not_null() & (count > 1))
        .then(2.0 * (rank - 1.0) / (count - 1.0) - 1.0)
        .otherwise(0.0)
        .cast(pl.Float32)
        .alias(f"x_{feature}")
    )


def main() -> None:
    args = parse_args(); started = time.time(); output = args.output_dir.resolve()
    self_path = output / "core20_self_built.parquet"
    bridge_path = output / "core20_bridged_raw.parquet"
    model_path = output / "core20_model_input.parquet"
    report_path = output / "quality_report.json"; manifest_path = output / "output_manifest.json"
    products = [self_path, bridge_path, model_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path in products: path.unlink(missing_ok=True)

    master_path = PROCESSED / "P1-G0-V006/us_equity_research_master.parquet"
    spread_path = PROCESSED / "P1-G0-V009/corwin_schultz_monthly.parquet"
    momentum_path = PROCESSED / "P1-G0-V010/momentum_characteristics.parquet"
    weekly_path = PROCESSED / "P1-G0-V011/weekly_market_characteristics.parquet"
    daily_path = PROCESSED / "P1-G0-V011/daily_rolling_characteristics.parquet"
    industry_path = PROCESSED / "P1-G0-V012/industry_momentum.parquet"
    delay_path = PROCESSED / "P1-G0-V013/price_delay.parquet"
    gkx_path = GKX_DIR / "gkx_core94_raw.parquet"
    inputs = [master_path, spread_path, momentum_path, weekly_path, daily_path, industry_path, delay_path, gkx_path]

    master = pl.scan_parquet(master_path)
    keys = master.select("permno", "month", "ret_fwd1")
    lag1 = shifted(master.select(
        "permno", "month",
        pl.col("market_cap").alias("mvel1"),
        pl.col("ret").alias("mom1m"),
        pl.col("max_daily_return").alias("maxret"),
        pl.col("daily_return_std").alias("retvol"),
        (pl.col("amihud_million") / 1_000_000.0).alias("ill"),
    ), 1)
    dolvol = shifted(master.select(
        "permno", "month",
        pl.when((pl.col("volume") > 0) & (pl.col("prc").abs() > 0))
        .then((pl.col("volume") * pl.col("prc").abs()).log()).otherwise(None).alias("dolvol"),
    ), 2)
    turnover = master.select("permno", "month", "turnover").with_columns(
        pl.col("turnover").rolling_mean(3, min_samples=3).over("permno").alias("_turn3"),
        pl.col("month").shift(2).over("permno").alias("_month_lag2"),
    ).with_columns(
        pl.when(pl.col("_month_lag2") == pl.col("month").dt.offset_by("-2mo"))
        .then(pl.col("_turn3") / 100.0).otherwise(None).alias("turn")
    ).select("permno", "month", "turn")
    turnover = shifted(turnover, 1)
    spread = shifted(pl.scan_parquet(spread_path).select(
        "permno", "month", pl.col("high_low_spread_monthly").alias("baspread")
    ), 1)
    momentum = pl.scan_parquet(momentum_path).select("permno", "month", "chmom", "mom6m", "mom12m", "mom36m")
    weekly = pl.scan_parquet(weekly_path).select(
        "permno", "month", pl.col("beta_weekly_3y").alias("beta"),
        pl.col("betasq_weekly_3y").alias("betasq"), pl.col("idiovol_weekly_3y").alias("idiovol")
    )
    daily = pl.scan_parquet(daily_path).select(
        "permno", "month", pl.col("std_dolvol_3m").alias("std_dolvol"),
        pl.col("std_turn_3m").alias("std_turn"), pl.col("zerotrade_3m").alias("zerotrade")
    )
    industry = pl.scan_parquet(industry_path).select("permno", "month", "indmom")
    delay = shifted(pl.scan_parquet(delay_path).select("permno", "month", "pricedelay"), 2)

    self_built = keys
    for part in [lag1, dolvol, turnover, spread, momentum, weekly, daily, industry, delay]:
        self_built = self_built.join(part, on=["permno", "month"], how="left")
    self_built = self_built.select("permno", "month", "ret_fwd1", *FEATURES).sort(["month", "permno"]).collect()
    self_built.write_parquet(self_path, compression="zstd")

    official = pl.scan_parquet(gkx_path).select(
        "permno", "month", *[pl.col(feature).alias(f"gkx_{feature}") for feature in FEATURES]
    ).collect()
    joined = self_built.join(official, on=["permno", "month"], how="left")
    bridge = joined.with_columns(
        *[pl.coalesce(pl.col(f"gkx_{feature}"), pl.col(feature)).alias(feature) for feature in FEATURES],
        pl.sum_horizontal(*[pl.col(f"gkx_{feature}").is_not_null().cast(pl.UInt8) for feature in FEATURES])
        .alias("official_feature_count"),
    ).with_columns(
        pl.when(pl.col("official_feature_count") == len(FEATURES)).then(pl.lit("official_full"))
        .when(pl.col("official_feature_count") > 0).then(pl.lit("official_partial"))
        .otherwise(pl.lit("self_built")).alias("feature_source")
    ).select("permno", "month", "ret_fwd1", "feature_source", "official_feature_count", *FEATURES)
    bridge.write_parquet(bridge_path, compression="zstd")
    model = bridge.select(
        "permno", "month", "ret_fwd1", "feature_source", "official_feature_count",
        *[rank_expression(feature) for feature in FEATURES],
        *[pl.col(feature).is_null().cast(pl.Int8).alias(f"missing_{feature}") for feature in FEATURES],
    )
    model.write_parquet(model_path, compression="zstd")

    nonmissing_self = self_built.select(*[pl.col(c).is_not_null().sum().alias(c) for c in FEATURES]).row(0, named=True)
    nonmissing_bridge = bridge.select(*[pl.col(c).is_not_null().sum().alias(c) for c in FEATURES]).row(0, named=True)
    source_counts = {row[0]: row[1] for row in bridge.group_by("feature_source").len().iter_rows()}
    report = {
        "schema_version": 1, "experiment_id": "P1-G0-V014", "rows": bridge.height,
        "unique_keys": bridge.select(pl.struct("permno", "month").n_unique()).item(),
        "first_month": str(bridge["month"].min()), "last_month": str(bridge["month"].max()),
        "features": FEATURES, "feature_lineage": FEATURE_LINEAGE, "self_built_nonmissing": nonmissing_self,
        "bridged_nonmissing": nonmissing_bridge, "source_counts": source_counts,
        "bridge_rule": "Official GKX value where available through 2021; audited self-built value otherwise, including 2022-2025.",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V014", "git_revision": git_revision(), "command": " ".join(os.sys.argv),
        "inputs": [{"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs],
        "outputs": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in [self_path, bridge_path, model_path, report_path]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
