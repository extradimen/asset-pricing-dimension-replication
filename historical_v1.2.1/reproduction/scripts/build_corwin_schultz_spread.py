#!/usr/bin/env python3
"""Build monthly Corwin-Schultz high-low spread estimates from CRSP CIZ daily data."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/raw/licensed/Daily Stock File_csv.zip")
READ_COLS = [
    "PERMNO",
    "DlyCalDt",
    "ConditionalType",
    "TradingStatusFlg",
    "DlyLow",
    "DlyHigh",
    "DlyBid",
    "DlyAsk",
    "DlyRet",
    "DlyVol",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size-mb", type=int, default=64)
    parser.add_argument("--max-rows", type=int)
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


def transform_complete_groups(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    if frame.is_empty():
        return pl.DataFrame(), {}
    active = frame.filter(
        (pl.col("ConditionalType") == "RW")
        & (pl.col("TradingStatusFlg") == "A")
    )
    keys = ["PERMNO", "DlyCalDt"]
    day_variants = active.select(
        keys + ["DlyLow", "DlyHigh", "DlyBid", "DlyAsk", "DlyRet", "DlyVol"]
    ).unique()
    days = active.select(keys).unique().height
    conflicts = day_variants.height - days
    if conflicts:
        raise ValueError(f"Found {conflicts} conflicting high-low security-days")
    daily = (
        active.unique(subset=keys, keep="first", maintain_order=True)
        .select(
            [
                "PERMNO",
                "DlyCalDt",
                "DlyLow",
                "DlyHigh",
                "DlyBid",
                "DlyAsk",
                "DlyRet",
                "DlyVol",
            ]
        )
        .sort(["PERMNO", "DlyCalDt"])
        .with_columns(
            (pl.col("DlyRet").is_not_null() & pl.col("DlyVol").is_not_null()).alias(
                "_analysis_eligible"
            )
        )
        .with_columns(
            pl.col("DlyCalDt").shift(1).over("PERMNO").alias("_previous_date"),
            pl.col("DlyLow").shift(1).over("PERMNO").alias("_previous_low"),
            pl.col("DlyHigh").shift(1).over("PERMNO").alias("_previous_high"),
            pl.col("_analysis_eligible").shift(1).over("PERMNO").alias("_previous_eligible"),
        )
    )
    valid = (
        (pl.col("DlyLow") > 0)
        & (pl.col("DlyHigh") >= pl.col("DlyLow"))
        & (pl.col("_previous_low") > 0)
        & (pl.col("_previous_high") >= pl.col("_previous_low"))
        & pl.col("_analysis_eligible")
        & pl.col("_previous_eligible")
        & ((pl.col("DlyCalDt") - pl.col("_previous_date")) <= pl.duration(days=7))
    )
    beta = (
        (pl.col("DlyHigh") / pl.col("DlyLow")).log().pow(2)
        + (pl.col("_previous_high") / pl.col("_previous_low")).log().pow(2)
    )
    gamma = (
        pl.max_horizontal("DlyHigh", "_previous_high")
        / pl.min_horizontal("DlyLow", "_previous_low")
    ).log().pow(2)
    denominator = 3.0 - 2.0 * math.sqrt(2.0)
    alpha = (
        (math.sqrt(2.0) * beta.sqrt() - beta.sqrt()) / denominator
        - (gamma / denominator).sqrt()
    )
    daily = daily.with_columns(
        pl.when(valid)
        .then(alpha.clip(lower_bound=0.0))
        .otherwise(None)
        .alias("_alpha")
    ).with_columns(
        (2.0 * (pl.col("_alpha").exp() - 1.0) / (1.0 + pl.col("_alpha").exp())).alias(
            "_cs_spread"
        ),
        pl.when((pl.col("DlyBid") > 0) & (pl.col("DlyAsk") >= pl.col("DlyBid")))
        .then((pl.col("DlyAsk") - pl.col("DlyBid")) / ((pl.col("DlyAsk") + pl.col("DlyBid")) / 2.0))
        .otherwise(None)
        .alias("_quoted_spread"),
        pl.when(
            pl.col("_analysis_eligible")
            & (pl.col("DlyLow") > 0)
            & (pl.col("DlyHigh") >= pl.col("DlyLow"))
        )
        .then((pl.col("DlyHigh") - pl.col("DlyLow")) / ((pl.col("DlyHigh") + pl.col("DlyLow")) / 2.0))
        .otherwise(None)
        .alias("_high_low_spread"),
        pl.col("DlyCalDt").dt.truncate("1mo").alias("month"),
    )
    monthly = (
        daily.group_by(["PERMNO", "month"], maintain_order=True)
        .agg(
            pl.col("_cs_spread").sum().alias("cs_spread_sum"),
            pl.col("_cs_spread").count().cast(pl.Int32).alias("cs_pair_days"),
            pl.col("_cs_spread").mean().alias("cs_spread_monthly"),
            pl.col("_quoted_spread").sum().alias("quoted_spread_sum"),
            pl.col("_quoted_spread").count().cast(pl.Int32).alias("quoted_spread_days"),
            pl.col("_quoted_spread").mean().alias("quoted_spread_monthly"),
            pl.col("_quoted_spread").drop_nulls().last().alias("quoted_spread_month_end"),
            pl.col("_high_low_spread").sum().alias("high_low_spread_sum"),
            pl.col("_high_low_spread").count().cast(pl.Int32).alias("high_low_spread_days"),
            pl.col("_high_low_spread").mean().alias("high_low_spread_monthly"),
        )
        .sort(["PERMNO", "month"])
        .with_columns(
            pl.col("cs_spread_sum")
            .rolling_sum(window_size=3, min_samples=3)
            .over("PERMNO")
            .alias("_sum_3m"),
            pl.col("cs_pair_days")
            .rolling_sum(window_size=3, min_samples=3)
            .over("PERMNO")
            .alias("cs_pair_days_3m"),
            pl.col("quoted_spread_sum")
            .rolling_sum(window_size=3, min_samples=3)
            .over("PERMNO")
            .alias("_quoted_sum_3m"),
            pl.col("quoted_spread_days")
            .rolling_sum(window_size=3, min_samples=3)
            .over("PERMNO")
            .alias("quoted_spread_days_3m"),
            pl.col("high_low_spread_sum")
            .rolling_sum(window_size=3, min_samples=3)
            .over("PERMNO")
            .alias("_high_low_sum_3m"),
            pl.col("high_low_spread_days")
            .rolling_sum(window_size=3, min_samples=3)
            .over("PERMNO")
            .alias("high_low_spread_days_3m"),
            pl.col("month").shift(1).over("PERMNO").alias("_previous_month"),
            pl.col("month").shift(2).over("PERMNO").alias("_previous_month_2"),
        )
        .with_columns(
            pl.when(pl.col("cs_pair_days_3m") > 0)
            .then(pl.col("_sum_3m") / pl.col("cs_pair_days_3m"))
            .otherwise(None)
            .alias("cs_spread_trailing3m")
            ,
            pl.when(pl.col("quoted_spread_days_3m") > 0)
            .then(pl.col("_quoted_sum_3m") / pl.col("quoted_spread_days_3m"))
            .otherwise(None)
            .alias("quoted_spread_trailing3m"),
            pl.when(
                (pl.col("high_low_spread_days_3m") >= 21)
                & (pl.col("_previous_month") == pl.col("month").dt.offset_by("-1mo"))
                & (pl.col("_previous_month_2") == pl.col("month").dt.offset_by("-2mo"))
            )
            .then(pl.col("_high_low_sum_3m") / pl.col("high_low_spread_days_3m"))
            .otherwise(None)
            .alias("high_low_spread_trailing3m"),
        )
        .drop(
            "_sum_3m",
            "_quoted_sum_3m",
            "_high_low_sum_3m",
            "_previous_month",
            "_previous_month_2",
        )
        .rename({"PERMNO": "permno"})
    )
    return monthly, {
        "raw_complete_rows": frame.height,
        "active_rows": active.height,
        "duplicate_active_rows": active.height - days,
        "high_low_conflicts": conflicts,
        "security_days": daily.height,
        "valid_pair_days": int(daily["_cs_spread"].is_not_null().sum()),
        "security_months": monthly.height,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    spread_path = output / "corwin_schultz_monthly.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [spread_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    convert = pacsv.ConvertOptions(
        include_columns=READ_COLS,
        column_types={
            "PERMNO": pa.int64(),
            "DlyCalDt": pa.date32(),
            "DlyLow": pa.float64(),
            "DlyHigh": pa.float64(),
            "DlyBid": pa.float64(),
            "DlyAsk": pa.float64(),
            "DlyRet": pa.float64(),
            "DlyVol": pa.float64(),
        },
        strings_can_be_null=True,
        null_values=[""],
    )
    read = pacsv.ReadOptions(block_size=args.block_size_mb * 1024 * 1024)
    counters: Counter[str] = Counter()
    writer: pq.ParquetWriter | None = None
    carry: pl.DataFrame | None = None
    unique_permnos: set[int] = set()
    month_min = month_max = None
    with zipfile.ZipFile(args.input) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV member, found {len(members)}")
        with archive.open(members[0]) as stream:
            for batch in pacsv.open_csv(stream, read_options=read, convert_options=convert):
                frame = pl.from_arrow(batch)
                if args.max_rows is not None:
                    remaining = args.max_rows - counters["raw_rows_read"]
                    if remaining <= 0:
                        break
                    frame = frame.head(remaining)
                counters["raw_rows_read"] += frame.height
                if carry is not None:
                    frame = pl.concat([carry, frame], how="vertical")
                last_permno = frame.item(-1, "PERMNO")
                complete = frame.filter(pl.col("PERMNO") != last_permno)
                carry = frame.filter(pl.col("PERMNO") == last_permno)
                monthly, stats = transform_complete_groups(complete)
                counters.update(stats)
                if not monthly.is_empty():
                    unique_permnos.update(monthly["permno"].unique().to_list())
                    lo, hi = monthly["month"].min(), monthly["month"].max()
                    month_min = min(month_min, lo) if month_min else lo
                    month_max = max(month_max, hi) if month_max else hi
                    table = monthly.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(spread_path, table.schema, compression="zstd")
                    writer.write_table(table, row_group_size=250_000)
                if args.max_rows is not None and counters["raw_rows_read"] >= args.max_rows:
                    break
    if carry is not None and not carry.is_empty():
        monthly, stats = transform_complete_groups(carry)
        counters.update(stats)
        if not monthly.is_empty():
            unique_permnos.update(monthly["permno"].unique().to_list())
            lo, hi = monthly["month"].min(), monthly["month"].max()
            month_min = min(month_min, lo) if month_min else lo
            month_max = max(month_max, hi) if month_max else hi
            table = monthly.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(spread_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250_000)
    if writer is None:
        raise RuntimeError("No spread observations produced")
    writer.close()

    report = {
        "schema_version": 1,
        "experiment_id": "P1-G0-V009",
        "data_snapshot_id": "wrds-us-equity-2025-12-v1",
        "partial_run": args.max_rows is not None,
        "max_rows": args.max_rows,
        **dict(counters),
        "unique_permnos": len(unique_permnos),
        "month_min": str(month_min),
        "month_max": str(month_max),
        "formula": "Corwin-Schultz (2012) two-day high-low spread; negative alpha truncated to zero",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V009",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "input": {"path": str(args.input.resolve()), "size_bytes": args.input.stat().st_size},
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [spread_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
