#!/usr/bin/env python3
"""Build the stock-day/six-factor panel used to validate Paper 2 beta forecasts."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import polars as pl

from audit_paper2_g0 import FF5, read_factor_file


CLASSIFICATION = ["PrimaryExch", "USIncFlg", "IssuerType", "SecurityType", "SecuritySubType", "ShareType"]
READ_COLUMNS = [
    "PERMNO", "DlyCalDt", "DlyDelFlg", "ConditionalType", "TradingStatusFlg", "DlyRetMissFlg",
    *CLASSIFICATION, "DlyRet",
]
DAILY_INVARIANTS = ["DlyRet", "DlyDelFlg", "DlyRetMissFlg"]
FACTOR_NAMES = ["mkt_rf", "smb", "hml", "rmw", "cma", "rf", "mom"]


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


def load_factors(ff5_path: Path, momentum_path: Path) -> pl.DataFrame:
    _, ff5 = read_factor_file(ff5_path, FF5, 8)
    _, momentum = read_factor_file(momentum_path, ["Mom"], 8)
    common = sorted(set(ff5) & set(momentum))
    rows = []
    for key in common:
        values = [float(value) for value in ff5[key]] + [float(momentum[key][0])]
        rows.append((datetime.strptime(key, "%Y%m%d").date(), *values))
    return pl.DataFrame(rows, schema={"date": pl.Date, **{name: pl.Float64 for name in FACTOR_NAMES}}, orient="row")


def load_eligible_keys(panel: Path, start: str, end: str) -> pl.DataFrame:
    return (
        pl.scan_parquet(panel)
        .select("permno", pl.col("month").alias("feature_month"), "target_month")
        .filter(pl.col("target_month").is_between(pl.lit(start).str.to_date(), pl.lit(end).str.to_date()))
        .unique()
        .collect()
    )


def transform_complete_groups(
    frame: pl.DataFrame,
    eligible: pl.DataFrame,
    factors: pl.DataFrame,
) -> tuple[pl.DataFrame, dict[str, int]]:
    if frame.is_empty():
        return pl.DataFrame(), {}
    active = (pl.col("ConditionalType") == "RW") & (pl.col("TradingStatusFlg") == "A")
    enriched = frame.with_columns(
        active.alias("_active_regular"),
        active.cast(pl.Int8).cum_sum().over("PERMNO").gt(0).alias("_active_seen"),
        *[
            pl.when(active).then(pl.col(column)).otherwise(None).forward_fill().over("PERMNO").alias(f"_info_{column}")
            for column in CLASSIFICATION
        ],
    )
    include = pl.col("_active_regular") | ((pl.col("DlyDelFlg") == "Y") & pl.col("_active_seen"))
    selected = enriched.filter(include)
    if selected.is_empty():
        return pl.DataFrame(), {"rows_in_complete_groups": frame.height, "included_rows": 0}
    keys = ["PERMNO", "DlyCalDt"]
    days = selected.select(keys).unique().height
    variants = selected.select(keys + DAILY_INVARIANTS).unique().height - days
    if variants:
        raise ValueError(f"Found {variants} conflicting same-security-day return records")
    daily = selected.unique(subset=keys, keep="first", maintain_order=True)
    common = (
        pl.col("_info_PrimaryExch").is_in(["N", "A", "Q"])
        & (pl.col("_info_USIncFlg") == "Y")
        & pl.col("_info_IssuerType").is_in(["ACOR", "CORP"])
        & (pl.col("_info_SecurityType") == "EQTY")
        & (pl.col("_info_SecuritySubType") == "COM")
        & (pl.col("_info_ShareType") == "NS")
    )
    common_daily = (
        daily.filter(common)
        .select(
            pl.col("PERMNO").alias("permno"),
            pl.col("DlyCalDt").alias("date"),
            pl.col("DlyCalDt").dt.truncate("1mo").alias("target_month"),
            pl.col("DlyRet").alias("stock_return"),
            (pl.col("DlyDelFlg") == "Y").alias("delist_flag"),
            pl.col("DlyRetMissFlg").alias("return_missing_flag"),
        )
    )
    matched_keys = common_daily.join(eligible, on=["permno", "target_month"], how="inner", validate="m:1")
    result = (
        matched_keys.join(factors, on="date", how="inner", validate="m:1")
        .with_columns((pl.col("stock_return") - pl.col("rf")).alias("stock_excess_return"))
        .select(
            "permno", "feature_month", "target_month", "date", "stock_return", "stock_excess_return",
            "delist_flag", "return_missing_flag", *FACTOR_NAMES,
        )
    )
    return result, {
        "rows_in_complete_groups": frame.height,
        "included_rows": selected.height,
        "duplicate_rows_collapsed": selected.height - days,
        "security_days": days,
        "common_security_days": common_daily.height,
        "eligible_security_days": matched_keys.height,
        "factor_matched_security_days": result.height,
        "missing_stock_returns": result["stock_return"].null_count(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--ff5-daily", type=Path, required=True)
    parser.add_argument("--momentum-daily", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--input-sha256", required=True)
    parser.add_argument("--target-start", default="1963-08-01")
    parser.add_argument("--target-end", default="2019-12-01")
    parser.add_argument("--block-size-mb", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--max-rows", type=int)
    args = parser.parse_args()

    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    pa.set_cpu_count(args.cpu_threads)

    started = time.time()
    output = args.output_dir.resolve()
    product = output / "stock_day_factor_panel.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)

    eligible = load_eligible_keys(args.panel, args.target_start, args.target_end)
    factors = load_factors(args.ff5_daily, args.momentum_daily)
    counters: Counter[str] = Counter()
    writer: pq.ParquetWriter | None = None
    carry: pl.DataFrame | None = None
    unique_permnos: set[int] = set()
    minimum_date: Any = None
    maximum_date: Any = None

    convert = pacsv.ConvertOptions(
        include_columns=READ_COLUMNS,
        column_types={"PERMNO": pa.int64(), "DlyCalDt": pa.date32(), "DlyRet": pa.float64()},
        strings_can_be_null=True,
        null_values=[""],
    )
    read = pacsv.ReadOptions(block_size=args.block_size_mb * 1024 * 1024)
    with zipfile.ZipFile(args.input) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in {args.input}")
        with archive.open(members[0]) as raw:
            reader = pacsv.open_csv(raw, read_options=read, convert_options=convert)
            try:
                for batch in reader:
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
                    result, stats = transform_complete_groups(complete, eligible, factors)
                    counters.update(stats)
                    if not result.is_empty():
                        unique_permnos.update(result["permno"].unique().to_list())
                        low, high = result["date"].min(), result["date"].max()
                        minimum_date = min(minimum_date, low) if minimum_date else low
                        maximum_date = max(maximum_date, high) if maximum_date else high
                        table = result.to_arrow()
                        if writer is None:
                            writer = pq.ParquetWriter(product, table.schema, compression="zstd")
                        writer.write_table(table, row_group_size=250_000)
                    if args.max_rows is not None and counters["raw_rows_read"] >= args.max_rows:
                        break
            finally:
                reader.close()
    if carry is not None and not carry.is_empty():
        result, stats = transform_complete_groups(carry, eligible, factors)
        counters.update(stats)
        if not result.is_empty():
            unique_permnos.update(result["permno"].unique().to_list())
            low, high = result["date"].min(), result["date"].max()
            minimum_date = min(minimum_date, low) if minimum_date else low
            maximum_date = max(maximum_date, high) if maximum_date else high
            table = result.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(product, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250_000)
    if writer is None:
        raise RuntimeError("No eligible stock-day observations were produced")
    writer.close()

    report = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "status": "smoke_completed" if args.max_rows is not None else "completed",
        "partial_run": args.max_rows is not None,
        "input_sha256": args.input_sha256,
        "eligible_stock_months": eligible.height,
        "factor_dates": factors.height,
        **dict(counters),
        "output_rows": pq.ParquetFile(product).metadata.num_rows,
        "unique_permnos": len(unique_permnos),
        "minimum_date": str(minimum_date),
        "maximum_date": str(maximum_date),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "run_id": args.run_id,
        "git_revision": git_revision(),
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
