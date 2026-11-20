#!/usr/bin/env python3
"""Stream CRSP CIZ into reusable security-week and equal-weighted market-week panels."""

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

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/raw/licensed/Daily Stock File_csv.zip")
READ_COLS = ["PERMNO", "DlyCalDt", "ConditionalType", "TradingStatusFlg", "DlyRet", "ewretd"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size-mb", type=int, default=128)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument("--week-convention", choices=["natural", "thursday"], default="natural")
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


def compound(column: str) -> pl.Expr:
    return (pl.col(column).drop_nulls() + 1.0).product() - 1.0


def thursday_week(column: str) -> pl.Expr:
    """Label Thursday-to-Wednesday return weeks by their Thursday start."""
    return ((pl.col(column) - pl.duration(days=3)).dt.truncate("1w") + pl.duration(days=3))


def week_label(column: str, convention: str) -> pl.Expr:
    return pl.col(column).dt.truncate("1w") if convention == "natural" else thursday_week(column)


def security_weeks(frame: pl.DataFrame, convention: str) -> tuple[pl.DataFrame, dict[str, int]]:
    active = frame.filter(
        (pl.col("ConditionalType") == "RW") & (pl.col("TradingStatusFlg") == "A")
    )
    daily = (
        active.unique(subset=["PERMNO", "DlyCalDt"], keep="first", maintain_order=True)
        .filter(pl.col("DlyRet").is_not_null())
        .with_columns(week_label("DlyCalDt", convention).alias("week"))
    )
    weekly = daily.group_by(["PERMNO", "week"], maintain_order=True).agg(
        pl.col("DlyCalDt").max().alias("week_end"),
        compound("DlyRet").alias("stock_weekly_return"),
        pl.len().cast(pl.Int16).alias("stock_trading_days"),
    ).rename({"PERMNO": "permno"})
    return weekly, {
        "raw_complete_rows": frame.height, "active_rows": active.height,
        "eligible_security_days": daily.height, "security_weeks": weekly.height,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    security_path = output / "security_weekly_returns.parquet"
    market_path = output / "market_weekly_returns.parquet"
    report_path = output / "weekly_panel_quality_report.json"
    manifest_path = output / "weekly_panel_output_manifest.json"
    products = [security_path, market_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    convert = pacsv.ConvertOptions(
        include_columns=READ_COLS,
        column_types={"PERMNO": pa.int64(), "DlyCalDt": pa.date32(), "DlyRet": pa.float64(), "ewretd": pa.float64()},
        strings_can_be_null=True, null_values=[""],
    )
    read = pacsv.ReadOptions(block_size=args.block_size_mb * 1024 * 1024)
    counters: Counter[str] = Counter()
    writer: pq.ParquetWriter | None = None
    carry: pl.DataFrame | None = None
    market_by_date: dict[object, float] = {}
    unique_permnos: set[int] = set()

    def collect_market(frame: pl.DataFrame) -> None:
        values = frame.select("DlyCalDt", "ewretd").drop_nulls().unique()
        conflicts = values.group_by("DlyCalDt").len().filter(pl.col("len") > 1)
        if conflicts.height:
            raise ValueError(f"Conflicting market returns on {conflicts.height} dates")
        for day, value in values.iter_rows():
            old = market_by_date.get(day)
            if old is not None and old != value:
                raise ValueError(f"Conflicting market return for {day}")
            market_by_date[day] = value

    def write_frame(frame: pl.DataFrame) -> None:
        nonlocal writer
        weekly, stats = security_weeks(frame, args.week_convention)
        counters.update(stats)
        if weekly.is_empty():
            return
        unique_permnos.update(weekly["permno"].unique().to_list())
        table = weekly.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(security_path, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=250_000)

    with zipfile.ZipFile(args.input) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV, found {len(members)}")
        with archive.open(members[0]) as stream:
            for batch in pacsv.open_csv(stream, read_options=read, convert_options=convert):
                frame = pl.from_arrow(batch)
                if args.max_rows is not None:
                    remaining = args.max_rows - counters["raw_rows_read"]
                    if remaining <= 0:
                        break
                    frame = frame.head(remaining)
                counters["raw_rows_read"] += frame.height
                collect_market(frame)
                if carry is not None:
                    frame = pl.concat([carry, frame], how="vertical")
                last_permno = frame.item(-1, "PERMNO")
                complete = frame.filter(pl.col("PERMNO") != last_permno)
                carry = frame.filter(pl.col("PERMNO") == last_permno)
                write_frame(complete)
                if args.max_rows is not None and counters["raw_rows_read"] >= args.max_rows:
                    break
    if carry is not None and not carry.is_empty():
        write_frame(carry)
    if writer is None:
        raise RuntimeError("No weekly observations produced")
    writer.close()

    market_daily = pl.DataFrame({
        "date": list(market_by_date.keys()), "ewretd": list(market_by_date.values())
    }).sort("date").with_columns(week_label("date", args.week_convention).alias("week"))
    market_weekly = market_daily.group_by("week", maintain_order=True).agg(
        pl.col("date").max().alias("market_week_end"), compound("ewretd").alias("market_weekly_return"),
        pl.len().cast(pl.Int16).alias("market_trading_days"),
    ).sort("week")
    market_weekly.write_parquet(market_path, compression="zstd")
    security_stats = pl.scan_parquet(security_path).select(
        pl.len().alias("rows"), pl.struct("permno", "week").n_unique().alias("unique_keys"),
        pl.col("week").min().alias("min_week"), pl.col("week").max().alias("max_week"),
    ).collect().row(0, named=True)
    report = {
        "schema_version": 1, "experiment_id": "P1-G0-V013", "partial_run": args.max_rows is not None,
        **dict(counters), "unique_permnos": len(unique_permnos), "market_daily_dates": len(market_by_date),
        "market_weeks": market_weekly.height,
        "week_convention": "calendar week (Monday label)" if args.week_convention == "natural" else "Thursday-to-Wednesday (Thursday label)",
        "security_panel": {k: str(v) if k.startswith("min_") or k.startswith("max_") else v for k, v in security_stats.items()},
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "experiment_id": "P1-G0-V013",
        "git_revision": git_revision(), "command": " ".join(os.sys.argv),
        "inputs": [{"path": str(args.input.resolve()), "size_bytes": args.input.stat().st_size, "sha256": sha256(args.input)}],
        "outputs": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in [security_path, market_path, report_path]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
