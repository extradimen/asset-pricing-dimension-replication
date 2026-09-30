#!/usr/bin/env python3
"""Build GHZ beta, beta squared, and idiovol from three years of weekly returns."""

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
READ_COLS = [
    "PERMNO", "DlyCalDt", "ConditionalType", "TradingStatusFlg", "DlyRet", "ewretd",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--block-size-mb", type=int, default=128)
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


def compound(column: str) -> pl.Expr:
    return ((pl.col(column).drop_nulls() + 1.0).product() - 1.0)


def transform_complete_groups(frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    if frame.is_empty():
        return pl.DataFrame(), {}
    active = frame.filter(
        (pl.col("ConditionalType") == "RW")
        & (pl.col("TradingStatusFlg") == "A")
    )
    daily = (
        active.unique(subset=["PERMNO", "DlyCalDt"], keep="first", maintain_order=True)
        .filter(pl.col("DlyRet").is_not_null() & pl.col("ewretd").is_not_null())
        .sort(["PERMNO", "DlyCalDt"])
        .with_columns(pl.col("DlyCalDt").dt.truncate("1w").alias("week"))
    )
    weekly = (
        daily.group_by(["PERMNO", "week"], maintain_order=True)
        .agg(
            pl.col("DlyCalDt").max().alias("week_end"),
            compound("DlyRet").alias("stock_weekly_return"),
            compound("ewretd").alias("market_weekly_return"),
            pl.len().cast(pl.Int16).alias("trading_days"),
        )
        .sort(["PERMNO", "week_end"])
        .with_columns(
            (pl.col("stock_weekly_return") * pl.col("market_weekly_return")).alias("xy"),
            pl.col("stock_weekly_return").pow(2).alias("x2"),
            pl.col("market_weekly_return").pow(2).alias("y2"),
        )
    )
    sums = weekly.rolling(
        "week_end", period="3y", group_by="PERMNO", closed="both"
    ).agg(
        pl.len().cast(pl.Int16).alias("weeks"),
        pl.col("stock_weekly_return").sum().alias("sum_x"),
        pl.col("market_weekly_return").sum().alias("sum_y"),
        pl.col("x2").sum().alias("sum_x2"),
        pl.col("y2").sum().alias("sum_y2"),
        pl.col("xy").sum().alias("sum_xy"),
    )
    n = pl.col("weeks").cast(pl.Float64)
    centered_x2 = pl.col("sum_x2") - pl.col("sum_x").pow(2) / n
    centered_y2 = pl.col("sum_y2") - pl.col("sum_y").pow(2) / n
    centered_xy = pl.col("sum_xy") - pl.col("sum_x") * pl.col("sum_y") / n
    beta = centered_xy / centered_y2
    residual_ss = centered_x2 - beta * centered_xy
    eligible = (n >= 52) & (centered_y2 > 0) & (residual_ss >= 0)
    weekly_characteristics = sums.with_columns(
        pl.when(eligible).then(beta).otherwise(None).alias("beta_weekly_3y"),
        pl.when(eligible & (n > 2)).then((residual_ss / (n - 2.0)).sqrt()).otherwise(None)
        .alias("idiovol_weekly_3y"),
    ).with_columns(
        pl.col("week_end").dt.truncate("1mo").alias("formation_month"),
    )
    # The published definition says the window ends in t-1, but the official
    # GKX values align most closely when the formation month is shifted by two
    # months.  We retain that implementation bridge explicitly and audit the
    # adjacent one-month shifts rather than silently overriding the discrepancy.
    result = (
        weekly_characteristics.group_by(["PERMNO", "formation_month"], maintain_order=True)
        .agg(
            pl.col("week_end").last(),
            pl.col("weeks").last(),
            pl.col("beta_weekly_3y").last(),
            pl.col("idiovol_weekly_3y").last(),
        )
        .with_columns(
            pl.col("formation_month").dt.offset_by("2mo").alias("month"),
            pl.col("beta_weekly_3y").pow(2).alias("betasq_weekly_3y"),
        )
        .select(
            pl.col("PERMNO").alias("permno"), "month", "week_end", "weeks",
            "beta_weekly_3y", "betasq_weekly_3y", "idiovol_weekly_3y",
        )
    )
    return result, {
        "raw_complete_rows": frame.height,
        "active_rows": active.height,
        "eligible_daily_rows": daily.height,
        "security_weeks": weekly.height,
        "security_months": result.height,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    panel_path = output / "weekly_market_characteristics.parquet"
    report_path = output / "weekly_quality_report.json"
    manifest_path = output / "weekly_output_manifest.json"
    products = [panel_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    convert = pacsv.ConvertOptions(
        include_columns=READ_COLS,
        column_types={
            "PERMNO": pa.int64(), "DlyCalDt": pa.date32(), "DlyRet": pa.float64(),
            "ewretd": pa.float64(),
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

    def write_frame(frame: pl.DataFrame) -> None:
        nonlocal writer, month_min, month_max
        panel, stats = transform_complete_groups(frame)
        counters.update(stats)
        if panel.is_empty():
            return
        unique_permnos.update(panel["permno"].unique().to_list())
        lo, hi = panel["month"].min(), panel["month"].max()
        month_min = min(month_min, lo) if month_min else lo
        month_max = max(month_max, hi) if month_max else hi
        table = panel.to_arrow()
        if writer is None:
            writer = pq.ParquetWriter(panel_path, table.schema, compression="zstd")
        writer.write_table(table, row_group_size=250_000)

    with zipfile.ZipFile(args.input) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(f"Expected one CRSP CSV, found {len(members)}")
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
                write_frame(complete)
                if args.max_rows is not None and counters["raw_rows_read"] >= args.max_rows:
                    break
    if carry is not None and not carry.is_empty():
        write_frame(carry)
    if writer is None:
        raise RuntimeError("No observations produced")
    writer.close()

    report = {
        "schema_version": 1, "experiment_id": "P1-G0-V011",
        "definition": "Three years of weekly stock returns regressed on equal-weighted market returns; minimum 52 weeks; formation month shifted two months to match the official GKX implementation, one month more than the published t-1 prose convention.",
        "input": str(args.input.resolve()), "partial_run": args.max_rows is not None,
        **dict(counters), "unique_permnos": len(unique_permnos),
        "first_target_month": str(month_min), "last_target_month": str(month_max),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V011", "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "inputs": [{"path": str(args.input.resolve()), "size_bytes": args.input.stat().st_size, "sha256": sha256(args.input)}],
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [panel_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
