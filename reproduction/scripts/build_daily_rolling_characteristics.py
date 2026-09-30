#!/usr/bin/env python3
"""Build three-month daily CRSP characteristics with official FF factors."""

from __future__ import annotations

import argparse
import hashlib
import io
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
DEFAULT_FACTORS = ROOT / "data/raw/public/F-F_Research_Data_Factors_daily_CSV.zip"
READ_COLS = [
    "PERMNO", "DlyCalDt", "ConditionalType", "TradingStatusFlg",
    "DlyRet", "DlyVol", "DlyPrc", "ShrOut",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--factors", type=Path, default=DEFAULT_FACTORS)
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


def read_ff_factors(path: Path) -> pl.DataFrame:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if not item.is_dir()]
        if len(members) != 1:
            raise ValueError(f"Expected one factor CSV, found {len(members)}")
        lines = io.TextIOWrapper(archive.open(members[0]), encoding="utf-8")
        rows: list[tuple[str, float, float]] = []
        for line in lines:
            fields = [field.strip() for field in line.split(",")]
            if len(fields) == 5 and len(fields[0]) == 8 and fields[0].isdigit():
                rows.append((fields[0], float(fields[1]) / 100.0, float(fields[4]) / 100.0))
    return (
        pl.DataFrame(rows, schema=["date_string", "mktrf", "rf"], orient="row")
        .with_columns(pl.col("date_string").str.to_date("%Y%m%d").alias("DlyCalDt"))
        .select("DlyCalDt", "mktrf", "rf")
    )


def sample_std(sum_col: str, sumsq_col: str, count_col: str) -> pl.Expr:
    n = pl.col(count_col).cast(pl.Float64)
    variance = (pl.col(sumsq_col) - pl.col(sum_col).pow(2) / n) / (n - 1.0)
    return pl.when((n >= 2) & (variance >= 0)).then(variance.sqrt()).otherwise(None)


def transform_complete_groups(
    frame: pl.DataFrame, factors: pl.DataFrame
) -> tuple[pl.DataFrame, dict[str, int]]:
    if frame.is_empty():
        return pl.DataFrame(), {}
    active = frame.filter(
        (pl.col("ConditionalType") == "RW")
        & (pl.col("TradingStatusFlg") == "A")
    )
    keys = ["PERMNO", "DlyCalDt"]
    daily = (
        active.unique(subset=keys, keep="first", maintain_order=True)
        .join(factors, on="DlyCalDt", how="left")
        .sort(keys)
        .with_columns(
            pl.col("DlyCalDt").dt.truncate("1mo").alias("month"),
            (pl.col("DlyRet") - pl.col("rf")).alias("_exret"),
            pl.when((pl.col("DlyVol") > 0) & (pl.col("DlyPrc").abs() > 0))
            .then((pl.col("DlyVol") * pl.col("DlyPrc").abs()).log())
            .otherwise(None)
            .alias("_log_dolvol"),
            pl.when((pl.col("ShrOut") > 0) & pl.col("DlyVol").is_not_null())
            .then(pl.col("DlyVol") / (pl.col("ShrOut") * 1000.0))
            .otherwise(None)
            .alias("_turn"),
        )
        .with_columns(
            (
                pl.col("DlyRet").is_not_null()
                & pl.col("DlyVol").is_not_null()
                & pl.col("mktrf").is_not_null()
                & pl.col("rf").is_not_null()
            ).alias("_reg_ok"),
            (pl.col("DlyRet").is_not_null() & pl.col("DlyVol").is_not_null()).alias("_base_ok"),
        )
    )
    monthly = (
        daily.group_by(["PERMNO", "month"], maintain_order=True)
        .agg(
            pl.col("_reg_ok").sum().cast(pl.Int32).alias("reg_n"),
            pl.col("_exret").filter(pl.col("_reg_ok")).sum().alias("sum_x"),
            pl.col("mktrf").filter(pl.col("_reg_ok")).sum().alias("sum_y"),
            pl.col("_exret").filter(pl.col("_reg_ok")).pow(2).sum().alias("sum_x2"),
            pl.col("mktrf").filter(pl.col("_reg_ok")).pow(2).sum().alias("sum_y2"),
            (pl.col("_exret") * pl.col("mktrf")).filter(pl.col("_reg_ok")).sum().alias("sum_xy"),
            pl.col("_log_dolvol").count().cast(pl.Int32).alias("dolvol_n"),
            pl.col("_log_dolvol").sum().alias("sum_log_dolvol"),
            pl.col("_log_dolvol").pow(2).sum().alias("sum_log_dolvol2"),
            pl.col("_turn").count().cast(pl.Int32).alias("turn_n"),
            pl.col("_turn").sum().alias("sum_turn"),
            pl.col("_turn").pow(2).sum().alias("sum_turn2"),
            (pl.col("DlyVol").filter(pl.col("_base_ok")) == 0).sum().cast(pl.Int32).alias("zero_days"),
            pl.col("_base_ok").sum().cast(pl.Int32).alias("base_n"),
        )
        .sort(["PERMNO", "month"])
    )
    sum_columns = [
        "reg_n", "sum_x", "sum_y", "sum_x2", "sum_y2", "sum_xy",
        "dolvol_n", "sum_log_dolvol", "sum_log_dolvol2",
        "turn_n", "sum_turn", "sum_turn2", "zero_days", "base_n",
    ]
    rolling = monthly.with_columns(
        *[
            pl.col(column).rolling_sum(3, min_samples=3).over("PERMNO").alias(f"{column}_3m")
            for column in sum_columns
        ],
        pl.col("month").shift(2).over("PERMNO").alias("_month_lag2"),
    ).with_columns(
        (pl.col("_month_lag2") == pl.col("month").dt.offset_by("-2mo")).alias("_contiguous")
    )
    n = pl.col("reg_n_3m").cast(pl.Float64)
    centered_x2 = pl.col("sum_x2_3m") - pl.col("sum_x_3m").pow(2) / n
    centered_y2 = pl.col("sum_y2_3m") - pl.col("sum_y_3m").pow(2) / n
    centered_xy = pl.col("sum_xy_3m") - pl.col("sum_x_3m") * pl.col("sum_y_3m") / n
    beta = centered_xy / centered_y2
    residual_ss = centered_x2 - beta * centered_xy
    eligible_reg = pl.col("_contiguous") & (n >= 21) & (centered_y2 > 0)
    eligible_base = pl.col("_contiguous") & (pl.col("base_n_3m") >= 21)
    result = (
        rolling.with_columns(
            pl.when(eligible_reg).then(beta).otherwise(None).alias("beta_daily_3m"),
            pl.when(eligible_reg & (residual_ss >= 0) & (n > 2))
            .then((residual_ss / (n - 2.0)).sqrt())
            .otherwise(None)
            .alias("idiovol_daily_capm_3m"),
            pl.when(eligible_base)
            .then(sample_std("sum_log_dolvol_3m", "sum_log_dolvol2_3m", "dolvol_n_3m"))
            .otherwise(None)
            .alias("std_dolvol_3m"),
            pl.when(eligible_base)
            .then(sample_std("sum_turn_3m", "sum_turn2_3m", "turn_n_3m"))
            .otherwise(None)
            .alias("std_turn_3m"),
            pl.when(eligible_base & (pl.col("sum_turn_3m") > 0))
            .then(
                (pl.col("zero_days_3m") + (1.0 / pl.col("sum_turn_3m")) / 11000.0)
                * 63.0 / pl.col("base_n_3m")
            )
            .otherwise(None)
            .alias("zerotrade_3m"),
        )
        .select(
            pl.col("PERMNO").alias("permno"), "month",
            "beta_daily_3m", "idiovol_daily_capm_3m",
            "std_dolvol_3m", "std_turn_3m", "zerotrade_3m",
            "reg_n_3m", "dolvol_n_3m", "turn_n_3m", "base_n_3m",
        )
    )
    return result, {
        "raw_complete_rows": frame.height,
        "active_rows": active.height,
        "security_days": daily.height,
        "factor_matched_days": int(daily["mktrf"].is_not_null().sum()),
        "security_months": result.height,
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    panel_path = output / "daily_rolling_characteristics.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [panel_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    factors = read_ff_factors(args.factors)
    convert = pacsv.ConvertOptions(
        include_columns=READ_COLS,
        column_types={
            "PERMNO": pa.int64(), "DlyCalDt": pa.date32(), "DlyRet": pa.float64(),
            "DlyVol": pa.float64(), "DlyPrc": pa.float64(), "ShrOut": pa.float64(),
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
        panel, stats = transform_complete_groups(frame, factors)
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
        "schema_version": 1,
        "experiment_id": "P1-G0-V011",
        "input": str(args.input.resolve()),
        "factor_input": str(args.factors.resolve()),
        "factor_sha256": sha256(args.factors),
        "partial_run": args.max_rows is not None,
        **dict(counters),
        "unique_permnos": len(unique_permnos),
        "first_month": str(month_min),
        "last_month": str(month_max),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V011",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [args.input, args.factors]
        ],
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [panel_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
