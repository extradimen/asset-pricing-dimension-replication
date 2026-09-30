#!/usr/bin/env python3
"""Build a reproducible CRSP CIZ security-month panel from the WRDS CSV archive.

The input is read directly from the ZIP so the 60 GB CSV never has to be
extracted.  CRSP CIZ already folds delisting returns into DlyRet.  Delisting
rows are retained and inherit the most recent active security classification;
same-day distribution rows are collapsed before daily returns are compounded.
"""

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


DEFAULT_INPUT = Path("data/raw/licensed/Daily Stock File_csv.zip")

ID_COLS = ["PERMNO", "PERMCO"]
INFO_COLS = [
    "CUSIP",
    "CUSIP9",
    "Ticker",
    "TradingSymbol",
    "SecurityNm",
    "PrimaryExch",
    "ExchangeTier",
    "ShareClass",
    "USIncFlg",
    "IssuerType",
    "SecurityType",
    "SecuritySubType",
    "ShareType",
    "SICCD",
    "NAICS",
]
NUMERIC_COLS = [
    "DlyPrc",
    "DlyCap",
    "DlyRet",
    "DlyRetx",
    "DlyOrdDivAmt",
    "DlyNonOrdDivAmt",
    "DlyVol",
    "DlyClose",
    "DlyLow",
    "DlyHigh",
    "DlyBid",
    "DlyAsk",
    "DlyOpen",
    "DlyNumTrd",
    "DlyMMCnt",
    "DlyPrcVol",
    "ShrOut",
    "vwretd",
    "vwretx",
    "ewretd",
    "ewretx",
    "sprtrn",
]
READ_COLS = (
    ID_COLS
    + INFO_COLS
    + [
        "DlyCalDt",
        "DlyDelFlg",
        "ConditionalType",
        "TradingStatusFlg",
        "DlyRetMissFlg",
    ]
    + NUMERIC_COLS
)
MARKET_COLS = ["vwretd", "vwretx", "ewretd", "ewretx", "sprtrn"]

# These fields should describe the same security-day even when CRSP emits more
# than one row for multiple distributions on that date.
STOCK_DAY_INVARIANTS = [
    "PERMCO",
    "DlyPrc",
    "DlyCap",
    "DlyRet",
    "DlyRetx",
    "DlyVol",
    "DlyClose",
    "DlyLow",
    "DlyHigh",
    "DlyBid",
    "DlyAsk",
    "DlyOpen",
    "DlyNumTrd",
    "DlyPrcVol",
    "ShrOut",
    "DlyDelFlg",
    "DlyRetMissFlg",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--block-size-mb", type=int, default=64)
    p.add_argument("--max-rows", type=int, help="Smoke-test row limit")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def sha256(path: Path, block: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def product_return(column: str, alias: str) -> pl.Expr:
    values = pl.col(column).drop_nulls()
    return (
        pl.when(pl.col(column).count() > 0)
        .then((values + 1.0).product() - 1.0)
        .otherwise(None)
        .cast(pl.Float64)
        .alias(alias)
    )


def last_regular(column: str, active: pl.Expr, alias: str, dtype: pl.DataType) -> pl.Expr:
    return (
        pl.col(column)
        .filter(active & pl.col(column).is_not_null())
        .last()
        .cast(dtype)
        .alias(alias)
    )


def transform_complete_groups(df: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    if df.is_empty():
        return pl.DataFrame(), {}

    active = (pl.col("ConditionalType") == "RW") & (pl.col("TradingStatusFlg") == "A")
    df = df.with_columns(
        active.alias("_active_regular"),
        active.cast(pl.Int8).cum_sum().over("PERMNO").gt(0).alias("_active_seen"),
        *[
            pl.when(active).then(pl.col(c)).otherwise(None).forward_fill().over("PERMNO").alias(f"_info_{c}")
            for c in INFO_COLS
        ],
    )

    include = pl.col("_active_regular") | (
        (pl.col("DlyDelFlg") == "Y") & pl.col("_active_seen")
    )
    selected = df.filter(include)
    if selected.is_empty():
        return pl.DataFrame(), {
            "rows_in_complete_groups": df.height,
            "rows_included_before_dedup": 0,
            "duplicate_rows": 0,
            "duplicate_invariant_variants": 0,
        }

    keys = ["PERMNO", "DlyCalDt"]
    n_days = selected.select(keys).unique().height
    invariant_variants = (
        selected.select(keys + STOCK_DAY_INVARIANTS).unique().height - n_days
    )
    duplicate_rows = selected.height - n_days
    if invariant_variants:
        raise ValueError(
            f"Found {invariant_variants} same-security-day variants in stock return/price fields"
        )

    daily = selected.unique(subset=keys, keep="first", maintain_order=True).with_columns(
        pl.col("DlyCalDt").dt.truncate("1mo").alias("month"),
        (pl.col("DlyPrc").abs() * pl.col("DlyVol")).alias("_dollar_volume"),
        pl.when((pl.col("DlyBid") > 0) & (pl.col("DlyAsk") >= pl.col("DlyBid")))
        .then((pl.col("DlyAsk") - pl.col("DlyBid")) / ((pl.col("DlyAsk") + pl.col("DlyBid")) / 2.0))
        .otherwise(None)
        .alias("_quoted_spread"),
        pl.when((pl.col("DlyPrc").abs() * pl.col("DlyVol")) > 0)
        .then(pl.col("DlyRet").abs() / (pl.col("DlyPrc").abs() * pl.col("DlyVol")) * 1_000_000)
        .otherwise(None)
        .alias("_amihud_million"),
    )

    active = pl.col("_active_regular")
    valid_pair = pl.col("DlyRet").is_not_null() & pl.col("vwretd").is_not_null()
    down_pair = valid_pair & (pl.col("vwretd") < 0)

    aggs: list[pl.Expr] = [
        pl.col("DlyCalDt").min().alias("first_date"),
        pl.col("DlyCalDt").max().alias("last_date"),
        pl.col("DlyCalDt").filter(active).max().alias("last_regular_trade_date"),
        product_return("DlyRet", "ret"),
        product_return("DlyRetx", "retx"),
        pl.col("DlyRet").count().cast(pl.Int32).alias("n_return_days"),
        pl.len().cast(pl.Int32).alias("n_observations"),
        pl.col("_active_regular").sum().cast(pl.Int32).alias("n_regular_days"),
        (pl.col("DlyDelFlg") == "Y").any().alias("delist_flag"),
        pl.col("DlyCalDt").filter(pl.col("DlyDelFlg") == "Y").max().alias("delist_date"),
        pl.col("DlyRetMissFlg").filter(pl.col("DlyDelFlg") == "Y").drop_nulls().last().alias("delist_return_missing_flag"),
        last_regular("PERMCO", active, "permco", pl.Int64),
        last_regular("DlyPrc", active, "prc", pl.Float64),
        last_regular("DlyCap", active, "market_cap", pl.Float64),
        last_regular("ShrOut", active, "shares_outstanding", pl.Float64),
        last_regular("DlyOpen", active, "open", pl.Float64),
        last_regular("DlyClose", active, "close", pl.Float64),
        pl.col("DlyLow").filter(active).min().cast(pl.Float64).alias("low"),
        pl.col("DlyHigh").filter(active).max().cast(pl.Float64).alias("high"),
        pl.col("DlyVol").filter(active).sum().cast(pl.Float64).alias("volume"),
        pl.col("_dollar_volume").filter(active).sum().cast(pl.Float64).alias("dollar_volume"),
        pl.col("DlyNumTrd").filter(active).sum().cast(pl.Float64).alias("number_of_trades"),
        ((pl.col("DlyVol").filter(active).fill_null(0) == 0).sum()).cast(pl.Int32).alias("zero_volume_days"),
        pl.col("_quoted_spread").filter(active).mean().cast(pl.Float64).alias("mean_quoted_spread"),
        pl.col("_amihud_million").filter(active).mean().cast(pl.Float64).alias("amihud_million"),
        pl.col("DlyRet").max().cast(pl.Float64).alias("max_daily_return"),
        pl.col("DlyRet").min().cast(pl.Float64).alias("min_daily_return"),
        pl.col("DlyRet").std(ddof=1).cast(pl.Float64).alias("daily_return_std"),
        pl.col("DlyRet").skew(bias=False).cast(pl.Float64).alias("daily_return_skew"),
        (pl.col("DlyRet").pow(2).sum()).cast(pl.Float64).alias("sum_ret2"),
        valid_pair.sum().cast(pl.Int32).alias("n_market_pairs"),
        pl.col("DlyRet").filter(valid_pair).sum().cast(pl.Float64).alias("sum_ri"),
        pl.col("vwretd").filter(valid_pair).sum().cast(pl.Float64).alias("sum_rm"),
        pl.col("DlyRet").filter(valid_pair).pow(2).sum().cast(pl.Float64).alias("sum_ri2"),
        pl.col("vwretd").filter(valid_pair).pow(2).sum().cast(pl.Float64).alias("sum_rm2"),
        (pl.col("DlyRet") * pl.col("vwretd")).filter(valid_pair).sum().cast(pl.Float64).alias("sum_ri_rm"),
        down_pair.sum().cast(pl.Int32).alias("n_down_market_pairs"),
        pl.col("DlyRet").filter(down_pair).sum().cast(pl.Float64).alias("sum_down_ri"),
        pl.col("vwretd").filter(down_pair).sum().cast(pl.Float64).alias("sum_down_rm"),
        pl.col("DlyRet").filter(down_pair).pow(2).sum().cast(pl.Float64).alias("sum_down_ri2"),
        pl.col("vwretd").filter(down_pair).pow(2).sum().cast(pl.Float64).alias("sum_down_rm2"),
        (pl.col("DlyRet") * pl.col("vwretd")).filter(down_pair).sum().cast(pl.Float64).alias("sum_down_ri_rm"),
    ]
    for c in INFO_COLS:
        aggs.append(pl.col(f"_info_{c}").drop_nulls().last().cast(pl.Utf8).alias(c.lower()))

    monthly = (
        daily.group_by(["PERMNO", "month"], maintain_order=True)
        .agg(aggs)
        .rename({"PERMNO": "permno"})
        .with_columns(
            pl.when(pl.col("shares_outstanding") > 0)
            .then(pl.col("volume") / pl.col("shares_outstanding"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("turnover"),
            (
                pl.col("primaryexch").is_in(["N", "A", "Q"])
                & (pl.col("usincflg") == "Y")
                & pl.col("issuertype").is_in(["ACOR", "CORP"])
                & (pl.col("securitytype") == "EQTY")
                & (pl.col("securitysubtype") == "COM")
                & (pl.col("sharetype") == "NS")
            ).alias("common_stock_flag"),
        )
    )
    stats = {
        "rows_in_complete_groups": df.height,
        "rows_included_before_dedup": selected.height,
        "duplicate_rows": duplicate_rows,
        "duplicate_invariant_variants": invariant_variants,
        "security_days": daily.height,
        "security_months": monthly.height,
    }
    return monthly, stats


def update_market_cache(df: pl.DataFrame, cache: dict[Any, tuple[Any, ...]]) -> int:
    market = (
        df.select(["DlyCalDt"] + MARKET_COLS)
        .filter(pl.col("DlyCalDt").is_not_null() & pl.col("vwretd").is_not_null())
        .unique()
    )
    conflicts = 0
    for row in market.iter_rows():
        date, values = row[0], tuple(row[1:])
        previous = cache.get(date)
        if previous is not None and previous != values:
            conflicts += 1
        else:
            cache[date] = values
    return conflicts


def make_market_month(cache: dict[Any, tuple[Any, ...]]) -> pl.DataFrame:
    rows = [(d, *values) for d, values in sorted(cache.items())]
    daily = pl.DataFrame(
        rows,
        schema={"date": pl.Date, **{c: pl.Float64 for c in MARKET_COLS}},
        orient="row",
    )
    return (
        daily.with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by("month", maintain_order=True)
        .agg(
            pl.col("date").min().alias("first_date"),
            pl.col("date").max().alias("last_date"),
            pl.len().cast(pl.Int32).alias("n_market_days"),
            *[product_return(c, c) for c in MARKET_COLS],
        )
    )


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    stock_path = output / "crsp_security_month.parquet"
    market_path = output / "crsp_market_month.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [stock_path, market_path, report_path, manifest_path]
    if any(p.exists() for p in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    for p in products:
        if p.exists():
            p.unlink()

    counters: Counter[str] = Counter()
    market_cache: dict[Any, tuple[Any, ...]] = {}
    writer: pq.ParquetWriter | None = None
    carry: pl.DataFrame | None = None
    first_month: str | None = None
    last_month: str | None = None
    common_months = 0
    delist_months = 0
    unique_permnos: set[int] = set()

    convert = pacsv.ConvertOptions(
        include_columns=READ_COLS,
        column_types={
            "PERMNO": pa.int64(),
            "PERMCO": pa.int64(),
            "DlyCalDt": pa.date32(),
            **{c: pa.float64() for c in NUMERIC_COLS},
        },
        strings_can_be_null=True,
        null_values=[""],
    )
    read = pacsv.ReadOptions(block_size=args.block_size_mb * 1024 * 1024)

    with zipfile.ZipFile(args.input) as zf:
        members = [i for i in zf.infolist() if not i.is_dir()]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV in archive, found {len(members)}")
        with zf.open(members[0]) as raw:
            reader = pacsv.open_csv(raw, read_options=read, convert_options=convert)
            for batch in reader:
                frame = pl.from_arrow(batch)
                if args.max_rows is not None:
                    remaining = args.max_rows - counters["raw_rows_read"]
                    if remaining <= 0:
                        break
                    frame = frame.head(remaining)
                counters["raw_rows_read"] += frame.height
                counters["market_value_conflicts"] += update_market_cache(frame, market_cache)
                if carry is not None:
                    frame = pl.concat([carry, frame], how="vertical")
                last_permno = frame.item(-1, "PERMNO")
                complete = frame.filter(pl.col("PERMNO") != last_permno)
                carry = frame.filter(pl.col("PERMNO") == last_permno)
                monthly, stats = transform_complete_groups(complete)
                counters.update(stats)
                if not monthly.is_empty():
                    unique_permnos.update(monthly["permno"].to_list())
                    common_months += int(monthly["common_stock_flag"].sum())
                    delist_months += int(monthly["delist_flag"].sum())
                    lo, hi = monthly["month"].min(), monthly["month"].max()
                    first_month = min(first_month, str(lo)) if first_month else str(lo)
                    last_month = max(last_month, str(hi)) if last_month else str(hi)
                    table = monthly.to_arrow()
                    if writer is None:
                        writer = pq.ParquetWriter(stock_path, table.schema, compression="zstd")
                    writer.write_table(table, row_group_size=250_000)
                if args.max_rows is not None and counters["raw_rows_read"] >= args.max_rows:
                    break

    if carry is not None and not carry.is_empty():
        monthly, stats = transform_complete_groups(carry)
        counters.update(stats)
        if not monthly.is_empty():
            unique_permnos.update(monthly["permno"].to_list())
            common_months += int(monthly["common_stock_flag"].sum())
            delist_months += int(monthly["delist_flag"].sum())
            lo, hi = monthly["month"].min(), monthly["month"].max()
            first_month = min(first_month, str(lo)) if first_month else str(lo)
            last_month = max(last_month, str(hi)) if last_month else str(hi)
            table = monthly.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(stock_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250_000)
    if writer is None:
        raise RuntimeError("No security-month observations were produced")
    writer.close()

    market_month = make_market_month(market_cache)
    market_month.write_parquet(market_path, compression="zstd")
    if counters["market_value_conflicts"]:
        raise ValueError(f"Found {counters['market_value_conflicts']} conflicting market returns")

    report = {
        "experiment_id": "P1-G0-V004",
        "input": str(args.input.resolve()),
        "partial_run": args.max_rows is not None,
        "max_rows": args.max_rows,
        "raw_rows_read": counters["raw_rows_read"],
        "rows_in_complete_groups": counters["rows_in_complete_groups"],
        "rows_included_before_dedup": counters["rows_included_before_dedup"],
        "duplicate_rows_collapsed": counters["duplicate_rows"],
        "duplicate_invariant_variants": counters["duplicate_invariant_variants"],
        "security_days": counters["security_days"],
        "security_months": counters["security_months"],
        "unique_permnos": len(unique_permnos),
        "common_stock_months": common_months,
        "delist_months": delist_months,
        "first_month": first_month,
        "last_month": last_month,
        "market_daily_dates": len(market_cache),
        "market_months": market_month.height,
        "market_value_conflicts": counters["market_value_conflicts"],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V004",
        "data_snapshot_id": "wrds-us-equity-2025-12-v1",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "input": {"path": str(args.input.resolve()), "size_bytes": args.input.stat().st_size},
        "outputs": [
            {
                "path": p.name,
                "size_bytes": p.stat().st_size,
                "sha256": sha256(p),
            }
            for p in [stock_path, market_path, report_path]
        ],
        "quality_report": report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
