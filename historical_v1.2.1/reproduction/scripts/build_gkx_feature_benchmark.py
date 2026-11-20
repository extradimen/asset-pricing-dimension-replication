#!/usr/bin/env python3
"""Convert the official GKX datashare and build the Structured Core-92 benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import zipfile
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "data/raw/public/gkx-datashare-2026-02-26/datashare.zip"
DEFAULT_MASTER = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V006/us_equity_research_master.parquet"

GKX_FEATURES = [
    "mvel1", "beta", "betasq", "chmom", "dolvol", "idiovol", "indmom", "mom1m",
    "mom6m", "mom12m", "mom36m", "pricedelay", "turn", "absacc", "acc", "age",
    "agr", "bm", "bm_ia", "cashdebt", "cashpr", "cfp", "cfp_ia", "chatoia",
    "chcsho", "chempia", "chinv", "chpmia", "convind", "currat", "depr", "divi",
    "divo", "dy", "egr", "ep", "gma", "grcapx", "grltnoa", "herf", "hire",
    "invest", "lev", "lgr", "mve_ia", "operprof", "orgcap", "pchcapx_ia",
    "pchcurrat", "pchdepr", "pchgm_pchsale", "pchquick", "pchsale_pchinvt",
    "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "pctacc", "ps", "quick",
    "rd", "rd_mve", "rd_sale", "realestate", "roic", "salecash", "saleinv",
    "salerec", "secured", "securedind", "sgr", "sin", "sp", "tang", "tb",
    "aeavol", "cash", "chtx", "cinvest", "ear", "nincr", "roaq", "roavol",
    "roeq", "rsup", "stdacc", "stdcf", "ms", "baspread", "ill", "maxret",
    "retvol", "std_dolvol", "std_turn", "zerotrade",
]
EXCLUDED_ANNOUNCEMENT_FEATURES = ["aeavol", "ear"]
CORE92 = [c for c in GKX_FEATURES if c not in EXCLUDED_ANNOUNCEMENT_FEATURES]
MASTER_COLUMNS = [
    "permno", "month", "ret", "ret_fwd1", "prc", "market_cap", "dollar_volume",
    "turnover", "mean_quoted_spread", "amihud_million", "primaryexch", "siccd",
    "gvkey",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--chunk-years", type=int, default=5)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


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


def convert_raw(input_path: Path, output_path: Path) -> dict[str, object]:
    writer: pq.ParquetWriter | None = None
    rows = 0
    min_month = max_month = None
    permnos: set[int] = set()
    with zipfile.ZipFile(input_path) as archive:
        with archive.open("datashare.csv") as stream:
            header = stream.readline().decode("utf-8-sig").rstrip("\r\n").split(",")
            stream.seek(0)
            observed_features = [c for c in header if c not in ["permno", "DATE", "sic2"]]
            if observed_features != GKX_FEATURES:
                missing = sorted(set(GKX_FEATURES) - set(observed_features))
                extra = sorted(set(observed_features) - set(GKX_FEATURES))
                raise ValueError(f"GKX feature schema mismatch; missing={missing}, extra={extra}")
            reader = pacsv.open_csv(
                stream,
                read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
                convert_options=pacsv.ConvertOptions(
                    include_columns=header,
                    column_types={
                        "permno": pa.int64(),
                        "DATE": pa.string(),
                        "sic2": pa.float64(),
                        **{c: pa.float64() for c in GKX_FEATURES},
                    },
                    strings_can_be_null=True,
                    null_values=[""],
                ),
            )
            for batch in reader:
                frame = (
                    pl.from_arrow(batch)
                    .with_columns(
                        pl.col("DATE").str.to_date("%Y%m%d", strict=True).dt.truncate("1mo").alias("month"),
                        pl.col("sic2").cast(pl.Int16, strict=False),
                    )
                    .drop("DATE")
                    .select(["permno", "month", "sic2"] + GKX_FEATURES)
                )
                rows += frame.height
                permnos.update(frame["permno"].unique().to_list())
                lo, hi = frame["month"].min(), frame["month"].max()
                min_month = min(min_month, lo) if min_month else lo
                max_month = max(max_month, hi) if max_month else hi
                table = frame.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=100_000)
    if writer is None:
        raise RuntimeError("No GKX rows produced")
    writer.close()
    return {
        "rows": rows,
        "columns": 97,
        "features": len(GKX_FEATURES),
        "unique_permnos": len(permnos),
        "month_min": str(min_month),
        "month_max": str(max_month),
    }


def build_overlap(
    raw_path: Path,
    master_path: Path,
    raw_overlap_path: Path,
    model_path: Path,
    chunk_years: int,
) -> dict[str, object]:
    raw_writer: pq.ParquetWriter | None = None
    model_writer: pq.ParquetWriter | None = None
    counters: Counter[str] = Counter()
    feature_nonmissing: Counter[str] = Counter()
    try:
        for first_year in range(1957, 2022, chunk_years):
            last_year = min(first_year + chunk_years - 1, 2021)
            start, end = date(first_year, 1, 1), date(last_year, 12, 1)
            gkx = pl.scan_parquet(raw_path).filter(
                (pl.col("month") >= start) & (pl.col("month") <= end)
            ).collect()
            master = (
                pl.scan_parquet(master_path)
                .select(MASTER_COLUMNS)
                .filter((pl.col("month") >= start) & (pl.col("month") <= end))
                .collect()
            )
            overlap = master.join(gkx, on=["permno", "month"], how="inner").sort(["month", "permno"])
            counters["master_rows_in_range"] += master.height
            counters["gkx_rows_in_range"] += gkx.height
            counters["overlap_rows"] += overlap.height
            counters["master_unmatched_rows"] += master.height - overlap.height
            for feature in GKX_FEATURES:
                feature_nonmissing[feature] += int(overlap[feature].is_not_null().sum())

            raw_table = overlap.to_arrow()
            if raw_writer is None:
                raw_writer = pq.ParquetWriter(raw_overlap_path, raw_table.schema, compression="zstd")
            raw_writer.write_table(raw_table, row_group_size=100_000)

            model = overlap.select(MASTER_COLUMNS + ["sic2"] + CORE92).with_columns(
                *[rank_expression(feature) for feature in CORE92],
                *[
                    pl.col(feature).is_null().cast(pl.Int8).alias(f"missing_{feature}")
                    for feature in CORE92
                ],
            ).drop(CORE92)
            model_table = model.to_arrow()
            if model_writer is None:
                model_writer = pq.ParquetWriter(model_path, model_table.schema, compression="zstd")
            model_writer.write_table(model_table, row_group_size=100_000)
            print(
                f"completed {first_year}-{last_year}: master={master.height}, "
                f"gkx={gkx.height}, overlap={overlap.height}",
                flush=True,
            )
    finally:
        if raw_writer is not None:
            raw_writer.close()
        if model_writer is not None:
            model_writer.close()
    if raw_writer is None or model_writer is None:
        raise RuntimeError("No GKX/master overlap produced")
    return {
        **dict(counters),
        "coverage_of_master": counters["overlap_rows"] / counters["master_rows_in_range"],
        "feature_nonmissing_rows": dict(feature_nonmissing),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    raw_path = output / "gkx_core94_raw.parquet"
    overlap_path = output / "gkx_core94_research_overlap.parquet"
    model_path = output / "structured_core92_model_input.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [raw_path, overlap_path, model_path, report_path, manifest_path]
    if any(p.exists() for p in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    raw = convert_raw(args.input, raw_path)
    overlap = build_overlap(raw_path, args.master, overlap_path, model_path, args.chunk_years)
    report = {
        "experiment_id": "P1-G0-V007",
        "data_snapshot_ids": ["wrds-us-equity-2025-12-v1", "gkx-datashare-2021-v1"],
        "gkx_raw": raw,
        "overlap": overlap,
        "core94_features": GKX_FEATURES,
        "core92_features": CORE92,
        "excluded_announcement_features": EXCLUDED_ANNOUNCEMENT_FEATURES,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V007",
        "data_snapshot_ids": report["data_snapshot_ids"],
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "inputs": [
            {"path": str(p.resolve()), "size_bytes": p.stat().st_size}
            for p in [args.input, args.master]
        ],
        "outputs": [
            {"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in [raw_path, overlap_path, model_path, report_path]
        ],
        "quality_report": report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "overlap"} | {
        "overlap": {k: v for k, v in overlap.items() if k != "feature_nonmissing_rows"}
    }, indent=2))


if __name__ == "__main__":
    main()
