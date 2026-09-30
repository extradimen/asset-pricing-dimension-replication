#!/usr/bin/env python3
"""Join CRSP, CCM, and point-in-time Compustat into a research master panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRSP = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V004/crsp_security_month.parquet"
DEFAULT_FUNDAMENTALS = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V005"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--crsp", type=Path, default=DEFAULT_CRSP)
    p.add_argument("--fundamentals-dir", type=Path, default=DEFAULT_FUNDAMENTALS)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--start-year", type=int, default=1950)
    p.add_argument("--end-year", type=int, default=2025)
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


def load_crsp_chunk(path: Path, start: date, end: date, has_following_month: bool) -> pl.DataFrame:
    target_end = end.replace(day=1)
    if has_following_month:
        target_end = pl.select(pl.lit(target_end).dt.offset_by("1mo")).item()
    frame = (
        pl.scan_parquet(path)
        .filter(
            pl.col("common_stock_flag")
            & (pl.col("month") >= start)
            & (pl.col("month") <= target_end)
        )
        .collect()
        .sort(["permno", "month"])
        .with_columns(
            pl.col("month").shift(-1).over("permno").alias("_next_month"),
            pl.col("ret").shift(-1).over("permno").alias("_next_ret"),
        )
        .with_columns(
            pl.when(pl.col("_next_month") == pl.col("month").dt.offset_by("1mo"))
            .then(pl.col("_next_ret"))
            .otherwise(None)
            .cast(pl.Float64)
            .alias("ret_fwd1"),
            pl.col("month").dt.month_end().alias("month_end"),
        )
        .filter(pl.col("month") <= end)
        .drop(["_next_month", "_next_ret"])
    )
    return frame


def resolve_ccm(crsp: pl.DataFrame, links: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    keys = crsp.select(["permno", "month", "month_end"])
    candidates = (
        keys.join(links, left_on="permno", right_on="LPERMNO", how="inner")
        .filter(
            (pl.col("month_end") >= pl.col("LINKDT"))
            & (pl.col("month_end") <= pl.col("LINKENDDT"))
        )
        .with_columns(
            pl.when(pl.col("LINKPRIM") == "P").then(0).otherwise(1).alias("_prim_rank"),
            pl.when(pl.col("LINKTYPE") == "LC").then(0).otherwise(1).alias("_type_rank"),
        )
    )
    ambiguity = (
        candidates.group_by(["permno", "month"])
        .agg(
            pl.len().alias("_candidate_count"),
            pl.col("gvkey").n_unique().alias("_gvkey_count"),
        )
        .filter(pl.col("_candidate_count") > 1)
    )
    mapping = (
        candidates.sort(["permno", "month", "_prim_rank", "_type_rank", "gvkey"])
        .unique(["permno", "month"], keep="first", maintain_order=True)
        .join(
            ambiguity.select(["permno", "month", "_candidate_count", "_gvkey_count"]),
            on=["permno", "month"],
            how="left",
        )
        .with_columns(
            pl.col("_candidate_count").fill_null(1).cast(pl.Int16).alias("ccm_candidate_count"),
            pl.col("_gvkey_count").fill_null(1).cast(pl.Int16).alias("ccm_gvkey_count"),
        )
        .select(
            "permno", "month", "gvkey", "LINKTYPE", "LINKPRIM", "LINKDT", "LINKENDDT",
            "ccm_candidate_count", "ccm_gvkey_count",
        )
    )
    result = crsp.join(mapping, on=["permno", "month"], how="left")
    return result, {
        "rows": crsp.height,
        "linked_rows": mapping.height,
        "unlinked_rows": crsp.height - mapping.height,
        "ambiguous_candidate_rows": ambiguity.height,
        "ambiguous_gvkey_rows": ambiguity.filter(pl.col("_gvkey_count") > 1).height,
    }


def load_statement_slice(
    path: Path,
    prefix: str,
    start: date,
    end: date,
    lookback: str,
) -> pl.DataFrame:
    lower = pl.select(pl.lit(start).dt.offset_by(lookback)).item()
    frame = (
        pl.scan_parquet(path)
        .filter(
            (pl.col("availability_date") >= lower)
            & (pl.col("availability_date") <= end)
        )
        .collect()
        .sort(["gvkey", "availability_date", "datadate"])
    )
    frame = frame.unique(["gvkey", "availability_date"], keep="last", maintain_order=True)
    frame = frame.rename({c: f"{prefix}_{c}" for c in frame.columns if c != "gvkey"})
    return frame.sort(["gvkey", f"{prefix}_availability_date"])


def availability_collision_stats(path: Path) -> dict[str, int]:
    collisions = (
        pl.scan_parquet(path)
        .select(["gvkey", "availability_date"])
        .group_by(["gvkey", "availability_date"])
        .len()
        .filter(pl.col("len") > 1)
        .collect()
    )
    return {
        "keys": collisions.height,
        "extra_rows": int((collisions["len"] - 1).sum()) if collisions.height else 0,
        "max_statements_same_date": int(collisions["len"].max()) if collisions.height else 1,
    }


def attach_statement(
    base: pl.DataFrame,
    statement: pl.DataFrame,
    prefix: str,
    freshness: str,
) -> pl.DataFrame:
    availability = f"{prefix}_availability_date"
    joined = base.sort(["gvkey", "month_end"]).join_asof(
        statement,
        left_on="month_end",
        right_on=availability,
        by="gvkey",
        strategy="backward",
        check_sortedness=False,
    )
    fresh_name = f"{prefix}_fresh"
    joined = joined.with_columns(
        (
            pl.col(availability).is_not_null()
            & (pl.col(availability) <= pl.col("month_end"))
            & (pl.col(availability) >= pl.col("month_end").dt.offset_by(freshness))
        ).alias(fresh_name)
    )
    # Remove stale values so results do not depend on the processing-window
    # lookback.  The boolean remains available as an explicit coverage flag.
    statement_columns = [c for c in statement.columns if c != "gvkey"]
    return joined.with_columns(
        *[
            pl.when(pl.col(fresh_name)).then(pl.col(c)).otherwise(None).alias(c)
            for c in statement_columns
        ]
    )


def main() -> None:
    args = parse_args()
    if args.start_year > args.end_year or args.chunk_years < 1:
        raise ValueError("Invalid year range or chunk size")
    started = time.time()
    output = args.output_dir.resolve()
    master_path = output / "us_equity_research_master.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    if any(p.exists() for p in [master_path, report_path, manifest_path]) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    for path in [master_path, report_path, manifest_path]:
        path.unlink(missing_ok=True)

    fundamental_dir = args.fundamentals_dir
    annual_path = fundamental_dir / "compustat_annual_pti.parquet"
    quarterly_path = fundamental_dir / "compustat_quarterly_pti.parquet"
    ccm_path = fundamental_dir / "ccm_link_history.parquet"
    links = (
        pl.read_parquet(ccm_path)
        .filter(pl.col("research_link_flag"))
        .select(["gvkey", "LPERMNO", "LINKTYPE", "LINKPRIM", "LINKDT", "LINKENDDT"])
    )
    annual_availability_collisions = availability_collision_stats(annual_path)
    quarterly_availability_collisions = availability_collision_stats(quarterly_path)

    counters: Counter[str] = Counter()
    writer: pq.ParquetWriter | None = None
    min_month = max_month = None
    try:
        for first_year in range(args.start_year, args.end_year + 1, args.chunk_years):
            last_year = min(first_year + args.chunk_years - 1, args.end_year)
            chunk_start = date(first_year, 1, 1)
            chunk_end = date(last_year, 12, 1)
            crsp = load_crsp_chunk(
                args.crsp,
                chunk_start,
                chunk_end,
                has_following_month=last_year < args.end_year,
            )
            if crsp.is_empty():
                continue
            base, link_stats = resolve_ccm(crsp, links)
            for key, value in link_stats.items():
                counters[f"ccm_{key}"] += value

            annual = load_statement_slice(
                annual_path, "a", chunk_start, date(last_year, 12, 31), "-18mo"
            )
            quarterly = load_statement_slice(
                quarterly_path, "q", chunk_start, date(last_year, 12, 31), "-12mo"
            )
            joined = attach_statement(base, annual, "a", "-18mo")
            joined = attach_statement(joined, quarterly, "q", "-12mo")
            joined = joined.sort(["permno", "month"])

            counters["output_rows"] += joined.height
            counters["target_nonnull"] += int(joined["ret_fwd1"].is_not_null().sum())
            counters["annual_attached"] += int(joined["a_availability_date"].is_not_null().sum())
            counters["annual_fresh"] += int(joined["a_fresh"].sum())
            counters["quarterly_attached"] += int(joined["q_availability_date"].is_not_null().sum())
            counters["quarterly_fresh"] += int(joined["q_fresh"].sum())
            counters["annual_lookahead_errors"] += int(
                (joined["a_availability_date"] > joined["month_end"]).fill_null(False).sum()
            )
            counters["quarterly_lookahead_errors"] += int(
                (joined["q_availability_date"] > joined["month_end"]).fill_null(False).sum()
            )
            lo, hi = joined["month"].min(), joined["month"].max()
            min_month = min(min_month, lo) if min_month else lo
            max_month = max(max_month, hi) if max_month else hi
            table = joined.to_arrow()
            if writer is None:
                writer = pq.ParquetWriter(master_path, table.schema, compression="zstd")
            elif table.schema != writer.schema:
                table = table.cast(writer.schema)
            writer.write_table(table, row_group_size=100_000)
            print(
                f"completed {first_year}-{last_year}: rows={joined.height}, "
                f"linked={link_stats['linked_rows']}, annual_fresh={int(joined['a_fresh'].sum())}, "
                f"quarterly_fresh={int(joined['q_fresh'].sum())}",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError("No research-master rows produced")

    report = {
        "experiment_id": "P1-G0-V006",
        "start_year": args.start_year,
        "end_year": args.end_year,
        "chunk_years": args.chunk_years,
        "month_min": str(min_month),
        "month_max": str(max_month),
        "annual_same_availability": annual_availability_collisions,
        "quarterly_same_availability": quarterly_availability_collisions,
        **dict(counters),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V006",
        "data_snapshot_id": "wrds-us-equity-2025-12-v1",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "inputs": [
            {"path": str(p.resolve()), "size_bytes": p.stat().st_size}
            for p in [args.crsp, annual_path, quarterly_path, ccm_path]
        ],
        "outputs": [
            {"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in [master_path, report_path]
        ],
        "quality_report": report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
