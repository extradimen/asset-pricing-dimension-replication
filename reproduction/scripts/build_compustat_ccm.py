#!/usr/bin/env python3
"""Build point-in-time Compustat statement files and a canonical CCM link table."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path
from typing import BinaryIO

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import polars as pl


DEFAULT_COMPUSTAT = Path("data/raw/licensed/【批量下载】Fundamentals Quarterly等.zip")
DEFAULT_CCM = Path("data/raw/licensed/link.zip")

COMMON_TEXT = [
    "gvkey", "tic", "conm", "cusip", "cik", "fic", "costat", "datafmt",
    "indfmt", "consol", "exchg", "fyr", "sic", "naics",
]
ANNUAL_DATES = ["datadate", "fdate", "pdate", "apdedate", "ipodate", "dldte"]
ANNUAL_TEXT = COMMON_TEXT + ["fyear", "curcd", "acctstd", "final", "upd", "src"]
ANNUAL_NUMERIC = [
    "at", "act", "che", "rect", "invt", "ppegt", "ppent", "intan", "gdwl",
    "ao", "aco", "lt", "lct", "dlc", "dltt", "txdb", "txditc", "pstk",
    "pstkrv", "pstkl", "seq", "ceq", "re", "ap", "lco", "lo", "mib",
    "wcap", "csho", "cshrc", "cshrt", "emp",
    "sale", "revt", "cogs", "xsga", "xrd", "xad", "dp", "oiadp", "oibdp",
    "ib", "ni", "pi", "txt", "xint", "ebit", "ebitda", "gp", "xido",
    "spi", "nopi",
    "oancf", "capx", "ivncf", "fincf", "dv", "dvc", "dvp", "prstkc",
    "sstk", "dlcch", "dltis", "dltr", "chech", "txp",
    "prcc_f", "mkvalt", "cshtr_f", "dvpsx_f", "adjex_f",
]

QUARTERLY_DATES = ["datadate", "rdq", "fdateq", "pdateq", "apdedateq", "ipodate", "dldte"]
QUARTERLY_TEXT = COMMON_TEXT + [
    "fyearq", "fqtr", "datafqtr", "datacqtr", "curcdq", "acctstdq", "finalq",
    "updq", "srcq",
]
QUARTERLY_NUMERIC = [
    "atq", "actq", "cheq", "rectq", "invtq", "ppegtq", "ppentq", "intanq",
    "gdwlq", "aoq", "acoq", "ltq", "lctq", "dlcq", "dlttq", "txdbq",
    "txditcq", "pstkq", "pstkrq", "seqq", "ceqq", "req", "apq", "lcoq",
    "loq", "mibq", "wcapq", "cshoq", "cshiq",
    "saleq", "revtq", "cogsq", "xsgaq", "xrdq", "dpq", "oiadpq", "oibdpq",
    "ibq", "niq", "piq", "txtq", "xintq", "xidoq", "spiq", "nopiq",
    "oancfy", "capxy", "ivncfy", "fincfy", "dvpy", "dvy", "prstkcy",
    "sstky", "dlcchy", "dltisy", "dltry", "chechy",
    "prccq", "mkvaltq", "cshtrq", "dvpsxq", "adjex",
]

CCM_DATES = ["LINKDT", "LINKENDDT", "ipodate", "dldte"]
CCM_NUMERIC = ["LPERMNO", "LPERMCO"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compustat-input", type=Path, default=DEFAULT_COMPUSTAT)
    p.add_argument("--ccm-input", type=Path, default=DEFAULT_CCM)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--block-size-mb", type=int, default=64)
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


def nested_zip_to_temp(outer: Path, marker: str) -> Path:
    with zipfile.ZipFile(outer) as archive:
        member = next(i for i in archive.infolist() if marker.lower() in i.filename.lower())
        target = tempfile.NamedTemporaryFile(prefix="compustat_", suffix=".zip", delete=False)
        with archive.open(member) as source:
            shutil.copyfileobj(source, target, 16 * 1024 * 1024)
        target.close()
    return Path(target.name)


def open_only_csv(archive_path: Path) -> tuple[zipfile.ZipFile, BinaryIO]:
    archive = zipfile.ZipFile(archive_path)
    members = [i for i in archive.infolist() if not i.is_dir() and i.filename.lower().endswith(".csv")]
    if len(members) != 1:
        archive.close()
        raise ValueError(f"Expected one CSV in {archive_path}, found {len(members)}")
    return archive, archive.open(members[0])


def available_columns(stream: BinaryIO) -> list[str]:
    header = stream.readline().decode("utf-8-sig").rstrip("\r\n")
    stream.seek(0)
    return header.split(",")


def process_statement_group(frame: pl.DataFrame, kind: str) -> tuple[pl.DataFrame, dict[str, int]]:
    if frame.is_empty():
        return frame, {}
    standard = (
        (pl.col("datafmt") == "STD")
        & (pl.col("indfmt") == "INDL")
        & (pl.col("consol") == "C")
    )
    selected = frame.filter(standard)
    if selected.is_empty():
        return selected, {"raw_rows_complete": frame.height, "standard_rows": 0}

    duplicate_keys = (
        selected.group_by(["gvkey", "datadate"])
        .len()
        .filter(pl.col("len") > 1)
    )
    duplicate_key_count = duplicate_keys.height
    duplicate_extra_rows = int((duplicate_keys["len"] - 1).sum()) if duplicate_key_count else 0

    if kind == "quarterly":
        # Every duplicate date in this snapshot is a fiscal-year transition.
        # Compustat identifies exactly one row with a nonblank calendar quarter;
        # keep that row because it supplies a unique calendar-time statement.
        if duplicate_key_count:
            duplicate_structure = (
                selected.join(duplicate_keys.select(["gvkey", "datadate"]), on=["gvkey", "datadate"])
                .group_by(["gvkey", "datadate"])
                .agg(
                    (
                        pl.col("datacqtr").is_not_null() & (pl.col("datacqtr") != "")
                    ).sum().alias("calendar_labels")
                )
            )
            invalid_structure = duplicate_structure.filter(pl.col("calendar_labels") != 1).height
            if invalid_structure:
                raise ValueError(
                    f"Quarterly file has {invalid_structure} duplicate keys without exactly one calendar-quarter row"
                )
        selected = selected.with_columns(
            (pl.col("datacqtr").is_not_null() & (pl.col("datacqtr") != ""))
            .cast(pl.Int8)
            .alias("_calendar_quarter_priority")
        ).sort(["gvkey", "datadate", "_calendar_quarter_priority"])
        canonical = selected.unique(["gvkey", "datadate"], keep="last", maintain_order=True)
        valid_rdq = (
            pl.col("rdq").is_not_null()
            & (pl.col("rdq") >= pl.col("datadate"))
            & (pl.col("rdq") <= pl.col("datadate").dt.offset_by("1y"))
        )
        canonical = canonical.with_columns(
            pl.when(valid_rdq).then(pl.col("rdq")).otherwise(pl.col("datadate").dt.offset_by("4mo"))
            .alias("availability_date"),
            pl.when(valid_rdq).then(pl.lit("actual_rdq")).otherwise(pl.lit("fallback_4m"))
            .alias("availability_rule"),
        ).drop("_calendar_quarter_priority")
        actual_dates = int(canonical.select(valid_rdq.sum()).item())
        fallback_dates = canonical.height - actual_dates
    else:
        if duplicate_key_count:
            raise ValueError(f"Annual file has {duplicate_key_count} duplicate gvkey-datadate keys")
        canonical = selected.with_columns(
            pl.col("datadate").dt.offset_by("6mo").alias("availability_date"),
            pl.lit("conservative_6m").alias("availability_rule"),
        )
        actual_dates = 0
        fallback_dates = canonical.height

    return canonical, {
        "raw_rows_complete": frame.height,
        "standard_rows": selected.height,
        "duplicate_keys": duplicate_key_count,
        "duplicate_extra_rows": duplicate_extra_rows,
        "canonical_rows": canonical.height,
        "actual_report_dates": actual_dates,
        "conservative_dates": fallback_dates,
    }


def build_statement(
    nested: Path,
    kind: str,
    output: Path,
    block_size: int,
) -> dict[str, object]:
    dates = ANNUAL_DATES if kind == "annual" else QUARTERLY_DATES
    texts = ANNUAL_TEXT if kind == "annual" else QUARTERLY_TEXT
    numerics = ANNUAL_NUMERIC if kind == "annual" else QUARTERLY_NUMERIC
    archive, stream = open_only_csv(nested)
    counters: Counter[str] = Counter()
    writer: pq.ParquetWriter | None = None
    carry: pl.DataFrame | None = None
    min_date: date | None = None
    max_date: date | None = None
    try:
        header = available_columns(stream)
        wanted = list(dict.fromkeys(texts + dates + numerics))
        present = [c for c in wanted if c in header]
        missing = [c for c in wanted if c not in header]
        critical = {"gvkey", "datadate", "datafmt", "indfmt", "consol"}
        if critical - set(present):
            raise ValueError(f"Missing critical {kind} columns: {sorted(critical - set(present))}")
        types = {
            **{c: pa.string() for c in texts if c in present},
            **{c: pa.date32() for c in dates if c in present},
            **{c: pa.float64() for c in numerics if c in present},
        }
        reader = pacsv.open_csv(
            stream,
            read_options=pacsv.ReadOptions(block_size=block_size),
            convert_options=pacsv.ConvertOptions(
                include_columns=present,
                column_types=types,
                strings_can_be_null=True,
                null_values=[""],
            ),
        )
        for batch in reader:
            frame = pl.from_arrow(batch)
            counters["raw_rows_read"] += frame.height
            if carry is not None:
                frame = pl.concat([carry, frame], how="vertical")
            last_gvkey = frame.item(-1, "gvkey")
            complete = frame.filter(pl.col("gvkey") != last_gvkey)
            carry = frame.filter(pl.col("gvkey") == last_gvkey)
            canonical, stats = process_statement_group(complete, kind)
            counters.update(stats)
            if not canonical.is_empty():
                lo, hi = canonical["datadate"].min(), canonical["datadate"].max()
                min_date = min(min_date, lo) if min_date else lo
                max_date = max(max_date, hi) if max_date else hi
                table = canonical.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=100_000)
        if carry is not None and not carry.is_empty():
            canonical, stats = process_statement_group(carry, kind)
            counters.update(stats)
            if not canonical.is_empty():
                lo, hi = canonical["datadate"].min(), canonical["datadate"].max()
                min_date = min(min_date, lo) if min_date else lo
                max_date = max(max_date, hi) if max_date else hi
                table = canonical.to_arrow()
                if writer is None:
                    writer = pq.ParquetWriter(output, table.schema, compression="zstd")
                writer.write_table(table, row_group_size=100_000)
    finally:
        stream.close()
        archive.close()
        if writer is not None:
            writer.close()
    if writer is None:
        raise RuntimeError(f"No {kind} rows produced")
    return {
        "kind": kind,
        **dict(counters),
        "date_min": str(min_date),
        "date_max": str(max_date),
        "present_columns": present,
        "missing_requested_columns": missing,
    }


def canonicalize_cross_group_quarterly_duplicates(path: Path) -> dict[str, int]:
    """Resolve rare duplicate keys that are non-contiguous in the source CSV."""
    keys = pl.scan_parquet(path).select(["gvkey", "datadate"]).collect()
    duplicates = keys.group_by(["gvkey", "datadate"]).len().filter(pl.col("len") > 1)
    if duplicates.is_empty():
        return {"cross_group_duplicate_keys": 0, "cross_group_extra_rows": 0}

    key_rows = duplicates.select(["gvkey", "datadate"]).to_dicts()
    key_set = {(row["gvkey"], row["datadate"]) for row in key_rows}
    predicate = pl.any_horizontal(
        *[
            (pl.col("gvkey") == row["gvkey"]) & (pl.col("datadate") == row["datadate"])
            for row in key_rows
        ]
    )
    candidates = pl.scan_parquet(path).filter(predicate).collect().with_columns(
        (pl.col("datacqtr").is_not_null() & (pl.col("datacqtr") != ""))
        .cast(pl.Int8)
        .alias("_calendar_quarter_priority")
    )
    structure = candidates.group_by(["gvkey", "datadate"]).agg(
        pl.col("_calendar_quarter_priority").sum().alias("calendar_labels")
    )
    invalid = structure.filter(pl.col("calendar_labels") != 1).height
    if invalid:
        raise ValueError(
            f"Quarterly file has {invalid} cross-group duplicate keys without exactly one calendar-quarter row"
        )
    canonical = (
        candidates.sort(["gvkey", "datadate", "_calendar_quarter_priority"])
        .unique(["gvkey", "datadate"], keep="last", maintain_order=True)
        .drop("_calendar_quarter_priority")
    )

    source = pq.ParquetFile(path)
    temporary = path.with_suffix(".deduplicated.parquet")
    writer: pq.ParquetWriter | None = None
    emitted: set[tuple[str, date]] = set()
    try:
        for batch in source.iter_batches(batch_size=100_000):
            frame = pl.from_arrow(batch)
            batch_keys = list(zip(frame["gvkey"].to_list(), frame["datadate"].to_list()))
            affected = {key for key in batch_keys if key in key_set}
            keep = pl.Series([key not in key_set for key in batch_keys])
            clean = frame.filter(keep)
            additions = [key for key in affected if key not in emitted]
            if additions:
                add_set = set(additions)
                add_mask = pl.Series(
                    [key in add_set for key in zip(canonical["gvkey"].to_list(), canonical["datadate"].to_list())]
                )
                clean = pl.concat([clean, canonical.filter(add_mask)], how="vertical")
                emitted.update(additions)
            if not clean.is_empty():
                table = clean.to_arrow().cast(source.schema_arrow)
                if writer is None:
                    writer = pq.ParquetWriter(temporary, source.schema_arrow, compression="zstd")
                writer.write_table(table, row_group_size=100_000)
    finally:
        if writer is not None:
            writer.close()
    if emitted != key_set:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("Failed to emit all cross-group quarterly canonical rows")
    temporary.replace(path)
    extra = int((duplicates["len"] - 1).sum())
    return {
        "cross_group_duplicate_keys": duplicates.height,
        "cross_group_extra_rows": extra,
    }


def assert_unique_statement_keys(path: Path, kind: str) -> None:
    keys = pl.scan_parquet(path).select(["gvkey", "datadate"]).collect()
    duplicates = keys.group_by(["gvkey", "datadate"]).len().filter(pl.col("len") > 1).height
    if duplicates:
        raise ValueError(f"{kind} output still has {duplicates} duplicate gvkey-datadate keys")


def build_ccm(input_path: Path, output: Path) -> dict[str, object]:
    archive, stream = open_only_csv(input_path)
    try:
        header = available_columns(stream)
        present = header
        table = pacsv.read_csv(
            stream,
            convert_options=pacsv.ConvertOptions(
                include_columns=present,
                column_types={c: pa.string() for c in present},
                strings_can_be_null=True,
                null_values=[""],
            ),
        )
    finally:
        stream.close()
        archive.close()
    frame = pl.from_arrow(table).with_columns(
        *[pl.col(c).str.to_date(strict=False) for c in CCM_DATES if c in present],
        *[pl.col(c).cast(pl.Int64, strict=False) for c in CCM_NUMERIC if c in present],
    ).with_columns(
        pl.col("LINKDT").fill_null(date(1900, 1, 1)),
        pl.col("LINKENDDT").fill_null(date(9999, 12, 31)),
        (
            pl.col("LINKTYPE").is_in(["LC", "LU"])
            & pl.col("LINKPRIM").is_in(["P", "C"])
            & pl.col("LPERMNO").is_not_null()
        ).alias("research_link_flag"),
        pl.col("LINKTYPE").is_in(["LC", "LU", "LS"]).alias("extended_link_flag"),
    ).sort(["gvkey", "LINKDT", "LINKENDDT", "LPERMNO"])
    frame.write_parquet(output, compression="zstd")

    eligible = frame.filter(pl.col("research_link_flag"))
    interval_errors = eligible.filter(pl.col("LINKENDDT") < pl.col("LINKDT")).height
    exact_duplicates = eligible.height - eligible.unique(
        ["gvkey", "LPERMNO", "LINKDT", "LINKENDDT", "LINKTYPE", "LINKPRIM"]
    ).height
    return {
        "raw_rows": frame.height,
        "research_link_rows": eligible.height,
        "extended_link_rows": frame.filter(pl.col("extended_link_flag")).height,
        "unique_research_gvkeys": eligible["gvkey"].n_unique(),
        "unique_research_permnos": eligible["LPERMNO"].n_unique(),
        "invalid_intervals": interval_errors,
        "exact_duplicate_links": exact_duplicates,
        "linktype_counts": frame.group_by("LINKTYPE").len().sort("LINKTYPE").to_dicts(),
        "linkprim_counts": frame.group_by("LINKPRIM").len().sort("LINKPRIM").to_dicts(),
    }


def main() -> None:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    annual_path = output / "compustat_annual_pti.parquet"
    quarterly_path = output / "compustat_quarterly_pti.parquet"
    ccm_path = output / "ccm_link_history.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [annual_path, quarterly_path, ccm_path, report_path, manifest_path]
    if any(p.exists() for p in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite or choose another directory")
    output.mkdir(parents=True, exist_ok=True)
    for p in products:
        if p.exists():
            p.unlink()

    annual_zip = nested_zip_to_temp(args.compustat_input, "Annual")
    quarterly_zip = nested_zip_to_temp(args.compustat_input, "Quarterly")
    try:
        annual = build_statement(annual_zip, "annual", annual_path, args.block_size_mb * 1024 * 1024)
        quarterly = build_statement(quarterly_zip, "quarterly", quarterly_path, args.block_size_mb * 1024 * 1024)
    finally:
        annual_zip.unlink(missing_ok=True)
        quarterly_zip.unlink(missing_ok=True)
    assert_unique_statement_keys(annual_path, "annual")
    cross_group = canonicalize_cross_group_quarterly_duplicates(quarterly_path)
    quarterly.update(cross_group)
    quarterly["duplicate_keys"] += cross_group["cross_group_duplicate_keys"]
    quarterly["duplicate_extra_rows"] += cross_group["cross_group_extra_rows"]
    assert_unique_statement_keys(quarterly_path, "quarterly")
    quarterly_rules = (
        pl.scan_parquet(quarterly_path)
        .group_by("availability_rule")
        .len()
        .collect()
    )
    rule_counts = dict(zip(quarterly_rules["availability_rule"], quarterly_rules["len"]))
    quarterly["canonical_rows"] = pq.ParquetFile(quarterly_path).metadata.num_rows
    quarterly["actual_report_dates"] = int(rule_counts.get("actual_rdq", 0))
    quarterly["conservative_dates"] = int(rule_counts.get("fallback_4m", 0))
    ccm = build_ccm(args.ccm_input, ccm_path)

    report = {
        "experiment_id": "P1-G0-V005",
        "annual": annual,
        "quarterly": quarterly,
        "ccm": ccm,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V005",
        "data_snapshot_id": "wrds-us-equity-2025-12-v1",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "inputs": [
            {"path": str(args.compustat_input.resolve()), "size_bytes": args.compustat_input.stat().st_size},
            {"path": str(args.ccm_input.resolve()), "size_bytes": args.ccm_input.stat().st_size},
        ],
        "outputs": [
            {"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)}
            for p in [annual_path, quarterly_path, ccm_path, report_path]
        ],
        "quality_report": report,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
