#!/usr/bin/env python3
"""Validate V028 without computing target or model-performance summaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

import polars as pl


LAST_FEATURE_MONTH = date(2019, 11, 1)
LAST_TARGET_MONTH = date(2019, 12, 1)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()

    feature_schema = pl.scan_parquet(args.features).collect_schema()
    panel_schema = pl.scan_parquet(args.input).collect_schema()
    expected_columns = [
        "permno",
        "month",
        "target_month",
        *[name for name in feature_schema.names() if name not in {"permno", "month"}],
        "ret_fwd1",
    ]
    expected_rows = (
        pl.scan_parquet(args.features)
        .select("month")
        .filter(pl.col("month") <= LAST_FEATURE_MONTH)
        .select(pl.len())
        .collect()
        .item()
    )
    observed = (
        pl.scan_parquet(args.input)
        .select(
            pl.len().alias("rows"),
            pl.struct("permno", "month").n_unique().alias("unique_keys"),
            pl.col("month").min().alias("minimum_feature_month"),
            pl.col("month").max().alias("maximum_feature_month"),
            pl.col("target_month").max().alias("maximum_target_month"),
            (pl.col("target_month") != pl.col("month").dt.offset_by("1mo"))
            .sum()
            .alias("calendar_mismatches"),
        )
        .collect()
        .row(0, named=True)
    )
    target_keys = (
        pl.scan_parquet(args.targets)
        .filter(pl.col("month") <= LAST_FEATURE_MONTH)
        .select("permno", "month")
        .unique()
    )
    missing_target_keys = (
        pl.scan_parquet(args.features)
        .filter(pl.col("month") <= LAST_FEATURE_MONTH)
        .select("permno", "month")
        .join(target_keys, on=["permno", "month"], how="anti")
        .select(pl.len())
        .collect()
        .item()
    )
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_inputs = {item["role"]: item for item in manifest["inputs"]}
    manifest_outputs = {item["path"]: item for item in manifest["outputs"]}
    checks = {
        "exact_schema": panel_schema.names() == expected_columns,
        "feature_types_preserved": all(
            panel_schema[name] == dtype for name, dtype in feature_schema.items()
        ),
        "target_type": panel_schema["ret_fwd1"] == pl.Float64,
        "expected_rows": observed["rows"] == expected_rows,
        "unique_keys": observed["rows"] == observed["unique_keys"],
        "feature_boundary": observed["maximum_feature_month"] == LAST_FEATURE_MONTH,
        "target_boundary": observed["maximum_target_month"] == LAST_TARGET_MONTH,
        "target_calendar_identity": observed["calendar_mismatches"] == 0,
        "target_source_covers_feature_keys": missing_target_keys == 0,
        "feature_source_hash": manifest_inputs["features"]["sha256"] == sha256(args.features),
        "target_source_hash": manifest_inputs["targets"]["sha256"] == sha256(args.targets),
        "output_hash": manifest_outputs[args.input.name]["sha256"] == sha256(args.input),
    }
    result = {
        "schema_version": 1,
        "experiment_id": "P1-G0-V028",
        "checks": checks,
        "rows": observed["rows"],
        "minimum_feature_month": str(observed["minimum_feature_month"]),
        "maximum_feature_month": str(observed["maximum_feature_month"]),
        "maximum_target_month": str(observed["maximum_target_month"]),
        "missing_target_source_keys": missing_target_keys,
        "target_values_summarized": False,
        "sealed_pricing_outputs_generated": False,
        "passed": all(checks.values()),
    }
    args.report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
