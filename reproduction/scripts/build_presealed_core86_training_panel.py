#!/usr/bin/env python3
"""Build the leakage-safe pre-2020 Core-86 training panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/wrds-us-equity-2025-12-v1"
DEFAULT_FEATURES = BASE / "P1-G0-V027/core86_model_input_pre2020.parquet"
DEFAULT_TARGETS = BASE / "P1-G0-V006/us_equity_research_master.parquet"
LAST_FEATURE_MONTH = date(2019, 11, 1)
LAST_TARGET_MONTH = date(2019, 12, 1)


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


def eligible_features(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Keep only feature months whose t+1 target remains before the seal."""
    return frame.filter(pl.col("month") <= LAST_FEATURE_MONTH)


def eligible_targets(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Project only the key and pre-sealed target before the join."""
    return (
        frame.filter(pl.col("month") <= LAST_FEATURE_MONTH)
        .select("permno", "month", "ret_fwd1")
        .with_columns(pl.lit(True).alias("_target_row_present"))
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--targets", type=Path, default=DEFAULT_TARGETS)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    started = time.time()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = output_dir / "core86_training_pre2020.parquet"
    report_path = output_dir / "quality_report.json"
    manifest_path = output_dir / "output_manifest.json"
    paths = [panel_path, report_path, manifest_path]
    if any(path.exists() for path in paths) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    for path in paths:
        path.unlink(missing_ok=True)

    features = eligible_features(pl.scan_parquet(args.features))
    targets = eligible_targets(pl.scan_parquet(args.targets))
    joined = features.join(targets, on=["permno", "month"], how="left", validate="1:1")

    join_audit = joined.select(
        pl.len().alias("rows"),
        pl.col("_target_row_present").is_null().sum().alias("missing_target_source_keys"),
    ).collect(engine="streaming").row(0, named=True)
    if join_audit["missing_target_source_keys"]:
        raise RuntimeError("Some Core-86 keys are absent from the target source")

    panel = (
        joined.drop("_target_row_present")
        .with_columns(pl.col("month").dt.offset_by("1mo").alias("target_month"))
        .select(
            "permno",
            "month",
            "target_month",
            *[
                name
                for name in pl.scan_parquet(args.features).collect_schema().names()
                if name not in {"permno", "month"}
            ],
            "ret_fwd1",
        )
    )
    panel.sink_parquet(panel_path, compression="zstd", mkdir=True)

    observed = (
        pl.scan_parquet(panel_path)
        .select(
            pl.len().alias("rows"),
            pl.struct("permno", "month").n_unique().alias("unique_keys"),
            pl.col("month").min().alias("minimum_feature_month"),
            pl.col("month").max().alias("maximum_feature_month"),
            pl.col("target_month").max().alias("maximum_target_month"),
            pl.col("ret_fwd1").is_not_null().sum().alias("non_null_targets"),
        )
        .collect()
        .row(0, named=True)
    )
    if observed["maximum_feature_month"] != LAST_FEATURE_MONTH:
        raise RuntimeError("Unexpected final feature month")
    if observed["maximum_target_month"] != LAST_TARGET_MONTH:
        raise RuntimeError("Unexpected final target month")

    report = {
        "schema_version": 1,
        "experiment_id": "P1-G0-V028",
        "rows": observed["rows"],
        "unique_keys": observed["unique_keys"],
        "minimum_feature_month": str(observed["minimum_feature_month"]),
        "maximum_feature_month": str(observed["maximum_feature_month"]),
        "maximum_target_month": str(observed["maximum_target_month"]),
        "non_null_targets": observed["non_null_targets"],
        "missing_target_source_keys": join_audit["missing_target_source_keys"],
        "target_distribution_summarized": False,
        "sealed_pricing_outputs_generated": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V028",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "boundary_rule": (
            "Feature month is at most 2019-11; target_month is exactly feature month + 1, "
            "so no 2020 return enters the development panel."
        ),
        "inputs": [
            {
                "role": role,
                "path": str(path.resolve()),
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for role, path in [("features", args.features), ("targets", args.targets)]
        ],
        "outputs": [
            {
                "path": path.name,
                "size_bytes": path.stat().st_size,
                "sha256": sha256(path),
            }
            for path in [panel_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
