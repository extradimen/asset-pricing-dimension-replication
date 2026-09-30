#!/usr/bin/env python3
"""Create a minimal, derived stock-return bridge for Paper 2 G4."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    product = output / "stock_outcomes.parquet"
    frame = (
        pl.scan_parquet(args.input)
        .select("permno", pl.col("target_month").alias("month"), pl.col("ret_fwd1").alias("ret"))
        .filter(pl.col("month") <= date.fromisoformat(config["maximum_target_month"]))
        .sort("month", "permno")
        .collect()
    )
    if frame.select(pl.struct("permno", "month").n_unique()).item() != len(frame):
        raise ValueError("Stock outcome bridge keys are not unique")
    frame.write_parquet(product, compression="zstd")
    report = {
        "schema_version": 1, "experiment_id": config["experiment_id"], "run_id": args.run_id,
        "status": "completed", "rows": len(frame), "securities": frame["permno"].n_unique(),
        "minimum_month": str(frame["month"].min()), "maximum_month": str(frame["month"].max()),
        "missing_returns": frame["ret"].null_count(), "raw_wrds_archive_included": False,
        "source_sha256": sha256(args.input),
    }
    report_path = output / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"], "run_id": args.run_id, "command": " ".join(os.sys.argv),
        "outputs": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in [product, report_path]],
    }
    (output / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
