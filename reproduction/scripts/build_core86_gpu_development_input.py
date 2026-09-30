#!/usr/bin/env python3
"""Add the audit-only market-cap column to the sealed-safe Core-86 panel."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/wrds-us-equity-2025-12-v1"
DEFAULT_PANEL = BASE / "P1-G0-V028/core86_training_pre2020.parquet"
DEFAULT_MASTER = BASE / "P1-G0-V006/us_equity_research_master.parquet"


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "core86_gpu_development_input.parquet"
    report_path = output_dir / "quality_report.json"
    manifest_path = output_dir / "output_manifest.json"
    paths = [output_path, report_path, manifest_path]
    if any(path.exists() for path in paths) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    for path in paths:
        path.unlink(missing_ok=True)

    panel_schema = pl.scan_parquet(args.panel).collect_schema()
    auxiliary = pl.scan_parquet(args.master).select("permno", "month", "market_cap")
    adapted = pl.scan_parquet(args.panel).join(
        auxiliary, on=["permno", "month"], how="left", validate="1:1"
    )
    output_columns = [
        "permno",
        "month",
        "target_month",
        "market_cap",
        *[
            name
            for name in panel_schema.names()
            if name not in {"permno", "month", "target_month"}
        ],
    ]
    adapted.select(output_columns).sink_parquet(output_path, compression="zstd", mkdir=True)

    observed = (
        pl.scan_parquet(output_path)
        .select(
            pl.len().alias("rows"),
            pl.struct("permno", "month").n_unique().alias("unique_keys"),
            pl.col("month").max().alias("maximum_feature_month"),
            pl.col("target_month").max().alias("maximum_target_month"),
            pl.col("market_cap").is_null().sum().alias("missing_market_cap"),
        )
        .collect()
        .row(0, named=True)
    )
    report = {
        "schema_version": 1,
        "experiment_id": "P1-G0-V029",
        **{key: str(value) if key.startswith("maximum_") else value for key, value in observed.items()},
        "added_columns": ["market_cap"],
        "market_cap_used_for_training_loss": False,
        "sealed_pricing_outputs_generated": False,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V029",
        "git_revision": git_revision(),
        "command": " ".join(os.sys.argv),
        "inputs": [
            {"role": role, "path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for role, path in [("panel", args.panel), ("master", args.master)]
        ],
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [output_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
