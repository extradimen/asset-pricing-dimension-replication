#!/usr/bin/env python3
"""Build equal-weighted two-digit-SIC industry momentum and audit against GKX."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MASTER = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V006/us_equity_research_master.parquet"
DEFAULT_MOMENTUM = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V010/momentum_characteristics.parquet"
DEFAULT_GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
PERIODS = {
    "full_1957_2021": (date(1957, 1, 1), date(2021, 12, 1)),
    "early_1957_1979": (date(1957, 1, 1), date(1979, 12, 1)),
    "middle_1980_1999": (date(1980, 1, 1), date(1999, 12, 1)),
    "recent_2000_2021": (date(2000, 1, 1), date(2021, 12, 1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master", type=Path, default=DEFAULT_MASTER)
    parser.add_argument("--momentum", type=Path, default=DEFAULT_MOMENTUM)
    parser.add_argument("--gkx", type=Path, default=DEFAULT_GKX)
    parser.add_argument("--output-dir", type=Path, required=True)
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


def sic2_expression() -> pl.Expr:
    raw = pl.coalesce(
        pl.col("a_sic").cast(pl.Int64, strict=False),
        pl.col("q_sic").cast(pl.Int64, strict=False),
        pl.col("siccd").cast(pl.Int64, strict=False),
    )
    code = (raw // 100).cast(pl.Int16)
    return pl.when(code.is_between(1, 99)).then(code).otherwise(None).alias("industry_sic2")


def audit_metric(frame: pl.DataFrame, shift: int, period: str) -> dict:
    clean = frame.select("month", "indmom", "candidate").drop_nulls()
    ranked = clean.with_columns(
        pl.col("indmom").rank(method="average").over("month").alias("a"),
        pl.col("candidate").rank(method="average").over("month").alias("b"),
        pl.len().over("month").alias("n"),
    ).with_columns(((pl.col("a") - pl.col("b")).abs() / (pl.col("n") - 1)).alias("error"))
    monthly = ranked.group_by("month").agg(
        pl.corr("a", "b").alias("correlation"), pl.mean("error").alias("error"), pl.len().alias("rows")
    )
    corr = monthly["correlation"].drop_nulls()
    corr = corr.filter(~corr.is_nan())
    return {
        "shift_months": shift, "period": period, "rows": clean.height, "months": monthly.height,
        "mean_monthly_rank_correlation": corr.mean(), "median_monthly_rank_correlation": corr.median(),
        "p05_monthly_rank_correlation": corr.quantile(0.05, interpolation="linear"),
        "mean_absolute_percentile_error": monthly["error"].mean(),
    }


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    panel_path = output / "industry_momentum.parquet"
    audit_path = output / "industry_momentum_audit.csv"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [panel_path, audit_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    base = (
        pl.scan_parquet(args.master)
        .select("permno", "month", "a_sic", "q_sic", "siccd")
        .join(pl.scan_parquet(args.momentum).select("permno", "month", "mom12m"), on=["permno", "month"])
        .with_columns(sic2_expression())
        .collect()
    )
    industries = base.group_by(["month", "industry_sic2"]).agg(
        pl.col("mom12m").mean().alias("indmom"),
        pl.col("mom12m").count().cast(pl.Int32).alias("industry_firms"),
    )
    panel = (
        base.select("permno", "month", "industry_sic2")
        .join(industries, on=["month", "industry_sic2"], how="left")
        .sort(["permno", "month"])
    )
    panel.write_parquet(panel_path, compression="zstd")

    gkx = pl.read_parquet(args.gkx, columns=["permno", "month", "sic2", "indmom"])
    rows = []
    for shift in [-1, 0, 1]:
        candidate = panel.select(
            "permno", pl.col("month").dt.offset_by(f"{shift}mo"),
            pl.col("indmom").alias("candidate"), pl.col("industry_sic2")
        )
        joined = gkx.join(candidate, on=["permno", "month"])
        for period, (start, end) in PERIODS.items():
            rows.append(audit_metric(joined.filter((pl.col("month") >= start) & (pl.col("month") <= end)), shift, period))
    audit = pl.DataFrame(rows)
    audit.write_csv(audit_path)
    full = audit.filter(pl.col("period") == "full_1957_2021").sort("mean_monthly_rank_correlation", descending=True)
    best = full.row(0, named=True)

    overlap = gkx.join(panel.select("permno", "month", "industry_sic2"), on=["permno", "month"]).drop_nulls(["sic2", "industry_sic2"])
    report = {
        "schema_version": 1, "experiment_id": "P1-G0-V012",
        "rows": panel.height, "unique_keys": panel.select(pl.struct("permno", "month").n_unique()).item(),
        "nonmissing_indmom": int(panel["indmom"].is_not_null().sum()),
        "first_month": str(panel["month"].min()), "last_month": str(panel["month"].max()),
        "industry_source_priority": ["Compustat annual SIC", "Compustat quarterly SIC", "CRSP SIC fallback"],
        "gkx_sic2_match_rate": float((overlap["sic2"] == overlap["industry_sic2"]).mean()),
        "selected_shift_months": best["shift_months"],
        "mean_monthly_rank_correlation": best["mean_monthly_rank_correlation"],
        "median_monthly_rank_correlation": best["median_monthly_rank_correlation"],
        "p05_monthly_rank_correlation": best["p05_monthly_rank_correlation"],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": "P1-G0-V012", "git_revision": git_revision(), "command": " ".join(os.sys.argv),
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [args.master, args.momentum, args.gkx]
        ],
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [panel_path, audit_path, report_path]
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
