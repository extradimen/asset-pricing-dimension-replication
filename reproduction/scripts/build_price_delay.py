#!/usr/bin/env python3
"""Build Hou-Moskowitz price delay from three years of weekly returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import polars as pl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEEKLY_DIR = ROOT / "data/processed/wrds-us-equity-2025-12-v1/P1-G0-V013"
DEFAULT_GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
PERIODS = {
    "full_1957_2021": (date(1957, 1, 1), date(2021, 12, 1)),
    "early_1957_1979": (date(1957, 1, 1), date(1979, 12, 1)),
    "middle_1980_1999": (date(1980, 1, 1), date(1999, 12, 1)),
    "recent_2000_2021": (date(2000, 1, 1), date(2021, 12, 1)),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weekly-dir", type=Path, default=DEFAULT_WEEKLY_DIR)
    parser.add_argument("--market-file", type=Path)
    parser.add_argument("--gkx", type=Path, default=DEFAULT_GKX)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-rows", type=int, default=500_000)
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def rolling_sums(values: np.ndarray, ends: np.ndarray, starts: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate([np.zeros((1,) + values.shape[1:]), np.cumsum(values, axis=0)], axis=0)
    return cumulative[ends + 1] - cumulative[starts]


def solve_batch(matrices: np.ndarray, vectors: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(matrices, vectors)
    except np.linalg.LinAlgError:
        result = np.full_like(vectors, np.nan, dtype=float)
        for index, (matrix, vector) in enumerate(zip(matrices, vectors)):
            try:
                result[index] = np.linalg.solve(matrix, vector)
            except np.linalg.LinAlgError:
                continue
        return result


def calculate_security(frame: pl.DataFrame, market: pl.DataFrame) -> pl.DataFrame:
    joined = frame.join(market, on="week", how="left").drop_nulls(
        ["stock_weekly_return", "market_weekly_return", "market_lag1", "market_lag2", "market_lag3", "market_lag4"]
    ).sort("week")
    if joined.height < 52:
        return pl.DataFrame()
    y = joined["stock_weekly_return"].to_numpy()
    x = np.column_stack([
        np.ones(joined.height),
        *[joined[column].to_numpy() for column in ["market_weekly_return", "market_lag1", "market_lag2", "market_lag3", "market_lag4"]],
    ])
    months = joined["week"].to_numpy().astype("datetime64[M]")
    weeks = joined["week"].to_numpy()
    ends = np.flatnonzero(np.r_[months[1:] != months[:-1], True])
    cutoffs = joined["week"].gather(ends).dt.offset_by("-3y").to_numpy()
    starts = np.searchsorted(weeks, cutoffs, side="left")
    counts = ends - starts + 1
    keep = counts >= 52
    ends, starts, counts = ends[keep], starts[keep], counts[keep]
    if ends.size == 0:
        return pl.DataFrame()
    xtx = rolling_sums(x[:, :, None] * x[:, None, :], ends, starts)
    xty = rolling_sums(x * y[:, None], ends, starts)
    y2 = rolling_sums(y[:, None] ** 2, ends, starts)[:, 0]
    sy = rolling_sums(y[:, None], ends, starts)[:, 0]
    unrestricted_beta = solve_batch(xtx, xty)
    restricted_beta = solve_batch(xtx[:, :2, :2], xty[:, :2])
    sst = y2 - sy**2 / counts
    unrestricted_sse = y2 - np.einsum("ij,ij->i", unrestricted_beta, xty)
    restricted_sse = y2 - np.einsum("ij,ij->i", restricted_beta, xty[:, :2])
    with np.errstate(divide="ignore", invalid="ignore"):
        unrestricted_r2 = 1.0 - unrestricted_sse / sst
        restricted_r2 = 1.0 - restricted_sse / sst
        delay_rsq = 1.0 - restricted_r2 / unrestricted_r2
        lag_weights = np.arange(1.0, 5.0)
        delay_slope = (unrestricted_beta[:, 2:] * lag_weights).sum(axis=1) / unrestricted_beta[:, 1:].sum(axis=1)
    inv_xtx = np.linalg.pinv(xtx)
    residual_variance = unrestricted_sse / np.maximum(counts - x.shape[1], 1)
    standard_errors = np.sqrt(np.maximum(residual_variance[:, None] * np.diagonal(inv_xtx, axis1=1, axis2=2), 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        t_statistics = unrestricted_beta / standard_errors
        delay_tstat = (t_statistics[:, 2:] * lag_weights).sum(axis=1) / t_statistics[:, 1:].sum(axis=1)
    delay_slope = np.where(np.isfinite(delay_slope), delay_slope, np.nan)
    delay_tstat = np.where(np.isfinite(delay_tstat), delay_tstat, np.nan)
    delay_rsq = np.where(np.isfinite(delay_rsq) & (unrestricted_r2 > 0), delay_rsq, np.nan)
    return pl.DataFrame({
        "permno": np.repeat(joined.item(0, "permno"), ends.size),
        "month": months[ends].astype("datetime64[D]"),
        "pricedelay": delay_rsq,
        "pricedelay_rsq": delay_rsq,
        "pricedelay_slope": delay_slope,
        "pricedelay_tstat": delay_tstat,
        "pricedelay_weeks": counts.astype(np.int16),
        "restricted_r2": restricted_r2,
        "unrestricted_r2": unrestricted_r2,
    }).with_columns(pl.col("month").cast(pl.Date))


def audit_metric(frame: pl.DataFrame, shift: int, period: str, measure: str) -> dict:
    clean = frame.select("month", "pricedelay", "candidate").drop_nulls()
    ranked = clean.with_columns(
        pl.col("pricedelay").rank().over("month").alias("a"),
        pl.col("candidate").rank().over("month").alias("b"),
        pl.len().over("month").alias("n"),
    ).with_columns(((pl.col("a") - pl.col("b")).abs() / (pl.col("n") - 1)).alias("error"))
    monthly = ranked.group_by("month").agg(
        pl.corr("a", "b").alias("correlation"), pl.mean("error").alias("error"), pl.len().alias("rows")
    )
    corr = monthly["correlation"].drop_nulls(); corr = corr.filter(~corr.is_nan())
    return {
        "measure": measure, "shift_months": shift, "period": period, "rows": clean.height, "months": monthly.height,
        "mean_monthly_rank_correlation": corr.mean(), "median_monthly_rank_correlation": corr.median(),
        "p05_monthly_rank_correlation": corr.quantile(0.05, interpolation="linear"),
        "mean_absolute_percentile_error": monthly["error"].mean(),
    }


def main() -> None:
    args = parse_args(); started = time.time(); output = args.output_dir.resolve()
    panel_path = output / "price_delay.parquet"; audit_path = output / "price_delay_audit.csv"
    report_path = output / "quality_report.json"; manifest_path = output / "output_manifest.json"
    products = [panel_path, audit_path, report_path, manifest_path]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path in products: path.unlink(missing_ok=True)

    security_path = args.weekly_dir / "security_weekly_returns.parquet"
    market_path = args.market_file or (args.weekly_dir / "market_weekly_returns.parquet")
    market = pl.read_parquet(market_path).sort("week").with_columns(
        *[pl.col("market_weekly_return").shift(lag).alias(f"market_lag{lag}") for lag in range(1, 5)]
    ).select("week", "market_weekly_return", "market_lag1", "market_lag2", "market_lag3", "market_lag4")
    reader = pq.ParquetFile(security_path)
    writer: pq.ParquetWriter | None = None
    carry: pl.DataFrame | None = None
    rows = nonmissing = 0
    for batch in reader.iter_batches(batch_size=args.batch_rows):
        frame = pl.from_arrow(batch)
        if carry is not None: frame = pl.concat([carry, frame], how="vertical")
        last_permno = frame.item(-1, "permno")
        complete = frame.filter(pl.col("permno") != last_permno)
        carry = frame.filter(pl.col("permno") == last_permno)
        pieces = [calculate_security(group, market) for group in complete.partition_by("permno", maintain_order=True)]
        pieces = [piece for piece in pieces if not piece.is_empty()]
        if pieces:
            result = pl.concat(pieces).sort(["permno", "month"])
            rows += result.height; nonmissing += int(result["pricedelay"].is_not_null().sum())
            table = result.to_arrow()
            if writer is None: writer = pq.ParquetWriter(panel_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250_000)
    if carry is not None and not carry.is_empty():
        result = calculate_security(carry, market)
        if not result.is_empty():
            rows += result.height; nonmissing += int(result["pricedelay"].is_not_null().sum())
            table = result.to_arrow()
            if writer is None: writer = pq.ParquetWriter(panel_path, table.schema, compression="zstd")
            writer.write_table(table, row_group_size=250_000)
    if writer is None: raise RuntimeError("No price-delay observations produced")
    writer.close()

    panel = pl.read_parquet(panel_path); gkx = pl.read_parquet(args.gkx, columns=["permno", "month", "pricedelay"])
    audit_rows = []
    for measure in ["pricedelay_rsq", "pricedelay_slope", "pricedelay_tstat"]:
        for shift in [-1, 0, 1, 2, 3]:
            candidate = panel.select("permno", pl.col("month").dt.offset_by(f"{shift}mo"), pl.col(measure).alias("candidate"))
            joined = gkx.join(candidate, on=["permno", "month"])
            for period, (start, end) in PERIODS.items():
                audit_rows.append(audit_metric(joined.filter((pl.col("month") >= start) & (pl.col("month") <= end)), shift, period, measure))
    audit = pl.DataFrame(audit_rows); audit.write_csv(audit_path)
    best = audit.filter(pl.col("period") == "full_1957_2021").sort("mean_monthly_rank_correlation", descending=True).row(0, named=True)
    report = {
        "schema_version": 1, "experiment_id": "P1-G0-V013", "rows": rows, "nonmissing_pricedelay": nonmissing,
        "unique_keys": panel.select(pl.struct("permno", "month").n_unique()).item(),
        "first_month": str(panel["month"].min()), "last_month": str(panel["month"].max()),
        "definition": "Primary pricedelay is Hou-Moskowitz D1 incremental R-squared. D2 weighted-slope and D3 weighted-t-statistic variants are retained separately; weekly stock returns on contemporaneous equal-weighted market return plus four lags; trailing three calendar years, minimum 52 observations.",
        "best_matching_measure": best["measure"], "selected_shift_months": best["shift_months"], "mean_monthly_rank_correlation": best["mean_monthly_rank_correlation"],
        "median_monthly_rank_correlation": best["median_monthly_rank_correlation"], "p05_monthly_rank_correlation": best["p05_monthly_rank_correlation"],
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "experiment_id": "P1-G0-V013",
        "git_revision": git_revision(), "command": " ".join(os.sys.argv),
        "inputs": [{"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in [security_path, market_path, args.gkx]],
        "outputs": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in [panel_path, audit_path, report_path]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n"); print(json.dumps(report, indent=2))


if __name__ == "__main__": main()
