#!/usr/bin/env python3
"""Build the eleven point-in-time quarterly Core-92 characteristics."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/wrds-us-equity-2025-12-v1"
GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
FEATURES = ["cash", "chtx", "cinvest", "ms", "nincr", "roaq", "roavol", "roeq", "rsup", "stdacc", "stdcf"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def safe_div(numerator: pl.Expr, denominator: pl.Expr) -> pl.Expr:
    return pl.when(denominator.is_not_null() & (denominator.abs() > 1e-12)).then(numerator / denominator).otherwise(None)


def lag(column: str, periods: int) -> pl.Expr:
    return pl.col(column).shift(periods).over("gvkey")


def continuous_lag(column: str, periods: int, index: str) -> pl.Expr:
    return pl.when(pl.col(index) - lag(index, periods) == periods).then(lag(column, periods)).otherwise(None)


def rank_expression(feature: str) -> pl.Expr:
    count = pl.col(feature).count().over("month")
    rank = pl.col(feature).rank(method="average").over("month")
    return pl.when(pl.col(feature).is_not_null() & (count > 1)).then(
        2.0 * (rank - 1.0) / (count - 1.0) - 1.0
    ).otherwise(0.0).cast(pl.Float32).alias(f"x_{feature}")


def quarterly_signals(path: Path) -> pl.DataFrame:
    columns = [
        "gvkey", "datadate", "fyearq", "fqtr", "sic", "atq", "actq", "cheq", "lctq", "dlcq",
        "ppentq", "saleq", "ibq", "txtq", "seqq", "ceqq", "pstkq", "pstkrq", "ltq",
        "prccq", "cshoq", "availability_date", "availability_rule",
    ]
    q = pl.read_parquet(path, columns=columns).sort(["gvkey", "fyearq", "fqtr", "datadate"])
    q = q.with_columns(
        (pl.col("fyearq").cast(pl.Int64) * 4 + pl.col("fqtr").cast(pl.Int64)).alias("quarter_id"),
        pl.coalesce(
            pl.col("seqq"),
            pl.col("ceqq") + pl.coalesce(pl.col("pstkrq"), pl.col("pstkq"), pl.lit(0.0)),
            pl.col("atq") - pl.col("ltq"),
        ).alias("book_equity_q"),
        pl.col("sic").cast(pl.Utf8).str.slice(0, 2).alias("sic2"),
    )
    for n in range(1, 17):
        needed = ["atq", "actq", "cheq", "lctq", "dlcq", "ppentq", "saleq", "ibq", "txtq", "book_equity_q"]
        q = q.with_columns(*[continuous_lag(col, n, "quarter_id").alias(f"l{n}_{col}") for col in needed])
    denominator = pl.when(pl.col("saleq") > 0).then(pl.col("saleq")).otherwise(0.01)
    sacc = (
        (pl.col("actq") - pl.col("l1_actq"))
        - (pl.col("cheq") - pl.col("l1_cheq"))
        - (pl.col("lctq") - pl.col("l1_lctq"))
        + (pl.col("dlcq") - pl.col("l1_dlcq"))
    ) / denominator
    invq = (pl.col("ppentq") - pl.col("l1_ppentq")) / denominator
    q = q.with_columns(
        (pl.col("cheq") / pl.col("atq")).alias("cash"),
        safe_div(pl.col("txtq") - pl.col("l4_txtq"), pl.col("l4_atq")).alias("chtx"),
        invq.alias("invq"),
        safe_div(pl.col("ibq"), pl.col("l1_atq")).alias("roaq"),
        safe_div(pl.col("ibq"), pl.col("l1_book_equity_q")).alias("roeq"),
        safe_div(pl.col("saleq") - pl.col("l4_saleq"), pl.col("prccq").abs() * pl.col("cshoq").abs()).alias("rsup"),
        sacc.alias("sacc"),
        (pl.col("ibq") / denominator - sacc).alias("scf"),
        pl.when(pl.col("l4_ibq").is_not_null()).then((pl.col("ibq") > pl.col("l4_ibq")).cast(pl.Int8)).otherwise(None).alias("earn_increase"),
    )
    # invq lags must be created after invq exists.
    q = q.with_columns(*[continuous_lag("invq", n, "quarter_id").alias(f"l{n}_invq") for n in (1, 2, 3)])
    q = q.with_columns(
        (pl.col("invq") - pl.mean_horizontal("l1_invq", "l2_invq", "l3_invq")).alias("cinvest"),
        pl.col("roaq").rolling_std(16, min_samples=16, ddof=1).over("gvkey").alias("roavol"),
        pl.col("sacc").rolling_std(16, min_samples=16, ddof=1).over("gvkey").alias("stdacc"),
        pl.col("scf").rolling_std(16, min_samples=16, ddof=1).over("gvkey").alias("stdcf"),
        pl.col("rsup").rolling_std(16, min_samples=16, ddof=1).over("gvkey").alias("sgrvol"),
        pl.col("earn_increase").rolling_sum(8, min_samples=8).over("gvkey").alias("nincr"),
    ).with_columns(
        pl.when(pl.col("quarter_id") - lag("quarter_id", 15) == 15).then(pl.col("roavol")).otherwise(None).alias("roavol"),
        pl.when(pl.col("quarter_id") - lag("quarter_id", 15) == 15).then(pl.col("stdacc")).otherwise(None).alias("stdacc"),
        pl.when(pl.col("quarter_id") - lag("quarter_id", 15) == 15).then(pl.col("stdcf")).otherwise(None).alias("stdcf"),
        pl.when(pl.col("quarter_id") - lag("quarter_id", 15) == 15).then(pl.col("sgrvol")).otherwise(None).alias("sgrvol"),
        pl.when(pl.col("quarter_id") - lag("quarter_id", 7) == 7).then(pl.col("nincr")).otherwise(None).alias("nincr"),
    )
    q = q.with_columns(
        pl.col("roavol").median().over(["fyearq", "fqtr", "sic2"]).alias("industry_roavol_median"),
        pl.col("sgrvol").median().over(["fyearq", "fqtr", "sic2"]).alias("industry_sgrvol_median"),
    ).with_columns(
        pl.when(pl.col("roavol").is_not_null() & pl.col("industry_roavol_median").is_not_null())
        .then((pl.col("roavol") < pl.col("industry_roavol_median")).cast(pl.Int8)).otherwise(None).alias("ms_q7"),
        pl.when(pl.col("sgrvol").is_not_null() & pl.col("industry_sgrvol_median").is_not_null())
        .then((pl.col("sgrvol") < pl.col("industry_sgrvol_median")).cast(pl.Int8)).otherwise(None).alias("ms_q8"),
    )
    return q.select("gvkey", "datadate", "availability_date", "availability_rule", *[f for f in FEATURES if f != "ms"], "ms_q7", "ms_q8")


def annual_ms_components(path: Path) -> pl.DataFrame:
    columns = ["gvkey", "datadate", "fyear", "sic", "at", "ni", "oancf", "xrd", "capx", "xad"]
    a = pl.read_parquet(path, columns=columns).with_columns(
        pl.col("fyear").cast(pl.Int64, strict=False)
    ).sort(["gvkey", "fyear", "datadate"]).with_columns(
        pl.col("sic").cast(pl.Utf8).str.slice(0, 2).alias("sic2"),
        (pl.col("fyear") - lag("fyear", 1)).alias("fyear_gap"),
        lag("at", 1).alias("lag_at_unchecked"),
    ).with_columns(
        pl.when(pl.col("fyear_gap") == 1).then(pl.col("lag_at_unchecked")).otherwise(None).alias("lag_at")
    ).with_columns(
        safe_div(pl.col("ni"), (pl.col("at") + pl.col("lag_at")) / 2).alias("ms_roa"),
        safe_div(pl.col("oancf"), (pl.col("at") + pl.col("lag_at")) / 2).alias("ms_cfroa"),
        safe_div(pl.col("xrd").fill_null(0.0), pl.col("lag_at")).alias("ms_xrdint"),
        safe_div(pl.col("capx"), pl.col("lag_at")).alias("ms_capxint"),
        safe_div(pl.col("xad").fill_null(0.0), pl.col("lag_at")).alias("ms_xadint"),
    )
    for col in ["ms_roa", "ms_cfroa", "ms_xrdint", "ms_capxint", "ms_xadint"]:
        a = a.with_columns(pl.col(col).median().over(["fyear", "sic2"]).alias(f"industry_{col}_median"))
    return a.with_columns(
        (pl.col("ms_roa") > pl.col("industry_ms_roa_median")).cast(pl.Int8).alias("ms_a1"),
        (pl.col("ms_cfroa") > pl.col("industry_ms_cfroa_median")).cast(pl.Int8).alias("ms_a2"),
        (pl.col("oancf") > pl.col("ni")).cast(pl.Int8).alias("ms_a3"),
        (pl.col("ms_xrdint") > pl.col("industry_ms_xrdint_median")).cast(pl.Int8).alias("ms_a4"),
        (pl.col("ms_capxint") > pl.col("industry_ms_capxint_median")).cast(pl.Int8).alias("ms_a5"),
        (pl.col("ms_xadint") > pl.col("industry_ms_xadint_median")).cast(pl.Int8).alias("ms_a6"),
    ).select("gvkey", "datadate", "ms_a1", "ms_a2", "ms_a3", "ms_a4", "ms_a5", "ms_a6")


def main() -> None:
    args = parse_args(); started = time.time(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    self_path = output / "quarterly11_self_built.parquet"
    bridge_path = output / "quarterly11_bridged_raw.parquet"
    model_path = output / "quarterly11_model_input.parquet"
    report_path = output / "quality_report.json"
    manifest_path = output / "output_manifest.json"
    products = [self_path, bridge_path, model_path, report_path, manifest_path]
    if any(p.exists() for p in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    for p in products: p.unlink(missing_ok=True)

    q_path = PROCESSED / "P1-G0-V005/compustat_quarterly_pti.parquet"
    a_path = PROCESSED / "P1-G0-V005/compustat_annual_pti.parquet"
    master_path = PROCESSED / "P1-G0-V006/us_equity_research_master.parquet"
    q = quarterly_signals(q_path)
    annual = annual_ms_components(a_path)
    keys = pl.read_parquet(master_path, columns=["permno", "month", "gvkey", "q_datadate", "a_datadate", "ret_fwd1"])
    self_built = keys.join(q, left_on=["gvkey", "q_datadate"], right_on=["gvkey", "datadate"], how="left")
    self_built = self_built.join(annual, left_on=["gvkey", "a_datadate"], right_on=["gvkey", "datadate"], how="left")
    components = [f"ms_a{i}" for i in range(1, 7)] + ["ms_q7", "ms_q8"]
    self_built = self_built.with_columns(
        pl.when(pl.all_horizontal(*[pl.col(c).is_not_null() for c in components]))
        .then(pl.sum_horizontal(*components)).otherwise(None).cast(pl.Float64).alias("ms")
    ).select("permno", "month", "ret_fwd1", "availability_date", "availability_rule", *FEATURES).sort(["month", "permno"])
    self_built.write_parquet(self_path, compression="zstd")

    official = pl.read_parquet(GKX, columns=["permno", "month", *FEATURES]).rename({f: f"gkx_{f}" for f in FEATURES})
    joined = self_built.join(official, on=["permno", "month"], how="left")
    bridge = joined.with_columns(
        *[pl.coalesce(pl.col(f"gkx_{f}"), pl.col(f)).alias(f) for f in FEATURES],
        pl.sum_horizontal(*[pl.col(f"gkx_{f}").is_not_null().cast(pl.UInt8) for f in FEATURES]).alias("official_feature_count"),
    ).with_columns(
        pl.when(pl.col("official_feature_count") == len(FEATURES)).then(pl.lit("official_full"))
        .when(pl.col("official_feature_count") > 0).then(pl.lit("official_partial"))
        .otherwise(pl.lit("self_built")).alias("feature_source")
    ).select("permno", "month", "ret_fwd1", "feature_source", "official_feature_count", *FEATURES)
    bridge.write_parquet(bridge_path, compression="zstd")
    model = bridge.select(
        "permno", "month", "ret_fwd1", "feature_source", "official_feature_count",
        *[rank_expression(f) for f in FEATURES],
        *[pl.col(f).is_null().cast(pl.Int8).alias(f"missing_{f}") for f in FEATURES],
    )
    model.write_parquet(model_path, compression="zstd")

    report = {
        "schema_version": 1, "experiment_id": "P1-G0-V015", "rows": bridge.height,
        "unique_keys": bridge.select(pl.struct("permno", "month").n_unique()).item(),
        "first_month": str(bridge["month"].min()), "last_month": str(bridge["month"].max()),
        "features": FEATURES,
        "self_built_nonmissing": self_built.select(*[pl.col(f).is_not_null().sum().alias(f) for f in FEATURES]).row(0, named=True),
        "bridged_nonmissing": bridge.select(*[pl.col(f).is_not_null().sum().alias(f) for f in FEATURES]).row(0, named=True),
        "source_counts": {row[0]: row[1] for row in bridge.group_by("feature_source").len().iter_rows()},
        "sealed_pricing_outputs_generated": False,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    inputs = [q_path, a_path, master_path, GKX]
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "experiment_id": "P1-G0-V015",
        "git_revision": git_revision(), "command": " ".join(os.sys.argv),
        "inputs": [{"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in inputs],
        "outputs": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in [self_path, bridge_path, model_path, report_path]],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
