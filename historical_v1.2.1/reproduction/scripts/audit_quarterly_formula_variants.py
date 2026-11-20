#!/usr/bin/env python3
"""Adjudicate source-documented definitions for the seven failed Core-11 features."""
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
GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
FEATURES = ["chtx", "cinvest", "ms", "nincr", "roaq", "roeq", "rsup"]
DISCRETE = {"ms", "nincr"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def safe_div(n: pl.Expr, d: pl.Expr) -> pl.Expr:
    return pl.when(d.is_not_null() & (d.abs() > 1e-12)).then(n / d).otherwise(None)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def git_revision() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def streak_expr(prefix: str) -> pl.Expr:
    """Length of the positive streak ending in the current quarter, capped at eight."""
    product = pl.lit(1, dtype=pl.Int16)
    terms: list[pl.Expr] = []
    for n in range(8):
        product = product * pl.col(f"{prefix}{n}").fill_null(False).cast(pl.Int16)
        terms.append(product)
    return pl.sum_horizontal(*terms)


def quarter_candidates(path: Path) -> pl.DataFrame:
    cols = [
        "gvkey", "datadate", "fyearq", "fqtr", "sic", "atq", "ppentq", "saleq", "ibq",
        "txtq", "seqq", "ceqq", "pstkq", "pstkrq", "ltq", "prccq", "cshoq",
    ]
    q = pl.read_parquet(path, columns=cols).with_columns(
        (pl.col("fyearq").cast(pl.Int64, strict=False) * 4 + pl.col("fqtr").cast(pl.Int64, strict=False)).alias("qid"),
        pl.coalesce(
            pl.col("seqq"),
            pl.col("ceqq") + pl.coalesce(pl.col("pstkrq"), pl.col("pstkq"), pl.lit(0.0)),
            pl.col("atq") - pl.col("ltq"),
        ).alias("beq"),
        (pl.col("prccq").abs() * pl.col("cshoq").abs()).alias("mveq"),
        pl.col("sic").cast(pl.Utf8).str.slice(0, 2).alias("sic2"),
    ).sort(["gvkey", "qid", "datadate"])
    base = ["atq", "ppentq", "saleq", "ibq", "txtq", "beq", "mveq"]
    q = q.with_columns(*[pl.col(c).shift(n).over("gvkey").alias(f"l{n}_{c}") for c in base for n in range(1, 9)])
    # Null out row lags that cross a missing fiscal quarter.
    for n in range(1, 9):
        q = q.with_columns(*[
            pl.when(pl.col("qid") - pl.col("qid").shift(n).over("gvkey") == n)
            .then(pl.col(f"l{n}_{c}")).otherwise(None).alias(f"l{n}_{c}") for c in base
        ])
    denom = pl.when(pl.col("saleq") > 0).then(pl.col("saleq")).otherwise(0.01)
    inv = safe_div(pl.col("ppentq") - pl.col("l1_ppentq"), denom)
    q = q.with_columns(
        inv.alias("inv"),
        safe_div(pl.col("txtq") - pl.col("l4_txtq"), pl.col("l4_atq")).alias("chtx_lag4"),
        safe_div(pl.col("txtq") - pl.col("l4_txtq"), pl.col("atq")).alias("chtx_current"),
        safe_div(pl.col("ibq"), pl.col("l1_atq")).alias("roaq_lag1"),
        safe_div(pl.col("ibq"), pl.col("atq")).alias("roaq_current"),
        safe_div(pl.col("ibq"), pl.col("l1_beq")).alias("roeq_lag1"),
        safe_div(pl.col("ibq"), pl.col("beq")).alias("roeq_current"),
        safe_div(pl.col("saleq") - pl.col("l4_saleq"), pl.col("mveq")).alias("rsup_current"),
        safe_div(pl.col("saleq") - pl.col("l4_saleq"), pl.col("l1_mveq")).alias("rsup_lag1"),
    )
    q = q.with_columns(*[pl.col("inv").shift(n).over("gvkey").alias(f"l{n}_inv") for n in range(1, 5)])
    for n in range(1, 5):
        q = q.with_columns(
            pl.when(pl.col("qid") - pl.col("qid").shift(n).over("gvkey") == n)
            .then(pl.col(f"l{n}_inv")).otherwise(None).alias(f"l{n}_inv")
        )
    q = q.with_columns(
        (pl.col("inv") - pl.mean_horizontal("l1_inv", "l2_inv", "l3_inv")).alias("cinvest_prev3"),
        (pl.col("inv") - pl.mean_horizontal("l1_inv", "l2_inv", "l3_inv", "l4_inv")).alias("cinvest_prev4"),
    )
    # Two documented earnings-growth conventions; both are consecutive streaks.
    q = q.with_columns(*[
        (pl.col("ibq").shift(n).over("gvkey") > pl.col("ibq").shift(n + 4).over("gvkey")).alias(f"yoy{n}")
        for n in range(8)
    ], *[
        (pl.col("ibq").shift(n).over("gvkey") > pl.col("ibq").shift(n + 1).over("gvkey")).alias(f"qoq{n}")
        for n in range(8)
    ])
    q = q.with_columns(
        streak_expr("yoy").alias("nincr_yoy"),
        streak_expr("qoq").alias("nincr_qoq"),
    )
    # Quarterly Mohanram components use 16 consecutive quarters.
    q = q.with_columns(
        pl.col("roaq_lag1").rolling_std(16, min_samples=16, ddof=1).over("gvkey").alias("roavol"),
        pl.col("rsup_current").rolling_std(16, min_samples=16, ddof=1).over("gvkey").alias("sgrvol"),
    ).with_columns(
        pl.when(pl.col("qid") - pl.col("qid").shift(15).over("gvkey") == 15).then(pl.col("roavol")).otherwise(None).alias("roavol"),
        pl.when(pl.col("qid") - pl.col("qid").shift(15).over("gvkey") == 15).then(pl.col("sgrvol")).otherwise(None).alias("sgrvol"),
    ).with_columns(
        pl.col("roavol").median().over(["fyearq", "fqtr", "sic2"]).alias("md_roavol"),
        pl.col("sgrvol").median().over(["fyearq", "fqtr", "sic2"]).alias("md_sgrvol"),
    ).with_columns(
        ((pl.col("roavol") < pl.col("md_roavol")).cast(pl.Int8) + (pl.col("sgrvol") < pl.col("md_sgrvol")).cast(pl.Int8)).alias("msq_low"),
        ((pl.col("roavol") > pl.col("md_roavol")).cast(pl.Int8) + (pl.col("sgrvol") > pl.col("md_sgrvol")).cast(pl.Int8)).alias("msq_high"),
        pl.col("datadate").dt.offset_by("4mo").dt.truncate("1mo").alias("fixed_month"),
    )
    keep = [c for c in q.columns if c.startswith(("chtx_", "cinvest_", "nincr_", "roaq_", "roeq_", "rsup_", "msq_"))]
    return q.select("gvkey", "datadate", "fixed_month", *keep)


def annual_ms(path: Path) -> pl.DataFrame:
    cols = ["gvkey", "datadate", "fyear", "sic", "at", "ni", "oancf", "xrd", "capx", "xad"]
    a = pl.read_parquet(path, columns=cols).with_columns(
        pl.col("fyear").cast(pl.Int64, strict=False),
        pl.col("sic").cast(pl.Utf8).str.slice(0, 2).alias("sic2"),
    ).sort(["gvkey", "fyear", "datadate"]).with_columns(
        pl.col("at").shift(1).over("gvkey").alias("lag_at"),
        (pl.col("fyear") - pl.col("fyear").shift(1).over("gvkey")).alias("gap"),
    ).with_columns(
        pl.when(pl.col("gap") == 1).then(pl.col("lag_at")).otherwise(None).alias("lag_at")
    ).with_columns(
        safe_div(pl.col("ni"), (pl.col("at") + pl.col("lag_at")) / 2).alias("roa"),
        safe_div(pl.col("oancf"), (pl.col("at") + pl.col("lag_at")) / 2).alias("cfroa"),
        safe_div(pl.col("xrd").fill_null(0), pl.col("lag_at")).alias("xrdint"),
        safe_div(pl.col("capx"), pl.col("lag_at")).alias("capxint"),
        safe_div(pl.col("xad").fill_null(0), pl.col("lag_at")).alias("xadint"),
    )
    for c in ["roa", "cfroa", "xrdint", "capxint", "xadint"]:
        a = a.with_columns(pl.col(c).median().over(["fyear", "sic2"]).alias(f"md_{c}"))
    return a.with_columns(
        (
            (pl.col("roa") > pl.col("md_roa")).cast(pl.Int8)
            + (pl.col("cfroa") > pl.col("md_cfroa")).cast(pl.Int8)
            + (pl.col("oancf") > pl.col("ni")).cast(pl.Int8)
            + (pl.col("xrdint") > pl.col("md_xrdint")).cast(pl.Int8)
            + (pl.col("capxint") > pl.col("md_capxint")).cast(pl.Int8)
            + (pl.col("xadint") > pl.col("md_xadint")).cast(pl.Int8)
        ).alias("msa"),
        pl.col("datadate").dt.offset_by("6mo").dt.truncate("1mo").alias("fixed_month"),
    ).select("gvkey", "datadate", "fixed_month", "msa")


def attach_timing(master: pl.DataFrame, q: pl.DataFrame, a: pl.DataFrame, timing: str) -> pl.DataFrame:
    if timing == "hybrid":
        out = master.join(q.drop("fixed_month"), left_on=["gvkey", "q_datadate"], right_on=["gvkey", "datadate"], how="left")
        return out.join(a.drop("fixed_month"), left_on=["gvkey", "a_datadate"], right_on=["gvkey", "datadate"], how="left")
    left = master.sort(["gvkey", "month"])
    qright = q.sort(["gvkey", "fixed_month", "datadate"]).unique(["gvkey", "fixed_month"], keep="last")
    aright = a.sort(["gvkey", "fixed_month", "datadate"]).unique(["gvkey", "fixed_month"], keep="last")
    out = left.join_asof(qright.drop("datadate"), left_on="month", right_on="fixed_month", by="gvkey", strategy="backward")
    return out.join_asof(aright.drop("datadate"), left_on="month", right_on="fixed_month", by="gvkey", strategy="backward", suffix="_annual")


def candidate_panel(master: pl.DataFrame, q: pl.DataFrame, a: pl.DataFrame) -> pl.DataFrame:
    frames = []
    for timing in ["hybrid", "fixed"]:
        x = attach_timing(master, q, a, timing)
        frames.append(x.select(
            "permno", "month",
            *[pl.col(c).alias(f"{timing}__{c}") for c in [
                "chtx_lag4", "chtx_current", "cinvest_prev3", "cinvest_prev4",
                "nincr_yoy", "nincr_qoq", "roaq_lag1", "roaq_current",
                "roeq_lag1", "roeq_current", "rsup_current", "rsup_lag1",
            ]],
            (pl.col("msa") + pl.col("msq_low")).alias(f"{timing}__ms_low"),
            (pl.col("msa") + pl.col("msq_high")).alias(f"{timing}__ms_high"),
        ))
    return frames[0].join(frames[1], on=["permno", "month"], how="inner")


def rank_expr(c: str, alias: str) -> pl.Expr:
    n = pl.col(c).count().over("month")
    r = pl.col(c).rank(method="average").over("month")
    return (2 * (r - 1) / (n - 1) - 1).alias(alias)


def metrics(panel: pl.DataFrame, candidate: str, official: str, start: str, end: str) -> dict[str, object]:
    feature = official.removeprefix("official_")
    pair = panel.filter(
        pl.col("month").is_between(pl.date(int(start[:4]), int(start[5:]), 1), pl.date(int(end[:4]), int(end[5:]), 1))
        & pl.col(candidate).is_not_null() & pl.col(official).is_not_null()
    ).select("month", candidate, official)
    monthly = pair.with_columns(rank_expr(candidate, "cr"), rank_expr(official, "or")).group_by("month").agg(
        pl.len().alias("n"),
        pl.corr(candidate, official, method="spearman").alias("spearman"),
        (pl.col("cr") - pl.col("or")).abs().mean().alias("rank_mae"),
    ).filter(pl.col("n") >= 30)
    return {
        "feature": feature, "candidate": candidate, "period_start": start, "period_end": end, "overlap_rows": pair.height,
        "months": monthly.height,
        "median_monthly_spearman": float(monthly["spearman"].median()) if monthly.height else None,
        "median_monthly_rank_mae": float(monthly["rank_mae"].median()) if monthly.height else None,
        "exact_agreement": float((pair[candidate] == pair[official]).mean()) if feature in DISCRETE and pair.height else None,
    }


def choose(rows: list[dict[str, object]], feature: str) -> dict[str, object]:
    eligible = [r for r in rows if r["feature"] == feature]
    if feature in DISCRETE:
        return max(eligible, key=lambda r: (r["exact_agreement"] or -1, r["median_monthly_spearman"] or -2, r["candidate"].endswith(("nincr_yoy", "ms_low"))))
    return max(eligible, key=lambda r: (r["median_monthly_spearman"] or -2, -(r["median_monthly_rank_mae"] or 9), r["candidate"]))


def main() -> None:
    args = parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    outputs = [out / n for n in ["candidate_panel.parquet", "selection_metrics.csv", "confirmation_metrics.csv", "result_summary.json", "output_manifest.json"]]
    if any(p.exists() for p in outputs) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    for p in outputs: p.unlink(missing_ok=True)
    qpath = BASE / "P1-G0-V005/compustat_quarterly_pti.parquet"
    apath = BASE / "P1-G0-V005/compustat_annual_pti.parquet"
    mpath = BASE / "P1-G0-V006/us_equity_research_master.parquet"
    master = pl.read_parquet(mpath, columns=["permno", "month", "gvkey", "q_datadate", "a_datadate"]).filter(
        pl.col("month").is_between(pl.date(2000, 1, 1), pl.date(2019, 12, 1)) & pl.col("gvkey").is_not_null()
    )
    panel = candidate_panel(master, quarter_candidates(qpath), annual_ms(apath))
    official = pl.read_parquet(GKX, columns=["permno", "month", *FEATURES]).rename({f: f"official_{f}" for f in FEATURES})
    panel = panel.join(official, on=["permno", "month"], how="inner").sort(["month", "permno"])
    panel.write_parquet(outputs[0], compression="zstd")
    mapping = {
        "chtx": ["chtx_lag4", "chtx_current"], "cinvest": ["cinvest_prev3", "cinvest_prev4"],
        "ms": ["ms_low", "ms_high"], "nincr": ["nincr_yoy", "nincr_qoq"],
        "roaq": ["roaq_lag1", "roaq_current"], "roeq": ["roeq_lag1", "roeq_current"],
        "rsup": ["rsup_current", "rsup_lag1"],
    }
    selection = []
    for feature, variants in mapping.items():
        for timing in ["hybrid", "fixed"]:
            for variant in variants:
                selection.append(metrics(panel, f"{timing}__{variant}", f"official_{feature}", "2000-01", "2009-12"))
    selected = {f: choose(selection, f) for f in FEATURES}
    confirmation = [metrics(panel, selected[f]["candidate"], f"official_{f}", "2010-01", "2019-12") for f in FEATURES]
    thresholds = {"spearman": .85, "rank_mae": .20, "nincr_exact": .75, "ms_exact": .65}
    for row in confirmation:
        f = row["feature"]
        if f in DISCRETE:
            row["passed"] = row["exact_agreement"] >= thresholds[f"{f}_exact"]
        else:
            row["passed"] = row["median_monthly_spearman"] >= thresholds["spearman"] and row["median_monthly_rank_mae"] <= thresholds["rank_mae"]
    pl.DataFrame(selection).write_csv(outputs[1])
    pl.DataFrame(confirmation).write_csv(outputs[2])
    result = {
        "schema_version": 1, "experiment_id": "P1-G0-V016", "status": "completed",
        "selection_period": ["2000-01", "2009-12"], "confirmation_period": ["2010-01", "2019-12"],
        "selected_candidates": {f: selected[f]["candidate"] for f in FEATURES},
        "selection_metrics": selected, "confirmation_metrics": confirmation,
        "passed_features": [r["feature"] for r in confirmation if r["passed"]],
        "failed_features": [r["feature"] for r in confirmation if not r["passed"]],
        "sealed_pricing_outputs_generated": False,
    }
    outputs[3].write_text(json.dumps(result, indent=2) + "\n")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "experiment_id": "P1-G0-V016",
        "git_revision": git_revision(), "command": " ".join(os.sys.argv),
        "inputs": [{"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in [qpath, apath, mpath, GKX]],
        "outputs": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha256(p)} for p in outputs[:-1]],
    }
    outputs[4].write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
