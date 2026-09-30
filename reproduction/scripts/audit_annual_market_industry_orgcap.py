#!/usr/bin/env python3
"""Build and audit the 17 annual market, industry, and orgcap characteristics."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/wrds-us-equity-2025-12-v1"
ANNUAL = BASE / "P1-G0-V020/compustat_annual_extended_pti.parquet"
MASTER = BASE / "P1-G0-V006/us_equity_research_master.parquet"
CCM = BASE / "P1-G0-V005/ccm_link_history.parquet"
GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
CPI_RAW = ROOT / "data/raw/public/bls-cpi-u-2026-09-24-v1"
FEATURES = "bm bm_ia cashpr cfp cfp_ia chatoia chempia chpmia dy ep herf lev mve_ia orgcap pchcapx_ia rd_mve sp".split()

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()

def div(n, d):
    return pl.when(d.is_not_null() & (d.abs() > 1e-12)).then(n / d).otherwise(None)

def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def gitrev():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None

def rank(column, alias):
    n = pl.col(column).count().over("month")
    r = pl.col(column).rank(method="average").over("month")
    return (2 * (r - 1) / (n - 1) - 1).alias(alias)

def load_cpi(raw_dir: Path) -> pl.DataFrame:
    rows = []
    for path in sorted(raw_dir.glob("bls_*.json")):
        payload = json.loads(path.read_text())
        if payload.get("status") != "REQUEST_SUCCEEDED":
            raise ValueError(f"BLS request failed in {path}")
        series = payload.get("Results", {}).get("series", [])
        if len(series) != 1 or series[0].get("seriesID") != "CUUR0000SA0":
            raise ValueError(f"Unexpected BLS series in {path}")
        rows.extend({"fyear": int(x["year"]), "cpi": float(x["value"])} for x in series[0]["data"] if x["period"] == "M13")
    cpi = pl.DataFrame(rows).unique("fyear", keep="none").sort("fyear")
    expected = list(range(1950, 2026))
    if cpi["fyear"].to_list() != expected:
        raise ValueError("CPI annual snapshot is incomplete or duplicated")
    return cpi

def add_orgcap(group: pl.DataFrame) -> pl.DataFrame:
    group = group.sort(["fyear", "datadate"])
    xsga, cpi = group["xsga"].to_list(), group["cpi"].to_list()
    stock = []
    prior = None
    for i, (expense, price_index) in enumerate(zip(xsga, cpi)):
        flow = None if expense is None or price_index is None or abs(price_index) <= 1e-12 else expense / price_index
        value = (None if flow is None else flow / .25) if i == 0 else (None if prior is None or flow is None else prior * .85 + flow)
        stock.append(value)
        prior = value
    return group.with_columns(pl.Series("orgcap_stock", stock, dtype=pl.Float64)).with_columns(div(pl.col("orgcap_stock"), pl.col("avgat")).alias("orgcap"))

def build_signals(annual_path: Path, ccm_path: Path, cpi: pl.DataFrame) -> pl.DataFrame:
    cols = ["gvkey", "datadate", "fyear", "sic", "at", "act", "che", "invt", "lct", "dlc", "txp", "dp", "ib", "oancf", "sale", "xsga", "xrd", "capx", "emp", "csho", "dvt", "ceq", "dltt", "lt", "prcc_f"]
    a = pl.read_parquet(annual_path, columns=cols).with_columns(pl.col("fyear").cast(pl.Int64, strict=False), pl.col("sic").cast(pl.Int64, strict=False)).sort(["gvkey", "fyear", "datadate"])
    for c in ["at", "act", "che", "lct", "dlc", "txp", "ib", "sale", "capx", "emp"]:
        a = a.with_columns(pl.col(c).shift(1).over("gvkey").alias("l1_" + c))
    a = a.with_columns(pl.col("at").shift(2).over("gvkey").alias("l2_at"))
    act = pl.coalesce(pl.col("act"), pl.col("che") + pl.col("invt"))
    lact = pl.coalesce(pl.col("l1_act"), pl.col("l1_che") + pl.col("invt").shift(1).over("gvkey"))
    lct = pl.col("lct")
    llct = pl.col("l1_lct")
    mve = pl.col("csho").abs() * pl.col("prcc_f").abs()
    accrual_bs = (act - lact) - (pl.col("che") - pl.col("l1_che")) - ((lct - llct) - (pl.col("dlc") - pl.col("l1_dlc")) - (pl.col("txp") - pl.col("l1_txp")) - pl.col("dp"))
    cfp = pl.when(pl.col("oancf").is_not_null()).then(div(pl.col("oancf"), mve)).otherwise(div(pl.col("ib") - accrual_bs, mve))
    avgat = (pl.col("at") + pl.col("l1_at")) / 2
    raw = a.with_columns(
        (pl.col("sic") // 100).cast(pl.Int16).alias("sic2"), mve.alias("mve_f"), avgat.alias("avgat"),
        div(pl.col("ceq"), mve).alias("bm"), div(pl.col("ib"), mve).alias("ep"),
        div(mve + pl.col("dltt") - pl.col("at"), pl.col("che")).alias("cashpr"), cfp.alias("cfp"),
        div(pl.col("dvt"), mve).alias("dy"), div(pl.col("lt"), mve).alias("lev"),
        div(pl.col("xrd"), mve).alias("rd_mve"), div(pl.col("sale"), mve).alias("sp"),
        (div(pl.col("ib"), pl.col("sale")) - div(pl.col("l1_ib"), pl.col("l1_sale"))).alias("chpm"),
        (div(pl.col("sale"), avgat) - div(pl.col("l1_sale"), (pl.col("l1_at") + pl.col("l2_at")) / 2)).alias("chato"),
        pl.when(pl.col("emp").is_null() | pl.col("l1_emp").is_null()).then(0.0).otherwise(div(pl.col("emp") - pl.col("l1_emp"), pl.col("l1_emp"))).alias("hire"),
        div(pl.col("capx") - pl.col("l1_capx"), pl.col("l1_capx")).alias("pchcapx")
    ).join(cpi, on="fyear", how="left")
    links = pl.read_parquet(ccm_path, columns=["gvkey", "sic", "LINKPRIM", "LINKTYPE", "LPERMNO", "LPERMCO", "LINKDT", "LINKENDDT"]).filter(
        pl.col("LINKPRIM").is_in(["P", "C"]) & pl.col("LINKTYPE").is_in(["LU", "LC", "LS"]) & pl.col("LPERMNO").is_not_null() & pl.col("LPERMCO").is_not_null()
    ).rename({"sic": "company_sic", "LPERMNO": "permno"})
    x = raw.join(links, on="gvkey", how="inner").filter(pl.col("datadate").is_between(pl.col("LINKDT"), pl.col("LINKENDDT"))).with_columns(
        pl.col("company_sic").str.slice(0, 2).cast(pl.Int16, strict=False).alias("sic2")
    ).sort(["permno", "fyear", "datadate"])
    x = x.unique(["permno", "fyear"], keep="first", maintain_order=True).group_by("permno", maintain_order=True).map_groups(add_orgcap)
    w = ["sic2", "fyear"]
    x = x.with_columns(
        (pl.col("bm") - pl.col("bm").mean().over(w)).alias("bm_ia"),
        (pl.col("cfp") - pl.col("cfp").mean().over(w)).alias("cfp_ia"),
        (pl.col("chato") - pl.col("chato").mean().over(w)).alias("chatoia"),
        (pl.col("hire") - pl.col("hire").mean().over(w)).alias("chempia"),
        (pl.col("chpm") - pl.col("chpm").mean().over(w)).alias("chpmia"),
        (pl.col("mve_f") - pl.col("mve_f").mean().over(w)).alias("mve_ia"),
        (pl.col("pchcapx") - pl.col("pchcapx").mean().over(w)).alias("pchcapx_ia"),
        pl.col("sale").sum().over(w).alias("indsale")
    ).with_columns((div(pl.col("sale"), pl.col("indsale")).pow(2).sum().over(w)).alias("herf"))
    return x.select("permno", "gvkey", "datadate", "fyear", "sic2", *FEATURES)

def metrics(panel, feature, start, end, sign=1):
    observed, reference = "v_" + feature, "g_" + feature
    x = panel.filter(pl.col("month").is_between(pl.date(start, 1, 1), pl.date(end, 12, 1)) & pl.col(feature).is_not_null() & pl.col(reference).is_not_null()).select("month", (pl.col(feature) * sign).alias(observed), reference)
    monthly = x.with_columns(rank(observed, "sr"), rank(reference, "gr")).group_by("month").agg(pl.len().alias("n"), pl.corr(observed, reference, method="spearman").alias("rho"), (pl.col("sr") - pl.col("gr")).abs().mean().alias("mae")).filter(pl.col("n") >= 30)
    return {"feature": feature, "sign": sign, "period_start": f"{start}-01", "period_end": f"{end}-12", "overlap_rows": x.height, "months": monthly.height, "median_monthly_spearman": float(monthly["rho"].median()), "median_monthly_rank_mae": float(monthly["mae"].median())}

def main():
    args = parse_args(); out = args.output_dir.resolve(); out.mkdir(parents=True, exist_ok=True)
    names = ["bls_cpi_u_annual.csv", "annual_complex_statement.parquet", "annual_complex_overlap.parquet", "selection_metrics.csv", "confirmation_metrics.csv", "result_summary.json", "output_manifest.json"]
    files = [out / n for n in names]
    if any(p.exists() for p in files) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    for path in files: path.unlink(missing_ok=True)
    cpi = load_cpi(CPI_RAW); cpi.write_csv(files[0])
    signals = build_signals(ANNUAL, CCM, cpi); signals.write_parquet(files[1], compression="zstd")
    keys = pl.read_parquet(MASTER, columns=["permno", "month", "a_datadate"]).filter(pl.col("month").is_between(pl.date(2000, 1, 1), pl.date(2019, 12, 1)))
    own = keys.join(signals, left_on=["permno", "a_datadate"], right_on=["permno", "datadate"], how="left").select("permno", "month", *FEATURES)
    ref = pl.read_parquet(GKX, columns=["permno", "month", *FEATURES]).rename({f: "g_" + f for f in FEATURES})
    panel = own.join(ref, on=["permno", "month"], how="inner").sort(["month", "permno"]); panel.write_parquet(files[2], compression="zstd")
    selection, signs = [], {}
    for feature in FEATURES:
        pos = metrics(panel, feature, 2000, 2009, 1); neg = metrics(panel, feature, 2000, 2009, -1)
        selection.extend([pos, neg]); signs[feature] = 1 if pos["median_monthly_spearman"] >= neg["median_monthly_spearman"] else -1
    confirmation = [metrics(panel, feature, 2010, 2019, signs[feature]) for feature in FEATURES]
    for row in confirmation:
        row["passed"] = row["median_monthly_spearman"] >= .85 and row["median_monthly_rank_mae"] <= .20
    pl.DataFrame(selection).write_csv(files[3]); pl.DataFrame(confirmation).write_csv(files[4])
    result = {"schema_version": 1, "experiment_id": "P1-G0-V022", "status": "completed", "selected_signs": signs, "confirmation_metrics": confirmation, "passed_features": [x["feature"] for x in confirmation if x["passed"]], "failed_features": [x["feature"] for x in confirmation if not x["passed"]], "sealed_pricing_outputs_generated": False}
    files[5].write_text(json.dumps(result, indent=2) + "\n")
    raw_inputs = sorted(CPI_RAW.glob("bls_*.json"))
    manifest = {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "experiment_id": "P1-G0-V022", "git_revision": gitrev(), "command": " ".join(os.sys.argv), "inputs": [{"path": str(p.resolve()), "size_bytes": p.stat().st_size, "sha256": sha(p)} for p in [ANNUAL, MASTER, CCM, GKX, *raw_inputs]], "outputs": [{"path": p.name, "size_bytes": p.stat().st_size, "sha256": sha(p)} for p in files[:-1]]}
    files[6].write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"passed": result["passed_features"], "failed": result["failed_features"], "selected_signs": signs}, indent=2))

if __name__ == "__main__":
    main()
