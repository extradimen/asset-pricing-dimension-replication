#!/usr/bin/env python3
"""Adjudicate three annual signals demeaned in stock-month SIC2 cells."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess
from datetime import datetime, timezone
from pathlib import Path
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "data/processed/wrds-us-equity-2025-12-v1"
ANNUAL = BASE / "P1-G0-V020/compustat_annual_extended_pti.parquet"
MASTER = BASE / "P1-G0-V006/us_equity_research_master.parquet"
GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
FEATURES = ["bm_ia", "chpmia", "pchcapx_ia"]

def div(n, d): return pl.when(d.is_not_null() & (d.abs() > 1e-12)).then(n / d).otherwise(None)
def rank(c, a):
    n=pl.col(c).count().over("month"); r=pl.col(c).rank(method="average").over("month")
    return (2*(r-1)/(n-1)-1).alias(a)
def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()
def gitrev():
    try:return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except Exception:return None

def annual_raw(path: Path) -> pl.DataFrame:
    a=pl.read_parquet(path,columns=["gvkey","datadate","fyear","ceq","csho","prcc_f","ib","sale","capx","ppent"]).with_columns(pl.col("fyear").cast(pl.Int64,strict=False)).sort(["gvkey","fyear","datadate"])
    a=a.with_columns(*[pl.col(c).shift(1).over("gvkey").alias("l1_"+c) for c in ["ib","sale","capx","ppent"]])
    capx=pl.when(pl.col("capx").is_null()&(pl.col("gvkey").cum_count().over("gvkey")>=2)).then(pl.col("ppent")-pl.col("l1_ppent")).otherwise(pl.col("capx"))
    a=a.with_columns(capx.alias("capx2")).with_columns(pl.col("capx2").shift(1).over("gvkey").alias("l1_capx2"))
    mve=pl.col("csho").abs()*pl.col("prcc_f").abs()
    return a.with_columns(div(pl.col("ceq"),mve).alias("bm"),(div(pl.col("ib"),pl.col("sale"))-div(pl.col("l1_ib"),pl.col("l1_sale"))).alias("chpm"),div(pl.col("capx2")-pl.col("l1_capx2"),pl.col("l1_capx2")).alias("pchcapx")).select("gvkey","datadate","bm","chpm","pchcapx")

def monthly_signals(master_path: Path, raw: pl.DataFrame) -> pl.DataFrame:
    x=pl.read_parquet(master_path,columns=["permno","month","gvkey","a_datadate","siccd"]).filter(pl.col("month").is_between(pl.date(1980,1,1),pl.date(2019,12,1))).with_columns(pl.col("siccd").str.zfill(4).str.slice(0,2).cast(pl.Int16,strict=False).alias("sic2"))
    x=x.join(raw,left_on=["gvkey","a_datadate"],right_on=["gvkey","datadate"],how="left")
    w=["month","sic2"]
    return x.with_columns((pl.col("bm")-pl.col("bm").mean().over(w)).alias("bm_ia"),(pl.col("chpm")-pl.col("chpm").mean().over(w)).alias("chpmia"),(pl.col("pchcapx")-pl.col("pchcapx").mean().over(w)).alias("pchcapx_ia")).select("permno","month","sic2",*FEATURES)

def metrics(panel, feature, start, end):
    g="g_"+feature
    x=panel.filter(pl.col("month").is_between(pl.date(start,1,1),pl.date(end,12,1))&pl.col(feature).is_not_null()&pl.col(g).is_not_null()).select("month",feature,g)
    m=x.with_columns(rank(feature,"sr"),rank(g,"gr")).group_by("month").agg(pl.len().alias("n"),pl.corr(feature,g,method="spearman").alias("rho"),(pl.col("sr")-pl.col("gr")).abs().mean().alias("mae")).filter(pl.col("n")>=30)
    return {"feature":feature,"period_start":f"{start}-01","period_end":f"{end}-12","overlap_rows":x.height,"months":m.height,"median_monthly_spearman":float(m["rho"].median()),"median_monthly_rank_mae":float(m["mae"].median())}

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--output-dir",type=Path,required=True);ap.add_argument("--overwrite",action="store_true");args=ap.parse_args()
    out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    files=[out/n for n in ["monthly_industry_signals.parquet","overlap.parquet","development_metrics.csv","confirmation_metrics.csv","postlock_metrics.csv","result_summary.json","output_manifest.json"]]
    if any(p.exists() for p in files) and not args.overwrite:raise FileExistsError("Output exists; use --overwrite")
    for p in files:p.unlink(missing_ok=True)
    signals=monthly_signals(MASTER,annual_raw(ANNUAL));signals.write_parquet(files[0],compression="zstd")
    ref=pl.read_parquet(GKX,columns=["permno","month",*FEATURES]).rename({f:"g_"+f for f in FEATURES})
    panel=signals.join(ref,on=["permno","month"],how="inner").sort(["month","permno"]);panel.write_parquet(files[1],compression="zstd")
    development=[metrics(panel,f,1980,1989) for f in FEATURES]
    confirmation=[metrics(panel,f,1990,1999) for f in FEATURES]
    postlock=[metrics(panel,f,2000,2019) for f in FEATURES]
    for row in confirmation:row["passed"]=row["median_monthly_spearman"]>=.85 and row["median_monthly_rank_mae"]<=.20
    pl.DataFrame(development).write_csv(files[2]);pl.DataFrame(confirmation).write_csv(files[3]);pl.DataFrame(postlock).write_csv(files[4])
    result={"schema_version":1,"experiment_id":"P1-G0-V023","status":"completed","confirmation_metrics":confirmation,"postlock_metrics":postlock,"passed_features":[x["feature"] for x in confirmation if x["passed"]],"failed_features":[x["feature"] for x in confirmation if not x["passed"]],"sealed_pricing_outputs_generated":False};files[5].write_text(json.dumps(result,indent=2)+"\n")
    manifest={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"experiment_id":"P1-G0-V023","git_revision":gitrev(),"command":" ".join(os.sys.argv),"inputs":[{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in [ANNUAL,MASTER,GKX]],"outputs":[{"path":p.name,"size_bytes":p.stat().st_size,"sha256":sha(p)} for p in files[:-1]]};files[6].write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps({"confirmation":confirmation,"postlock":postlock},indent=2))
if __name__=="__main__":main()
