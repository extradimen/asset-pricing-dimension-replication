#!/usr/bin/env python3
"""Build the production Core-10 quarterly bridge through 2025."""
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
BASE = ROOT / "data/processed/wrds-us-equity-2025-12-v1"
GKX = ROOT / "data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
V015 = BASE / "P1-G0-V015/quarterly11_self_built.parquet"
FEATURES = ["cash", "chtx", "cinvest", "nincr", "roaq", "roavol", "roeq", "rsup", "stdacc", "stdcf"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__); p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--overwrite", action="store_true"); return p.parse_args()


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def git_rev() -> str | None:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): return None


def quarterly_nincr(path: Path) -> pl.DataFrame:
    q=pl.read_parquet(path,columns=["gvkey","datadate","fyearq","fqtr","ibq"]).with_columns(
        (pl.col("fyearq").cast(pl.Int64,strict=False)*4+pl.col("fqtr").cast(pl.Int64,strict=False)).alias("qid")
    ).sort(["gvkey","qid","datadate"])
    for n in range(1,9):
        q=q.with_columns(
            pl.when(pl.col("qid")-pl.col("qid").shift(n).over("gvkey")==n).then(pl.col("ibq").shift(n).over("gvkey")).otherwise(None).alias(f"l{n}")
        )
    prod=pl.lit(1,dtype=pl.Int16); streak=pl.lit(0,dtype=pl.Int16)
    for n in range(8):
        current=pl.col("ibq") if n==0 else pl.col(f"l{n}")
        indicator=(current>pl.col(f"l{n+1}")).fill_null(False).cast(pl.Int16)
        prod=prod*indicator; streak=streak+prod
    return q.with_columns(streak.alias("nincr")).select("gvkey","datadate","nincr")


def rank_expr(f: str) -> pl.Expr:
    n=pl.col(f).count().over("month"); r=pl.col(f).rank(method="average").over("month")
    return pl.when(pl.col(f).is_not_null()&(n>1)).then(2*(r-1)/(n-1)-1).otherwise(0.0).cast(pl.Float32).alias(f"x_{f}")


def main() -> None:
    a=parse_args(); started=time.time(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    outputs=[out/n for n in ["quarterly10_self_built.parquet","quarterly10_bridged_raw.parquet","quarterly10_model_input.parquet","quality_report.json","output_manifest.json"]]
    if any(p.exists() for p in outputs) and not a.overwrite: raise FileExistsError("Output exists; use --overwrite")
    for p in outputs: p.unlink(missing_ok=True)
    qpath=BASE/"P1-G0-V005/compustat_quarterly_pti.parquet"; mpath=BASE/"P1-G0-V006/us_equity_research_master.parquet"
    keys=pl.read_parquet(mpath,columns=["permno","month","gvkey","q_datadate"])
    nine=[f for f in FEATURES if f!="nincr"]
    source=pl.read_parquet(V015,columns=["permno","month",*nine])
    nq=keys.join(quarterly_nincr(qpath),left_on=["gvkey","q_datadate"],right_on=["gvkey","datadate"],how="left").select("permno","month","nincr")
    source=source.join(nq,on=["permno","month"],how="left").select("permno",pl.col("month").dt.offset_by("6mo").alias("month"),*FEATURES)
    target=keys.select("permno","month").unique().sort(["month","permno"])
    self_built=target.join(source,on=["permno","month"],how="left").sort(["month","permno"])
    self_built.write_parquet(outputs[0],compression="zstd")
    official=pl.read_parquet(GKX,columns=["permno","month",*FEATURES]).rename({f:f"gkx_{f}" for f in FEATURES})
    joined=self_built.join(official,on=["permno","month"],how="left")
    bridge=joined.with_columns(
        *[pl.coalesce(pl.col(f"gkx_{f}"),pl.col(f)).alias(f) for f in FEATURES],
        pl.sum_horizontal(*[pl.col(f"gkx_{f}").is_not_null().cast(pl.UInt8) for f in FEATURES]).alias("official_feature_count"),
    ).with_columns(
        pl.when(pl.col("official_feature_count")==len(FEATURES)).then(pl.lit("official_full"))
        .when(pl.col("official_feature_count")>0).then(pl.lit("official_partial")).otherwise(pl.lit("self_built")).alias("feature_source")
    ).select("permno","month","feature_source","official_feature_count",*FEATURES)
    bridge.write_parquet(outputs[1],compression="zstd")
    model=bridge.select("permno","month","feature_source","official_feature_count",*[rank_expr(f) for f in FEATURES],*[pl.col(f).is_null().cast(pl.Int8).alias(f"missing_{f}") for f in FEATURES])
    model.write_parquet(outputs[2],compression="zstd")
    report={
        "schema_version":1,"experiment_id":"P1-G0-V019","rows":bridge.height,"unique_keys":bridge.select(pl.struct("permno","month").n_unique()).item(),
        "first_month":str(bridge["month"].min()),"last_month":str(bridge["month"].max()),"features":FEATURES,"excluded_features":["ms"],
        "self_built_nonmissing":self_built.select(*[pl.col(f).is_not_null().sum().alias(f) for f in FEATURES]).row(0,named=True),
        "bridged_nonmissing":bridge.select(*[pl.col(f).is_not_null().sum().alias(f) for f in FEATURES]).row(0,named=True),
        "forbidden_output_columns_present":sorted(set(["ret","ret_fwd1","pricing_error","model_prediction"])&set(bridge.columns+model.columns)),
        "sealed_pricing_outputs_generated":False,"elapsed_seconds":round(time.time()-started,3),
    }
    outputs[3].write_text(json.dumps(report,indent=2)+"\n")
    manifest={
        "schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"experiment_id":"P1-G0-V019","git_revision":git_rev(),"command":" ".join(os.sys.argv),
        "inputs":[{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in [qpath,mpath,V015,GKX]],
        "outputs":[{"path":p.name,"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in outputs[:-1]],
    }
    outputs[4].write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(report,indent=2))


if __name__=="__main__": main()
