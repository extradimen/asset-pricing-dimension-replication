#!/usr/bin/env python3
"""Rebuild the canonical annual Compustat table with specialized signal fields."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_compustat_ccm as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = Path("data/raw/licensed/【批量下载】Fundamentals Quarterly等.zip")
EXTRA_NUMERIC = [
    "ajex", "dc", "dcpstk", "dcvt", "dm", "drc", "drlt", "dvt", "fatb", "fatl",
    "ob", "gdwlia", "gdwlip", "gwo", "ivao", "np", "dpc", "txdc", "txdi",
    "txfed", "txfo", "scstkc",
]


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser(description=__doc__); p.add_argument("--input",type=Path,default=DEFAULT_INPUT); p.add_argument("--output-dir",type=Path,required=True); p.add_argument("--block-size-mb",type=int,default=64); p.add_argument("--overwrite",action="store_true"); return p.parse_args()


def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(8*1024*1024),b""): h.update(b)
    return h.hexdigest()


def git_rev() -> str | None:
    try: return subprocess.check_output(["git","rev-parse","HEAD"],text=True,stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError): return None


def main() -> None:
    a=parse_args(); started=time.time(); out=a.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    table=out/"compustat_annual_extended_pti.parquet"; report=out/"quality_report.json"; manifest=out/"output_manifest.json"
    if any(p.exists() for p in [table,report,manifest]) and not a.overwrite: raise FileExistsError("Output exists; use --overwrite")
    for p in [table,report,manifest]: p.unlink(missing_ok=True)
    nested=base.nested_zip_to_temp(a.input,"Annual")
    try:
        base.ANNUAL_NUMERIC=list(dict.fromkeys(base.ANNUAL_NUMERIC+EXTRA_NUMERIC))
        quality=base.build_statement(nested,"annual",table,a.block_size_mb*1024*1024)
    finally:
        nested.unlink(missing_ok=True)
    missing_extra=sorted(set(EXTRA_NUMERIC)-set(quality["present_columns"]))
    result={"schema_version":1,"experiment_id":"P1-G0-V020","extra_numeric_fields":EXTRA_NUMERIC,"missing_extra_fields":missing_extra,"annual":quality,"sealed_pricing_outputs_generated":False,"elapsed_seconds":round(time.time()-started,3)}
    report.write_text(json.dumps(result,indent=2)+"\n")
    man={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"experiment_id":"P1-G0-V020","git_revision":git_rev(),"command":" ".join(os.sys.argv),"inputs":[{"path":str(a.input.resolve()),"size_bytes":a.input.stat().st_size,"sha256":sha256(a.input)}],"outputs":[{"path":p.name,"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in [table,report]]}
    manifest.write_text(json.dumps(man,indent=2)+"\n"); print(json.dumps({"rows":quality["canonical_rows"],"columns":len(quality["present_columns"])+2,"missing_extra_fields":missing_extra,"elapsed_seconds":result["elapsed_seconds"]},indent=2))


if __name__=="__main__": main()
