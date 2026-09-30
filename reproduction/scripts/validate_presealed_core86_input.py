#!/usr/bin/env python3
"""Validate a physically pre-2020 Core-86 input without reading sealed features."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import date
from pathlib import Path
import polars as pl
FORBIDDEN={'ret','ret_fwd1','pricing_error','model_prediction'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--input',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args();cut=date(2020,1,1)
 ss=pl.scan_parquet(a.source).collect_schema();os=pl.scan_parquet(a.input).collect_schema()
 # The full-period source is projected to month only; sealed feature values are never scanned.
 expected=pl.scan_parquet(a.source).select('month').filter(pl.col('month')<cut).select(pl.len()).collect().item()
 observed=pl.scan_parquet(a.input).select(pl.len().alias('rows'),pl.struct('permno','month').n_unique().alias('unique'),pl.col('month').min().alias('min'),pl.col('month').max().alias('max')).collect().row(0,named=True)
 m=json.loads(a.manifest.read_text());checks={'schema_identity':ss==os,'expected_rows':expected==observed['rows'],'unique_keys':observed['rows']==observed['unique'],'strict_boundary':observed['max']==date(2019,12,1) and observed['max']<cut,'forbidden_absent':not(FORBIDDEN&set(os.names())),'source_hash':m['source']['sha256']==sha(a.source),'output_hash':m['output']['sha256']==sha(a.input),'manifest_counts':m['output']['rows']==observed['rows'] and m['output']['maximum_month']==str(observed['max'])}
 r={'schema_version':1,'experiment_id':'P1-G0-V027','checks':checks,'rows':observed['rows'],'minimum_month':str(observed['min']),'maximum_month':str(observed['max']),'passed':all(checks.values()),'sealed_feature_values_read':False,'sealed_pricing_outputs_generated':False};a.report.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));raise SystemExit(0 if r['passed'] else 1)
if __name__=='__main__':main()
