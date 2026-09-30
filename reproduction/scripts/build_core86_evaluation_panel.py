#!/usr/bin/env python3
"""Join outcome-free Core-86 features to next-month returns for a frozen interval."""
from __future__ import annotations
import argparse,hashlib,json
from datetime import date,datetime,timezone
from pathlib import Path
import polars as pl

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--features',type=Path,required=True);p.add_argument('--master',type=Path,required=True)
 p.add_argument('--feature-start',required=True);p.add_argument('--feature-end',required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--sealed-protocol',type=Path)
 a=p.parse_args();start=date.fromisoformat(a.feature_start+'-01');end=date.fromisoformat(a.feature_end+'-01');sealed=end>=date(2019,12,1)
 if sealed:
  if not a.sealed_protocol:raise RuntimeError('Sealed interval requires protocol')
  protocol=json.loads(a.sealed_protocol.read_text())
  if not protocol.get('protocol_frozen') or [a.feature_start,a.feature_end]!=protocol['sealed_feature_months']:raise RuntimeError('Protocol mismatch')
 out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);panel=out/'core86_evaluation_input.parquet';report=out/'quality_report.json';manifest=out/'output_manifest.json'
 if any(x.exists() for x in [panel,report,manifest]):raise FileExistsError('Output exists')
 f=pl.scan_parquet(a.features).filter(pl.col('month').is_between(start,end));schema=f.collect_schema().names()
 t=pl.scan_parquet(a.master).filter(pl.col('month').is_between(start,end)).select('permno','month','ret_fwd1','market_cap',pl.lit(True).alias('_target_present'))
 joined=f.join(t,on=['permno','month'],how='left',validate='1:1');audit=joined.select(pl.len().alias('rows'),pl.col('_target_present').is_null().sum().alias('missing_target_keys')).collect().row(0,named=True)
 if audit['missing_target_keys']:raise RuntimeError('Feature keys missing from target source')
 joined.drop('_target_present').with_columns(pl.col('month').dt.offset_by('1mo').alias('target_month')).select('permno','month','target_month','market_cap',*[x for x in schema if x not in {'permno','month'}],'ret_fwd1').sink_parquet(panel,compression='zstd',mkdir=True)
 observed=pl.scan_parquet(panel).select(pl.len().alias('rows'),pl.struct('permno','month').n_unique().alias('unique_keys'),pl.col('month').min().alias('first_feature_month'),pl.col('month').max().alias('last_feature_month'),pl.col('target_month').min().alias('first_target_month'),pl.col('target_month').max().alias('last_target_month'),pl.col('ret_fwd1').is_not_null().sum().alias('non_null_targets')).collect().row(0,named=True)
 result={'schema_version':1,'sealed_period_accessed':sealed,**{k:(str(v) if 'month' in k else v) for k,v in observed.items()},'missing_target_keys':audit['missing_target_keys'],'target_distribution_summarized':False}
 report.write_text(json.dumps(result,indent=2)+'\n');items=[('features',a.features),('master',a.master)]
 m={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'sealed_period_accessed':sealed,'inputs':[{'role':r,'path':str(x.resolve()),'size_bytes':x.stat().st_size,'sha256':sha(x)} for r,x in items],'outputs':[{'path':x.name,'size_bytes':x.stat().st_size,'sha256':sha(x)} for x in [panel,report]]}
 manifest.write_text(json.dumps(m,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
