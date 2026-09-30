#!/usr/bin/env python3
"""Audit OpenSourceAP ChInvIA as the source-backed pchcapx_ia construction."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
import polars as pl
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"data/processed/wrds-us-equity-2025-12-v1"
ANNUAL=BASE/"P1-G0-V020/compustat_annual_extended_pti.parquet";MASTER=BASE/"P1-G0-V006/us_equity_research_master.parquet";GKX=ROOT/"data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
def div(n,d):return pl.when(d.is_not_null()&(d.abs()>1e-12)).then(n/d).otherwise(None)
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def gitrev():
 try:return subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
 except:return None
def rank(c,a):
 n=pl.col(c).count().over('month');r=pl.col(c).rank(method='average').over('month');return (2*(r-1)/(n-1)-1).alias(a)
def raw_signal(path):
 a=pl.read_parquet(path,columns=['gvkey','datadate','fyear','capx','ppent']).with_columns(pl.col('fyear').cast(pl.Int64,strict=False)).sort(['gvkey','fyear','datadate'])
 a=a.with_columns(pl.col('ppent').shift(1).over('gvkey').alias('l1_ppent'))
 a=a.with_columns(pl.coalesce(pl.col('capx'),pl.col('ppent')-pl.col('l1_ppent')).alias('capx2'))
 a=a.with_columns(pl.col('capx2').shift(1).over('gvkey').alias('l1_capx'),pl.col('capx2').shift(2).over('gvkey').alias('l2_capx'))
 avg=(pl.col('l1_capx')+pl.col('l2_capx'))/2
 primary=div(pl.col('capx2')-avg,avg);fallback=div(pl.col('capx2')-pl.col('l1_capx'),pl.col('l1_capx'))
 return a.with_columns(pl.coalesce(primary,fallback).alias('pchcapx')).select('gvkey','datadate','pchcapx')
def monthly_signal(master,raw):
 x=pl.read_parquet(master,columns=['permno','month','gvkey','a_datadate','siccd']).filter(pl.col('month').is_between(pl.date(1960,1,1),pl.date(2019,12,1))).with_columns(pl.col('siccd').str.zfill(4).str.slice(0,2).cast(pl.Int16,strict=False).alias('sic2')).join(raw,left_on=['gvkey','a_datadate'],right_on=['gvkey','datadate'],how='left')
 return x.with_columns((pl.col('pchcapx')-pl.col('pchcapx').mean().over(['month','sic2'])).alias('pchcapx_ia')).select('permno','month','sic2','pchcapx_ia')
def metrics(panel,start,end):
 x=panel.filter(pl.col('month').is_between(pl.date(start,1,1),pl.date(end,12,1))&pl.col('pchcapx_ia').is_not_null()&pl.col('g').is_not_null()).select('month','pchcapx_ia','g')
 m=x.with_columns(rank('pchcapx_ia','sr'),rank('g','gr')).group_by('month').agg(pl.len().alias('n'),pl.corr('pchcapx_ia','g',method='spearman').alias('rho'),(pl.col('sr')-pl.col('gr')).abs().mean().alias('mae')).filter(pl.col('n')>=30)
 return {'feature':'pchcapx_ia','period_start':f'{start}-01','period_end':f'{end}-12','overlap_rows':x.height,'months':m.height,'median_monthly_spearman':float(m['rho'].median()),'median_monthly_rank_mae':float(m['mae'].median())}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--overwrite',action='store_true');a=p.parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 files=[out/n for n in ['monthly_signal.parquet','overlap.parquet','development_metrics.csv','confirmation_metrics.csv','postlock_metrics.csv','result_summary.json','output_manifest.json']]
 if any(x.exists() for x in files) and not a.overwrite:raise FileExistsError('Output exists; use --overwrite')
 for x in files:x.unlink(missing_ok=True)
 sig=monthly_signal(MASTER,raw_signal(ANNUAL));sig.write_parquet(files[0],compression='zstd');g=pl.read_parquet(GKX,columns=['permno','month','pchcapx_ia']).rename({'pchcapx_ia':'g'});panel=sig.join(g,on=['permno','month'],how='inner').sort(['month','permno']);panel.write_parquet(files[1],compression='zstd')
 dev=metrics(panel,1960,1969);conf=metrics(panel,1970,1979);post=metrics(panel,1980,2019);conf['passed']=conf['median_monthly_spearman']>=.85 and conf['median_monthly_rank_mae']<=.20
 pl.DataFrame([dev]).write_csv(files[2]);pl.DataFrame([conf]).write_csv(files[3]);pl.DataFrame([post]).write_csv(files[4]);result={'schema_version':1,'experiment_id':'P1-G0-V024','status':'completed','confirmation_metrics':conf,'postlock_metrics':post,'passed_features':['pchcapx_ia'] if conf['passed'] else [],'failed_features':[] if conf['passed'] else ['pchcapx_ia'],'sealed_pricing_outputs_generated':False};files[5].write_text(json.dumps(result,indent=2)+'\n')
 man={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'experiment_id':'P1-G0-V024','git_revision':gitrev(),'command':' '.join(os.sys.argv),'inputs':[{'path':str(x.resolve()),'size_bytes':x.stat().st_size,'sha256':sha(x)} for x in [ANNUAL,MASTER,GKX]],'outputs':[{'path':x.name,'size_bytes':x.stat().st_size,'sha256':sha(x)} for x in files[:-1]]};files[6].write_text(json.dumps(man,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
