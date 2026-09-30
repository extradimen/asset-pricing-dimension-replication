#!/usr/bin/env python3
"""Build the production Core-56 annual characteristic bridge through 2025."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import polars as pl

ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/"data/processed/wrds-us-equity-2025-12-v1"
MASTER=BASE/"P1-G0-V006/us_equity_research_master.parquet";DIRECT=BASE/"P1-G0-V021/annual_direct_statement.parquet";COMPLEX=BASE/"P1-G0-V022/annual_complex_statement.parquet";GKX=ROOT/"data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
DIRECT_FEATURES="absacc acc age agr cashdebt chcsho chinv convind currat depr divi divo egr gma grcapx grltnoa hire invest lgr operprof pchcurrat pchdepr pchgm_pchsale pchquick pchsale_pchinvt pchsale_pchrect pchsale_pchxsga pchsaleinv pctacc quick rd rd_sale realestate roic salecash saleinv salerec secured securedind sgr sin tang".split()
COMPLEX_FEATURES="bm cashpr cfp cfp_ia chatoia chempia dy ep herf lev mve_ia orgcap rd_mve sp".split()
FEATURES=DIRECT_FEATURES+COMPLEX_FEATURES
EXCLUDED={"ps","tb","bm_ia","chpmia","pchcapx_ia"};FORBIDDEN={"ret","ret_fwd1","pricing_error","model_prediction"}
def args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--overwrite',action='store_true');return p.parse_args()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def gitrev():
 try:return subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
 except:return None
def rank_expr(f):
 n=pl.col(f).count().over('month');r=pl.col(f).rank(method='average').over('month')
 return pl.when(pl.col(f).is_not_null()&(n>1)).then(2*(r-1)/(n-1)-1).otherwise(0.0).cast(pl.Float32).alias('x_'+f)
def main():
 a=args();started=time.time();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 files=[out/n for n in ['annual56_self_built.parquet','annual56_bridged_raw.parquet','annual56_model_input.parquet','quality_report.json','output_manifest.json']]
 if any(p.exists() for p in files) and not a.overwrite:raise FileExistsError('Output exists; use --overwrite')
 for p in files:p.unlink(missing_ok=True)
 admitted21=set(json.loads((ROOT/'experiments/P1-G0-V021/result_summary.json').read_text())['passed_features']);admitted22=set(json.loads((ROOT/'experiments/P1-G0-V022/result_summary.json').read_text())['passed_features'])
 if admitted21!=set(DIRECT_FEATURES) or admitted22!=set(COMPLEX_FEATURES):raise ValueError('Frozen feature lists disagree with admission records')
 keys=pl.read_parquet(MASTER,columns=['permno','month','gvkey','a_datadate'])
 direct=pl.read_parquet(DIRECT,columns=['gvkey','datadate',*DIRECT_FEATURES]).rename({'datadate':'a_datadate'})
 complex_=pl.read_parquet(COMPLEX,columns=['permno','datadate',*COMPLEX_FEATURES]).rename({'datadate':'a_datadate'})
 selfbuilt=keys.join(direct,on=['gvkey','a_datadate'],how='left').with_columns((-pl.col('agr')).alias('agr')).join(complex_,on=['permno','a_datadate'],how='left').select('permno','month',*FEATURES).sort(['month','permno'])
 selfbuilt.write_parquet(files[0],compression='zstd')
 official=pl.read_parquet(GKX,columns=['permno','month',*FEATURES]).rename({f:'gkx_'+f for f in FEATURES})
 joined=selfbuilt.join(official,on=['permno','month'],how='left')
 bridge=joined.with_columns(*[pl.coalesce(pl.col('gkx_'+f),pl.col(f)).alias(f) for f in FEATURES],pl.sum_horizontal(*[pl.col('gkx_'+f).is_not_null().cast(pl.UInt8) for f in FEATURES]).alias('official_feature_count')).with_columns(pl.when(pl.col('official_feature_count')==len(FEATURES)).then(pl.lit('official_full')).when(pl.col('official_feature_count')>0).then(pl.lit('official_partial')).otherwise(pl.lit('self_built')).alias('feature_source')).select('permno','month','feature_source','official_feature_count',*FEATURES)
 bridge.write_parquet(files[1],compression='zstd')
 model=bridge.select('permno','month','feature_source','official_feature_count',*[rank_expr(f) for f in FEATURES],*[pl.col(f).is_null().cast(pl.Int8).alias('missing_'+f) for f in FEATURES]);model.write_parquet(files[2],compression='zstd')
 report={'schema_version':1,'experiment_id':'P1-G0-V025','rows':bridge.height,'unique_keys':bridge.select(pl.struct('permno','month').n_unique()).item(),'first_month':str(bridge['month'].min()),'last_month':str(bridge['month'].max()),'feature_count':len(FEATURES),'features':FEATURES,'excluded_features':sorted(EXCLUDED),'self_built_nonmissing':selfbuilt.select(*[pl.col(f).is_not_null().sum().alias(f) for f in FEATURES]).row(0,named=True),'bridged_nonmissing':bridge.select(*[pl.col(f).is_not_null().sum().alias(f) for f in FEATURES]).row(0,named=True),'source_counts':bridge.group_by('feature_source').len().sort('feature_source').to_dicts(),'forbidden_output_columns_present':sorted(FORBIDDEN&set(bridge.columns+model.columns)),'sealed_pricing_outputs_generated':False,'elapsed_seconds':round(time.time()-started,3)};files[3].write_text(json.dumps(report,indent=2)+'\n')
 manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'experiment_id':'P1-G0-V025','git_revision':gitrev(),'command':' '.join(os.sys.argv),'inputs':[{'path':str(p.resolve()),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in [MASTER,DIRECT,COMPLEX,GKX]],'outputs':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in files[:-1]]};files[4].write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
