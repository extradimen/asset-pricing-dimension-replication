#!/usr/bin/env python3
"""Assemble the outcome-free Structured Core-86 panel."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess,time
from datetime import datetime,timezone
from pathlib import Path
import polars as pl
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'data/processed/wrds-us-equity-2025-12-v1'
MRAW=BASE/'P1-G0-V014/core20_self_built.parquet';MBRIDGE=BASE/'P1-G0-V014/core20_bridged_raw.parquet';QRAW=BASE/'P1-G0-V019/quarterly10_self_built.parquet';QBRIDGE=BASE/'P1-G0-V019/quarterly10_bridged_raw.parquet';ARAW=BASE/'P1-G0-V025/annual56_self_built.parquet';ABRIDGE=BASE/'P1-G0-V025/annual56_bridged_raw.parquet'
MONTHLY="baspread beta betasq chmom dolvol idiovol ill indmom maxret mom12m mom1m mom36m mom6m mvel1 pricedelay retvol std_dolvol std_turn turn zerotrade".split()
QUARTERLY="cash chtx cinvest nincr roaq roavol roeq rsup stdacc stdcf".split()
ANNUAL="absacc acc age agr cashdebt chcsho chinv convind currat depr divi divo egr gma grcapx grltnoa hire invest lgr operprof pchcurrat pchdepr pchgm_pchsale pchquick pchsale_pchinvt pchsale_pchrect pchsale_pchxsga pchsaleinv pctacc quick rd rd_sale realestate roic salecash saleinv salerec secured securedind sgr sin tang bm cashpr cfp cfp_ia chatoia chempia dy ep herf lev mve_ia orgcap rd_mve sp".split();FEATURES=MONTHLY+QUARTERLY+ANNUAL
EXCLUDED={'aeavol','ear','ms','ps','tb','bm_ia','chpmia','pchcapx_ia'};FORBIDDEN={'ret','ret_fwd1','pricing_error','model_prediction'}
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024*1024),b''):h.update(x)
 return h.hexdigest()
def gitrev():
 try:return subprocess.check_output(['git','rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
 except:return None
def rank(f):
 n=pl.col(f).count().over('month');r=pl.col(f).rank(method='average').over('month');return pl.when(pl.col(f).is_not_null()&(n>1)).then(2*(r-1)/(n-1)-1).otherwise(0.0).cast(pl.Float32).alias('x_'+f)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--overwrite',action='store_true');a=p.parse_args();started=time.time();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
 files=[out/n for n in ['core86_self_built.parquet','core86_bridged_raw.parquet','core86_model_input.parquet','quality_report.json','output_manifest.json']]
 if any(x.exists() for x in files) and not a.overwrite:raise FileExistsError('Output exists; use --overwrite')
 for x in files:x.unlink(missing_ok=True)
 # Explicit projections are a data-firewall: V014's legacy ret_fwd1 column is never read.
 mr=pl.read_parquet(MRAW,columns=['permno','month',*MONTHLY]);qr=pl.read_parquet(QRAW,columns=['permno','month',*QUARTERLY]);ar=pl.read_parquet(ARAW,columns=['permno','month',*ANNUAL])
 selfbuilt=mr.join(qr,on=['permno','month'],how='inner',validate='1:1').join(ar,on=['permno','month'],how='inner',validate='1:1').sort(['month','permno']);selfbuilt.write_parquet(files[0],compression='zstd')
 mb=pl.read_parquet(MBRIDGE,columns=['permno','month','feature_source','official_feature_count',*MONTHLY]).rename({'feature_source':'monthly_source','official_feature_count':'monthly_official_count'});qb=pl.read_parquet(QBRIDGE,columns=['permno','month','feature_source','official_feature_count',*QUARTERLY]).rename({'feature_source':'quarterly_source','official_feature_count':'quarterly_official_count'});ab=pl.read_parquet(ABRIDGE,columns=['permno','month','feature_source','official_feature_count',*ANNUAL]).rename({'feature_source':'annual_source','official_feature_count':'annual_official_count'})
 bridge=mb.join(qb,on=['permno','month'],how='inner',validate='1:1').join(ab,on=['permno','month'],how='inner',validate='1:1').sort(['month','permno']);bridge.write_parquet(files[1],compression='zstd')
 meta=['monthly_source','monthly_official_count','quarterly_source','quarterly_official_count','annual_source','annual_official_count'];model=bridge.select('permno','month',*meta,*[rank(f) for f in FEATURES],*[pl.col(f).is_null().cast(pl.Int8).alias('missing_'+f) for f in FEATURES]);model.write_parquet(files[2],compression='zstd')
 report={'schema_version':1,'experiment_id':'P1-G0-V026','rows':bridge.height,'unique_keys':bridge.select(pl.struct('permno','month').n_unique()).item(),'first_month':str(bridge['month'].min()),'last_month':str(bridge['month'].max()),'feature_count':len(FEATURES),'feature_groups':{'monthly_and_market':MONTHLY,'quarterly':QUARTERLY,'annual':ANNUAL},'excluded_features':sorted(EXCLUDED),'forbidden_output_columns_present':sorted(FORBIDDEN&set(selfbuilt.columns+bridge.columns+model.columns)),'legacy_v014_target_read':False,'sealed_pricing_outputs_generated':False,'elapsed_seconds':round(time.time()-started,3)};files[3].write_text(json.dumps(report,indent=2)+'\n')
 inputs=[MRAW,MBRIDGE,QRAW,QBRIDGE,ARAW,ABRIDGE];manifest={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'experiment_id':'P1-G0-V026','git_revision':gitrev(),'command':' '.join(os.sys.argv),'input_projection_policy':'Only explicitly listed feature and provenance columns were read; V014 ret_fwd1 was excluded at scan time.','inputs':[{'path':str(x.resolve()),'size_bytes':x.stat().st_size,'sha256':sha(x)} for x in inputs],'outputs':[{'path':x.name,'size_bytes':x.stat().st_size,'sha256':sha(x)} for x in files[:-1]]};files[4].write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
