#!/usr/bin/env python3
"""Build and audit 44 direct annual accounting characteristics."""
from __future__ import annotations
import argparse,hashlib,json,os,subprocess
from datetime import datetime,timezone
from pathlib import Path
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"data/processed/wrds-us-equity-2025-12-v1"
ANNUAL=BASE/"P1-G0-V020/compustat_annual_extended_pti.parquet"
MASTER=BASE/"P1-G0-V006/us_equity_research_master.parquet"
GKX=ROOT/"data/processed/gkx-datashare-2021-v1/P1-G0-V007/gkx_core94_raw.parquet"
FEATURES="absacc acc age agr cashdebt chcsho chinv convind currat depr divi divo egr gma grcapx grltnoa hire invest lgr operprof pchcurrat pchdepr pchgm_pchsale pchquick pchsale_pchinvt pchsale_pchrect pchsale_pchxsga pchsaleinv pctacc ps quick rd rd_sale realestate roic salecash saleinv salerec secured securedind sgr sin tang tb".split()
DISCRETE={"convind","divi","divo","ps","rd","securedind","sin"}

def parse_args():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--overwrite",action="store_true");return p.parse_args()
def div(n,d): return pl.when(d.is_not_null()&(d.abs()>1e-12)).then(n/d).otherwise(None)
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

def annual_signals(path:Path)->pl.DataFrame:
 cols=['gvkey','datadate','fyear','sic','naics','at','act','che','rect','invt','ppegt','ppent','aco','intan','ao','ap','lco','lo','lt','lct','dlc','dltt','txp','dp','ib','ni','oancf','revt','sale','cogs','xsga','xint','xrd','xad','ceq','capx','emp','csho','dvt','fatb','fatl','dm','dc','dcvt','dcpstk','pstk','cshrc','drc','drlt','txfo','txfed','txt','txdi','scstkc','ebit','nopi']
 a=pl.read_parquet(path,columns=cols).with_columns(pl.col('fyear').cast(pl.Int64,strict=False),pl.col('sic').cast(pl.Int64,strict=False),pl.col('naics').cast(pl.Utf8)).sort(['gvkey','fyear','datadate'])
 lagcols=[c for c in cols if c not in {'gvkey','datadate','fyear','sic','naics'}]
 a=a.with_columns(*[pl.col(c).shift(n).over('gvkey').alias(f'l{n}_{c}') for c in lagcols for n in (1,2)])
 act=pl.coalesce(pl.col('act'),pl.col('che')+pl.col('rect')+pl.col('invt'))
 lct=pl.coalesce(pl.col('lct'),pl.col('ap'))
 capx=pl.when(pl.col('capx').is_null()&(pl.col('gvkey').cum_count().over('gvkey')>=2)).then(pl.col('ppent')-pl.col('l1_ppent')).otherwise(pl.col('capx'))
 dr=pl.when(pl.col('drc').is_not_null()&pl.col('drlt').is_not_null()).then(pl.col('drc')+pl.col('drlt')).when(pl.col('drc').is_not_null()).then(pl.col('drc')).when(pl.col('drlt').is_not_null()).then(pl.col('drlt')).otherwise(None)
 dc=pl.when(pl.col('dcvt').is_null()&pl.col('dcpstk').is_not_null()&pl.col('pstk').is_not_null()&(pl.col('dcpstk')>pl.col('pstk'))).then(pl.col('dcpstk')-pl.col('pstk')).when(pl.col('dcvt').is_null()&pl.col('dcpstk').is_not_null()&pl.col('pstk').is_null()).then(pl.col('dcpstk')).when(pl.col('dc').is_null()).then(pl.col('dcvt')).otherwise(None)
 tr=pl.when(pl.col('fyear')<=1978).then(.48).when(pl.col('fyear')<=1986).then(.46).when(pl.col('fyear')==1987).then(.40).when(pl.col('fyear')<=1992).then(.34).otherwise(.35)
 a=a.with_columns(act.alias('act2'),lct.alias('lct2'),capx.alias('capx2'),dr.alias('dr2'),dc.alias('dc2'),tr.alias('tr2'),pl.col('xint').fill_null(0).alias('xint0'),pl.col('xsga').fill_null(0).alias('xsga0'),div(pl.col('xrd'),pl.col('l1_at')).alias('xrd_h'))
 for c in ['act2','lct2','capx2','dr2','dc2','xrd_h']:
  a=a.with_columns(pl.col(c).shift(1).over('gvkey').alias('l1_'+c))
 a=a.with_columns(pl.col('capx2').shift(2).over('gvkey').alias('l2_capx2'))
 avgat=(pl.col('at')+pl.col('l1_at'))/2
 accrual_bs=(pl.col('act2')-pl.col('l1_act2'))-(pl.col('che')-pl.col('l1_che'))-((pl.col('lct2')-pl.col('l1_lct2'))-(pl.col('dlc')-pl.col('l1_dlc'))-(pl.col('txp')-pl.col('l1_txp'))-pl.col('dp'))
 acc=pl.when(pl.col('oancf').is_null()).then(div(accrual_bs,avgat)).otherwise(div(pl.col('ib')-pl.col('oancf'),avgat))
 sales_growth=div(pl.col('sale')-pl.col('l1_sale'),pl.col('l1_sale'))
 inv_growth=div(pl.col('invt')-pl.col('l1_invt'),pl.col('l1_invt'))
 rect_growth=div(pl.col('rect')-pl.col('l1_rect'),pl.col('l1_rect'))
 xsga_growth=div(pl.col('xsga')-pl.col('l1_xsga'),pl.col('l1_xsga'))
 gm_growth=div((pl.col('sale')-pl.col('cogs'))-(pl.col('l1_sale')-pl.col('l1_cogs')),pl.col('l1_sale')-pl.col('l1_cogs'))
 curr=div(pl.col('act2'),pl.col('lct2')); lcurr=div(pl.col('l1_act2'),pl.col('l1_lct2'))
 quick=div(pl.col('act2')-pl.col('invt'),pl.col('lct2')); lquick=div(pl.col('l1_act2')-pl.col('l1_invt'),pl.col('l1_lct2'))
 saleinv=div(pl.col('sale'),pl.col('invt')); lsaleinv=div(pl.col('l1_sale'),pl.col('l1_invt'))
 ppe_current=pl.when(pl.col('ppegt').is_null()).then(pl.col('ppent')).otherwise(pl.col('ppegt')); ppe_lag=pl.when(pl.col('l1_ppegt').is_null()).then(pl.col('l1_ppent')).otherwise(pl.col('l1_ppegt'))
 ona=pl.col('rect')+pl.col('invt')+pl.col('ppent')+pl.col('aco')+pl.col('intan')+pl.col('ao')-pl.col('ap')-pl.col('lco')-pl.col('lo')
 lona=pl.col('l1_rect')+pl.col('l1_invt')+pl.col('l1_ppent')+pl.col('l1_aco')+pl.col('l1_intan')+pl.col('l1_ao')-pl.col('l1_ap')-pl.col('l1_lco')-pl.col('l1_lo')
 working=(pl.col('rect')-pl.col('l1_rect'))+(pl.col('invt')-pl.col('l1_invt'))+(pl.col('aco')-pl.col('l1_aco'))-((pl.col('ap')-pl.col('l1_ap'))+(pl.col('lco')-pl.col('l1_lco')))-pl.col('dp')
 ps=[(pl.col('ni')>0),(pl.col('oancf')>0),div(pl.col('ni'),pl.col('at'))>div(pl.col('l1_ni'),pl.col('l1_at')),pl.col('oancf')>pl.col('ni'),div(pl.col('dltt'),pl.col('at'))>div(pl.col('l1_dltt'),pl.col('l1_at')),curr>lcurr,div(pl.col('sale')-pl.col('cogs'),pl.col('sale'))>div(pl.col('l1_sale')-pl.col('l1_cogs'),pl.col('l1_sale')),div(pl.col('sale'),pl.col('at'))>div(pl.col('l1_sale'),pl.col('l1_at')),pl.col('scstkc')==0]
 tb=pl.when(pl.col('txfo').is_null()|pl.col('txfed').is_null()).then(div(div(pl.col('txt')-pl.col('txdi'),pl.col('tr2')),pl.col('ib'))).when((pl.col('txfo')+pl.col('txfed')>0)|((pl.col('txt')>pl.col('txdi'))&(pl.col('ib')<=0))).then(1.0).otherwise(div(div(pl.col('txfo')-pl.col('txfed'),pl.col('tr2')),pl.col('ib')))
 a=a.with_columns(
  acc.abs().alias('absacc'),acc.alias('acc'),pl.col('gvkey').cum_count().over('gvkey').cast(pl.Float64).alias('age'),(div(pl.col('at'),pl.col('l1_at'))-1).alias('agr'),div(pl.col('ib')+pl.col('dp'),(pl.col('lt')+pl.col('l1_lt'))/2).alias('cashdebt'),(div(pl.col('csho'),pl.col('l1_csho'))-1).alias('chcsho'),div(pl.col('invt')-pl.col('l1_invt'),avgat).alias('chinv'),((pl.col('dc2').is_not_null()&(pl.col('dc2')!=0))|(pl.col('cshrc').is_not_null()&(pl.col('cshrc')!=0))).cast(pl.Int8).alias('convind'),curr.alias('currat'),div(pl.col('dp'),pl.col('ppent')).alias('depr'),((pl.col('dvt').is_not_null()&(pl.col('dvt')>0))&((pl.col('l1_dvt').is_null())|(pl.col('l1_dvt')==0))).cast(pl.Int8).alias('divi'),((pl.col('dvt').is_null()|(pl.col('dvt')==0))&(pl.col('l1_dvt').is_not_null())&(pl.col('l1_dvt')>0)).cast(pl.Int8).alias('divo'),div(pl.col('ceq')-pl.col('l1_ceq'),pl.col('l1_ceq')).alias('egr'),div(pl.col('revt')-pl.col('cogs'),pl.col('l1_at')).alias('gma'),div(pl.col('capx2')-pl.col('l2_capx2'),pl.col('l2_capx2')).alias('grcapx'),div((ona-lona)-working,avgat).alias('grltnoa'),pl.when(pl.col('emp').is_null()|pl.col('l1_emp').is_null()).then(0.0).otherwise(div(pl.col('emp')-pl.col('l1_emp'),pl.col('l1_emp'))).alias('hire'),div((ppe_current-ppe_lag)+(pl.col('invt')-pl.col('l1_invt')),pl.col('l1_at')).alias('invest'),(div(pl.col('lt'),pl.col('l1_lt'))-1).alias('lgr'),div(pl.col('revt')-pl.col('cogs')-pl.col('xsga0')-pl.col('xint0'),pl.col('l1_ceq')).alias('operprof'),div(curr-lcurr,lcurr).alias('pchcurrat'),div(div(pl.col('dp'),pl.col('ppent'))-div(pl.col('l1_dp'),pl.col('l1_ppent')),div(pl.col('l1_dp'),pl.col('l1_ppent'))).alias('pchdepr'),(gm_growth-sales_growth).alias('pchgm_pchsale'),div(quick-lquick,lquick).alias('pchquick'),(sales_growth-inv_growth).alias('pchsale_pchinvt'),(sales_growth-rect_growth).alias('pchsale_pchrect'),(sales_growth-xsga_growth).alias('pchsale_pchxsga'),div(saleinv-lsaleinv,lsaleinv).alias('pchsaleinv'),pl.when(pl.col('ib')==0).then((pl.col('ib')-pl.col('oancf'))/.01).when(pl.col('oancf').is_null()).then(div(accrual_bs,pl.col('ib').abs())).otherwise(div(pl.col('ib')-pl.col('oancf'),pl.col('ib').abs())).alias('pctacc'),pl.sum_horizontal(*[x.fill_null(False).cast(pl.Int8) for x in ps]).alias('ps'),quick.alias('quick'),(((div(pl.col('xrd'),pl.col('at'))-pl.col('l1_xrd_h'))/pl.col('l1_xrd_h'))>.05).fill_null(False).cast(pl.Int8).alias('rd'),div(pl.col('xrd'),pl.col('sale')).alias('rd_sale'),div(pl.col('fatb')+pl.col('fatl'),ppe_current).alias('realestate'),div(pl.col('ebit')-pl.col('nopi'),pl.col('ceq')+pl.col('lt')-pl.col('che')).alias('roic'),div(pl.col('sale'),pl.col('che')).alias('salecash'),saleinv.alias('saleinv'),div(pl.col('sale'),pl.col('rect')).alias('salerec'),div(pl.col('dm'),pl.col('dltt')).alias('secured'),(pl.col('dm').is_not_null()&(pl.col('dm')!=0)).cast(pl.Int8).alias('securedind'),(div(pl.col('sale'),pl.col('l1_sale'))-1).alias('sgr'),(((pl.col('sic').is_between(2100,2199))|pl.col('sic').is_between(2080,2085)|pl.col('naics').is_in(['7132','71312','713210','71329','713290','72112','721120']))).cast(pl.Int8).alias('sin'),div(pl.col('che')+.715*pl.col('rect')+.547*pl.col('invt')+.535*pl.col('ppent'),pl.col('at')).alias('tang'),tb.alias('tb'))
 return a.select('gvkey','datadate',*FEATURES)

def metrics(panel,f,start,end,sign=1):
 o='g_'+f;c='v_'+f
 x=panel.filter(pl.col('month').is_between(pl.date(start,1,1),pl.date(end,12,1))&pl.col(f).is_not_null()&pl.col(o).is_not_null()).select('month',(pl.col(f)*sign).alias(c),o)
 m=x.with_columns(rank(c,'sr'),rank(o,'gr')).group_by('month').agg(pl.len().alias('n'),pl.corr(c,o,method='spearman').alias('rho'),(pl.col('sr')-pl.col('gr')).abs().mean().alias('mae')).filter(pl.col('n')>=30)
 return {'feature':f,'sign':sign,'period_start':f'{start}-01','period_end':f'{end}-12','overlap_rows':x.height,'months':m.height,'median_monthly_spearman':float(m['rho'].median()),'median_monthly_rank_mae':float(m['mae'].median()),'exact_agreement':float((x[c]==x[o]).mean()) if f in DISCRETE else None}

def main():
 a=parse_args();out=a.output_dir.resolve();out.mkdir(parents=True,exist_ok=True);files=[out/n for n in ['annual_direct_statement.parquet','annual_direct_overlap.parquet','selection_metrics.csv','confirmation_metrics.csv','result_summary.json','output_manifest.json']]
 if any(p.exists() for p in files) and not a.overwrite:raise FileExistsError('Output exists; use --overwrite')
 for p in files:p.unlink(missing_ok=True)
 st=annual_signals(ANNUAL);st.write_parquet(files[0],compression='zstd')
 keys=pl.read_parquet(MASTER,columns=['permno','month','gvkey','a_datadate']).filter(pl.col('month').is_between(pl.date(2000,1,1),pl.date(2019,12,1)))
 selfp=keys.join(st,left_on=['gvkey','a_datadate'],right_on=['gvkey','datadate'],how='left').select('permno','month',*FEATURES)
 g=pl.read_parquet(GKX,columns=['permno','month',*FEATURES]).rename({f:'g_'+f for f in FEATURES});panel=selfp.join(g,on=['permno','month'],how='inner').sort(['month','permno']);panel.write_parquet(files[1],compression='zstd')
 selection=[];signs={}
 for f in FEATURES:
  raw=metrics(panel,f,2000,2009,1);selection.append(raw)
  if f in DISCRETE:signs[f]=1
  else:
   neg=metrics(panel,f,2000,2009,-1);selection.append(neg);signs[f]=1 if raw['median_monthly_spearman']>=neg['median_monthly_spearman'] else -1
 confirmation=[metrics(panel,f,2010,2019,signs[f]) for f in FEATURES]
 for r in confirmation:
  f=r['feature'];r['passed']=r['exact_agreement']>=.75 if f in DISCRETE else r['median_monthly_spearman']>=.85 and r['median_monthly_rank_mae']<=.20
 pl.DataFrame(selection).write_csv(files[2]);pl.DataFrame(confirmation).write_csv(files[3])
 result={'schema_version':1,'experiment_id':'P1-G0-V021','status':'completed','selected_signs':signs,'confirmation_metrics':confirmation,'passed_features':[r['feature'] for r in confirmation if r['passed']],'failed_features':[r['feature'] for r in confirmation if not r['passed']],'sealed_pricing_outputs_generated':False};files[4].write_text(json.dumps(result,indent=2)+'\n')
 man={'schema_version':1,'created_at':datetime.now(timezone.utc).isoformat(),'experiment_id':'P1-G0-V021','git_revision':gitrev(),'command':' '.join(os.sys.argv),'inputs':[{'path':str(p.resolve()),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in [ANNUAL,MASTER,GKX]],'outputs':[{'path':p.name,'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in files[:-1]]};files[5].write_text(json.dumps(man,indent=2)+'\n');print(json.dumps({'passed':result['passed_features'],'failed':result['failed_features'],'selected_signs':signs},indent=2))
if __name__=='__main__':main()
