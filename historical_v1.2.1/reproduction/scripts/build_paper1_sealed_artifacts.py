#!/usr/bin/env python3
"""Build the frozen sealed-period paper tables and figures from P1-G2-V001."""
from __future__ import annotations
import argparse,csv,json,math
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KS=[1,2,3,4,5,8];SEEDS=[20260924,20260925,20260926,20260927,20260928]
LABEL={'size_bm_25':'25 Size–B/M','industry_49':'49 Industries','combined_74':'Combined 74','size_op_25':'25 Size–OP','size_inv_25':'25 Size–Inv','size_mom_25':'25 Size–Mom','size_accruals_25':'25 Size–Accruals','size_beta_25':'25 Size–Beta','size_resvar_25':'25 Size–Residual variance'}
COL={1:'#1B3A5D',2:'#0072B2',3:'#009E73',4:'#E69F00',5:'#D55E00',8:'#7A5195'}
def read_factors(path):
 with path.open() as f:
  rows=list(csv.DictReader(f));cols=[x for x in rows[0] if x.startswith('factor_')];return np.asarray([[float(r[x]) for x in cols] for r in rows])
def save(fig,stem,out):
 for x in ['pdf','svg','png']:fig.savefig(out/f'{stem}.{x}',dpi=300 if x=='png' else None,bbox_inches='tight',facecolor='white')
 plt.close(fig)
def main():
 p=argparse.ArgumentParser();p.add_argument('--result',type=Path,required=True);p.add_argument('--sealed-root',type=Path,required=True);p.add_argument('--development-result',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
 d=json.loads(a.result.read_text());dev=json.loads(a.development_result.read_text());out=a.output_dir;td=out/'tables';fd=out/'figures';td.mkdir(parents=True,exist_ok=True);fd.mkdir(parents=True,exist_ok=True)
 mpl.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'legend.frameon':False,'pdf.fonttype':42})
 families=[];dims=[];boots=[]
 for f,r in d['family_results'].items():
  b=d['bootstrap']['families'][f];families.append({'asset_family':f,'asset_label':LABEL[f],'best_K_gamma0':r['best_dimension']['0.0'],'best_K_gamma1':r['best_dimension']['1.0'],'A_raw':r['A_raw'],'A_low':b['A_raw']['ci95'][0],'A_high':b['A_raw']['ci95'][1],'attenuation':r['market_direction_attenuation'],'atten_low':b['attenuation']['ci95'][0],'atten_high':b['attenuation']['ci95'][1],'completion_raw':r['completion_minus_k1']['raw_moment'],'raw_low':b['completion_raw']['ci95'][0],'raw_high':b['completion_raw']['ci95'][1],'completion_alpha':r['completion_minus_k1']['alpha'],'alpha_low':b['completion_alpha']['ci95'][0],'alpha_high':b['completion_alpha']['ci95'][1]})
  for k,v in r['learned_dimension_seed_mean'].items():dims.append({'asset_family':f,'asset_label':LABEL[f],'K':int(k),'distance_gamma0':v['distance']['0.0'],'distance_gamma1':v['distance']['1.0'],'raw_moment':v['raw_moment'],'alpha':v['alpha']})
 fam=pd.DataFrame(families);dim=pd.DataFrame(dims);hyp=pd.DataFrame([{'hypothesis':k,**v} for k,v in d['preregistered_hypotheses'].items()])
 comp=[]
 for f in dev['family_results']:
  comp.append({'asset_family':f,'asset_label':LABEL[f],'development_A':dev['family_results'][f]['A_raw'],'sealed_A':d['family_results'][f]['A_raw'],'development_attenuation':dev['family_results'][f]['market_direction_attenuation'],'sealed_attenuation':d['family_results'][f]['market_direction_attenuation'],'development_completion_raw':dev['family_results'][f]['completion_minus_k1']['raw_moment'],'sealed_completion_raw':d['family_results'][f]['completion_minus_k1']['raw_moment'],'development_completion_alpha':dev['family_results'][f]['completion_minus_k1']['alpha'],'sealed_completion_alpha':d['family_results'][f]['completion_minus_k1']['alpha']})
 comp=pd.DataFrame(comp);perf=[];curves={}
 for k in KS:
  curves[k]=[]
  for seed in SEEDS:
   x=read_factors(a.sealed_root/f'factors-k{k}-seed{seed}'/'monthly_factor_returns.csv');w=np.cumprod(1+x,axis=0);portfolio=x.mean(1);wealth=np.cumprod(1+portfolio);dd=wealth/np.maximum.accumulate(wealth)-1
   perf.append({'K':k,'seed':seed,'factor_count':x.shape[1],'annualized_equal_factor_return':portfolio.mean()*12,'annualized_equal_factor_volatility':portfolio.std(ddof=1)*math.sqrt(12),'equal_factor_sharpe':portfolio.mean()/portfolio.std(ddof=1)*math.sqrt(12),'maximum_drawdown':dd.min()})
   curves[k].append(wealth)
 perf=pd.DataFrame(perf);perf_k=perf.groupby('K',as_index=False).agg({'annualized_equal_factor_return':['mean','std'],'annualized_equal_factor_volatility':['mean','std'],'equal_factor_sharpe':['mean','std'],'maximum_drawdown':['mean','std']});perf_k.columns=['_'.join(str(x) for x in c if x!='') for c in perf_k.columns]
 tables={'sealed_table_01_family_inference':fam,'sealed_table_02_dimension_metrics':dim,'sealed_table_03_hypothesis_gates':hyp,'sealed_table_04_development_comparison':comp,'sealed_table_05_factor_performance_by_seed':perf,'sealed_table_06_factor_performance_by_dimension':perf_k}
 for name,x in tables.items():x.to_csv(td/f'{name}.csv',index=False,float_format='%.10g');(td/f'{name}.tex').write_text(x.to_latex(index=False,longtable=len(x)>40,float_format=lambda z:f'{z:.3f}'))
 # Endpoint dumbbell.
 fig,ax=plt.subplots(figsize=(8,5));y=np.arange(len(fam));ax.hlines(y,fam.best_K_gamma0,fam.best_K_gamma1,color='#999');ax.scatter(fam.best_K_gamma0,y,color='#0072B2',label='γ=0');ax.scatter(fam.best_K_gamma1,y,color='#D55E00',marker='s',label='γ=1');ax.set_yticks(y,fam.asset_label);ax.set_xticks(KS);ax.invert_yaxis();ax.set_xlabel('Best K');ax.set_title('Sealed-period preferred dimension by pricing geometry',fontweight='bold');ax.legend();save(fig,'sealed_figure_01_endpoint_dimensions',fd)
 def forest(point,lo,hi,title,stem):
  fig,ax=plt.subplots(figsize=(8.2,5));y=np.arange(len(fam));v=fam[point].to_numpy();ax.hlines(y,fam[lo],fam[hi],color='#1B3A5D',lw=1.5);ax.scatter(v,y,color='#1B3A5D',s=26);ax.axvline(0,color='#333',ls='--',lw=.8);ax.set_yticks(y,fam.asset_label);ax.invert_yaxis();ax.set_title(title,fontweight='bold');ax.set_xlabel('Point estimate and 95% block-bootstrap interval');save(fig,stem,fd)
 forest('A_raw','A_low','A_high','Sealed geometry sensitivity A','sealed_figure_02_A_forest');forest('attenuation','atten_low','atten_high','Sealed market-direction attenuation','sealed_figure_03_attenuation_forest')
 fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True);y=np.arange(len(fam))
 for ax,pt,lo,hi,title in [(axes[0],'completion_raw','raw_low','raw_high','Raw moment'),(axes[1],'completion_alpha','alpha_low','alpha_high','Alpha')]:
  v=fam[pt].to_numpy();ax.hlines(y,fam[lo],fam[hi],color='#3C6E71',lw=1.5);ax.scatter(v,y,color='#3C6E71',s=26);ax.axvline(0,color='#333',ls='--',lw=.8);ax.set_title(title);ax.set_xlabel('K=1 + market minus K=1')
 axes[0].set_yticks(y,fam.asset_label);axes[0].invert_yaxis();fig.suptitle('Sealed-period market-factor completion',fontweight='bold');save(fig,'sealed_figure_04_completion_forest',fd)
 fig,axes=plt.subplots(1,2,figsize=(10,4))
 for ax,x,yv,title in [(axes[0],'development_A','sealed_A','Geometry A'),(axes[1],'development_completion_raw','sealed_completion_raw','Raw-moment completion')]:
  ax.scatter(comp[x],comp[yv],s=42,color='#7A5195');lo=min(comp[x].min(),comp[yv].min());hi=max(comp[x].max(),comp[yv].max());ax.plot([lo,hi],[lo,hi],color='#555',ls='--');ax.set_xlabel('2010–2019 development');ax.set_ylabel('2020–2025 sealed');ax.set_title(title)
 fig.suptitle('Development findings do not mechanically carry into the sealed period',fontweight='bold');save(fig,'sealed_figure_05_development_comparison',fd)
 fig,ax=plt.subplots(figsize=(9,4.2));
 for k in KS:
  arr=np.vstack(curves[k]);mean=arr.mean(0);se=arr.std(0,ddof=1)/np.sqrt(len(arr));ax.plot(np.arange(len(mean)),mean,color=COL[k],label=f'K={k}');ax.fill_between(np.arange(len(mean)),mean-2*se,mean+2*se,color=COL[k],alpha=.08)
 ax.set_xlabel('Sealed test month');ax.set_ylabel('Cumulative wealth of equal-factor portfolio');ax.set_title('Frozen-model factor performance in the sealed period',fontweight='bold');ax.legend(ncol=3);save(fig,'sealed_figure_06_factor_wealth',fd)
 summary={'schema_version':1,'experiment_id':d['experiment_id'],'sealed_period_accessed':d['sealed_period_accessed'],'tables':len(tables),'figures':6,'all_primary_hypotheses_supported':d['all_primary_hypotheses_supported']};(out/'result_summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
