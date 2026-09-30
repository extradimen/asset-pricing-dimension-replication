#!/usr/bin/env python3
"""Run the frozen paper-one confirmation statistics on the one-time sealed factor returns."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np
import audit_external_asset_geometry as ext
import audit_hj_alpha_estimands as estimands
import evaluate_multi_factor_teacher as multi
import evaluate_teacher_pricing as base

KS=ext.KS;SEEDS=ext.SEEDS
PRIMARY={'size_bm_25':('25_Portfolios_5x5_CSV.zip',25,'size_bm::'),'industry_49':('49_Industry_Portfolios_CSV.zip',49,'industry::')}

def months(a,b):return [str(x) for x in np.arange(np.datetime64(a),np.datetime64(b)+1,dtype='datetime64[M]')]
def ci(rows,key):
 v=np.asarray([r[key] for r in rows]);return {'ci95':np.quantile(v,[.025,.975]).tolist(),'nonpositive_fraction':float(np.mean(v<=0))}
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--sealed-root',type=Path,required=True);p.add_argument('--validation-root',type=Path,required=True);p.add_argument('--primary-dir',type=Path,required=True);p.add_argument('--external-dir',type=Path,required=True);p.add_argument('--ff5',type=Path,required=True);p.add_argument('--validation-start',default='1999-12');p.add_argument('--validation-end',default='2009-11');p.add_argument('--test-start',default='2019-12');p.add_argument('--test-end',default='2025-11');p.add_argument('--expected-test-months',type=int,default=72);p.add_argument('--bootstrap-draws',type=int,default=500);p.add_argument('--bootstrap-seed',type=int,default=20260924);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 vm,tm=months(a.validation_start,a.validation_end),months(a.test_start,a.test_end)
 if len(vm)!=120 or len(tm)!=a.expected_test_months:raise RuntimeError('Evaluation month count differs from requested protocol')
 ff=base.read_factor_file(a.ff5,['Mkt-RF','SMB','HML','RMW','CMA','RF']);rf={m:x[-1] for m,x in ff.items()};market={'validation':np.asarray([ff[base.next_month(m)][0] for m in vm]),'test':np.asarray([ff[base.next_month(m)][0] for m in tm])}
 raw={}
 for label,(name,n,prefix) in PRIMARY.items():raw[label]=base.read_first_value_weighted_monthly(a.primary_dir/name,n,prefix)[1]
 for label,name in ext.FAMILIES.items():raw[label]=base.read_first_value_weighted_monthly(a.external_dir/name,25,f'{label}::')[1]
 returns={f:{'validation':base.asset_matrix(vm,r,rf),'test':base.asset_matrix(tm,r,rf)} for f,r in raw.items()}
 returns['combined_74']={s:np.column_stack([returns['size_bm_25'][s],returns['industry_49'][s]]) for s in ['validation','test']}
 factor_maps={k:{seed:{**multi.read_factor_matrix(a.validation_root/f'factors-k{k}-seed{seed}'/'monthly_factor_returns.csv'),**multi.read_factor_matrix(a.sealed_root/f'factors-k{k}-seed{seed}'/'monthly_factor_returns.csv')} for seed in SEEDS} for k in KS}
 factors={period:{k:{seed:multi.align_matrix(ms,factor_maps[k][seed]) for seed in SEEDS} for k in KS} for period,ms in [('validation',vm),('test',tm)]}
 loadings={k:{seed:estimands.self_pricing_loadings(factors['validation'][k][seed]) for seed in SEEDS} for k in KS}
 completion={seed:estimands.self_pricing_loadings(np.column_stack([factors['validation'][1][seed],market['validation']])) for seed in SEEDS}
 design=np.column_stack([np.ones(len(vm)),market['validation']]);betas={f:(np.linalg.pinv(design)@r['validation'])[1] for f,r in returns.items()}
 hedged={f:r['test']-market['test'][:,None]*betas[f][None,:] for f,r in returns.items()}
 family_results={f:ext.evaluate_period(r['test'],hedged[f],factors['test'],market['test'],loadings,completion) for f,r in returns.items()}
 rng=np.random.default_rng(a.bootstrap_seed);draws={f:[] for f in returns}
 for _ in range(a.bootstrap_draws):
  ix=ext.circular_indices(len(tm),12,rng)
  for f,r in returns.items():
   row=ext.evaluate_period(r['test'][ix],hedged[f][ix],{k:{s:factors['test'][k][s][ix] for s in SEEDS} for k in KS},market['test'][ix],loadings,completion)
   draws[f].append({'A_raw':row['A_raw'],'attenuation':row['market_direction_attenuation'],'completion_raw':row['completion_minus_k1']['raw_moment'],'completion_alpha':row['completion_minus_k1']['alpha']})
 bootstrap={f:{k:ci(rows,k) for k in rows[0]} for f,rows in draws.items()}
 external=list(ext.FAMILIES);changed=sum(family_results[f]['best_dimension']['0.0']!=family_results[f]['best_dimension']['1.0'] for f in external);atten=sum(family_results[f]['market_direction_attenuation']>0 for f in external);complete=sum(family_results[f]['completion_minus_k1']['raw_moment']<0 and family_results[f]['completion_minus_k1']['alpha']<0 for f in external);unique=sorted({v for f in external for v in family_results[f]['best_dimension'].values()})
 combined=family_results['combined_74'];hyp={
  'H1_geometry_dependence':{'combined_endpoint_dimensions_differ':combined['best_dimension']['0.0']!=combined['best_dimension']['1.0'],'external_changed_count':changed,'required_external':4},
  'H2_market_direction':{'combined_positive_attenuation':combined['market_direction_attenuation']>0,'external_positive_count':atten,'required_external':4},
  'H3_economic_completion':{'external_joint_improvement_count':complete,'required_external':4},
  'H4_asset_family_dependence':{'unique_external_best_dimensions':unique,'required_count':3}}
 hyp['H1_geometry_dependence']['supported']=bool(hyp['H1_geometry_dependence']['combined_endpoint_dimensions_differ'] and changed>=4);hyp['H2_market_direction']['supported']=bool(hyp['H2_market_direction']['combined_positive_attenuation'] and atten>=4);hyp['H3_economic_completion']['supported']=bool(complete>=4);hyp['H4_asset_family_dependence']['supported']=bool(len(unique)>=3)
 sealed=bool(a.test_start=='2019-12' and a.test_end=='2025-11' and len(tm)==72)
 result={'schema_version':1,'experiment_id':'P1-G2-V001' if sealed else 'P1-G1-V024','sealed_period_accessed':sealed,'periods':{'validation_feature_months':[vm[0],vm[-1],len(vm)],'test_feature_months':[tm[0],tm[-1],len(tm)],'test_target_months':[str(np.datetime64(tm[0])+1),str(np.datetime64(tm[-1])+1),len(tm)]},'models_retrained':False,'family_results':family_results,'bootstrap':{'draws':a.bootstrap_draws,'block_months':12,'families':bootstrap},'preregistered_hypotheses':hyp,'all_primary_hypotheses_supported':bool(all(x['supported'] for x in hyp.values()))}
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(hyp,indent=2))
if __name__=='__main__':main()
