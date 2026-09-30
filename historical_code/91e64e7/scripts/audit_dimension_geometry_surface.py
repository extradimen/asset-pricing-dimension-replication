#!/usr/bin/env python3
"""Estimate the seed-replicated dimension-geometry surface and crossing points."""
from __future__ import annotations
import argparse,json,math,re
from pathlib import Path
import numpy as np
import audit_hj_alpha_estimands as estimands
import evaluate_multi_factor_teacher as multi
import evaluate_teacher_pricing as base

GAMMAS=np.round(np.arange(0,1.0001,.05),2); PATTERN=re.compile(r"factors-k(\d+)-seed(\d+)$")
def eigensystem(returns):
 second=returns.T@returns/len(returns); values,vectors=np.linalg.eigh(second); ridge=1e-4*np.trace(second)/second.shape[0]; return values,vectors,ridge
def distance(moment,values,vectors,ridge,gamma):
 diagonal=(values+ridge)**(-gamma); diagonal*=len(values)/np.sum(diagonal*values); return float(math.sqrt(max(np.sum((vectors.T@moment)**2*diagonal),0))*math.sqrt(12))
def crossing(gammas,gaps):
 for i in range(len(gaps)-1):
  if gaps[i]==0:return float(gammas[i])
  if gaps[i]*gaps[i+1]<0:return float(gammas[i]+(gammas[i+1]-gammas[i])*gaps[i]/(gaps[i]-gaps[i+1]))
 return None
def circular_indices(length,block,rng):
 parts=[]
 while sum(len(x) for x in parts)<length:
  start=int(rng.integers(length)); parts.append((start+np.arange(block))%length)
 return np.concatenate(parts)[:length]
def parse_args():
 p=argparse.ArgumentParser(description=__doc__); p.add_argument("--sweep-root",type=Path,required=True)
 for x in ["portfolios-25","industries-49","ff5","output"]: p.add_argument(f"--{x}",type=Path,required=True)
 p.add_argument("--experiment-id",required=True); p.add_argument("--bootstrap-draws",type=int,default=500); p.add_argument("--bootstrap-seed",type=int,default=20260924); return p.parse_args()
def main():
 a=parse_args(); names25,raw25=base.read_first_value_weighted_monthly(a.portfolios_25,25,"size_bm::"); names49,raw49=base.read_first_value_weighted_monthly(a.industries_49,49,"industry::"); ff=base.read_factor_file(a.ff5,["Mkt-RF","SMB","HML","RMW","CMA","RF"]); rf={m:x[-1] for m,x in ff.items()}; periods={"validation":sorted(m for m in raw25 if "2000-01"<=m<="2009-12"),"development":sorted(m for m in raw25 if "2010-01"<=m<="2019-12")}; assets={}
 for label,source in [("size_bm_25",raw25),("industry_49",raw49)]: assets[label]={p:base.asset_matrix(ms,source,rf) for p,ms in periods.items()}
 assets["combined_74"]={p:np.column_stack([assets["size_bm_25"][p],assets["industry_49"][p]]) for p in periods}; models={}
 for directory in sorted(a.sweep_root.iterdir()):
  match=PATTERN.match(directory.name)
  if not match or not directory.is_dir(): continue
  count,seed=map(int,match.groups()); mapping=multi.read_factor_matrix(directory/"monthly_factor_returns.csv"); vf=multi.align_matrix(periods["validation"],mapping); df=multi.align_matrix(periods["development"],mapping); loading=estimands.self_pricing_loadings(vf); key=f"k{count}_seed{seed}"; models[key]={"factor_count":count,"seed":seed,"development_factors":df,"loading":loading,"families":{}}
  for family,r in assets.items():
   moment=estimands.moment_vector(df,r["development"],loading); values,vectors,ridge=eigensystem(r["development"]); alpha=estimands.alpha_vector(df,r["development"]); models[key]["families"][family]={"mean_absolute_alpha_annualized":float(np.abs(alpha).mean()*12),"mean_absolute_euler_moment_annualized":float(np.abs(moment).mean()*12),"distances":{str(g):distance(moment,values,vectors,ridge,float(g)) for g in GAMMAS}}
 expected={(k,s) for k in [1,2,3,4,5,8] for s in [20260924,20260925,20260926,20260927,20260928]}; observed={(m["factor_count"],m["seed"]) for m in models.values()};
 if observed!=expected: raise RuntimeError(f"Incomplete surface: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
 aggregate={}; rankings={}
 for family in assets:
  aggregate[family]={}; rankings[family]={}
  for g in GAMMAS:
   rows=[]
   for count in [1,2,3,4,5,8]:
    values=np.asarray([models[f"k{count}_seed{s}"]["families"][family]["distances"][str(g)] for s in [20260924,20260925,20260926,20260927,20260928]]); rows.append((float(values.mean()),count)); aggregate[family].setdefault(str(count),{})[str(g)]={"mean":float(values.mean()),"standard_error":float(values.std(ddof=1)/np.sqrt(len(values))),"seed_values":values.tolist()}
   rankings[family][str(g)]=[count for _,count in sorted(rows)]
 crossings={}
 for seed in [20260924,20260925,20260926,20260927,20260928]:
  gaps=[models[f"k1_seed{seed}"]["families"]["combined_74"]["distances"][str(g)]-models[f"k5_seed{seed}"]["families"]["combined_74"]["distances"][str(g)] for g in GAMMAS]; crossings[str(seed)]={"gap_gamma_0":gaps[0],"gap_gamma_1":gaps[-1],"crossing_gamma":crossing(GAMMAS,gaps)}
 rng=np.random.default_rng(a.bootstrap_seed); returns=assets["combined_74"]["development"]; draws=[]
 for _ in range(a.bootstrap_draws):
  idx=circular_indices(len(returns),12,rng); values,vectors,ridge=eigensystem(returns[idx]); gaps=[]
  for g in GAMMAS:
   seed_gaps=[]
   for seed in [20260924,20260925,20260926,20260927,20260928]:
    vals=[]
    for count in [1,5]:
     model=models[f"k{count}_seed{seed}"]; moment=estimands.moment_vector(model["development_factors"][idx],returns[idx],model["loading"]); vals.append(distance(moment,values,vectors,ridge,float(g)))
    seed_gaps.append(vals[0]-vals[1])
   gaps.append(float(np.mean(seed_gaps)))
  draws.append({"gap_gamma_0":gaps[0],"gap_gamma_1":gaps[-1],"crossing_gamma":crossing(GAMMAS,gaps)})
 finite=np.asarray([x["crossing_gamma"] for x in draws if x["crossing_gamma"] is not None]); bootstrap={"draws":a.bootstrap_draws,"gap_gamma_0_ci95":np.quantile([x["gap_gamma_0"] for x in draws],[.025,.975]).tolist(),"gap_gamma_1_ci95":np.quantile([x["gap_gamma_1"] for x in draws],[.025,.975]).tolist(),"crossing_found_fraction":float(len(finite)/len(draws)),"crossing_gamma_ci95":np.quantile(finite,[.025,.975]).tolist() if len(finite) else None}
 replicated=sum(x["gap_gamma_0"]>0 and x["gap_gamma_1"]<0 and x["crossing_gamma"] is not None for x in crossings.values())>=4
 serializable={k:{x:y for x,y in v.items() if x not in ["development_factors","loading"]} for k,v in models.items()}; result={"schema_version":1,"experiment_id":a.experiment_id,"sealed_period_accessed":False,"gammas":GAMMAS.tolist(),"models":serializable,"aggregate":aggregate,"dimension_rankings":rankings,"paired_k1_k5_crossings":crossings,"block_bootstrap":bootstrap,"preregistered_replication_rule_passed":replicated}; a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"crossings":crossings,"bootstrap":bootstrap,"replicated":replicated},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
