#!/usr/bin/env python3
"""Trace pricing-error rankings from equal-weight to HJ asset geometry."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import audit_hj_alpha_estimands as estimands
import evaluate_multi_factor_teacher as multi
import evaluate_teacher_pricing as base

GAMMAS=(0.0,0.25,0.5,0.75,1.0)

def geometry_weight(returns:np.ndarray,gamma:float):
 second=returns.T@returns/returns.shape[0]; values,vectors=np.linalg.eigh(second); ridge=1e-4*np.trace(second)/second.shape[0]; diagonal=(values+ridge)**(-gamma); scale=returns.shape[1]/np.sum(diagonal*values); diagonal*=scale; return (vectors*diagonal)@vectors.T,values,vectors,diagonal,ridge

def fit_joint(factors:np.ndarray,assets:np.ndarray,gamma:float)->np.ndarray:
 priced=np.column_stack([assets,factors]); weight,*_=geometry_weight(priced,gamma); mean=priced.mean(0); direction=factors.T@priced/factors.shape[0]; system=direction@weight@direction.T; ridge=1e-6*max(float(np.trace(system))/system.shape[0],1e-12); return np.linalg.solve(system+ridge*np.eye(system.shape[0]),direction@weight@mean)

def metric(factors,returns,loadings,gamma,names):
 moment=estimands.moment_vector(factors,returns,loadings); alpha=estimands.alpha_vector(factors,returns); weight,values,vectors,diagonal,ridge=geometry_weight(returns,gamma); projected=vectors.T@moment; contribution=projected**2*diagonal; shares=contribution/max(contribution.sum(),1e-20); order=np.argsort(shares)[::-1]; blocks=np.array_split(np.arange(len(values)),10)
 leading=[]
 for index in order[:5]:
  assets=np.argsort(np.abs(vectors[:,index]))[::-1][:8]; leading.append({"ascending_eigen_index":int(index),"eigenvalue":float(values[index]),"distance_share":float(shares[index]),"top_assets":[{"asset":names[i],"loading":float(vectors[i,index])} for i in assets]})
 return {"distance_annualized":float(math.sqrt(max(moment@weight@moment,0))*math.sqrt(12)),"mean_absolute_euler_moment_annualized":float(np.abs(moment).mean()*12),"root_mean_squared_euler_moment_annualized":float(np.sqrt(np.mean(moment**2))*12),"mean_absolute_alpha_annualized":float(np.abs(alpha).mean()*12),"factor_self_pricing_mean_absolute_moment_annualized":float(np.abs(estimands.moment_vector(factors,factors,loadings)).mean()*12),"effective_error_direction_count":float(1/max(np.sum(shares**2),1e-20)),"eigenvalue_decile_distance_shares":[float(shares[b].sum()) for b in blocks],"leading_error_directions":leading,"weight_ridge":float(ridge)}

def parse_args():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ["v008-k1","v008-k3","v008-k5","v011-group-dro","v012-adversarial-8","portfolios-25","industries-49","ff5","momentum","output"]: p.add_argument(f"--{x}",type=Path,required=True)
 p.add_argument("--experiment-id",required=True); return p.parse_args()

def main():
 a=parse_args(); names25,raw25=base.read_first_value_weighted_monthly(a.portfolios_25,25,"size_bm::"); names49,raw49=base.read_first_value_weighted_monthly(a.industries_49,49,"industry::"); ff5=base.read_factor_file(a.ff5,["Mkt-RF","SMB","HML","RMW","CMA","RF"]); mom=base.read_factor_file(a.momentum,["Mom"]); rf={m:x[-1] for m,x in ff5.items()}; periods={"validation":sorted(m for m in raw25 if "2000-01"<=m<="2009-12"),"development":sorted(m for m in raw25 if "2010-01"<=m<="2019-12")}; assets={}
 for label,source,names in [("size_bm_25",raw25,names25),("industry_49",raw49,names49)]: assets[label]={"names":names,**{p:base.asset_matrix(ms,source,rf) for p,ms in periods.items()}}
 assets["combined_74"]={"names":names25+names49,**{p:np.column_stack([assets["size_bm_25"][p],assets["industry_49"][p]]) for p in periods}}
 months=periods["validation"]+periods["development"]; mappings={"ff5_momentum_6factor":{m:np.concatenate([ff5[base.next_month(m)][:-1],mom[base.next_month(m)]]) for m in months},"v008_k1":multi.read_factor_matrix(a.v008_k1),"v008_k3":multi.read_factor_matrix(a.v008_k3),"v008_k5":multi.read_factor_matrix(a.v008_k5),"v011_group_dro":multi.read_factor_matrix(a.v011_group_dro),"v012_adversarial_8":multi.read_factor_matrix(a.v012_adversarial_8)}; models={}
 for model,mapping in mappings.items():
  vf=multi.align_matrix(periods["validation"],mapping); df=multi.align_matrix(periods["development"],mapping); strict=estimands.self_pricing_loadings(vf); models[model]={"factor_count":vf.shape[1],"families":{}}
  for family,data in assets.items():
   entry={}
   for gamma in GAMMAS:
    joint=fit_joint(vf,data["validation"],gamma); entry[str(gamma)]={"strict_self_pricing":{"validation":metric(vf,data["validation"],strict,gamma,data["names"]),"development":metric(df,data["development"],strict,gamma,data["names"])},"geometry_specific_joint":{"validation":metric(vf,data["validation"],joint,gamma,data["names"]),"development":metric(df,data["development"],joint,gamma,data["names"])}}
   models[model]["families"][family]=entry
 rankings={case:{str(g):sorted(models,key=lambda n:models[n]["families"]["combined_74"][str(g)][case]["development"]["distance_annualized"]) for g in GAMMAS} for case in ["strict_self_pricing","geometry_specific_joint"]}
 result={"schema_version":1,"experiment_id":a.experiment_id,"sealed_period_accessed":False,"geometry_exponents":GAMMAS,"weight_normalization":"trace(W_gamma S)/N=1","models":models,"development_combined_rankings":rankings}; a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps(rankings,indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
