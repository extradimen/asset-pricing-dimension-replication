#!/usr/bin/env python3
"""Audit HJ, traded-factor self-pricing and regression-alpha estimands."""
from __future__ import annotations
import argparse,json,math
from pathlib import Path
import numpy as np
import evaluate_multi_factor_teacher as multi
import evaluate_teacher_pricing as base

def self_pricing_loadings(factors:np.ndarray)->np.ndarray:
 second=factors.T@factors/factors.shape[0]; ridge=1e-6*max(float(np.trace(second))/second.shape[0],1e-12)
 return np.linalg.solve(second+ridge*np.eye(second.shape[0]),factors.mean(0))

def moment_vector(factors,returns,loadings): return ((1-factors@loadings)[:,None]*returns).mean(0)
def alpha_vector(factors,returns): return (np.linalg.pinv(np.column_stack([np.ones(len(factors)),factors]))@returns)[0]

def eigen_decomposition(returns:np.ndarray,moment:np.ndarray,alpha:np.ndarray)->dict:
 second=returns.T@returns/returns.shape[0]; values,vectors=np.linalg.eigh(second); ridge=1e-4*np.trace(second)/second.shape[0]
 contributions=(vectors.T@moment)**2/(values+ridge); alpha_energy=(vectors.T@alpha)**2; half=len(values)//2
 total=max(float(contributions.sum()),1e-20); atotal=max(float(alpha_energy.sum()),1e-20)
 order=np.argsort(contributions)[::-1]
 return {"ridge":float(ridge),"hj_squared_from_components":float(contributions.sum()),"lowest_variance_half_hj_share":float(contributions[:half].sum()/total),"highest_variance_half_hj_share":float(contributions[half:].sum()/total),"top_5_hj_direction_share":float(contributions[order[:5]].sum()/total),"lowest_variance_half_alpha_energy_share":float(alpha_energy[:half].sum()/atotal),"highest_variance_half_alpha_energy_share":float(alpha_energy[half:].sum()/atotal),"top_hj_directions":[{"ascending_eigen_index":int(i),"eigenvalue":float(values[i]),"hj_share":float(contributions[i]/total),"alpha_energy_share":float(alpha_energy[i]/atotal)} for i in order[:10]]}

def diagnostics(factors,returns,loadings)->dict:
 moment=moment_vector(factors,returns,loadings); alpha=alpha_vector(factors,returns); factor_moment=moment_vector(factors,factors,loadings); weight=multi.second_weight(returns)
 denom=np.linalg.norm(moment)*np.linalg.norm(alpha); corr=np.corrcoef(np.abs(moment),np.abs(alpha))[0,1] if len(moment)>1 else float("nan")
 return {"hj_ridge_annualized":float(math.sqrt(max(moment@weight@moment,0))*math.sqrt(12)),"mean_absolute_pricing_moment_annualized":float(np.abs(moment).mean()*12),"mean_absolute_alpha_annualized":float(np.abs(alpha).mean()*12),"maximum_absolute_alpha_annualized":float(np.abs(alpha).max()*12),"factor_self_pricing_mean_absolute_moment_annualized":float(np.abs(factor_moment).mean()*12),"factor_self_pricing_maximum_absolute_moment_annualized":float(np.abs(factor_moment).max()*12),"moment_alpha_cosine":float(moment@alpha/denom) if denom>0 else float("nan"),"absolute_moment_alpha_correlation":float(corr),"sdf_loading_norm":float(np.linalg.norm(loadings)),"eigen_decomposition":eigen_decomposition(returns,moment,alpha)}

def parse_args():
 p=argparse.ArgumentParser(description=__doc__)
 for x in ["v008-k5","v011-group-dro","v012-adversarial-8","portfolios-25","industries-49","ff5","momentum","output"]: p.add_argument(f"--{x}",type=Path,required=True)
 p.add_argument("--experiment-id",required=True); return p.parse_args()

def main():
 a=parse_args(); _,raw25=base.read_first_value_weighted_monthly(a.portfolios_25,25,"size_bm::"); _,raw49=base.read_first_value_weighted_monthly(a.industries_49,49,"industry::"); ff5=base.read_factor_file(a.ff5,["Mkt-RF","SMB","HML","RMW","CMA","RF"]); mom=base.read_factor_file(a.momentum,["Mom"]); rf={m:x[-1] for m,x in ff5.items()}
 periods={"validation":sorted(m for m in raw25 if "2000-01"<=m<="2009-12"),"development":sorted(m for m in raw25 if "2010-01"<=m<="2019-12")}; assets={}
 for label,source in [("size_bm_25",raw25),("industry_49",raw49)]: assets[label]={p:base.asset_matrix(ms,source,rf) for p,ms in periods.items()}
 assets["combined_74"]={p:np.column_stack([assets["size_bm_25"][p],assets["industry_49"][p]]) for p in periods}
 months=periods["validation"]+periods["development"]; mappings={"ff5_momentum_6factor":{m:np.concatenate([ff5[base.next_month(m)][:-1],mom[base.next_month(m)]]) for m in months},"v008_k5":multi.read_factor_matrix(a.v008_k5),"v011_group_dro":multi.read_factor_matrix(a.v011_group_dro),"v012_adversarial_8":multi.read_factor_matrix(a.v012_adversarial_8)}
 output={}
 for model,mapping in mappings.items():
  vf=multi.align_matrix(periods["validation"],mapping); df=multi.align_matrix(periods["development"],mapping); strict=self_pricing_loadings(vf); output[model]={}
  for family,r in assets.items():
   unrestricted=multi.fit_loadings(vf,r["validation"]); joint=multi.fit_loadings(vf,np.column_stack([r["validation"],vf])); output[model][family]={"loadings":{"unrestricted_test_asset_hj":unrestricted.tolist(),"joint_hj_with_traded_factors":joint.tolist(),"strict_factor_self_pricing":strict.tolist()},"validation":{},"development":{}}
   for name,b in [("unrestricted_test_asset_hj",unrestricted),("joint_hj_with_traded_factors",joint),("strict_factor_self_pricing",strict)]: output[model][family]["validation"][name]=diagnostics(vf,r["validation"],b); output[model][family]["development"][name]=diagnostics(df,r["development"],b)
 result={"schema_version":1,"experiment_id":a.experiment_id,"sealed_period_accessed":False,"loading_estimation_period":"2000-01 to 2009-12","models":output}; a.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"models":list(output),"families":list(assets)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
