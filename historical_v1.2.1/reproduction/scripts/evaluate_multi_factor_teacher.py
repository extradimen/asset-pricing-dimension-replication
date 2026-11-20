#!/usr/bin/env python3
"""Evaluate multi-direction neural SDF teachers on public test assets."""

from __future__ import annotations
import argparse, csv, json, math
from pathlib import Path
import numpy as np
import evaluate_teacher_pricing as base

VALID_START,VALID_END="2000-01","2009-12"; DEV_START,DEV_END="2010-01","2019-12"

def read_factor_matrix(path: Path) -> dict[str,np.ndarray]:
    with path.open(newline="") as stream:
        rows=csv.DictReader(stream); columns=[x for x in rows.fieldnames or [] if x.startswith("factor_")]
        return {row["month"]:np.asarray([float(row[x]) for x in columns]) for row in rows}

def align_matrix(months:list[str], mapping:dict[str,np.ndarray]) -> np.ndarray:
    return np.vstack([mapping[x] for x in months])

def second_weight(returns:np.ndarray) -> np.ndarray:
    second=returns.T@returns/returns.shape[0]; ridge=1e-4*np.trace(second)/second.shape[0]
    return np.linalg.inv(second+ridge*np.eye(second.shape[0]))

def fit_loadings(factors:np.ndarray, returns:np.ndarray) -> np.ndarray:
    mean=returns.mean(0); direction=factors.T@returns/returns.shape[0]; weight=second_weight(returns)
    system=direction@weight@direction.T; ridge=1e-6*max(float(np.trace(system))/max(system.shape[0],1),1e-12)
    return np.linalg.solve(system+ridge*np.eye(system.shape[0]),direction@weight@mean)

def fit_tangency(factors:np.ndarray) -> np.ndarray:
    covariance=np.atleast_2d(np.cov(factors,rowvar=False)); mean=factors.mean(0)
    ridge=1e-6*max(float(np.trace(covariance))/covariance.shape[0],1e-12)
    return np.linalg.solve(covariance+ridge*np.eye(covariance.shape[0]),mean)

def metrics(factors:np.ndarray, returns:np.ndarray, loadings:np.ndarray, tangency:np.ndarray) -> dict[str,float]:
    sdf_factor=factors@loadings; moment=((1-sdf_factor)[:,None]*returns).mean(0); weight=second_weight(returns)
    design=np.column_stack([np.ones(factors.shape[0]),factors]); alpha=(np.linalg.pinv(design)@returns)[0]
    portfolio=factors@tangency; std=portfolio.std(ddof=1)
    return {"factor_count":factors.shape[1],"factor_span_sharpe":float(portfolio.mean()/std*math.sqrt(12)) if std>0 else float("nan"),"hj_ridge_annualized":float(math.sqrt(max(moment@weight@moment,0))*math.sqrt(12)),"mean_absolute_pricing_moment_annualized":float(np.abs(moment).mean()*12),"mean_absolute_alpha_annualized":float(np.abs(alpha).mean()*12),"maximum_absolute_alpha_annualized":float(np.abs(alpha).max()*12)}

def evaluate(mapping,periods,asset_sets):
    output={}
    for asset_name,values in asset_sets.items():
        valid=align_matrix(periods["validation"],mapping); dev=align_matrix(periods["development"],mapping)
        loadings=fit_loadings(valid,values["validation"]); tangency=fit_tangency(valid)
        output[asset_name]={"validation":metrics(valid,values["validation"],loadings,tangency),"development":metrics(dev,values["development"],loadings,tangency),"validation_sdf_loadings":loadings.tolist(),"validation_tangency_weights":tangency.tolist()}
    return output

def circular_indices(length,block,rng):
    parts=[]
    while sum(x.size for x in parts)<length:
        start=int(rng.integers(0,length)); parts.append((start+np.arange(block))%length)
    return np.concatenate(parts)[:length]

def main()->int:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidate-root",type=Path,required=True); p.add_argument("--linear-predictions",type=Path,required=True)
    for name in ["portfolios-25","industries-49","ff5","momentum","output"]: p.add_argument(f"--{name}",type=Path,required=True)
    p.add_argument("--experiment-id",required=True); p.add_argument("--bootstrap-draws",type=int,default=500); p.add_argument("--bootstrap-seed",type=int,default=20260924)
    args=p.parse_args(); names25,raw25=base.read_first_value_weighted_monthly(args.portfolios_25,25,"size_bm::"); names49,raw49=base.read_first_value_weighted_monthly(args.industries_49,49,"industry::")
    ff5=base.read_factor_file(args.ff5,["Mkt-RF","SMB","HML","RMW","CMA","RF"]); momentum=base.read_factor_file(args.momentum,["Mom"]); rf={m:x[-1] for m,x in ff5.items()}
    periods={"validation":sorted(m for m in raw25 if VALID_START<=m<=VALID_END),"development":sorted(m for m in raw25 if DEV_START<=m<=DEV_END)}
    assets={}
    for label,source in [("size_bm_25",raw25),("industry_49",raw49)]: assets[label]={s:base.asset_matrix(ms,source,rf) for s,ms in periods.items()}
    assets["combined_74"]={s:np.column_stack([assets["size_bm_25"][s],assets["industry_49"][s]]) for s in periods}
    all_months=periods["validation"]+periods["development"]
    mappings={"linear_core92":{m:np.asarray([v]) for m,v in base.linear_factor(args.linear_predictions).items()}}
    ff_matrix={m:np.concatenate([ff5[base.next_month(m)][:-1],momentum[base.next_month(m)]]) for m in all_months}; valid_ff=np.vstack([ff_matrix[m] for m in periods["validation"]]); mappings["ff5_momentum_6factor"]=ff_matrix
    for directory in sorted(args.candidate_root.glob("factors-*")):
        if directory.is_dir(): mappings[directory.name]=read_factor_matrix(directory/"monthly_factor_returns.csv")
    evaluations={name:evaluate(mapping,periods,assets) for name,mapping in mappings.items()}
    baseline_names=["linear_core92","ff5_momentum_6factor"]; selected=min(baseline_names,key=lambda n:evaluations[n]["industry_49"]["validation"]["hj_ridge_annualized"])
    rng=np.random.default_rng(args.bootstrap_seed); comparisons={}
    baseline_valid=align_matrix(periods["validation"],mappings[selected]); baseline_dev=align_matrix(periods["development"],mappings[selected]); returns_valid=assets["combined_74"]["validation"]; returns_dev=assets["combined_74"]["development"]
    b_load=fit_loadings(baseline_valid,returns_valid); b_tan=fit_tangency(baseline_valid); b_point=metrics(baseline_dev,returns_dev,b_load,b_tan)
    for name,mapping in mappings.items():
        if not name.startswith("factors-"): continue
        valid=align_matrix(periods["validation"],mapping); dev=align_matrix(periods["development"],mapping); loading=fit_loadings(valid,returns_valid); tangency=fit_tangency(valid); point=metrics(dev,returns_dev,loading,tangency)
        draws=[]
        for _ in range(args.bootstrap_draws):
            idx=circular_indices(len(dev),12,rng); t=metrics(dev[idx],returns_dev[idx],loading,tangency); b=metrics(baseline_dev[idx],returns_dev[idx],b_load,b_tan)
            draws.append([(b["hj_ridge_annualized"]-t["hj_ridge_annualized"])/b["hj_ridge_annualized"],(b["mean_absolute_alpha_annualized"]-t["mean_absolute_alpha_annualized"])/b["mean_absolute_alpha_annualized"]])
        array=np.asarray(draws); comparisons[name]={"point_estimates":{"hj_ridge_relative_improvement":(b_point["hj_ridge_annualized"]-point["hj_ridge_annualized"])/b_point["hj_ridge_annualized"],"mean_absolute_alpha_relative_improvement":(b_point["mean_absolute_alpha_annualized"]-point["mean_absolute_alpha_annualized"])/b_point["mean_absolute_alpha_annualized"],"factor_span_sharpe_difference":point["factor_span_sharpe"]-b_point["factor_span_sharpe"]},"block_bootstrap":{"hj_ridge_ci95":np.quantile(array[:,0],[.025,.975]).tolist(),"mean_absolute_alpha_ci95":np.quantile(array[:,1],[.025,.975]).tolist()}}
    result={"schema_version":1,"experiment_id":args.experiment_id,"sealed_period_accessed":False,"timing":"characteristic month t paired with official return month t+1","training_asset_family":"size_bm_25","held_out_asset_family":"industry_49","test_assets":{"size_bm_25":names25,"industry_49":names49},"periods":periods,"selected_validation_baseline":selected,"evaluations":evaluations,"comparisons":comparisons}
    args.output.write_text(json.dumps(result,indent=2)+"\n"); print(json.dumps({"selected_baseline":selected,"candidates":list(comparisons)},indent=2)); return 0
if __name__=="__main__": raise SystemExit(main())
