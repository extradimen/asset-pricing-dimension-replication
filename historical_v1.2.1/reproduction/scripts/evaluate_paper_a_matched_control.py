#!/usr/bin/env python3
"""Evaluate the frozen paired feature ablation without reselecting specifications.

All validation SDF coefficients remain fixed in resampling. Evaluation second
moments are recomputed in each common time-block draw, as are all K distances.
Intervals are pointwise descriptive intervals conditional on trained networks.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import platform
import sys
from pathlib import Path
import numpy as np


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2)+'\n')


def write_csv(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def coefficient(f):
    second=f.T@f/len(f)
    ridge=1e-6*max(float(np.trace(second))/f.shape[1],1e-12)
    return np.linalg.solve(second+ridge*np.eye(f.shape[1]),f.mean(0))


def distances(sdf,returns,gammas):
    """sdf is arms x K x seeds x months; final axis of output is gamma."""
    second=returns.T@returns/len(returns)
    values,vectors=np.linalg.eigh(second)
    ridge=1e-4*np.trace(second)/returns.shape[1]
    diagonal=(values[:,None]+ridge)**(-np.asarray(gammas)[None,:])
    diagonal*=returns.shape[1]/(values[:,None]*diagonal).sum(0)
    moments=np.einsum('akst,tj->aksj',sdf,returns)/len(returns)
    components=np.einsum('aksj,jl->aksl',moments,vectors)**2
    return np.sqrt(np.maximum(12*np.einsum('aksl,lg->aksg',components,diagonal),0))


def circular_blocks(rng,n,block):
    starts=rng.integers(0,n,size=int(np.ceil(n/block)))
    return ((starts[:,None]+np.arange(block))%n).ravel()[:n]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['config','data-dir','training-dir','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());out=a.output_dir
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    manifest=json.loads((a.training_dir/'output_manifest.json').read_text())
    for item in manifest['outputs']:
        if sha(a.training_dir/item['path'])!=item['sha256']:raise RuntimeError('Training hash mismatch')
    with np.load(a.data_dir/'development.npz') as z:
        returns=z['assets'].astype(np.float64);months=z['target_months']
    with np.load(a.data_dir/'validation.npz') as z:vmonths=z['target_months']
    arms=c['arms'];ks=c['factor_counts'];seeds=c['seeds'];gammas=c['geometry_grid']
    sdf=np.empty((len(arms),len(ks),len(seeds),len(months)));coefficients=[];init={}
    for ai,arm in enumerate(arms):
        for ki,k in enumerate(ks):
            for si,seed in enumerate(seeds):
                folder=a.training_dir/(arm+'-k'+str(k)+'-seed'+str(seed))
                rows=list(csv.DictReader((folder/'monthly_factor_returns.csv').open()))
                vs=[r for r in rows if r['split']=='validation'];ds=[r for r in rows if r['split']=='development']
                for subset,expected in [(vs,vmonths),(ds,months)]:
                    observed=np.array([r['target_month'] for r in subset],dtype='datetime64[M]').astype(int)
                    np.testing.assert_array_equal(observed,expected)
                vf=np.array([[float(r['factor_'+str(i)]) for i in range(k)] for r in vs])
                df=np.array([[float(r['factor_'+str(i)]) for i in range(k)] for r in ds])
                b=coefficient(vf);sdf[ai,ki,si]=1-df@b
                coefficients.append({'arm':arm,'K':k,'seed':seed,'b':b.tolist()})
                q=json.loads((folder/'quality_report.json').read_text())
                if ai==0:init[k,seed]=q['initialization_sha256']
                else:assert init[k,seed]==q['initialization_sha256']
    selections={'size_bm_25':slice(0,25),'industry_49':slice(25,74),'combined_74':slice(0,74)}
    surface=[];summary=[];rankings=[];agreement=[];point={}
    for family,cols in selections.items():
        values=distances(sdf,returns[:,cols],gammas);point[family]=values
        means=values.mean(2)
        for ai,arm in enumerate(arms):
            for gi,gamma in enumerate(gammas):
                order=np.argsort(means[ai,:,gi],kind='stable')
                rankings.append({'asset_family':family,'arm':arm,'gamma':gamma,
                    'best_K':ks[order[0]],'runner_up_K':ks[order[1]],
                    'margin':float(means[ai,order[1],gi]-means[ai,order[0],gi]),
                    'complete_order':' > '.join(str(ks[i]) for i in order)})
                for ki,k in enumerate(ks):
                    summary.append({'asset_family':family,'arm':arm,'K':k,'gamma':gamma,
                        'mean_distance':float(means[ai,ki,gi]),'seed_sd':float(values[ai,ki,:,gi].std(ddof=1))})
                    for si,seed in enumerate(seeds):surface.append({'asset_family':family,'arm':arm,'K':k,
                        'seed':seed,'gamma':gamma,'distance':float(values[ai,ki,si,gi])})
            for si,seed in enumerate(seeds):
                for gi,gamma in enumerate(gammas):
                    order=np.argsort(values[ai,:,si,gi],kind='stable')
                    # Preserve all seed-specific orderings, not just the mean winner.
                    rankings.append({'asset_family':family,'arm':arm+'-seed'+str(seed),'gamma':gamma,
                        'best_K':ks[order[0]],'runner_up_K':ks[order[1]],
                        'margin':float(values[ai,order[1],si,gi]-values[ai,order[0],si,gi]),
                        'complete_order':' > '.join(str(ks[i]) for i in order)})
        for gi,gamma in enumerate(gammas):
            r0=np.argsort(np.argsort(means[0,:,gi]));r1=np.argsort(np.argsort(means[1,:,gi]))
            agreement.append({'asset_family':family,'gamma':gamma,'spearman':float(np.corrcoef(r0,r1)[0,1]),
                'same_mean_winner':bool(np.argmin(means[0,:,gi])==np.argmin(means[1,:,gi]))})
    rng=np.random.default_rng(c['inference']['seed']);draws=c['inference']['draws']
    boot=np.empty((draws,len(ks),2));bsw=np.zeros((2,len(ks),2),int)
    for draw in range(draws):
        ti=circular_blocks(rng,len(months),c['inference']['month_block'])
        si=rng.integers(0,len(seeds),size=len(seeds))
        d=distances(sdf[:,:,:,ti],returns[ti],[0.,1.])[:,:,si,:].mean(2)
        boot[draw]=d[1]-d[0]
        for ai in range(2):
            for gi in range(2):bsw[ai,np.argmin(d[ai,:,gi]),gi]+=1
    intervals=[]
    for ki,k in enumerate(ks):
        for gi,gamma in enumerate([0.,1.]):
            original=point['combined_74'][:,:,:,0 if gi==0 else -1].mean(2)
            lo,hi=np.quantile(boot[:,ki,gi],[.025,.975])
            intervals.append({'asset_family':'combined_74','K':k,'gamma':gamma,
                'full92':float(original[0,ki]),'masked86':float(original[1,ki]),
                'difference_masked_minus_full':float(original[1,ki]-original[0,ki]),
                'ci95_low':float(lo),'ci95_high':float(hi),
                'full92_bootstrap_winner_frequency':float(bsw[0,ki,gi]/draws),
                'masked86_bootstrap_winner_frequency':float(bsw[1,ki,gi]/draws)})
    for name,rows in [('seed_geometry_surface',surface),('mean_geometry_surface',summary),('rankings',rankings),
                      ('rank_agreement',agreement),('paired_endpoint_inference',intervals)]:write_csv(out/(name+'.csv'),rows)
    write_json(out/'validation_coefficients.json',coefficients)
    write_json(out/'quality_report.json',{'status':'completed','models':sdf.shape[:3],'development_months':len(months),
        'all_30_initialization_pairs_match':True,'evidence_class':c['evidence_class'],'sealed_period_accessed':False,
        'bootstrap_draws':draws,'metric_second_moment_reestimated_in_each_draw':True,
        'interpretation':'Positive paired difference means that masking six features worsens pricing distance. Intervals are pointwise and conditional on existing training/validation estimates; no joint significance or prospective validation claim.'})
    write_json(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()})
    write_json(out/'output_manifest.json',{'inputs':[{'path':str(f),'sha256':sha(f)} for f in [a.config,
        a.data_dir/'output_manifest.json',a.training_dir/'output_manifest.json',Path(__file__)]],
        'outputs':[{'path':f.name,'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(out.iterdir()) if f.is_file()]})
    print(json.dumps({'status':'completed','paired_endpoints':intervals}),flush=True)


if __name__=='__main__':main()
