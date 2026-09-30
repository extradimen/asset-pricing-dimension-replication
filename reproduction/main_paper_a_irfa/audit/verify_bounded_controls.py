#!/usr/bin/env python3
"""Read-only scientific identity checks on frozen control outputs; no fitting."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'main_paper_a_irfa';DATA=PAPER/'visuals/figure_data'
checks=[]
def record(name,n,passed):checks.append({'check':name,'values':int(n),'passed':bool(passed)})
def close(a,b):return np.allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True)

a=ROOT/'experiments/P1-G4-V001/P1-G4-V001-R001'
b=ROOT/'experiments/P1-G4-V002/P1-G4-V002-EVAL001'
for prefix,source in [('architecture',a),('matched',b)]:
    for p in DATA.glob(prefix+'_*.csv'):
        original=source/p.name.removeprefix(prefix+'_')
        record('Copied control source: '+p.name,1,p.read_bytes()==original.read_bytes())
g=pd.read_csv(a/'monthly_geometry.csv');s=pd.read_csv(a/'paired_summary.csv')
record('Width-normalized participation rank',len(g),close(g.participation_rank/g.width,g.rank_fraction))
record('Unique trained/random spectral cells',len(g),len(g[g.condition=='trained'])==32400 and len(g[g.condition=='random'])==5400)
for _,r in s.iterrows():
    z=g[(g.normalization==r.normalization)&(g.layer=='hidden'+str(r.layer))]
    t=z[z.condition=='trained'][r.metric];n=z[z.condition=='random'][r.metric]
    record('Architecture means '+r.normalization+'/'+str(r.layer)+'/'+r.metric,3,
           close([t.mean(),n.mean(),t.mean()-n.mean()],[r.trained_mean,r.random_mean,r.paired_mean_difference]))
surface=pd.read_csv(b/'seed_geometry_surface.csv');mean=pd.read_csv(b/'mean_geometry_surface.csv')
keys=['asset_family','arm','K','gamma']
expected=surface.groupby(keys).distance.agg(['mean','std']).reset_index()
z=mean.merge(expected,on=keys,validate='one_to_one')
record('Complete matched grid',len(surface),len(surface)==3780 and len(z)==756)
record('Matched seed means and dispersion',len(z)*2,close(z.mean_distance,z['mean']) and close(z.seed_sd,z['std']))
endpoint=pd.read_csv(b/'paired_endpoint_inference.csv')
for _,r in endpoint.iterrows():
    q=mean[(mean.asset_family=='combined_74')&(mean.K==r.K)&(mean.gamma==r.gamma)].set_index('arm')
    record('Matched endpoint K='+str(r.K)+' gamma='+str(r.gamma),3,
           close([q.loc['full92','mean_distance'],q.loc['masked86','mean_distance'],r.masked86-r.full92],
                 [r.full92,r.masked86,r.difference_masked_minus_full]))
rank=pd.read_csv(b/'rankings.csv');r=rank[rank.arm.isin(['full92','masked86'])]
ok=[]
for _,row in r.iterrows():
    q=mean[(mean.asset_family==row.asset_family)&(mean.arm==row.arm)&(mean.gamma==row.gamma)].sort_values('mean_distance',kind='stable')
    ok.append(row.best_K==q.iloc[0].K and close(row.margin,q.iloc[1].mean_distance-q.iloc[0].mean_distance))
record('All mean winners and margins',len(ok),all(ok))
train=ROOT/'experiments/P1-G4-V002/P1-G4-V002-R002'
report=json.loads((train/'quality_report.json').read_text());models=report['models_summary']
initial={};pairs=[]
for m in models:
    key=(m['factor_count'],m['seed'])
    if key in initial:pairs.append(initial[key]==m['initialization_sha256'])
    else:initial[key]=m['initialization_sha256']
record('Thirty initialization pairs',len(pairs),len(models)==60 and len(pairs)==30 and all(pairs))
record('Both CUDA workers completed',2,json.loads((train/'worker_exit_status.json').read_text())['return_codes']==[0,0])
out={'scope':'Identity and source-output verification, not an additional experiment','checks':checks,'all_passed':all(x['passed'] for x in checks)}
(PAPER/'audit/bounded_control_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({'checks':len(checks),'all_passed':out['all_passed'],'verified_values':sum(x['values'] for x in checks)},indent=2))
raise SystemExit(0 if out['all_passed'] else 1)
