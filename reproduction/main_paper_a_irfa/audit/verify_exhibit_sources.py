#!/usr/bin/env python3
"""Read-only verification of frozen numerical exhibits; no new estimation."""
from pathlib import Path
import json, hashlib, re
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
PAPER=ROOT/'main_paper_a_irfa'; DATA=PAPER/'visuals/figure_data'
checks=[]
def record(name,n,passed,detail=''):
    checks.append(dict(check=name,items=int(n),passed=bool(passed),detail=detail))
def read(p):return json.loads((ROOT/p).read_text())
def close(a,b):return np.allclose(a,b,rtol=1e-8,atol=5e-9,equal_nan=True)
manifest=json.loads((DATA/'figure_data_manifest.json').read_text())
for x in manifest['items']:
    p=DATA/x['file']; h=hashlib.sha256(p.read_bytes()).hexdigest()
    record('Exhibit checksum: '+x['file'],1,h==x['copy_sha256'])
    if x.get('identity_verified'):
        record('Source identity: '+x['file'],1,p.read_bytes()==(ROOT/x['source']).read_bytes())
sources={k:read(v) for k,v in {'Core-92':'experiments/P1-G1-V015/dimension_geometry_surface_R001.json','Core-86':'experiments/P1-G1-V022/core86_geometry_surface_R001.json'}.items()}
d=pd.read_csv(DATA/'table_01_geometry_surface_long.csv'); failures=[]
for _,r in d.iterrows():
    z=sources[r.data_definition]['aggregate'][r.asset_family][str(r.K)][str(r.gamma)]
    if not close([r.mean_distance,r.standard_error,r.lower_2se,r.upper_2se],[z['mean'],z['standard_error'],z['mean']-2*z['standard_error'],z['mean']+2*z['standard_error']]):failures.append(r.to_dict())
record('Every dimension-surface mean, SE, and band',len(d)*4,not failures,str(failures[:2]))
s=pd.read_csv(DATA/'table_02_geometry_seed_long.csv');failures=[]
for _,r in s.iterrows():
    z=sources[r.data_definition]['models'][f'k{r.K}_seed{r.seed}']['families'][r.asset_family]['distances'][str(r.gamma)]
    if not close(r.distance,z):failures.append(r.to_dict())
record('Every individual-seed pricing distance',len(s),not failures,str(failures[:2]))
rank=d.groupby(['data_definition','asset_family','gamma']).mean_distance.rank(method='min')
record('All surface ranks',len(d),close(rank,d.rank_within_surface))
z=read('experiments/P1-G2-V001/sealed_confirmation_R001.json');d=pd.read_csv(DATA/'sealed_table_01_family_inference.csv');failure=[]
for _,r in d.iterrows():
    f=z['family_results'][r.asset_family];b=z['bootstrap']['families'][r.asset_family]
    pairs=[(r.best_K_gamma0,f['best_dimension']['0.0']),(r.best_K_gamma1,f['best_dimension']['1.0']),(r.A_raw,f['A_raw']),(r.attenuation,f['market_direction_attenuation']),(r.completion_raw,f['completion_minus_k1']['raw_moment']),(r.completion_alpha,f['completion_minus_k1']['alpha'])]
    for col,key in [('A','A_raw'),('atten','attenuation'),('raw','completion_raw'),('alpha','completion_alpha')]:
        pairs.extend([(r[col+'_low'],b[key]['ci95'][0]),(r[col+'_high'],b[key]['ci95'][1])])
    if not all(close(a,b) for a,b in pairs):failure.append(r.asset_family)
record('Locked evaluation: all nine families and intervals',len(d)*14,not failure,str(failure))
sim=pd.read_csv(DATA/'identification_condition_summary.csv')
record('Simulation frequency sums',len(sim),close(sim.exact_rate+sim.under_rate+sim.over_rate,1))
record('Simulation oracle exact recovery 64.2%',1,round(sim[sim.evaluation=='oracle'].exact_rate.mean()*100,1)==64.2)
record('Simulation feasible exact recovery 32.6%',1,round(sim[sim.evaluation=='feasible'].exact_rate.mean()*100,1)==32.6)
groot=ROOT/'experiments/P7-G1-V001/P7-G1-V001-CPU-REPLAY001'
g=pd.read_csv(groot/'monthly_geometry.csv'); drift=pd.read_csv(groot/'subspace_drift.csv'); cka=pd.read_csv(groot/'layer_cka.csv');ep=pd.read_csv(groot/'model_geometry_endpoints.csv')
actual=pd.read_csv(DATA/'p7_geometry_distribution.csv'); keys=['factor_count','seed','target_month','layer','normalization']
expected=g[keys+['high_volatility','participation_rank']].merge(drift,on=keys,how='left').rename(columns={'grassmann_distance':'subspace_drift'})
a=actual.sort_values(keys).reset_index(drop=True);b=expected.sort_values(keys).reset_index(drop=True)
record('All geometry-distribution identities',len(a),a[keys+['high_volatility']].equals(b[keys+['high_volatility']]) and close(a[['participation_rank','subspace_drift']],b[['participation_rank','subspace_drift']]))
matrix=pd.read_csv(DATA/'p7_model_evidence_matrix.csv');failure=[]
for _,r in matrix.iterrows():
    match=lambda t:t[(t.factor_count==r.factor_count)&(t.seed==r.seed)]
    gg=match(g);dd=match(drift);cc=match(cka);ee=match(ep).iloc[0]
    pairs=[(r[f'hidden{i}_participation_rank'],gg[(gg.layer==f'hidden{i}')&(gg.normalization=='raw_centered')].participation_rank.median()) for i in [1,2,3]]
    pairs += [(r['cka_'+pair],cc[cc.layer_pair==pair].linear_cka.median()) for pair in ['hidden1-hidden2','hidden2-hidden3']]
    pairs += [(r.hidden3_drift,dd[(dd.layer=='hidden3')&(dd.normalization=='raw_centered')].grassmann_distance.median())]
    pairs += [(r[k],ee[k]) for k in ['development_hj_loss','development_factor_sharpe']]
    if not all(close(a,b) for a,b in pairs):failure.append([r.factor_count,r.seed])
record('All model-level geometry and pricing endpoints',len(matrix)*8,not failure,str(failure))
td=pd.read_csv(DATA/'temporal_decomposition.csv')
record('All temporal loss-decomposition identities',len(td),close(td.A-td.twoB,td.loss_diff))
# Numeric source reports remain available to audit every rounded narrative claim.
text='\n'.join(p.read_text() for p in (PAPER/'manuscript/sections').glob('*.tex'))
bib=(PAPER/'manuscript/references/references.bib').read_text();bibkeys=set(re.findall(r'@\w+\s*\{\s*([^,]+)',bib))
citations={k.strip() for group in re.findall(r'\\cite\w*(?:\[[^]]*\])*\{([^}]+)\}',text) for k in group.split(',')}
record('Bibliography keys resolve',len(citations),not(citations-bibkeys),str(sorted(citations-bibkeys)))
out=dict(scope='Frozen-output audit; not a new experiment or clean-room retraining',checks=checks,all_passed=all(x['passed'] for x in checks),bibliography_entries=len(bibkeys),cited_research_entries=len(citations),uncited_keys=sorted(bibkeys-citations))
(PAPER/'audit/numerical_verification.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
raise SystemExit(0 if out['all_passed'] else 1)
