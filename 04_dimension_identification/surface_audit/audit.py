from pathlib import Path
import sys,json,csv,hashlib,itertools,statistics,math
import numpy as np
root=Path(__file__).resolve().parents[2];home=Path(__file__).parent;out=home/'results';out.mkdir(exist_ok=False)
config=json.loads((home/'protocol.json').read_text());run=root/config['input_run'];original=json.loads((root/'configs/paper1/P1-G4-V003.json').read_text())
sys.path.insert(0,str(root/'scripts'));from paper_a_cae_core import sdf_coefficient,pricing_weight,pricing_distance
manifest=json.loads((run/'output_manifest.json').read_text()); indexed={r['path']:r for r in manifest['outputs']};sources=[]
def verify(p):
 rel=p.relative_to(run).as_posix();h=hashlib.sha256(p.read_bytes()).hexdigest();assert h==indexed[rel]['sha256'],rel;sources.append({'path':str(p.relative_to(root)),'sha256':h});return p
surface=list(csv.DictReader(verify(run/'analysis/pricing_surface.csv').open()));dim=list(csv.DictReader(verify(run/'analysis/dimension_summary.csv').open()));bench=list(csv.DictReader(verify(run/'analysis/benchmark_surface.csv').open()))
expected=set(itertools.product(config['families'],config['gammas'],config['arms'],config['factor_counts'],config['seeds']))
actual=[(r['family'],float(r['gamma']),r['arm'],int(r['K']),int(r['seed'])) for r in surface];assert len(actual)==len(set(actual))==900;assert set(actual)==expected
v=np.load(verify(run/'prepared/validation.npz'));d=np.load(verify(run/'prepared/development.npz'));assert len(v['months'])==len(d['months'])==120
arrays={};varrays={}
for arm,k,seed in itertools.product(config['arms'],config['factor_counts'],config['seeds']):
 f=np.load(verify(run/f'{arm}-k{k}-seed{seed}'/'factors.npz'));b=sdf_coefficient(f['validation']);arrays[arm,k,seed]=1-f['development']@b;varrays[arm,k,seed]=1-f['validation']@b
slices={'size_bm_25':slice(0,25),'industry_49':slice(25,74),'combined_74':slice(0,74)};errors=[];cache={}
for family,gamma in itertools.product(config['families'],config['gammas']):
 sl=slices[family];vr=v['assets'][:,sl].astype(np.float64);dr=d['assets'][:,sl].astype(np.float64);cache[family,gamma]=(vr,dr,pricing_weight(vr,gamma,original['pricing_ridge']))
for r,key in zip(surface,actual):
 fam,g,arm,k,seed=key;vr,dr,w=cache[fam,g]
 for name,sdf,R in [('development_distance',arrays[arm,k,seed],dr),('validation_distance',varrays[arm,k,seed],vr)]:
  x=float(pricing_distance(sdf,R,w));a=float(r[name]);assert math.isfinite(a) and a>=0;errors.append(abs(x-a));assert abs(x-a)<=config['numerical_tolerance']*max(1,abs(a)),key
means=[];lookup={}
for fam,g,arm,k in itertools.product(config['families'],config['gammas'],config['arms'],config['factor_counts']):
 sub=[r for r in surface if (r['family'],float(r['gamma']),r['arm'],int(r['K']))==(fam,g,arm,k)];assert len(sub)==5
 row={'family':fam,'gamma':g,'arm':arm,'K':k,'validation_mean':statistics.mean(float(r['validation_distance']) for r in sub),'development_mean':statistics.mean(float(r['development_distance']) for r in sub),'seed_sd':statistics.stdev(float(r['development_distance']) for r in sub),'n_seeds':5};means.append(row);lookup[fam,g,arm,k]=row
for r in dim:assert abs(lookup['combined_74',1,r['arm'],int(r['K'])]['development_mean']-float(r['HJ_distance']))<1e-12
bf=sdf_coefficient(v['benchmark']);bsdf=1-d['benchmark']@bf
for r in bench:
 vr,dr,w=cache[r['family'],float(r['gamma'])];assert abs(float(pricing_distance(bsdf,dr,w))-float(r['FF5_MOM_distance']))<1e-10
summary=[];transfers=[]
for fam,g,arm in itertools.product(config['families'],config['gammas'],config['arms']):
 rs=[lookup[fam,g,arm,k] for k in config['factor_counts']];best=min(rs,key=lambda r:r['development_mean']);vbest=min(rs,key=lambda r:r['validation_mean']);ordered=sorted(rs,key=lambda r:r['development_mean']);b=best['development_mean'];cost=lambda r:100*(r['development_mean']/b-1)
 row={'family':fam,'gamma':g,'arm':arm,'development_best_K':best['K'],'minimum':b,'second_K':ordered[1]['K'],'second_excess_pct':cost(ordered[1]),'validation_best_K':vbest['K'],'validation_selected_excess_pct':cost(vbest),'K5_excess_pct':cost(lookup[fam,g,arm,5]),'K8_excess_pct':cost(lookup[fam,g,arm,8])};summary.append(row)
 for r in rs:r['excess_distance']=r['development_mean']-b;r['excess_pct']=cost(r)
 if g in [0,1]:
  other=min([lookup[fam,1-g,arm,k] for k in config['factor_counts']],key=lambda r:r['development_mean']);transfers.append({'family':fam,'arm':arm,'target_gamma':g,'best_K':best['K'],'transfer_K':other['K'],'excess_pct':cost(lookup[fam,g,arm,other['K']])})
def write(name,rows):
 with (out/name).open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
write('all_180_candidate_means.csv',means);write('all_30_task_summaries.csv',summary);write('all_12_endpoint_transfers.csv',transfers)
receipt={'status':'passed','frozen_rows':900,'recomputed_distances':1800,'max_absolute_error':max(errors),'cells':180,'tasks':30,'source_manifest_checked':True,'primary_12_means_match':True,'all_15_benchmarks_match':True,'new_training':False,'new_resampling':False,'sources':sources,'scripts':[{'path':str(p.relative_to(root)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in [Path(__file__),root/'scripts/paper_a_cae_core.py',root/'scripts/evaluate_paper_a_cae_control.py']], 'interpretation':'Descriptive point estimates; SD across seeds is not sampling uncertainty; development minima retrospective; validation pricing selection differs from original predictive-R2 competence rule.'}
(out/'audit_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:v for k,v in receipt.items() if k not in ['sources','scripts']}));print(json.dumps(summary,indent=2))
