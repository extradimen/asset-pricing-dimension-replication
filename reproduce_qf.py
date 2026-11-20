"""Frozen aggregate integrity and exhibit replay only; no new experiment or training."""
from pathlib import Path
import csv,hashlib,json,itertools,statistics,math,subprocess,sys
root=Path(__file__).resolve().parent
a=root/'04_dimension_identification/surface_audit';man=root/'01_overview/manuscript'
for line in (root/'SHA256SUMS').read_text().splitlines():
 h,n=line.split('  ',1);assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h,n
rows=list(csv.DictReader((a/'frozen_inputs/pricing_surface.csv').open()))
families=['size_bm_25','industry_49','combined_74'];gammas=[0,.25,.5,.75,1];arms=['linear','cae'];ks=[1,2,3,4,5,8];seeds=list(range(20261005,20261010))
expected=set(itertools.product(families,gammas,arms,ks,seeds));keys=[(r['family'],float(r['gamma']),r['arm'],int(r['K']),int(r['seed'])) for r in rows]
assert len(rows)==len(set(keys))==900 and set(keys)==expected
lookup={};summaries={}
for fam,g,arm,k in itertools.product(families,gammas,arms,ks):
 sub=[r for r in rows if (r['family'],float(r['gamma']),r['arm'],int(r['K']))==(fam,g,arm,k)]
 lookup[fam,g,arm,k]={'validation_mean':statistics.mean(float(r['validation_distance']) for r in sub),'development_mean':statistics.mean(float(r['development_distance']) for r in sub),'seed_sd':statistics.stdev(float(r['development_distance']) for r in sub),'n_seeds':5}
for fam,g,arm in itertools.product(families,gammas,arms):
 order=sorted(ks,key=lambda k:lookup[fam,g,arm,k]['development_mean']);best=order[0];minimum=lookup[fam,g,arm,best]['development_mean'];vb=min(ks,key=lambda k:lookup[fam,g,arm,k]['validation_mean']);cost=lambda k:100*(lookup[fam,g,arm,k]['development_mean']/minimum-1)
 summaries[fam,g,arm]={'development_best_K':best,'minimum':minimum,'second_K':order[1],'second_excess_pct':cost(order[1]),'validation_best_K':vb,'validation_selected_excess_pct':cost(vb),'K5_excess_pct':cost(5),'K8_excess_pct':cost(8)}
 for k in ks:lookup[fam,g,arm,k].update(excess_distance=lookup[fam,g,arm,k]['development_mean']-minimum,excess_pct=cost(k))
def check(a,b,label):assert math.isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-11),(label,a,b)
means=list(csv.DictReader((a/'results/all_180_candidate_means.csv').open()));assert len(means)==180
for r in means:
 x=lookup[r['family'],float(r['gamma']),r['arm'],int(r['K'])]
 for n,v in x.items():check(v,r[n],n)
tasks=list(csv.DictReader((a/'results/all_30_task_summaries.csv').open()));assert len(tasks)==30
for r in tasks:
 for n,v in summaries[r['family'],float(r['gamma']),r['arm']].items():check(v,r[n],n)
trans=list(csv.DictReader((a/'results/all_12_endpoint_transfers.csv').open()));assert len(trans)==12
for r in trans:
 fam,g,arm=r['family'],float(r['target_gamma']),r['arm'];best=summaries[fam,g,arm]['development_best_K'];other=summaries[fam,1-g,arm]['development_best_K'];minimum=summaries[fam,g,arm]['minimum']
 check(best,r['best_K'],'best');check(other,r['transfer_K'],'transfer');check(100*(lookup[fam,g,arm,other]['development_mean']/minimum-1),r['excess_pct'],'transfer cost')
table=(man/'cae_surface_audit.tex').read_text()
for fam,g in itertools.product(families,gammas):
 l,c=summaries[fam,g,'linear'],summaries[fam,g,'cae'];expected=f" & {g:g} & {l['development_best_K']} & {l['minimum']:.3f} & {l['K5_excess_pct']:.2f} & {c['development_best_K']} & {c['minimum']:.3f} & {c['K5_excess_pct']:.2f}"
 assert expected in table,expected
meta=json.loads((root/'.zenodo.json').read_text());assert meta['version']=='1.3.0'
assert 'version: 1.3.0' in (root/'CITATION.cff').read_text()
assert meta['doi'] == '10.5281/zenodo.24398249'
for text in [(root/'CITATION.cff').read_text(), (root/'README.md').read_text(), (man/'main.tex').read_text()]:assert meta['doi'] in text
report={'status':'passed','frozen_surface_rows':900,'candidate_means':180,'task_summaries':30,'endpoint_transfers':12,'main_table_rows':15,'new_training':False,'new_inference':False,'licensed_inputs_required_for_full_distance_recomputation':True,'archive_publication_status':json.loads((root/'PUBLICATION_STATUS.json').read_text())['status']}
print(json.dumps(report,indent=2))
if '--no-pdf' not in sys.argv:
 for main in ['main.tex','audit_supplement.tex']:subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error',main],cwd=man,check=True)
