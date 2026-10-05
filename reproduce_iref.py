"""Public exhibit checks and typesetting only; no new statistical experiment."""
from pathlib import Path
import csv, gzip, hashlib, json, math, subprocess, sys
ROOT=Path(__file__).resolve().parent
REPRO=ROOT/'reproduction'
checks=0
for line in (ROOT/'SHA256SUMS').read_text().splitlines():
    h,n=line.split('  ',1)
    assert hashlib.sha256((ROOT/n).read_bytes()).hexdigest()==h,n
    checks+=1
v3=REPRO/'experiments/P1-G4-V003/P1-G4-V003-R001/analysis'
v4=REPRO/'experiments/P1-G4-V004/P1-G4-V004-R001'
paper=REPRO/'main_paper_a_iref/manuscript'
rows=list(csv.DictReader((v3/'dimension_summary.csv').open()))
table=(paper/'cae_candidate_comparison.tex').read_text()
for k in [1,2,3,4,5,8]:
    values=[]
    for arm in ['linear','cae']:
        r=next(r for r in rows if r['arm']==arm and int(r['K'])==k)
        values.extend([f"{100*float(r['predictive_R2']):.4f}",f"{float(r['HJ_distance']):.4f}",f"{100*float(r['development_winner_frequency']):.2f}"])
    assert str(k)+' & '+' & '.join(values) in table,(k,values)
outer=json.loads(gzip.decompress((v4/'outer_results.json.gz').read_bytes()))
summary=list(csv.DictReader((v4/'calibration_summary.csv').open()))
assert len(outer)==len(summary)==30
table=(paper/'inference_calibration_appendix.tex').read_text()
for row in summary:
    data=outer[row['id']];assert len(data)==400
    for key in ['simultaneous_coverage','false_rejection_any_true_null','K8_bootstrap_winner_frequency','critical_value','mean_interval_width','mean_point_bias']:
        mean=sum(r[key] for r in data)/400
        assert math.isclose(mean,float(row[key]),rel_tol=1e-12,abs_tol=1e-12),(row['id'],key)
    if float(row['gap'])==0:
        vals=[row['months'],row['moment_rank'],f"{float(row['base_distance']):g}",f"{float(row['phi']):.1f}",f"{100*float(row['simultaneous_coverage']):.2f}",f"{100*float(row['false_rejection_any_true_null']):.2f}"]
        assert ' & '.join(vals) in table,vals
print(json.dumps({'verified_files':checks,'candidate_table_numeric_cells':36,'simulation_scenarios':30,'outer_replications':12000,'calibration_table_rows':10,'statistical_calibration_status':'failed_as_reported'}))
if '--no-pdf' not in sys.argv:
    subprocess.run(['latexmk','-pdf','-interaction=nonstopmode','-halt-on-error','main.tex'],cwd=paper,check=True)
    print('Current authored article: '+str(paper/'main.pdf'))
