#!/usr/bin/env python3
"""Reformat archived aggregate reports; no estimation or model execution."""
from pathlib import Path
import csv,json,hashlib
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'visuals/figure_data';SRC=ROOT/'audit/source_evidence/temporal'
def read(e):return json.loads((SRC/f'{e}_report.json').read_text())
def write(name,rows):
 p=DATA/name
 if not (SRC/('transcribed_'+name)).exists(): (SRC/('transcribed_'+name)).write_bytes(p.read_bytes())
 old=list(csv.DictReader(p.open()))
 with p.open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 errors=[]
 for a,b in zip(old,rows):
  for k,v in b.items():
   if isinstance(v,(float,int)) and abs(float(a[k])-v)>1e-3:errors.append([k,a[k],v])
 return {'file':name,'rows':len(rows),'gross_transcription_mismatches':errors,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
n=read('P3-TOP-M002')['summary'];r=read('P3-TOP-MR001')['summary'];b=read('P3-TOP-BREAK001')['summaries']['tree'];rows=[]
for model,source,periods in [('Neural',n,['development_2000_2009','promotion_2010_2019']),('Ridge',r,['2000_2009','2010_2019']),('Tree',b,['development_2000_2009','promotion_2010_2019'])]:
 for update,arm in [('Rolling 60','rolling60'),('Expanding','expanding')]:
  for period,label in zip(periods,['2000-2009','2010-2019']):
   z=source[arm][period];rows.append(dict(model=model,update=update,period=label,A=z['adjustment_cost_A'],twoB=z['alignment_benefit_2B'],loss_diff=z['updated_minus_frozen_loss'],lambda_star=z.get('ex_post_lambda_star_from_mean_terms',z.get('ex_post_lambda_star'))))
out=[write('temporal_decomposition.csv',rows)];d=read('P3-TOP-PORT001')['portfolio_summary'];rows=[]
for model,key in [('Neural','neural'),('Ridge','ridge'),('Tree','tree')]:
 for update,arm in [('Frozen','frozen'),('Rolling 60','rolling60'),('Expanding','expanding')]:
  q=d[key][arm]['value']['25'];rows.append(dict(model=model,update=update,net_return=q['annualized_mean'],sharpe=q['annualized_sharpe'],turnover=q['mean_turnover'],ce_gamma5=q['annualized_certainty_equivalent_gamma5']))
out.append(write('temporal_portfolio.csv',rows))
(SRC/'exhibit_reformat_audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))

b=read('P3-TOP-BREAK001')['common_break'];q=read('P3-TOP-SEQ001')['detector']
def month(x):return f'{int(x)//100:04d}-{int(x)%100:02d}'
rows=[dict(event='retrospective_break',date=month(b['estimated_first_month_after_break']),label='Retrospective common break',status='exploratory'),dict(event='realtime_trigger',date=month(q['crossing_month']),label='Lagged CUSUM crossing',status='internal_failed_gate'),dict(event='policy_active',date=month(q['first_active_month']),label='Updating starts next month',status='internal_failed_gate')]
out.append(write('temporal_events.csv',rows))
(SRC/'exhibit_reformat_audit.json').write_text(json.dumps(out,indent=2)+'\n')
manifest=json.loads((DATA/'figure_data_manifest.json').read_text())
for x in manifest['items']:
 if x['file'].startswith('temporal_'):
  x.pop('manual_transcription_checked_against_verified_reports',None);x.pop('source_reports',None)
  x['derivation']='Automatic extraction by audit/rebuild_temporal_exhibit_data.py from archived aggregate JSON reports'
  x['source_reports']=[{'path':str(p.relative_to(ROOT)),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(SRC.glob('*_report.json'))]
  x['copy_sha256']=hashlib.sha256((DATA/x['file']).read_bytes()).hexdigest()
manifest['updated_at']='2026-09-30'
(DATA/'figure_data_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
