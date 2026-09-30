#!/usr/bin/env python3
"""Validate hashes, dimensions, decision rules, and formats for P1-G2-V001."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];EXP=ROOT/'experiments/P1-G2-V001';RUN=EXP/'outputs/P1-G2-V001-R001';ART=ROOT/'paper1_sealed_artifacts'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 d=json.loads((RUN/'sealed_confirmation.json').read_text());inf=json.loads((RUN/'sealed_factors/inference_manifest.json').read_text())
 assert d['sealed_period_accessed'] is True and inf['sealed_period_accessed'] is True
 assert inf['months']==72 and len(inf['models'])==30 and inf['rows']==286565
 assert d['bootstrap']['draws']==500 and d['bootstrap']['block_months']==12 and len(d['family_results'])==9
 support={k:v['supported'] for k,v in d['preregistered_hypotheses'].items()};assert support=={'H1_geometry_dependence':False,'H2_market_direction':True,'H3_economic_completion':False,'H4_asset_family_dependence':False}
 assert len(list((ART/'tables').glob('*.csv')))==6 and len(list((ART/'tables').glob('*.tex')))==6
 assert len(list((ART/'figures').glob('*.pdf')))==6 and len(list((ART/'figures').glob('*.svg')))==6 and len(list((ART/'figures').glob('*.png')))==6
 for p in (ART/'tables').glob('*.csv'):assert len(pd.read_csv(p))>0
 inputs=[RUN/'sealed_confirmation.json',RUN/'sealed_factors/inference_manifest.json',ROOT/'data/processed/wrds-us-equity-2025-12-v1/P1-G2-V001/quality_report.json',ROOT/'data/processed/wrds-us-equity-2025-12-v1/P1-G2-V001/output_manifest.json']
 arts=sorted(p for p in ART.rglob('*') if p.is_file() and p.name!='artifact_manifest.json')
 manifest={'schema_version':1,'experiment_id':'P1-G2-V001','sealed_period_accessed':True,'inputs':[{'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in inputs],'artifacts':[{'path':str(p.relative_to(ROOT)),'size_bytes':p.stat().st_size,'sha256':sha(p)} for p in arts]}
 (ART/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print(json.dumps({'experiment_id':'P1-G2-V001','validated':True,'models':30,'months':72,'bootstrap_draws':500,'tables':6,'figures':6,'hypotheses':support},indent=2))
if __name__=='__main__':main()
