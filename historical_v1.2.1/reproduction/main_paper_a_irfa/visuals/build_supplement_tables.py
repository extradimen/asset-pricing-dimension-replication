#!/usr/bin/env python3
"""Render archived long tables without new estimation."""
from pathlib import Path
import csv,json
ROOT=Path(__file__).resolve().parents[2]
DATA=ROOT/'main_paper_a_irfa/visuals/figure_data'
OUT=ROOT/'main_paper_a_irfa/manuscript/supplement'
def read(p):return list(csv.DictReader(p.open()))
def begin(spec,caption,label,header):
 return ['{\\small\\setlength{\\tabcolsep}{4pt}',r'\begin{longtable}{@{}'+spec+'@{}}',r'\caption{'+caption+r'}\label{'+label+r'}\\',r'\toprule',header+r'\\\midrule\endfirsthead',r'\multicolumn{'+str(len(spec))+'}{l}{'+caption+r' (continued)}\\\toprule',header+r'\\\midrule\endhead',r'\bottomrule\endfoot']
def end(lines,name):
 (OUT/name).write_text('\n'.join(lines+[r'\end{longtable}',r'}'])+'\n')
f=read(ROOT/'paper1_artifacts/tables/table_12_core86_feature_dictionary.csv')
l=begin('rlll','Core-86 characteristic membership','tab:dictionary','No. & Characteristic & Component & Input treatment')
for i,q in enumerate(f,1):
 label=q['group_label'].replace('Monthly/market','Monthly/market')
 l.append(f"{i} & \\texttt{{{q['feature'].replace('_',r'\_')}}} & {label} & Rank + missing flag"+r'\\')
end(l,'generated_feature_dictionary.tex')
d=read(DATA/'identification_condition_summary.csv')
l=begin('rrrllllrrr','Complete known-truth selection frequencies','tab:simlong',r'$K_0$ & $T$ & $N$ & Prices & Span & $\rho$ & Metric/eval. & Exact & Under & Over')
for q in d:
 l.append(' & '.join([q['true_dimension'],q['months'],q['asset_count'],q['signal'],q['spanning'].replace('weak_tail','weak'),q['rho'],q['geometry']+'/'+('O' if q['evaluation']=='oracle' else 'F')]+[f"{100*float(q[k]):.1f}" for k in ['exact_rate','under_rate','over_rate']])+r'\\')
end(l,'generated_simulation_long.tex')
d=read(DATA/'temporal_decomposition.csv')
l=begin('lllrrrr','Temporal update-loss components (scaled by 1,000)','tab:decomposition',r'Learner & Rule & Period & $A$ & $2B$ & $A-2B$ & $\lambda^*$')
for q in d:l.append(' & '.join([q['model'],q['update'],q['period']]+[f"{1000*float(q[k]):.4f}" for k in ['A','twoB','loss_diff']]+[f"{float(q['lambda_star']):.4f}"])+r'\\')
l += [r'\end{longtable}',r'}']
p=json.loads((ROOT/'main_paper_a_irfa/audit/source_evidence/temporal/P3-TOP-PORT001_report.json').read_text())['portfolio_summary']
l+=begin('lllrrrr','Portfolio performance across weighting and cost specifications','tab:portlong',r'Learner & Rule/weight & Cost (bp) & Return (\%) & Sharpe & Turnover & CE (\%)')
for m,arms in p.items():
 for a,weights in arms.items():
  for w,costs in weights.items():
   for c,q in costs.items():
    l.append(' & '.join([m,a+'/'+('VW' if w=='value' else 'EW'),c,f"{100*q['annualized_mean']:.2f}",f"{q['annualized_sharpe']:.3f}",f"{q['mean_turnover']:.3f}",f"{100*q['annualized_certainty_equivalent_gamma5']:.2f}"])+r'\\')
end(l,'generated_temporal_tables.tex')
print('Supplement tables: 86 characteristics, 576 simulation rows, 12 decomposition and 72 portfolio rows.')
