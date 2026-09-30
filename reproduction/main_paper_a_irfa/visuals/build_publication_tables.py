#!/usr/bin/env python3
"""Format frozen summaries as comparable manuscript tables; no estimation."""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parents[2]
DATA = Path(__file__).resolve().parent / 'figure_data'
OUT = ROOT / 'main_paper_a_irfa/manuscript/tables'
SOURCES = {}


def read(name):
    path = DATA / name
    SOURCES[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open() as f:
        return list(csv.DictReader(f))


def row(*values):
    return ' & '.join(str(v) for v in values) + r' \\'


def start(caption, label):
    return [r'\begin{widetable}[!tbp]', r'\centering',
            r'\begin{minipage}{\textwidth}',
            r'\caption{' + caption + '}', r'\label{' + label + '}',
            r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
            r'\renewcommand{\arraystretch}{1.16}']


def grid(spec):
    return r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}' + spec + r'@{}}'


def panel(n, label):
    return r'\multicolumn{' + str(n) + r'}{@{}l}{\textit{' + label + r'}} \\[2pt]'


def finish(lines, note, name):
    lines += [r'\end{tabular*}', r'\par\vspace{4pt}',
              r'{\footnotesize\raggedright\textit{Notes:} ' + note + r'\par}',
              r'\end{minipage}', r'\end{widetable}']
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text('\n'.join(lines) + '\n')


def dimension():
    d = read('table_01_geometry_surface_long.csv')
    lines = start('Pricing distances for every candidate dimension, archived development windows', 'tab:dimension-evidence')
    lines += [grid('crrrr'), r'\toprule',
              row('', r'\multicolumn{2}{c}{Core-92}', r'\multicolumn{2}{c}{Core-86}'),
              r'\cmidrule(lr){2-3}\cmidrule(l){4-5}',
              row('$K$', 'Equal moments', 'HJ', 'Equal moments', 'HJ'), r'\midrule']
    contexts = [('size_bm_25', 'A. 25 size--book-to-market portfolios'),
                ('industry_49', 'B. 49 industry portfolios'),
                ('combined_74', 'C. Combined 74 portfolios')]
    for i, (asset, label) in enumerate(contexts):
        if i: lines += [r'\addlinespace[5pt]']
        lines += [panel(5, label)]
        for k in [1, 2, 3, 4, 5, 8]:
            cells = [f'${k}$']
            for definition in ['Core-92', 'Core-86']:
                for gamma in [0, 1]:
                    group = [r for r in d if r['asset_family'] == asset and r['data_definition'] == definition
                             and float(r['gamma']) == gamma]
                    q = next(r for r in group if int(r['K']) == k)
                    best = min(group, key=lambda r: float(r['mean_distance']))
                    value = f"{float(q['mean_distance']):.3f}"
                    if q['K'] == best['K']: value = r'\textbf{' + value + '}'
                    cells.append(value + f" ({float(q['standard_error']):.3f})")
            lines += [row(*cells)]
    lines += [r'\bottomrule']
    finish(lines, r'Each entry is a mean pricing distance over five fixed training seeds; parentheses give the standard error across seeds. Lower is better. Boldface marks the smallest mean within each asset set, data definition, and loss geometry. Seed standard errors describe algorithmic dispersion, not time-series sampling uncertainty. Core-92 return months are February 2010--January 2020; Core-86 return months are January 2010--December 2019. Construction and coverage also differ. All six candidates are shown, including those that never win.', 'dimension_comparison.tex')


def sealed():
    d = read('sealed_table_01_family_inference.csv')
    lookup = {r['asset_family']: r for r in d}
    # Asset labels come from the frozen file; preserve its actual family keys.
    short = {'25 Size–B/M':'Size--B/M','49 Industries':'Industries','Combined 74':'Combined 74',
             '25 Size–OP':'Size--OP','25 Size–Inv':'Size--Inv','25 Size–Mom':'Size--Mom',
             '25 Size–Accruals':'Size--Accruals','25 Size–Beta':'Size--Beta',
             '25 Size–Residual variance':'Size--Residual var.'}
    ordered = sorted(d, key=lambda r: (r['asset_family'] not in ['size_bm_25','industry_49','combined_74'],
                                     list(lookup).index(r['asset_family'])))
    lines = start('Pricing dimensions and economic diagnostics in the locked 2020--2025 evaluation', 'tab:sealed-evidence')
    lines += [grid('lccrrrr'), r'\toprule',
              row('Test assets', r'\multicolumn{2}{c}{Best $K$}', 'Geometry', 'Market', r'\multicolumn{2}{c}{Completion (pp)}'),
              r'\cmidrule(lr){2-3}\cmidrule(l){6-7}',
              row('', 'EW','HJ','contrast $A$','attenuation','Raw moments','Alpha'),r'\midrule']
    for i, q in enumerate(ordered):
        if i in [0,3]:
            if i: lines += [r'\addlinespace[4pt]']
            lines += [panel(7, 'Primary assets' if i == 0 else 'External assets')]
        label = short.get(q['asset_label'], q['asset_label'].replace('–','--'))
        lines += [row(label,q['best_K_gamma0'],q['best_K_gamma1'],
                      f"{float(q['A_raw']):.2f}",f"{float(q['attenuation']):.2f}",
                      f"{100*float(q['completion_raw']):.2f}",f"{100*float(q['completion_alpha']):.2f}")]
        cells=['','','']
        for low,high,scale in [('A_low','A_high',1),('atten_low','atten_high',1),('raw_low','raw_high',100),('alpha_low','alpha_high',100)]:
            cells.append(f"[{float(q[low])*scale:.2f}, {float(q[high])*scale:.2f}]")
        lines += [row(*cells)]
    lines += [r'\bottomrule']
    finish(lines, r'Brackets below each diagnostic are 95\% twelve-month circular-block bootstrap intervals (500 draws). Positive market attenuation means hedging reduces the absolute geometry contrast. Completion is the error of $K=1$ plus the market factor minus the error of $K=1$, multiplied by 100; negative values indicate improvement. EW and HJ denote the equal-moment and covariance-weighted endpoints. All rows use the frozen Core-86 models.', 'sealed_comparison.tex')


def simulation():
    d = read('identification_table_02_mechanism_aggregates.csv')
    h = read('identification_hypothesis_summary.csv')
    labels = {'evaluation':'Evaluation','months':'Sample months','signal':'Risk prices','spanning':'Asset spanning',
              'true_dimension':'True dimension','asset_count':'Test assets','rho':'Factor correlation','geometry':'Pricing geometry'}
    levels = {'feasible':'Feasible','oracle':'Oracle','strong':'Strong','weak':'Weak','full':'Full',
              'weak_tail':'Weak tail','equal':'Equal moments','hj':'HJ'}
    lines = start('Known-truth dimension selection: outcomes and prespecified mechanism contrasts', 'tab:sim-mechanisms')
    lines += [grid('llrrrr'),r'\toprule',panel(6,'A. Marginal selection outcomes'),
              row('Design axis','Level',r'Exact (\%)',r'Under (\%)',r'Over (\%)','Mean error'),r'\midrule']
    previous = None
    for q in d:
        first = q['mechanism'] != previous
        if first and previous: lines += [r'\addlinespace[2pt]']
        lines += [row(labels[q['mechanism']] if first else '', levels.get(q['level'],q['level']),
                      *(f"{100*float(q[c]):.1f}" for c in ['exact_rate','under_rate','over_rate']),
                      f"{float(q['mean_absolute_error']):.3f}")]
        previous=q['mechanism']
    lines += [r'\bottomrule',r'\end{tabular*}',r'\par\vspace{7pt}',
              grid('lrrr'),r'\toprule',panel(4,'B. Prespecified mechanism contrasts'),
              row('Contrast and outcome','Cells','Effect (pp)',r'95\% cell interval (pp)'),r'\midrule']
    effect_labels=['More time: exact recovery$^{a}$','Strong minus weak prices: exact recovery',
                   'Weak minus strong prices: under-selection','Full minus weak-tail span: exact recovery',
                   'Weak-tail minus full span: asset-family divergence','Equal--HJ disagreement: level$^{b}$',
                   'Complex minus baseline designs: disagreement$^{c}$','Oracle minus feasible evaluation: exact recovery']
    for q,label in zip(h,effect_labels):
        lines += [row(label,q['matched_cells'],f"{100*float(q['mean_difference']):.1f}",
                      f"[{100*float(q['ci_low']):.1f}, {100*float(q['ci_high']):.1f}]")]
    lines += [r'\bottomrule']
    finish(lines, r'Panel A averages the remaining frozen design axes; its rows do not hold every other axis fixed. Exact, under-, and over-selection sum to 100\% before rounding; mean error is $|\widehat K-K_0|$. Oracle over-selection is zero by construction. Panel B uses percentage points (pp). Its archived intervals are the mean $\pm1.96$ standard errors across design-cell contrasts; they summarize cell heterogeneity, not uncertainty from independent Monte Carlo draws. $^{a}$600 minus 72 months, restricted to strong prices, full span, and $\rho=0$. $^{b}$An average level, not a difference. $^{c}$Correlation and/or weak-tail designs relative to full spanning with $\rho=0$. There are 500 Monte Carlo repetitions per base condition; six asset families share each repetition.', 'simulation_comparison.tex')


def temporal():
    d=read('temporal_decomposition.csv'); p=read('temporal_portfolio.csv')
    lines=start('Forecast updating and portfolio performance by learner and update rule','tab:temporal-economic')
    lines += [grid('llrrrrrr'),r'\toprule',
              row('','',r'\multicolumn{2}{c}{Updated minus frozen loss}', r'\multicolumn{4}{c}{Value-weighted portfolios; 25 bp cost}'),
              r'\cmidrule(lr){3-4}\cmidrule(l){5-8}',
              row('Learner','Rule','2000--09','2010--19',r'Return (\%)','Sharpe','Turnover',r'CE (\%)'),r'\midrule']
    for model in ['Neural','Ridge','Tree']:
        if model!='Neural': lines += [r'\addlinespace[5pt]']
        for update in ['Frozen','Rolling 60','Expanding']:
            q=next(r for r in p if r['model']==model and r['update']==update)
            losses=[]
            for period in ['2000-2009','2010-2019']:
                if update=='Frozen': losses.append('0.000')
                else:
                    z=next(r for r in d if r['model']==model and r['update']==update and r['period']==period)
                    losses.append(f"{1000*float(z['loss_diff']):.3f}")
            lines += [row(model if update=='Frozen' else '',update,*losses,
                          f"{100*float(q['net_return']):.2f}",f"{float(q['sharpe']):.3f}",
                          f"{float(q['turnover']):.3f}",f"{100*float(q['ce_gamma5']):.2f}")]
    lines += [r'\bottomrule']
    finish(lines, r'Loss differences are multiplied by $10^3$; negative values favor updating. Zero for the frozen rule is the comparison baseline. Portfolio return and certainty equivalent (CE, risk aversion five) are annualized percentages over February 2000--December 2019. Turnover is monthly for unit long and unit short positions after return drift. Portfolio columns cover the full period, not the individual decades in the loss columns. These are separate return-prediction learners, not the pricing-dimension networks. None of the six update-minus-frozen return differences passes the Holm-adjusted 5\% gate.', 'temporal_comparison.tex')


def geometry():
    path=ROOT/'experiments/P7-G1-V001/P7-G1-V001-REPORT001/result_summary.json'
    SOURCES[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
    d=json.loads(path.read_text())
    lines=start('Layerwise representation geometry under three normalization rules','tab:geometry')
    lines += [grid('lrrrrrrr'),r'\toprule',
              row('Layer','Participation','Entropy','PCA95','TWO-NN','LB ($k=20$)','Tangent angle','Drift'),
              row('','rank','rank','rank','dimension','dimension','(radians)','(rank 5)'),r'\midrule']
    for i,(norm,label) in enumerate([('raw_centered','A. Raw centering'),('feature_zscore','B. Feature z-score'),('row_l2_centered',r'C. Row-$L_2$ normalization')]):
        if i: lines += [r'\addlinespace[5pt]']
        lines += [panel(8,label)]
        for layer in range(1,4):
            key=f'hidden{layer}|{norm}'; q=d['geometry_summary_medians'][key]
            cells=[f'Hidden {layer}']
            for col in ['participation_rank','entropy_effective_rank','pca95_rank','twonn_dimension','mle20_dimension','curvature_proxy_radians']:
                value=q[col]
                cells.append('--' if value is None else (f'{value:.0f}' if col=='pca95_rank' else f'{value:.3f}'))
            cells.append(f"{d['drift_medians'][key]:.3f}")
            lines += [row(*cells)]
    lines += [r'\bottomrule']
    finish(lines, r'Entries are archived medians for 30 frozen networks and 120 development months. Spectral summaries use 3,600 observations per group; drift uses 3,570 adjacent-month pairs. Neighbor dimensions and tangent angles use only 300 observations per raw/z-score group (ten sampled months). LB is the Levina--Bickel estimator. Dashes denote diagnostics not computed under row normalization, not zero values. The numerical-collapse rate is zero; it is not a classification neural-collapse test. The two adjacent-layer CKA medians are 0.875 and 0.848. These are descriptive, finite-scale diagnostics; the separate economic-function test is reported in Figure~\ref{fig:evidence-ladder} and Section~\ref{sec:function-gate}.', 'geometry_comparison.tex')


def main():
    dimension(); sealed(); simulation(); temporal(); geometry()
    outputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*_comparison.tex'))}
    (OUT/'table_manifest.json').write_text(json.dumps({'purpose':'Presentation of frozen summaries; no new estimation',
                                                    'sources':SOURCES,'outputs':outputs},indent=2)+'\n')
    print(f'Formatted {len(outputs)} tables from {len(SOURCES)} frozen inputs.')


if __name__=='__main__':
    main()
