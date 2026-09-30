#!/usr/bin/env python3
"""Render frozen post-publication controls; never train or bootstrap here."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from build_publication_figures import style, clean, save, BLUE, RED, MID, LIGHT, INK, KCOL, KMARK
from build_publication_tables import start, grid, row, panel, finish

DATA=Path(__file__).parent/'figure_data'


def architecture():
    d=pd.read_csv(DATA/'architecture_monthly_geometry.csv')
    s=pd.read_csv(DATA/'architecture_paired_summary.csv')
    groups=[(norm,label,layer) for norm,label in [('raw_centered','Raw centered'),
                ('feature_zscore','Feature z-score'),('row_l2_centered','Row L2')] for layer in [1,2,3]]
    fig,(ax,effect)=plt.subplots(1,2,figsize=(11.2,6.4),sharey=True,
                               gridspec_kw={'width_ratios':[2.15,1],'wspace':.17})
    xs=np.linspace(0,.72,500)
    for i,(norm,label,layer) in enumerate(groups):
        y=8-i
        if i%3==0:
            for a in [ax,effect]:a.axhspan(y-2.45,y+.45,color='#F4F7F9',zorder=-2)
        q=d[(d.normalization==norm)&(d.layer=='hidden'+str(layer))]
        for condition,color,sign in [('trained',BLUE,1),('random',MID,-1)]:
            v=q[q.condition==condition].rank_fraction.to_numpy()
            density=gaussian_kde(v)(xs);density=density/max(density)*.36
            ax.fill_between(100*xs,y,y+sign*density,color=color,alpha=.23,lw=0)
            ax.plot(100*xs,y+sign*density,color=color,lw=1)
            ax.plot([100*v.mean()]*2,[y,y+sign*.32],color=color,lw=1.8)
        t=s[(s.normalization==norm)&(s.layer==layer)&(s.metric=='rank_fraction')].iloc[0]
        m=100*t.paired_mean_difference;lo=100*t.paired_bootstrap_low;hi=100*t.paired_bootstrap_high
        effect.plot([lo,hi],[y,y],color=BLUE,lw=2)
        effect.scatter(m,y,color=BLUE,s=22,zorder=3)
        effect.text(3.2,y,f'{m:+.2f}',va='center',ha='right',fontsize=8,color=INK)
    ax.set_yticks(range(8,-1,-1),[f'{label}  |  H{layer} ({[128,64,32][layer-1]})' for _,label,layer in groups])
    ax.set_xlim(7,47);ax.set_ylim(-.6,8.6);ax.set_xlabel('Participation rank / layer width (%)')
    effect.axvline(0,color=INK,lw=.8);effect.set_xlim(-10,4);effect.set_xticks([-10,-5,0])
    effect.set_xlabel('Trained minus random (pp)');effect.tick_params(left=False,labelleft=False)
    for a in [ax,effect]:clean(a,'x');a.spines['left'].set_visible(False);a.spines['bottom'].set_position(('outward',5))
    ax.set_title('Distributions on identical monthly inputs',loc='left',pad=16)
    effect.set_title('Paired mean gap and 95% interval',loc='left',pad=16)
    fig.legend(handles=[Line2D([],[],color=BLUE,lw=3,label='Trained: 30 networks'),
                        Line2D([],[],color=MID,lw=3,label='Random: 5 unique hidden networks')],
               loc='upper center',bbox_to_anchor=(.57,.915),ncol=2)
    fig.suptitle('Training adds concentration beyond the narrowing architecture',fontsize=13,color=INK,fontweight='bold',y=.99)
    fig.text(.23,.035,'H1/H2/H3 widths: 128/64/32. Each row: 3,600 trained and 600 random model-months.\n'
             'Intervals resample common 12-month blocks and paired seeds; all K stay paired. Pointwise, conditional inference.',
             fontsize=7.3,color=MID)
    fig.subplots_adjust(left=.23,right=.98,top=.82,bottom=.16)
    save(fig,'figure_07_geometry_function')
    lines=start('Trained and random-network concentration on matched inputs','tab:architecture-control')
    lines += [grid('crrrrr'),r'\toprule',row('Layer','Width','Trained PR','Random PR',r'Gap / width (pp)',r'95\% interval'),r'\midrule']
    for j,(norm,label) in enumerate([('raw_centered','A. Raw centered'),('feature_zscore','B. Feature z-score'),('row_l2_centered',r'C. Row-$L_2$')]):
        if j:lines += [r'\addlinespace[4pt]']
        lines += [panel(6,label)]
        for layer in [1,2,3]:
            q=s[(s.normalization==norm)&(s.layer==layer)]
            a=q[q.metric=='participation_rank'].iloc[0];b=q[q.metric=='rank_fraction'].iloc[0]
            lines += [row(f'H{layer}',int(a.width),f'{a.trained_mean:.3f}',f'{a.random_mean:.3f}',
                f'{100*b.paired_mean_difference:.3f}',f'[{100*b.paired_bootstrap_low:.3f}, {100*b.paired_bootstrap_high:.3f}]')]
    lines += [r'\bottomrule']
    finish(lines,r'PR denotes participation rank. Entries are means, unlike the archived medians in Table~\ref{tab:geometry}. Width-normalized gaps are trained minus random. The primary endpoint is raw-centered H3; all other rows are descriptive. The 1,000 draws resample common twelve-month circular blocks and five paired seed clusters. Random hidden weights are shared across all six $K$ values. Intervals are pointwise and conditional on the archived sample and trained weights. This control was designed after the earlier study; it does not retune the economic-function probe.','architecture_control.tex')


def matched_features():
    d=pd.read_csv(DATA/'matched_seed_geometry_surface.csv')
    summary=pd.read_csv(DATA/'matched_mean_geometry_surface.csv')
    inference=pd.read_csv(DATA/'matched_paired_endpoint_inference.csv')
    keys=['asset_family','K','seed','gamma']
    full=d[d.arm=='full92'].drop(columns='arm');masked=d[d.arm=='masked86'].drop(columns='arm')
    paired=full.merge(masked,on=keys,suffixes=('_full','_masked'),validate='one_to_one')
    paired['gap']=paired.distance_masked-paired.distance_full
    # One common coordinate system: every curve is the same paired ablation.
    fig,ax=plt.subplots(figsize=(11.2,6.0))
    q=paired[paired.asset_family=='combined_74']
    for k in sorted(q.K.unique()):
        group=q[q.K==k]
        for seed,z in group.groupby('seed'):
            z=z.sort_values('gamma');ax.plot(z.gamma,z.gap,color=KCOL[k],lw=.7,alpha=.22)
        m=group.groupby('gamma').gap.mean().sort_index()
        ax.plot(m.index,m.values,color=KCOL[k],lw=2.2,marker=KMARK[k],markevery=[0,10,20],ms=5,label=f'K={k}')
    ax.axhline(0,color=INK,lw=1)
    ax.set_xlim(-.015,1.015);ax.set_xlabel('Geometry weight: equal moments (0) to HJ-type loss (1)')
    ax.set_ylabel('Pricing distance: masked 86 minus full 92')
    clean(ax,'both')
    ax.legend(ncol=6,loc='upper center',bbox_to_anchor=(.5,1.13),columnspacing=2)
    fig.suptitle('Same stocks, dates and initialization: the effect of masking six features',
                 y=.98,fontsize=12.5,fontweight='bold',color=INK)
    fig.text(.12,.025,'Combined 74 assets, 2010-2019. Thin curves: five paired seeds per K. Thick curves: seed means.\n'
             'Positive values mean masking worsens pricing. Both arms retain 184 inputs and identical hidden widths; only 12 input columns are zeroed.',
             fontsize=7.4,color=MID)
    fig.subplots_adjust(left=.12,right=.98,top=.79,bottom=.19)
    save(fig,'figure_08_matched_feature_control')
    lines=start('Matched six-feature ablation: pricing distances and paired uncertainty','tab:matched-features')
    lines += [grid('crrrrc'),r'\toprule',row('$K$','Full 92','Masked 86','Difference',r'95\% interval',r'$K^*$ frequency'),r'\midrule']
    for i,gamma in enumerate([0.,1.]):
        if i:lines += [r'\addlinespace[5pt]']
        lines += [panel(6,'A. Equal-moment endpoint' if i==0 else 'B. HJ-type endpoint')]
        for _,r in inference[inference.gamma==gamma].sort_values('K').iterrows():
            lines += [row(int(r.K),f'{r.full92:.3f}',f'{r.masked86:.3f}',f'{r.difference_masked_minus_full:+.3f}',
                f'[{r.ci95_low:.3f}, {r.ci95_high:.3f}]',f'{100*r.full92_bootstrap_winner_frequency:.1f} / {100*r.masked86_bootstrap_winner_frequency:.1f}')]
    lines += [r'\bottomrule']
    finish(lines,r'Combined 74 assets, January 2010--December 2019. Distances average five seeds; difference is masked minus full, so positive values favor retaining all features. Both arms use the same mother panel, ordered stock-month rows, ranks, target-month clock, and paired initial weights. The six characteristic columns and their missingness flags are zeroed only in the masked arm. Intervals use 1,000 common twelve-month block draws with paired seed resampling and fixed validation coefficients; the evaluation second moment is re-estimated within each draw. These are pointwise intervals conditional on fitted models. $K^*$ frequency lists full/masked bootstrap winner percentages, not posterior probabilities of a true dimension.','matched_feature_control.tex')
    # Full secondary grid is a readable long table in the supplement.
    lines=[r'\subsection{Complete matched-ablation endpoint grid}',r'\small',
           r'\begin{longtable}{@{}lcrrrr@{}}',r'\caption{Both endpoint distances in every fixed asset family.}\label{tab:s-matched-grid}\\',
           r'\toprule',row('Asset family','$K$',r'Full $\gamma=0$',r'Masked $\gamma=0$',r'Full $\gamma=1$',r'Masked $\gamma=1$'),
           r'\midrule\endfirsthead',r'\toprule',row('Asset family','$K$',r'Full $\gamma=0$',r'Masked $\gamma=0$',r'Full $\gamma=1$',r'Masked $\gamma=1$'),r'\midrule\endhead']
    for family,label in [('size_bm_25','Size--B/M 25'),('industry_49','Industry 49'),('combined_74','Combined 74')]:
        for k in [1,2,3,4,5,8]:
            cells=[label,k]
            for g in [0.,1.]:
                for arm in ['full92','masked86']:
                    z=summary[(summary.asset_family==family)&(summary.K==k)&(summary.gamma==g)&(summary.arm==arm)]
                    assert len(z)==1;cells.append(f'{z.iloc[0].mean_distance:.4f}')
            lines.append(row(*cells))
        lines.append(r'\addlinespace')
    lines += [r'\bottomrule\end{longtable}',r'\normalsize']
    (DATA.parents[1]/'manuscript/supplement/generated_matched_features.tex').write_text('\n'.join(lines)+'\n')


def main():
    style();architecture()
    if (DATA/'matched_seed_geometry_surface.csv').exists():matched_features()
    print('Rendered available frozen bounded controls.')


if __name__=='__main__':main()
