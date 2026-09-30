#!/usr/bin/env python3
"""Build the manuscript's unified publication figures from frozen summaries.

This script changes presentation only. It reads copied, hash-traceable summary
tables and never estimates a model, changes a specification, or selects a result.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(__file__).resolve().parent / "figure_data"
OUT = ROOT / "manuscript" / "figures"

BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
RED = "#D55E00"
PURPLE = "#CC79A7"
SKY = "#56B4E9"
YELLOW = "#F0E442"
INK = "#243447"
MID = "#66788A"
LIGHT = "#D9E1E8"
PALE = "#F4F7F9"
KS = [1, 2, 3, 4, 5, 8]
KCOL = {1: BLUE, 2: SKY, 3: GREEN, 4: ORANGE, 5: RED, 8: PURPLE}
KMARK = {1: "o", 2: "s", 3: "^", 4: "D", 5: "P", 8: "X"}


def style() -> None:
    mpl.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.titlesize": 10.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9,
        "axes.labelcolor": INK,
        "axes.edgecolor": MID,
        "axes.linewidth": 0.7,
        "xtick.color": MID,
        "ytick.color": MID,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.frameon": False,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.hashsalt": "main-paper-a-2026-09-28",
    })


def clean(ax, grid="y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid, color=LIGHT, linewidth=0.55, alpha=0.75)
    ax.set_axisbelow(True)


def panel(ax, letter: str, title: str) -> None:
    ax.text(0.0, 1.045, f"{letter}   {title}", transform=ax.transAxes,
            fontsize=9.2, fontweight="bold", color=INK, va="bottom", linespacing=1.08)


def metadata_for(ext: str) -> dict:
    if ext == "pdf":
        fixed = datetime(2026, 9, 28, tzinfo=timezone.utc)
        return {"Creator": "Main Paper A figure builder",
                "CreationDate": fixed, "ModDate": fixed}
    if ext == "svg":
        return {"Creator": "Main Paper A figure builder", "Date": "2026-09-28"}
    return {"Software": "Main Paper A figure builder"}


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=360 if ext == "png" else None,
                    bbox_inches="tight", pad_inches=0.04, metadata=metadata_for(ext))
    plt.close(fig)


def evidence_ladder() -> None:
    """Out-of-seed pricing predictions with and without hidden-state geometry."""
    d = pd.read_csv(DATA / "p7_model_evidence_matrix.csv")
    summary = json.loads((DATA / "p7_canonical_return_summary.json").read_text())
    probe = {item["endpoint"]: item for item in summary["functional_probes"]}

    def predictions(endpoint: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        factor=d.factor_count.to_numpy(); seeds=d.seed.to_numpy()
        x=d.hidden3_participation_rank.to_numpy(); y=d[endpoint].to_numpy()
        baseline=np.empty(len(d)); geometry=np.empty(len(d))
        for held_seed in np.unique(seeds):
            train=seeds != held_seed; test=np.flatnonzero(~train)
            residual_x=x.copy(); residual_y=y.copy()
            for value in np.unique(factor):
                mask=train & (factor == value)
                residual_x[mask] -= x[mask].mean()
                residual_y[mask] -= y[mask].mean()
            slope=np.dot(residual_x[train],residual_y[train]) / max(np.dot(residual_x[train],residual_x[train]),1e-12)
            for idx in test:
                same=train & (factor == factor[idx])
                baseline[idx]=y[same].mean()
                geometry[idx]=baseline[idx] + slope * (x[idx]-x[same].mean())
        return y,baseline,geometry

    fig,axes=plt.subplots(1,2,figsize=(11.2,5.8))
    specs=[
        ("development_hj_loss","A","Primary endpoint: HJ loss",False),
        ("development_factor_sharpe","B","Secondary endpoint: factor Sharpe",True),
    ]
    for ax,(endpoint,letter,title,higher_better) in zip(axes,specs):
        actual,base,geo=predictions(endpoint)
        lower=min(actual.min(),base.min(),geo.min()); upper=max(actual.max(),base.max(),geo.max())
        pad=(upper-lower)*.09; lo=lower-pad; hi=upper+pad
        ax.plot([lo,hi],[lo,hi],color=INK,lw=1.0,zorder=0)
        improved=np.abs(actual-geo) < np.abs(actual-base)
        for i in range(len(d)):
            arrow_color=GREEN if improved[i] else RED
            ax.annotate("",xy=(actual[i],geo[i]),xytext=(actual[i],base[i]),
                        arrowprops=dict(arrowstyle="->",color=arrow_color,lw=.9,alpha=.55),zorder=1)
            ax.scatter(actual[i],base[i],s=23,facecolors="white",edgecolors=MID,lw=.8,zorder=2)
            k=int(d.factor_count.iloc[i])
            ax.scatter(actual[i],geo[i],s=34,color=KCOL[k],marker=KMARK[k],
                       edgecolors="white",lw=.45,zorder=3)
        p=probe[endpoint]
        rmse0=float(np.sqrt(np.mean((actual-base)**2))); rmse1=float(np.sqrt(np.mean((actual-geo)**2)))
        gain=100*(rmse0-rmse1)/rmse0
        direction="higher" if higher_better else "lower"
        ax.text(.03,.965,
                f"K-only RMSE {rmse0:.3f}  →  + geometry {rmse1:.3f}\n"
                f"gain {gain:.3f}%  ·  p={p['permutation_p_two_sided']:.3f}  ·  {improved.sum()}/30 arrows move closer",
                transform=ax.transAxes,va="top",fontsize=7.4,color=INK,
                bbox=dict(boxstyle="round,pad=.35",fc="white",ec=LIGHT,lw=.7,alpha=.94))
        ax.text(.97,.04,f"Observed performance ({direction} is better)",transform=ax.transAxes,
                ha="right",fontsize=6.8,color=MID)
        ax.set_xlim(lo,hi); ax.set_ylim(lo,hi); ax.set_aspect("equal",adjustable="box")
        ax.set_xlabel(f"Observed {title.split(': ')[1]}")
        ax.set_ylabel("Leave-one-seed-out prediction")
        clean(ax,"both"); panel(ax,letter,title)
    handles=[]
    for k in KS:
        handles.append(mpl.lines.Line2D([],[],color=KCOL[k],marker=KMARK[k],lw=0,ms=5,label=f"K={k}"))
    handles.extend([
        mpl.lines.Line2D([],[],color=MID,marker="o",mfc="white",lw=0,ms=4,label="K-only prediction"),
        mpl.lines.Line2D([],[],color=GREEN,lw=1.2,label="geometry improves error"),
        mpl.lines.Line2D([],[],color=RED,lw=1.2,label="geometry worsens error"),
    ])
    fig.legend(handles=handles,ncol=9,loc="upper center",bbox_to_anchor=(.5,.895),fontsize=7.0,
               handletextpad=.35,columnspacing=.8)
    fig.suptitle("Hidden-state rank does not reliably transport to out-of-seed pricing performance",
                 y=.99,fontsize=13,fontweight="bold",color=INK)
    fig.text(.985,.02,
             "Each arrow is one frozen network. Open circle: prediction from nominal K. Colored endpoint: prediction after adding hidden-3 participation rank.",
             ha="right",fontsize=7.4,color=MID)
    fig.subplots_adjust(top=.80,bottom=.13,left=.09,right=.985,wspace=.24)
    save(fig, "figure_01_evidence_ladder")


def graphical_abstract() -> None:
    """A concise summary of distinct measurements, without a compression funnel."""
    d = pd.read_csv(DATA / "architecture_paired_summary.csv")
    q = d[(d.normalization == 'raw_centered') & (d.metric == 'rank_fraction')].sort_values('layer')
    fig = plt.figure(figsize=(12.5, 5.0))
    ax = fig.add_axes([.075, .24, .38, .49])
    widths = [128, 64, 32]
    for field, label, color, marker in [('random_mean', 'Matched random network', MID, 's'),
                                       ('trained_mean', 'Trained network', BLUE, 'o')]:
        ax.plot([1,2,3], 100*q[field], marker=marker, color=color, lw=2, label=label)
    ax.set_xticks([1,2,3], [f'Hidden {i+1}\nwidth {w}' for i,w in enumerate(widths)])
    ax.set_ylabel('Participation rank / layer width (%)')
    ax.set_ylim(10,38)
    clean(ax)
    ax.legend(loc='upper left',fontsize=8)
    ax.set_title('Architecture-matched representation control',loc='left',fontsize=11,pad=14)
    fig.text(.55,.715,'IDENTIFICATION',fontsize=10,color=INK,weight='bold')
    fig.text(.55,.655,'Dimension rankings depend on the evaluation geometry.',fontsize=10,color=INK)
    fig.text(.55,.605,'Matched feature removal does not reproduce the archived rank reversal.',fontsize=8.7,color=MID)
    fig.text(.55,.495,'PRICING FUNCTION',fontsize=10,color=INK,weight='bold')
    fig.text(.55,.435,'Adding hidden-state rank: 3.915% HJ-loss RMSE gain',fontsize=10,color=INK)
    fig.text(.55,.385,'Below the specified 5% threshold; permutation p = 0.149.',fontsize=9,color=MID)
    fig.text(.55,.275,'INTERPRETATION',fontsize=10,color=INK,weight='bold')
    fig.text(.55,.215,'Concentration alone does not identify an invariant pricing dimension.',fontsize=9.4,color=INK)
    fig.text(.05,.92,'Low-dimensional representation and economic identification',fontsize=18,color=INK,weight='bold')
    fig.text(.05,.105,'Training adds concentration beyond random initialization. Absolute rank decline also reflects narrowing layer width.',fontsize=10,color=INK)
    fig.text(.05,.055,'Later controls are exploratory; historical evaluation is not an untouched research-wide holdout.',fontsize=9,color=MID)
    OUT.mkdir(parents=True,exist_ok=True)
    with mpl.rc_context({'savefig.bbox':None}):
        for ext in ('pdf','svg','png'):
            fig.savefig(OUT/f'graphical_abstract.{ext}',dpi=200 if ext=='png' else None,bbox_inches=None,metadata=metadata_for(ext))
    plt.close(fig)


def dimension_sensitivity() -> None:
    d = pd.read_csv(DATA / "table_01_geometry_surface_long.csv")
    seeds = pd.read_csv(DATA / "table_02_geometry_seed_long.csv")
    contexts = [
        ("Core-92", "size_bm_25", "25 Size–B/M  ·  Core-92"),
        ("Core-86", "size_bm_25", "25 Size–B/M  ·  Core-86"),
        ("Core-92", "industry_49", "49 Industries  ·  Core-92"),
        ("Core-86", "industry_49", "49 Industries  ·  Core-86"),
        ("Core-92", "combined_74", "Combined 74  ·  Core-92"),
        ("Core-86", "combined_74", "Combined 74  ·  Core-86"),
    ]
    gammas = np.sort(d.gamma.unique())
    matrix = np.zeros((len(contexts) * len(KS), len(gammas)))
    mean_winner_paths = []
    vote_points = []

    for block, (definition, asset, label) in enumerate(contexts):
        q = d[(d.data_definition == definition) & (d.asset_family == asset)]
        pivot = q.pivot(index="K", columns="gamma", values="mean_distance").loc[KS, gammas]
        best = pivot.min(axis=0)
        matrix[block*6:(block+1)*6, :] = 100 * (pivot.to_numpy() / best.to_numpy() - 1)
        winners = pivot.idxmin(axis=0).map({k:i for i,k in enumerate(KS)}).to_numpy() + block*6
        mean_winner_paths.append(winners)

        sq = seeds[(seeds.data_definition == definition) & (seeds.asset_family == asset)]
        seed_winners = (sq.loc[sq.groupby(["gamma","seed"]).distance.idxmin()]
                          .groupby(["gamma","K"]).size().rename("votes").reset_index())
        for _, row in seed_winners.iterrows():
            vote_points.append((np.where(np.isclose(gammas, row.gamma))[0][0],
                                block*6 + KS.index(int(row.K)), int(row.votes)))

    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        "dimension_regret", ["#08306B", "#2171B5", "#6BAED6", "#D9EAF4",
                             "#FFF4D6", "#F6B04A", "#C74A00"])
    norm = mpl.colors.PowerNorm(gamma=.62, vmin=0, vmax=50, clip=True)
    fig, ax = plt.subplots(figsize=(11.4, 8.0))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)

    for block, path in enumerate(mean_winner_paths):
        line, = ax.plot(np.arange(len(gammas)), path, color="white", lw=2.4,
                        marker="D",ms=3.3,mec=INK,mew=.55,mfc="white",zorder=4)
        line.set_path_effects([pe.Stroke(linewidth=3.8, foreground=INK), pe.Normal()])
    for x, y, votes in vote_points:
        ax.scatter(x, y, s=8 + 10*votes, facecolor="white", edgecolor=INK,
                   linewidth=.65, alpha=.78, zorder=5)

    for boundary in np.arange(5.5, 36, 6):
        ax.axhline(boundary, color="white", lw=3.2, zorder=6)
        ax.axhline(boundary, color=INK, lw=.65, zorder=6)
    for boundary in [11.5, 23.5]:
        ax.axhline(boundary, color=INK, lw=2.0, zorder=7)

    ax.set_xticks(np.arange(0, 21, 2), [f"{g:.1f}" for g in gammas[::2]])
    ax.set_xlabel("Pricing geometry  γ     equal moments  ⟶  Hansen–Jagannathan")
    ax.set_yticks(np.arange(36), [f"K={k}" for _ in contexts for k in KS], fontsize=7.1)
    ax.tick_params(axis="y", length=0, pad=3)
    ax.spines[:].set_visible(False)

    for block, (_, _, label) in enumerate(contexts):
        y = block*6 + 2.5
        ax.text(-1.0, y, label, ha="right", va="center", fontsize=8.2,
                fontweight="semibold", color=INK, clip_on=False)
    for y, label in [(5.5, "SIZE–VALUE"), (17.5, "INDUSTRIES"), (29.5, "COMBINED")]:
        ax.text(20.65, y, label, rotation=90, ha="left", va="center",
                fontsize=7.1, fontweight="bold", color=MID, clip_on=False)

    cax = fig.add_axes([.20, .045, .51, .021])
    cb = fig.colorbar(im, cax=cax, orientation="horizontal", ticks=[0, 2, 5, 10, 20, 50])
    cb.set_label("Excess pricing distance relative to the pointwise best model (%)", fontsize=8)
    cb.outline.set_visible(False)
    legend_handles = [
        mpl.lines.Line2D([],[],color=INK,lw=1.8,marker="D",mfc="white",mec=INK,
                         ms=4,label="mean-distance winner"),
    ]
    legend_handles += [ax.scatter([], [], s=8+10*v, facecolor="white", edgecolor=INK,
                                  linewidth=.65, label=f"seed votes {v}/5") for v in [1,3,5]]
    fig.legend(handles=legend_handles, title="Two distinct selection summaries", ncol=2,
               loc="lower right", bbox_to_anchor=(.965,.018), fontsize=6.9, title_fontsize=7.3,
               columnspacing=.9,handletextpad=.5)
    fig.suptitle("Dimension competition across payoff geometry, feature lineage, and test assets",
                 y=.985, fontsize=13.2, fontweight="bold", color=INK)
    fig.text(.50, .925,
             "Diamond path: winner of the five-seed mean  ·  Hollow circles: seed-level winner votes (size = consensus)",
             ha="center", fontsize=8.0, color=MID)
    fig.subplots_adjust(top=.90, bottom=.12, left=.23, right=.93)
    save(fig, "figure_02_dimension_sensitivity")


def data_definition_instability() -> None:
    d = pd.read_csv(DATA / "table_04_data_definition_rank_agreement.csv")
    assets = ["25 Size–B/M", "49 Industries", "Combined 74"]
    fig = plt.figure(figsize=(11.0, 5.2))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.0, 1.35], hspace=0.46)
    ax = fig.add_subplot(gs[0])
    for label, color, marker in zip(assets, [BLUE, ORANGE, GREEN], ["o", "s", "^"]):
        q = d[d.asset_label == label].sort_values("gamma")
        ax.plot(q.gamma, q.spearman_rho, color=color, lw=2.0, marker=marker,
                markevery=4, ms=4.2, label=label)
    clean(ax); ax.axhline(0, color=INK, lw=0.8)
    ax.set_xlim(0, 1); ax.set_ylim(-0.68, 0.68)
    ax.set_ylabel("Spearman agreement of all six K ranks")
    ax.set_xlabel("Geometry weight  γ")
    panel(ax, "A", "Rank agreement is low or negative across most geometries")
    ax.legend(ncol=3, loc="lower right")

    ax2 = fig.add_subplot(gs[1])
    rows=[]; labels=[]
    for asset in assets:
        q=d[d.asset_label==asset].sort_values("gamma")
        rows += [q.best_K_core92.to_numpy(), q.best_K_core86.to_numpy()]
        labels += [f"{asset} · Core-92", f"{asset} · Core-86"]
    mat=np.vstack(rows)
    cmap=ListedColormap([KCOL[k] for k in KS]); norm=BoundaryNorm(np.arange(-.5,6.5,1), cmap.N)
    idx=np.vectorize({k:i for i,k in enumerate(KS)}.get)(mat)
    ax2.imshow(idx, aspect="auto", cmap=cmap, norm=norm, interpolation="nearest")
    ax2.set_yticks(range(len(labels)), labels)
    ticks=np.arange(0,21,5); ax2.set_xticks(ticks, [f"{x/20:.2f}" for x in ticks])
    ax2.set_xlabel("Geometry weight  γ")
    ax2.spines[:].set_visible(False)
    panel(ax2, "B", "The identity of the best K changes after the six-feature lineage audit")
    cax=fig.add_axes([0.20, 0.005, 0.60, 0.025])
    cb=mpl.colorbar.ColorbarBase(cax,cmap=cmap,norm=norm,orientation="horizontal",ticks=range(6))
    cb.ax.set_xticklabels([f"K={k}" for k in KS]); cb.outline.set_visible(False)
    fig.suptitle("A near-tie explanation is insufficient: the full dimension ranking reverses",
                 y=1.015, fontsize=13, fontweight="bold", color=INK)
    save(fig, "figure_03_data_definition_instability")


def sealed_falsification() -> None:
    fam = pd.read_csv(DATA / "sealed_table_01_family_inference.csv")
    comp = pd.read_csv(DATA / "sealed_table_04_development_comparison.csv")
    y=np.arange(len(fam))
    fig, axes = plt.subplots(1,5,figsize=(11.4,6.55),sharey=True,
                            gridspec_kw={"width_ratios":[1.05,1.55,1.65,1.65,1.65],"wspace":.08})
    titles=["Endpoint\ndimension","Geometry\ncontrast A","Market-direction\nattenuation",
            "Raw-moment\ncompletion","Alpha\ncompletion"]
    for ax,title in zip(axes,titles):
        ax.set_title(title,fontsize=9.0,fontweight="bold",color=INK,pad=10)
        ax.set_ylim(-.7,len(fam)-.3); ax.invert_yaxis(); clean(ax,"x")
        ax.tick_params(axis="y",length=0)
    axes[0].set_yticks(y,fam.asset_label,fontsize=7.6)
    for ax in axes[1:]: ax.tick_params(labelleft=False)

    ax=axes[0]
    ax.hlines(y,fam.best_K_gamma0,fam.best_K_gamma1,color=LIGHT,lw=2.6,zorder=1)
    ax.scatter(fam.best_K_gamma0,y,color=BLUE,s=28,label="EW",zorder=3)
    ax.scatter(fam.best_K_gamma1,y,color=RED,s=30,marker="s",label="HJ",zorder=3)
    ax.set_xticks(KS); ax.set_xlim(.55,8.45); ax.set_xlabel("Best K")

    metric_specs=[
        ("A_raw","A_low","A_high","development_A",1.0,"Contrast"),
        ("attenuation","atten_low","atten_high","development_attenuation",1.0,"Attenuation"),
        ("completion_raw","raw_low","raw_high","development_completion_raw",100.0,"Completion (pp)"),
        ("completion_alpha","alpha_low","alpha_high","development_completion_alpha",100.0,"Completion (pp)"),
    ]
    comp_index=comp.set_index("asset_family")
    for ax,(point,lo,hi,dev_col,scale,xlabel) in zip(axes[1:],metric_specs):
        ax.axvline(0,color=INK,lw=.8,zorder=1)
        ax.hlines(y,fam[lo]*scale,fam[hi]*scale,color=MID,lw=1.15,zorder=2)
        ax.scatter(fam[point]*scale,y,color=GREEN,s=28,zorder=4)
        for i,row in fam.iterrows():
            if row.asset_family in comp_index.index:
                dev=float(comp_index.loc[row.asset_family,dev_col])*scale
                sealed=float(row[point])*scale
                ax.annotate("",xy=(sealed,i),xytext=(dev,i),
                            arrowprops=dict(arrowstyle="-|>",color=LIGHT,lw=1.0),zorder=2)
                ax.scatter(dev,i,facecolor="white",edgecolor=BLUE,marker="D",s=20,
                           linewidth=.9,zorder=5)
        ax.set_xlabel(xlabel,fontsize=7.8)

    for ax in axes:
        for b in [1.5,4.5,7.5]: ax.axhline(b,color=LIGHT,lw=.55,zorder=0)
    fig.legend(handles=[
        mpl.lines.Line2D([],[],marker="o",color="none",markerfacecolor=BLUE,markeredgecolor=BLUE,label="EW endpoint"),
        mpl.lines.Line2D([],[],marker="s",color="none",markerfacecolor=RED,markeredgecolor=RED,label="HJ endpoint"),
        mpl.lines.Line2D([],[],marker="D",color="none",markerfacecolor="white",markeredgecolor=BLUE,label="Development"),
        mpl.lines.Line2D([],[],marker="o",color="none",markerfacecolor=GREEN,markeredgecolor=GREEN,label="Locked-period point; line = 95% interval"),
    ],loc="lower center",ncol=4,bbox_to_anchor=(.57,.018),fontsize=7.4)
    fig.suptitle("Locked-period evidence: dimension, geometry, market direction, and completion",
                 y=.985,fontsize=13,fontweight="bold",color=INK)
    fig.text(.58,.925,
             "Rows are preserved across every diagnostic; arrows connect development to the 2020–2025 locked estimate",
             ha="center",fontsize=8.0,color=MID)
    fig.subplots_adjust(top=.83,bottom=.14,left=.18,right=.985)
    save(fig,"figure_04_sealed_falsification")


def identification_mechanisms() -> None:
    conditions=pd.read_csv(DATA/"identification_condition_summary.csv")
    geom=pd.read_csv(DATA/"identification_table_04_geometry_disagreement.csv")
    fig=plt.figure(figsize=(11.4,8.9))
    gs=fig.add_gridspec(5,2,height_ratios=[1,1,1,1,1.05],hspace=.50,wspace=.16)
    row_specs=[("strong","full"),("strong","weak_tail"),("weak","full"),("weak","weak_tail")]
    row_names={"strong":"Strong prices","weak":"Weak prices","full":"Full span","weak_tail":"Weak tail"}
    months=np.array([72,240,600])
    for r,(sig,span) in enumerate(row_specs):
        for c,rho in enumerate([0.0,.7]):
            ax=fig.add_subplot(gs[r,c])
            for k0 in [1,3,5]:
                lines={}
                for ev,ls,alpha in [("oracle","-",1.0),("feasible","--",.92)]:
                    raw=conditions[(conditions.signal==sig)&(conditions.spanning==span)&
                                   (conditions.true_dimension==k0)&(conditions.evaluation==ev)&
                                   np.isclose(conditions.rho,rho)]
                    q=raw.groupby("months").exact_rate.mean().reindex(months)
                    lines[ev]=q.to_numpy()
                    for month in months:
                        values=(raw[raw.months==month].sort_values(["asset_count","geometry"])
                                .exact_rate.to_numpy())
                        offsets=np.linspace(-10,10,len(values))
                        if ev=="oracle":
                            ax.scatter(month+offsets,values,s=8,color=KCOL[k0],alpha=.25,
                                       edgecolors="none",zorder=2)
                        else:
                            ax.scatter(month+offsets,values,s=8,facecolors="white",edgecolors=KCOL[k0],
                                       linewidths=.45,alpha=.48,zorder=2)
                        ax.vlines(month,values.min(),values.max(),color=KCOL[k0],lw=.55,alpha=.22,zorder=1)
                    ax.plot(months,lines[ev],color=KCOL[k0],marker=KMARK[k0],ms=3.8,
                            lw=1.45,ls=ls,alpha=alpha,zorder=3)
                ax.fill_between(months,lines["feasible"],lines["oracle"],color=KCOL[k0],alpha=.055,zorder=1)
            ax.set_ylim(-.02,1.03); ax.set_xlim(55,620); ax.set_xticks(months)
            if r<3: ax.set_xticklabels([])
            if c==1: ax.set_yticklabels([])
            clean(ax,"both")
            ax.text(.02,.90,f"{row_names[sig]} · {row_names[span]}",transform=ax.transAxes,
                    fontsize=7.8,fontweight="semibold",color=INK,
                    bbox=dict(boxstyle="round,pad=.18",fc="white",ec="none",alpha=.82))
            if r==0:
                panel(ax,chr(65+c),f"Factor correlation ρ={rho:g}")

    ax=fig.add_subplot(gs[4,:])
    line_specs=[
        ("oracle","full",BLUE,"Oracle · full span","o"),
        ("oracle","weak_tail",SKY,"Oracle · weak tail","s"),
        ("feasible","full",RED,"Feasible · full span","^"),
        ("feasible","weak_tail",ORANGE,"Feasible · weak tail","D"),
    ]
    rho_positions=[0,1]
    for line_index,(ev,sp,color,label,marker) in enumerate(line_specs):
        vals=[]
        for rho in [0.0,.7]:
            vals.append(geom[(geom.evaluation==ev)&(geom.spanning==sp)&np.isclose(geom.rho,rho)]
                        .geometry_disagreement_rate.iloc[0])
        ax.plot(rho_positions,vals,color=color,marker=marker,ms=5,lw=1.8,label=label)
        label_shift={0:.012,1:-.012,2:.016,3:-.016}[line_index]
        ax.text(1.022,vals[1]+label_shift,f"{vals[1]:.1%}",va="center",fontsize=7,
                color=color,fontweight="semibold")
    ax.set_xlim(-.035,1.12); ax.set_ylim(-.01,.58); ax.set_xticks(rho_positions,["ρ=0","ρ=0.7"])
    ax.set_ylabel("Geometry disagreement")
    clean(ax,"both"); panel(ax,"C","Two prespecified correlation levels reveal geometry dependence")
    ax.legend(ncol=4,loc="upper left",bbox_to_anchor=(0,.87),fontsize=7.1)

    style_handles=[]
    for k0 in [1,3,5]:
        style_handles.append(mpl.lines.Line2D([],[],color=KCOL[k0],marker=KMARK[k0],lw=1.5,label=f"true K₀={k0}"))
    style_handles.extend([
        mpl.lines.Line2D([],[],color=INK,lw=1.5,ls="-",label="oracle evaluation"),
        mpl.lines.Line2D([],[],color=INK,lw=1.5,ls="--",label="feasible evaluation"),
        mpl.lines.Line2D([],[],color=MID,lw=0,marker=".",ms=4,label="asset-count × geometry cell"),
    ])
    fig.legend(handles=style_handles,ncol=6,loc="upper center",bbox_to_anchor=(.5,.925),fontsize=7.2)
    fig.suptitle("More observations recover missing directions but do not eliminate feasible over-selection",
                 y=.995,fontsize=13,fontweight="bold",color=INK)
    fig.text(.014,.555,"Exact recovery probability",rotation=90,va="center",ha="center",
             fontsize=8.5,color=INK)
    fig.text(.985,.018,
             "Each trajectory averages two asset counts and two pricing geometries; shaded gaps separate oracle from feasible evaluation.",
             ha="right",fontsize=7.2,color=MID)
    fig.subplots_adjust(top=.86,bottom=.075,left=.095,right=.965)
    save(fig,"figure_05_identification_mechanisms")


def temporal_usability() -> None:
    dec=pd.read_csv(DATA/"temporal_decomposition.csv")
    port=pd.read_csv(DATA/"temporal_portfolio.csv")
    events=pd.read_csv(DATA/"temporal_events.csv").set_index("event")["date"]
    break_date=pd.Timestamp(events["retrospective_break"]); trigger_date=pd.Timestamp(events["realtime_trigger"])
    bx=break_date.year+(break_date.month-1)/12; tx=trigger_date.year+(trigger_date.month-1)/12
    delay=(trigger_date.year-break_date.year)*12+trigger_date.month-break_date.month
    fig=plt.figure(figsize=(8.3,7.2))
    gs=fig.add_gridspec(2,2,height_ratios=[1.08,1],hspace=.50,wspace=.42)
    axes=[fig.add_subplot(gs[0,:]),fig.add_subplot(gs[1,0]),fig.add_subplot(gs[1,1])]
    ax=axes[0]
    labels=[]; vals_early=[]; vals_late=[]
    for model in ["Neural","Ridge","Tree"]:
        for update in ["Rolling 60","Expanding"]:
            q=dec[(dec["model"]==model)&(dec["update"]==update)]
            labels.append(f"{model} · {update}")
            vals_early.append(q[q["period"]=="2000-2009"].loss_diff.iloc[0]*10000)
            vals_late.append(q[q["period"]=="2010-2019"].loss_diff.iloc[0]*10000)
    y=np.arange(len(labels));
    for yi,a,b in zip(y,vals_early,vals_late): ax.hlines(yi,min(a,b),max(a,b),color=LIGHT,lw=2.5)
    ax.scatter(vals_early,y,color=ORANGE,marker="s",s=34,label="2000–2009",zorder=3)
    ax.scatter(vals_late,y,color=BLUE,marker="o",s=34,label="2010–2019",zorder=3)
    ax.axvline(0,color=INK,lw=.8)
    ax.set_yticks(y,labels,fontsize=7.2); ax.invert_yaxis(); clean(ax,"x")
    ax.set_xlabel("Updated minus frozen loss  (×10⁴; left is better)")
    panel(ax,"A","Update value reverses by decade\nand memory rule"); ax.legend(loc="lower right")

    ax=axes[1]; ax.set_xlim(2009.5,2020); ax.set_ylim(0,1); ax.axis("off")
    ax.hlines(.52,2010,2019.9,color=LIGHT,lw=4)
    ax.vlines(bx,.36,.70,color=BLUE,lw=2); ax.scatter([bx],[.52],color=BLUE,s=42,zorder=3)
    ax.text(bx,.75,break_date.strftime("%b %Y")+"\nretrospective",ha="center",color=BLUE,fontweight="bold",fontsize=8)
    ax.vlines(tx,.36,.70,color=RED,lw=2); ax.scatter([tx],[.52],color=RED,marker="s",s=42,zorder=3)
    ax.text(tx,.22,trigger_date.strftime("%b %Y")+"\nlagged trigger",ha="center",color=RED,fontweight="bold",fontsize=8)
    ax.annotate(f"{delay}-month detection delay",xy=(2016.45,.56),ha="center",color=MID,fontsize=7.8)
    panel(ax,"B","Retrospective break and\nlagged detection")

    ax=axes[2]; p=port.copy(); p["label"]=p["model"]+" · "+p["update"]
    yy=np.arange(len(p)); colors=[BLUE if u=="Frozen" else ORANGE if u=="Expanding" else PURPLE for u in p["update"]]
    ax.hlines(yy,0,p.ce_gamma5*100,color=LIGHT,lw=2); ax.scatter(p.ce_gamma5*100,yy,color=colors,s=34,zorder=3)
    ax.axvline(0,color=INK,lw=.85); ax.set_yticks(yy,p.label,fontsize=7.1); ax.invert_yaxis(); clean(ax,"x")
    ax.set_xlabel("Annualized certainty equivalent, γ=5 (%)")
    panel(ax,"C","After-cost certainty equivalents\nare all negative")
    fig.suptitle("Updating in a retrospective stock sample: losses, detection, and portfolio costs",
                 y=.995,fontsize=13,fontweight="bold",color=INK)
    fig.subplots_adjust(top=.89,bottom=.09,left=.18,right=.98)
    save(fig,"figure_06_temporal_usability")


def geometry_function() -> None:
    d=pd.read_csv(DATA/"p7_geometry_distribution.csv")
    summary=json.loads((DATA/"p7_canonical_return_summary.json").read_text())
    norms=[("raw_centered","Raw centered"),("feature_zscore","Feature z-score"),
           ("row_l2_centered","Row L2 centered")]
    layers=[("hidden1","Hidden 1",SKY),("hidden2","Hidden 2",BLUE),("hidden3","Hidden 3","#004C6D")]
    rows=[]
    for norm,nlabel in norms:
        for layer,llabel,color in layers: rows.append((norm,nlabel,layer,llabel,color))
    fig,ax=plt.subplots(figsize=(11.2,6.7))
    grid=np.linspace(3.5,28,180)
    edges=np.linspace(3.5,28,181)
    drift_norm=mpl.colors.Normalize(vmin=.60,vmax=1.00)
    drift_cmap=mpl.cm.viridis_r
    medians_by_norm={}
    for row,(norm,nlabel,layer,llabel,color) in enumerate(rows):
        base=8-row
        q=d[(d.normalization==norm)&(d.layer==layer)]
        all_hist,_=np.histogram(q.participation_rank,bins=edges,density=True)
        all_density=gaussian_filter1d(all_hist.astype(float),2.0)
        all_density=.74*all_density/max(all_density.max(),1e-12)
        ax.fill_between(grid,base,base+all_density,color=color,alpha=.38,lw=0,zorder=1)
        ax.plot(grid,base+all_density,color=color,lw=1.25,zorder=2)
        for flag,ls,c in [(False,"--",MID),(True,"-",RED)]:
            vals=q[q.high_volatility.eq(flag)].participation_rank
            hist,_=np.histogram(vals,bins=edges,density=True)
            density=gaussian_filter1d(hist.astype(float),2.0)
            density=.74*density/max(density.max(),1e-12)
            ax.plot(grid,base+density,color=c,lw=.8,ls=ls,alpha=.85,zorder=3)
        median=float(q.participation_rank.median())
        drift=float(q.subspace_drift.median())
        ax.scatter(median,base+.04,s=48,marker="D",color=drift_cmap(drift_norm(drift)),
                   edgecolor="white",linewidth=.7,zorder=5)
        medians_by_norm.setdefault(norm,[]).append((median,base))
        ax.text(27.85,base+.12,f"median {median:.2f}   drift {drift:.3f}",ha="right",
                va="bottom",fontsize=6.9,color=INK)
    for norm,points in medians_by_norm.items():
        xs=[p[0] for p in points]; ys=[p[1]+.04 for p in points]
        ax.plot(xs,ys,color=INK,lw=1.15,alpha=.75,zorder=4)
        ax.annotate("",xy=(xs[-1],ys[-1]),xytext=(xs[-2],ys[-2]),
                    arrowprops=dict(arrowstyle="-|>",color=INK,lw=1.15),zorder=4)
    for boundary in [5.5,2.5]: ax.axhline(boundary,color=INK,lw=.8)
    ax.set_xlim(3.5,28); ax.set_ylim(-.15,9.05)
    labels=[f"{nlabel}  ·  {llabel}" for _,nlabel,_,llabel,_ in rows]
    ax.set_yticks(range(8,-1,-1),labels,fontsize=7.7)
    ax.set_xlabel("Participation rank across frozen model–month cells")
    ax.set_ylabel("")
    clean(ax,"x"); ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0,pad=7)
    ax.plot([],[],color=MID,ls="--",lw=1,label="Other months")
    ax.plot([],[],color=RED,lw=1,label="High-volatility months")
    ax.scatter([],[],s=45,marker="D",color=drift_cmap(drift_norm(.8)),edgecolor="white",
               label="Median; color encodes drift")
    ax.legend(ncol=3,loc="upper right",fontsize=7.4)
    cax=fig.add_axes([.735,.072,.20,.018])
    cb=mpl.colorbar.ColorbarBase(cax,cmap=drift_cmap,norm=drift_norm,orientation="horizontal",
                                 ticks=[.6,.7,.8,.9,1.0])
    cb.set_label("Median rank-5 Grassmann drift",fontsize=7.2); cb.outline.set_visible(False)
    probe=summary["functional_probes"][0]
    fig.suptitle("Layerwise compression is pervasive across 32,400 model–month representations",
                 y=.99,fontsize=13,fontweight="bold",color=INK)
    fig.text(.20,.025,
             f"Functional boundary: HJ gain {probe['cross_validated_rmse_improvement_pct']:.3f}% < 5% gate; "
             f"permutation p={probe['permutation_p_two_sided']:.3f}",
             ha="left",fontsize=7.5,color=RED,fontweight="semibold")
    fig.subplots_adjust(top=.86,bottom=.16,left=.20,right=.985)
    save(fig,"figure_07_geometry_function")


def main() -> None:
    style()
    evidence_ladder()
    graphical_abstract()
    dimension_sensitivity()
    data_definition_instability()
    sealed_falsification()
    identification_mechanisms()
    temporal_usability()
    from build_bounded_control_exhibits import architecture
    architecture()
    print("built 7 publication figures in PDF, SVG, and PNG")


if __name__ == "__main__":
    main()
