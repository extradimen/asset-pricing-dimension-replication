#!/usr/bin/env python3
"""Build reproducible paper-one long tables, statistics, and figures from frozen JSON results."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata, spearmanr

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [20260924, 20260925, 20260926, 20260927, 20260928]
KS = [1, 2, 3, 4, 5, 8]
PRIMARY = {"Core-92": "experiments/P1-G1-V015/dimension_geometry_surface_R001.json",
           "Core-86": "experiments/P1-G1-V022/core86_geometry_surface_R001.json"}
EXTERNAL = {"Core-92": "experiments/P1-G1-V021/external_asset_geometry_R001.json",
            "Core-86": "experiments/P1-G1-V022/core86_external_six_families_R001.json"}
MARKET = {"Core-92": "experiments/P1-G1-V017/market_hedged_geometry_R001.json",
          "Core-86": "experiments/P1-G1-V022/core86_market_hedge_R001.json"}
SPECTRAL = "experiments/P1-G1-V016/spectral_economic_decomposition_R001.json"
FEATURES = "experiments/P1-G0-V026/result_summary.json"
CONFIG = "configs/paper1/P1-G1-V023.json"
ASSET_LABEL = {"size_bm_25": "25 Size–B/M", "industry_49": "49 Industries", "combined_74": "Combined 74",
               "size_op_25": "25 Size–OP", "size_inv_25": "25 Size–Inv", "size_mom_25": "25 Size–Mom",
               "size_accruals_25": "25 Size–Accruals", "size_beta_25": "25 Size–Beta",
               "size_resvar_25": "25 Size–Residual variance"}
COLORS = {1: "#1B3A5D", 2: "#0072B2", 3: "#009E73", 4: "#E69F00", 5: "#D55E00", 8: "#7A5195"}


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fmt(x, digits=3):
    if pd.isna(x):
        return ""
    return f"{x:.{digits}f}"


def write_table(df: pd.DataFrame, stem: str, table_dir: Path, latex_cols=None, caption="", label=""):
    csv = table_dir / f"{stem}.csv"
    df.to_csv(csv, index=False, float_format="%.10g")
    show = df if latex_cols is None else df[latex_cols]
    (table_dir / f"{stem}.tex").write_text(show.to_latex(
        index=False, escape=True, na_rep="", float_format=lambda x: f"{x:.3f}",
        caption=caption or None, label=label or None, longtable=len(show) > 45,
    ))


def setup_style():
    mpl.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 11,
        "axes.labelsize": 9.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .6,
        "legend.frameon": False, "figure.dpi": 130, "savefig.bbox": "tight",
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def savefig(fig, stem: str, fig_dir: Path):
    for ext in ("pdf", "svg", "png"):
        fig.savefig(fig_dir / f"{stem}.{ext}", dpi=300 if ext == "png" else None,
                    facecolor="white", bbox_inches="tight")
    plt.close(fig)


def geometry_tables():
    agg_rows, seed_rows, endpoint_rows = [], [], []
    for dataset, path in PRIMARY.items():
        d = load(path)
        for family, by_k in d["aggregate"].items():
            for k, by_g in by_k.items():
                for gamma, s in by_g.items():
                    vals = s["seed_values"]
                    agg_rows.append({"data_definition": dataset, "asset_family": family,
                                     "asset_label": ASSET_LABEL[family], "K": int(k), "gamma": float(gamma),
                                     "mean_distance": s["mean"], "standard_error": s["standard_error"],
                                     "lower_2se": s["mean"] - 2*s["standard_error"],
                                     "upper_2se": s["mean"] + 2*s["standard_error"]})
                    for seed, value in zip(SEEDS, vals):
                        seed_rows.append({"data_definition": dataset, "asset_family": family,
                                          "asset_label": ASSET_LABEL[family], "K": int(k),
                                          "gamma": float(gamma), "seed": seed, "distance": value})
        # Endpoint frequency by seed is fixed before aggregation.
        sdf = pd.DataFrame(seed_rows)
        sdf = sdf[sdf.data_definition.eq(dataset)]
        for family in d["aggregate"]:
            for gamma in (0.0, 1.0):
                q = sdf[(sdf.asset_family == family) & (sdf.gamma == gamma)]
                best = q.loc[q.groupby("seed").distance.idxmin(), ["seed", "K"]]
                freq = Counter(best.K)
                mean_best = int(q.groupby("K").distance.mean().idxmin())
                for k in KS:
                    endpoint_rows.append({"data_definition": dataset, "asset_family": family,
                                          "asset_label": ASSET_LABEL[family], "gamma": gamma, "K": k,
                                          "best_seed_count": freq.get(k, 0),
                                          "best_seed_fraction": freq.get(k, 0)/len(SEEDS),
                                          "five_seed_mean_best_K": mean_best})
    agg = pd.DataFrame(agg_rows)
    agg["rank_within_surface"] = agg.groupby(["data_definition", "asset_family", "gamma"])["mean_distance"].rank(method="min")
    return agg.sort_values(["data_definition","asset_family","gamma","K"]), pd.DataFrame(seed_rows), pd.DataFrame(endpoint_rows)


def agreement_table(geometry: pd.DataFrame):
    rows = []
    for family in ["size_bm_25", "industry_49", "combined_74"]:
        for gamma in sorted(geometry.gamma.unique()):
            q = geometry[(geometry.asset_family == family) & (geometry.gamma == gamma)]
            a = q[q.data_definition == "Core-92"].set_index("K").loc[KS, "mean_distance"]
            b = q[q.data_definition == "Core-86"].set_index("K").loc[KS, "mean_distance"]
            rho, rp = spearmanr(a, b)
            tau, tp = kendalltau(a, b)
            k92, k86 = int(a.idxmin()), int(b.idxmin())
            rows.append({"asset_family": family, "asset_label": ASSET_LABEL[family], "gamma": gamma,
                         "spearman_rho": rho, "spearman_p": rp, "kendall_tau": tau, "kendall_p": tp,
                         "best_K_core92": k92, "best_K_core86": k86, "exact_best_K_agreement": k92 == k86})
    return pd.DataFrame(rows)


def external_tables():
    summary, dim_rows, sub_rows = [], [], []
    for dataset, path in EXTERNAL.items():
        d = load(path)
        for family, r in d["family_results"].items():
            boot = d["bootstrap"]["families"][family]
            row = {"data_definition": dataset, "asset_family": family, "asset_label": ASSET_LABEL[family],
                   "best_K_gamma0": r["best_dimension"]["0.0"], "best_K_gamma1": r["best_dimension"]["1.0"],
                   "A_raw": r["A_raw"], "A_ci_low": boot["A_raw"]["ci95"][0], "A_ci_high": boot["A_raw"]["ci95"][1],
                   "market_attenuation": r["market_direction_attenuation"],
                   "attenuation_ci_low": boot["attenuation"]["ci95"][0], "attenuation_ci_high": boot["attenuation"]["ci95"][1],
                   "completion_raw": r["completion_minus_k1"]["raw_moment"],
                   "completion_raw_ci_low": boot["completion_raw"]["ci95"][0], "completion_raw_ci_high": boot["completion_raw"]["ci95"][1],
                   "completion_alpha": r["completion_minus_k1"]["alpha"],
                   "completion_alpha_ci_low": boot["completion_alpha"]["ci95"][0], "completion_alpha_ci_high": boot["completion_alpha"]["ci95"][1]}
            summary.append(row)
            for k, v in r["learned_dimension_seed_mean"].items():
                dim_rows.append({"data_definition": dataset, "asset_family": family, "asset_label": ASSET_LABEL[family],
                                 "K": int(k), "distance_gamma0": v["distance"]["0.0"],
                                 "distance_gamma1": v["distance"]["1.0"], "raw_moment": v["raw_moment"], "alpha": v["alpha"]})
        for period, families in d["subperiod_results"].items():
            for family, r in families.items():
                sub_rows.append({"data_definition": dataset, "period": period, "asset_family": family,
                                 "asset_label": ASSET_LABEL[family], "best_K_gamma0": r["best_dimension"]["0.0"],
                                 "best_K_gamma1": r["best_dimension"]["1.0"], "A_raw": r["A_raw"],
                                 "A_hedged": r["A_market_beta_hedged"],
                                 "market_attenuation": r["market_direction_attenuation"],
                                 "completion_raw": r["completion_minus_k1"]["raw_moment"],
                                 "completion_alpha": r["completion_minus_k1"]["alpha"]})
    return pd.DataFrame(summary), pd.DataFrame(dim_rows), pd.DataFrame(sub_rows)


def mechanism_tables():
    rows, curves = [], []
    for dataset, path in MARKET.items():
        d = load(path)
        bd = d["basis_diagnostics"]
        a = d.get("endpoint_change_A", {"original": np.nan, "market_beta_hedged": np.nan})
        if dataset == "Core-92":
            # V017 predates standardized endpoint fields; derive them from frozen curves.
            ag = d["aggregate"]
            a = {mode: ag[mode]["mean_squared_distance_gap_by_gamma"]["1.0"] - ag[mode]["mean_squared_distance_gap_by_gamma"]["0.0"] for mode in ["original","market_beta_hedged"]}
        for mode in ["original", "market_beta_hedged"]:
            b = bd[mode]
            rows.append({"data_definition": dataset, "mode": mode,
                         "largest_eigenvalue_share": b.get("largest_eigenvalue_second_moment_share", b.get("largest_eigenvalue_variance_share")),
                         "equal_weight_cosine": b["largest_direction_absolute_cosine_with_equal_weight_vector"],
                         "same_sign_asset_share": b["largest_direction_same_sign_asset_share"], "endpoint_change_A": a[mode]})
            for g, val in d["aggregate"][mode]["mean_squared_distance_gap_by_gamma"].items():
                curves.append({"data_definition": dataset, "mode": mode, "gamma": float(g), "K1_minus_K5_squared_gap": val,
                               "standard_error": d["aggregate"][mode]["standard_error_by_gamma"][g]})
    spectral = load(SPECTRAL)
    srows = pd.DataFrame(spectral["directions"])
    deciles = pd.DataFrame(spectral["eigenvalue_deciles"])
    return pd.DataFrame(rows), pd.DataFrame(curves), srows, deciles


def feature_table():
    d = load(FEATURES)
    rows=[]
    labels = {"monthly_and_market": "Monthly/market", "quarterly": "Quarterly accounting", "annual": "Annual accounting"}
    for group, features in d["feature_groups"].items():
        for pos, feature in enumerate(features, 1):
            rows.append({"feature": feature, "group": group, "group_label": labels[group],
                         "within_group_index": pos, "frequency": group.split("_")[0].capitalize(),
                         "transformation": "Monthly cross-sectional rank scaled to [-1,1]; missing value=0 plus missing indicator"})
    return pd.DataFrame(rows)


def hypothesis_table(agreement, external):
    rows=[]
    for dataset, geo_file, ext_file in [("Core-92", PRIMARY["Core-92"], EXTERNAL["Core-92"]), ("Core-86", PRIMARY["Core-86"], EXTERNAL["Core-86"])]:
        g, e = load(geo_file), load(ext_file)
        def h(name):
            value=e["preregistered_hypotheses"][name]
            detail={k:v for k,v in value.items() if k!="supported"}
            return bool(value["supported"]), json.dumps(detail,ensure_ascii=False,sort_keys=True)
        h1,h2,h3=h("H1_geometry_dependence"),h("H2_market_direction"),h("H3_economic_completion")
        rows.extend([
            {"data_definition": dataset, "hypothesis": "Primary geometry rule", "passed": bool(g.get("v022_primary_geometry_rule_passed", g["preregistered_replication_rule_passed"])), "evidence": "Different endpoint K and bootstrap interval restriction"},
            {"data_definition": dataset, "hypothesis": "External-family geometry", "passed": h1[0], "evidence": h1[1]},
            {"data_definition": dataset, "hypothesis": "Market-direction mechanism", "passed": h2[0], "evidence": h2[1]},
            {"data_definition": dataset, "hypothesis": "Economic completion", "passed": h3[0], "evidence": h3[1]},
        ])
    return pd.DataFrame(rows)


def make_figures(geo, endpoint, agreement, external, subperiod, mech, curves, directions, deciles, fig_dir):
    # 1. Geometry surfaces, one row per data definition and one column per asset set.
    fig, axes = plt.subplots(2, 3, figsize=(12.2, 6.7), sharex=True)
    for i, dataset in enumerate(["Core-92", "Core-86"]):
        for j, family in enumerate(["size_bm_25", "industry_49", "combined_74"]):
            ax=axes[i,j]; q=geo[(geo.data_definition==dataset)&(geo.asset_family==family)]
            for k in KS:
                s=q[q.K==k]; ax.plot(s.gamma,s.mean_distance,color=COLORS[k],lw=1.8,label=f"K={k}")
                ax.fill_between(s.gamma,s.lower_2se,s.upper_2se,color=COLORS[k],alpha=.06)
            ax.set_title(f"{dataset}: {ASSET_LABEL[family]}"); ax.set_xlabel(r"Geometry weight $\gamma$")
            if j==0: ax.set_ylabel("Pricing distance")
    axes[0,2].legend(ncol=2,fontsize=8,loc="upper right")
    fig.suptitle("Pricing dimension depends on the loss geometry and data definition",fontweight="bold",y=1.01)
    savefig(fig,"figure_01_geometry_surfaces",fig_dir)

    # 2. Endpoint rank heatmaps.
    fig,axes=plt.subplots(2,2,figsize=(10,6.1),sharex=True,sharey=True)
    for i,dataset in enumerate(["Core-92","Core-86"]):
        for j,gamma in enumerate([0.,1.]):
            ax=axes[i,j]; q=geo[(geo.data_definition==dataset)&(geo.gamma==gamma)]
            mat=q.pivot(index="asset_label",columns="K",values="rank_within_surface").reindex(columns=KS)
            im=ax.imshow(mat.values,cmap="YlGnBu_r",vmin=1,vmax=6,aspect="auto")
            for y in range(mat.shape[0]):
                for x in range(mat.shape[1]): ax.text(x,y,int(mat.iloc[y,x]),ha="center",va="center",fontsize=8)
            ax.set_xticks(range(6),KS); ax.set_yticks(range(len(mat)),mat.index); ax.set_title(f"{dataset}, γ={gamma:.0f}")
            ax.set_xlabel("Latent dimension K")
    fig.colorbar(im,ax=axes.ravel().tolist(),label="Rank (1 = best)",shrink=.78)
    fig.suptitle("Endpoint rankings shift across geometry, assets, and feature definitions",fontweight="bold")
    savefig(fig,"figure_02_endpoint_rank_heatmap",fig_dir)

    # 3. Data-definition rank agreement.
    fig,axes=plt.subplots(1,2,figsize=(10.3,3.7),sharex=True)
    for family in agreement.asset_family.unique():
        q=agreement[agreement.asset_family==family]; axes[0].plot(q.gamma,q.spearman_rho,lw=2,label=ASSET_LABEL[family])
        axes[1].plot(q.gamma,q.exact_best_K_agreement.astype(int),lw=2,label=ASSET_LABEL[family])
    axes[0].axhline(0,color="#555",lw=.7); axes[0].set_ylabel("Spearman rank agreement"); axes[0].set_xlabel(r"$\gamma$")
    axes[1].set_ylabel("Exact best-K agreement"); axes[1].set_yticks([0,1],["No","Yes"]); axes[1].set_xlabel(r"$\gamma$")
    axes[0].legend(fontsize=8); fig.suptitle("Core-92 and Core-86 do not preserve one universal dimension ranking",fontweight="bold")
    savefig(fig,"figure_03_data_definition_agreement",fig_dir)

    def forest(cols, title, stem, xline=0):
        point,lo,hi=cols; labels=[]; ys=[]; colors=[]; values=[]; lows=[]; highs=[]
        y=0
        for dataset in ["Core-92","Core-86"]:
            for _,r in external[external.data_definition==dataset].iterrows():
                labels.append(f"{dataset} · {r.asset_label}"); ys.append(y); colors.append("#1B3A5D" if dataset=="Core-92" else "#D55E00")
                values.append(r[point]); lows.append(r[lo]); highs.append(r[hi]); y+=1
            y+=.7
        fig,ax=plt.subplots(figsize=(8.8,6.3)); vals=np.array(values); low=np.array(lows); high=np.array(highs)
        for value,yv,lv,hv,color in zip(vals,ys,low,high,colors):
            ax.errorbar(value,yv,xerr=[[value-lv],[hv-value]],fmt="o",color=color,
                        ecolor=color,markersize=4.5,elinewidth=1.6,capsize=3,zorder=3)
        ax.axvline(xline,color="#333",lw=.9,ls="--")
        ax.set_yticks(ys,labels); ax.invert_yaxis(); ax.set_xlabel("Point estimate and 95% block-bootstrap interval"); ax.set_title(title,fontweight="bold")
        savefig(fig,stem,fig_dir)
    forest(("A_raw","A_ci_low","A_ci_high"),"Geometry sensitivity A across external test assets","figure_04_external_A_forest")
    forest(("market_attenuation","attenuation_ci_low","attenuation_ci_high"),"Attenuation after removing the market direction","figure_05_market_attenuation_forest")

    # 6. Completion effects, two estimands.
    fig,axes=plt.subplots(1,2,figsize=(11.3,6.2),sharey=True)
    ordered=external.copy(); ordered["order"]=ordered.data_definition.map({"Core-92":0,"Core-86":1})
    ordered=ordered.sort_values(["order","asset_family"]); y=np.arange(len(ordered))
    for ax,point,lo,hi,title in [(axes[0],"completion_raw","completion_raw_ci_low","completion_raw_ci_high","Raw Euler moment"),(axes[1],"completion_alpha","completion_alpha_ci_low","completion_alpha_ci_high","Mean absolute alpha")]:
        vals=ordered[point].to_numpy(); colors=ordered.data_definition.map({"Core-92":"#1B3A5D","Core-86":"#D55E00"})
        lows=ordered[lo].to_numpy(); highs=ordered[hi].to_numpy()
        for value,yv,lv,hv,color in zip(vals,y,lows,highs,colors):
            ax.errorbar(value,yv,xerr=[[value-lv],[hv-value]],fmt="o",color=color,
                        ecolor=color,markersize=4.3,elinewidth=1.5,capsize=3,zorder=3)
        ax.axvline(0,color="#333",lw=.9,ls="--"); ax.set_title(title); ax.set_xlabel("K=1 + market minus K=1")
    axes[0].set_yticks(y,[f"{d} · {a}" for d,a in zip(ordered.data_definition,ordered.asset_label)]); axes[0].invert_yaxis()
    fig.suptitle("A market factor closes raw moments more reliably than alpha",fontweight="bold")
    savefig(fig,"figure_06_completion_forest",fig_dir)

    # 7. Market hedge gap curves.
    fig,axes=plt.subplots(1,2,figsize=(10.5,3.8),sharey=True)
    for ax,dataset in zip(axes,["Core-92","Core-86"]):
        for mode,color in [("original","#1B3A5D"),("market_beta_hedged","#D55E00")]:
            q=curves[(curves.data_definition==dataset)&(curves["mode"]==mode)]
            label="Original" if mode=="original" else "Market-beta hedged"
            ax.plot(q.gamma,q.K1_minus_K5_squared_gap,color=color,lw=2,label=label)
            ax.fill_between(q.gamma,q.K1_minus_K5_squared_gap-2*q.standard_error,q.K1_minus_K5_squared_gap+2*q.standard_error,color=color,alpha=.09)
        ax.axhline(0,color="#333",lw=.8); ax.set_title(dataset); ax.set_xlabel(r"$\gamma$"); ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Squared-distance gap, $K=1-K=5$")
    fig.suptitle("Removing market exposure attenuates the geometry shift, with substantial seed uncertainty",fontweight="bold")
    savefig(fig,"figure_07_market_hedge_curves",fig_dir)

    # 8. Spectral concentration.
    fig,axes=plt.subplots(1,2,figsize=(10.4,3.8))
    axes[0].bar(deciles.ascending_eigenvalue_decile,deciles.absolute_endpoint_change_mass_share,color="#3C6E71")
    axes[0].set_xlabel("Ascending eigenvalue decile"); axes[0].set_ylabel("Absolute endpoint-change mass share")
    axes[1].scatter(directions.eigenvalue,directions.mean_endpoint_change,s=22,c=directions.absolute_endpoint_change_share,cmap="magma",alpha=.85)
    axes[1].set_xscale("log"); axes[1].axhline(0,color="#333",lw=.8); axes[1].set_xlabel("Return second-moment eigenvalue (log scale)"); axes[1].set_ylabel("Mean endpoint change")
    fig.suptitle("Geometry sensitivity is concentrated in a small set of high-variance directions",fontweight="bold")
    savefig(fig,"figure_08_spectral_concentration",fig_dir)

    # 9. Subperiod robustness heatmap. Colors are normalized within statistic;
    # cells retain raw estimates so unlike scales remain interpretable.
    fig,axes=plt.subplots(1,2,figsize=(11,4.4),sharey=True)
    metrics=["A_raw","market_attenuation","completion_raw","completion_alpha"]
    global_scales={m:max(float(subperiod[m].abs().max()),1e-12) for m in metrics}
    for ax,dataset in zip(axes,["Core-92","Core-86"]):
        q=subperiod[subperiod.data_definition==dataset].copy(); q["row"]=q.period+" · "+q.asset_label
        mat=q.set_index("row")[metrics]; normalized=mat.copy()
        for m in metrics: normalized[m]=normalized[m]/global_scales[m]
        im=ax.imshow(normalized.values,cmap="RdBu_r",vmin=-1,vmax=1,aspect="auto")
        for y in range(mat.shape[0]):
            for x in range(mat.shape[1]): ax.text(x,y,fmt(mat.iloc[y,x],2),ha="center",va="center",fontsize=6.6)
        ax.set_xticks(range(4),["A","Market atten.","Moment comp.","Alpha comp."],rotation=25,ha="right")
        ax.set_yticks(range(len(mat)),mat.index); ax.set_title(dataset)
    fig.colorbar(im,ax=axes.ravel().tolist(),shrink=.8,label="Signed estimate / statistic-specific maximum")
    fig.suptitle("Mechanism estimates vary materially across development subperiods",fontweight="bold")
    savefig(fig,"figure_09_subperiod_robustness",fig_dir)

    # 10. Seed-level endpoint selection stability.
    q=endpoint[endpoint.best_seed_count>0].copy()
    fig,axes=plt.subplots(2,3,figsize=(11.5,6.2),sharex=True,sharey=True)
    for i,dataset in enumerate(["Core-92","Core-86"]):
        for j,family in enumerate(["size_bm_25","industry_49","combined_74"]):
            ax=axes[i,j]; s=q[(q.data_definition==dataset)&(q.asset_family==family)]
            for gamma,marker,color in [(0.,"o","#0072B2"),(1.,"s","#D55E00")]:
                z=s[s.gamma==gamma]
                ax.scatter(z.K,[gamma]*len(z),s=90*z.best_seed_fraction+20,marker=marker,color=color,alpha=.82,label=f"γ={gamma:.0f}")
                for _,r in z.iterrows(): ax.text(r.K,gamma+(.10 if gamma==0 else -.10),f"{int(r.best_seed_count)}/5",ha="center",fontsize=7)
            ax.set_title(f"{dataset}: {ASSET_LABEL[family]}"); ax.set_xticks(KS); ax.set_yticks([0,1]); ax.set_ylim(-.28,1.28)
            ax.set_xlabel("Seed-selected K")
    axes[0,0].set_ylabel("Geometry endpoint"); axes[1,0].set_ylabel("Geometry endpoint")
    axes[0,2].legend(fontsize=8,loc="center right")
    fig.suptitle("No primary asset panel has unanimous best-K selection across seeds",fontweight="bold")
    savefig(fig,"figure_10_seed_selection_stability",fig_dir)

    # 11. Feature-frequency composition.
    f=feature_table().groupby("group_label").size().sort_values()
    fig,ax=plt.subplots(figsize=(6.8,3.6)); bars=ax.barh(f.index,f.values,color=["#3C6E71","#D9A441","#1B3A5D"])
    for bar,value in zip(bars,f.values): ax.text(value+.7,bar.get_y()+bar.get_height()/2,str(value),va="center",fontweight="bold")
    ax.set_xlabel("Number of characteristics"); ax.set_title("Core-86 combines three information frequencies",fontweight="bold")
    ax.set_xlim(0,max(f.values)*1.12); savefig(fig,"figure_11_core86_feature_composition",fig_dir)


def main():
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument("--output-dir",type=Path,default=ROOT/"paper1_artifacts"); args=ap.parse_args()
    out=args.output_dir.resolve(); tables=out/"tables"; figures=out/"figures"; manifests=out/"manifests"
    for p in (tables,figures,manifests): p.mkdir(parents=True,exist_ok=True)
    setup_style()
    geo,seeds,endpoint=geometry_tables(); agreement=agreement_table(geo)
    external,external_dim,subperiod=external_tables(); mech,curves,directions,deciles=mechanism_tables(); features=feature_table()
    hypotheses=hypothesis_table(agreement,external)
    endpoint_main=(geo[geo.gamma.isin([0.,1.])]
        .sort_values("mean_distance").groupby(["data_definition","asset_family","asset_label","gamma"],as_index=False).first()
        .pivot(index=["data_definition","asset_family","asset_label"],columns="gamma",values=["K","mean_distance","standard_error"]).reset_index())
    endpoint_main.columns=["_".join(str(x) for x in c if str(x)!="").replace(".0","") if isinstance(c,tuple) else c for c in endpoint_main.columns]
    agreement_main=(agreement.groupby(["asset_family","asset_label"],as_index=False)
        .agg(exact_best_K_agreement_rate=("exact_best_K_agreement","mean"),mean_spearman=("spearman_rho","mean"),
             min_spearman=("spearman_rho","min"),max_spearman=("spearman_rho","max"),mean_kendall=("kendall_tau","mean")))
    outputs={
        "main_table_01_endpoint_best_dimensions": endpoint_main,
        "main_table_02_data_definition_sensitivity": agreement_main,
        "main_table_03_external_family_inference": external,
        "main_table_04_hypothesis_gates": hypotheses,
        "table_01_geometry_surface_long": geo, "table_02_geometry_seed_long": seeds,
        "table_03_endpoint_seed_stability": endpoint, "table_04_data_definition_rank_agreement": agreement,
        "table_05_external_family_inference": external, "table_06_external_dimension_metrics": external_dim,
        "table_07_external_subperiods": subperiod, "table_08_market_mechanism": mech,
        "table_09_market_hedge_curves": curves, "table_10_spectral_directions": directions,
        "table_11_spectral_deciles": deciles, "table_12_core86_feature_dictionary": features,
        "table_13_hypothesis_gates": hypotheses,
    }
    for stem,df in outputs.items(): write_table(df,stem,tables,caption=stem.replace("_"," ").title(),label=f"tab:{stem}")
    make_figures(geo,endpoint,agreement,external,subperiod,mech,curves,directions,deciles,figures)
    # Machine-readable result summary contains only prespecified descriptive statistics.
    summary={
        "schema_version":1,"experiment_id":"P1-G1-V023","sealed_period_accessed":False,
        "row_counts":{k:int(len(v)) for k,v in outputs.items()},
        "best_K_exact_agreement_rate":{f:float(q.exact_best_K_agreement.mean()) for f,q in agreement.groupby("asset_family")},
        "mean_rank_agreement":{f:{"spearman":float(q.spearman_rho.mean()),"kendall":float(q.kendall_tau.mean())} for f,q in agreement.groupby("asset_family")},
        "endpoint_seed_unanimity_rate":float((endpoint.groupby(["data_definition","asset_family","gamma"]).best_seed_count.max()==5).mean()),
        "external_ci_excludes_zero":{dataset:{
            "A":int(((q.A_ci_low>0)|(q.A_ci_high<0)).sum()),
            "market_attenuation":int(((q.attenuation_ci_low>0)|(q.attenuation_ci_high<0)).sum()),
            "completion_raw":int(((q.completion_raw_ci_low>0)|(q.completion_raw_ci_high<0)).sum()),
            "completion_alpha":int(((q.completion_alpha_ci_low>0)|(q.completion_alpha_ci_high<0)).sum())
        } for dataset,q in external.groupby("data_definition")},
        "interpretation":"Descriptive reporting only; no V022 primary decision is changed."
    }
    (out/"result_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2)+"\n")
    input_paths=[ROOT/p for p in [*PRIMARY.values(),*EXTERNAL.values(),*MARKET.values(),SPECTRAL,FEATURES,CONFIG]]
    artifact_paths=sorted([p for p in out.rglob("*") if p.is_file() and "artifact_manifest" not in p.name])
    manifest={"schema_version":1,"experiment_id":"P1-G1-V023","sealed_period_accessed":False,
              "inputs":[{"path":str(p.relative_to(ROOT)),"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in input_paths],
              "artifacts":[{"path":str(p.relative_to(ROOT)),"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in artifact_paths]}
    (manifests/"artifact_manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
