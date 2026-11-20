#!/usr/bin/env python3
"""Build tables and publication figures for P1-G3-V001."""

from __future__ import annotations

import hashlib
import json
import plistlib
import subprocess
from pathlib import Path

_CHECK_OUTPUT = subprocess.check_output


def _sandbox_safe_check_output(command, *args, **kwargs):
    # macOS system_profiler can hang in a restricted execution environment.
    if isinstance(command, (list, tuple)) and command and command[0] == "system_profiler":
        return plistlib.dumps([{"_items": []}])
    return _CHECK_OUTPUT(command, *args, **kwargs)


subprocess.check_output = _sandbox_safe_check_output

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/P1-G3-V001/outputs/P1-G3-V001-R001"
DEST = ROOT / "paper1_identification_artifacts"
TABLES = DEST / "tables"
FIGURES = DEST / "figures"
COLORS = {"oracle": "#1f4e79", "feasible": "#c55a11", "strong": "#2f5597", "weak": "#a61c3c"}


def save_table(frame: pd.DataFrame, stem: str) -> list[Path]:
    csv_path = TABLES / f"{stem}.csv"
    tex_path = TABLES / f"{stem}.tex"
    frame.to_csv(csv_path, index=False)
    tex_path.write_text(frame.to_latex(index=False, float_format=lambda value: f"{value:.4f}"), encoding="utf-8")
    return [csv_path, tex_path]


def save_figure(fig: plt.Figure, stem: str) -> list[Path]:
    paths = [FIGURES / f"{stem}.{suffix}" for suffix in ("pdf", "svg", "png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], bbox_inches="tight")
    fig.savefig(paths[2], bbox_inches="tight", dpi=300)
    plt.close(fig)
    return paths


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    condition = pd.read_csv(SOURCE / "condition_summary.csv")
    frequencies = pd.read_csv(SOURCE / "selection_frequencies.csv")
    disagreement = pd.read_csv(SOURCE / "disagreement_summary.csv")
    hypotheses = pd.read_csv(SOURCE / "hypothesis_summary.csv")
    artifacts: list[Path] = []

    artifacts += save_table(hypotheses, "identification_table_01_frozen_hypotheses")

    blocks = []
    for variable in ("evaluation", "months", "signal", "spanning", "true_dimension", "asset_count", "rho", "geometry"):
        block = condition.groupby(variable, as_index=False)[["exact_rate", "under_rate", "over_rate", "mean_absolute_error"]].mean()
        block.insert(0, "mechanism", variable)
        block = block.rename(columns={variable: "level"})
        blocks.append(block)
    mechanism = pd.concat(blocks, ignore_index=True)
    artifacts += save_table(mechanism, "identification_table_02_mechanism_aggregates")

    ideal = condition.query("signal == 'strong' and spanning == 'full' and rho == 0.0")
    ideal = ideal.groupby(["evaluation", "geometry", "months", "true_dimension"], as_index=False)[["exact_rate", "under_rate", "over_rate"]].mean()
    artifacts += save_table(ideal, "identification_table_03_ideal_recovery")

    geometry = disagreement.query("geometry == 'equal_vs_hj'").groupby(
        ["evaluation", "rho", "spanning"], as_index=False
    )["family_divergence_rate"].mean().rename(columns={"family_divergence_rate": "geometry_disagreement_rate"})
    artifacts += save_table(geometry, "identification_table_04_geometry_disagreement")

    family = disagreement.query("geometry != 'equal_vs_hj'").groupby(
        ["evaluation", "geometry", "signal", "spanning"], as_index=False
    )[["family_divergence_rate", "family_unanimity_rate", "mean_unique_dimensions"]].mean()
    artifacts += save_table(family, "identification_table_05_asset_family_instability")
    artifacts += save_table(condition, "identification_table_06_all_conditions")
    artifacts += save_table(frequencies, "identification_table_07_all_selection_frequencies")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    grouped = condition.groupby(["evaluation", "signal", "spanning", "months"], as_index=False)["exact_rate"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, evaluation in zip(axes, ("oracle", "feasible")):
        subset = grouped[grouped.evaluation == evaluation]
        for (signal, spanning), values in subset.groupby(["signal", "spanning"]):
            values = values.sort_values("months")
            axis.plot(values.months, values.exact_rate, marker="o", label=f"{signal}, {spanning}")
        axis.set_title(f"{evaluation.capitalize()} evaluation")
        axis.set_xlabel("Months per training/evaluation window")
        axis.set_ylim(0, 1)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Exact recovery probability")
    axes[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Known pricing dimension is recoverable only under strong identification")
    artifacts += save_figure(fig, "identification_figure_01_recovery_by_sample_length")

    ideal_plot = ideal.groupby(["evaluation", "months", "true_dimension"], as_index=False)["exact_rate"].mean()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for axis, evaluation in zip(axes, ("oracle", "feasible")):
        subset = ideal_plot[ideal_plot.evaluation == evaluation]
        for dimension, values in subset.groupby("true_dimension"):
            values = values.sort_values("months")
            axis.plot(values.months, values.exact_rate, marker="o", label=f"true K={dimension}")
        axis.set_title(f"{evaluation.capitalize()} evaluation")
        axis.set_xlabel("Months")
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.25)
        axis.legend(frameon=False)
    axes[0].set_ylabel("Exact recovery probability")
    fig.suptitle("Ideal DGP: evaluation noise prevents exact-dimension consistency")
    artifacts += save_figure(fig, "identification_figure_02_ideal_recovery_by_true_k")

    errors = condition.groupby(["evaluation", "months"], as_index=False)[["exact_rate", "under_rate", "over_rate"]].mean()
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    for axis, evaluation in zip(axes, ("oracle", "feasible")):
        values = errors[errors.evaluation == evaluation].sort_values("months")
        bottom = np.zeros(len(values))
        for column, color, label in (("under_rate", "#a61c3c", "Underselect"), ("exact_rate", "#70ad47", "Exact"), ("over_rate", "#4472c4", "Overselect")):
            axis.bar(values.months.astype(str), values[column], bottom=bottom, color=color, label=label)
            bottom += values[column].to_numpy()
        axis.set_title(evaluation.capitalize())
        axis.set_xlabel("Months")
    axes[0].set_ylabel("Probability")
    axes[1].legend(frameon=False, loc="lower right")
    fig.suptitle("Oracle evaluation removes over-selection; feasible evaluation does not")
    artifacts += save_figure(fig, "identification_figure_03_selection_error_decomposition")

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.8), sharey=True)
    for axis, evaluation in zip(axes, ("oracle", "feasible")):
        pivot = geometry[geometry.evaluation == evaluation].pivot(index="spanning", columns="rho", values="geometry_disagreement_rate")
        image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=max(0.55, geometry.geometry_disagreement_rate.max()), cmap="YlOrRd", aspect="auto")
        axis.set_xticks(range(len(pivot.columns)), [str(value) for value in pivot.columns])
        axis.set_yticks(range(len(pivot.index)), pivot.index)
        axis.set_xlabel("Factor correlation")
        axis.set_title(evaluation.capitalize())
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                axis.text(j, i, f"{pivot.iloc[i, j]:.1%}", ha="center", va="center")
    axes[0].set_ylabel("Asset spanning")
    fig.colorbar(image, ax=axes, label="Equal–HJ disagreement", shrink=0.85)
    fig.suptitle("Correlated directions make selected dimension geometry-dependent")
    artifacts += save_figure(fig, "identification_figure_04_geometry_disagreement")

    family_plot = family.groupby(["evaluation", "signal", "spanning"], as_index=False)["family_divergence_rate"].mean()
    categories = [
        ("strong", "full", "strong/full"),
        ("strong", "weak_tail", "strong/weak-tail"),
        ("weak", "full", "weak/full"),
        ("weak", "weak_tail", "weak/weak-tail"),
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(categories))
    width = 0.36
    for offset, evaluation in ((-width / 2, "oracle"), (width / 2, "feasible")):
        values = []
        for signal, spanning, _ in categories:
            value = family_plot.query("evaluation == @evaluation and signal == @signal and spanning == @spanning").family_divergence_rate.iloc[0]
            values.append(value)
        ax.bar(x + offset, values, width, label=evaluation.capitalize(), color=COLORS[evaluation])
    ax.set_xticks(x, [display for _, _, display in categories], rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Probability six asset families select multiple K")
    ax.legend(frameon=False)
    ax.set_title("Exact selected dimension is rarely invariant across asset families")
    artifacts += save_figure(fig, "identification_figure_05_asset_family_divergence")

    representative = frequencies.query(
        "evaluation == 'feasible' and months == 72 and asset_count == 74 and rho == 0.7 and true_dimension == 3"
    )
    representative = representative.groupby(["signal", "spanning", "geometry", "candidate_dimension"], as_index=False)["selection_rate"].mean()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, (signal, spanning) in zip(axes.flat, (("strong", "full"), ("strong", "weak_tail"), ("weak", "full"), ("weak", "weak_tail"))):
        subset = representative.query("signal == @signal and spanning == @spanning")
        for geometry_name, values in subset.groupby("geometry"):
            axis.plot(values.candidate_dimension, values.selection_rate, marker="o", label=geometry_name)
        axis.axvline(3, color="black", linestyle="--", linewidth=1)
        axis.set_title(f"{signal}, {spanning}")
        axis.grid(alpha=0.2)
    axes[1, 0].set_xlabel("Selected K")
    axes[1, 1].set_xlabel("Selected K")
    axes[0, 0].set_ylabel("Selection probability")
    axes[1, 0].set_ylabel("Selection probability")
    axes[0, 1].legend(frameon=False)
    fig.suptitle("Representative 72-month selection distribution (true K=3)")
    artifacts += save_figure(fig, "identification_figure_06_selection_distribution")

    fig, ax = plt.subplots(figsize=(7.5, 4))
    for evaluation in ("oracle", "feasible"):
        values = condition.loc[condition.evaluation == evaluation, "exact_rate"]
        ax.hist(values, bins=np.linspace(0, 1, 21), alpha=0.55, label=evaluation.capitalize(), color=COLORS[evaluation])
    ax.set_xlabel("Condition-level exact recovery rate")
    ax.set_ylabel("Number of frozen conditions")
    ax.legend(frameon=False)
    ax.set_title("Recovery performance varies widely across identification conditions")
    artifacts += save_figure(fig, "identification_figure_07_recovery_distribution")

    readme = DEST / "README.md"
    readme.write_text(
        "# P1-G3-V001 identification artifacts\n\n"
        "Generated only from the frozen synthetic output of P1-G3-V001-R001. "
        "The directory contains seven CSV/LaTeX table pairs and seven figures in PDF, SVG and 300-dpi PNG.\n",
        encoding="utf-8",
    )
    artifacts.append(readme)
    manifest = {
        "schema_version": 1,
        "experiment_id": "P1-G3-V001",
        "run_id": "P1-G3-V001-R001",
        "source_manifest_sha256": sha256(SOURCE / "output_manifest.json"),
        "artifacts": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(artifacts)
        ],
    }
    manifest_path = DEST / "artifact_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"tables": 7, "figures": 7, "artifacts": len(artifacts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
