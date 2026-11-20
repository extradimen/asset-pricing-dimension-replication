#!/usr/bin/env python3
"""Rebuild small, auditable figure inputs from frozen experiment outputs."""
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/P7-G1-V001/P7-G1-V001-CPU-REPLAY001"
OUTPUT = Path(__file__).resolve().parent / "figure_data/p7_model_evidence_matrix.csv"
GEOMETRY_OUTPUT = Path(__file__).resolve().parent / "figure_data/p7_geometry_distribution.csv"


def main() -> None:
    monthly = pd.read_csv(SOURCE / "monthly_geometry.csv")
    participation = (
        monthly[monthly.normalization.eq("raw_centered")]
        .groupby(["factor_count", "seed", "layer"]).participation_rank.median()
        .unstack("layer")
        .rename(columns=lambda value: f"{value}_participation_rank")
        .reset_index()
    )
    cka = (
        pd.read_csv(SOURCE / "layer_cka.csv")
        .groupby(["factor_count", "seed", "layer_pair"]).linear_cka.median()
        .unstack("layer_pair")
        .rename(columns=lambda value: f"cka_{value}")
        .reset_index()
    )
    drift_raw = pd.read_csv(SOURCE / "subspace_drift.csv")
    drift = (
        drift_raw[drift_raw.normalization.eq("raw_centered") & drift_raw.layer.eq("hidden3")]
        .groupby(["factor_count", "seed"]).grassmann_distance.median()
        .rename("hidden3_drift")
        .reset_index()
    )
    endpoints = pd.read_csv(SOURCE / "model_geometry_endpoints.csv")[[
        "factor_count", "seed", "development_hj_loss", "development_factor_sharpe"
    ]]
    result = (
        participation.merge(cka, on=["factor_count", "seed"])
        .merge(drift, on=["factor_count", "seed"])
        .merge(endpoints, on=["factor_count", "seed"])
        .sort_values(["factor_count", "seed"])
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"wrote {len(result)} model rows to {OUTPUT}")

    geometry_columns = [
        "factor_count", "seed", "target_month", "layer", "normalization",
        "high_volatility", "participation_rank",
    ]
    geometry = monthly[geometry_columns].merge(
        drift_raw.rename(columns={"grassmann_distance": "subspace_drift"}),
        on=["factor_count", "seed", "target_month", "layer", "normalization"],
        how="left",
    )
    geometry.to_csv(GEOMETRY_OUTPUT, index=False)
    print(f"wrote {len(geometry)} model-month-layer rows to {GEOMETRY_OUTPUT}")


if __name__ == "__main__":
    main()
