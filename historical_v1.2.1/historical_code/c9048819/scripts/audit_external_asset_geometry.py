#!/usr/bin/env python3
"""Confirm metric-dependent compressibility on six frozen public asset families."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import audit_hj_alpha_estimands as estimands
import evaluate_multi_factor_teacher as multi
import evaluate_teacher_pricing as base

KS = (1, 2, 3, 4, 5, 8)
SEEDS = (20260924, 20260925, 20260926, 20260927, 20260928)
GAMMAS = np.round(np.arange(0, 1.0001, 0.05), 2)
FAMILIES = {
    "size_op_25": "25_Portfolios_ME_OP_5x5_CSV.zip",
    "size_inv_25": "25_Portfolios_ME_INV_5x5_CSV.zip",
    "size_mom_25": "25_Portfolios_ME_Prior_12_2_CSV.zip",
    "size_accruals_25": "25_Portfolios_ME_AC_5x5_CSV.zip",
    "size_beta_25": "25_Portfolios_ME_BETA_5x5_CSV.zip",
    "size_resvar_25": "25_Portfolios_ME_RESVAR_5x5_CSV.zip",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--external-dir", type=Path, required=True)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--ff5", type=Path, required=True)
    parser.add_argument("--momentum", type=Path, required=True)
    parser.add_argument("--experiment-id", default="P1-G1-V021")
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def geometry(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    second = returns.T @ returns / len(returns)
    values, vectors = np.linalg.eigh(second)
    ridge = 1e-4 * np.trace(second) / second.shape[0]
    return values, vectors, float(ridge)


def squared_distance(moment: np.ndarray, system: tuple[np.ndarray, np.ndarray, float], gamma: float) -> float:
    values, vectors, ridge = system
    diagonal = (values + ridge) ** (-gamma)
    diagonal *= len(values) / np.sum(diagonal * values)
    return float(12 * np.sum((vectors.T @ moment) ** 2 * diagonal))


def metrics(factors: np.ndarray, returns: np.ndarray, loading: np.ndarray, system, gammas=(0.0, 1.0)) -> dict:
    moment = estimands.moment_vector(factors, returns, loading)
    alpha = estimands.alpha_vector(factors, returns)
    return {
        "distance": {str(g): squared_distance(moment, system, float(g)) for g in gammas},
        "raw_moment": float(np.abs(moment).mean() * 12),
        "alpha": float(np.abs(alpha).mean() * 12),
    }


def circular_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    pieces = []
    while sum(len(piece) for piece in pieces) < length:
        start = int(rng.integers(length))
        pieces.append((start + np.arange(block)) % length)
    return np.concatenate(pieces)[:length]


def mean_seed_metrics(rows: list[dict]) -> dict:
    return {
        "distance": {
            str(g): float(np.mean([row["distance"][str(g)] for row in rows]))
            for g in (0.0, 1.0)
        },
        "raw_moment": float(np.mean([row["raw_moment"] for row in rows])),
        "alpha": float(np.mean([row["alpha"] for row in rows])),
    }


def evaluate_period(
    returns: np.ndarray,
    hedged: np.ndarray,
    factors: dict[int, dict[int, np.ndarray]],
    market: np.ndarray,
    loadings: dict[int, dict[int, np.ndarray]],
    completion_loadings: dict[int, np.ndarray],
) -> dict:
    systems = {"raw": geometry(returns), "hedged": geometry(hedged)}
    by_k = {}
    by_k_hedged = {}
    for k in KS:
        rows = [metrics(factors[k][seed], returns, loadings[k][seed], systems["raw"]) for seed in SEEDS]
        hedge_rows = [metrics(factors[k][seed], hedged, loadings[k][seed], systems["hedged"]) for seed in SEEDS]
        by_k[str(k)] = mean_seed_metrics(rows)
        by_k_hedged[str(k)] = mean_seed_metrics(hedge_rows)
    completed_rows = [
        metrics(
            np.column_stack([factors[1][seed], market]),
            returns,
            completion_loadings[seed],
            systems["raw"],
        )
        for seed in SEEDS
    ]
    completed = mean_seed_metrics(completed_rows)
    a_raw = ((by_k["1"]["distance"]["0.0"] - by_k["5"]["distance"]["0.0"])
             - (by_k["1"]["distance"]["1.0"] - by_k["5"]["distance"]["1.0"]))
    a_hedged = ((by_k_hedged["1"]["distance"]["0.0"] - by_k_hedged["5"]["distance"]["0.0"])
                - (by_k_hedged["1"]["distance"]["1.0"] - by_k_hedged["5"]["distance"]["1.0"]))
    best = {
        str(g): int(min(KS, key=lambda k: by_k[str(k)]["distance"][str(g)]))
        for g in (0.0, 1.0)
    }
    return {
        "learned_dimension_seed_mean": by_k,
        "hedged_learned_dimension_seed_mean": by_k_hedged,
        "best_dimension": best,
        "A_raw": float(a_raw),
        "A_market_beta_hedged": float(a_hedged),
        "market_direction_attenuation": float(abs(a_raw) - abs(a_hedged)),
        "k1_plus_market": completed,
        "completion_minus_k1": {
            "raw_moment": float(completed["raw_moment"] - by_k["1"]["raw_moment"]),
            "alpha": float(completed["alpha"] - by_k["1"]["alpha"]),
        },
    }


def main() -> int:
    args = parse_args()
    ff5 = base.read_factor_file(args.ff5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    momentum = base.read_factor_file(args.momentum, ["Mom"])
    rf = {month: row[-1] for month, row in ff5.items()}
    raw = {
        label: base.read_first_value_weighted_monthly(args.external_dir / name, 25, f"{label}::")[1]
        for label, name in FAMILIES.items()
    }
    months = {
        "validation": [str(month) for month in np.arange(np.datetime64("2000-01"), np.datetime64("2010-01"), dtype="datetime64[M]")],
        "development": [str(month) for month in np.arange(np.datetime64("2010-01"), np.datetime64("2020-01"), dtype="datetime64[M]")],
    }
    returns = {
        family: {period: base.asset_matrix(period_months, values, rf) for period, period_months in months.items()}
        for family, values in raw.items()
    }
    market = {
        period: np.asarray([ff5[base.next_month(month)][0] for month in period_months])
        for period, period_months in months.items()
    }
    benchmark = {
        period: np.vstack([
            np.concatenate([ff5[base.next_month(month)][:-1], momentum[base.next_month(month)]])
            for month in period_months
        ])
        for period, period_months in months.items()
    }
    benchmark_loading = estimands.self_pricing_loadings(benchmark["validation"])
    factor_maps = {
        k: {
            seed: multi.read_factor_matrix(args.sweep_root / f"factors-k{k}-seed{seed}" / "monthly_factor_returns.csv")
            for seed in SEEDS
        }
        for k in KS
    }
    factors = {
        period: {
            k: {seed: multi.align_matrix(period_months, factor_maps[k][seed]) for seed in SEEDS}
            for k in KS
        }
        for period, period_months in months.items()
    }
    loadings = {
        k: {seed: estimands.self_pricing_loadings(factors["validation"][k][seed]) for seed in SEEDS}
        for k in KS
    }
    completion_loadings = {
        seed: estimands.self_pricing_loadings(
            np.column_stack([factors["validation"][1][seed], market["validation"]])
        )
        for seed in SEEDS
    }
    hedged = {}
    market_betas = {}
    validation_design = np.column_stack([np.ones(len(market["validation"])), market["validation"]])
    for family in FAMILIES:
        beta = (np.linalg.pinv(validation_design) @ returns[family]["validation"])[1]
        market_betas[family] = beta
        hedged[family] = returns[family]["development"] - market["development"][:, None] * beta[None, :]

    family_results = {
        family: evaluate_period(
            returns[family]["development"], hedged[family], factors["development"], market["development"],
            loadings, completion_loadings,
        )
        for family in FAMILIES
    }
    for family, row in family_results.items():
        row["ff5_plus_momentum"] = metrics(
            benchmark["development"], returns[family]["development"], benchmark_loading,
            geometry(returns[family]["development"]),
        )
    changed_best_count = sum(row["best_dimension"]["0.0"] != row["best_dimension"]["1.0"] for row in family_results.values())
    attenuation_count = sum(row["market_direction_attenuation"] > 0 for row in family_results.values())
    completion_count = sum(
        row["completion_minus_k1"]["raw_moment"] < 0 and row["completion_minus_k1"]["alpha"] < 0
        for row in family_results.values()
    )
    unique_best = sorted({value for row in family_results.values() for value in row["best_dimension"].values()})

    subperiods = {"2010_2014": slice(0, 60), "2015_2019": slice(60, 120)}
    subperiod_results = {
        name: {
            family: evaluate_period(
                returns[family]["development"][slc], hedged[family][slc],
                {k: {seed: factors["development"][k][seed][slc] for seed in SEEDS} for k in KS},
                market["development"][slc], loadings, completion_loadings,
            )
            for family in FAMILIES
        }
        for name, slc in subperiods.items()
    }
    for name, slc in subperiods.items():
        for family, row in subperiod_results[name].items():
            row["ff5_plus_momentum"] = metrics(
                benchmark["development"][slc], returns[family]["development"][slc], benchmark_loading,
                geometry(returns[family]["development"][slc]),
            )

    rng = np.random.default_rng(args.bootstrap_seed)
    bootstrap_draws = {family: [] for family in FAMILIES}
    for _ in range(args.bootstrap_draws):
        index = circular_indices(120, 12, rng)
        for family in FAMILIES:
            row = evaluate_period(
                returns[family]["development"][index], hedged[family][index],
                {k: {seed: factors["development"][k][seed][index] for seed in SEEDS} for k in KS},
                market["development"][index], loadings, completion_loadings,
            )
            bootstrap_draws[family].append({
                "A_raw": row["A_raw"],
                "attenuation": row["market_direction_attenuation"],
                "completion_raw": row["completion_minus_k1"]["raw_moment"],
                "completion_alpha": row["completion_minus_k1"]["alpha"],
            })
    bootstrap = {}
    for family, rows in bootstrap_draws.items():
        bootstrap[family] = {}
        for key in rows[0]:
            values = [row[key] for row in rows]
            bootstrap[family][key] = {
                "ci95": np.quantile(values, [0.025, 0.975]).tolist(),
                "nonpositive_fraction": float(np.mean(np.asarray(values) <= 0)),
            }

    hypotheses = {
        "H1_geometry_dependence": {"count": int(changed_best_count), "required": 4, "supported": bool(changed_best_count >= 4)},
        "H2_market_direction": {
            "positive_attenuation_count": int(attenuation_count),
            "median_attenuation": float(np.median([row["market_direction_attenuation"] for row in family_results.values()])),
            "supported": bool(attenuation_count >= 4 and np.median([row["market_direction_attenuation"] for row in family_results.values()]) > 0),
        },
        "H3_economic_completion": {"joint_improvement_count": int(completion_count), "required": 4, "supported": bool(completion_count >= 4)},
        "H4_asset_family_dependence": {"unique_best_dimensions": unique_best, "required_count": 3, "supported": bool(len(unique_best) >= 3)},
    }
    result = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "source_git_revision": args.source_git_revision,
        "parent_run_id": "P1-G1-V020-R001",
        "sealed_period_accessed": False,
        "asset_snapshot_id": "ken-french-external-six-families-2026-07-v1",
        "periods": {key: [value[0], value[-1], len(value)] for key, value in months.items()},
        "frozen_models_retrained": False,
        "market_beta_summary": {
            family: {"minimum": float(beta.min()), "median": float(np.median(beta)), "maximum": float(beta.max())}
            for family, beta in market_betas.items()
        },
        "family_results": family_results,
        "subperiod_results": subperiod_results,
        "bootstrap": {"draws": args.bootstrap_draws, "block_months": 12, "families": bootstrap},
        "preregistered_hypotheses": hypotheses,
        "all_primary_hypotheses_supported": bool(all(row["supported"] for row in hypotheses.values())),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(hypotheses, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
