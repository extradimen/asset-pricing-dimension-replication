#!/usr/bin/env python3
"""Evaluate whether adding Mkt-RF completes a one-factor learned pricing span."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import audit_hj_alpha_estimands as estimands
import evaluate_multi_factor_teacher as multi
import evaluate_teacher_pricing as base

GAMMAS = np.round(np.arange(0, 1.0001, 0.05), 2)
SEEDS = (20260924, 20260925, 20260926, 20260927, 20260928)
MODEL_ORDER = ("market_only", "learned_k1", "learned_k1_plus_market", "learned_k5")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--portfolios-25", type=Path, required=True)
    parser.add_argument("--industries-49", type=Path, required=True)
    parser.add_argument("--ff5", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def geometry(returns: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    second = returns.T @ returns / len(returns)
    values, vectors = np.linalg.eigh(second)
    ridge = 1e-4 * np.trace(second) / second.shape[0]
    return values, vectors, float(ridge)


def squared_distance(moment: np.ndarray, values: np.ndarray, vectors: np.ndarray, ridge: float, gamma: float) -> float:
    diagonal = (values + ridge) ** (-gamma)
    diagonal *= len(values) / np.sum(diagonal * values)
    return float(12 * np.sum((vectors.T @ moment) ** 2 * diagonal))


def crossing(gaps: list[float]) -> float | None:
    for index in range(len(gaps) - 1):
        if gaps[index] == 0:
            return float(GAMMAS[index])
        if gaps[index] * gaps[index + 1] < 0:
            return float(GAMMAS[index] + 0.05 * gaps[index] / (gaps[index] - gaps[index + 1]))
    return None


def main() -> int:
    args = parse_args()
    _, raw25 = base.read_first_value_weighted_monthly(args.portfolios_25, 25, "size_bm::")
    _, raw49 = base.read_first_value_weighted_monthly(args.industries_49, 49, "industry::")
    ff5 = base.read_factor_file(args.ff5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    risk_free = {month: row[-1] for month, row in ff5.items()}
    periods = {
        "validation": sorted(month for month in raw25 if "2000-01" <= month <= "2009-12"),
        "development": sorted(month for month in raw25 if "2010-01" <= month <= "2019-12"),
    }
    returns = {
        period: np.column_stack([
            base.asset_matrix(months, raw25, risk_free),
            base.asset_matrix(months, raw49, risk_free),
        ])
        for period, months in periods.items()
    }
    market = {
        period: np.asarray([ff5[base.next_month(month)][0] for month in months])[:, None]
        for period, months in periods.items()
    }
    eigensystem = geometry(returns["development"])

    seed_results: dict[str, dict] = {}
    for seed in SEEDS:
        factor_matrices = {}
        for count in (1, 5):
            mapping = multi.read_factor_matrix(args.sweep_root / f"factors-k{count}-seed{seed}" / "monthly_factor_returns.csv")
            factor_matrices[count] = {
                period: multi.align_matrix(months, mapping)
                for period, months in periods.items()
            }
        models = {
            "market_only": market,
            "learned_k1": factor_matrices[1],
            "learned_k1_plus_market": {
                period: np.column_stack([factor_matrices[1][period], market[period]])
                for period in periods
            },
            "learned_k5": factor_matrices[5],
        }
        seed_results[str(seed)] = {}
        for name in MODEL_ORDER:
            factors = models[name]
            loading = estimands.self_pricing_loadings(factors["validation"])
            moment = estimands.moment_vector(factors["development"], returns["development"], loading)
            alpha = estimands.alpha_vector(factors["development"], returns["development"])
            self_moment = estimands.moment_vector(factors["development"], factors["development"], loading)
            seed_results[str(seed)][name] = {
                "factor_count": int(factors["development"].shape[1]),
                "mean_absolute_euler_moment_annualized": float(np.abs(moment).mean() * 12),
                "mean_absolute_alpha_annualized": float(np.abs(alpha).mean() * 12),
                "factor_self_pricing_mean_absolute_moment_annualized": float(np.abs(self_moment).mean() * 12),
                "squared_distance_by_gamma": {
                    str(gamma): squared_distance(moment, *eigensystem, float(gamma))
                    for gamma in GAMMAS
                },
            }

    aggregate = {}
    for name in MODEL_ORDER:
        aggregate[name] = {
            "mean_squared_distance_by_gamma": {},
            "standard_error_by_gamma": {},
            "mean_absolute_euler_moment_annualized": float(np.mean([
                seed_results[str(seed)][name]["mean_absolute_euler_moment_annualized"] for seed in SEEDS
            ])),
            "mean_absolute_alpha_annualized": float(np.mean([
                seed_results[str(seed)][name]["mean_absolute_alpha_annualized"] for seed in SEEDS
            ])),
            "factor_self_pricing_mean_absolute_moment_annualized": float(np.mean([
                seed_results[str(seed)][name]["factor_self_pricing_mean_absolute_moment_annualized"] for seed in SEEDS
            ])),
        }
        for gamma in GAMMAS:
            values = np.asarray([
                seed_results[str(seed)][name]["squared_distance_by_gamma"][str(gamma)] for seed in SEEDS
            ])
            aggregate[name]["mean_squared_distance_by_gamma"][str(gamma)] = float(values.mean())
            aggregate[name]["standard_error_by_gamma"][str(gamma)] = float(values.std(ddof=1) / np.sqrt(len(values)))

    paired = {}
    original_curves = []
    completed_curves = []
    for seed in SEEDS:
        original = []
        completed = []
        for gamma in GAMMAS:
            k5 = seed_results[str(seed)]["learned_k5"]["squared_distance_by_gamma"][str(gamma)]
            original.append(seed_results[str(seed)]["learned_k1"]["squared_distance_by_gamma"][str(gamma)] - k5)
            completed.append(seed_results[str(seed)]["learned_k1_plus_market"]["squared_distance_by_gamma"][str(gamma)] - k5)
        original_curves.append(original)
        completed_curves.append(completed)
        paired[str(seed)] = {
            "original_k1_minus_k5_gamma_0": float(original[0]),
            "completed_k1_market_minus_k5_gamma_0": float(completed[0]),
            "completed_k1_market_minus_k5_gamma_1": float(completed[-1]),
            "completed_crossing_gamma": crossing(completed),
        }
    original_mean = np.mean(original_curves, axis=0)
    completed_mean = np.mean(completed_curves, axis=0)
    closure = float((original_mean[0] - completed_mean[0]) / original_mean[0])
    nonpositive_seed_count = int(sum(row["completed_k1_market_minus_k5_gamma_0"] <= 0 for row in paired.values()))
    supported = bool(closure >= 0.75 and completed_mean[0] <= 0 and completed_mean[-1] <= 0 and nonpositive_seed_count >= 4)
    result = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "source_git_revision": args.source_git_revision,
        "parent_run_id": "P1-G1-V015-R001",
        "sealed_period_accessed": False,
        "gammas": GAMMAS.tolist(),
        "aggregate": aggregate,
        "seed_results": seed_results,
        "paired_primary_contrast": paired,
        "primary_mean_gap_by_gamma": {
            str(gamma): {
                "original_k1_minus_k5": float(original),
                "completed_k1_market_minus_k5": float(completed),
            }
            for gamma, original, completed in zip(GAMMAS, original_mean, completed_mean)
        },
        "preregistered_diagnostics": {
            "gamma_0_gap_closure_fraction": closure,
            "completed_mean_gap_gamma_0": float(completed_mean[0]),
            "completed_mean_gap_gamma_1": float(completed_mean[-1]),
            "completed_gamma_0_nonpositive_seed_count": nonpositive_seed_count,
            "completed_mean_crossing_gamma": crossing(completed_mean.tolist()),
        },
        "preregistered_market_completion_supported": supported,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "aggregate_endpoints": {
            name: {
                "gamma_0": aggregate[name]["mean_squared_distance_by_gamma"]["0.0"],
                "gamma_1": aggregate[name]["mean_squared_distance_by_gamma"]["1.0"],
                "alpha": aggregate[name]["mean_absolute_alpha_annualized"],
                "raw_moment": aggregate[name]["mean_absolute_euler_moment_annualized"],
            }
            for name in MODEL_ORDER
        },
        "diagnostics": result["preregistered_diagnostics"],
        "supported": supported,
        "paired": paired,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
