#!/usr/bin/env python3
"""Test whether validation-estimated market-beta hedging removes the geometry reversal."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sweep-root", type=Path, required=True)
    parser.add_argument("--portfolios-25", type=Path, required=True)
    parser.add_argument("--industries-49", type=Path, required=True)
    parser.add_argument("--ff5", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--validation-start", default="2000-01")
    parser.add_argument("--validation-end", default="2009-12")
    parser.add_argument("--development-start", default="2010-01")
    parser.add_argument("--development-end", default="2019-12")
    parser.add_argument("--parent-run-id", default="P1-G1-V015-R001")
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


def basis_diagnostics(returns: np.ndarray) -> dict[str, float]:
    values, vectors, _ = geometry(returns)
    equal = np.ones(returns.shape[1]) / np.sqrt(returns.shape[1])
    return {
        "largest_eigenvalue_second_moment_share": float(values[-1] / values.sum()),
        "largest_direction_absolute_cosine_with_equal_weight_vector": float(abs(vectors[:, -1] @ equal)),
        "largest_direction_same_sign_asset_share": float(max(np.mean(vectors[:, -1] > 0), np.mean(vectors[:, -1] < 0))),
    }


def main() -> int:
    args = parse_args()
    _, raw25 = base.read_first_value_weighted_monthly(args.portfolios_25, 25, "size_bm::")
    _, raw49 = base.read_first_value_weighted_monthly(args.industries_49, 49, "industry::")
    ff5 = base.read_factor_file(args.ff5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    risk_free = {month: row[-1] for month, row in ff5.items()}
    validation_months = sorted(month for month in raw25 if args.validation_start <= month <= args.validation_end)
    development_months = sorted(month for month in raw25 if args.development_start <= month <= args.development_end)
    if len(validation_months) != 120 or len(development_months) != 120:
        raise ValueError("Expected 120 characteristic months in validation and development")
    original = {}
    for label, months in (("validation", validation_months), ("development", development_months)):
        original[label] = np.column_stack([
            base.asset_matrix(months, raw25, risk_free),
            base.asset_matrix(months, raw49, risk_free),
        ])
    market = {
        "validation": np.asarray([ff5[base.next_month(month)][0] for month in validation_months]),
        "development": np.asarray([ff5[base.next_month(month)][0] for month in development_months]),
    }
    design = np.column_stack([np.ones(len(validation_months)), market["validation"]])
    market_betas = (np.linalg.pinv(design) @ original["validation"])[1]
    hedged = original["development"] - market["development"][:, None] * market_betas[None, :]
    payoff_sets = {"original": original["development"], "market_beta_hedged": hedged}
    systems = {label: geometry(returns) for label, returns in payoff_sets.items()}

    model_moments: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    for seed in SEEDS:
        model_moments[seed] = {}
        for count in (1, 5):
            mapping = multi.read_factor_matrix(args.sweep_root / f"factors-k{count}-seed{seed}" / "monthly_factor_returns.csv")
            validation_factors = multi.align_matrix(validation_months, mapping)
            development_factors = multi.align_matrix(development_months, mapping)
            loading = estimands.self_pricing_loadings(validation_factors)
            model_moments[seed][count] = {
                label: estimands.moment_vector(development_factors, returns, loading)
                for label, returns in payoff_sets.items()
            }

    seed_results = {}
    aggregate = {}
    for label in payoff_sets:
        aggregate[label] = {}
        seed_curves = []
        for seed in SEEDS:
            gaps = []
            for gamma in GAMMAS:
                distances = [
                    squared_distance(model_moments[seed][count][label], *systems[label], float(gamma))
                    for count in (1, 5)
                ]
                gaps.append(distances[0] - distances[1])
            seed_curves.append(gaps)
            seed_results.setdefault(str(seed), {})[label] = {
                "squared_distance_gap_by_gamma": {str(gamma): float(gap) for gamma, gap in zip(GAMMAS, gaps)},
                "crossing_gamma": crossing(gaps),
            }
        seed_curves_array = np.asarray(seed_curves)
        means = seed_curves_array.mean(axis=0)
        standard_errors = seed_curves_array.std(axis=0, ddof=1) / np.sqrt(len(SEEDS))
        aggregate[label] = {
            "mean_squared_distance_gap_by_gamma": {str(gamma): float(value) for gamma, value in zip(GAMMAS, means)},
            "standard_error_by_gamma": {str(gamma): float(value) for gamma, value in zip(GAMMAS, standard_errors)},
            "crossing_gamma": crossing(means.tolist()),
            "has_sign_change": bool(np.min(means) < 0 < np.max(means)),
        }

    original_gamma0 = aggregate["original"]["mean_squared_distance_gap_by_gamma"]["0.0"]
    hedged_gamma0 = aggregate["market_beta_hedged"]["mean_squared_distance_gap_by_gamma"]["0.0"]
    reduction = float((original_gamma0 - hedged_gamma0) / original_gamma0)
    seed_reductions = [
        seed_results[str(seed)]["original"]["squared_distance_gap_by_gamma"]["0.0"]
        - seed_results[str(seed)]["market_beta_hedged"]["squared_distance_gap_by_gamma"]["0.0"]
        for seed in SEEDS
    ]
    smaller_count = int(sum(value > 0 for value in seed_reductions))
    endpoint_change = {
        label: float(
            row["mean_squared_distance_gap_by_gamma"]["0.0"]
            - row["mean_squared_distance_gap_by_gamma"]["1.0"]
        )
        for label, row in aggregate.items()
    }
    endpoint_change_attenuation = float(
        abs(endpoint_change["original"]) - abs(endpoint_change["market_beta_hedged"])
    )
    supported = bool(
        reduction >= 0.75
        and not aggregate["market_beta_hedged"]["has_sign_change"]
        and smaller_count >= 4
    )
    result = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "source_git_revision": args.source_git_revision,
        "parent_run_id": args.parent_run_id,
        "sealed_period_accessed": False,
        "gammas": GAMMAS.tolist(),
        "market_beta_estimation": "2000-2009 intercept plus Mkt-RF; beta frozen for 2010-2019 hedging",
        "market_beta_summary": {
            "minimum": float(market_betas.min()),
            "median": float(np.median(market_betas)),
            "maximum": float(market_betas.max()),
        },
        "basis_diagnostics": {label: basis_diagnostics(returns) for label, returns in payoff_sets.items()},
        "seed_results": seed_results,
        "aggregate": aggregate,
        "endpoint_change_A": endpoint_change,
        "endpoint_change_market_attenuation": endpoint_change_attenuation,
        "v022_primary_market_mechanism_supported": endpoint_change_attenuation > 0,
        "preregistered_diagnostics": {
            "gamma_0_disadvantage_reduction_fraction": reduction,
            "seeds_with_smaller_gamma_0_disadvantage": smaller_count,
            "hedged_curve_has_no_sign_change": not aggregate["market_beta_hedged"]["has_sign_change"],
        },
        "preregistered_market_mode_explanation_supported": supported,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "original": aggregate["original"],
        "market_beta_hedged": aggregate["market_beta_hedged"],
        "diagnostics": result["preregistered_diagnostics"],
        "supported": supported,
        "basis": result["basis_diagnostics"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
