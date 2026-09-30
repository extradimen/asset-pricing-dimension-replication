#!/usr/bin/env python3
"""Run the frozen P1-G3-V001 synthetic pricing-dimension audit."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/paper1/P1-G3-V001.json"
EXPECTED_EXPERIMENT = "P1-G3-V001"


def toeplitz_correlation(size: int, rho: float) -> np.ndarray:
    indices = np.arange(size)
    return rho ** np.abs(indices[:, None] - indices[None, :])


def make_loading_matrix(
    rng: np.random.Generator,
    asset_count: int,
    max_dimension: int,
    true_dimension: int,
    spanning: str,
    weak_tail_scale: float,
) -> np.ndarray:
    raw = rng.standard_normal((asset_count, max_dimension))
    q, _ = np.linalg.qr(raw, mode="reduced")
    scales = np.ones(max_dimension)
    if spanning == "weak_tail":
        first_weak = math.ceil(true_dimension / 2)
        scales[first_weak:true_dimension] = weak_tail_scale
    elif spanning != "full":
        raise ValueError(f"unknown spanning regime: {spanning}")
    return q * np.sqrt(asset_count) * scales


def geometry_matrix(covariance: np.ndarray, geometry: str, ridge_share: float) -> np.ndarray:
    size = covariance.shape[0]
    if geometry == "equal":
        return np.eye(size) / size
    if geometry != "hj":
        raise ValueError(f"unknown geometry: {geometry}")
    ridge = ridge_share * float(np.trace(covariance) / size)
    inverse = np.linalg.inv(covariance + ridge * np.eye(size))
    return inverse / float(np.trace(inverse))


def projection_matrix(loadings: np.ndarray, weight: np.ndarray, dimension: int) -> np.ndarray:
    design = loadings[:, :dimension]
    gram = design.T @ weight @ design
    return design @ np.linalg.solve(gram, design.T @ weight)


def quadratic_losses(residuals: np.ndarray, weight: np.ndarray) -> np.ndarray:
    return np.einsum("ri,ij,rj->r", residuals, weight, residuals, optimize=True)


def select_dimensions(losses: np.ndarray, candidates: list[int]) -> np.ndarray:
    # np.argmin returns the first minimum, implementing the frozen smaller-K tie break.
    return np.asarray(candidates, dtype=int)[np.argmin(losses, axis=1)]


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return center - half, center + half


def mean_difference_interval(values: Iterable[float]) -> tuple[float, float, float]:
    array = np.asarray(list(values), dtype=float)
    mean = float(np.mean(array))
    if len(array) <= 1:
        return mean, float("nan"), float("nan")
    standard_error = float(np.std(array, ddof=1) / np.sqrt(len(array)))
    return mean, mean - 1.959963984540054 * standard_error, mean + 1.959963984540054 * standard_error


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _rate_row(base: dict, selected: np.ndarray, true_dimension: int) -> dict:
    flat = selected.reshape(-1)
    total = int(flat.size)
    exact = int(np.sum(flat == true_dimension))
    under = int(np.sum(flat < true_dimension))
    over = int(np.sum(flat > true_dimension))
    low, high = wilson_interval(exact, total)
    return {
        **base,
        "observations": total,
        "exact_count": exact,
        "exact_rate": exact / total,
        "exact_ci_low": low,
        "exact_ci_high": high,
        "under_rate": under / total,
        "over_rate": over / total,
        "mean_absolute_error": float(np.mean(np.abs(flat - true_dimension))),
        "median_selected_k": float(np.median(flat)),
    }


def run_simulation(config: dict) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rng = np.random.default_rng(int(config["random_seed"]))
    repetitions = int(config["monte_carlo_repetitions"])
    max_dimension = max(config["candidate_dimensions"])
    candidates = [int(value) for value in config["candidate_dimensions"]]
    family_count = int(config["asset_families"])
    factor_vols = np.linspace(*map(float, config["factor_volatility_range"]), max_dimension)
    idio_vol = float(config["idiosyncratic_volatility"])

    condition_rows: list[dict] = []
    frequency_rows: list[dict] = []
    disagreement_rows: list[dict] = []
    selection_cache: dict[tuple, np.ndarray] = {}

    condition_number = 0
    total_conditions = (
        len(config["true_dimensions"])
        * len(config["months_per_window"])
        * len(config["asset_counts"])
        * len(config["signal_levels"])
        * len(config["factor_correlations"])
        * len(config["spanning_regimes"])
    )
    for true_dimension in config["true_dimensions"]:
        signs = np.where(np.arange(max_dimension) % 2 == 0, 1.0, -1.0)
        for months in config["months_per_window"]:
            for asset_count in config["asset_counts"]:
                for signal_name, signal_value in config["signal_levels"].items():
                    prices = np.zeros(max_dimension)
                    prices[:true_dimension] = float(signal_value) * signs[:true_dimension]
                    for rho in config["factor_correlations"]:
                        factor_covariance = (
                            factor_vols[:, None]
                            * toeplitz_correlation(max_dimension, float(rho))
                            * factor_vols[None, :]
                        )
                        for spanning in config["spanning_regimes"]:
                            condition_number += 1
                            family_selections = {
                                (geometry, evaluation): np.empty((repetitions, family_count), dtype=int)
                                for geometry in config["geometries"]
                                for evaluation in config["evaluation_modes"]
                            }
                            for family in range(family_count):
                                loadings = make_loading_matrix(
                                    rng,
                                    int(asset_count),
                                    max_dimension,
                                    int(true_dimension),
                                    spanning,
                                    float(config["weak_tail_loading_scale"]),
                                )
                                mean = loadings @ prices
                                covariance = loadings @ factor_covariance @ loadings.T + idio_vol**2 * np.eye(asset_count)
                                mean_covariance = covariance / int(months)
                                training_means = rng.multivariate_normal(mean, mean_covariance, size=repetitions)
                                evaluation_means = rng.multivariate_normal(mean, mean_covariance, size=repetitions)
                                for geometry in config["geometries"]:
                                    weight = geometry_matrix(covariance, geometry, float(config["hj_ridge_share"]))
                                    oracle_losses = np.empty((repetitions, len(candidates)))
                                    feasible_losses = np.empty_like(oracle_losses)
                                    for candidate_index, candidate in enumerate(candidates):
                                        projector = projection_matrix(loadings, weight, candidate)
                                        predicted = training_means @ projector.T
                                        oracle_losses[:, candidate_index] = quadratic_losses(mean - predicted, weight)
                                        feasible_losses[:, candidate_index] = quadratic_losses(evaluation_means - predicted, weight)
                                    family_selections[(geometry, "oracle")][:, family] = select_dimensions(oracle_losses, candidates)
                                    family_selections[(geometry, "feasible")][:, family] = select_dimensions(feasible_losses, candidates)

                            common = {
                                "true_dimension": int(true_dimension),
                                "months": int(months),
                                "asset_count": int(asset_count),
                                "signal": signal_name,
                                "rho": float(rho),
                                "spanning": spanning,
                            }
                            for (geometry, evaluation), selected in family_selections.items():
                                key = (
                                    int(true_dimension), int(months), int(asset_count), signal_name,
                                    float(rho), spanning, geometry, evaluation,
                                )
                                selection_cache[key] = selected.copy()
                                condition_rows.append(_rate_row({**common, "geometry": geometry, "evaluation": evaluation}, selected, int(true_dimension)))
                                counts = Counter(selected.reshape(-1).tolist())
                                for candidate in candidates:
                                    low, high = wilson_interval(counts[candidate], repetitions * family_count)
                                    frequency_rows.append({
                                        **common,
                                        "geometry": geometry,
                                        "evaluation": evaluation,
                                        "candidate_dimension": candidate,
                                        "selected_count": counts[candidate],
                                        "selection_rate": counts[candidate] / (repetitions * family_count),
                                        "selection_ci_low": low,
                                        "selection_ci_high": high,
                                    })
                                diverse = np.asarray([len(set(row.tolist())) > 1 for row in selected])
                                unanimous = np.asarray([len(set(row.tolist())) == 1 for row in selected])
                                disagreement_rows.append({
                                    **common,
                                    "geometry": geometry,
                                    "evaluation": evaluation,
                                    "family_divergence_rate": float(np.mean(diverse)),
                                    "family_unanimity_rate": float(np.mean(unanimous)),
                                    "mean_unique_dimensions": float(np.mean([len(set(row.tolist())) for row in selected])),
                                })
                            for evaluation in config["evaluation_modes"]:
                                equal = family_selections[("equal", evaluation)]
                                hj = family_selections[("hj", evaluation)]
                                disagreement_rows.append({
                                    **common,
                                    "geometry": "equal_vs_hj",
                                    "evaluation": evaluation,
                                    "family_divergence_rate": float(np.mean(equal != hj)),
                                    "family_unanimity_rate": float("nan"),
                                    "mean_unique_dimensions": float("nan"),
                                })
                            print(f"condition {condition_number}/{total_conditions} complete", flush=True)

    hypothesis_rows = build_hypothesis_summary(condition_rows, disagreement_rows)
    return condition_rows, frequency_rows, disagreement_rows, hypothesis_rows


def _indexed(rows: list[dict], value_name: str, omit: str) -> dict[tuple, float]:
    dimensions = [
        "true_dimension", "months", "asset_count", "signal", "rho", "spanning", "geometry", "evaluation"
    ]
    keys = [name for name in dimensions if name != omit]
    return {tuple(row[name] for name in keys): float(row[value_name]) for row in rows}


def _paired_differences(rows: list[dict], value: str, field: str, left, right, filters: dict | None = None) -> list[float]:
    filters = filters or {}
    subset = [row for row in rows if all(row[name] == expected for name, expected in filters.items())]
    left_rows = [row for row in subset if row[field] == left]
    right_rows = [row for row in subset if row[field] == right]
    left_index = _indexed(left_rows, value, field)
    right_index = _indexed(right_rows, value, field)
    shared = sorted(set(left_index) & set(right_index), key=str)
    if not shared:
        raise ValueError(f"no matched cells for {field}: {left} versus {right}")
    return [left_index[key] - right_index[key] for key in shared]


def build_hypothesis_summary(condition_rows: list[dict], disagreement_rows: list[dict]) -> list[dict]:
    tests: list[tuple[str, str, list[float], str]] = []
    tests.append((
        "H1_sample_length",
        "exact_rate_T600_minus_T72_in_full_strong_rho0",
        _paired_differences(condition_rows, "exact_rate", "months", 600, 72, {"signal": "strong", "spanning": "full", "rho": 0.0}),
        "positive",
    ))
    tests.append((
        "H2_weak_prices",
        "exact_rate_strong_minus_weak",
        _paired_differences(condition_rows, "exact_rate", "signal", "strong", "weak"),
        "positive",
    ))
    tests.append((
        "H2_weak_prices",
        "under_rate_weak_minus_strong",
        _paired_differences(condition_rows, "under_rate", "signal", "weak", "strong"),
        "positive",
    ))
    tests.append((
        "H3_incomplete_span",
        "exact_rate_full_minus_weak_tail",
        _paired_differences(condition_rows, "exact_rate", "spanning", "full", "weak_tail"),
        "positive",
    ))
    family_rows = [row for row in disagreement_rows if row["geometry"] != "equal_vs_hj"]
    tests.append((
        "H3_incomplete_span",
        "family_divergence_weak_tail_minus_full",
        _paired_differences(family_rows, "family_divergence_rate", "spanning", "weak_tail", "full"),
        "positive",
    ))
    geometry_rows = [row for row in disagreement_rows if row["geometry"] == "equal_vs_hj"]
    tests.append((
        "H4_geometry",
        "geometry_disagreement_level",
        [float(row["family_divergence_rate"]) for row in geometry_rows],
        "positive",
    ))
    baseline_geometry = {
        (row["true_dimension"], row["months"], row["asset_count"], row["signal"], row["evaluation"]):
        float(row["family_divergence_rate"])
        for row in geometry_rows if row["rho"] == 0.0 and row["spanning"] == "full"
    }
    complex_differences = []
    for row in geometry_rows:
        if row["rho"] == 0.0 and row["spanning"] == "full":
            continue
        key = (row["true_dimension"], row["months"], row["asset_count"], row["signal"], row["evaluation"])
        complex_differences.append(float(row["family_divergence_rate"]) - baseline_geometry[key])
    tests.append((
        "H4_geometry",
        "geometry_disagreement_complex_minus_baseline",
        complex_differences,
        "positive",
    ))
    tests.append((
        "H5_oracle_vs_feasible",
        "exact_rate_oracle_minus_feasible",
        _paired_differences(condition_rows, "exact_rate", "evaluation", "oracle", "feasible"),
        "nonnegative",
    ))
    output = []
    for hypothesis, estimand, differences, expected in tests:
        mean, low, high = mean_difference_interval(differences)
        passes = mean > 0 if expected == "positive" else mean >= 0
        output.append({
            "hypothesis": hypothesis,
            "estimand": estimand,
            "matched_cells": len(differences),
            "mean_difference": mean,
            "ci_low": low,
            "ci_high": high,
            "expected_direction": expected,
            "point_direction_pass": bool(passes),
        })
    return output


def validate_config(config: dict) -> None:
    if config.get("experiment_id") != EXPECTED_EXPERIMENT:
        raise ValueError("unexpected experiment ID")
    if not config.get("protocol_frozen"):
        raise ValueError("protocol must be frozen")
    if not config.get("synthetic_only") or config.get("sealed_outputs_allowed"):
        raise ValueError("P1-G3-V001 must be synthetic-only and barred from sealed outputs")
    if config.get("candidate_dimensions") != [1, 2, 3, 4, 5, 8]:
        raise ValueError("candidate dimension grid differs from frozen protocol")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    validate_config(config)
    output_root = ROOT / config["output_root"]
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit(f"formal output directory already populated: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    condition_rows, frequency_rows, disagreement_rows, hypothesis_rows = run_simulation(config)
    paths = {
        "condition_summary": output_root / "condition_summary.csv",
        "selection_frequencies": output_root / "selection_frequencies.csv",
        "disagreement_summary": output_root / "disagreement_summary.csv",
        "hypothesis_summary": output_root / "hypothesis_summary.csv",
    }
    write_csv(paths["condition_summary"], condition_rows)
    write_csv(paths["selection_frequencies"], frequency_rows)
    write_csv(paths["disagreement_summary"], disagreement_rows)
    write_csv(paths["hypothesis_summary"], hypothesis_rows)
    result_summary = {
        "schema_version": 1,
        "experiment_id": EXPECTED_EXPERIMENT,
        "run_id": "P1-G3-V001-R001",
        "status": "completed",
        "synthetic_only": True,
        "sealed_outputs_accessed": False,
        "conditions": len(condition_rows),
        "frequency_rows": len(frequency_rows),
        "disagreement_rows": len(disagreement_rows),
        "hypotheses": hypothesis_rows,
    }
    result_path = output_root / "result_summary.json"
    result_path.write_text(json.dumps(result_summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPECTED_EXPERIMENT,
        "run_id": "P1-G3-V001-R001",
        "config_path": str(config_path.relative_to(ROOT)),
        "config_sha256": sha256(config_path),
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [*paths.values(), result_path]
        ],
    }
    manifest_path = output_root / "output_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result_summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
