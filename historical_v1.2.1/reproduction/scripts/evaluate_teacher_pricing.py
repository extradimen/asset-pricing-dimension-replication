#!/usr/bin/env python3
"""Evaluate neural and non-neural factors on public test-asset pricing metrics."""

from __future__ import annotations

import argparse
import csv
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


VALID_START, VALID_END = "2000-01", "2009-12"
DEV_START, DEV_END = "2010-01", "2019-12"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed-root", type=Path, required=True)
    parser.add_argument("--linear-predictions", type=Path, required=True)
    parser.add_argument("--portfolios-25", type=Path, required=True)
    parser.add_argument("--industries-49", type=Path, required=True)
    parser.add_argument("--ff5", type=Path, required=True)
    parser.add_argument("--momentum", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=500)
    parser.add_argument("--bootstrap-seed", type=int, default=20260924)
    parser.add_argument("--experiment-id", default="P1-G1-V003")
    return parser.parse_args()


def zip_lines(path: Path) -> list[str]:
    with zipfile.ZipFile(path) as archive:
        members = [name for name in archive.namelist() if not name.endswith("/")]
        if len(members) != 1:
            raise ValueError(f"Expected one member in {path}")
        return archive.read(members[0]).decode("utf-8-sig", errors="strict").splitlines()


def read_first_value_weighted_monthly(path: Path, expected_assets: int, prefix: str) -> tuple[list[str], dict[str, np.ndarray]]:
    lines = zip_lines(path)
    marker = next(index for index, line in enumerate(lines) if "Average Value Weighted Returns -- Monthly" in line)
    header = next(csv.reader([lines[marker + 1]]))
    names = [f"{prefix}{value.strip()}" for value in header[1:]]
    if len(names) != expected_assets:
        raise ValueError(f"Expected {expected_assets} assets in {path}, found {len(names)}")
    result: dict[str, np.ndarray] = {}
    for line in lines[marker + 2 :]:
        fields = next(csv.reader([line])) if line else []
        if not fields or len(fields[0].strip()) != 6 or not fields[0].strip().isdigit():
            if result:
                break
            continue
        values = np.asarray([float(value.strip()) / 100.0 for value in fields[1:]], dtype=np.float64)
        if values.size != expected_assets:
            raise ValueError(f"Unexpected row width in {path}: {line[:80]}")
        values[(values <= -0.999) | ~np.isfinite(values)] = np.nan
        key = f"{fields[0].strip()[:4]}-{fields[0].strip()[4:]}"
        result[key] = values
    return names, result


def read_factor_file(path: Path, expected: list[str]) -> dict[str, np.ndarray]:
    lines = zip_lines(path)
    header_index = next(
        index
        for index, line in enumerate(lines)
        if (lambda fields: bool(fields) and fields[0].strip() == "" and all(name in fields[1:] for name in expected))(
            [value.strip() for value in next(csv.reader([line]))]
        )
    )
    header = [value.strip() for value in next(csv.reader([lines[header_index]]))[1:]]
    result: dict[str, np.ndarray] = {}
    for line in lines[header_index + 1 :]:
        fields = next(csv.reader([line])) if line else []
        if not fields or len(fields[0].strip()) != 6 or not fields[0].strip().isdigit():
            if result:
                break
            continue
        key = f"{fields[0].strip()[:4]}-{fields[0].strip()[4:]}"
        result[key] = np.asarray([float(value.strip()) / 100.0 for value in fields[1:]], dtype=np.float64)
    if header != expected:
        raise ValueError(f"Factor header mismatch in {path}: {header}")
    return result


def read_monthly_factor(path: Path) -> dict[str, float]:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = csv.DictReader(stream)
        return {row["month"]: float(row["factor_return"]) for row in rows}


def linear_factor(path: Path) -> dict[str, float]:
    table = pq.read_table(path, columns=["month", "ret_excess_fwd1", "prediction"])
    months = table.column("month").to_numpy().astype("datetime64[M]").astype(str)
    returns = table.column("ret_excess_fwd1").to_numpy().astype(np.float64)
    scores = table.column("prediction").to_numpy().astype(np.float64)
    order = np.argsort(months, kind="stable"); months = months[order]; returns = returns[order]; scores = scores[order]
    unique, starts, counts = np.unique(months, return_index=True, return_counts=True)
    result: dict[str, float] = {}
    for month, start, count in zip(unique, starts, counts):
        score = scores[start : start + count]; ret = returns[start : start + count]
        centered = score - score.mean(); weight = centered / max(np.abs(centered).sum(), 1e-12)
        result[str(month)] = float(weight @ ret)
    return result


def aligned(months: list[str], mapping: dict[str, float]) -> np.ndarray:
    return np.asarray([mapping[month] for month in months], dtype=np.float64)


def next_month(month: str) -> str:
    return str(np.datetime64(month, "M") + 1)


def asset_matrix(months: list[str], assets: dict[str, np.ndarray], rf: dict[str, float]) -> np.ndarray:
    realization_months = [next_month(month) for month in months]
    matrix = np.vstack([assets[month] for month in realization_months])
    rates = np.asarray([rf[month] for month in realization_months])[:, None]
    result = matrix - rates
    if not np.isfinite(result).all():
        raise ValueError("Missing test-asset return in evaluation period")
    return result


def fit_sdf_scale(factor: np.ndarray, returns: np.ndarray) -> float:
    mean_returns = returns.mean(axis=0)
    covariance_direction = (factor[:, None] * returns).mean(axis=0)
    denominator = float(covariance_direction @ covariance_direction)
    return float(covariance_direction @ mean_returns / denominator) if denominator > 0 else 0.0


def metrics(factor: np.ndarray, returns: np.ndarray, sdf_scale: float) -> dict[str, float]:
    moment = ((1.0 - sdf_scale * factor)[:, None] * returns).mean(axis=0)
    second = returns.T @ returns / returns.shape[0]
    pinv = np.linalg.pinv(second, rcond=1e-8)
    ridge = 1e-4 * float(np.trace(second) / second.shape[0])
    hj = math.sqrt(max(float(moment @ pinv @ moment), 0.0)) * math.sqrt(12.0)
    hj_ridge = math.sqrt(max(float(moment @ np.linalg.inv(second + ridge * np.eye(second.shape[0])) @ moment), 0.0)) * math.sqrt(12.0)
    design = np.column_stack([np.ones(factor.size), factor])
    coefficient = np.linalg.pinv(design) @ returns
    alpha = coefficient[0]
    mean = float(factor.mean()); std = float(factor.std(ddof=1))
    return {
        "sdf_scale": sdf_scale,
        "factor_sharpe": mean / std * math.sqrt(12.0) if std > 0 else float("nan"),
        "hj_pinv_annualized": hj,
        "hj_ridge_annualized": hj_ridge,
        "mean_absolute_pricing_moment_annualized": float(np.abs(moment).mean() * 12.0),
        "root_mean_square_pricing_moment_annualized": float(np.sqrt(np.mean(moment ** 2)) * 12.0),
        "mean_absolute_alpha_annualized": float(np.abs(alpha).mean() * 12.0),
        "maximum_absolute_alpha_annualized": float(np.abs(alpha).max() * 12.0),
        "effective_second_moment_rank": int(np.linalg.matrix_rank(second, tol=np.linalg.svd(second, compute_uv=False)[0] * 1e-8)),
    }


def evaluate_factor(factor_map: dict[str, float], periods: dict[str, list[str]], asset_sets: dict[str, dict[str, np.ndarray]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for asset_name, split_returns in asset_sets.items():
        valid_factor = aligned(periods["validation"], factor_map)
        development_factor = aligned(periods["development"], factor_map)
        scale = fit_sdf_scale(valid_factor, split_returns["validation"])
        output[asset_name] = {
            "validation": metrics(valid_factor, split_returns["validation"], scale),
            "development": metrics(development_factor, split_returns["development"], scale),
        }
    return output


def circular_block_indices(length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    pieces: list[np.ndarray] = []
    while sum(piece.size for piece in pieces) < length:
        start = int(rng.integers(0, length))
        pieces.append((start + np.arange(block)) % length)
    return np.concatenate(pieces)[:length]


def bootstrap_comparison(
    teacher: np.ndarray,
    baseline: np.ndarray,
    returns: np.ndarray,
    teacher_scale: float,
    baseline_scale: float,
    draws: int,
    rng: np.random.Generator,
) -> dict[str, object]:
    values = {"hj_ridge": [], "mean_absolute_alpha": []}
    for _ in range(draws):
        index = circular_block_indices(len(teacher), 12, rng)
        t = metrics(teacher[index], returns[index], teacher_scale)
        b = metrics(baseline[index], returns[index], baseline_scale)
        values["hj_ridge"].append((b["hj_ridge_annualized"] - t["hj_ridge_annualized"]) / b["hj_ridge_annualized"])
        values["mean_absolute_alpha"].append((b["mean_absolute_alpha_annualized"] - t["mean_absolute_alpha_annualized"]) / b["mean_absolute_alpha_annualized"])
    return {
        key: {
            "mean_relative_improvement": float(np.mean(items)),
            "ci95": [float(np.quantile(items, 0.025)), float(np.quantile(items, 0.975))],
        }
        for key, items in values.items()
    }


def main() -> int:
    args = parse_args()
    names25, raw25 = read_first_value_weighted_monthly(args.portfolios_25, 25, "size_bm::")
    names49, raw49 = read_first_value_weighted_monthly(args.industries_49, 49, "industry::")
    ff5 = read_factor_file(args.ff5, ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "RF"])
    momentum = read_factor_file(args.momentum, ["Mom"])
    rf = {month: values[-1] for month, values in ff5.items()}
    periods = {
        "validation": sorted(month for month in raw25 if VALID_START <= month <= VALID_END),
        "development": sorted(month for month in raw25 if DEV_START <= month <= DEV_END),
    }
    for split, months in periods.items():
        if len(months) != 120:
            raise ValueError(f"Expected 120 {split} months, found {len(months)}")
    asset_sets: dict[str, dict[str, np.ndarray]] = {}
    for label, source in [("size_bm_25", raw25), ("industry_49", raw49)]:
        asset_sets[label] = {split: asset_matrix(months, source, rf) for split, months in periods.items()}
    asset_sets["combined_74"] = {
        split: np.column_stack([asset_sets["size_bm_25"][split], asset_sets["industry_49"][split]])
        for split in periods
    }

    factors: dict[str, dict[str, float]] = {"linear_core92": linear_factor(args.linear_predictions)}
    all_months = periods["validation"] + periods["development"]
    ff_matrix = {
        month: np.concatenate([ff5[next_month(month)][:-1], momentum[next_month(month)]])
        for month in all_months
    }
    valid_ff = np.vstack([ff_matrix[month] for month in periods["validation"]])
    covariance = np.cov(valid_ff, rowvar=False)
    ridge = 1e-6 * np.trace(covariance) / covariance.shape[0]
    tangency_weight = np.linalg.solve(covariance + ridge * np.eye(covariance.shape[0]), valid_ff.mean(axis=0))
    factors["ff5_momentum_tangency"] = {month: float(ff_matrix[month] @ tangency_weight) for month in all_months}
    for seed_dir in sorted(args.seed_root.glob("seed-*")):
        if seed_dir.is_dir():
            factors[f"neural_{seed_dir.name}"] = read_monthly_factor(seed_dir / "monthly_factor_returns.csv")

    evaluations = {name: evaluate_factor(mapping, periods, asset_sets) for name, mapping in factors.items()}
    baseline_names = ["linear_core92", "ff5_momentum_tangency"]
    selected_baseline = min(
        baseline_names,
        key=lambda name: evaluations[name]["combined_74"]["validation"]["hj_ridge_annualized"],
    )
    baseline_map = factors[selected_baseline]
    baseline_valid = aligned(periods["validation"], baseline_map)
    baseline_dev = aligned(periods["development"], baseline_map)
    combined_valid = asset_sets["combined_74"]["validation"]
    combined_dev = asset_sets["combined_74"]["development"]
    baseline_scale = fit_sdf_scale(baseline_valid, combined_valid)
    baseline_metrics = evaluations[selected_baseline]["combined_74"]["development"]
    rng = np.random.default_rng(args.bootstrap_seed)
    seed_gate: dict[str, object] = {}
    for name, mapping in factors.items():
        if not name.startswith("neural_"):
            continue
        teacher_valid = aligned(periods["validation"], mapping)
        teacher_dev = aligned(periods["development"], mapping)
        teacher_scale = fit_sdf_scale(teacher_valid, combined_valid)
        teacher_metrics = evaluations[name]["combined_74"]["development"]
        point = {
            "hj_ridge_relative_improvement": (baseline_metrics["hj_ridge_annualized"] - teacher_metrics["hj_ridge_annualized"]) / baseline_metrics["hj_ridge_annualized"],
            "mean_absolute_alpha_relative_improvement": (baseline_metrics["mean_absolute_alpha_annualized"] - teacher_metrics["mean_absolute_alpha_annualized"]) / baseline_metrics["mean_absolute_alpha_annualized"],
            "factor_sharpe_difference": teacher_metrics["factor_sharpe"] - baseline_metrics["factor_sharpe"],
        }
        bootstrap = bootstrap_comparison(
            teacher_dev, baseline_dev, combined_dev, teacher_scale, baseline_scale,
            args.bootstrap_draws, rng,
        )
        seed_gate[name] = {"point_estimates": point, "block_bootstrap": bootstrap}
    direction_count = sum(
        int(item["point_estimates"]["hj_ridge_relative_improvement"] > 0 and item["point_estimates"]["mean_absolute_alpha_relative_improvement"] > 0)
        for item in seed_gate.values()
    )
    admission_pass = direction_count >= 4 and all(
        item["point_estimates"]["hj_ridge_relative_improvement"] >= 0.05
        and item["point_estimates"]["mean_absolute_alpha_relative_improvement"] >= 0.05
        and item["point_estimates"]["factor_sharpe_difference"] >= 0
        and (
            item["block_bootstrap"]["hj_ridge"]["ci95"][0] > 0
            or item["block_bootstrap"]["mean_absolute_alpha"]["ci95"][0] > 0
        )
        for item in seed_gate.values()
    )
    result = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "sealed_period_accessed": False,
        "periods": periods,
        "timing": "Each model row is indexed by characteristic month t and priced against official test-asset and benchmark-factor returns realized in month t+1.",
        "test_assets": {"size_bm_25": names25, "industry_49": names49},
        "sdf_scale_fit_period": "2000-01 through 2009-12 validation period",
        "hj_definition": "sqrt(g' E[RR']^{-1} g), annualized; ridge version uses 1e-4 times average second-moment eigenvalue",
        "evaluations": evaluations,
        "selected_validation_baseline": selected_baseline,
        "seed_gate": seed_gate,
        "positive_direction_seed_count": direction_count,
        "partial_admission_gate_passed": admission_pass,
        "gate_note": "Partial gate covers public 25 size-BM and 49 industry assets plus linear and FF5+momentum baselines. IPCA and model-out double sorts remain outstanding before final strong-teacher admission.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected_baseline": selected_baseline, "positive_direction_seed_count": direction_count, "partial_admission_gate_passed": admission_pass}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
