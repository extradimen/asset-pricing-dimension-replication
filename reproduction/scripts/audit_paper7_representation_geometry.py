#!/usr/bin/env python3
"""Independently audit returned paper-7 representation-geometry outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def fixed_effect_probe(rows: list[dict[str, str]], endpoint: str, permutations: int, seed: int) -> dict[str, float]:
    factor = np.array([int(row["factor_count"]) for row in rows])
    seeds = np.array([int(row["seed"]) for row in rows])
    x = np.array([float(row["geometry_value"]) for row in rows]); y = np.array([float(row[endpoint]) for row in rows])
    xw, yw = x.copy(), y.copy()
    for value in np.unique(factor):
        mask = factor == value; xw[mask] -= x[mask].mean(); yw[mask] -= y[mask].mean()
    observed = abs(float(np.corrcoef(xw, yw)[0, 1])); rng = np.random.default_rng(seed); exceed = 0
    for _ in range(permutations):
        permuted = x.copy()
        for value in np.unique(factor):
            indices = np.flatnonzero(factor == value); permuted[indices] = rng.permutation(permuted[indices])
        within = permuted.copy()
        for value in np.unique(factor):
            mask = factor == value; within[mask] -= permuted[mask].mean()
        exceed += abs(float(np.corrcoef(within, yw)[0, 1])) >= observed
    base_errors = []; geometry_errors = []
    for held_seed in np.unique(seeds):
        train = seeds != held_seed; test_indices = np.flatnonzero(~train)
        residual_x, residual_y = x.copy(), y.copy()
        for value in np.unique(factor):
            mask = train & (factor == value); residual_x[mask] -= x[mask].mean(); residual_y[mask] -= y[mask].mean()
        slope = float(np.dot(residual_x[train], residual_y[train]) / max(np.dot(residual_x[train], residual_x[train]), 1e-12))
        for index in test_indices:
            same = train & (factor == factor[index]); base = y[same].mean(); extended = base + slope * (x[index] - x[same].mean())
            base_errors.append((y[index] - base) ** 2); geometry_errors.append((y[index] - extended) ** 2)
    base_rmse = float(np.sqrt(np.mean(base_errors))); geometry_rmse = float(np.sqrt(np.mean(geometry_errors)))
    return {"permutation_p_two_sided": float((exceed + 1) / (permutations + 1)),
            "leave_one_seed_out_baseline_rmse": base_rmse, "leave_one_seed_out_geometry_rmse": geometry_rmse,
            "cross_validated_rmse_improvement_pct": float(100 * (base_rmse - geometry_rmse) / base_rmse)}


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return bool(np.isclose(left, right, atol=tolerance, rtol=tolerance))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    config = json.loads(args.config.read_text(encoding="utf-8")); source_config = json.loads(Path(config["source_config"]).read_text(encoding="utf-8"))
    if sha256(Path(config["source_config"])) != config["source_config_sha256"]:
        raise RuntimeError("source config hash mismatch")
    returned_summary = Path(config["returned_summary"])
    if sha256(returned_summary) != config["returned_summary_sha256"]:
        raise RuntimeError("returned A100 summary hash mismatch")
    source_manifest_path = args.run_dir / "output_manifest.json"
    if sha256(source_manifest_path) != config["source_output_manifest_sha256"]:
        raise RuntimeError("source output manifest hash mismatch")
    manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    hash_checks = {item["path"]: sha256(args.run_dir / item["path"]) == item["sha256"] for item in manifest["outputs"]}
    geometry = load_csv(args.run_dir / "monthly_geometry.csv"); drift = load_csv(args.run_dir / "subspace_drift.csv")
    cka = load_csv(args.run_dir / "layer_cka.csv"); models = load_csv(args.run_dir / "model_geometry_endpoints.csv")
    source_report = json.loads((args.run_dir / "quality_report.json").read_text(encoding="utf-8"))
    source_probe = json.loads((args.run_dir / "functional_probe.json").read_text(encoding="utf-8"))[0]
    recomputed_probe = fixed_effect_probe(models, "development_hj_loss", source_config["functional_probe"]["permutations"], source_config["functional_probe"]["seed"])
    raw_hidden3 = [row for row in geometry if row["layer"] == "hidden3" and row["normalization"] == "raw_centered"]
    high = np.array([float(row["participation_rank"]) for row in raw_hidden3 if row["high_volatility"] == "True"])
    other = np.array([float(row["participation_rank"]) for row in raw_hidden3 if row["high_volatility"] == "False"])
    raw = np.array([float(row["participation_rank"]) for row in raw_hidden3])
    zscore = np.array([float(row["participation_rank"]) for row in geometry if row["layer"] == "hidden3" and row["normalization"] == "feature_zscore"])
    layers = {layer: float(np.median([float(row["participation_rank"]) for row in geometry if row["layer"] == layer and row["normalization"] == "raw_centered"])) for layer in ["hidden1", "hidden2", "hidden3"]}
    gates = {
        "all_output_hashes": all(hash_checks.values()), "geometry_rows_32400": len(geometry) == 32400,
        "drift_rows_32130": len(drift) == 32130, "cka_rows_7200": len(cka) == 7200,
        "models_30": len(models) == 30, "six_factor_counts_five_seeds": len({(row["factor_count"], row["seed"]) for row in models}) == 30,
        "state_difference_reproduced": close(float(high.mean() - other.mean()), source_report["hidden3_participation_rank_high_vol_minus_other"]),
        "normalization_difference_reproduced": close(float(np.median(zscore) - np.median(raw)), source_report["hidden3_zscore_minus_raw_participation_rank"]),
        "functional_p_reproduced": close(recomputed_probe["permutation_p_two_sided"], source_probe["permutation_p_two_sided"]),
        "functional_cv_reproduced": close(recomputed_probe["cross_validated_rmse_improvement_pct"], source_probe["cross_validated_rmse_improvement_pct"]),
        "a100_cpu_canonical_summary_exact_match": (args.run_dir / "canonical_return_summary.json").read_bytes() == returned_summary.read_bytes(),
    }
    report = {"schema_version": 1, "experiment_id": config["experiment_id"], "run_id": args.run_id,
              "status": "completed" if all(gates.values()) else "completed_with_failed_gate", "source_run": str(args.run_dir),
              "returned_a100_summary": str(returned_summary), "returned_a100_summary_sha256": sha256(returned_summary),
              "hash_checks": hash_checks, "gates": gates, "raw_centered_layer_median_participation_rank": layers,
              "hidden3_minus_hidden1": layers["hidden3"] - layers["hidden1"], "recomputed_primary_probe": recomputed_probe}
    (args.output_dir / "audit_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
