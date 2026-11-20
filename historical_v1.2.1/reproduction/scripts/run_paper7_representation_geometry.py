#!/usr/bin/env python3
"""Extract and audit hidden-representation geometry from frozen paper-1 networks."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn

from paper7_geometry_core import (
    geometry_metrics,
    grassmann_distance,
    linear_cka,
    normalize_representation,
    pca_basis,
    sha256,
    spearman,
    write_json,
)


MODEL_RE = re.compile(r"factors-k(?P<k>\d+)-seed(?P<seed>\d+)")


class FrozenTeacher(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], factor_count: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        layers.extend([nn.Linear(previous, factor_count), nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def representations(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        result: dict[str, torch.Tensor] = {}
        value = x
        hidden_index = 0
        for layer in self.network:
            value = layer(value)
            if isinstance(layer, nn.SiLU):
                hidden_index += 1
                result[f"hidden{hidden_index}"] = value
        return result


def load_sample_payload(path: Path, expected_stocks: int) -> tuple[list[int], dict[int, dict[str, np.ndarray]], dict[int, dict[str, float | bool]], list[str]]:
    with np.load(path, allow_pickle=False) as payload:
        months = payload["target_months"].astype(np.int32)
        feature_months = payload["feature_months"].astype(np.int32)
        permno = payload["permno"].astype(np.int64)
        x = payload["x"].astype(np.float32)
        feature_names = [str(value) for value in payload["feature_names"]]
        market_return = payload["feature_market_return"].astype(np.float64)
        volatility = payload["trailing_12m_volatility"].astype(np.float64)
        high_volatility = payload["high_volatility"].astype(bool)
        down_market = payload["down_market"].astype(bool)
    if tuple(x.shape) != (120, expected_stocks, 172) or len(feature_names) != 172:
        raise RuntimeError(f"unexpected frozen sample shape {x.shape}")
    if not np.all(feature_months == months - 1) or not np.isfinite(x).all():
        raise RuntimeError("frozen sample alignment or finiteness check failed")
    samples = {int(month): {"permno": permno[index], "x": x[index], "feature_month": np.full(expected_stocks, feature_months[index], dtype=np.int32)} for index, month in enumerate(months)}
    states = {int(month): {"feature_market_return": float(market_return[index]), "trailing_12m_volatility": float(volatility[index]), "high_volatility": bool(high_volatility[index]), "down_market": bool(down_market[index])} for index, month in enumerate(months)}
    return [int(value) for value in months], samples, states, feature_names


def read_models(checkpoint_manifest: Path) -> list[dict[str, object]]:
    manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    result = []
    for item in manifest["checkpoints"]:
        path = Path(item["path"])
        match = MODEL_RE.search(path.as_posix())
        if not match:
            raise RuntimeError(path)
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"checkpoint hash mismatch: {path}")
        result.append({"path": path, "factor_count": int(match.group("k")), "seed": int(match.group("seed")), "sha256": item["sha256"]})
    if len(result) != 30:
        raise RuntimeError(f"expected 30 checkpoints, found {len(result)}")
    return result


def endpoint_from_checkpoint(path: Path) -> dict[str, float]:
    report = json.loads(path.with_name("quality_report.json").read_text(encoding="utf-8"))
    development = report["performance"]["development"]
    return {
        "development_hj_loss": float(development["normalized_hj_span_loss"]),
        "development_factor_sharpe": float(development["factor_span_sharpe"]),
        "development_hhi": float(development["mean_scaled_hhi"]),
    }


def fixed_effect_probe(rows: list[dict[str, float | int]], endpoint: str, permutations: int, seed: int) -> dict[str, float | int | str]:
    factor = np.array([int(row["factor_count"]) for row in rows])
    seeds = np.array([int(row["seed"]) for row in rows])
    x = np.array([float(row["geometry_value"]) for row in rows])
    y = np.array([float(row[endpoint]) for row in rows])
    x_within = x.copy(); y_within = y.copy()
    for value in np.unique(factor):
        mask = factor == value
        x_within[mask] -= x[mask].mean(); y_within[mask] -= y[mask].mean()
    slope = float(np.dot(x_within, y_within) / np.dot(x_within, x_within))
    observed = abs(float(np.corrcoef(x_within, y_within)[0, 1]))
    rng = np.random.default_rng(seed)
    exceed = 0
    for _ in range(permutations):
        permuted = x.copy()
        for value in np.unique(factor):
            mask = np.flatnonzero(factor == value)
            permuted[mask] = rng.permutation(permuted[mask])
        within = permuted.copy()
        for value in np.unique(factor):
            mask = factor == value
            within[mask] -= permuted[mask].mean()
        statistic = abs(float(np.corrcoef(within, y_within)[0, 1]))
        exceed += statistic >= observed
    baseline_errors = []; extended_errors = []
    for held_seed in np.unique(seeds):
        train = seeds != held_seed; test = ~train
        base_prediction = np.zeros(test.sum()); extended_prediction = np.zeros(test.sum())
        test_indices = np.flatnonzero(test)
        for output_index, row_index in enumerate(test_indices):
            same = train & (factor == factor[row_index])
            base_prediction[output_index] = y[same].mean()
            x_mean = x[same].mean(); y_mean = y[same].mean()
            residual_x = x.copy(); residual_y = y.copy()
            for value in np.unique(factor):
                mask = train & (factor == value)
                residual_x[mask] -= x[mask].mean()
                residual_y[mask] -= y[mask].mean()
            local_slope = float(np.dot(residual_x[train], residual_y[train]) / max(np.dot(residual_x[train], residual_x[train]), 1e-12))
            extended_prediction[output_index] = y_mean + local_slope * (x[row_index] - x_mean)
        baseline_errors.extend(np.square(y[test] - base_prediction)); extended_errors.extend(np.square(y[test] - extended_prediction))
    baseline_rmse = float(np.sqrt(np.mean(baseline_errors))); extended_rmse = float(np.sqrt(np.mean(extended_errors)))
    return {
        "endpoint": endpoint,
        "n_models": len(rows),
        "within_factor_count_slope": slope,
        "within_factor_count_correlation": float(np.corrcoef(x_within, y_within)[0, 1]),
        "within_factor_count_spearman": spearman(x_within, y_within),
        "permutation_p_two_sided": float((exceed + 1) / (permutations + 1)),
        "leave_one_seed_out_baseline_rmse": baseline_rmse,
        "leave_one_seed_out_geometry_rmse": extended_rmse,
        "cross_validated_rmse_improvement_pct": float(100.0 * (baseline_rmse - extended_rmse) / baseline_rmse),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    sample = Path(config["inputs"]["sample_payload"]); checkpoints = Path(config["inputs"]["checkpoint_manifest"])
    for path, expected in [(sample, config["inputs"]["sample_payload_sha256"]), (checkpoints, config["inputs"]["checkpoint_manifest_sha256"])]:
        if sha256(path) != expected:
            raise RuntimeError(f"input hash mismatch: {path}")
    months, samples, states, feature_columns = load_sample_payload(sample, config["sampling"]["stocks_per_month"])
    models = read_models(checkpoints)
    device = torch.device(args.device)
    metric_rows: list[dict[str, object]] = []
    drift_rows: list[dict[str, object]] = []
    cka_rows: list[dict[str, object]] = []
    model_summaries: list[dict[str, object]] = []
    annual_anchor = set(months[::12])

    for model_item in models:
        try:
            checkpoint = torch.load(model_item["path"], map_location="cpu", weights_only=False)
        except TypeError:
            checkpoint = torch.load(model_item["path"], map_location="cpu")
        model = FrozenTeacher(checkpoint["input_dim"], checkpoint["hidden"], checkpoint["factor_count"]).to(device)
        model.load_state_dict(checkpoint["model_state_dict"]); model.eval()
        endpoints = endpoint_from_checkpoint(model_item["path"])
        prior_basis: dict[tuple[str, str], np.ndarray] = {}
        model_metric_rows: list[dict[str, object]] = []
        with torch.no_grad():
            for target_month in months:
                x = torch.from_numpy(samples[target_month]["x"]).to(device)
                representations = {name: value.detach().cpu().numpy() for name, value in model.representations(x).items()}
                for layer, values in representations.items():
                    for normalization in config["geometry"]["normalizations"]:
                        neighbor = target_month in annual_anchor and normalization in config["geometry"]["neighbor_normalizations"]
                        metrics = geometry_metrics(values, normalization, neighbor_metrics=neighbor, curvature_seed=int(model_item["seed"]) + target_month)
                        row = {
                            "factor_count": model_item["factor_count"], "seed": model_item["seed"], "target_month": str(np.datetime64(target_month, "M")),
                            "feature_month": str(np.datetime64(int(samples[target_month]["feature_month"][0]), "M")), "layer": layer,
                            **states[target_month], **metrics,
                        }
                        metric_rows.append(row); model_metric_rows.append(row)
                        centered = normalize_representation(values, normalization)
                        basis = pca_basis(centered, config["geometry"]["trajectory_rank"])
                        key = (layer, normalization)
                        if key in prior_basis:
                            drift_rows.append({"factor_count": model_item["factor_count"], "seed": model_item["seed"], "target_month": row["target_month"], "layer": layer, "normalization": normalization, "grassmann_distance": grassmann_distance(prior_basis[key], basis)})
                        prior_basis[key] = basis
                cka_rows.extend([
                    {"factor_count": model_item["factor_count"], "seed": model_item["seed"], "target_month": str(np.datetime64(target_month, "M")), "layer_pair": "hidden1-hidden2", "linear_cka": linear_cka(representations["hidden1"], representations["hidden2"])},
                    {"factor_count": model_item["factor_count"], "seed": model_item["seed"], "target_month": str(np.datetime64(target_month, "M")), "layer_pair": "hidden2-hidden3", "linear_cka": linear_cka(representations["hidden2"], representations["hidden3"])},
                ])
        selected = [row for row in model_metric_rows if row["layer"] == "hidden3" and row["normalization"] == "raw_centered"]
        raw_pr = float(np.median([float(row["participation_rank"]) for row in selected]))
        raw_er = float(np.median([float(row["entropy_effective_rank"]) for row in selected]))
        raw_top = float(np.median([float(row["top_eigenvalue_share"]) for row in selected]))
        collapsed_rate = float(np.mean([bool(row["collapsed"]) for row in selected]))
        model_summaries.append({"factor_count": model_item["factor_count"], "seed": model_item["seed"], "geometry_value": raw_pr, "median_hidden3_raw_participation_rank": raw_pr, "median_hidden3_raw_entropy_rank": raw_er, "median_hidden3_raw_top_share": raw_top, "hidden3_collapse_rate": collapsed_rate, **endpoints})
        print(json.dumps({"completed_model": model_item["path"].parent.name, "elapsed_seconds": round(time.time() - started, 1)}), flush=True)

    outputs = [args.output_dir / name for name in ["monthly_geometry.csv", "subspace_drift.csv", "layer_cka.csv", "model_geometry_endpoints.csv"]]
    for path, rows in zip(outputs, [metric_rows, drift_rows, cka_rows, model_summaries]):
        write_csv(path, rows)
    probes = [fixed_effect_probe(model_summaries, endpoint, config["functional_probe"]["permutations"], config["functional_probe"]["seed"]) for endpoint in ["development_hj_loss", "development_factor_sharpe"]]
    write_json(args.output_dir / "functional_probe.json", probes)
    primary = probes[0]
    high = [float(row["participation_rank"]) for row in metric_rows if row["layer"] == "hidden3" and row["normalization"] == "raw_centered" and row["high_volatility"]]
    normal = [float(row["participation_rank"]) for row in metric_rows if row["layer"] == "hidden3" and row["normalization"] == "raw_centered" and not row["high_volatility"]]
    raw = [float(row["participation_rank"]) for row in metric_rows if row["layer"] == "hidden3" and row["normalization"] == "raw_centered"]
    zscore = [float(row["participation_rank"]) for row in metric_rows if row["layer"] == "hidden3" and row["normalization"] == "feature_zscore"]
    report = {
        "schema_version": 1, "experiment_id": config["experiment_id"], "run_id": args.run_id, "status": "completed", "evidence_class": config["evidence_class"],
        "device": str(device), "models": len(models), "target_months": len(months), "stocks_per_month": config["sampling"]["stocks_per_month"], "feature_columns": len(feature_columns),
        "geometry_rows": len(metric_rows), "subspace_drift_rows": len(drift_rows), "cka_rows": len(cka_rows), "collapse_rate": float(np.mean([bool(row["collapsed"]) for row in metric_rows])),
        "hidden3_participation_rank_high_vol_minus_other": float(np.mean(high) - np.mean(normal)),
        "hidden3_zscore_minus_raw_participation_rank": float(np.median(zscore) - np.median(raw)),
        "primary_functional_probe": primary,
        "functional_gate": {
            "criterion": "geometry must improve leave-one-seed-out HJ-loss RMSE by at least 5% and have within-K permutation p <= 0.05",
            "passed": bool(primary["cross_validated_rmse_improvement_pct"] >= 5.0 and primary["permutation_p_two_sided"] <= 0.05),
            "stop_if_failed": "Do not search alternative layers, metrics, states, or endpoints for a positive association; report the preregistered negative result."
        },
        "interpretation": "finite-scale hidden-representation geometry of frozen networks; not an economic compression frontier and not a claim of unique topological dimension",
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(args.output_dir / "quality_report.json", report)
    write_json(args.output_dir / "environment.json", {"created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform()})
    all_outputs = [*outputs, args.output_dir / "functional_probe.json", args.output_dir / "quality_report.json", args.output_dir / "environment.json"]
    write_json(args.output_dir / "output_manifest.json", {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(), "experiment_id": config["experiment_id"], "run_id": args.run_id, "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in [args.config, sample, checkpoints]], "outputs": [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in all_outputs]})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
