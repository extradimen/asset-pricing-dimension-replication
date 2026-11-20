#!/usr/bin/env python3
"""Fit and evaluate Paper 2 BL and BG exposure baselines."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch import nn


FACTORS = ["mkt_rf", "smb", "hml", "rmw", "cma", "mom"]
PAIR_NAMES = [(left, right) for index, left in enumerate(FACTORS) for right in FACTORS[index:]]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def required_moment_columns() -> list[str]:
    columns = ["n_days", "sum_y", "sum_y2"]
    columns += [f"sum_{factor}" for factor in FACTORS]
    columns += [f"sum_{factor}_y" for factor in FACTORS]
    columns += [f"sum_{left}_{right}" for left, right in PAIR_NAMES]
    return columns


def centered_moments(frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = frame["n_days"].to_numpy().astype(np.float32)
    sum_y = frame["sum_y"].to_numpy().astype(np.float64)
    sum_y2 = frame["sum_y2"].to_numpy().astype(np.float64)
    sum_f = np.column_stack([frame[f"sum_{factor}"].to_numpy() for factor in FACTORS]).astype(np.float64)
    sum_fy = np.column_stack([frame[f"sum_{factor}_y"].to_numpy() for factor in FACTORS]).astype(np.float64)
    cross = np.empty((len(frame), len(FACTORS), len(FACTORS)), dtype=np.float64)
    for left_index, left in enumerate(FACTORS):
        for right_index, right in enumerate(FACTORS[left_index:], left_index):
            values = frame[f"sum_{left}_{right}"].to_numpy().astype(np.float64)
            cross[:, left_index, right_index] = values
            cross[:, right_index, left_index] = values
    centered_y2 = sum_y2 - np.square(sum_y) / n
    centered_fy = sum_fy - sum_f * sum_y[:, None] / n[:, None]
    centered_ff = cross - np.einsum("ni,nj->nij", sum_f, sum_f) / n[:, None, None]
    return centered_y2.astype(np.float32), centered_fy.astype(np.float32), centered_ff.astype(np.float32)


def residual_sum_squares(
    beta: np.ndarray, centered_y2: np.ndarray, centered_fy: np.ndarray, centered_ff: np.ndarray
) -> np.ndarray:
    return centered_y2 - 2.0 * np.einsum("ni,ni->n", beta, centered_fy) + np.einsum(
        "ni,nij,nj->n", beta, centered_ff, beta
    )


class FeatureMap:
    def __init__(self, continuous: list[str], indicators: list[str], interactions: list[list[str]], knots: list[float]):
        self.continuous = continuous
        self.indicators = indicators
        self.interactions = interactions
        self.knots = np.asarray(knots, dtype=np.float32)
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None
        self.index = {name: offset for offset, name in enumerate(continuous)}

    def fit(self, frame: pl.DataFrame) -> None:
        values = frame.select(self.continuous).to_numpy().astype(np.float64)
        self.mean = np.nanmean(values, axis=0).astype(np.float32)
        self.scale = np.nanstd(values, axis=0).astype(np.float32)
        self.scale = np.where(self.scale > 1e-6, self.scale, 1.0).astype(np.float32)

    def standardized(self, frame: pl.DataFrame) -> np.ndarray:
        if self.mean is None or self.scale is None:
            raise RuntimeError("FeatureMap must be fitted on the training sample")
        values = frame.select(self.continuous).to_numpy().astype(np.float32)
        values = np.where(np.isfinite(values), values, self.mean)
        return ((values - self.mean) / self.scale).astype(np.float32)

    def transform(self, frame: pl.DataFrame, model: str) -> np.ndarray:
        continuous = self.standardized(frame)
        indicators = frame.select(self.indicators).to_numpy().astype(np.float32)
        interactions = np.column_stack([
            continuous[:, self.index[left]] * continuous[:, self.index[right]]
            for left, right in self.interactions
        ]).astype(np.float32)
        linear = np.column_stack([continuous, indicators, interactions]).astype(np.float32)
        if model == "BL":
            return linear
        if model != "BG":
            raise ValueError(f"Unknown model: {model}")
        hinges = np.maximum(continuous[:, :, None] - self.knots[None, None, :], 0.0)
        return np.column_stack([linear, hinges.reshape(len(frame), -1)]).astype(np.float32)

    def metadata(self) -> dict:
        if self.mean is None or self.scale is None:
            raise RuntimeError("FeatureMap has not been fitted")
        return {
            "continuous": self.continuous,
            "indicators": self.indicators,
            "interactions": self.interactions,
            "spline_knots_standardized": self.knots.tolist(),
            "training_mean": self.mean.tolist(),
            "training_scale": self.scale.tolist(),
        }


def month_balancing_weights(months: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(months, return_inverse=True, return_counts=True)
    weights = 1.0 / counts[inverse]
    return (weights / weights.mean()).astype(np.float32)


def torch_quadratic_loss(
    beta: torch.Tensor,
    centered_fy: torch.Tensor,
    centered_ff: torch.Tensor,
    days: torch.Tensor,
    weights: torch.Tensor,
) -> torch.Tensor:
    variable_sse = -2.0 * (beta * centered_fy).sum(dim=1) + torch.einsum("bi,bij,bj->b", beta, centered_ff, beta)
    return (weights * variable_sse / days).mean()


def iter_batches(rows: int, batch_size: int, order: torch.Tensor | None = None):
    if order is None:
        order = torch.arange(rows)
    for start in range(0, rows, batch_size):
        yield order[start : start + batch_size]


def validation_loss(
    model: nn.Module, design: np.ndarray, fy: np.ndarray, ff: np.ndarray, days: np.ndarray,
    weights: np.ndarray, device: torch.device, batch_size: int, loss_scale: float,
) -> float:
    model.eval()
    total = 0.0
    total_weight = 0
    with torch.no_grad():
        for indices in iter_batches(len(design), batch_size):
            idx = indices.numpy()
            x = torch.from_numpy(design[idx]).to(device)
            loss = torch_quadratic_loss(
                model(x), torch.from_numpy(fy[idx]).to(device), torch.from_numpy(ff[idx]).to(device),
                torch.from_numpy(days[idx]).to(device), torch.from_numpy(weights[idx]).to(device),
            )
            total += float(loss.item()) * len(idx)
            total_weight += len(idx)
    return loss_scale * total / total_weight


def fit_model(
    name: str,
    feature_map: FeatureMap,
    train: pl.DataFrame,
    validation: pl.DataFrame,
    train_moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    validation_moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    config: dict,
    device: torch.device,
) -> tuple[nn.Module, dict]:
    settings = config["optimizer"]
    train_design = feature_map.transform(train, name)
    validation_design = feature_map.transform(validation, name)
    train_y2, train_fy, train_ff = train_moments
    _, validation_fy, validation_ff = validation_moments
    train_days = train["n_days"].to_numpy().astype(np.float32)
    validation_days = validation["n_days"].to_numpy().astype(np.float32)
    train_weights = month_balancing_weights(train["feature_month"].to_numpy())
    validation_weights = month_balancing_weights(validation["feature_month"].to_numpy())
    del train_y2

    torch.manual_seed(config["seed"] + (0 if name == "BL" else 1))
    model = nn.Linear(train_design.shape[1], len(FACTORS)).to(device)
    nn.init.zeros_(model.weight)
    nn.init.zeros_(model.bias)
    learning_rate = settings["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=settings["weight_decay"])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(config["seed"])
    best_loss = float("inf")
    best_state = None
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, settings["maximum_epochs"] + 1):
        model.train()
        order = torch.randperm(len(train), generator=generator)
        for indices in iter_batches(len(train), settings["batch_size"], order):
            idx = indices.numpy()
            optimizer.zero_grad(set_to_none=True)
            beta = model(torch.from_numpy(train_design[idx]).to(device))
            loss = settings["loss_scale"] * torch_quadratic_loss(
                beta, torch.from_numpy(train_fy[idx]).to(device), torch.from_numpy(train_ff[idx]).to(device),
                torch.from_numpy(train_days[idx]).to(device), torch.from_numpy(train_weights[idx]).to(device),
            )
            loss.backward()
            optimizer.step()
        current = validation_loss(
            model, validation_design, validation_fy, validation_ff, validation_days, validation_weights,
            device, settings["batch_size"], settings["loss_scale"],
        )
        history.append({"epoch": epoch, "validation_variable_loss_scaled": current})
        threshold = abs(best_loss) * settings["minimum_relative_improvement"] if np.isfinite(best_loss) else 0.0
        if current < best_loss - threshold:
            best_loss = current
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if stale >= settings["patience"]:
            break
    if best_state is None:
        raise RuntimeError(f"{name} did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "model": name,
        "input_dimension": train_design.shape[1],
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "learning_rate": learning_rate,
        "best_epoch": best_epoch,
        "epochs_run": epoch,
        "best_validation_variable_loss_scaled": best_loss,
        "history": history,
    }


def predict(model: nn.Module, design: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    parts = []
    with torch.no_grad():
        for indices in iter_batches(len(design), batch_size):
            parts.append(model(torch.from_numpy(design[indices.numpy()]).to(device)).cpu().numpy())
    return np.vstack(parts).astype(np.float32)


def joint_calibration(beta: np.ndarray, fy: np.ndarray, ff: np.ndarray, weights: np.ndarray) -> float:
    numerator = np.sum(weights * np.einsum("ni,ni->n", beta, fy))
    denominator = np.sum(weights * np.einsum("ni,nij,nj->n", beta, ff, beta))
    return float(numerator / denominator) if denominator > 0 else float("nan")


def evaluate(
    frame: pl.DataFrame,
    moments: tuple[np.ndarray, np.ndarray, np.ndarray],
    predictions: dict[str, np.ndarray],
) -> tuple[dict, pl.DataFrame]:
    y2, fy, ff = moments
    days = frame["n_days"].to_numpy().astype(np.float32)
    months = frame["feature_month"].to_numpy()
    splits = frame["split"].to_numpy()
    b0 = np.column_stack([frame[f"beta_{factor}"].to_numpy() for factor in FACTORS]).astype(np.float32)
    predictions = {"B0": b0, **predictions}
    records = []
    monthly_records = []
    for split in ["train", "validation", "development_oos"]:
        split_mask = splits == split
        for sample, sample_mask in [("all", split_mask), ("b0_common", split_mask & frame["b0_eligible"].to_numpy())]:
            for name, beta in predictions.items():
                mask = sample_mask & np.all(np.isfinite(beta), axis=1)
                if not mask.any():
                    continue
                weights = month_balancing_weights(months[mask])
                rss = residual_sum_squares(beta[mask], y2[mask], fy[mask], ff[mask])
                moment = fy[mask] - np.einsum("nij,nj->ni", ff[mask], beta[mask])
                records.append({
                    "split": split,
                    "sample": sample,
                    "model": name,
                    "rows": int(mask.sum()),
                    "month_balanced_residual_variance": float(np.average(rss / days[mask], weights=weights)),
                    "month_balanced_orthogonality_loss": float(np.average(np.square(moment).sum(axis=1) / days[mask], weights=weights)),
                    "joint_calibration_slope": joint_calibration(beta[mask], fy[mask], ff[mask], weights),
                })
        split_indices = np.flatnonzero(split_mask)
        split_months = months[split_indices]
        unique_months, starts = np.unique(split_months, return_index=True)
        stops = np.append(starts[1:], len(split_indices))
        for month, start, stop in zip(unique_months, starts, stops):
            month_indices = split_indices[start:stop]
            for name, beta in predictions.items():
                indices = month_indices[np.all(np.isfinite(beta[month_indices]), axis=1)]
                if not len(indices):
                    continue
                rss = residual_sum_squares(beta[indices], y2[indices], fy[indices], ff[indices])
                moment = fy[indices] - np.einsum("nij,nj->ni", ff[indices], beta[indices])
                monthly_records.append({
                    "feature_month": str(month),
                    "split": split,
                    "model": name,
                    "rows": int(len(indices)),
                    "residual_variance": float(np.mean(rss / days[indices])),
                    "orthogonality_loss": float(np.mean(np.square(moment).sum(axis=1) / days[indices])),
                })
    monthly = pl.DataFrame(monthly_records).with_columns(pl.col("feature_month").str.to_date())
    return {"metrics": records}, monthly


def load_frame(path: Path, columns: list[str], max_rows_per_split: int | None) -> pl.DataFrame:
    scan = pl.scan_parquet(path).select(columns)
    if max_rows_per_split is None:
        return scan.collect()
    return pl.concat([
        scan.filter(pl.col("split") == split).head(max_rows_per_split).collect()
        for split in ["train", "validation", "development_oos"]
    ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--max-rows-per-split", type=int)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if config["factors"] != FACTORS:
        raise ValueError("Factor order changed")
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Output directory already exists: {output}")
    output.mkdir(parents=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the BL/BG exposure run")
    device = torch.device(config["device"])
    started = time.time()

    schema = pl.scan_parquet(args.input).collect_schema().names()
    company = [name for name in schema if name.startswith("x_")]
    states = [name for name in schema if name.startswith("z_")]
    indicators = [name for name in schema if name.startswith("missing_")]
    continuous = ["log_market_cap", *company, *states]
    columns = [
        "permno", "feature_month", "target_month", "split", "b0_eligible",
        *continuous, *indicators, *required_moment_columns(), *[f"beta_{factor}" for factor in FACTORS],
    ]
    frame = load_frame(args.input, columns, args.max_rows_per_split)
    if frame["target_month"].max().year > 2019:
        raise ValueError("Confirmation-period target entered the exposure run")
    train = frame.filter(pl.col("split") == "train")
    validation = frame.filter(pl.col("split") == "validation")
    feature_map = FeatureMap(continuous, indicators, config["interactions"], config["spline_knots_standardized"])
    feature_map.fit(train)
    train_moments = centered_moments(train)
    validation_moments = centered_moments(validation)
    models = {}
    training = {}
    for name in ["BL", "BG"]:
        model, diagnostics = fit_model(
            name, feature_map, train, validation, train_moments, validation_moments, config, device
        )
        models[name] = model
        training[name] = diagnostics
        torch.save(model.state_dict(), output / f"{name.lower()}_checkpoint.pt")

    moments = centered_moments(frame)
    predictions = {
        name: predict(model, feature_map.transform(frame, name), device, config["optimizer"]["batch_size"])
        for name, model in models.items()
    }
    metrics, monthly = evaluate(frame, moments, predictions)
    prediction_frame = frame.select("permno", "feature_month", "target_month", "split", "b0_eligible")
    for name, values in predictions.items():
        prediction_frame = prediction_frame.with_columns([
            pl.Series(f"{name.lower()}_beta_{factor}", values[:, index]) for index, factor in enumerate(FACTORS)
        ])
    predictions_path = output / "exposure_predictions.parquet"
    monthly_path = output / "monthly_metrics.parquet"
    prediction_frame.write_parquet(predictions_path, compression="zstd")
    monthly.write_parquet(monthly_path, compression="zstd")
    preprocessing_path = output / "preprocessing.json"
    preprocessing_path.write_text(json.dumps(feature_map.metadata(), indent=2) + "\n")
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": args.run_id,
        "status": "completed",
        "input_sha256": sha256(args.input),
        "config_sha256": sha256(args.config),
        "torch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(device),
        "smoke_max_rows_per_split": args.max_rows_per_split,
        "rows": len(frame),
        "split_rows": frame.group_by("split").len().sort("split").to_dicts(),
        "training": training,
        **metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    report_path = output / "quality_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    outputs = [predictions_path, monthly_path, preprocessing_path, report_path, output / "bl_checkpoint.pt", output / "bg_checkpoint.pt"]
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "run_id": args.run_id,
        "command": " ".join(os.sys.argv),
        "outputs": [{"path": path.name, "bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs],
    }
    (output / "output_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["experiment_id", "run_id", "status", "gpu", "rows", "split_rows", "elapsed_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
