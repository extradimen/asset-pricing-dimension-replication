#!/usr/bin/env python3
"""Train a sealed-safe multi-direction neural SDF teacher."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from run_neural_sdf_teacher import Teacher, load_panel, load_pricing_targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--pricing-targets", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--factor-count", type=int, required=True)
    parser.add_argument("--hidden", default="128,64,32")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--month-batch-size", type=int, default=120)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--hj-penalty", type=float, default=3.0)
    parser.add_argument("--span-sharpe-reward", type=float, default=0.25)
    parser.add_argument("--diversity-penalty", type=float, default=0.05)
    parser.add_argument("--concentration-penalty", type=float, default=1e-3)
    parser.add_argument("--pricing-ridge-multiplier", type=float, default=1e-4)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class MultiTeacher(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int], factors: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.SiLU()]); previous = width
        layers.extend([nn.Linear(previous, factors), nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def month_factor_returns(scores: torch.Tensor, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    centered = scores - scores.mean(dim=0, keepdim=True)
    weights = centered / centered.abs().sum(dim=0, keepdim=True).clamp_min(1e-8)
    factor_returns = weights.T @ returns
    scaled_hhi = weights.square().sum(dim=0) * weights.shape[0]
    return factor_returns, scaled_hhi


def factor_span_sharpe(factors: torch.Tensor, ridge_multiplier: float = 1e-4) -> torch.Tensor:
    mean = factors.mean(dim=0)
    centered = factors - mean
    covariance = centered.T @ centered / max(factors.shape[0] - 1, 1)
    ridge = ridge_multiplier * torch.trace(covariance) / covariance.shape[0]
    solved = torch.linalg.solve(
        covariance + ridge.clamp_min(1e-10) * torch.eye(covariance.shape[0], device=factors.device), mean
    )
    return torch.sqrt((mean @ solved).clamp_min(1e-12) * 12.0)


def hj_span_loss(
    factors: torch.Tensor,
    asset_returns: torch.Tensor,
    ridge_multiplier: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    complete = torch.isfinite(asset_returns).all(dim=0)
    returns = asset_returns[:, complete]
    if returns.shape[1] < 2:
        raise ValueError("HJ span batch has fewer than two complete assets")
    mean_returns = returns.mean(dim=0)
    direction = factors.T @ returns / returns.shape[0]
    second = returns.T @ returns / returns.shape[0]
    ridge = ridge_multiplier * torch.trace(second) / second.shape[0]
    weight = torch.linalg.inv(second + ridge.clamp_min(1e-12) * torch.eye(second.shape[0], device=returns.device))
    system = direction @ weight @ direction.T
    loading_ridge = ridge_multiplier * torch.trace(system) / max(system.shape[0], 1)
    loadings = torch.linalg.solve(
        system + loading_ridge.clamp_min(1e-12) * torch.eye(system.shape[0], device=returns.device),
        direction @ weight @ mean_returns,
    )
    moment = mean_returns - direction.T @ loadings
    distance = moment @ weight @ moment
    zero_distance = (mean_returns @ weight @ mean_returns).detach().clamp_min(1e-12)
    return distance / zero_distance, loadings


def factor_diversity_loss(factors: torch.Tensor) -> torch.Tensor:
    if factors.shape[1] == 1:
        return factors.new_zeros(())
    centered = factors - factors.mean(dim=0, keepdim=True)
    standardized = centered / centered.square().mean(dim=0, keepdim=True).sqrt().clamp_min(1e-6)
    correlation = standardized.T @ standardized / standardized.shape[0]
    off_diagonal = correlation - torch.diag(torch.diag(correlation))
    return off_diagonal.square().sum() / (factors.shape[1] * (factors.shape[1] - 1))


def evaluate_monthly(
    model: MultiTeacher,
    split,
    x: torch.Tensor,
    y: torch.Tensor,
    pricing_assets: torch.Tensor,
    ridge_multiplier: float,
) -> tuple[dict[str, float], list[dict[str, object]]]:
    model.eval(); rows: list[dict[str, object]] = []; factors: list[torch.Tensor] = []; hhis: list[torch.Tensor] = []
    with torch.no_grad():
        for month, start, end in zip(split.unique_months, split.starts, split.ends):
            factor, hhi = month_factor_returns(model(x[int(start):int(end)]), y[int(start):int(end)])
            factors.append(factor); hhis.append(hhi)
            row: dict[str, object] = {"split": split.name, "month": str(np.datetime64(int(month), "M"))}
            row.update({f"factor_{index:03d}": float(value) for index, value in enumerate(factor.cpu())})
            rows.append(row)
    matrix = torch.stack(factors)
    hj_loss, loadings = hj_span_loss(matrix, pricing_assets, ridge_multiplier)
    metrics = {
        "months": len(rows),
        "factor_count": matrix.shape[1],
        "factor_span_sharpe": float(factor_span_sharpe(matrix, ridge_multiplier).cpu()),
        "normalized_hj_span_loss": float(hj_loss.cpu()),
        "mean_scaled_hhi": float(torch.stack(hhis).mean().cpu()),
        "maximum_absolute_sdf_loading": float(loadings.abs().max().cpu()),
    }
    return metrics, rows


def main() -> int:
    args = parse_args(); started = time.time(); output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)
    products = [output / name for name in ["checkpoint.pt", "training_history.csv", "monthly_factor_returns.csv", "quality_report.json", "environment.json", "output_manifest.json"]]
    if any(path.exists() for path in products):
        raise FileExistsError("Output exists")
    if args.factor_count < 1:
        raise ValueError("factor-count must be positive")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available(): raise RuntimeError("CUDA required")
    splits = load_panel(args.input, args.factors)
    tensors = {name: (torch.from_numpy(split.x).to(device), torch.from_numpy(split.y).to(device)) for name, split in splits.items()}
    pricing = load_pricing_targets(args.pricing_targets, splits, device, "size_bm_25")
    hidden = [int(value) for value in args.hidden.split(",") if value]
    model = MultiTeacher(splits["train"].x.shape[1], hidden, args.factor_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed); history: list[dict[str, object]] = []
    best_score = -float("inf"); best_epoch = 0; best_state = None; stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train(); order = rng.permutation(len(splits["train"].unique_months)); batch_losses=[]
        for offset in range(0, len(order), args.month_batch_size):
            selection = order[offset:offset + args.month_batch_size]
            split = splits["train"]; x, y = tensors["train"]
            x_batch = torch.cat([x[int(split.starts[i]):int(split.ends[i])] for i in selection])
            y_batch = torch.cat([y[int(split.starts[i]):int(split.ends[i])] for i in selection])
            counts = [int(split.ends[i] - split.starts[i]) for i in selection]
            scores = model(x_batch); factor_rows=[]; hhis=[]; cursor=0
            for count in counts:
                factor, hhi = month_factor_returns(scores[cursor:cursor+count], y_batch[cursor:cursor+count])
                factor_rows.append(factor); hhis.append(hhi); cursor += count
            factor_matrix = torch.stack(factor_rows)
            hj_loss, _ = hj_span_loss(factor_matrix, pricing["train"][selection], args.pricing_ridge_multiplier)
            span_sharpe = factor_span_sharpe(factor_matrix, args.pricing_ridge_multiplier)
            diversity = factor_diversity_loss(factor_matrix)
            loss = args.hj_penalty * hj_loss - args.span_sharpe_reward * span_sharpe + args.diversity_penalty * diversity + args.concentration_penalty * torch.stack(hhis).mean()
            optimizer.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step()
            batch_losses.append(float(loss.detach().cpu()))
        validation, _ = evaluate_monthly(model, splits["validation"], *tensors["validation"], pricing["validation"], args.pricing_ridge_multiplier)
        selection_score = args.span_sharpe_reward * validation["factor_span_sharpe"] - args.hj_penalty * validation["normalized_hj_span_loss"]
        row = {"epoch": epoch, "train_batch_loss": float(np.mean(batch_losses)), "validation_selection_score": selection_score, **{f"validation_{k}": v for k, v in validation.items()}}
        history.append(row); print(json.dumps(row), flush=True)
        if selection_score > best_score + 1e-4:
            best_score = selection_score; best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}; stale = 0
        else: stale += 1
        if stale >= args.patience: break
    if best_state is None: raise RuntimeError("No valid checkpoint")
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "input_dim": splits["train"].x.shape[1], "hidden": hidden, "factor_count": args.factor_count, "seed": args.seed}, products[0])
    with products[1].open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    metrics={}; monthly=[]
    for name in ["train", "validation", "development"]:
        value, rows = evaluate_monthly(model, splits[name], *tensors[name], pricing[name], args.pricing_ridge_multiplier)
        metrics[name]=value; monthly.extend(rows)
    with products[2].open("w", newline="") as stream:
        writer=csv.DictWriter(stream, fieldnames=list(monthly[0])); writer.writeheader(); writer.writerows(monthly)
    report={
        "schema_version":1,"experiment_id":args.experiment_id,"status":"completed","evidence_class":"development",
        "sealed_period_accessed":False,"source_git_revision":args.source_git_revision,"device":str(device),"seed":args.seed,
        "factor_count":args.factor_count,"architecture":[splits["train"].x.shape[1],*hidden,args.factor_count],
        "parameter_count":sum(p.numel() for p in model.parameters()),"training_asset_family":"size_bm_25",
        "objective":"multi-direction normalized ridge-HJ span loss plus factor-span Sharpe, diversity and concentration regularization",
        "hj_penalty":args.hj_penalty,"span_sharpe_reward":args.span_sharpe_reward,"diversity_penalty":args.diversity_penalty,
        "best_epoch":best_epoch,"best_validation_selection_score":best_score,"performance":metrics,"elapsed_seconds":round(time.time()-started,3),
    }
    products[3].write_text(json.dumps(report,indent=2)+"\n")
    environment={"created_at":datetime.now(timezone.utc).isoformat(),"hostname":platform.node(),"python":sys.version,"numpy":np.__version__,"torch":torch.__version__,"cuda_runtime":torch.version.cuda,"gpu":torch.cuda.get_device_name(device),"deterministic_algorithms":torch.are_deterministic_algorithms_enabled()}
    products[4].write_text(json.dumps(environment,indent=2)+"\n")
    manifest={"schema_version":1,"created_at":datetime.now(timezone.utc).isoformat(),"experiment_id":args.experiment_id,"source_git_revision":args.source_git_revision,"command":" ".join(sys.argv),"inputs":[{"path":str(p.resolve()),"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in [args.input,args.factors,args.pricing_targets]],"outputs":[{"path":p.name,"size_bytes":p.stat().st_size,"sha256":sha256(p)} for p in products[:-1]]}
    products[5].write_text(json.dumps(manifest,indent=2)+"\n"); print(json.dumps(report,indent=2),flush=True); return 0


if __name__ == "__main__":
    raise SystemExit(main())
