#!/usr/bin/env python3
"""Train a sealed-safe, one-seed feed-forward neural SDF teacher pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import platform
import random
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import nn

GKX_FEATURES = [
    "mvel1", "beta", "betasq", "chmom", "dolvol", "idiovol", "indmom", "mom1m",
    "mom6m", "mom12m", "mom36m", "pricedelay", "turn", "absacc", "acc", "age",
    "agr", "bm", "bm_ia", "cashdebt", "cashpr", "cfp", "cfp_ia", "chatoia",
    "chcsho", "chempia", "chinv", "chpmia", "convind", "currat", "depr", "divi",
    "divo", "dy", "egr", "ep", "gma", "grcapx", "grltnoa", "herf", "hire",
    "invest", "lev", "lgr", "mve_ia", "operprof", "orgcap", "pchcapx_ia",
    "pchcurrat", "pchdepr", "pchgm_pchsale", "pchquick", "pchsale_pchinvt",
    "pchsale_pchrect", "pchsale_pchxsga", "pchsaleinv", "pctacc", "ps", "quick",
    "rd", "rd_mve", "rd_sale", "realestate", "roic", "salecash", "saleinv",
    "salerec", "secured", "securedind", "sgr", "sin", "sp", "tang", "tb",
    "aeavol", "cash", "chtx", "cinvest", "ear", "nincr", "roaq", "roavol",
    "roeq", "rsup", "stdacc", "stdcf", "ms", "baspread", "ill", "maxret",
    "retvol", "std_dolvol", "std_turn", "zerotrade",
]
CORE92 = [feature for feature in GKX_FEATURES if feature not in {"aeavol", "ear"}]


TRAIN_START, TRAIN_END = np.datetime64("1963-07"), np.datetime64("1999-12")
VALID_START, VALID_END = np.datetime64("2000-01"), np.datetime64("2009-12")
DEV_START, DEV_END = np.datetime64("2010-01"), np.datetime64("2019-12")
SEALED_START = np.datetime64("2020-01")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--experiment-id", default="P1-G1-V002")
    parser.add_argument("--seed", type=int, default=20260924)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--hidden", default="128,64,32")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--month-batch-size", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--concentration-penalty", type=float, default=1e-3)
    parser.add_argument("--pricing-targets", type=Path)
    parser.add_argument("--pricing-penalty", type=float, default=0.0)
    parser.add_argument("--pricing-loss", choices=["standardized", "hj_ridge"], default="standardized")
    parser.add_argument("--pricing-asset-family", choices=["all", "size_bm_25", "industry_49"], default="all")
    parser.add_argument("--pricing-ridge-multiplier", type=float, default=1e-4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def monthly_rf(path: Path) -> dict[int, float]:
    rows: dict[int, list[float]] = {}
    with zipfile.ZipFile(path) as archive:
        members = [x for x in archive.infolist() if not x.is_dir()]
        if len(members) != 1:
            raise ValueError("Expected one factor CSV")
        stream = io.TextIOWrapper(archive.open(members[0]), encoding="utf-8")
        for line in stream:
            fields = [x.strip() for x in line.split(",")]
            if len(fields) == 5 and len(fields[0]) == 8 and fields[0].isdigit():
                day = np.datetime64(f"{fields[0][:4]}-{fields[0][4:6]}-{fields[0][6:]}")
                rows.setdefault(int(day.astype("datetime64[M]").astype(int)), []).append(float(fields[4]) / 100.0)
    return {month: float(np.prod(1.0 + np.asarray(values)) - 1.0) for month, values in rows.items()}


@dataclass
class SplitData:
    name: str
    months: np.ndarray
    permno: np.ndarray
    market_cap: np.ndarray
    x: np.ndarray
    y: np.ndarray
    unique_months: np.ndarray
    starts: np.ndarray
    ends: np.ndarray


def load_panel(path: Path, factors: Path) -> dict[str, SplitData]:
    feature_columns = [f"x_{name}" for name in CORE92] + [f"missing_{name}" for name in CORE92]
    columns = ["permno", "month", "ret_fwd1", "market_cap"] + feature_columns
    rf = monthly_rf(factors)
    chunks: dict[str, dict[str, list[np.ndarray]]] = {
        name: {key: [] for key in ["months", "permno", "market_cap", "x", "y"]}
        for name in ["train", "validation", "development"]
    }
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=50_000, columns=columns):
        month_days = batch.column("month").to_numpy(zero_copy_only=False).astype("datetime64[D]")
        months = month_days.astype("datetime64[M]").astype(np.int32)
        if months.size and np.datetime64(int(months.max()), "M") >= SEALED_START:
            raise RuntimeError("Input contains the sealed period; use a physically pre-sealed snapshot")
        future_rf = np.array([rf.get(int(value + 1), np.nan) for value in months], dtype=np.float32)
        returns = batch.column("ret_fwd1").to_numpy(zero_copy_only=False).astype(np.float32)
        y = returns - future_rf
        x = np.column_stack([
            batch.column(column).to_numpy(zero_copy_only=False).astype(np.float32)
            for column in feature_columns
        ])
        permno = batch.column("permno").to_numpy(zero_copy_only=False).astype(np.int64)
        market_cap = batch.column("market_cap").to_numpy(zero_copy_only=False).astype(np.float32)
        finite = np.isfinite(y) & np.isfinite(x).all(axis=1)
        month_values = months.astype("datetime64[M]")
        split_masks = {
            "train": finite & (month_values >= TRAIN_START) & (month_values <= TRAIN_END),
            "validation": finite & (month_values >= VALID_START) & (month_values <= VALID_END),
            "development": finite & (month_values >= DEV_START) & (month_values <= DEV_END),
        }
        for name, keep in split_masks.items():
            if not keep.any():
                continue
            for key, values in [("months", months), ("permno", permno), ("market_cap", market_cap), ("x", x), ("y", y)]:
                chunks[name][key].append(values[keep])

    result: dict[str, SplitData] = {}
    for name, values in chunks.items():
        arrays = {key: np.concatenate(parts) for key, parts in values.items()}
        if not arrays["months"].size:
            raise RuntimeError(f"No rows for {name}")
        order = np.lexsort((arrays["permno"], arrays["months"]))
        arrays = {key: value[order] for key, value in arrays.items()}
        unique, starts, counts = np.unique(arrays["months"], return_index=True, return_counts=True)
        result[name] = SplitData(
            name=name,
            months=arrays["months"],
            permno=arrays["permno"],
            market_cap=arrays["market_cap"],
            x=arrays["x"],
            y=arrays["y"],
            unique_months=unique,
            starts=starts,
            ends=starts + counts,
        )
    return result


class Teacher(nn.Module):
    def __init__(self, input_dim: int, hidden: list[int]) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = input_dim
        for width in hidden:
            layers.extend([nn.Linear(previous, width), nn.SiLU()])
            previous = width
        layers.extend([nn.Linear(previous, 1), nn.Tanh()])
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def month_portfolio(score: torch.Tensor, returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    centered = score - score.mean()
    weight = centered / centered.abs().sum().clamp_min(1e-8)
    factor_return = torch.dot(weight, returns)
    scaled_hhi = weight.square().sum() * weight.numel()
    return factor_return, scaled_hhi, weight


def pricing_moment_loss(factor_returns: torch.Tensor, asset_returns: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a scale-free Euler-moment loss and its analytic SDF scale."""
    available = torch.isfinite(asset_returns)
    safe_returns = torch.nan_to_num(asset_returns)
    counts = available.sum(dim=0)
    usable = counts >= 3
    if not bool(usable.any()):
        raise ValueError("Pricing-moment batch has no asset with at least three observations")
    denominator = counts.clamp_min(1)
    mean_returns = (safe_returns * available).sum(dim=0) / denominator
    direction = (factor_returns[:, None] * safe_returns * available).sum(dim=0) / denominator
    mean_returns = mean_returns[usable]
    direction = direction[usable]
    scale = torch.dot(direction, mean_returns) / torch.dot(direction, direction).clamp_min(1e-12)
    moment = mean_returns - scale * direction
    rms = ((safe_returns.square() * available).sum(dim=0) / denominator).sqrt()[usable].detach().clamp_min(1e-4)
    return (moment / rms).square().mean(), scale


def pricing_hj_loss(
    factor_returns: torch.Tensor,
    asset_returns: torch.Tensor,
    ridge_multiplier: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return normalized ridge-HJ squared distance and its weighted analytic scale."""
    complete = torch.isfinite(asset_returns).all(dim=0)
    if int(complete.sum()) < 2:
        raise ValueError("HJ batch has fewer than two complete pricing assets")
    returns = asset_returns[:, complete]
    mean_returns = returns.mean(dim=0)
    direction = (factor_returns[:, None] * returns).mean(dim=0)
    second = returns.T @ returns / returns.shape[0]
    ridge = ridge_multiplier * torch.trace(second) / second.shape[0]
    weight = torch.linalg.inv(second + ridge.clamp_min(1e-12) * torch.eye(second.shape[0], device=second.device))
    scale = (direction @ weight @ mean_returns) / (direction @ weight @ direction).clamp_min(1e-12)
    moment = mean_returns - scale * direction
    distance_squared = moment @ weight @ moment
    zero_distance_squared = (mean_returns @ weight @ mean_returns).detach().clamp_min(1e-12)
    return distance_squared / zero_distance_squared, scale


def load_pricing_targets(
    path: Path,
    splits: dict[str, SplitData],
    device: torch.device,
    asset_family: str = "all",
) -> dict[str, torch.Tensor]:
    table = pq.read_table(path)
    columns = table.column_names
    asset_columns = [name for name in columns if name.startswith("asset_")]
    if len(asset_columns) != 74:
        raise ValueError(f"Expected 74 pricing assets, found {len(asset_columns)}")
    months = table.column("month").to_numpy(zero_copy_only=False).astype("datetime64[M]").astype(np.int32)
    values = np.column_stack([
        table.column(name).to_numpy(zero_copy_only=False).astype(np.float32) for name in asset_columns
    ])
    mapping = {int(month): values[index] for index, month in enumerate(months)}
    slices = {"all": slice(None), "size_bm_25": slice(0, 25), "industry_49": slice(25, 74)}
    selected = slices[asset_family]
    output: dict[str, torch.Tensor] = {}
    for name, split in splits.items():
        missing = [int(month) for month in split.unique_months if int(month) not in mapping]
        if missing:
            raise ValueError(f"Pricing targets miss {len(missing)} months in {name}")
        matrix = np.vstack([mapping[int(month)] for month in split.unique_months])[:, selected]
        if (np.isfinite(matrix).sum(axis=1) < 25).any():
            raise ValueError(f"Pricing targets contain fewer than 25 available assets in a {name} month")
        output[name] = torch.from_numpy(matrix).to(device=device)
    return output


def train_epoch(
    model: Teacher,
    split: SplitData,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    month_batch_size: int,
    concentration_penalty: float,
    rng: np.random.Generator,
    pricing_assets: torch.Tensor | None = None,
    pricing_penalty: float = 0.0,
    pricing_loss_name: str = "standardized",
    pricing_ridge_multiplier: float = 1e-4,
) -> dict[str, float]:
    model.train()
    order = rng.permutation(len(split.unique_months))
    losses: list[float] = []
    sharpes: list[float] = []
    pricing_losses: list[float] = []
    for offset in range(0, len(order), month_batch_size):
        selection = order[offset : offset + month_batch_size]
        x_batch = torch.cat([x[int(split.starts[i]) : int(split.ends[i])] for i in selection])
        y_batch = torch.cat([y[int(split.starts[i]) : int(split.ends[i])] for i in selection])
        counts = [int(split.ends[i] - split.starts[i]) for i in selection]
        scores = model(x_batch)
        returns: list[torch.Tensor] = []
        concentrations: list[torch.Tensor] = []
        cursor = 0
        for count in counts:
            factor_return, hhi, _ = month_portfolio(scores[cursor : cursor + count], y_batch[cursor : cursor + count])
            returns.append(factor_return)
            concentrations.append(hhi)
            cursor += count
        monthly = torch.stack(returns)
        sharpe = monthly.mean() / monthly.std(unbiased=True).clamp_min(1e-6) * math.sqrt(12.0)
        if pricing_assets is not None:
            if pricing_loss_name == "hj_ridge":
                pricing_loss, _ = pricing_hj_loss(monthly, pricing_assets[selection], pricing_ridge_multiplier)
            else:
                pricing_loss, _ = pricing_moment_loss(monthly, pricing_assets[selection])
        else:
            pricing_loss = monthly.new_zeros(())
        loss = -sharpe + pricing_penalty * pricing_loss + concentration_penalty * torch.stack(concentrations).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        sharpes.append(float(sharpe.detach().cpu()))
        pricing_losses.append(float(pricing_loss.detach().cpu()))
    return {
        "loss": float(np.mean(losses)),
        "batch_sharpe": float(np.mean(sharpes)),
        "batch_pricing_moment_loss": float(np.mean(pricing_losses)),
    }


def rank_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 3:
        return float("nan")
    ar = np.empty(a.size, dtype=np.float64); ar[np.argsort(a, kind="stable")] = np.arange(a.size)
    br = np.empty(b.size, dtype=np.float64); br[np.argsort(b, kind="stable")] = np.arange(b.size)
    return float(np.corrcoef(ar, br)[0, 1])


def evaluate(model: Teacher, split: SplitData, x: torch.Tensor, y: torch.Tensor, keep_rows: bool) -> tuple[dict[str, float], list[dict[str, object]], dict[str, np.ndarray]]:
    model.eval()
    monthly_rows: list[dict[str, object]] = []
    score_parts: list[np.ndarray] = []
    weight_parts: list[np.ndarray] = []
    with torch.no_grad():
        for month, start, end in zip(split.unique_months, split.starts, split.ends):
            score = model(x[int(start) : int(end)])
            factor_return, scaled_hhi, weight = month_portfolio(score, y[int(start) : int(end)])
            score_np = score.cpu().numpy()
            weight_np = weight.cpu().numpy()
            returns_np = split.y[int(start) : int(end)]
            order = np.argsort(score_np, kind="stable")
            tenth = max(1, len(order) // 10)
            spread = float(returns_np[order[-tenth:]].mean() - returns_np[order[:tenth]].mean())
            monthly_rows.append({
                "split": split.name,
                "month": str(np.datetime64(int(month), "M")),
                "rows": int(end - start),
                "factor_return": float(factor_return.cpu()),
                "rank_ic": rank_correlation(score_np, returns_np),
                "decile_spread": spread,
                "scaled_hhi": float(scaled_hhi.cpu()),
                "maximum_absolute_weight": float(np.abs(weight_np).max()),
            })
            if keep_rows:
                score_parts.append(score_np.astype(np.float32))
                weight_parts.append(weight_np.astype(np.float32))
    factor_returns = np.asarray([row["factor_return"] for row in monthly_rows])
    mean = float(factor_returns.mean())
    std = float(factor_returns.std(ddof=1))
    wealth = np.cumprod(1.0 + factor_returns)
    drawdown = wealth / np.maximum.accumulate(wealth) - 1.0
    metrics = {
        "rows": int(split.y.size),
        "months": int(len(monthly_rows)),
        "annualized_factor_return": mean * 12.0,
        "annualized_factor_volatility": std * math.sqrt(12.0),
        "annualized_factor_sharpe": mean / std * math.sqrt(12.0) if std > 0 else float("nan"),
        "mean_monthly_rank_ic": float(np.nanmean([row["rank_ic"] for row in monthly_rows])),
        "annualized_decile_spread": float(np.mean([row["decile_spread"] for row in monthly_rows]) * 12.0),
        "maximum_drawdown": float(drawdown.min()),
        "mean_scaled_hhi": float(np.mean([row["scaled_hhi"] for row in monthly_rows])),
        "maximum_absolute_weight": float(np.max([row["maximum_absolute_weight"] for row in monthly_rows])),
    }
    row_data = {
        "score": np.concatenate(score_parts) if score_parts else np.empty(0, dtype=np.float32),
        "weight": np.concatenate(weight_parts) if weight_parts else np.empty(0, dtype=np.float32),
    }
    return metrics, monthly_rows, row_data


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main() -> int:
    args = parse_args()
    started = time.time()
    output = args.output_dir.resolve()
    products = [
        output / "checkpoint.pt", output / "training_history.csv", output / "monthly_factor_returns.csv",
        output / "teacher_scores_weights.parquet", output / "quality_report.json", output / "environment.json",
        output / "output_manifest.json",
    ]
    if any(path.exists() for path in products) and not args.overwrite:
        raise FileExistsError("Output exists; use --overwrite")
    output.mkdir(parents=True, exist_ok=True)
    for path in products:
        path.unlink(missing_ok=True)

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("This experiment requires CUDA")

    splits = load_panel(args.input, args.factors)
    tensors = {
        name: (
            torch.from_numpy(split.x).to(device=device),
            torch.from_numpy(split.y).to(device=device),
        )
        for name, split in splits.items()
    }
    if (args.pricing_targets is None) != (args.pricing_penalty == 0.0):
        raise ValueError("Use --pricing-targets together with a positive --pricing-penalty")
    pricing_targets = (
        load_pricing_targets(args.pricing_targets, splits, device, args.pricing_asset_family)
        if args.pricing_targets else None
    )
    hidden = [int(value) for value in args.hidden.split(",") if value]
    model = Teacher(splits["train"].x.shape[1], hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    rng = np.random.default_rng(args.seed)
    history: list[dict[str, object]] = []
    best_epoch = 0
    best_score = -float("inf")
    best_validation_sharpe = -float("inf")
    best_validation_pricing_loss = float("nan")
    best_state: dict[str, torch.Tensor] | None = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        training = train_epoch(
            model, splits["train"], *tensors["train"], optimizer, args.month_batch_size,
            args.concentration_penalty, rng,
            pricing_targets["train"] if pricing_targets else None, args.pricing_penalty,
            args.pricing_loss, args.pricing_ridge_multiplier,
        )
        validation, validation_monthly, _ = evaluate(model, splits["validation"], *tensors["validation"], keep_rows=False)
        if pricing_targets:
            validation_factor = torch.tensor(
                [row["factor_return"] for row in validation_monthly], device=device, dtype=torch.float32
            )
            validation_loss_function = pricing_hj_loss if args.pricing_loss == "hj_ridge" else pricing_moment_loss
            validation_loss_args = (
                (validation_factor, pricing_targets["validation"], args.pricing_ridge_multiplier)
                if args.pricing_loss == "hj_ridge" else (validation_factor, pricing_targets["validation"])
            )
            validation_pricing_loss = float(validation_loss_function(*validation_loss_args)[0].detach().cpu())
        else:
            validation_pricing_loss = 0.0
        validation_score = validation["annualized_factor_sharpe"] - args.pricing_penalty * validation_pricing_loss
        row = {
            "epoch": epoch, **training,
            "validation_sharpe": validation["annualized_factor_sharpe"],
            "validation_rank_ic": validation["mean_monthly_rank_ic"],
            "validation_pricing_moment_loss": validation_pricing_loss,
            "validation_selection_score": validation_score,
        }
        history.append(row)
        print(json.dumps(row), flush=True)
        if validation_score > best_score + 1e-4:
            best_score = validation_score
            best_validation_sharpe = validation["annualized_factor_sharpe"]
            best_validation_pricing_loss = validation_pricing_loss
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    if best_state is None:
        raise RuntimeError("No valid checkpoint")
    model.load_state_dict(best_state)
    torch.save({"model_state_dict": best_state, "input_dim": splits["train"].x.shape[1], "hidden": hidden, "seed": args.seed}, products[0])
    write_csv(products[1], history)

    metrics: dict[str, dict[str, float]] = {}
    monthly: list[dict[str, object]] = []
    score_tables: list[pa.Table] = []
    for name in ["train", "validation", "development"]:
        split = splits[name]
        keep_rows = name != "train"
        split_metrics, split_monthly, row_data = evaluate(model, split, *tensors[name], keep_rows=keep_rows)
        metrics[name] = split_metrics; monthly.extend(split_monthly)
        if not keep_rows:
            continue
        score_tables.append(pa.table({
            "permno": pa.array(split.permno),
            "month": pa.array(split.months.astype("datetime64[M]").astype("datetime64[D]")),
            "split": pa.array([name] * split.y.size),
            "ret_excess_fwd1": pa.array(split.y),
            "market_cap": pa.array(split.market_cap),
            "teacher_score": pa.array(row_data["score"]),
            "teacher_weight": pa.array(row_data["weight"]),
        }))
    write_csv(products[2], monthly)
    pq.write_table(pa.concat_tables(score_tables), products[3], compression="zstd", row_group_size=100_000)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    report = {
        "schema_version": 1,
        "experiment_id": args.experiment_id,
        "status": "completed",
        "evidence_class": "development",
        "sealed_period_accessed": False,
        "source_git_revision": args.source_git_revision,
        "device": str(device),
        "seed": args.seed,
        "features": len(CORE92) * 2,
        "architecture": [len(CORE92) * 2, *hidden, 1],
        "parameter_count": parameter_count,
        "objective": (
            f"month-batched annualized long-short SDF Sharpe plus {args.pricing_loss} analytic-scale pricing loss and scaled-HHI regularization"
            if pricing_targets else "month-batched annualized long-short SDF Sharpe with scaled-HHI regularization"
        ),
        "pricing_penalty": args.pricing_penalty,
        "pricing_loss": args.pricing_loss,
        "pricing_asset_family": args.pricing_asset_family,
        "pricing_ridge_multiplier": args.pricing_ridge_multiplier,
        "best_epoch": best_epoch,
        "best_validation_selection_score": best_score,
        "best_validation_sharpe": best_validation_sharpe,
        "best_validation_pricing_moment_loss": best_validation_pricing_loss,
        "performance": metrics,
        "elapsed_seconds": round(time.time() - started, 3),
    }
    products[4].write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    environment = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "numpy": np.__version__,
        "pyarrow": pa.__version__,
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(device),
        "gpu_count": torch.cuda.device_count(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
    }
    products[5].write_text(json.dumps(environment, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": args.experiment_id,
        "source_git_revision": args.source_git_revision,
        "command": " ".join(sys.argv),
        "inputs": [
            {"path": str(path.resolve()), "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in [args.input, args.factors, *([args.pricing_targets] if args.pricing_targets else [])]
        ],
        "outputs": [
            {"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in products[:-1]
        ],
    }
    products[6].write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
