#!/usr/bin/env python3
"""Exact frozen loss functions extracted from run_multi_factor_sdf_teacher.py.
Original file SHA256: 5c5af1f0c240f053959ef9569c1c141267e37dc919ac1c9511404055c3b456d6
Loss algebra is unchanged. Matrix right-hand sides avoid the PyTorch 1.13 K=1 vector-solve backward shape bug; solutions are squeezed back to vectors.
"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn

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
        covariance + ridge.clamp_min(1e-10) * torch.eye(covariance.shape[0], device=factors.device), mean.unsqueeze(-1)
    ).squeeze(-1)
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
        (direction @ weight @ mean_returns).unsqueeze(-1),
    ).squeeze(-1)
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
