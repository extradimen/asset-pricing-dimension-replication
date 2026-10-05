"""Conditional autoencoder mechanics; no data access or experiment on import."""
from __future__ import annotations
import numpy as np
import torch
from torch import nn


class ConditionalAE(nn.Module):
    def __init__(self, p, k, hidden):
        super().__init__()
        layers = []
        for width in hidden:
            layers.extend([nn.Linear(p, width), nn.ReLU()])
            p = width
        layers.append(nn.Linear(p, k))
        self.beta = nn.Sequential(*layers)
        # Caller supplies P+1 characteristic-managed portfolios, including constant.
        self.factor = None

    def set_encoder(self, managed_dim, k):
        self.factor = nn.Linear(managed_dim, k, bias=False)
        return self

    def forward(self, characteristics, managed_returns, month_index):
        factors = self.factor(managed_returns)
        return (self.beta(characteristics) * factors[month_index]).sum(-1)


def managed_portfolio(z, y, ridge):
    """Ridge-stabilized cross-sectional characteristic portfolio, current returns."""
    z = np.column_stack([np.ones(len(z)), z]).astype(np.float64)
    gram = z.T @ z / len(z)
    penalty = ridge * max(np.trace(gram) / len(gram), 1e-12)
    return np.linalg.solve(gram + penalty * np.eye(len(gram)), z.T @ y / len(z))


def lagged_factor_mean(history, current):
    """Forecast at t excludes realized f_t and every later realization."""
    history = np.asarray(history)
    current = np.asarray(current)
    previous = np.vstack([np.zeros((1, current.shape[1])), np.cumsum(current, axis=0)[:-1]])
    return (history.sum(0) + previous) / (len(history) + np.arange(len(current)))[:, None]


def sdf_coefficient(factors, ridge=1e-6):
    second = factors.T @ factors / len(factors)
    penalty = ridge * max(np.trace(second) / len(second), 1e-12)
    return np.linalg.solve(second + penalty * np.eye(len(second)), factors.mean(0))


def pricing_weight(reference_returns, gamma, ridge=1e-4):
    second = reference_returns.T @ reference_returns / len(reference_returns)
    values, vectors = np.linalg.eigh(second)
    diagonal = (np.maximum(values, 0) + ridge * max(np.trace(second) / len(second), 1e-12)) ** (-gamma)
    diagonal *= len(second) / np.sum(values * diagonal)
    return (vectors * diagonal) @ vectors.T


def pricing_distance(sdf, returns, weight):
    moments = np.asarray(sdf) @ returns / len(returns)
    return np.sqrt(np.maximum(12 * np.einsum('...i,ij,...j->...', moments, weight, moments), 0))


def block_indices(rng, n, block):
    starts = rng.integers(0, n, int(np.ceil(n / block)))
    return ((starts[:, None] + np.arange(block)) % n).ravel()[:n]
