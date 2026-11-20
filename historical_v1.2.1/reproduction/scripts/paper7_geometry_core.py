#!/usr/bin/env python3
"""Shared, dependency-light geometry estimators for paper 7.

The estimators in this module are deliberately described as finite-scale
geometry diagnostics.  They are not treated as oracle estimates of a unique
topological dimension.
"""

from __future__ import annotations

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


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_representation(x: np.ndarray, mode: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 4:
        raise ValueError("representation must be a two-dimensional array with at least four rows")
    if mode == "raw_centered":
        return x - x.mean(axis=0, keepdims=True)
    if mode == "feature_zscore":
        centered = x - x.mean(axis=0, keepdims=True)
        scale = centered.std(axis=0, ddof=1, keepdims=True)
        active = scale > 1e-12
        return np.divide(centered, scale, out=np.zeros_like(centered), where=active)
    if mode == "row_l2_centered":
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        scaled = np.divide(x, norm, out=np.zeros_like(x), where=norm > 1e-12)
        return scaled - scaled.mean(axis=0, keepdims=True)
    raise ValueError(f"unknown normalization mode: {mode}")


def covariance_eigenvalues(x: np.ndarray) -> np.ndarray:
    if x.shape[0] <= x.shape[1]:
        singular = np.linalg.svd(x, compute_uv=False, full_matrices=False)
        values = singular.square() / max(x.shape[0] - 1, 1)
    else:
        covariance = x.T @ x / max(x.shape[0] - 1, 1)
        values = np.linalg.eigvalsh(covariance)[::-1]
    values = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    return values[values > max(values[0] if values.size else 0.0, 1.0) * 1e-14]


def pairwise_distances(x: np.ndarray) -> np.ndarray:
    squared = np.sum(x * x, axis=1, keepdims=True)
    distances2 = squared + squared.T - 2.0 * (x @ x.T)
    np.maximum(distances2, 0.0, out=distances2)
    distances = np.sqrt(distances2, out=distances2)
    np.fill_diagonal(distances, np.inf)
    return distances


def twonn_dimension(distances: np.ndarray, trim_fraction: float = 0.0) -> float:
    nearest = np.partition(distances, kth=1, axis=1)[:, :2]
    nearest.sort(axis=1)
    valid = (nearest[:, 0] > 1e-12) & np.isfinite(nearest[:, 1])
    log_ratio = np.log(nearest[valid, 1] / nearest[valid, 0])
    if log_ratio.size < 20:
        return float("nan")
    if trim_fraction > 0:
        cutoff = np.quantile(log_ratio, 1.0 - trim_fraction)
        log_ratio = log_ratio[log_ratio <= cutoff]
    mean_log = float(log_ratio.mean())
    return float(1.0 / mean_log) if mean_log > 0 else float("nan")


def levina_bickel_dimension(distances: np.ndarray, k: int) -> float:
    if k < 4 or distances.shape[0] <= k:
        return float("nan")
    nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
    nearest.sort(axis=1)
    radius = nearest[:, k - 1]
    logs = np.log(np.maximum(radius[:, None], 1e-12) / np.maximum(nearest[:, : k - 1], 1e-12))
    denominator = logs.sum(axis=1)
    local = np.divide(k - 2.0, denominator, out=np.full_like(denominator, np.nan), where=denominator > 0)
    finite = local[np.isfinite(local)]
    return float(np.median(finite)) if finite.size else float("nan")


def tangent_curvature_proxy(
    x: np.ndarray,
    distances: np.ndarray,
    *,
    anchors: int = 8,
    small_k: int = 16,
    large_k: int = 48,
    tangent_rank: int = 3,
    seed: int = 0,
) -> float:
    """Median largest principal angle between small- and large-scale tangents."""
    if x.shape[0] <= large_k or x.shape[1] == 1:
        return float("nan")
    rank = min(tangent_rank, x.shape[1], small_k - 1)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(x.shape[0], size=min(anchors, x.shape[0]), replace=False)
    angles: list[float] = []
    for index in chosen:
        order = np.argpartition(distances[index], large_k)[:large_k]
        order = order[np.argsort(distances[index, order])]
        small = x[order[:small_k]] - x[order[:small_k]].mean(axis=0, keepdims=True)
        large = x[order] - x[order].mean(axis=0, keepdims=True)
        _, _, vh_small = np.linalg.svd(small, full_matrices=False)
        _, _, vh_large = np.linalg.svd(large, full_matrices=False)
        singular = np.linalg.svd(vh_small[:rank] @ vh_large[:rank].T, compute_uv=False)
        principal = np.arccos(np.clip(singular, -1.0, 1.0))
        angles.append(float(np.max(principal)))
    return float(np.median(angles)) if angles else float("nan")


def geometry_metrics(
    representation: np.ndarray,
    normalization: str,
    *,
    neighbor_metrics: bool = True,
    curvature_seed: int = 0,
) -> dict[str, float | int | str]:
    x = normalize_representation(representation, normalization)
    eigenvalues = covariance_eigenvalues(x)
    total = float(eigenvalues.sum()) if eigenvalues.size else 0.0
    probabilities = eigenvalues / total if total > 0 else np.array([], dtype=np.float64)
    entropy_rank = float(np.exp(-(probabilities * np.log(np.maximum(probabilities, 1e-300))).sum())) if probabilities.size else 0.0
    participation = float(total * total / np.square(eigenvalues).sum()) if eigenvalues.size else 0.0
    stable_rank = float(total / eigenvalues[0]) if eigenvalues.size and eigenvalues[0] > 0 else 0.0
    top_share = float(eigenvalues[0] / total) if total > 0 else 1.0
    cumulative = np.cumsum(probabilities) if probabilities.size else np.array([])
    metrics: dict[str, float | int | str] = {
        "normalization": normalization,
        "n": int(x.shape[0]),
        "ambient_dimension": int(x.shape[1]),
        "active_spectral_rank": int(eigenvalues.size),
        "total_variance": total,
        "entropy_effective_rank": entropy_rank,
        "participation_rank": participation,
        "stable_rank": stable_rank,
        "top_eigenvalue_share": top_share,
        "pca90_rank": int(np.searchsorted(cumulative, 0.90) + 1) if cumulative.size else 0,
        "pca95_rank": int(np.searchsorted(cumulative, 0.95) + 1) if cumulative.size else 0,
        "collapsed": bool(total <= 1e-12 or participation <= 1.05),
    }
    if not neighbor_metrics:
        metrics.update({"twonn_dimension": float("nan"), "mle10_dimension": float("nan"), "mle20_dimension": float("nan"), "curvature_proxy_radians": float("nan")})
        return metrics
    distances = pairwise_distances(x)
    twonn = twonn_dimension(distances)
    tangent_rank = int(np.clip(np.rint(twonn), 1, min(x.shape[1], 12))) if np.isfinite(twonn) else 3
    metrics.update(
        {
            "twonn_dimension": twonn,
            "mle10_dimension": levina_bickel_dimension(distances, 10),
            "mle20_dimension": levina_bickel_dimension(distances, 20),
            "curvature_tangent_rank": tangent_rank,
            "curvature_proxy_radians": tangent_curvature_proxy(x, distances, tangent_rank=tangent_rank, seed=curvature_seed),
        }
    )
    return metrics


def pca_basis(x: np.ndarray, rank: int) -> np.ndarray:
    centered = x - x.mean(axis=0, keepdims=True)
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    return vh[: min(rank, vh.shape[0])].T


def grassmann_distance(left: np.ndarray, right: np.ndarray) -> float:
    rank = min(left.shape[1], right.shape[1])
    singular = np.linalg.svd(left[:, :rank].T @ right[:, :rank], compute_uv=False)
    angles = np.arccos(np.clip(singular, -1.0, 1.0))
    return float(np.linalg.norm(angles))


def linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    x = left - left.mean(axis=0, keepdims=True)
    y = right - right.mean(axis=0, keepdims=True)
    cross = np.square(x.T @ y).sum()
    denominator = np.sqrt(np.square(x.T @ x).sum() * np.square(y.T @ y).sum())
    return float(cross / denominator) if denominator > 0 else float("nan")


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def spearman(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.corrcoef(rankdata(np.asarray(left)), rankdata(np.asarray(right)))[0, 1])
