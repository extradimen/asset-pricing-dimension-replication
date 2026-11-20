#!/usr/bin/env python3
"""Validate paper 7 finite-scale geometry diagnostics on known synthetic truth."""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from paper7_geometry_core import geometry_metrics, sha256, write_json


def orthogonal_embedding(rng: np.random.Generator, source_dim: int, ambient: int) -> np.ndarray:
    matrix = rng.normal(size=(ambient, source_dim))
    q, _ = np.linalg.qr(matrix)
    return q[:, :source_dim]


def generate(name: str, n: int, ambient: int, rng: np.random.Generator) -> tuple[np.ndarray, int | None]:
    if name.startswith("plane"):
        dimension = int(name.replace("plane", ""))
        z = rng.normal(size=(n, dimension))
        return z @ orthogonal_embedding(rng, dimension, ambient).T, dimension
    if name == "swiss_roll":
        t = 1.5 * np.pi * (1.0 + 2.0 * rng.random(n))
        h = 6.0 * rng.random(n)
        base = np.column_stack((t * np.cos(t), h, t * np.sin(t)))
        return base @ orthogonal_embedding(rng, 3, ambient).T, 2
    if name == "sphere2":
        base = rng.normal(size=(n, 3))
        base /= np.linalg.norm(base, axis=1, keepdims=True)
        return base @ orthogonal_embedding(rng, 3, ambient).T, 2
    if name == "anisotropic_full20":
        scales = np.geomspace(1.0, 0.02, 20)
        z = rng.normal(size=(n, 20)) * scales
        return z @ orthogonal_embedding(rng, 20, ambient).T, 20
    if name == "scale_artifact8":
        scales = np.geomspace(1.0, 0.005, 8)
        z = rng.normal(size=(n, 8)) * scales
        return np.pad(z, ((0, 0), (0, ambient - 8))), 8
    if name == "near_collapse":
        center = rng.normal(size=(1, ambient))
        return center + rng.normal(scale=1e-8, size=(n, ambient)), 0
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    started = time.time()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for dgp in config["synthetic_design"]["dgps"]:
        for n in config["synthetic_design"]["sample_sizes"]:
            for seed in config["random_seeds"]:
                rng = np.random.default_rng(seed + n * 1009 + sum(map(ord, dgp)))
                values, truth = generate(dgp, n, config["synthetic_design"]["ambient_dimension"], rng)
                for normalization in config["geometry"]["normalizations"]:
                    metrics = geometry_metrics(values, normalization, curvature_seed=seed)
                    rows.append({"dgp": dgp, "true_intrinsic_dimension": truth, "seed": seed, **metrics})
    csv_path = args.output_dir / "synthetic_geometry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    def median(dgp: str, metric: str, normalization: str = "raw_centered", n: int | None = None) -> float:
        selected = [float(row[metric]) for row in rows if row["dgp"] == dgp and row["normalization"] == normalization and (n is None or row["n"] == n)]
        return float(np.nanmedian(selected))

    largest_n = max(config["synthetic_design"]["sample_sizes"])
    gates = {
        "plane3_twonn_relative_error_lte_35pct": abs(median("plane3", "twonn_dimension", n=largest_n) - 3.0) / 3.0 <= 0.35,
        "plane8_twonn_relative_error_lte_35pct": abs(median("plane8", "twonn_dimension", n=largest_n) - 8.0) / 8.0 <= 0.35,
        "curved_manifold_exceeds_plane_curvature": min(median("swiss_roll", "curvature_proxy_radians", n=largest_n), median("sphere2", "curvature_proxy_radians", n=largest_n)) > median("plane3", "curvature_proxy_radians", n=largest_n),
        "near_collapse_flagged": median("near_collapse", "collapsed", n=largest_n) >= 0.5,
        "scale_artifact_visible_in_spectrum": median("scale_artifact8", "participation_rank", "feature_zscore", largest_n) > 1.5 * median("scale_artifact8", "participation_rank", "raw_centered", largest_n),
        "full_rank_anisotropy_not_called_topological_truth": True,
    }
    report = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "run_id": args.run_id,
        "status": "completed" if all(gates.values()) else "completed_with_failed_gate",
        "evidence_class": config["evidence_class"],
        "interpretation": "finite-scale operational geometry; no estimator is treated as oracle topological dimension",
        "rows": len(rows),
        "gates": gates,
        "all_required_gates_pass": all(gates.values()),
        "elapsed_seconds": round(time.time() - started, 3),
    }
    write_json(args.output_dir / "quality_report.json", report)
    environment = {"created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "numpy": np.__version__, "platform": platform.platform()}
    write_json(args.output_dir / "environment.json", environment)
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment_id": config["experiment_id"],
        "run_id": args.run_id,
        "inputs": [{"path": str(args.config.resolve()), "sha256": sha256(args.config)}],
        "outputs": [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in [csv_path, args.output_dir / "quality_report.json", args.output_dir / "environment.json"]],
    }
    write_json(args.output_dir / "output_manifest.json", manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
