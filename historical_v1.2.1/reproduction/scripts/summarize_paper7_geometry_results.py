#!/usr/bin/env python3
"""Summarize the frozen, audited paper-7 geometry outputs without new searches."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def finite_median(rows: list[dict[str, str]], key: str) -> float | None:
    values = np.array([float(row[key]) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)
    geometry = read_csv(args.run_dir / "monthly_geometry.csv")
    drift = read_csv(args.run_dir / "subspace_drift.csv"); cka = read_csv(args.run_dir / "layer_cka.csv")
    probes = json.loads((args.run_dir / "functional_probe.json").read_text(encoding="utf-8"))
    quality = json.loads((args.run_dir / "quality_report.json").read_text(encoding="utf-8"))
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    metrics = ["participation_rank", "entropy_effective_rank", "stable_rank", "top_eigenvalue_share", "pca90_rank", "pca95_rank", "twonn_dimension", "mle10_dimension", "mle20_dimension", "curvature_proxy_radians"]
    geometry_summary = {}
    for layer in ["hidden1", "hidden2", "hidden3"]:
        for normalization in ["raw_centered", "feature_zscore", "row_l2_centered"]:
            selected = [row for row in geometry if row["layer"] == layer and row["normalization"] == normalization]
            geometry_summary[layer + "|" + normalization] = {key: finite_median(selected, key) for key in metrics}
    drift_summary = {layer + "|" + norm: finite_median([row for row in drift if row["layer"] == layer and row["normalization"] == norm], "grassmann_distance") for layer in ["hidden1", "hidden2", "hidden3"] for norm in ["raw_centered", "feature_zscore", "row_l2_centered"]}
    cka_summary = {pair: finite_median([row for row in cka if row["layer_pair"] == pair], "linear_cka") for pair in ["hidden1-hidden2", "hidden2-hidden3"]}
    raw_layer = {layer: geometry_summary[layer + "|raw_centered"]["participation_rank"] for layer in ["hidden1", "hidden2", "hidden3"]}
    raw3 = raw_layer["hidden3"]; z3 = geometry_summary["hidden3|feature_zscore"]["participation_rank"]
    report = {
        "schema_version": 1, "experiment_id": "P7-G1-V001-REPORT001", "source_run": str(args.run_dir),
        "audit_status": audit["status"], "geometry_summary_medians": geometry_summary,
        "raw_centered_layer_participation_rank_medians": raw_layer,
        "hidden3_minus_hidden1_participation_rank": raw3 - raw_layer["hidden1"],
        "hidden3_zscore_minus_raw_participation_rank": z3 - raw3,
        "hidden3_zscore_to_raw_participation_ratio": z3 / raw3,
        "collapse_rate": quality["collapse_rate"], "hidden3_high_vol_minus_other": quality["hidden3_participation_rank_high_vol_minus_other"],
        "drift_medians": drift_summary, "cka_medians": cka_summary, "functional_probes": probes,
        "decisions": {
            "descriptive_layer_compression": bool(raw_layer["hidden3"] < raw_layer["hidden2"] < raw_layer["hidden1"]),
            "collapse_stop_triggered": bool(quality["collapse_rate"] > 0.10),
            "normalization_artifact_stop_triggered": bool(abs(z3 - raw3) > 0.50 * raw3),
            "primary_functional_gate_passed": bool(quality["functional_gate"]["passed"]),
            "functional_search_stopped": not bool(quality["functional_gate"]["passed"]),
        },
        "interpretation": "Descriptive finite-scale hidden geometry is retained; the preregistered functional gate failed, so no alternative geometry-function search is permitted."
    }
    (args.output_dir / "result_summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
