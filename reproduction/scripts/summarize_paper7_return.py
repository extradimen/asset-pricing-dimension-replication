#!/usr/bin/env python3
"""Create the deterministic rounded summary used for paper-7 result return."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def rows(root: Path, name: str) -> list[dict[str, str]]:
    with (root / name).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-run", default="P7-G1-V001-R001")
    args = parser.parse_args()
    g = rows(args.run_dir, "monthly_geometry.csv"); d = rows(args.run_dir, "subspace_drift.csv")
    c = rows(args.run_dir, "layer_cka.csv"); m = rows(args.run_dir, "model_geometry_endpoints.csv")
    mean = lambda values: sum(values) / len(values)
    q = lambda value: round(float(value), 3)
    raw3 = [row for row in g if row["layer"] == "hidden3" and row["normalization"] == "raw_centered"]
    high = [float(row["participation_rank"]) for row in raw3 if row["high_volatility"] == "True"]
    other = [float(row["participation_rank"]) for row in raw3 if row["high_volatility"] == "False"]
    layer_medians = {layer: q(statistics.median([float(row["participation_rank"]) for row in g if row["layer"] == layer and row["normalization"] == "raw_centered"])) for layer in ["hidden1", "hidden2", "hidden3"]}
    norm_medians = {norm: q(statistics.median([float(row["participation_rank"]) for row in g if row["layer"] == "hidden3" and row["normalization"] == norm])) for norm in ["raw_centered", "feature_zscore", "row_l2_centered"]}
    cka_medians = {pair: q(statistics.median([float(row["linear_cka"]) for row in c if row["layer_pair"] == pair])) for pair in ["hidden1-hidden2", "hidden2-hidden3"]}
    drift_medians = {layer + "|" + norm: q(statistics.median([float(row["grassmann_distance"]) for row in d if row["layer"] == layer and row["normalization"] == norm])) for layer in ["hidden1", "hidden2", "hidden3"] for norm in ["raw_centered", "feature_zscore", "row_l2_centered"]}
    probe = json.loads((args.run_dir / "functional_probe.json").read_text(encoding="utf-8"))
    probe_q = [{key: (q(value) if isinstance(value, float) else value) for key, value in item.items()} for item in probe]
    models = [{"factor_count": int(row["factor_count"]), "seed": int(row["seed"]), "participation_rank": q(row["geometry_value"]), "hj_loss": q(row["development_hj_loss"]), "factor_sharpe": q(row["development_factor_sharpe"])} for row in m]
    output = {"schema_version": 1, "source_run": args.source_run, "precision_decimals": 3,
              "counts": {"geometry": len(g), "drift": len(d), "cka": len(c), "models": len(m)},
              "collapse_rate": q(mean([row["collapsed"] == "True" for row in g])),
              "layer_raw_participation_medians": layer_medians,
              "hidden3_normalization_participation_medians": norm_medians,
              "hidden3_high_vol_minus_other": q(mean(high) - mean(other)), "cka_medians": cka_medians,
              "drift_medians": drift_medians, "functional_probes": probe_q, "model_endpoints": models}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
