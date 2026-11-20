#!/usr/bin/env python3
"""Freeze the deterministic paper-7 development sample as a portable NPZ payload."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow
import pyarrow.parquet as pq

from paper7_geometry_core import sha256, write_json


def load_month_samples(panel: Path, sample_size: int, sample_seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    parquet = pq.ParquetFile(panel)
    names = parquet.schema_arrow.names
    features = [name for name in names if name.startswith("x_")] + [name for name in names if name.startswith("missing_")]
    if len(features) != 172:
        raise RuntimeError(f"expected 172 frozen Core-86 inputs, found {len(features)}")
    storage: dict[int, dict[str, list[np.ndarray]]] = defaultdict(lambda: {"permno": [], "x": [], "feature_month": []})
    for batch in parquet.iter_batches(batch_size=50_000, columns=["permno", "month", "target_month", *features]):
        target = batch.column("target_month").to_numpy(zero_copy_only=False).astype("datetime64[M]").astype(np.int32)
        keep = (target >= np.datetime64("2010-01", "M").astype(int)) & (target <= np.datetime64("2019-12", "M").astype(int))
        if not keep.any():
            continue
        target = target[keep]
        permno = batch.column("permno").to_numpy(zero_copy_only=False).astype(np.int64)[keep]
        feature_month = batch.column("month").to_numpy(zero_copy_only=False).astype("datetime64[M]").astype(np.int32)[keep]
        x = np.column_stack([batch.column(name).to_numpy(zero_copy_only=False)[keep].astype(np.float32) for name in features])
        for month in np.unique(target):
            selected = target == month
            storage[int(month)]["permno"].append(permno[selected])
            storage[int(month)]["feature_month"].append(feature_month[selected])
            storage[int(month)]["x"].append(x[selected])
    months = np.array(sorted(storage), dtype=np.int32)
    if len(months) != 120:
        raise RuntimeError(f"expected 120 development target months, found {len(months)}")
    all_permno = []; all_x = []; feature_months = []
    for month in months:
        pieces = storage[int(month)]
        permno = np.concatenate(pieces["permno"])
        x = np.concatenate(pieces["x"])
        feature_month = np.concatenate(pieces["feature_month"])
        if len(permno) < sample_size:
            raise RuntimeError(f"month {month} has only {len(permno)} rows")
        rng = np.random.default_rng(sample_seed + int(month) * 1009)
        chosen = np.sort(rng.choice(len(permno), size=sample_size, replace=False))
        if not np.all(feature_month[chosen] == int(month) - 1):
            raise RuntimeError(f"feature/target month alignment failed at {month}")
        all_permno.append(permno[chosen]); all_x.append(x[chosen]); feature_months.append(int(feature_month[chosen][0]))
    return months, np.array(feature_months, dtype=np.int32), np.stack(all_permno), np.stack(all_x), features


def market_states(market_path: Path, target_months: np.ndarray) -> dict[str, np.ndarray | float]:
    table = pq.read_table(market_path, columns=["month", "vwretd"])
    months = table["month"].to_numpy(zero_copy_only=False).astype("datetime64[M]").astype(np.int32)
    returns = table["vwretd"].to_numpy(zero_copy_only=False).astype(np.float64)
    keep = np.isfinite(returns)
    months, returns = months[keep], returns[keep]
    order = np.argsort(months); months, returns = months[order], returns[order]
    trailing = np.full(len(months), np.nan)
    for index in range(11, len(months)):
        trailing[index] = np.std(returns[index - 11:index + 1], ddof=1)
    training = (months >= np.datetime64("1963-07", "M").astype(int)) & (months <= np.datetime64("1999-12", "M").astype(int))
    high_threshold = float(np.nanquantile(trailing[training], 0.80))
    down_threshold = float(np.nanquantile(returns[training], 0.20))
    lookup = {int(month): index for index, month in enumerate(months)}
    indices = np.array([lookup[int(month) - 1] for month in target_months])
    return {
        "feature_market_return": returns[indices],
        "trailing_12m_volatility": trailing[indices],
        "high_volatility": trailing[indices] >= high_threshold,
        "down_market": returns[indices] <= down_threshold,
        "high_volatility_threshold": high_threshold,
        "down_market_threshold": down_threshold,
    }


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
    panel = Path(config["inputs"]["panel"]); market = Path(config["inputs"]["market_month"])
    for path, expected in [(panel, config["inputs"]["panel_sha256"]), (market, config["inputs"]["market_month_sha256"])]:
        if sha256(path) != expected:
            raise RuntimeError(f"input hash mismatch: {path}")
    months, feature_months, permno, x, feature_names = load_month_samples(panel, config["sampling"]["stocks_per_month"], config["sampling"]["seed"])
    states = market_states(market, months)
    payload = args.output_dir / "paper7_representation_sample.npz"
    np.savez_compressed(payload, target_months=months, feature_months=feature_months, permno=permno, x=x,
                        feature_names=np.array(feature_names), **states)
    gates = {
        "month_count_120": len(months) == 120,
        "equal_count_384": tuple(x.shape[:2]) == (120, config["sampling"]["stocks_per_month"]),
        "feature_count_172": x.shape[2] == 172,
        "finite_features": bool(np.isfinite(x).all()),
        "one_month_lag": bool(np.all(feature_months == months - 1)),
        "unique_security_within_month": bool(all(len(np.unique(row)) == len(row) for row in permno)),
    }
    report = {"schema_version": 1, "experiment_id": config["experiment_id"], "run_id": args.run_id,
              "status": "completed" if all(gates.values()) else "completed_with_failed_gate", "evidence_class": config["evidence_class"],
              "shape": list(x.shape), "target_month_start": str(np.datetime64(int(months[0]), "M")), "target_month_end": str(np.datetime64(int(months[-1]), "M")),
              "payload_size_bytes": payload.stat().st_size, "payload_sha256": sha256(payload), "gates": gates,
              "elapsed_seconds": round(time.time() - started, 3)}
    write_json(args.output_dir / "quality_report.json", report)
    write_json(args.output_dir / "environment.json", {"created_at": datetime.now(timezone.utc).isoformat(), "python": sys.version, "numpy": np.__version__, "pyarrow": pyarrow.__version__, "platform": platform.platform()})
    outputs = [payload, args.output_dir / "quality_report.json", args.output_dir / "environment.json"]
    write_json(args.output_dir / "output_manifest.json", {"schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
              "experiment_id": config["experiment_id"], "run_id": args.run_id,
              "inputs": [{"path": str(path.resolve()), "sha256": sha256(path)} for path in [args.config, panel, market]],
              "outputs": [{"path": path.name, "size_bytes": path.stat().st_size, "sha256": sha256(path)} for path in outputs]})
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(gates.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
