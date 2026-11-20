#!/usr/bin/env python3
"""Run a deterministic neural-SDF seed grid across assigned CUDA devices."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--factors", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-git-revision", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--seeds", default="20260924,20260925,20260926,20260927,20260928")
    parser.add_argument("--cuda-devices", default="0,1")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--pricing-targets", type=Path)
    parser.add_argument("--pricing-penalty", type=float, default=0.0)
    args = parser.parse_args()

    seeds = [int(value) for value in args.seeds.split(",") if value]
    devices = [value for value in args.cuda_devices.split(",") if value]
    if not seeds or not devices:
        raise ValueError("At least one seed and CUDA device are required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    entrypoint = Path(__file__).resolve().with_name("run_neural_sdf_teacher.py")

    def run(index_seed: tuple[int, int]) -> dict[str, object]:
        index, seed = index_seed
        physical_device = devices[index % len(devices)]
        output = args.output_root / f"seed-{seed}"
        log_path = args.output_root / f"seed-{seed}.log"
        if output.exists() or log_path.exists():
            raise FileExistsError(f"Seed output exists: {seed}")
        command = [
            args.python, str(entrypoint), "--input", str(args.input), "--factors", str(args.factors),
            "--output-dir", str(output), "--source-git-revision", args.source_git_revision,
            "--experiment-id", args.experiment_id,
            "--seed", str(seed), "--device", "cuda:0", "--epochs", "80", "--patience", "12",
            "--month-batch-size", "24", "--learning-rate", "0.0003", "--weight-decay", "0.00001",
            "--concentration-penalty", "0.001",
        ]
        if args.pricing_targets:
            command.extend(["--pricing-targets", str(args.pricing_targets), "--pricing-penalty", str(args.pricing_penalty)])
        environment = os.environ.copy(); environment["CUDA_VISIBLE_DEVICES"] = physical_device
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(command, env=environment, stdout=log, stderr=subprocess.STDOUT, text=True)
        result: dict[str, object] = {
            "seed": seed, "physical_cuda_device": physical_device, "returncode": completed.returncode,
            "command": command, "log": log_path.name, "log_sha256": sha256(log_path),
        }
        if completed.returncode == 0:
            report_path = output / "quality_report.json"
            result["quality_report"] = json.loads(report_path.read_text(encoding="utf-8"))
            result["output_manifest_sha256"] = sha256(output / "output_manifest.json")
        return result

    results: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=len(devices)) as pool:
        futures = [pool.submit(run, item) for item in enumerate(seeds)]
        for future in as_completed(futures):
            result = future.result(); results.append(result); print(json.dumps(result), flush=True)
    results.sort(key=lambda row: int(row["seed"]))
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_git_revision": args.source_git_revision,
        "experiment_id": args.experiment_id,
        "input": {"path": str(args.input.resolve()), "sha256": sha256(args.input)},
        "factors": {"path": str(args.factors.resolve()), "sha256": sha256(args.factors)},
        "seeds": seeds,
        "devices": devices,
        "runs": results,
    }
    manifest_path = args.output_root / "seed_grid_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if any(int(row["returncode"]) != 0 for row in results):
        raise RuntimeError("At least one seed failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
