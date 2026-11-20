"""CUDA implementation of historical-only monthly MLP refits.

This is intentionally separate from the active CPU implementation so an in-flight
CPU run and its recorded source hashes cannot be changed by GPU preparation.
"""
import argparse
import contextlib
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn


class ReturnMLP(nn.Module):
    def __init__(self, inputs=214, widths=(32, 32)):
        super().__init__()
        layers = []
        for width in widths:
            layers.extend([nn.Linear(inputs, width), nn.SiLU()])
            inputs = width
        final = nn.Linear(inputs, 1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.network = nn.Sequential(*layers, final)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def weights(months):
    _, inverse, counts = np.unique(months, return_inverse=True, return_counts=True)
    value = 1 / counts[inverse]
    return (value / value.mean()).astype(np.float32)


def training_bounds(months, target, arm):
    ordinal = (months // 100) * 12 + (months % 100)
    target_ordinal = (target // 100) * 12 + (target % 100)
    stop = int(np.searchsorted(ordinal, target_ordinal, side="left"))
    start = 0 if arm == "expanding" else int(np.searchsorted(ordinal, target_ordinal - 60, side="left"))
    if arm not in ["expanding", "rolling60"]:
        raise ValueError(arm)
    if stop <= start or np.any(months[start:stop] >= target):
        raise ValueError("Invalid historical training slice")
    if arm == "rolling60" and len(np.unique(ordinal[start:stop])) != 60:
        raise ValueError("Incomplete rolling calendar support")
    return start, stop


def train_device(x, y, month, seed, cfg, out, device):
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    # Initialize on CPU so seeds map to the same initial parameters as the CPU implementation.
    model = ReturnMLP(x.shape[1], cfg["hidden_widths"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    tx = torch.from_numpy(x).to(device)
    ty = torch.from_numpy(y).to(device)
    tw = torch.from_numpy(weights(month)).to(device)
    generator = torch.Generator().manual_seed(seed)
    history = []
    started = time.time()
    for epoch in range(cfg["epochs"]):
        order = torch.randperm(len(x), generator=generator)
        total = 0.0
        for lo in range(0, len(x), cfg["batch_size"]):
            idx = order[lo:lo + cfg["batch_size"]].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = (tw[idx] * (model(tx[idx]) - ty[idx]).square()).mean()
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite training loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clip_norm"])
            optimizer.step()
            total += float(loss.detach().cpu()) * len(idx)
        history.append({"epoch": epoch + 1, "online_training_loss": total / len(x)})
        print(f"seed {seed} epoch {epoch + 1}/{cfg['epochs']}", flush=True)
    torch.save(model.state_dict(), out / f"updated_seed{seed}.pt")
    return model, {"seed": seed, "seconds": time.time() - started, "history": history}


def predict(model, x, batch_size, device):
    result = np.empty(len(x), np.float32)
    model.eval()
    with torch.no_grad():
        for lo in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[lo:lo + batch_size]).to(device)
            result[lo:lo + len(batch)] = model(batch).detach().cpu().numpy()
    return result


def summarize(rows, arms):
    loss = {arm: float(np.mean([row["loss"][arm] for row in rows])) for arm in arms + ["zero"]}
    return {
        "months": len(rows), "stock_months": sum(row["stocks"] for row in rows), "loss": loss,
        "r2_against_zero": {arm: 1 - loss[arm] / loss["zero"] for arm in arms},
        "relative_gain": {arm: (loss["frozen"] - loss[arm]) / loss["frozen"] for arm in arms[1:]},
    }


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if cfg.get("device") != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("registered CUDA execution requires an available CUDA device")
    device = torch.device("cuda", int(cfg.get("cuda_device", 0)))
    torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True)
    source = Path(cfg["frozen_dir"])
    out = Path(args.output)
    if sha(source / "output_manifest.json") != cfg["frozen_manifest_sha256"]:
        raise RuntimeError("frozen manifest checksum mismatch")
    if sha(cfg["training_config"]) != cfg["training_config_sha256"]:
        raise RuntimeError("training config checksum mismatch")
    settings = json.loads(Path(cfg["training_config"]).read_text())
    manifest = json.loads((source / "output_manifest.json").read_text())
    for name, digest in manifest["outputs"].items():
        if sha(source / name) != digest:
            raise RuntimeError(f"frozen parent checksum mismatch: {name}")
    frozen_rows = json.loads((source / "monthly_losses.json").read_text())
    selected_rows = frozen_rows[: int(cfg.get("max_months", len(frozen_rows)))]
    if not selected_rows:
        raise RuntimeError("no target months selected")
    years = sorted(int(path.name.split("_")[0]) for path in (source / "arrays").glob("*_month.npy"))
    n = sum(len(np.load(source / "arrays" / f"{year}_month.npy", mmap_mode="r")) for year in years)
    x = np.empty((n, 214), np.float32)
    y = np.empty(n, np.float32)
    months = np.empty(n, np.int32)
    offset = 0
    for year in years:
        ym = np.load(source / "arrays" / f"{year}_month.npy")
        stop = offset + len(ym)
        x[offset:stop] = np.load(source / "arrays" / f"{year}_x.npy")
        y[offset:stop] = np.load(source / "arrays" / f"{year}_y.npy")
        months[offset:stop] = ym
        offset = stop
    if not np.all(months[1:] >= months[:-1]) or np.max(months) > 201912:
        raise RuntimeError("array month ordering failed")
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    parity = []
    started = time.time()
    for frozen_row in selected_rows:
        target = frozen_row["month"]
        lo = int(np.searchsorted(months, target, "left"))
        hi = int(np.searchsorted(months, target, "right"))
        if hi - lo != frozen_row["stocks"]:
            raise RuntimeError("evaluation row count mismatch")
        xm, ym = x[lo:hi], y[lo:hi]
        record = {
            "month": target, "stocks": hi - lo, "environment": frozen_row["environment"],
            "loss": {"frozen": frozen_row["loss"]["frozen"], "zero": frozen_row["loss"]["zero"]},
            "training": {},
        }
        folder = out / str(target)
        folder.mkdir()
        for arm in cfg["arms"]:
            arm_dir = folder / arm
            arm_dir.mkdir()
            start, stop = training_bounds(months, target, arm)
            predicted = np.empty((hi - lo, len(settings["seeds"])), np.float32)
            histories = []
            for index, seed in enumerate(settings["seeds"]):
                with (arm_dir / f"training_seed{seed}.log").open("w") as log, contextlib.redirect_stdout(log):
                    model, history = train_device(
                        x[start:stop], y[start:stop], months[start:stop], seed, settings, arm_dir, device,
                    )
                predicted[:, index] = predict(model, xm, settings["batch_size"], device)
                histories.append(history)
                del model
                torch.cuda.empty_cache()
            if not np.isfinite(predicted).all():
                raise RuntimeError("nonfinite prediction")
            if target == 200002 and arm == "expanding":
                original = np.load(source / "2000_seed_predictions.npy")[: hi - lo]
                delta = np.abs(predicted.astype(np.float64) - original.astype(np.float64))
                parity.append({
                    "month": target, "arm": arm, "max_abs_prediction_difference": float(delta.max()),
                    "mean_abs_prediction_difference": float(delta.mean()),
                    "cpu_loss": float(frozen_row["loss"]["frozen"]),
                    "cuda_loss": float(np.mean((ym - predicted.mean(axis=1)).astype(np.float64) ** 2)),
                })
            np.save(arm_dir / "seed_predictions.npy", predicted)
            ensemble = predicted.mean(axis=1)
            record["loss"][arm] = float(np.mean((ym - ensemble).astype(np.float64) ** 2))
            record["training"][arm] = {
                "first_target": int(months[start]), "last_target": int(months[stop - 1]),
                "months": int(len(np.unique(months[start:stop]))), "rows": stop - start,
                "seeds": histories,
            }
            (arm_dir / "training.json").write_text(json.dumps(record["training"][arm], indent=2) + "\n")
        rows.append(record)
        (folder / "month_result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
        (out / "progress.json").write_text(json.dumps({
            "completed_months": len(rows), "total_months": len(selected_rows),
            "last_completed_target": target, "elapsed_seconds": time.time() - started,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2) + "\n")
        print(f"completed {target}: {len(rows)}/{len(selected_rows)}", flush=True)
    arms = ["frozen"] + cfg["arms"]
    report = {
        "scope": cfg["scope"], "evaluation": summarize(rows, arms), "parity": parity,
        "by_environment": {
            key: {str(state): summarize([row for row in rows if row["environment"][key] == state], arms)
                  for state in [0, 1] if any(row["environment"][key] == state for row in rows)}
            for key in ["A", "B1", "B2", "C"]
        },
        "elapsed_seconds": time.time() - started,
        "fits": len(rows) * len(cfg["arms"]) * len(settings["seeds"]),
    }
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    (out / "monthly_losses.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    final = {
        "config_sha256": sha(args.config), "script_sha256": sha(__file__),
        "parent_manifest_sha256": cfg["frozen_manifest_sha256"],
        "runtime": {
            "python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
            "device": str(device), "gpu": torch.cuda.get_device_name(device), "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(), "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        },
        "outputs": {str(path.relative_to(out)): sha(path) for path in out.rglob("*") if path.is_file()},
    }
    (out / "output_manifest.json").write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
