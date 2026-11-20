"""Exact adjustment-cost/alignment-benefit decomposition for archived predictions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def components(y, frozen, updated):
    y = np.asarray(y, dtype=np.float64)
    frozen = np.asarray(frozen, dtype=np.float64)
    updated = np.asarray(updated, dtype=np.float64)
    error = y - frozen
    increment = updated - frozen
    adjustment_cost = float(np.mean(increment ** 2))
    alignment = float(np.mean(error * increment))
    direct = float(np.mean((y - updated) ** 2 - (y - frozen) ** 2))
    reconstructed = adjustment_cost - 2 * alignment
    return {
        "adjustment_cost_A": adjustment_cost,
        "alignment_B": alignment,
        "alignment_benefit_2B": 2 * alignment,
        "updated_minus_frozen_loss": direct,
        "reconstructed_loss_difference": reconstructed,
        "identity_error": direct - reconstructed,
        "ex_post_lambda_star": float(np.clip(alignment / adjustment_cost, 0, 1)) if adjustment_cost > 0 else 0.0,
    }


def summarize(rows, arm):
    keys = ["adjustment_cost_A", "alignment_B", "alignment_benefit_2B",
            "updated_minus_frozen_loss", "reconstructed_loss_difference"]
    result = {key: float(np.mean([row[arm][key] for row in rows])) for key in keys}
    a, b = result["adjustment_cost_A"], result["alignment_B"]
    result["ex_post_lambda_star_from_mean_terms"] = float(np.clip(b / a, 0, 1)) if a > 0 else 0.0
    result["months"] = len(rows)
    result["stock_months"] = int(sum(row["stocks"] for row in rows))
    result["months_full_update_helped"] = int(sum(row[arm]["updated_minus_frozen_loss"] < 0 for row in rows))
    result["mean_frozen_seed_variance"] = float(np.mean([row[arm]["frozen_seed_variance"] for row in rows]))
    result["mean_updated_seed_variance"] = float(np.mean([row[arm]["updated_seed_variance"] for row in rows]))
    return result


def verify(path, manifest_outputs, root):
    relative = str(path.relative_to(root))
    expected = manifest_outputs.get(relative)
    if expected is None or sha(path) != expected:
        raise RuntimeError(f"parent output verification failed: {relative}")


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    frozen_root = Path(cfg["frozen_dir"])
    updated_root = Path(cfg["updated_dir"])
    frozen_manifest_path = frozen_root / "output_manifest.json"
    updated_manifest_path = updated_root / "output_manifest.json"
    if sha(frozen_manifest_path) != cfg["frozen_manifest_sha256"]:
        raise RuntimeError("frozen manifest mismatch")
    if sha(updated_manifest_path) != cfg["updated_manifest_sha256"]:
        raise RuntimeError("updated manifest mismatch")
    frozen_outputs = json.loads(frozen_manifest_path.read_text())["outputs"]
    updated_outputs = json.loads(updated_manifest_path.read_text())["outputs"]
    monthly_path = updated_root / "monthly_losses.json"
    verify(monthly_path, updated_outputs, updated_root)
    archived = {row["month"]: row for row in json.loads(monthly_path.read_text())}
    expected_months = [year * 100 + month for year in range(2000, 2020) for month in range(1, 13)][1:]
    if sorted(archived) != expected_months:
        raise RuntimeError("archived month calendar mismatch")

    rows = []
    used_inputs = {}
    for year in range(2000, 2020):
        paths = {
            "months": frozen_root / "arrays" / f"{year}_month.npy",
            "outcomes": frozen_root / "arrays" / f"{year}_y.npy",
            "frozen_predictions": frozen_root / f"{year}_seed_predictions.npy",
        }
        for path in paths.values():
            verify(path, frozen_outputs, frozen_root)
            used_inputs[str(path)] = frozen_outputs[str(path.relative_to(frozen_root))]
        months_all = np.load(paths["months"])
        use = months_all >= 200002
        months = months_all[use]
        outcomes = np.load(paths["outcomes"])[use]
        frozen_seeds = np.load(paths["frozen_predictions"])
        if len(months) != len(outcomes) or len(months) != len(frozen_seeds):
            raise RuntimeError(f"year alignment mismatch: {year}")
        cursor = 0
        for month in np.unique(months):
            count = int(np.sum(months == month))
            sl = slice(cursor, cursor + count)
            frozen_slice = frozen_seeds[sl]
            frozen = frozen_slice.mean(axis=1)
            y = outcomes[sl]
            archived_row = archived[int(month)]
            if count != archived_row["stocks"]:
                raise RuntimeError(f"stock count mismatch: {month}")
            row = {"month": int(month), "stocks": count, "environment": archived_row["environment"]}
            for arm in cfg["arms"]:
                path = updated_root / str(int(month)) / arm / "seed_predictions.npy"
                verify(path, updated_outputs, updated_root)
                used_inputs[str(path)] = updated_outputs[str(path.relative_to(updated_root))]
                updated_seeds = np.load(path)
                if updated_seeds.shape != frozen_slice.shape:
                    raise RuntimeError(f"seed prediction shape mismatch: {month}:{arm}")
                updated = updated_seeds.mean(axis=1)
                result = components(y, frozen, updated)
                result["frozen_seed_variance"] = float(np.mean(np.var(frozen_slice.astype(np.float64), axis=1)))
                result["updated_seed_variance"] = float(np.mean(np.var(updated_seeds.astype(np.float64), axis=1)))
                archived_delta = archived_row["loss"][arm] - archived_row["loss"]["frozen"]
                result["archived_loss_difference"] = float(archived_delta)
                # The archived trainer subtracts float32 arrays before promoting the
                # residual to float64 for squaring. Reproduce that arithmetic for
                # archive verification while retaining float64 for the exact identity.
                archived_numeric_delta = (
                    np.mean((y.astype(np.float32) - updated.astype(np.float32)).astype(np.float64) ** 2)
                    - np.mean((y.astype(np.float32) - frozen.astype(np.float32)).astype(np.float64) ** 2)
                )
                result["archived_numeric_loss_difference"] = float(archived_numeric_delta)
                result["archived_reconstruction_error"] = float(archived_numeric_delta - archived_delta)
                if abs(result["identity_error"]) > cfg["identity_absolute_tolerance"]:
                    raise RuntimeError(f"identity tolerance failed: {month}:{arm}")
                if abs(result["archived_reconstruction_error"]) > cfg["archived_loss_absolute_tolerance"]:
                    raise RuntimeError(f"archive reconstruction failed: {month}:{arm}")
                row[arm] = result
            rows.append(row)
            cursor += count
        if cursor != len(months):
            raise RuntimeError(f"year cursor mismatch: {year}")

    development = [row for row in rows if row["month"] <= cfg["development_end"]]
    promotion = [row for row in rows if row["month"] >= cfg["promotion_start"]]
    summaries = {
        arm: {
            "full_2000_2019": summarize(rows, arm),
            "development_2000_2009": summarize(development, arm),
            "promotion_2010_2019": summarize(promotion, arm),
            "by_environment": {
                version: {str(state): summarize([r for r in rows if r["environment"][version] == state], arm)
                          for state in [0, 1]}
                for version in ["A", "B1", "B2", "C"]
            },
        } for arm in cfg["arms"]
    }
    report = {
        "experiment_id": cfg["experiment_id"], "scope": cfg["scope"],
        "claim_boundary": cfg["claim_boundary"], "identity": cfg["protocol"],
        "months": len(rows), "stock_months": int(sum(row["stocks"] for row in rows)),
        "max_absolute_identity_error": max(abs(row[arm]["identity_error"]) for row in rows for arm in cfg["arms"]),
        "max_absolute_archived_reconstruction_error": max(
            abs(row[arm]["archived_reconstruction_error"]) for row in rows for arm in cfg["arms"]),
        "summary": summaries,
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "monthly_decomposition.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = [
        "# 机械更新的精确损失分解", "",
        "定义A=E[(u-f)^2]为调整成本，2B=2E[(y-f)(u-f)]为对齐收益；更新相对冻结的损失差精确等于A-2B。本轮仅分解已观察的2000–2019结果，不估计新的可行更新规则。", "",
        "| 更新臂 | 时期 | A调整成本 | 2B对齐收益 | 更新-冻结损失 | 事后lambda* | 更新有益月数 |", "|---|---|---:|---:|---:|---:|---:|",
    ]
    labels = [("development_2000_2009", "2000–2009"), ("promotion_2010_2019", "2010–2019"),
              ("full_2000_2019", "2000–2019")]
    for arm in cfg["arms"]:
        for key, label in labels:
            value = summaries[arm][key]
            lines.append(f"|{arm}|{label}|{value['adjustment_cost_A']:.8f}|{value['alignment_benefit_2B']:.8f}|{value['updated_minus_frozen_loss']:.8f}|{value['ex_post_lambda_star_from_mean_terms']:.4f}|{value['months_full_update_helped']}/{value['months']}|")
    lines += ["", f"最大恒等式误差为{report['max_absolute_identity_error']:.3e}；最大存档损失重建误差为{report['max_absolute_archived_reconstruction_error']:.3e}。", "",
              "事后lambda*仅用于诊断信号尺度，不是可交易策略或晋级证据。"]
    (out / "分解结果.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "config_sha256": sha(args.config), "script_sha256": sha(__file__),
        "parent_manifests": {str(frozen_manifest_path): cfg["frozen_manifest_sha256"],
                             str(updated_manifest_path): cfg["updated_manifest_sha256"]},
        "verified_used_inputs": used_inputs,
        "runtime": {"python": platform.python_version(), "executable": sys.executable, "numpy": np.__version__},
        "outputs": {path.name: sha(path) for path in out.iterdir() if path.is_file()},
    }
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
