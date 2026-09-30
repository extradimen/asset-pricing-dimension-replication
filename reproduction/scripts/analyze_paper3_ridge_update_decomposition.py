"""Exact update decomposition for archived Ridge coefficients and sufficient statistics."""
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


def summarize(rows, arm):
    keys = ["adjustment_cost_A", "alignment_benefit_2B", "updated_minus_frozen_loss"]
    value = {key: float(np.mean([row[arm][key] for row in rows])) for key in keys}
    value["alignment_B"] = value["alignment_benefit_2B"] / 2
    value["ex_post_lambda_star"] = float(np.clip(value["alignment_B"] / value["adjustment_cost_A"], 0, 1))
    value["months"] = len(rows)
    value["months_full_update_helped"] = sum(row[arm]["updated_minus_frozen_loss"] < 0 for row in rows)
    return value


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg["source_dir"])
    manifest_path = root / "output_manifest.json"
    if sha(manifest_path) != cfg["source_manifest_sha256"]:
        raise RuntimeError("source manifest checksum mismatch")
    manifest = json.loads(manifest_path.read_text())["outputs"]
    required = ["month_sufficient_statistics.npz", "monthly_coefficients.npy", "monthly_losses.json"]
    for name in required:
        if sha(root / name) != manifest[name]:
            raise RuntimeError(f"source output checksum mismatch: {name}")
    stats = np.load(root / "month_sufficient_statistics.npz")
    stat_months = np.array([int(value[:4] + value[5:7]) for value in stats["months"]])
    index = {month: i for i, month in enumerate(stat_months)}
    coefficients = np.load(root / "monthly_coefficients.npy")
    archived = json.loads((root / "monthly_losses.json").read_text())
    if coefficients.shape != (len(archived), len(cfg["arms"]), stats["gram"].shape[1]):
        raise RuntimeError("coefficient geometry mismatch")
    rows = []
    max_identity_error = 0.0
    max_archive_error = 0.0
    for row_index, archived_row in enumerate(archived):
        month = archived_row["month"]
        j = index[month]
        gram, rhs = stats["gram"][j], stats["rhs"][j]
        frozen = coefficients[row_index, 0]
        result = {"month": month, "environment": archived_row["environment"]}
        for arm_index, arm in enumerate(cfg["arms"][1:], start=1):
            delta = coefficients[row_index, arm_index] - frozen
            a = float(delta @ gram @ delta)
            b = float((rhs - gram @ frozen) @ delta)
            reconstructed = a - 2 * b
            archived_delta = archived_row["loss"][arm] - archived_row["loss"]["frozen"]
            identity_error = reconstructed - archived_delta
            max_identity_error = max(max_identity_error, abs(identity_error))
            max_archive_error = max(max_archive_error, abs(identity_error))
            result[arm] = {
                "adjustment_cost_A": a, "alignment_B": b, "alignment_benefit_2B": 2 * b,
                "updated_minus_frozen_loss": float(archived_delta),
                "reconstructed_loss_difference": reconstructed, "identity_error": identity_error,
            }
        rows.append(result)
    if max_identity_error > 1e-10:
        raise RuntimeError(f"Ridge decomposition identity failed: {max_identity_error}")
    periods = {
        "2000_2009": [row for row in rows if row["month"] <= 200912],
        "2010_2019": [row for row in rows if row["month"] >= 201001],
        "2000_2019": rows,
    }
    report = {
        "experiment_id": cfg["experiment_id"], "scope": cfg["scope"], "purpose": cfg["purpose"],
        "months": len(rows), "max_absolute_identity_error": max_identity_error,
        "summary": {arm: {name: summarize(values, arm) for name, values in periods.items()}
                    for arm in cfg["arms"][1:]},
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "monthly_decomposition.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# Ridge更新损失分解", "", "| 更新臂 | 时期 | A调整成本 | 2B对齐收益 | 更新-冻结损失 | 事后lambda* |", "|---|---|---:|---:|---:|---:|"]
    for arm in cfg["arms"][1:]:
        for period in ["2000_2009", "2010_2019", "2000_2019"]:
            value = report["summary"][arm][period]
            lines.append(f"|{arm}|{period.replace('_', '–')}|{value['adjustment_cost_A']:.8f}|{value['alignment_benefit_2B']:.8f}|{value['updated_minus_frozen_loss']:.8f}|{value['ex_post_lambda_star']:.4f}|")
    lines += ["", f"最大数值恒等式误差：{max_identity_error:.3e}。", "", "本轮只是已存档Ridge结果的机制分解，不是新模型或后续期确证。"]
    (out / "分解结果.md").write_text("\n".join(lines) + "\n")
    output_manifest = {
        "config_sha256": sha(args.config), "script_sha256": sha(__file__),
        "parent_manifest_sha256": cfg["source_manifest_sha256"],
        "verified_inputs": {name: manifest[name] for name in required},
        "runtime": {"python": platform.python_version(), "executable": sys.executable, "numpy": np.__version__},
        "outputs": {path.name: sha(path) for path in out.iterdir() if path.is_file()},
    }
    (out / "output_manifest.json").write_text(json.dumps(output_manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
