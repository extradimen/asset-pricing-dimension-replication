"""Controlled Q2 pilot. Read-only official archives; outputs are append-only."""
import argparse
import csv
import hashlib
import io
import json
import os
import platform
from pathlib import Path
import sys
import zipfile

import numpy as np


FILES = {
    "size_bm_25": ("ken-french-monthly-2026-07/25_Portfolios_5x5_CSV.zip", "a6522ad4ea890fafb3971938292844e256128de0e2f6929166ff2f0f7d0d573a", 25),
    "industry_49": ("ken-french-monthly-2026-07/49_Industry_Portfolios_CSV.zip", "b197eab1059757f3cfa2b92fd520197372237104d3d46a3000c8061e5066a302", 49),
    "size_op_25": ("ken-french-external-six-families-2026-07/25_Portfolios_ME_OP_5x5_CSV.zip", "7cbe57f3cca13039b6fce13f80c4fbfbab59292cfa2984cacec4a6d4e46a022e", 25),
    "size_inv_25": ("ken-french-external-six-families-2026-07/25_Portfolios_ME_INV_5x5_CSV.zip", "88e66145f9e0589ffcb7f8b96c518e917b654d9a4a0407e85562bfae73831f84", 25),
    "size_mom_25": ("ken-french-external-six-families-2026-07/25_Portfolios_ME_Prior_12_2_CSV.zip", "cc76640d778055a500192fe9e8419e7e7516649b5a9c052550be297d5f18a0be", 25),
    "size_beta_25": ("ken-french-external-six-families-2026-07/25_Portfolios_ME_BETA_5x5_CSV.zip", "214b387d364f3dad2c7d2aad78fbbbabd8b1b264f1160672ccbb9bbf05eac155", 25),
    "size_resvar_25": ("ken-french-external-six-families-2026-07/25_Portfolios_ME_RESVAR_5x5_CSV.zip", "a09d29b8bcffa490d1977eb59698c3e2bdaef2ed9fec9e3eb4138b084bae0c6c", 25),
}
RF = ("ken-french-monthly-2026-07/F-F_Research_Data_5_Factors_2x3_CSV.zip", "b8653b411cc5e28917e7ef643bb42f6d2d3703f84bc170eb6ae38d5267c65807")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(path, expected):
    actual = sha(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch: {path}: {actual}")
    return {"path": str(path), "sha256": actual}


def parse_monthly(text, stop, factor=False):
    """Select one monthly table, preserve sentinel missingness, stop before sealed years."""
    lines = text.splitlines()
    if factor:
        start = next(i for i, s in enumerate(lines) if s.strip().startswith(",") and "Mkt-RF" in s and "RF" in s)
    else:
        start = next(i for i, s in enumerate(lines) if "Average Value Weighted Returns -- Monthly" in s) + 1
        while not lines[start].strip():
            start += 1
    header = [x.strip() for x in next(csv.reader([lines[start]]))[1:]]
    rows = {}
    for row in csv.reader(lines[start + 1:]):
        key = row[0].strip() if row else ""
        if len(key) != 6 or not key.isdigit():
            if rows:
                break
            continue
        month = int(key)
        if month > stop:
            break
        if len(row) != len(header) + 1 or month in rows:
            raise ValueError("Malformed/duplicate monthly row")
        values = np.array([float(v) for v in row[1:]])
        values[values <= -99.99] = np.nan
        rows[month] = values / 100.0
    return header, rows


def load_zip(path, stop, factor=False):
    with zipfile.ZipFile(path) as archive:
        members = [n for n in archive.namelist() if n.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError("Expected one CSV per archive")
        return parse_monthly(archive.read(members[0]).decode("utf-8-sig"), stop, factor)


def features(y):
    x = np.full((len(y), 4), np.nan)
    for t in range(12, len(y)):
        history = y[t-12:t]
        if np.isfinite(history).all():
            x[t] = [history[-1], history[-3:].mean(), history.mean(), history.std(ddof=1)]
    return x


def fit(x, y, alpha):
    z = np.column_stack((np.ones(len(x)), x))
    penalty = np.diag([0.0] + [alpha] * x.shape[1])
    return np.linalg.solve(z.T @ z / len(y) + penalty, z.T @ y / len(y))


def predict_asset(y, months, cfg):
    x = features(y)
    valid = np.isfinite(x).all(axis=1) & np.isfinite(y)
    source = valid & (months >= cfg["source_start"]) & (months <= cfg["source_end"])
    if source.sum() < cfg["minimum_source_rows"]:
        return [], {"excluded_source": True}
    mu = x[source].mean(axis=0)
    scale = np.maximum(x[source].std(axis=0, ddof=1), 1e-8)
    x = (x - mu) / scale
    sigma = max(float(y[source].std(ddof=1)), 0.005)
    frozen = fit(x[source], y[source], cfg["ridge_alpha"])
    records = []
    for t in np.flatnonzero((months >= cfg["target_start"]) & (months <= cfg["target_end"])):
        # Prediction eligibility uses past covariates and training availability only.
        if not np.isfinite(x[t]).all():
            continue
        expanding = valid & (months >= cfg["source_start"]) & (np.arange(len(y)) < t)
        rolling = expanding & (np.arange(len(y)) >= t - cfg["rolling_months"])
        if rolling.sum() < cfg["minimum_rolling_rows"]:
            continue
        betas = [frozen, fit(x[rolling], y[rolling], cfg["ridge_alpha"]), fit(x[expanding], y[expanding], cfg["ridge_alpha"])]
        train_masks = [source, rolling, expanding]
        stops = [int(months[m][-1]) for m in train_masks]
        assert all(s < months[t] for s in stops)
        predictions = [float(np.r_[1., x[t]] @ b) for b in betas]
        if not np.isfinite(y[t]):
            continue  # Actual outcome availability, same across arms; not a fitting rule.
        records.append({"month": int(months[t]), "actual": float(y[t]), "sigma": sigma,
                        "predictions": dict(zip(cfg["arms"], predictions)),
                        "train_stop": dict(zip(cfg["arms"], stops)),
                        "train_n": dict(zip(cfg["arms"], [int(m.sum()) for m in train_masks]))})
    return records, {"excluded_source": False, "source_n": int(source.sum()), "sigma": sigma,
                     "feature_mean": mu.tolist(), "feature_scale": scale.tolist(), "evaluated_months": len(records)}


def summarize(rows, arms):
    if not rows:
        return {"n": 0, "loss": None, "gain": None, "relative_reduction": None}
    loss = {a: float(np.mean([r["loss"][a] for r in rows])) for a in arms}
    gain = {a: loss["frozen"] - loss[a] for a in arms[1:]}
    return {"n": len(rows), "loss": loss, "gain": gain,
            "relative_reduction": {a: v / loss["frozen"] for a, v in gain.items()}}


def main():
    # The global controller starts in its registered project, not this worktree.
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    root = Path(cfg["data_root"])
    inputs = [verify(root / p, h) for p, h in cfg["snapshots"]]
    inputs.append(verify(cfg["labels"], cfg["labels_sha256"]))
    label_rows = json.loads(Path(cfg["labels"]).read_text())
    labels = {int(r["month"][:7].replace("-", "")): {k: r[k] for k in ["A", "B1", "B2", "C"]} for r in label_rows}
    rf_path = root / "data/raw/public" / RF[0]
    inputs.append(verify(rf_path, RF[1]))
    rf_header, rf_rows = load_zip(rf_path, cfg["target_end"], factor=True)
    rf_idx = rf_header.index("RF")
    months = np.array([y*100+m for y in range(1963, 2020) for m in range(1, 13) if 196307 <= y*100+m <= cfg["target_end"]])
    monthly = {}
    asset_audit = {}
    with (out / "predictions.jsonl").open("w") as sink:
        for family, (relative, digest, count) in FILES.items():
            path = root / "data/raw/public" / relative
            inputs.append(verify(path, digest))
            header, rows = load_zip(path, cfg["target_end"])
            assert len(header) == count
            for j, name in enumerate(header):
                y = np.array([rows[m][j] - rf_rows[m][rf_idx] if m in rows and m in rf_rows else np.nan for m in months])
                predictions, audit = predict_asset(y, months, cfg)
                asset_id = f"{family}:{j}:{name}"
                asset_audit[asset_id] = audit
                for r in predictions:
                    sink.write(json.dumps({"asset": asset_id, **r}, allow_nan=False) + "\n")
                    loss = [(r["actual"] - r["predictions"][a])**2 / r["sigma"]**2 for a in cfg["arms"]]
                    monthly.setdefault(r["month"], {}).setdefault(family, []).append(loss)
            print(f"completed {family}: {count} assets", flush=True)
    panel = []
    for m, families in sorted(monthly.items()):
        assert set(families) == set(FILES), "Do not silently alter family weights"
        loss = np.mean([np.mean(v, axis=0) for v in families.values()], axis=0)
        panel.append({"month": m, "loss": dict(zip(cfg["arms"], loss.tolist())), "environment": labels[m],
                      "family_counts": {k: len(v) for k, v in families.items()}})
    report = {"experiment_id": cfg["experiment_id"], "scope": cfg["decision"], "assets": len(asset_audit),
              "months": len(panel), "first_month": panel[0]["month"], "last_month": panel[-1]["month"],
              "minimum_assets_per_month": min(sum(r["family_counts"].values()) for r in panel),
              "temporal_audit": "Every recorded training target strictly precedes its prediction target; all feature and loss scalers frozen from source only.",
              "splits": {}}
    for split, lo, hi in [("2000_2009", 200001, 200912), ("2010_2019", 201001, 201912), ("pooled_exploratory", 200001, 201912)]:
        subset = [r for r in panel if lo <= r["month"] <= hi]
        report["splits"][split] = {"overall": summarize(subset, cfg["arms"]), "environments": {}}
        for v in ["A", "B1", "B2", "C"]:
            report["splits"][split]["environments"][v] = {str(state): summarize([r for r in subset if r["environment"][v] == state], cfg["arms"]) for state in [0, 1]}
    for name, payload in [("report.json", report), ("monthly_losses.json", panel), ("asset_audit.json", asset_audit)]:
        (out / name).write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = ["# 同一模型冻结与更新：公开组合基准", "", "样本：2000年1月—2019年12月，7类199个公开组合。初始训练1965—1999年。", "",
             "三组均为逐组合岭回归，惩罚系数0.1（平均平方误差尺度），截距不惩罚。输入为自身超额收益滞后1期、过去3期均值、过去12期均值和标准差。预处理及损失标准化只用初始训练样本。", "",
             "更新收益＝冻结损失−更新损失；正数表示更新更好。损失先在组合家族内等权，再在7个家族间等权。下表改善率是平均损失差除以冻结平均损失，不是投资收益率。", "",
             "没有根据结果选择窗口或模型，没有报告显著性。事后环境使用冻结标签；公开收益为2026年快照，不是历史数据版本回测。这是直接组合收益预测起步实验，不是个股机器学习最终结果，也不能解释β／λ。"]
    for split, result in report["splits"].items():
        lines += ["", f"## {split}", "", "|环境|月数|冻结损失|滚动60月损失|扩展历史损失|滚动改善率|扩展改善率|", "|---|---:|---:|---:|---:|---:|---:|"]
        groups = [("全样本", result["overall"])] + [(v+"="+state, s) for v, states in result["environments"].items() for state, s in states.items()]
        for name, s in groups:
            if not s["n"]:
                lines.append(f"|{name}|0|不适用|不适用|不适用|不适用|不适用|")
            else:
                values = [f'{s["loss"][a]:.6f}' for a in cfg["arms"]] + [f'{s["relative_reduction"][a]:.2%}' for a in cfg["arms"][1:]]
                lines.append(f'|{name}|{s["n"]}|' + "|".join(values) + "|")
    (out / "冻结与更新结果.md").write_text("\n".join(lines) + "\n")
    manifest = {"inputs": inputs, "config": verify(args.config, sha(args.config)), "script_sha256": sha(__file__),
                "runtime": {"executable": sys.executable, "python": platform.python_version(), "numpy": np.__version__},
                "outputs": {p.name: sha(p) for p in out.iterdir() if p.is_file()},
                "run_environment_ids": {k: os.environ[k] for k in ["EXPERIMENT_RUN_ID", "EXPERIMENT_ID"] if k in os.environ}}
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"completed": cfg["experiment_id"], "months": len(panel), "assets": len(asset_audit)}), flush=True)


if __name__ == "__main__":
    main()
