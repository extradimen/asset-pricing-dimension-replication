"""Evaluate the locked historical-only confidence-gated anchored update rule."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
from scipy.stats import norm
import scipy
import statsmodels
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.sandwich_covariance import cov_hac


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def hac_mean_se(values, lag):
    values = np.asarray(values, dtype=float)
    fit = sm.OLS(values, np.ones((len(values), 1))).fit()
    covariance = cov_hac(fit, nlags=lag, use_correction=True)
    return float(values.mean()), float(np.sqrt(covariance[0, 0]))


def historical_lambda(history, arm, lag, z):
    a = float(np.mean([row[arm]["adjustment_cost_A"] for row in history]))
    b, se = hac_mean_se([row[arm]["alignment_B"] for row in history], lag)
    lower = b - z * se
    value = float(np.clip(b / a, 0, 1)) if a > 0 and lower > 0 else 0.0
    return value, {"A_hat": a, "B_hat": b, "B_hac_se": se, "B_lower_one_sided_95": lower}


def circular_bootstrap(values, draws, block, seed):
    values = np.asarray(values, dtype=float)
    observed = float(values.mean())
    centered = values - observed
    rng = np.random.default_rng(seed)
    samples = np.empty(draws, dtype=float)
    blocks = int(np.ceil(len(values) / block))
    offsets = np.arange(block)
    for draw in range(draws):
        starts = rng.integers(0, len(values), size=blocks)
        index = ((starts[:, None] + offsets) % len(values)).ravel()[:len(values)]
        samples[draw] = centered[index].mean()
    p = float((1 + np.sum(np.abs(samples) >= abs(observed))) / (draws + 1))
    quantiles = np.quantile(samples, [0.025, 0.975])
    return {"estimate": observed, "basic_ci95": [float(observed - quantiles[1]), float(observed - quantiles[0])],
            "p_two_sided_centered": p}


def stability(values, months):
    values = np.asarray(values, dtype=float)
    months = np.asarray(months, dtype=int)
    return {
        "2010_2014": float(values[months <= 201412].mean()),
        "2015_2019": float(values[months >= 201501].mean()),
        "leave_one_year_out": {str(year): float(values[months // 100 != year].mean()) for year in range(2010, 2020)},
    }


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    protocol_path = Path(cfg["protocol"])
    source_path = Path(cfg["monthly_decomposition"])
    if sha(source_path) != cfg["monthly_decomposition_sha256"]:
        raise RuntimeError("decomposition checksum mismatch")
    protocol = json.loads(protocol_path.read_text())
    if protocol["parent_decomposition_sha256"] != cfg["monthly_decomposition_sha256"]:
        raise RuntimeError("protocol is not bound to decomposition")
    rows = json.loads(source_path.read_text())
    arm = cfg["arm"]
    h = cfg["history_months"]
    decisions = []
    for index in range(h, len(rows)):
        current = rows[index]
        history = rows[index-h:index]
        if history[-1]["month"] >= current["month"]:
            raise RuntimeError("nonhistorical update information")
        lam, estimates = historical_lambda(history, arm, cfg["hac_lag"], cfg["one_sided_z"])
        term = current[arm]
        anchored_delta = lam * lam * term["adjustment_cost_A"] - 2 * lam * term["alignment_B"]
        full_delta = term["updated_minus_frozen_loss"]
        decisions.append({
            "month": current["month"], "lambda": lam, **estimates,
            "anchored_minus_frozen_loss": float(anchored_delta),
            "full_minus_frozen_loss": float(full_delta),
            "frozen_minus_anchored_improvement": float(-anchored_delta),
            "full_minus_anchored_improvement": float(full_delta - anchored_delta),
        })
    promotion = [row for row in decisions if cfg["promotion_start"] <= row["month"] <= cfg["promotion_end"]]
    if len(promotion) != 120:
        raise RuntimeError("promotion calendar mismatch")
    months = np.array([row["month"] for row in promotion], dtype=int)
    comparisons = {
        "anchored_vs_frozen": np.array([row["frozen_minus_anchored_improvement"] for row in promotion]),
        "anchored_vs_full": np.array([row["full_minus_anchored_improvement"] for row in promotion]),
    }
    inference = {
        key: {**circular_bootstrap(value, cfg["bootstrap"]["draws"], cfg["bootstrap"]["block_months"],
                                   cfg["bootstrap"]["seed"] + index),
              "stability": stability(value, months)}
        for index, (key, value) in enumerate(comparisons.items())
    }
    corrected = multipletests([value["p_two_sided_centered"] for value in inference.values()], method="holm")[1]
    for value, p_holm in zip(inference.values(), corrected):
        value["p_holm"] = float(p_holm)
        stable = value["stability"]
        value["positive_both_halves"] = stable["2010_2014"] > 0 and stable["2015_2019"] > 0
        value["positive_all_leave_one_year_out"] = min(stable["leave_one_year_out"].values()) > 0
        value["passes"] = (value["estimate"] > 0 and value["basic_ci95"][0] > 0 and value["p_holm"] < 0.05
                           and value["positive_both_halves"] and value["positive_all_leave_one_year_out"])
    statistical_pass = all(value["passes"] for value in inference.values())
    report = {
        "experiment_id": cfg["experiment_id"], "scope": cfg["scope"], "stop_rule": cfg["stop_rule"],
        "decision_months": len(decisions), "promotion_months": len(promotion),
        "lambda_summary_promotion": {
            "mean": float(np.mean([row["lambda"] for row in promotion])),
            "median": float(np.median([row["lambda"] for row in promotion])),
            "open_months": int(sum(row["lambda"] > 0 for row in promotion)),
            "full_update_months": int(sum(row["lambda"] == 1 for row in promotion)),
        },
        "inference": inference, "statistical_promotion_gate_pass": statistical_pass,
        "next_action": "freeze economic gate before portfolio outcomes" if statistical_pass else "stop this anchored rule; do not access later-period model outcomes",
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "monthly_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2) + "\n")
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    lines = ["# 历史可行锚定更新：统计晋级门", "",
             f"结论：**{'\u901a过' if statistical_pass else '\u672a通过'}**。规则在2010–2019的平均lambda为{report['lambda_summary_promotion']['mean']:.4f}，120个月中开启{report['lambda_summary_promotion']['open_months']}个月。", "",
             "| 比较 | 损失改善 | 95%区间 | Holm p | 两半期同号 | 逐年剔除同号 | 通过 |", "|---|---:|---|---:|---|---|---|"]
    for key, value in inference.items():
        lines.append(f"|{key}|{value['estimate']:.8f}|[{value['basic_ci95'][0]:.8f}, {value['basic_ci95'][1]:.8f}]|{value['p_holm']:.5f}|{value['positive_both_halves']}|{value['positive_all_leave_one_year_out']}|{value['passes']}|")
    lines += ["", report["next_action"]]
    (out / "晋级门结果.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "config_sha256": sha(args.config), "protocol_sha256": sha(protocol_path), "script_sha256": sha(__file__),
        "input_sha256": {str(source_path): sha(source_path)},
        "runtime": {"python": platform.python_version(), "executable": sys.executable, "numpy": np.__version__,
                    "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "outputs": {path.name: sha(path) for path in out.iterdir() if path.is_file()},
    }
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
