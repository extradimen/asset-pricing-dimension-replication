"""Exploratory calendar-month HAC inference for the completed neural update run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys

import numpy as np
import scipy
from scipy.stats import norm
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


def regression(y, state, lag):
    y = np.asarray(y, dtype=float)
    x = np.ones((len(y), 1)) if state is None else np.column_stack([np.ones(len(y)), state])
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("comparison is not identified")
    result = sm.OLS(y, x).fit()
    covariance = cov_hac(result, nlags=lag, use_correction=True)
    estimate = float(result.params[-1])
    se = float(np.sqrt(covariance[-1, -1]))
    if not np.isfinite(se) or se <= 0:
        raise ValueError("degenerate HAC standard error")
    return {
        "estimate": estimate,
        "se": se,
        "p": float(2 * norm.sf(abs(estimate / se))),
        "ci95": [estimate - norm.ppf(.975) * se, estimate + norm.ppf(.975) * se],
    }


def adjust_family(results):
    adjusted = multipletests([value["p"] for value in results.values()], method="holm")[1]
    for value, corrected in zip(results.values(), adjusted):
        value["p_holm"] = float(corrected)
    return results


def q1_results(losses, environments, lag):
    return adjust_family({
        f"{arm}:{version}": regression(values, environments[version], lag)
        for arm, values in losses.items() for version in environments
    })


def q2_results(gains, environments, lag):
    result = {}
    for arm, values in gains.items():
        result[f"{arm}:overall"] = regression(values, None, lag)
        for version, state in environments.items():
            result[f"{arm}:{version}"] = regression(values, state, lag)
    return adjust_family(result)


def spells(state):
    state = np.asarray(state)
    return int(np.sum((state == 1) & np.r_[True, state[:-1] != 1]))


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    source = Path(cfg["input_dir"])
    if sha(source / "monthly_losses.json") != cfg["monthly_sha256"]:
        raise RuntimeError("monthly loss checksum mismatch")
    if sha(source / "output_manifest.json") != cfg["output_manifest_sha256"]:
        raise RuntimeError("output manifest checksum mismatch")
    rows = json.loads((source / "monthly_losses.json").read_text())
    months = np.array([row["month"] for row in rows], dtype=int)
    expected = [year * 100 + month for year in range(2000, 2020) for month in range(1, 13)][1:]
    if months.tolist() != expected:
        raise RuntimeError("expected the complete 2000-02 through 2019-12 calendar")
    environments = {name: np.array([row["environment"][name] for row in rows], dtype=int)
                    for name in cfg["environments"]}
    losses = {arm: np.array([row["loss"][arm] for row in rows], dtype=float) for arm in cfg["arms"]}
    gains = {arm: losses["frozen"] - losses[arm] for arm in cfg["updated_arms"]}
    primary_lag = int(cfg["primary_hac_lag"])
    sensitivity = [int(value) for value in cfg["sensitivity_hac_lags"]]
    report = {
        "status": cfg["status"],
        "n_months": len(rows),
        "stock_months": int(sum(row["stocks"] for row in rows)),
        "definitions": {
            "q1": "pressure-state monthly loss minus ordinary-state monthly loss; positive means worse in pressure",
            "q2_overall": "frozen monthly loss minus updated monthly loss; positive means updating helps",
            "q2_environment": "pressure-state update gain minus ordinary-state update gain",
        },
        "primary_hac_lag": primary_lag,
        "q1_primary_holm12": q1_results(losses, environments, primary_lag),
        "q2_primary_holm10": q2_results(gains, environments, primary_lag),
        "q1_lag_sensitivity": {str(lag): q1_results(losses, environments, lag) for lag in sensitivity},
        "q2_lag_sensitivity": {str(lag): q2_results(gains, environments, lag) for lag in sensitivity},
        "state_support": {name: {"pressure_months": int(state.sum()), "pressure_spells": spells(state)}
                          for name, state in environments.items()},
        "leave_year_out_gain_means": {
            str(year): {arm: float(values[months // 100 != year].mean()) for arm, values in gains.items()}
            for year in range(2000, 2020)
        },
        "largest_absolute_gain_months": {
            arm: [{"month": int(months[index]), "gain": float(values[index])}
                  for index in np.argsort(-np.abs(values), kind="stable")[:10]]
            for arm, values in gains.items()
        },
    }
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    lines = [
        "# 神经网络跨环境表现与更新价值：探索性推断", "",
        "样本为2000-02至2019-12的239个共同评价月、1,043,511个股票月。检验以月为观测单位，主带宽为Newey–West/Bartlett HAC 12个月；另固定报告6、24、60个月。Q1的12项和Q2的10项分别做Holm校正。", "",
        "Q1系数是压力月损失减普通月损失；正值表示压力期预测误差更高。Q2总体系数是冻结损失减更新损失；正值表示更新有益。Q2环境系数是压力期更新收益减普通期更新收益。损失是个股原始收益平方误差的月度横截面均值，不是投资收益或风险调整alpha。", "",
        "## Q1：跨环境损失差异（HAC12，Holm 12项）", "",
        "|模型:环境|损失差|95%区间|Holm p|", "|---|---:|---|---:|",
    ]
    for key, value in report["q1_primary_holm12"].items():
        lines.append(f"|{key}|{value['estimate']:.8f}|[{value['ci95'][0]:.8f}, {value['ci95'][1]:.8f}]|{value['p_holm']:.5f}|")
    lines += ["", "## Q2：更新价值及环境交互（HAC12，Holm 10项）", "",
              "|更新:比较|损失改善|95%区间|Holm p|", "|---|---:|---|---:|"]
    for key, value in report["q2_primary_holm10"].items():
        lines.append(f"|{key}|{value['estimate']:.8f}|[{value['ci95'][0]:.8f}, {value['ci95'][1]:.8f}]|{value['p_holm']:.5f}|")
    lines += ["", "## 证据边界", "",
              "这是完整神经网络结果完成后的探索性渐近推断，不是未看结果的独立确认。HAC与Holm不能处理超参数历史选择、有限危机段数或总体学习算法重训不确定性。Q1/Q2均不得解释为β或λ结构改变；Q3仍需独立测量与误差方法认证。", "",
              "完整带宽敏感性、逐年剔除和最大影响月份见report.json。"]
    (out / "推断结果.md").write_text("\n".join(lines) + "\n")
    manifest = {
        "config_sha256": sha(args.config), "script_sha256": sha(__file__),
        "inputs": {str(source / name): sha(source / name) for name in ["monthly_losses.json", "output_manifest.json"]},
        "runtime": {"python": platform.python_version(), "executable": sys.executable,
                    "numpy": np.__version__, "scipy": scipy.__version__, "statsmodels": statsmodels.__version__},
        "outputs": {path.name: sha(path) for path in out.iterdir() if path.is_file()},
    }
    (out / "output_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"q1": report["q1_primary_holm12"], "q2": report["q2_primary_holm10"]}, indent=2))


if __name__ == "__main__":
    main()
