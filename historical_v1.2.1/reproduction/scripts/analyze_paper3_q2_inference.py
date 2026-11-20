"""Exploratory HAC/interaction inference and exhaustive influence diagnostics."""
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
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regression(y, state, lag, family_size=10):
    y = np.asarray(y, float)
    x = np.ones((len(y), 1)) if state is None else np.column_stack([np.ones(len(y)), state])
    if np.linalg.matrix_rank(x) != x.shape[1]:
        return None
    result = sm.OLS(y, x).fit()
    covariance = cov_hac(result, nlags=lag, use_correction=True)
    estimate = float(result.params[-1])
    se = float(np.sqrt(covariance[-1, -1]))
    if not np.isfinite(se) or se <= 0:
        raise ValueError('Degenerate standard error')
    return dict(estimate=estimate, se=se, p=float(2*norm.sf(abs(estimate/se))),
                ci95=[estimate-norm.ppf(.975)*se, estimate+norm.ppf(.975)*se],
                simultaneous_ci95=[estimate-norm.ppf(1-.05/(2*family_size))*se,
                                   estimate+norm.ppf(1-.05/(2*family_size))*se])


def family_results(gain, environments, lag):
    results = {}
    for arm, y in gain.items():
        results[arm + ':overall'] = regression(y, None, lag)
        for version, state in environments.items():
            results[arm + ':' + version] = regression(y, state, lag)
    if any(r is None for r in results.values()):
        raise ValueError('Full-sample interaction not identified')
    adjusted = multipletests([r['p'] for r in results.values()], method='holm')[1]
    for r, p in zip(results.values(), adjusted):
        r['p_holm10'] = float(p)
    return results


def points(y, environments):
    return {'overall': float(y.mean()), **{k: float(y[s == 1].mean()-y[s == 0].mean())
                                          if len(np.unique(s)) == 2 else None for k, s in environments.items()}}


def spells(state):
    return int(np.sum((state == 1) & np.r_[True, state[:-1] != 1]))


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    source = Path(cfg['input_dir'])
    for name, key in [('monthly_losses.json', 'monthly_sha256'), ('predictions.jsonl', 'predictions_sha256')]:
        assert sha(source / name) == cfg[key], name
    rows = json.loads((source / 'monthly_losses.json').read_text())
    months = np.array([r['month'] for r in rows])
    assert months.tolist() == [y*100+m for y in range(2000, 2020) for m in range(1, 13)]
    environments = {k: np.array([r['environment'][k] for r in rows]) for k in cfg['environments']}
    gain = {a: np.array([r['loss']['frozen']-r['loss'][a] for r in rows]) for a in cfg['arms']}
    report = {'status': cfg['status'], 'n_months': len(rows), 'gain_definition': 'frozen loss minus updated loss; positive means update helps',
              'primary': family_results(gain, environments, cfg['primary_hac_lag']),
              'lag_sensitivity': {str(lag): family_results(gain, environments, lag) for lag in cfg['sensitivity_hac_lags']},
              'state_support': {k: {'pressure_months': int(s.sum()), 'pressure_spells': spells(s)} for k, s in environments.items()}}
    report['decades_descriptive'] = {}
    for first, last in [(200001, 200912), (201001, 201912)]:
        use = (months >= first) & (months <= last)
        report['decades_descriptive'][str(first)] = {a: points(y[use], {k: s[use] for k, s in environments.items()}) for a, y in gain.items()}
    report['leave_year_out_descriptive'] = {}
    for year in range(2000, 2020):
        use = months//100 != year
        report['leave_year_out_descriptive'][str(year)] = {a: points(y[use], {k: s[use] for k, s in environments.items()}) for a, y in gain.items()}
    report['largest_absolute_months'] = {a: [{'month': int(months[t]), 'gain': float(y[t]), 'contribution_to_full_mean': float(y[t]/len(y))}
                                                for t in np.argsort(-abs(y), kind='stable')[:5]] for a, y in gain.items()}
    # Reconstruct family means, preserving the original equal-family weighting.
    assets = {}
    with (source / 'predictions.jsonl').open() as f:
        for line in f:
            r = json.loads(line)
            family = r['asset'].split(':')[0]
            losses = [(r['actual']-r['predictions'][a])**2 / r['sigma']**2 for a in ['frozen']+cfg['arms']]
            assets.setdefault(family, {}).setdefault(r['month'], []).append(losses)
    families = {k: np.array([np.mean(v[int(m)], axis=0) for m in months]) for k, v in assets.items()}
    pooled = np.mean(list(families.values()), axis=0)
    np.testing.assert_allclose(pooled, [[r['loss'][a] for a in ['frozen']+cfg['arms']] for r in rows], rtol=1e-12)
    report['leave_family_out_descriptive'] = {}
    report['family_means_descriptive'] = {}
    for k, values in families.items():
        leave = np.mean([v for name, v in families.items() if name != k], axis=0)
        report['leave_family_out_descriptive'][k] = {a: points(leave[:, 0]-leave[:, j+1], environments) for j, a in enumerate(cfg['arms'])}
        report['family_means_descriptive'][k] = {a: float(np.mean(values[:, 0]-values[:, j+1])) for j, a in enumerate(cfg['arms'])}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    lines = ['# 冻结与更新：探索性统计检验', '',
             '沿用2000—2019年240个月与199个组合，不重新训练、不改变环境定义。正的损失差表示更新更好；环境交互为压力状态减普通状态的更新收益差。单位均为标准化平方预测误差，不是投资回报率。', '',
             '主分析：逐月配对，Newey–West/Bartlett HAC滞后12个月，小样本因子T/(T−k)，双侧渐近正态检验。2个总体均值及8个环境交互共10项采用Holm修正。环境回归保留完整日历顺序，不拼接不连续的压力月份。', '',
             '这是看过均值后的探索性分析，不是独立确认。12个月只是固定的工作带宽；另完整报告6、24、60个月，不挑最显著版本。HAC依赖弱相关等渐近条件，少数危机事件、长期估计误差依赖可能影响有效性；Holm不能修复错误的单项p值。没有重新估计模型，也没有认证总体学习算法的重复抽样性能。', '',
             '## 主分析（HAC 12个月）', '',
             '|更新方式/比较|损失差|单项95%区间|十项同时95%区间（Bonferroni）|Holm修正p|', '|---|---:|---|---|---:|']
    for key, r in report['primary'].items():
        ci = lambda v: f'[{v[0]:.6f}, {v[1]:.6f}]'
        lines.append(f"|{key}|{r['estimate']:.6f}|{ci(r['ci95'])}|{ci(r['simultaneous_ci95'])}|{r['p_holm10']:.5f}|")
    lines += ['', '## 带宽敏感性', '', '|比较|6月修正p|12月修正p|24月修正p|60月修正p|', '|---|---:|---:|---:|---:|']
    for key in report['primary']:
        values = [report['lag_sensitivity']['6'][key], report['primary'][key], report['lag_sensitivity']['24'][key], report['lag_sensitivity']['60'][key]]
        lines.append('|'+key+'|'+'|'.join(f"{r['p_holm10']:.5f}" for r in values)+'|')
    lines += ['', '## 压力状态支持', '']
    for k, s in report['state_support'].items():
        lines.append(f"- {k}：{s['pressure_months']}个月，{s['pressure_spells']}段连续压力区间。月份数不等于独立危机事件数。")
    lines += ['', '2010—2019年A/B1没有压力状态，不能用于重复验证这两项环境差异。', '', '## 全部逐年/逐家族剔除诊断', '',
              '以下仅检查影响程度，所有年份和家族均逐一报告，主分析不剔除任何数据；不对这些诊断挑选p值。', '', '|方式|完整均值|逐年剔除后的均值范围|逐家族剔除后的均值范围|', '|---|---:|---|---|']
    for a in cfg['arms']:
        yr = [r[a]['overall'] for r in report['leave_year_out_descriptive'].values()]
        fm = [r[a]['overall'] for r in report['leave_family_out_descriptive'].values()]
        lines.append(f"|{a}|{gain[a].mean():.6f}|[{min(yr):.6f}, {max(yr):.6f}]|[{min(fm):.6f}, {max(fm):.6f}]|")
    lines += ['', '完整逐年、逐家族、两十年及最大影响月份见同目录report.json。', '',
              '方法参考：[statsmodels HAC](https://www.statsmodels.org/stable/generated/statsmodels.stats.sandwich_covariance.cov_hac.html)、[多重检验](https://www.statsmodels.org/stable/generated/statsmodels.stats.multitest.multipletests.html)。实现版本记录在清单，不以网页新版本替代本地实际版本。']
    (out/'统计检验与影响诊断.md').write_text('\n'.join(lines)+'\n')
    manifest = {'config_sha256': sha(args.config), 'script_sha256': sha(__file__),
                'inputs': {str(source/name): sha(source/name) for name in ['monthly_losses.json', 'predictions.jsonl']},
                'runtime': {'python': platform.python_version(), 'executable': sys.executable, 'numpy': np.__version__, 'scipy': scipy.__version__, 'statsmodels': statsmodels.__version__},
                'outputs': {p.name: sha(p) for p in out.iterdir()}}
    (out/'output_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['primary'], indent=2))


if __name__ == '__main__':
    main()
