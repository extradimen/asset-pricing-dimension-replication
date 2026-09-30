"""Descriptive labels only; evidence execution requires experimentctl launch.

Read original workbook without editing it. Produce JSON/Markdown, not a workbook.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics
import sys
from datetime import datetime, timezone

VARIABLES = ['dp', 'ep', 'bm', 'ntis', 'tbl', 'tms', 'dfy', 'svar']
NAMES = dict(zip(VARIABLES, ['股息价格比', '盈利价格比', '账面市值比', '净股权发行', '短期利率', '期限利差', '信用利差', '市场方差']))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def shift(month, n):
    y, m = map(int, month[:7].split('-'))
    y, m = divmod(y * 12 + m - 1 + n, 12)
    return f'{y:04d}-{m + 1:02d}-01'


def months(start, end):
    while start <= end:
        yield start
        start = shift(start, 1)


def quantile(values, p):
    a = sorted(values)
    h = (len(a) - 1) * p
    i = math.floor(h)
    return a[i] + (h - i) * (a[min(i + 1, len(a) - 1)] - a[i])


def label(value, cut):
    if value is None or not math.isfinite(value):
        return '缺失'
    if cut['q1'] >= cut['q2']:
        raise ValueError('Collapsed thresholds')
    return '低' if value <= cut['q1'] else '中' if value <= cut['q2'] else '高'


def episodes(rows, key):
    result = []
    for row in rows:
        tag = row['labels'][key]
        date = row['formation_month']
        if result and tag == result[-1]['label'] and date == shift(result[-1]['end'], 1):
            result[-1]['end'] = date
            result[-1]['months'] += 1
        else:
            result.append({'variable': key, 'label': tag, 'start': date, 'end': date, 'months': 1})
    return result


def number(x):
    try:
        x = float(x)
        return x if math.isfinite(x) else None
    except (ValueError, TypeError):
        return None


def derive(row):
    def logratio(a, b):
        x, y = number(row[a]), number(row[b])
        return math.log(x / y) if x is not None and y is not None and x > 0 and y > 0 else None
    def difference(a, b):
        x, y = number(row[a]), number(row[b])
        return x - y if x is not None and y is not None else None
    return {'dp': logratio('D12', 'Index'), 'ep': logratio('E12', 'Index'),
            'bm': number(row['b/m']), 'ntis': number(row['ntis']), 'tbl': number(row['tbl']),
            'tms': difference('lty', 'tbl'), 'dfy': difference('BAA', 'AAA'), 'svar': number(row['svar'])}


def calibrate(data, config, keys):
    result = {}
    expected = list(months(config['calibration_start'], config['calibration_end']))
    for key in keys:
        values = [data.get(m, {}).get(key) for m in expected]
        if any(x is None or not math.isfinite(x) for x in values):
            raise ValueError(f'Incomplete calibration for {key}')
        mean, sd = statistics.mean(values), statistics.pstdev(values)
        if sd <= 0:
            raise ValueError('Constant calibration')
        q1, q2 = [quantile(values, p) for p in config['quantiles']]
        if q1 >= q2:
            raise ValueError('Non-distinct quantiles')
        result[key] = {'q1': q1, 'q2': q2, 'mean': mean, 'sd_population': sd, 'months': len(values)}
    return result


def main():
    import openpyxl
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    root = Path(__file__).resolve().parents[1]
    source = Path(cfg['source_path'])
    assert digest(source) == cfg['source_sha256'], 'Input hash changed'
    assert cfg['calibration_end'] < shift(cfg['formation_start'], -1)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source, out / source.name)
    wb = openpyxl.load_workbook(source, read_only=True, data_only=True)
    iterator = wb[cfg['source_sheet']].iter_rows(values_only=True)
    header = next(iterator)
    required = ['yyyymm', 'Index', 'D12', 'E12', 'b/m', 'ntis', 'tbl', 'lty', 'BAA', 'AAA', 'svar']
    assert all(header.count(k) == 1 for k in required)
    data = {}
    for cells in iterator:
        row = dict(zip(header, cells))
        if row['yyyymm'] is None:
            continue
        date = str(int(row['yyyymm']))
        assert len(date) == 6 and 1 <= int(date[4:]) <= 12
        month = date[:4] + '-' + date[4:] + '-01'
        assert month not in data, 'Duplicate month'
        data[month] = derive(row)
    wb.close()
    cuts = calibrate(data, cfg, VARIABLES)
    rows = []
    for t in months(cfg['formation_start'], cfg['formation_end']):
        obs = shift(t, -cfg['observation_lag_months'])
        values = {k: data.get(obs, {}).get(k) for k in VARIABLES}
        rows.append({'formation_month': t, 'observation_month': obs, 'prediction_month': shift(t, 1),
                     'evidence_class': 'retrospective_latest_history_not_PIT', 'values': values,
                     'labels': {k: label(values[k], cuts[k]) for k in VARIABLES},
                     'standardized': {k: (values[k] - cuts[k]['mean']) / cuts[k]['sd_population'] if values[k] is not None else None for k in VARIABLES}})
    credit_manifest = root / cfg['credit_manifest']
    assert digest(credit_manifest) == cfg['credit_manifest_sha256']
    manifest = json.loads(credit_manifest.read_text())
    base = credit_manifest.parent
    calibration_record = next(r for r in manifest['inputs'] if r['requested_vintage'] == cfg['credit_calibration_vintage'])
    vintage_file = base / 'vintages' / (cfg['credit_calibration_vintage'] + '.csv')
    assert digest(vintage_file) == calibration_record['sha256']
    credit_data = {}
    with vintage_file.open() as f:
        reader = csv.DictReader(f)
        suffix = cfg['credit_calibration_vintage'].replace('-', '')
        assert reader.fieldnames == ['observation_date', 'BAA_' + suffix, 'AAA_' + suffix]
        for r in reader:
            a, b = number(r['BAA_' + suffix]), number(r['AAA_' + suffix])
            credit_data[r['observation_date']] = {'dfy': round(a - b, 10) if a is not None and b is not None else None}
    credit_cuts = calibrate(credit_data, cfg, ['dfy'])
    credit_path = base / 'labels.json'
    assert digest(credit_path) == cfg['credit_labels_sha256']
    old_rows = json.loads(credit_path.read_text())
    assert [r['origin_month'] for r in old_rows] == [r['formation_month'] for r in rows]
    credit_rows = []
    for r in old_rows:
        value = r.get('spread') if r['available'] else None
        credit_rows.append({'formation_month': r['origin_month'], 'observation_month': r.get('required_observation'),
                            'archived_vintage': r.get('vintage_date'), 'values': {'dfy': value},
                            'labels': {'dfy': label(value, credit_cuts['dfy'])},
                            'evidence_class': 'ALFRED_archived_credit_only_semantics_pending'})
    runs = [e for k in VARIABLES for e in episodes(rows, k)]
    credit_runs = episodes(credit_rows, 'dfy')
    summary = {'experiment_id': cfg['experiment_id'], 'formation_months': len(rows), 'thresholds': cuts,
               'credit_asof_thresholds_percentage_points': credit_cuts,
               'counts': {k: {tag: sum(r['labels'][k] == tag for r in rows) for tag in ['低', '中', '高', '缺失']} for k in VARIABLES},
               'credit_asof_counts': {tag: sum(r['labels']['dfy'] == tag for r in credit_rows) for tag in ['低', '中', '高', '缺失']},
               'formal_eight_variable_PIT_panel_ready': False, 'limitations': cfg['limitations'],
               'source_columns_used': required, 'model_results_accessed': False}
    for name, obj in [('monthly_labels', rows), ('intervals', runs), ('credit_archived_monthly_labels', credit_rows),
                      ('credit_archived_intervals', credit_runs), ('summary', summary)]:
        (out / (name + '.json')).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# 环境划分首版：2000—2025', '', '**八指标历史回顾版，不是已认证的实时预测环境。** 作者最新历史文件可能含修订；滞后一个月不能消除修订或发布延迟。', '',
             '日期均为形成月：例如2000-01使用1999-12的观测值，对应2000-02预测；不把形成月标签解释为该月同期事件。', '',
             '校准观测期：1963-01至1999-11，共443个月。两个三分位阈值冻结用于全部2000—2025形成月，不要求后续每组占三分之一。', '',
             '低：x≤q1；中：q1<x≤q2；高：x>q2。连续同标签月份合并；缺失单列，不跨缺失合并。没有唯一综合类别。', '',
             '公式：dp=ln(D12/Index)，ep=ln(E12/Index)，bm=b/m，ntis取原列，tbl取原列，tms=lty−tbl，dfy=BAA−AAA，svar取原列。', '',
             'dp/ep为对数比；bm和ntis为比率；作者文件的利率/利差为小数（0.01=1个百分点）；svar为收益平方单位。信用历史版本表为百分点，不能与作者文件未经换算混用。', '',
             '八指标来源：[Goyal作者主页](https://sites.google.com/view/agoyal145)。lty从2022年改用FRED来源；信用AAA构成变化仍待语义审计。', '',
             '## 固定阈值与覆盖', '', '| 指标 | q1 | q2 | 低月数 | 中月数 | 高月数 | 缺失 |', '|---|---:|---:|---:|---:|---:|---:|']
    for k in VARIABLES:
        c, n = cuts[k], summary['counts'][k]
        lines.append(f'| {k} {NAMES[k]} | {c["q1"]:.8g} | {c["q2"]:.8g} | {n["低"]} | {n["中"]} | {n["高"]} | {n["缺失"]} |')
    for k in VARIABLES:
        lines += ['', f'## {k}：{NAMES[k]}全部连续区间（历史回顾版）', '', '| 起始形成月 | 结束形成月 | 状态 | 月数 |', '|---|---|---|---:|']
        for e in runs:
            if e['variable'] == k:
                lines.append(f'| {e["start"][:7]} | {e["end"][:7]} | {e["label"]} | {e["months"]} |')
    lines += ['', '## 信用利差：历史版本单独核验', '',
              '仅此维度使用各形成月的ALFRED存档值，校准阈值来自2000-01-31存档中1963-01至1999-11的观测。不是完整八维实时环境，也不是最早发布日或债券构成不变的证明。', '',
              f'q1={credit_cuts["dfy"]["q1"]:.8g}个百分点，q2={credit_cuts["dfy"]["q2"]:.8g}个百分点。', '',
              '| 起始形成月 | 结束形成月 | 状态 | 月数 |', '|---|---|---|---:|']
    for e in credit_runs:
        lines.append(f'| {e["start"][:7]} | {e["end"][:7]} | {e["label"]} | {e["months"]} |')
    lines += ['', '## 八指标逐月标签（历史回顾版）', '', '| 形成月 | 观测月 | dp | ep | bm | ntis | tbl | tms | dfy | svar |', '|---|---|---|---|---|---|---|---|---|---|']
    for r in rows:
        lines.append('| ' + ' | '.join([r['formation_month'][:7], r['observation_month'][:7]] + [r['labels'][k] for k in VARIABLES]) + ' |')
    (out / '环境划分报告.md').write_text('\n'.join(lines) + '\n')
    record = {'experiment_id': cfg['experiment_id'], 'created_at': datetime.now(timezone.utc).isoformat(),
              'config_sha256': digest(args.config), 'script_sha256': digest(__file__), 'python': sys.version,
              'openpyxl': openpyxl.__version__, 'source_url': cfg['source_url'], 'source_sha256': digest(source),
              'credit_manifest_sha256': digest(credit_manifest),
              'outputs': [{'path': p.name, 'sha256': digest(p)} for p in sorted(out.iterdir()) if p.is_file()]}
    (out / 'output_manifest.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
