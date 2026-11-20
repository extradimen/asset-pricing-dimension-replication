"""Stock-level same-model update comparison using month sufficient statistics."""
import argparse
from datetime import date
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
import polars as pl
from threadpoolctl import threadpool_limits
from run_paper3_q2_baseline import load_zip
from analyze_paper3_q2_inference import family_results


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def transform(frame, meta):
    raw = frame.select(meta['continuous']).to_numpy().astype(np.float32)
    mu, scale = np.array(meta['training_mean'], np.float32), np.array(meta['training_scale'], np.float32)
    standardized = (np.where(np.isfinite(raw), raw, mu)-mu)/scale
    indicators = frame.select(meta['indicators']).to_numpy().astype(np.float32)
    index = {k: i for i, k in enumerate(meta['continuous'])}
    cross = np.column_stack([standardized[:, index[a]]*standardized[:, index[b]] for a,b in meta['interactions']])
    x = np.column_stack([np.ones(len(frame)), standardized, indicators, cross]).astype(np.float64)
    if not np.isfinite(x).all():
        raise ValueError('Nonfinite feature design')
    return x


def solve(grams, rhs, alpha):
    penalty = np.eye(grams.shape[-1])*alpha
    penalty[0,0] = 0
    return np.linalg.solve(np.mean(grams, axis=0)+penalty, np.mean(rhs, axis=0))


def month_index(d):
    return d.year*12+d.month


def summary(rows, arms):
    if not rows:
        return {'months': 0, 'loss': None}
    loss = {a: float(np.mean([r['loss'][a] for r in rows])) for a in arms+['zero']}
    return {'months': len(rows), 'stock_months': sum(r['stocks'] for r in rows), 'loss': loss,
            'r2_against_zero': {a: 1-loss[a]/loss['zero'] for a in arms},
            'relative_gain': {a: (loss['frozen']-loss[a])/loss['frozen'] for a in arms[1:]}}


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    root = Path(cfg['root'])
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    inputs = {}
    for key in ['panel', 'outcomes', 'preprocessing', 'rf', 'labels']:
        path = Path(cfg[key]) if key == 'labels' else root/cfg[key]
        inputs[str(path)] = sha(path)
        assert inputs[str(path)] == cfg[key+'_sha256'], key
    meta = json.loads((root/cfg['preprocessing']).read_text())
    assert len(meta['continuous'])+len(meta['indicators'])+len(meta['interactions']) == cfg['dimension']
    assert all(k.startswith(('x_', 'z_')) or k=='log_market_cap' for k in meta['continuous'])
    names, values = load_zip(root/cfg['rf'], 201912, factor=True)
    ri = names.index('RF')
    rf = {date(m//100,m%100,1): float(v[ri]) for m,v in values.items()}
    labels = {r['month']: {k:r[k] for k in ['A','B1','B2','C']} for r in json.loads(Path(cfg['labels']).read_text())}
    end = date.fromisoformat(cfg['source_target_end'])
    first = date.fromisoformat(cfg['evaluation_start'])
    final = date.fromisoformat(cfg['evaluation_end'])
    columns = ['permno','feature_month','target_month']+meta['continuous']+meta['indicators']
    panel = pl.scan_parquet(root/cfg['panel']).select(columns).filter(pl.col('target_month') <= final)
    outcomes = pl.scan_parquet(root/cfg['outcomes']).rename({'month':'target_month'}).filter(pl.col('target_month') <= final)
    def year_frame(year):
        lo, hi = date(year,1,1), date(year,12,31)
        return (panel.filter(pl.col('target_month').is_between(lo, hi))
                .join(outcomes.filter(pl.col('target_month').is_between(lo, hi)), on=['permno','target_month'], how='inner', validate='1:1')
                .filter(pl.col('ret').is_finite()).sort('target_month','permno').collect())
    stats = {}
    with threadpool_limits(limits=2):
        for year in range(1963, 2020):
            frame = year_frame(year)
            if frame.select((pl.col('feature_month').dt.offset_by('1mo') != pl.col('target_month')).sum()).item():
                raise ValueError('Feature timing mismatch')
            for part in frame.partition_by('target_month', maintain_order=True):
                m = part['target_month'][0]
                x = transform(part, meta)
                y = part['ret'].to_numpy()-rf[m]
                assert x.shape[1] == cfg['dimension']+1 and np.isfinite(y).all()
                stats[m] = (x.T@x/len(x), x.T@y/len(x), len(x))
            if year%5==0:
                print(f'month statistics through {year}; {len(stats)} months', flush=True)
        dates = sorted(stats)
        grams = np.array([stats[m][0] for m in dates])
        rhs = np.array([stats[m][1] for m in dates])
        source = np.array([m <= end for m in dates])
        assert source.sum() >= 120
        frozen = solve(grams[source], rhs[source], cfg['alpha'])
        monthly, audit, coeff = [], [], []
        for year in range(2000, 2020):
            frame = year_frame(year)
            for part in frame.partition_by('target_month', maintain_order=True):
                m = part['target_month'][0]
                if m < first:
                    continue
                history = np.array([d < m for d in dates])
                rolling = history & np.array([month_index(d) >= month_index(m)-cfg['rolling_calendar_months'] for d in dates])
                assert rolling.sum() == cfg['rolling_calendar_months']
                masks = [source, rolling, history]
                stops = [max(d for d,use in zip(dates, mask) if use) for mask in masks]
                assert all(stop < m for stop in stops)
                betas = [frozen, solve(grams[rolling], rhs[rolling], cfg['alpha']), solve(grams[history], rhs[history], cfg['alpha'])]
                x = transform(part, meta)
                y = part['ret'].to_numpy()-rf[m]
                predicted = x@np.array(betas).T
                loss = {a: float(np.mean((y-predicted[:,i])**2)) for i,a in enumerate(cfg['arms'])}
                loss['zero'] = float(np.mean(y*y))
                monthly.append({'month': m.year*100+m.month, 'stocks': len(y), 'loss': loss, 'environment': labels[str(m)]})
                audit.append({'month':str(m), 'train_stop':dict(zip(cfg['arms'],map(str,stops))), 'training_months':dict(zip(cfg['arms'],[int(v.sum()) for v in masks]))})
                coeff.append(np.array(betas))
            print(f'predictions through {year}', flush=True)
    assert len(monthly) == 239
    env = {k: np.array([r['environment'][k] for r in monthly]) for k in ['A','B1','B2','C']}
    gains = {a: np.array([r['loss']['frozen']-r['loss'][a] for r in monthly]) for a in cfg['arms'][1:]}
    report = {'scope':cfg['scope'], 'source_months':int(source.sum()), 'source_stock_months':sum(stats[d][2] for d in dates if d<=end),
              'evaluation':summary(monthly,cfg['arms']), 'minimum_stocks_per_month': min(r['stocks'] for r in monthly),
              'by_environment':{k:{str(s):summary([r for r in monthly if r['environment'][k]==s],cfg['arms']) for s in [0,1]} for k in env},
              'by_decade':{str(y):summary([r for r in monthly if y*100+1<=r['month']<=(y+9)*100+12],cfg['arms']) for y in [2000,2010]},
              'exploratory_hac12_holm10':family_results(gains,env,12),
              'hac_sensitivity':{str(lag):family_results(gains,env,lag) for lag in [6,24,60]},
              'elapsed_seconds':time.time()-started, 'confirmation_period_accessed':False}
    for name, obj in [('report.json',report),('monthly_losses.json',monthly),('temporal_audit.json',audit)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    np.save(out/'monthly_coefficients.npy',np.array(coeff))
    np.savez_compressed(out/'month_sufficient_statistics.npz', months=np.array([str(m) for m in dates]),gram=grams,rhs=rhs,counts=np.array([stats[m][2] for m in dates]))
    manifest={'config_sha256':sha(args.config),'script_sha256':sha(__file__),'inputs':inputs,
              'imported_scripts':{n:sha(Path(__file__).parent/n) for n in ['run_paper3_q2_baseline.py','analyze_paper3_q2_inference.py']},
              'runtime':{'python':sys.version,'numpy':np.__version__,'polars':pl.__version__},
              'outputs':{p.name:sha(p) for p in out.iterdir() if p.is_file()}}
    (out/'output_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['evaluation'],indent=2),flush=True)


if __name__=='__main__':
    main()
