#!/usr/bin/env python3
"""Frozen P1 conditional-autoencoder control, immutable inputs and child-wait ledger."""
from __future__ import annotations
import argparse
import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import time
import zipfile
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '2')
import numpy as np
import torch
from paper_a_cae_core import ConditionalAE, managed_portfolio, lagged_factor_mean


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def csvout(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def zip_monthly(path, columns):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.lower().endswith('.csv'))
        text = archive.read(name).decode('utf-8-sig')
    result = {}
    for row in csv.reader(io.StringIO(text)):
        date = row[0].strip() if row else ''
        if len(date) == 6 and date.isdigit() and len(row) >= columns+1:
            key = int(np.datetime64(date[:4]+'-'+date[4:], 'M').astype(int))
            if key in result: raise ValueError('duplicate monthly factor date')
            values = np.array([float(x) for x in row[1:columns+1]]) / 100
            if np.any(values <= -.99): raise ValueError('factor missing code')
            result[key] = values
    return result


def load_split(folder, name):
    with np.load(folder/(name+'.npz'), allow_pickle=False) as z:
        result = {k: z[k] for k in ['x', 'y', 'starts', 'ends', 'target_months', 'feature_months', 'assets']}
    # Use the 92 original numerical characteristics, without missing-indicator expansion.
    result['x'] = np.ascontiguousarray(result['x'][:, :92], dtype=np.float32)
    assert np.isfinite(result['x']).all() and np.isfinite(result['y']).all()
    assert np.all(result['target_months'] == result['feature_months']+1)
    assert np.all(np.diff(result['target_months']) == 1)
    assert result['starts'][0] == 0 and result['ends'][-1] == len(result['y'])
    assert np.array_equal(result['starts'][1:], result['ends'][:-1])
    assert result['target_months'][-1] < int(np.datetime64('2020-01', 'M').astype(int))
    return result


def prepare(a, c):
    out = a.output_dir/'prepared'; out.mkdir()
    for entry in c['inputs']:
        if sha(entry['path']) != entry['sha256']: raise ValueError('Input checksum: '+entry['path'])
    ff = zip_monthly(Path(c['ff5_zip']), 6)
    mom = zip_monthly(Path(c['momentum_zip']), 1)
    scale = None; xscale = None; summaries = {}
    for name in ['train', 'validation', 'development']:
        d = load_split(Path(c['data_dir']), name)
        raw = np.stack([managed_portfolio(d['x'][s:e], d['y'][s:e], c['managed_ridge'])
                        for s, e in zip(d['starts'], d['ends'])])
        if name == 'train':
            scale = float(np.sqrt(np.mean(d['y'].astype(np.float64)**2)))
            xscale = np.maximum(raw.std(axis=0), 1e-6)
        benchmark = np.stack([np.r_[ff[int(m)][:5], mom[int(m)]] for m in d['target_months']])
        np.savez_compressed(out/(name+'.npz'), managed=(raw/xscale).astype(np.float32),
                            assets=d['assets'], months=d['target_months'], benchmark=benchmark)
        summaries[name] = {'rows': len(d['y']), 'months': len(raw),
                           'first': str(np.datetime64(int(d['target_months'][0]), 'M')),
                           'last': str(np.datetime64(int(d['target_months'][-1]), 'M'))}
        print(json.dumps({'phase': 'prepared', 'split': name, **summaries[name]}), flush=True)
    js(out/'scaling.json', {'return_rms_train': scale, 'managed_std_train': xscale.tolist(), 'splits': summaries})


def tensors(a, c, name, device, scale):
    d = load_split(Path(c['data_dir']), name)
    with np.load(a.output_dir/'prepared'/(name+'.npz')) as z: managed = z['managed']
    counts = d['ends']-d['starts']
    return {**d, 'xt': torch.from_numpy(d['x']).to(device),
            'yt': torch.from_numpy((d['y']/scale).astype(np.float32)).to(device),
            'mt': torch.from_numpy(managed).to(device),
            'it': torch.from_numpy(np.repeat(np.arange(len(counts)), counts)).long().to(device)}


def reconstruction_sse(model, d):
    model.eval(); total = 0.
    with torch.no_grad():
        for start in range(0, len(d['yt']), 65536):
            sl = slice(start, start+65536)
            err = model(d['xt'][sl], d['mt'], d['it'][sl])-d['yt'][sl]
            total += float(err.square().sum().cpu())
    return total/len(d['yt'])


def monthly_output(model, d, history, scale):
    model.eval()
    with torch.no_grad(): factors = model.factor(d['mt']).cpu().numpy().astype(np.float64)
    forecasts = lagged_factor_mean(history, factors)
    rows = []
    with torch.no_grad():
        for j, (s, e) in enumerate(zip(d['starts'], d['ends'])):
            beta = model.beta(d['xt'][s:e]).cpu().numpy().astype(np.float64)
            y = d['y'][s:e].astype(np.float64)
            reconstruction = beta @ factors[j] * scale
            prediction = beta @ forecasts[j] * scale
            rows.append({'month': int(d['target_months'][j]), 'n': int(e-s),
                         'zero_sse': float(y@y), 'total_sse': float(np.sum((y-reconstruction)**2)),
                         'predictive_sse': float(np.sum((y-prediction)**2))})
    return factors, rows


def worker(a, c):
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    device = torch.device('cuda:'+str(a.worker_index))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
    scale = json.loads((a.output_dir/'prepared/scaling.json').read_text())['return_rms_train']
    train = tensors(a, c, 'train', device, scale)
    val = tensors(a, c, 'validation', device, scale)
    tasks = [(arm, k, seed) for arm in c['architectures'] for k in c['factor_counts'] for seed in c['seeds']]
    selected = []
    for ti, (arm, k, seed) in enumerate(tasks):
        if ti % a.workers != a.worker_index: continue
        started = time.time(); folder = a.output_dir/(arm+'-k'+str(k)+'-seed'+str(seed)); folder.mkdir()
        best = None; candidates = []
        for l1 in c['l1_grid']:
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
            rng = np.random.default_rng(seed)
            model = ConditionalAE(92, k, c['architectures'][arm]).set_encoder(93, k).to(device)
            initial_sha = hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            optimizer = torch.optim.Adam(model.parameters(), lr=c['learning_rate'])
            best_loss = float('inf'); state = None; stale = 0; best_epoch = 0; history = []
            for epoch in range(1, c['max_epochs']+1):
                model.train(); order = rng.permutation(len(train['starts'])); losses = []
                for offset in range(0, len(order), c['month_batch']):
                    indices = np.concatenate([np.arange(train['starts'][j], train['ends'][j])
                                              for j in order[offset:offset+c['month_batch']]])
                    ix = torch.from_numpy(indices).long().to(device)
                    pred = model(train['xt'][ix], train['mt'], train['it'][ix])
                    mse = (pred-train['yt'][ix]).square().mean()
                    penalty = sum(p.abs().sum() for p in model.parameters())
                    loss = mse + l1*penalty
                    if not torch.isfinite(loss): raise ValueError('Nonfinite training loss')
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                    optimizer.step(); losses.append(float(mse.detach().cpu()))
                score = reconstruction_sse(model, val)
                history.append({'epoch': epoch, 'train_mse_scaled': float(np.mean(losses)), 'validation_mse_scaled': score})
                if score < best_loss-1e-6:
                    best_loss = score; best_epoch = epoch; stale = 0
                    state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                else: stale += 1
                if epoch % 5 == 0:
                    print(json.dumps({'phase': 'training', 'arm': arm, 'K': k, 'seed': seed,
                                      'l1': l1, 'epoch': epoch, 'validation_mse': score}), flush=True)
                if stale >= c['patience']: break
            candidate = {'l1': l1, 'validation_mse_scaled': best_loss, 'best_epoch': best_epoch,
                         'epochs': len(history), 'initialization_sha256': initial_sha}
            candidates.append(candidate); csvout(folder/('history-l1-'+str(l1)+'.csv'), history)
            if best is None or best_loss < best['validation_mse_scaled']:
                best = {**candidate, 'state': state}
            del model, optimizer
        assert best is not None
        state = best.pop('state'); js(folder/'selection.json', {'selected': best, 'candidates': candidates, 'development_used': False})
        torch.save({'state': state, 'arm': arm, 'K': k, 'seed': seed, 'hidden': c['architectures'][arm], 'return_scale': scale}, folder/'checkpoint.pt')
        selected.append({'arm': arm, 'K': k, 'seed': seed, **best, 'elapsed_seconds': time.time()-started})
        print(json.dumps({'phase': 'model_selected', **selected[-1]}), flush=True)
    # Development evaluation begins only after this worker freezes all its models.
    dev = tensors(a, c, 'development', device, scale)
    for row in selected:
        arm, k, seed = row['arm'], row['K'], row['seed']
        folder = a.output_dir/(arm+'-k'+str(k)+'-seed'+str(seed))
        checkpoint = torch.load(folder/'checkpoint.pt', map_location='cpu')
        model = ConditionalAE(92, k, c['architectures'][arm]).set_encoder(93, k).to(device)
        model.load_state_dict(checkpoint['state']); model.eval()
        with torch.no_grad(): tf = model.factor(train['mt']).cpu().numpy().astype(np.float64)
        vf, vm = monthly_output(model, val, tf, scale)
        df, dm = monthly_output(model, dev, np.vstack([tf, vf]), scale)
        np.savez_compressed(folder/'factors.npz', train=tf, validation=vf, development=df)
        csvout(folder/'monthly_metrics.csv', [{'split': 'validation', **r} for r in vm]+[{'split': 'development', **r} for r in dm])
        js(folder/'quality.json', {'status': 'complete', **row, 'forecast_rule': 'expanding mean of strictly earlier factors; fixed model',
                                  'total_R2_uses_contemporaneous_factors': True, 'sealed_period_accessed': False})
        print(json.dumps({'phase': 'evaluated', 'arm': arm, 'K': k, 'seed': seed}), flush=True)
    js(a.output_dir/('worker-'+str(a.worker_index)+'.json'), selected)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True); p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=2); p.add_argument('--worker-index', type=int)
    a = p.parse_args(); c = json.loads(a.config.read_text())
    if a.worker_index is not None: worker(a, c); return
    a.output_dir.mkdir(parents=True, exist_ok=False)
    js(a.output_dir/'run_start.json', {'config_sha256': sha(a.config), 'pid': os.getpid(), 'cwd': os.getcwd(), 'started_at_unix': time.time()})
    prepare(a, c)
    children = []; streams = []
    for index in range(a.workers):
        stream = (a.output_dir/('worker-'+str(index)+'.log')).open('w'); streams.append(stream)
        cmd = [sys.executable, __file__, '--config', str(a.config), '--output-dir', str(a.output_dir),
               '--workers', str(a.workers), '--worker-index', str(index)]
        children.append(subprocess.Popen(cmd, stdout=stream, stderr=subprocess.STDOUT))
    codes = [child.wait() for child in children]
    for stream in streams: stream.close()
    js(a.output_dir/'worker_exit_status.json', {'return_codes': codes, 'evidence': 'subprocess child wait'})
    if any(codes): raise RuntimeError('Worker failed; preserve run, inspect child waits and logs')
    rows = sum([json.loads((a.output_dir/('worker-'+str(i)+'.json')).read_text()) for i in range(a.workers)], [])
    assert len(rows) == len(c['architectures'])*len(c['factor_counts'])*len(c['seeds'])
    csvout(a.output_dir/'selected_models.csv', rows)
    js(a.output_dir/'environment.json', {'python': sys.version, 'torch': torch.__version__, 'numpy': np.__version__,
                                        'cuda': torch.version.cuda, 'platform': platform.platform()})
    # Analysis is frozen in this same snapshot; no adaptive decision between training and analysis.
    from evaluate_paper_a_cae_control import evaluate
    evaluate(a.output_dir, c)
    files = [{'path': f.relative_to(a.output_dir).as_posix(), 'sha256': sha(f), 'bytes': f.stat().st_size}
             for f in sorted(a.output_dir.rglob('*')) if f.is_file()]
    js(a.output_dir/'output_manifest.json', {'config_sha256': sha(a.config), 'outputs': files})
    print(json.dumps({'status': 'completed', 'selected_models': len(rows), 'outputs': len(files)}), flush=True)


if __name__ == '__main__': main()
