#!/usr/bin/env python3
"""Frozen, paired trained/random architecture diagnostic (no model fitting)."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from paper7_geometry_core import geometry_metrics, sha256, write_json
from run_paper7_representation_geometry import FrozenTeacher, read_models


def circular_indices(rng, length, block):
    starts = rng.integers(0, length, size=int(np.ceil(length / block)))
    return np.concatenate([(s + np.arange(block)) % length for s in starts])[:length]


def interval(paired, draws, block, seed):
    """Resample seed clusters and shared time blocks; retain every paired K."""
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for i in range(draws):
        seeds = rng.integers(0, paired.shape[0], size=paired.shape[0])
        months = circular_indices(rng, paired.shape[1], block)
        estimates[i] = paired[seeds][:, months].mean()
    return np.quantile(estimates, [.025, .975]).tolist()


def save_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--run-id', required=True)
    a = p.parse_args(); start = time.time()
    if a.output_dir.exists():
        raise FileExistsError(a.output_dir)
    c = json.loads(a.config.read_text())
    for item in c['inputs']:
        if sha256(Path(item['path'])) != item['sha256']:
            raise RuntimeError('Input hash mismatch: '+item['path'])
    a.output_dir.mkdir(parents=True)
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    with np.load(c['inputs'][0]['path'], allow_pickle=False) as data:
        x = data['x'].astype(np.float32)
        months = data['target_months'].astype(int)
    assert x.shape == (c['months'], c['stocks_per_month'], 172)
    models = read_models(Path(c['inputs'][1]['path']))
    metrics = ['participation_rank', 'rank_fraction', 'top_eigenvalue_share']
    shape = (len(c['seeds']), len(months), len(c['factor_counts']), 3, 3, 3)
    trained = np.zeros(shape)
    random = np.zeros((shape[0], shape[1], 1, 3, 3, 3))
    rows = []

    def measure(model, si, ki, condition):
        target = trained if condition == 'trained' else random
        with torch.no_grad():
            for mi, month in enumerate(months):
                hidden = model.representations(torch.from_numpy(x[mi]))
                for li, (layer, value) in enumerate(hidden.items()):
                    for ni, norm in enumerate(c['normalizations']):
                        g = geometry_metrics(value.numpy(), norm, neighbor_metrics=False)
                        fraction = float(g['participation_rank']) / int(g['ambient_dimension'])
                        values = [float(g['participation_rank']), fraction, float(g['top_eigenvalue_share'])]
                        target[si, mi, ki, ni, li] = values
                        rows.append({'condition':condition, 'seed':c['seeds'][si],
                                     'factor_count':c['factor_counts'][ki] if condition=='trained' else 'shared',
                                     'target_month':str(np.datetime64(int(month),'M')),
                                     'normalization':norm, 'layer':layer,
                                     'width':g['ambient_dimension'], **dict(zip(metrics,values))})

    for si, seed in enumerate(c['seeds']):
        torch.manual_seed(seed)
        random_model = FrozenTeacher(172, c['hidden'], 1).eval()
        measure(random_model, si, 0, 'random')
        print(json.dumps({'condition':'random','seed':seed,'elapsed':round(time.time()-start,1)}),flush=True)
    for item in models:
        state = torch.load(item['path'], map_location='cpu', weights_only=False)
        assert state['input_dim'] == 172 and state['hidden'] == c['hidden']
        model = FrozenTeacher(172, state['hidden'], int(item['factor_count'])).eval()
        model.load_state_dict(state['model_state_dict'])
        measure(model,c['seeds'].index(item['seed']),c['factor_counts'].index(item['factor_count']),'trained')
        print(json.dumps({'condition':'trained','seed':item['seed'],'K':item['factor_count'],
                          'elapsed':round(time.time()-start,1)}),flush=True)
    save_csv(a.output_dir/'monthly_geometry.csv',rows)
    summaries=[]; seed_summaries=[]
    diff = trained - random
    for ni,norm in enumerate(c['normalizations']):
        for li,width in enumerate(c['hidden']):
            for ei,metric in enumerate(metrics):
                d=diff[:,:,:,ni,li,ei]
                ci=interval(d,c['inference']['draws'],c['inference']['month_block'],c['inference']['seed'])
                summaries.append({'normalization':norm,'layer':li+1,'width':width,'metric':metric,
                                  'trained_mean':float(trained[:,:,:,ni,li,ei].mean()),
                                  'random_mean':float(random[:,:,:,ni,li,ei].mean()),
                                  'trained_median':float(np.median(trained[:,:,:,ni,li,ei])),
                                  'random_median':float(np.median(random[:,:,:,ni,li,ei])),
                                  'paired_mean_difference':float(d.mean()),
                                  'paired_bootstrap_low':ci[0],'paired_bootstrap_high':ci[1]})
                for si,seed in enumerate(c['seeds']):
                    for ki,k in enumerate(c['factor_counts']):
                        seed_summaries.append({'normalization':norm,'layer':li+1,'metric':metric,
                                              'seed':seed,'factor_count':k,
                                              'paired_month_mean_difference':float(d[si,:,ki].mean())})
    save_csv(a.output_dir/'paired_summary.csv',summaries)
    save_csv(a.output_dir/'seed_summary.csv',seed_summaries)
    primary=next(s for s in summaries if s['normalization']=='raw_centered' and s['layer']==3 and s['metric']=='rank_fraction')
    report={'experiment_id':c['experiment_id'],'run_id':a.run_id,'status':'completed',
            'trained_models':30,'unique_random_hidden_networks':5,'months':len(months),
            'primary':primary,'elapsed_seconds':time.time()-start,
            'interpretation':'Paired descriptive post-publication control, conditional on architecture and archived sample. Random hidden layers are shared across K; 5 seeds limit inference. No pricing-functional endpoint retuned.'}
    write_json(a.output_dir/'quality_report.json',report)
    write_json(a.output_dir/'environment.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
                                              'platform':platform.platform(),'torch_threads':torch.get_num_threads()})
    inputs=[{'path':str(a.config),'sha256':sha256(a.config)},*c['inputs'],
            {'path':__file__,'sha256':sha256(Path(__file__))}]
    outputs=[{'path':f.name,'bytes':f.stat().st_size,'sha256':sha256(f)} for f in sorted(a.output_dir.iterdir()) if f.is_file()]
    write_json(a.output_dir/'output_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),
               'experiment_id':c['experiment_id'],'run_id':a.run_id,'inputs':inputs,'outputs':outputs,
               'checkpoint_hashes':json.loads(Path(c['inputs'][1]['path']).read_text())['checkpoints']})
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
