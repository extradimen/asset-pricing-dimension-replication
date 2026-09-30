#!/usr/bin/env python3
"""Fixed paired six-feature ablation; one long-lived worker per allocated GPU."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from torch import nn
from paper_a_control_core import MultiTeacher, month_factor_returns, hj_span_loss, factor_span_sharpe, factor_diversity_loss


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2)+'\n')


def write_csv(path, rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def factors_for(model, split, x, y, selection):
    counts=[int(split['ends'][i]-split['starts'][i]) for i in selection]
    xb=torch.cat([x[int(split['starts'][i]):int(split['ends'][i])] for i in selection])
    yb=torch.cat([y[int(split['starts'][i]):int(split['ends'][i])] for i in selection])
    scores=model(xb);rows=[];hhis=[];cursor=0
    for count in counts:
        f,h=month_factor_returns(scores[cursor:cursor+count],yb[cursor:cursor+count])
        rows.append(f);hhis.append(h);cursor+=count
    return torch.stack(rows),torch.stack(hhis)


def evaluate(model, split, x, y, assets, ridge):
    # Evaluate month by month, preserving the archived validation selection definition.
    model.eval();factors=[];hhis=[]
    with torch.no_grad():
        for start,end in zip(split['starts'],split['ends']):
            f,h=month_factor_returns(model(x[int(start):int(end)]),y[int(start):int(end)])
            factors.append(f);hhis.append(h)
        factors=torch.stack(factors)
        loss,_=hj_span_loss(factors,assets[:,:25],ridge)
        sharpe=factor_span_sharpe(factors,ridge)
        metrics={'normalized_hj_span_loss':float(loss.cpu()),'factor_span_sharpe':float(sharpe.cpu()),
                 'mean_scaled_hhi':float(torch.stack(hhis).mean().cpu())}
    return metrics,factors.cpu().numpy()


def worker(a,c):
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    torch.use_deterministic_algorithms(True)
    device=torch.device('cuda:'+str(a.worker_index))
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    design=json.loads((a.data_dir/'array_design.json').read_text())
    dat={};tensors={};assets={}
    for name in c['splits']:
        with np.load(a.data_dir/(name+'.npz'),allow_pickle=False) as z:
            dat[name]={k:z[k] for k in ['starts','ends','feature_months','target_months']}
            x=z['x'];y=z['y'];ar=z['assets']
        assert x.shape[1]==184 and np.isfinite(x).all() and np.isfinite(y).all()
        tensors[name]=(torch.from_numpy(x).to(device),torch.from_numpy(y).to(device))
        assets[name]=torch.from_numpy(ar).to(device)
    cfg=c['training'];tasks=[(k,s) for k in c['factor_counts'] for s in c['seeds']]
    mask=torch.ones(184,device=device);mask[design['masked_indices']]=0
    summaries=[]
    for task_index,(k,seed) in enumerate(tasks):
        if task_index%a.workers!=a.worker_index:continue
        initialization=None
        for arm in c['arms']:
            started=time.time();out=a.output_dir/(arm+'-k'+str(k)+'-seed'+str(seed))
            out.mkdir(exist_ok=False)
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);np.random.seed(seed)
            model=MultiTeacher(184,c['hidden'],k).to(device)
            initial_hash=hashlib.sha256(b''.join(v.detach().cpu().numpy().tobytes() for v in model.state_dict().values())).hexdigest()
            if initialization is None:initialization=initial_hash
            assert initial_hash==initialization
            xs={name:xy[0] if arm=='full92' else xy[0]*mask for name,xy in tensors.items()}
            optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            rng=np.random.default_rng(seed);best=-float('inf');best_epoch=0;state=None;stale=0;history=[]
            for epoch in range(1,cfg['epochs']+1):
                model.train();order=rng.permutation(len(dat['train']['starts']));losses=[]
                for offset in range(0,len(order),cfg['month_batch_size']):
                    selection=order[offset:offset+cfg['month_batch_size']]
                    f,h=factors_for(model,dat['train'],xs['train'],tensors['train'][1],selection)
                    hj,_=hj_span_loss(f,assets['train'][selection,:25],cfg['pricing_ridge_multiplier'])
                    loss=cfg['hj_penalty']*hj-cfg['span_sharpe_reward']*factor_span_sharpe(f,cfg['pricing_ridge_multiplier'])+cfg['diversity_penalty']*factor_diversity_loss(f)+cfg['concentration_penalty']*h.mean()
                    optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.0);optimizer.step()
                    losses.append(float(loss.detach().cpu()))
                val,_=evaluate(model,dat['validation'],xs['validation'],tensors['validation'][1],assets['validation'],cfg['pricing_ridge_multiplier'])
                score=cfg['span_sharpe_reward']*val['factor_span_sharpe']-cfg['hj_penalty']*val['normalized_hj_span_loss']
                history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'selection_score':score,**val})
                if score>best+1e-4:
                    best=score;best_epoch=epoch;state={key:v.detach().cpu().clone() for key,v in model.state_dict().items()};stale=0
                else:stale+=1
                if stale>=cfg['patience']:break
            if state is None:raise RuntimeError('No finite selected state')
            model.load_state_dict(state)
            torch.save({'model_state_dict':state,'input_dim':184,'hidden':c['hidden'],'factor_count':k,
                        'seed':seed,'arm':arm,'masked_indices':design['masked_indices'] if arm=='masked86' else []},out/'checkpoint.pt')
            performance={};monthly=[]
            for name in c['splits']:
                met,fs=evaluate(model,dat[name],xs[name],tensors[name][1],assets[name],cfg['pricing_ridge_multiplier'])
                performance[name]=met
                for mi,row in enumerate(fs):
                    monthly.append({'split':name,'feature_month':str(np.datetime64(int(dat[name]['feature_months'][mi]),'M')),
                                    'target_month':str(np.datetime64(int(dat[name]['target_months'][mi]),'M')),
                                    **{'factor_'+str(i):float(v) for i,v in enumerate(row)}})
            write_csv(out/'training_history.csv',history);write_csv(out/'monthly_factor_returns.csv',monthly)
            report={'experiment_id':c['experiment_id'],'arm':arm,'factor_count':k,'seed':seed,'epochs':len(history),
                    'best_epoch':best_epoch,'initialization_sha256':initial_hash,'performance':performance,
                    'elapsed_seconds':time.time()-started,'status':'completed'}
            write_json(out/'quality_report.json',report);summaries.append(report)
            print(json.dumps(report),flush=True)
            del xs,model,optimizer,state
            torch.cuda.empty_cache()
    write_json(a.output_dir/('worker-'+str(a.worker_index)+'.json'),summaries)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--data-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--workers',type=int,default=2)
    p.add_argument('--worker-index',type=int,default=None)
    a=p.parse_args();c=json.loads(a.config.read_text())
    manifest=json.loads((a.data_dir/'output_manifest.json').read_text())
    for item in manifest['outputs']:
        if sha(a.data_dir/item['path'])!=item['sha256']:raise RuntimeError('Array input checksum mismatch')
    if a.worker_index is not None:
        worker(a,c);return 0
    if a.output_dir.exists():raise FileExistsError(a.output_dir)
    a.output_dir.mkdir(parents=True)
    streams=[];children=[];returns=[]
    for index in range(a.workers):
        stream=(a.output_dir/('worker-'+str(index)+'.log')).open('w');streams.append(stream)
        command=[sys.executable,__file__,'--config',str(a.config),'--data-dir',str(a.data_dir),
                 '--output-dir',str(a.output_dir),'--workers',str(a.workers),'--worker-index',str(index)]
        children.append(subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT))
    for child in children:returns.append(child.wait())
    for stream in streams:stream.close()
    write_json(a.output_dir/'worker_exit_status.json',{'return_codes':returns,'evidence':'subprocess child wait'})
    if any(returns):raise RuntimeError('Worker failure; see frozen logs. Do not overwrite this run.')
    summaries=[]
    for i in range(a.workers):summaries.extend(json.loads((a.output_dir/('worker-'+str(i)+'.json')).read_text()))
    assert len(summaries)==60
    write_json(a.output_dir/'quality_report.json',{'experiment_id':c['experiment_id'],'status':'completed',
               'models':len(summaries),'arms':c['arms'],'models_summary':summaries})
    write_json(a.output_dir/'environment.json',{'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,
               'platform':platform.platform(),'cuda':torch.version.cuda,'gpus':[torch.cuda.get_device_name(i) for i in range(a.workers)]})
    outputs=[{'path':f.relative_to(a.output_dir).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)} for f in sorted(a.output_dir.rglob('*')) if f.is_file()]
    write_json(a.output_dir/'output_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),
               'config_sha256':sha(a.config),'array_manifest_sha256':sha(a.data_dir/'output_manifest.json'),
               'code':[{'path':__file__,'sha256':sha(__file__)},{'path':'scripts/paper_a_control_core.py','sha256':sha('scripts/paper_a_control_core.py')}],
               'outputs':outputs})
    print(json.dumps({'status':'completed','models':60}),flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
