#!/usr/bin/env python3
"""Frozen V004 evidence runner: simulation, progress, known truth and checksum manifest."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch
from paper_a_inference_calibration import one_replication, summarize


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def js(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True); p.add_argument('--device',choices=['cuda','cpu'],required=True)
    a=p.parse_args(); c=json.loads(a.config.read_text()); a.output_dir.mkdir(parents=True,exist_ok=False)
    started=time.time(); device=torch.device(a.device)
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False
    for e in c['inputs']: assert sha(e['path'])==e['sha256'],e['path']
    js(a.output_dir/'run_start.json',{'config_sha256':sha(a.config),'started_at_unix':started,
        'pid':os.getpid(),'cwd':os.getcwd(),'device':str(device)})
    js(a.output_dir/'environment.json',{'python':sys.version,'numpy':np.__version__,
        'torch':torch.__version__,'cuda':torch.version.cuda,'platform':platform.platform(),
        'gpu':torch.cuda.get_device_name(0) if a.device=='cuda' else None,'float_dtype':'float64'})
    all_summaries=[]
    for s in c['scenarios']:
        scenario_start=time.time(); rows=[]
        with (a.output_dir/(s['id']+'.jsonl')).open('x') as f:
            for rep in range(c['outer_replications']):
                row=one_replication(c,s,rep,device);rows.append(row)
                f.write(json.dumps(row,allow_nan=False)+'\n')
                if (rep+1)%50==0:
                    f.flush();progress={'phase':'calibrating','scenario':s['id'],
                        'scenario_index':s['index'],'replications_completed':rep+1,
                        'total_scenarios':len(c['scenarios']),'elapsed_seconds':time.time()-started}
                    js(a.output_dir/'progress.json',progress);print(json.dumps(progress),flush=True)
        result=summarize(s,rows);result['elapsed_seconds']=time.time()-scenario_start
        all_summaries.append(result);js(a.output_dir/(s['id']+'-summary.json'),result)
    with (a.output_dir/'calibration_summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(all_summaries[0]));w.writeheader();w.writerows(all_summaries)
    js(a.output_dir/'calibration_summary.json',all_summaries)
    js(a.output_dir/'quality_report.json',{'status':'complete','scenarios':len(all_summaries),
        'replications_per_scenario':c['outer_replications'],
        'outer_replications':len(all_summaries)*c['outer_replications'],
        'inner_bootstrap_draws':c['bootstrap_draws'],'simultaneous_pairs':15,
        'elapsed_seconds':time.time()-started,'inputs_verified':True,
        'empirical_model_retrained':False,'sealed_period_accessed':False,
        'limitations':c['limitations']})
    files=[{'path':f.relative_to(a.output_dir).as_posix(),'bytes':f.stat().st_size,'sha256':sha(f)}
           for f in sorted(a.output_dir.rglob('*')) if f.is_file()]
    js(a.output_dir/'output_manifest.json',{'config_sha256':sha(a.config),'outputs':files})
    print(json.dumps({'status':'complete','scenarios':len(all_summaries),'seconds':time.time()-started}),flush=True)


if __name__=='__main__': main()
