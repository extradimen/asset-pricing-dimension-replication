"""Build audited yearly arrays and fit a fixed five-seed neural anchor on CPU."""
import argparse
from datetime import date
import json
import os
from pathlib import Path
import time
import numpy as np
import polars as pl
import torch
from torch import nn
from run_paper3_stock_anchor import sha, transform, summary
from run_paper3_q2_baseline import load_zip


class ReturnMLP(nn.Module):
    def __init__(self, inputs=214, widths=(32,32)):
        super().__init__()
        layers=[]
        for width in widths:
            layers.extend([nn.Linear(inputs,width),nn.SiLU()])
            inputs=width
        final=nn.Linear(inputs,1)
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)
        self.network=nn.Sequential(*layers,final)

    def forward(self,x):
        return self.network(x).squeeze(-1)


def weights(months):
    _,inverse,counts=np.unique(months,return_inverse=True,return_counts=True)
    w=1/counts[inverse]
    return (w/w.mean()).astype(np.float32)


def train(x,y,month,seed,cfg,out):
    torch.manual_seed(seed)
    model=ReturnMLP(x.shape[1],cfg['hidden_widths'])
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
    tx,ty,tw=map(torch.from_numpy,[x,y,weights(month)])
    generator=torch.Generator().manual_seed(seed)
    history=[]
    start=time.time()
    for epoch in range(cfg['epochs']):
        order=torch.randperm(len(x),generator=generator)
        total=0.
        for lo in range(0,len(x),cfg['batch_size']):
            idx=order[lo:lo+cfg['batch_size']]
            optimizer.zero_grad(set_to_none=True)
            loss=(tw[idx]*(model(tx[idx])-ty[idx]).square()).mean()
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),cfg['gradient_clip_norm'])
            optimizer.step()
            total+=float(loss.detach())*len(idx)
        history.append({'epoch':epoch+1,'online_training_loss':total/len(x)})
        print(f'seed {seed} epoch {epoch+1}/{cfg["epochs"]}',flush=True)
    torch.save(model.state_dict(),out/f'frozen_seed{seed}.pt')
    return model,{'seed':seed,'seconds':time.time()-start,'history':history}


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    assert sha(cfg['data_config'])==cfg['data_config_sha256']
    data=json.loads(Path(cfg['data_config']).read_text())
    root=Path(data['root'])
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    arrays=out/'arrays'; arrays.mkdir()
    torch.set_num_threads(cfg['cpu_threads'])
    torch.use_deterministic_algorithms(True)
    inputs={}
    for key in ['panel','outcomes','preprocessing','rf','labels']:
        path=Path(data[key]) if key=='labels' else root/data[key]
        inputs[str(path)]=sha(path)
        assert inputs[str(path)]==data[key+'_sha256'],key
    meta=json.loads((root/data['preprocessing']).read_text())
    names,values=load_zip(root/data['rf'],201912,factor=True)
    rf={date(m//100,m%100,1):v[names.index('RF')] for m,v in values.items()}
    labels={r['month']:{k:r[k] for k in ['A','B1','B2','C']} for r in json.loads(Path(data['labels']).read_text())}
    columns=['permno','feature_month','target_month']+meta['continuous']+meta['indicators']
    panel=pl.scan_parquet(root/data['panel']).select(columns)
    outcomes=pl.scan_parquet(root/data['outcomes']).rename({'month':'target_month'})
    counts={}
    for year in range(1963,2020):
        lo,hi=date(year,1,1),date(year,12,31)
        frame=(panel.filter(pl.col('target_month').is_between(lo,hi))
               .join(outcomes.filter(pl.col('target_month').is_between(lo,hi)),on=['permno','target_month'],how='inner',validate='1:1')
               .filter(pl.col('ret').is_finite()).sort('target_month','permno').collect())
        if not len(frame):
            continue
        assert frame.select((pl.col('feature_month').dt.offset_by('1mo')!=pl.col('target_month')).sum()).item()==0
        dates=frame['target_month'].to_list()
        x=transform(frame,meta)[:,1:].astype(np.float32)
        y=(frame['ret'].to_numpy()-np.array([rf[m] for m in dates])).astype(np.float32)
        months=np.array([m.year*100+m.month for m in dates],np.int32)
        assert x.shape[1]==214 and np.isfinite(x).all() and np.isfinite(y).all()
        for name,v in [('x',x),('y',y),('month',months),('permno',frame['permno'].to_numpy())]:
            np.save(arrays/f'{year}_{name}.npy',v,allow_pickle=False)
        counts[str(year)]=len(y)
    del frame,x,y,months
    print('yearly arrays complete',flush=True)
    pieces={k:[] for k in ['x','y','month']}
    for year in sorted(map(int,counts)):
        if year>2000: break
        months=np.load(arrays/f'{year}_month.npy')
        use=months<=200001
        for k in pieces:
            pieces[k].append(np.load(arrays/f'{year}_{k}.npy')[use])
    source={k:np.concatenate(v) for k,v in pieces.items()}
    del pieces
    assert int(source['month'].max())==200001
    models,training=[],[]
    for seed in cfg['seeds']:
        model,record=train(source['x'],source['y'],source['month'],seed,cfg,out)
        model.eval(); models.append(model); training.append(record)
    source_rows=len(source['y'])
    del source
    rows=[]
    for year in range(2000,2020):
        months=np.load(arrays/f'{year}_month.npy')
        use=months>=200002
        x=np.load(arrays/f'{year}_x.npy')[use]
        y=np.load(arrays/f'{year}_y.npy')[use]
        months=months[use]
        prediction=np.empty((len(x),len(models)),np.float32)
        with torch.no_grad():
            for lo in range(0,len(x),cfg['batch_size']):
                batch=torch.from_numpy(x[lo:lo+cfg['batch_size']])
                for i,model in enumerate(models):
                    prediction[lo:lo+len(batch),i]=model(batch).numpy()
        np.save(out/f'{year}_seed_predictions.npy',prediction)
        ensemble=prediction.mean(axis=1)
        for m in np.unique(months):
            mask=months==m
            losses={'frozen':float(np.mean((y[mask]-ensemble[mask]).astype(np.float64)**2)),
                    'zero':float(np.mean(y[mask].astype(np.float64)**2))}
            losses.update({f'seed{s}':float(np.mean((y[mask]-prediction[mask,i]).astype(np.float64)**2)) for i,s in enumerate(cfg['seeds'])})
            rows.append({'month':int(m),'stocks':int(mask.sum()),'loss':losses,'environment':labels[f'{m//100:04d}-{m%100:02d}-01']})
    report={'scope':cfg['scope'],'source_rows':source_rows,'evaluation':summary(rows,['frozen']),
            'seed_mse':{str(s):float(np.mean([r['loss'][f'seed{s}'] for r in rows])) for s in cfg['seeds']},
            'by_environment':{k:{str(s):summary([r for r in rows if r['environment'][k]==s],['frozen']) for s in [0,1]} for k in ['A','B1','B2','C']},
            'training':training,'year_rows':counts,'confirmation_period_accessed':False,'rolling_and_expanding_run':False}
    for name,obj in [('report.json',report),('monthly_losses.json',rows)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    manifest={'config_sha256':sha(args.config),'script_sha256':sha(__file__),'inputs':inputs,
              'imported_scripts':{n:sha(Path(__file__).parent/n) for n in ['run_paper3_stock_anchor.py','run_paper3_q2_baseline.py']},
              'runtime':{'torch':torch.__version__,'numpy':np.__version__,'polars':pl.__version__,'device':'cpu','threads':cfg['cpu_threads']},
              'outputs':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'output_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['evaluation'],indent=2),flush=True)


if __name__=='__main__':
    main()
