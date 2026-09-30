"""Historical-only monthly refits of the fixed five-seed MLP; checkpoint each month."""
import argparse
import contextlib
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
from run_paper3_stock_anchor import sha, summary
from run_paper3_neural_frozen import train
from analyze_paper3_q2_inference import family_results


def training_bounds(months,target,arm):
    ordinal=(months//100)*12+(months%100)
    target_ordinal=(target//100)*12+(target%100)
    stop=int(np.searchsorted(ordinal,target_ordinal,side='left'))
    start=0 if arm=='expanding' else int(np.searchsorted(ordinal,target_ordinal-60,side='left'))
    if arm not in ['expanding','rolling60']: raise ValueError(arm)
    if stop<=start or np.any(months[start:stop]>=target): raise ValueError('Invalid historical training slice')
    if arm=='rolling60' and len(np.unique(ordinal[start:stop]))!=60: raise ValueError('Incomplete rolling calendar support')
    return start,stop


def main():
    os.chdir(Path(__file__).resolve().parents[1])
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text())
    source=Path(cfg['frozen_dir']); out=Path(args.output)
    assert sha(source/'output_manifest.json')==cfg['frozen_manifest_sha256']
    assert sha(cfg['training_config'])==cfg['training_config_sha256']
    settings=json.loads(Path(cfg['training_config']).read_text())
    manifest=json.loads((source/'output_manifest.json').read_text())
    # Verify all immutable parent outputs, including arrays and frozen predictions.
    for name,digest in manifest['outputs'].items():
        assert sha(source/name)==digest,name
    for name,digest in manifest['imported_scripts'].items():
        assert sha(Path(__file__).parent/name)==digest,name
    assert sha(Path(__file__).parent/'run_paper3_neural_frozen.py')==manifest['script_sha256']
    frozen_rows=json.loads((source/'monthly_losses.json').read_text())
    assert len(frozen_rows)==cfg['months']
    years=sorted(int(p.name.split('_')[0]) for p in (source/'arrays').glob('*_month.npy'))
    n=sum(len(np.load(source/'arrays'/f'{year}_month.npy',mmap_mode='r')) for year in years)
    x=np.empty((n,214),np.float32); y=np.empty(n,np.float32); months=np.empty(n,np.int32)
    offset=0
    for year in years:
        ym=np.load(source/'arrays'/f'{year}_month.npy'); stop=offset+len(ym)
        x[offset:stop]=np.load(source/'arrays'/f'{year}_x.npy')
        y[offset:stop]=np.load(source/'arrays'/f'{year}_y.npy')
        months[offset:stop]=ym; offset=stop
    assert np.all(months[1:]>=months[:-1]) and np.max(months)<=201912
    out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(settings['cpu_threads']); torch.use_deterministic_algorithms(True)
    rows=[]; started=time.time()
    for frozen_row in frozen_rows:
        target=frozen_row['month']
        lo,hi=int(np.searchsorted(months,target,'left')),int(np.searchsorted(months,target,'right'))
        assert hi-lo==frozen_row['stocks']
        xm,ym=x[lo:hi],y[lo:hi]
        record={'month':target,'stocks':hi-lo,'environment':frozen_row['environment'],
                'loss':{'frozen':frozen_row['loss']['frozen'],'zero':frozen_row['loss']['zero']},'training':{}}
        folder=out/str(target); folder.mkdir()
        for arm in cfg['arms']:
            arm_dir=folder/arm; arm_dir.mkdir()
            start,stop=training_bounds(months,target,arm)
            predicted=np.empty((hi-lo,len(settings['seeds'])),np.float32)
            histories=[]
            for i,seed in enumerate(settings['seeds']):
                with (arm_dir/f'training_seed{seed}.log').open('w') as log, contextlib.redirect_stdout(log):
                    model,history=train(x[start:stop],y[start:stop],months[start:stop],seed,settings,arm_dir)
                model.eval()
                with torch.no_grad():
                    for p in range(0,len(xm),settings['batch_size']):
                        batch=torch.from_numpy(xm[p:p+settings['batch_size']])
                        predicted[p:p+len(batch),i]=model(batch).numpy()
                histories.append(history)
                del model
            assert np.isfinite(predicted).all()
            # The very first expanding fit must exactly reproduce the frozen ensemble.
            if target==200002 and arm=='expanding':
                original=np.load(source/'2000_seed_predictions.npy')[:hi-lo]
                np.testing.assert_allclose(predicted,original,rtol=0,atol=1e-7)
            np.save(arm_dir/'seed_predictions.npy',predicted)
            ensemble=predicted.mean(axis=1)
            record['loss'][arm]=float(np.mean((ym-ensemble).astype(np.float64)**2))
            record['training'][arm]={'first_target':int(months[start]),'last_target':int(months[stop-1]),
                                     'months':int(len(np.unique(months[start:stop]))),'rows':stop-start,
                                     'seeds':histories,'seed_losses':{str(seed):float(np.mean((ym-predicted[:,i]).astype(np.float64)**2)) for i,seed in enumerate(settings['seeds'])}}
            (arm_dir/'training.json').write_text(json.dumps(record['training'][arm],indent=2)+'\n')
        rows.append(record)
        (folder/'month_result.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
        # Progress is operational, not the immutable final report; no model selection.
        (out/'progress.json').write_text(json.dumps({'completed_months':len(rows),'total_months':cfg['months'],
            'last_completed_target':target,'elapsed_seconds':time.time()-started,'updated_at':datetime.now(timezone.utc).isoformat()},indent=2)+'\n')
        print(f'completed {target}: {len(rows)}/{cfg["months"]}',flush=True)
    environments={k:np.array([r['environment'][k] for r in rows]) for k in ['A','B1','B2','C']}
    gains={a:np.array([r['loss']['frozen']-r['loss'][a] for r in rows]) for a in cfg['arms']}
    arms=['frozen']+cfg['arms']
    report={'scope':cfg['inference'],'evaluation':summary(rows,arms),
            'by_environment':{k:{str(s):summary([r for r in rows if r['environment'][k]==s],arms) for s in [0,1]} for k in environments},
            'hac12_holm10':family_results(gains,environments,12),
            'hac_sensitivity':{str(lag):family_results(gains,environments,lag) for lag in [6,24,60]},
            'elapsed_seconds':time.time()-started,'fits':len(rows)*len(cfg['arms'])*len(settings['seeds'])}
    for name,obj in [('report.json',report),('monthly_losses.json',rows)]:
        (out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    final={'config_sha256':sha(args.config),'script_sha256':sha(__file__),
           'parent_manifest_sha256':cfg['frozen_manifest_sha256'],
           'runtime':{'torch':torch.__version__,'numpy':np.__version__,'device':'cpu','threads':settings['cpu_threads']},
           'imported_scripts':{name:sha(Path(__file__).parent/name) for name in ['run_paper3_neural_frozen.py','run_paper3_stock_anchor.py','analyze_paper3_q2_inference.py']},
           'outputs':{str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}}
    (out/'output_manifest.json').write_text(json.dumps(final,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__': main()
