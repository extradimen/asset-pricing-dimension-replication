#!/usr/bin/env python3
"""Construct one mother array; arm membership never changes rows or ranks."""
from __future__ import annotations
import argparse
import json
import sys
import platform
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import pyarrow.parquet as pq
from run_neural_sdf_teacher import CORE92, CORE86, monthly_rf
from paper7_geometry_core import sha256, write_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();c=json.loads(a.config.read_text());out=a.output_dir
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    source=Path(c['mother_panel']);factors=Path(c['factors']);assets=Path(c['pricing_targets'])
    for item in c['input_manifest']:
        if sha256(Path(item['path']))!=item['sha256']:raise RuntimeError('Source hash mismatch')
    assert set(CORE92)-set(CORE86)==set(c['excluded_features'])
    features=['x_'+f for f in CORE92]+['missing_'+f for f in CORE92]
    rf=monthly_rf(factors)
    values={s:{key:[] for key in ['month','permno','x','y']} for s in c['splits']}
    total=0;excluded=0
    for batch in pq.ParquetFile(source).iter_batches(batch_size=50000,columns=['month','permno','ret_fwd1',*features]):
        m=batch.column('month').to_numpy(zero_copy_only=False).astype('datetime64[M]').astype(np.int32)
        pid=batch.column('permno').to_numpy(zero_copy_only=False).astype(np.int64)
        x=np.column_stack([batch.column(f).to_numpy(zero_copy_only=False).astype(np.float32) for f in features])
        y=batch.column('ret_fwd1').to_numpy(zero_copy_only=False).astype(np.float32)-np.array([rf.get(int(i+1),np.nan) for i in m],dtype=np.float32)
        valid=np.isfinite(x).all(1)&np.isfinite(y)
        total+=len(m);excluded+=int((~valid).sum())
        for name,(first,last) in c['splits'].items():
            keep=valid&(m+1>=int(np.datetime64(first,'M').astype(int)))&(m+1<=int(np.datetime64(last,'M').astype(int)))
            for key,v in [('month',m),('permno',pid),('x',x),('y',y)]:values[name][key].append(v[keep])
    table=pq.read_table(assets); am=table['month'].to_numpy(zero_copy_only=False).astype('datetime64[M]').astype(int)
    assetcols=[x for x in table.column_names if x.startswith('asset_')]
    assert len(assetcols)==74
    ar=np.column_stack([table[f].to_numpy(zero_copy_only=False).astype(np.float32) for f in assetcols]); lookup={int(m):ar[i] for i,m in enumerate(am)}
    reports={}
    for name in c['splits']:
        v={k:np.concatenate(xs) for k,xs in values.pop(name).items()}
        order=np.lexsort((v['permno'],v['month']));v={k:xs[order] for k,xs in v.items()}
        months,starts,counts=np.unique(v['month'],return_index=True,return_counts=True)
        assert not np.any((v['permno'][1:]==v['permno'][:-1])&(v['month'][1:]==v['month'][:-1]))
        targets=np.vstack([lookup[int(m)] for m in months])
        required_targets=targets[:,:25] if name=='train' else targets
        if not np.isfinite(required_targets).all():raise ValueError('Incomplete required public pricing asset coverage')
        path=out/(name+'.npz')
        np.savez_compressed(path,x=v['x'],y=v['y'],feature_months=months,target_months=months+1,
                            starts=starts,ends=starts+counts,assets=targets)
        keys=np.column_stack([v['permno'],v['month']])
        import hashlib
        reports[name]={'rows':len(v['y']),'months':len(months),'first_target':str(np.datetime64(int(months[0]+1),'M')),
                       'last_target':str(np.datetime64(int(months[-1]+1),'M')),
                       'ordered_key_sha256':hashlib.sha256(keys.tobytes()).hexdigest(),
                       'x_sha256':hashlib.sha256(v['x'].tobytes()).hexdigest(),'y_sha256':hashlib.sha256(v['y'].tobytes()).hexdigest()}
        print(json.dumps({'split':name,**reports[name]}),flush=True)
    masks=[i for i,f in enumerate(features) if f.removeprefix('x_').removeprefix('missing_') in c['excluded_features']]
    assert len(masks)==12
    write_json(out/'array_design.json',{'feature_names':features,'masked_indices':masks,'splits':reports,
               'arm_invariant':'Both arms load these exact arrays; masked86 sets only 12 input columns to zero at forward pass.'})
    write_json(out/'quality_report.json',{'experiment_id':c['experiment_id'],'phase':'data_preparation','status':'completed',
               'source_rows':total,'nonfinite_target_or_feature_rows':excluded,'splits':reports,'excluded_features':c['excluded_features'],
               'last_target_before_2020':True,'input_dim_both_arms':184,'active_characteristics':[92,86]})
    write_json(out/'environment.json',{'python':sys.version,'numpy':np.__version__,'platform':platform.platform()})
    write_json(out/'output_manifest.json',{'created_at':datetime.now(timezone.utc).isoformat(),'inputs':c['input_manifest'],
               'config_sha256':sha256(a.config),'script_sha256':sha256(Path(__file__)),
               'outputs':[{'path':f.name,'bytes':f.stat().st_size,'sha256':sha256(f)} for f in sorted(out.iterdir()) if f.is_file()]})


if __name__=='__main__':main()
