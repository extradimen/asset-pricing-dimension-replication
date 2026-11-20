"""Prespecified analysis of P1-G4-V003. Conditional inference, not structural K recovery."""
from __future__ import annotations
import csv
import itertools
import json
from pathlib import Path
import numpy as np
from paper_a_cae_core import sdf_coefficient, pricing_weight, pricing_distance, block_indices


def js(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def cs(path, rows):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def evaluate(root, c):
    out = root/'analysis'; out.mkdir()
    arms = list(c['architectures']); ks = c['factor_counts']; seeds = c['seeds']
    data = {}
    for split in ['validation', 'development']:
        with np.load(root/'prepared'/(split+'.npz')) as z: data[split] = {key: z[key] for key in z.files}
    n = len(data['development']['months']); nv = len(data['validation']['months'])
    sdf = np.empty((len(arms), len(ks), len(seeds), n))
    vsdf = np.empty((len(arms), len(ks), len(seeds), nv))
    sse = np.empty((2, len(arms), len(ks), len(seeds), n, 3))
    metrics = []; coefficients = []
    for ai, arm in enumerate(arms):
        for ki, k in enumerate(ks):
            for si, seed in enumerate(seeds):
                folder = root/(arm+'-k'+str(k)+'-seed'+str(seed))
                with np.load(folder/'factors.npz') as f:
                    b = sdf_coefficient(f['validation'])
                    vsdf[ai, ki, si] = 1-f['validation']@b
                    sdf[ai, ki, si] = 1-f['development']@b
                coefficients.append({'arm': arm, 'K': k, 'seed': seed, 'validation_sdf_b': b.tolist()})
                rows = list(csv.DictReader((folder/'monthly_metrics.csv').open()))
                for pi, split in enumerate(['validation', 'development']):
                    sub = [r for r in rows if r['split'] == split]
                    assert [int(r['month']) for r in sub] == data[split]['months'].tolist()
                    v = np.array([[float(r[key]) for key in ['zero_sse', 'total_sse', 'predictive_sse']] for r in sub])
                    sse[pi, ai, ki, si] = v
                    metrics.append({'split': split, 'arm': arm, 'K': k, 'seed': seed,
                                    'total_R2': float(1-v[:,1].sum()/v[:,0].sum()),
                                    'predictive_R2': float(1-v[:,2].sum()/v[:,0].sum())})
    cs(out/'stock_prediction_metrics.csv', metrics); js(out/'sdf_coefficients.json', coefficients)
    families = {'size_bm_25': slice(0,25), 'industry_49': slice(25,74), 'combined_74': slice(0,74)}
    gammas = c['geometry_grid']; surface = []; benchmark_rows = []; point = {}
    bf = data['validation']['benchmark']; b = sdf_coefficient(bf)
    bsdf = 1-data['development']['benchmark']@b
    for family, sl in families.items():
        vr = data['validation']['assets'][:,sl].astype(np.float64)
        dr = data['development']['assets'][:,sl].astype(np.float64)
        assert np.isfinite(vr).all() and np.isfinite(dr).all()
        for gamma in gammas:
            w = pricing_weight(vr, gamma, c['pricing_ridge'])
            d = pricing_distance(sdf, dr, w); v = pricing_distance(vsdf, vr, w)
            point[family, gamma] = d
            benchmark_rows.append({'family': family, 'gamma': gamma,
                                   'FF5_MOM_distance': float(pricing_distance(bsdf, dr, w))})
            for ai, arm in enumerate(arms):
                for ki,k in enumerate(ks):
                    for si,seed in enumerate(seeds):
                        surface.append({'family': family, 'gamma': gamma, 'arm': arm, 'K': k, 'seed': seed,
                                        'validation_distance': float(v[ai,ki,si]), 'development_distance': float(d[ai,ki,si])})
    cs(out/'pricing_surface.csv', surface); cs(out/'benchmark_surface.csv', benchmark_rows)
    # Primary: fixed validation HJ-type weight and SDF coefficients, common time blocks,
    # common seed draws. This measures evaluation/algorithm uncertainty conditional on training.
    vr = data['validation']['assets'].astype(np.float64); dr = data['development']['assets'].astype(np.float64)
    w = pricing_weight(vr, 1., c['pricing_ridge'])
    rng = np.random.default_rng(c['inference']['seed']); draws = c['inference']['draws']
    boot = np.empty((draws,len(arms),len(ks)))
    bootstrap_benchmark = np.empty(draws)
    validation_boot = np.empty_like(boot)
    prediction_boot = np.empty((draws,len(arms),len(ks)))
    for j in range(draws):
        ii = block_indices(rng,n,c['inference']['block_months'])
        vi = block_indices(rng,nv,c['inference']['block_months'])
        ss = rng.integers(0,len(seeds),len(seeds))
        boot[j] = pricing_distance(sdf[...,ii],dr[ii],w)[:,:,ss].mean(2)
        validation_boot[j] = pricing_distance(vsdf[...,vi],vr[vi],w)[:,:,ss].mean(2)
        bootstrap_benchmark[j] = pricing_distance(bsdf[ii],dr[ii],w)
        selected_sse = np.take(np.take(sse[1],ss,axis=2),ii,axis=3).sum(3).mean(2)
        prediction_boot[j] = 1-selected_sse[...,2]/selected_sse[...,0]
    mean = point['combined_74',1.].mean(2)
    pairs = list(itertools.combinations(range(len(ks)),2)); ai = arms.index('cae')
    diff = np.stack([boot[:,ai,i]-boot[:,ai,j] for i,j in pairs],axis=1)
    observed = np.array([mean[ai,i]-mean[ai,j] for i,j in pairs])
    se = np.maximum(diff.std(0,ddof=1),1e-12)
    max_t = np.max(np.abs((diff-observed)/se),axis=1)
    critical = float(np.quantile(max_t,.95))
    pairrows = []
    for index,(i,j) in enumerate(pairs):
        low,high = observed[index]+np.array([-1,1])*critical*se[index]
        pairrows.append({'K_a': ks[i], 'K_b': ks[j], 'difference_a_minus_b': float(observed[index]),
                         'simultaneous_95_low': float(low), 'simultaneous_95_high': float(high),
                         'includes_zero': bool(low<=0<=high)})
    cs(out/'cae_pairwise_HJ_uncertainty.csv',pairrows)
    summaries = []; validation_means = []; qualification = []
    for aidx,arm in enumerate(arms):
        val_r2 = 1-sse[0,aidx,:,:,:,2].sum(2).mean(1)/sse[0,aidx,:,:,:,0].sum(2).mean(1)
        dev_r2 = 1-sse[1,aidx,:,:,:,2].sum(2).mean(1)/sse[1,aidx,:,:,:,0].sum(2).mean(1)
        chosen = int(np.argmax(val_r2))
        validation_means.append({'arm':arm,'K_chosen_by_validation_predictive_R2':ks[chosen],
                                 'validation_predictive_R2':float(val_r2[chosen]),
                                 'development_predictive_R2':float(dev_r2[chosen])})
        for ki,k in enumerate(ks):
            lo,hi = np.quantile(boot[:,aidx,ki],[.025,.975])
            plo,phi = np.quantile(prediction_boot[:,aidx,ki],[.025,.975])
            summaries.append({'arm':arm,'K':k,'HJ_distance':float(mean[aidx,ki]),
                              'HJ_pointwise_95_low':float(lo),'HJ_pointwise_95_high':float(hi),
                              'development_winner_frequency':float(np.mean(np.argmin(boot[:,aidx],axis=1)==ki)),
                              'validation_winner_frequency':float(np.mean(np.argmin(validation_boot[:,aidx],axis=1)==ki)),
                              'predictive_R2':float(dev_r2[ki]),'predictive_R2_95_low':float(plo),'predictive_R2_95_high':float(phi)})
        if arm=='cae':
            reference=float(pricing_distance(bsdf,dr,w))
            bd=boot[:,aidx,chosen]-bootstrap_benchmark
            qualification={'selection':'K chosen by validation predictive R2 only', 'chosen_K':ks[chosen],
                           'validation_predictive_R2_positive':bool(val_r2[chosen]>0),
                           'development_predictive_R2_positive':bool(dev_r2[chosen]>0),
                           'development_HJ_not_worse_than_FF5_MOM_point_estimate':bool(mean[aidx,chosen]<=reference),
                           'HJ_difference_to_FF5_MOM':float(mean[aidx,chosen]-reference),
                           'HJ_difference_pointwise_95':np.quantile(bd,[.025,.975]).tolist()}
    qualification['all_point_criteria_pass']=all(qualification[k] for k in ['validation_predictive_R2_positive','development_predictive_R2_positive','development_HJ_not_worse_than_FF5_MOM_point_estimate'])
    qualification['interpretation']='Prespecified diagnostic gate, not proof of global competitiveness or inferential noninferiority. Report failures without retuning.'
    cs(out/'dimension_summary.csv',summaries);js(out/'validation_selected_models.json',validation_means)
    js(out/'competence_gate.json',qualification)
    benchmark=float(pricing_distance(bsdf,dr,w))
    for aidx,arm in enumerate(arms):
        gap=mean[aidx]-mean[aidx].min()
        js(out/(arm+'_descriptive_tolerance_set.json'),{'tolerance':c['descriptive_tolerance']*benchmark,
           'tolerance_rule':'1% of FF5+MOM development HJ-type distance; descriptive only, not a confidence set',
           'members':[k for k,g in zip(ks,gap) if g<=c['descriptive_tolerance']*benchmark]})
    js(out/'quality_report.json',{'status':'complete','models':int(np.prod(sdf.shape[:3])),
        'candidate_fits':int(np.prod(sdf.shape[:3])*len(c['l1_grid'])),'bootstrap_draws':draws,
        'primary_simultaneous_comparisons':len(pairs),'simultaneous_max_t_critical':critical,
        'sealed_period_accessed':False,'economic_task_changed_by_gamma':True,
        'test_returns_used_for_forecast':False,'pricing_weight_estimated_on_validation_only':True,
        'sdf_coefficients_estimated_on_validation_only':True,'bootstrap_retrains_networks':False,
        'limitations':c['limitations']})
    np.savez_compressed(out/'bootstrap_primary.npz',pricing_distance=boot,validation_distance=validation_boot,
                         predictive_r2=prediction_boot,benchmark_distance=bootstrap_benchmark)
