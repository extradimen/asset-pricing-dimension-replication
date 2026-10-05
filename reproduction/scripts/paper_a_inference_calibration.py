"""Known-truth, conditional pricing-moment calibration; no model fitting or data access."""
from __future__ import annotations
import itertools
import math
import numpy as np
import torch


def simultaneous_intervals(point, bootstrap):
    """Exact statistic used in V003, in float64 on CPU or CUDA.

    Columns are dimensions; both inputs already average the five seed distances.
    The bootstrap centers on the observed pair difference, uses ddof=1, and
    takes the .95 linear-interpolated quantile of the maximum standardized error.
    """
    pairs = list(itertools.combinations(range(len(point)), 2))
    i = torch.tensor([p[0] for p in pairs], device=point.device)
    j = torch.tensor([p[1] for p in pairs], device=point.device)
    observed = point[i] - point[j]
    differences = bootstrap[:, i] - bootstrap[:, j]
    se = differences.std(dim=0, unbiased=True).clamp_min(1e-12)
    critical = torch.quantile(((differences-observed)/se).abs().amax(dim=1), .95)
    return observed-critical*se, observed+critical*se, critical


def bootstrap_design(rng, n, draws, block, seeds, device):
    # Preserve V003 RNG order: development blocks, unused validation blocks,
    # then shared seed draws. This also makes fixture parity directly testable.
    counts = np.zeros((draws, n), dtype=np.float64)
    seed_counts = np.zeros((draws, seeds), dtype=np.float64)
    for b in range(draws):
        starts = rng.integers(0, n, math.ceil(n/block))
        ii = ((starts[:, None]+np.arange(block)) % n).ravel()[:n]
        rng.integers(0, 120, math.ceil(120/block))
        ss = rng.integers(0, seeds, seeds)
        counts[b] = np.bincount(ii, minlength=n)/n
        seed_counts[b] = np.bincount(ss, minlength=seeds)/seeds
    return (torch.as_tensor(counts, device=device),
            torch.as_tensor(seed_counts, device=device))


def model_means(q, distances, seeds, device):
    # Fixed orthogonal rotations preserve each population pricing distance.
    angles = torch.linspace(0., math.pi/2, len(distances), dtype=torch.float64,
                            device=device)[:, None]
    angles = angles + torch.linspace(-.15, .15, seeds, dtype=torch.float64,
                                    device=device)[None, :]
    mu = torch.zeros((len(distances), seeds, q), dtype=torch.float64, device=device)
    radius = torch.tensor(distances, dtype=torch.float64, device=device)[:, None]/math.sqrt(12.)
    mu[:, :, 0] = radius*angles.cos()
    mu[:, :, 1] = radius*angles.sin()
    return mu


def latent_series(rng, n, q, phi, k, seeds, device):
    # Independent stationary AR(1) components. One common component, K
    # dimension-specific components, and S seed-specific components.
    x = rng.standard_normal((n, 1+k+seeds, q))
    if phi:
        for t in range(1, n):
            x[t] = phi*x[t-1]+math.sqrt(1-phi*phi)*x[t]
    return torch.as_tensor(x, device=device)


def distances_from_latent(averages, mu, sigma, shares):
    k, seeds, q = mu.shape
    noise = (math.sqrt(shares[0])*averages[..., 0, :][..., None, None, :]
             + math.sqrt(shares[1])*averages[..., 1:1+k, :][..., :, None, :]
             + math.sqrt(shares[2])*averages[..., 1+k:1+k+seeds, :][..., None, :, :])
    # Fixed total noise energy: q*(sigma*sqrt(4/q))**2 = 4*sigma**2.
    moments = mu + sigma*math.sqrt(4./q)*noise
    return math.sqrt(12.)*torch.linalg.vector_norm(moments, dim=-1)


def one_replication(c, scenario, rep, device):
    rng = np.random.default_rng(np.random.SeedSequence([c['random_seed'],
                                                       scenario['index'], rep]))
    n, q, phi = scenario['months'], scenario['moment_rank'], scenario['phi']
    k, seeds = len(c['factor_counts']), c['model_seeds']
    d = [scenario['base_distance']+scenario['gap']]* (k-1)+[scenario['base_distance']]
    mu = model_means(q, d, seeds, device)
    x = latent_series(rng, n, q, phi, k, seeds, device)
    weights, seed_weights = bootstrap_design(rng, n, c['bootstrap_draws'],
                                             c['block_months'], seeds, device)
    point = distances_from_latent(x.mean(0), mu, c['noise_sigma'], c['noise_shares']).mean(-1)
    bm = (weights @ x.reshape(n, -1)).reshape(c['bootstrap_draws'], 1+k+seeds, q)
    by_seed = distances_from_latent(bm, mu, c['noise_sigma'], c['noise_shares'])
    boot = (by_seed*seed_weights[:, None, :]).sum(-1)
    lo, hi, critical = simultaneous_intervals(point, boot)
    pairs = list(itertools.combinations(range(k), 2))
    truth = torch.tensor([d[i]-d[j] for i, j in pairs], dtype=torch.float64, device=device)
    cover = (lo<=truth) & (truth<=hi)
    null = truth.abs()<1e-12
    reject = (lo>0) | (hi<0)
    correct = torch.where(truth>0, lo>0, hi<0) & ~null
    values = {
        'replication': rep, 'simultaneous_coverage': bool(cover.all()),
        'false_rejection_any_true_null': bool((reject & null).any()),
        'point_winner_K': c['factor_counts'][int(point.argmin())],
        'K8_bootstrap_winner_frequency': float((boot.argmin(1)==k-1).double().mean()),
        'critical_value': float(critical), 'mean_interval_width': float((hi-lo).mean()),
        'detect_any_true_gap': bool(correct.any()) if scenario['gap']>0 else None,
        'detect_all_five_K8_gaps': bool(correct[~null].all()) if scenario['gap']>0 else None,
        'all_K8_comparisons_cover_truth': bool(cover[[b==k-1 for a,b in pairs]].all()),
        'mean_point_bias': float((point-torch.tensor(d,device=device)).mean())}
    return values


def wilson(successes, total):
    z = 1.959963984540054
    p = successes/total; den = 1+z*z/total
    center = (p+z*z/(2*total))/den
    half = z*math.sqrt(p*(1-p)/total+z*z/(4*total*total))/den
    return [max(0., center-half), min(1., center+half)]


def summarize(scenario, rows):
    result = dict(scenario); result['replications'] = len(rows)
    for key in ['simultaneous_coverage', 'false_rejection_any_true_null',
                'detect_any_true_gap', 'detect_all_five_K8_gaps']:
        if rows[0][key] is None:
            result[key] = None; result[key+'_mc95_low'] = None; result[key+'_mc95_high'] = None
        else:
            hits = sum(r[key] for r in rows)
            result[key] = hits/len(rows)
            result[key+'_mc95_low'], result[key+'_mc95_high'] = wilson(hits, len(rows))
    for key in ['critical_value', 'mean_interval_width', 'K8_bootstrap_winner_frequency',
                'mean_point_bias']:
        result[key] = float(np.mean([r[key] for r in rows]))
    result['K8_point_selection_frequency'] = sum(r['point_winner_K']==8 for r in rows)/len(rows)
    # A diagnostic flag, not acceptance of exact calibration or an empirical power claim.
    result['material_undercoverage_flag'] = result['simultaneous_coverage_mc95_high'] < .90
    result['material_false_rejection_flag'] = result['false_rejection_any_true_null_mc95_low'] > .10
    return result
