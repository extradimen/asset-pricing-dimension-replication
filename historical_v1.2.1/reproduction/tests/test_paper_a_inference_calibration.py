"""Small deterministic software checks; not scientific calibration evidence."""
import itertools
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
import torch
from paper_a_inference_calibration import (simultaneous_intervals, bootstrap_design,
                                         distances_from_latent,model_means,wilson)
from paper_a_cae_core import block_indices,pricing_distance


def test_intervals_match_frozen_v003_numpy_expression():
    rng=np.random.default_rng(6);point=rng.normal(size=6);boot=point+rng.normal(size=(199,6))
    pairs=list(itertools.combinations(range(6),2))
    diff=np.stack([boot[:,i]-boot[:,j] for i,j in pairs],axis=1)
    observed=np.array([point[i]-point[j] for i,j in pairs])
    se=np.maximum(diff.std(0,ddof=1),1e-12)
    critical=np.quantile(np.max(np.abs((diff-observed)/se),axis=1),.95)
    lo,hi,crit=simultaneous_intervals(torch.tensor(point),torch.tensor(boot))
    np.testing.assert_allclose(lo,observed-critical*se,atol=1e-12)
    np.testing.assert_allclose(hi,observed+critical*se,atol=1e-12)
    np.testing.assert_allclose(crit,critical,atol=1e-12)


def test_joint_block_and_seed_resampling_matches_v003_rng_order():
    a=np.random.default_rng(8);b=np.random.default_rng(8)
    weights,ss=bootstrap_design(a,29,11,12,5,'cpu')
    for j in range(11):
        ix=block_indices(b,29,12);block_indices(b,120,12);seeds=b.integers(0,5,5)
        np.testing.assert_allclose(weights[j],np.bincount(ix,minlength=29)/29)
        np.testing.assert_allclose(ss[j],np.bincount(seeds,minlength=5)/5)


def test_metric_matches_parent_function_and_analytic_truth():
    mu=model_means(4,[5.,5.5,6.,7.,8.,9.],5,'cpu')
    zero=torch.zeros((12,4),dtype=torch.float64)
    d=distances_from_latent(zero,mu,4.,[.5,.3,.2]).numpy()
    np.testing.assert_allclose(d,np.broadcast_to(np.array([5,5.5,6,7,8,9])[:,None],(6,5)))
    # Arbitrary moment vectors can be passed to the parent norm as one-row
    # returns and unit SDF. This tests scaling and the square-root operation.
    for k in range(6):
        np.testing.assert_allclose(d[k,0],pricing_distance(np.ones(1),mu[k,0].numpy()[None,:],np.eye(4)))


def test_count_matrix_and_mean_seed_distances_do_not_change_estimand():
    rng=np.random.default_rng(9);x=rng.normal(size=(20,12,4))
    mu=model_means(4,[5.]*6,5,'cpu')
    weights,ss=bootstrap_design(rng,20,13,12,5,'cpu')
    means=(weights@torch.tensor(x.reshape(20,-1))).reshape(13,12,4)
    fast=(distances_from_latent(means,mu,4.,[.5,.3,.2])*ss[:,None,:]).sum(-1)
    for b in range(13):
        # Explicit repeated time rows and seed indices, versus weighted counts.
        ti=np.repeat(np.arange(20),np.rint(weights[b].numpy()*20).astype(int))
        si=np.repeat(np.arange(5),np.rint(ss[b].numpy()*5).astype(int))
        slow=distances_from_latent(torch.tensor(x[ti].mean(0)),mu,4.,[.5,.3,.2])[:,si].mean(-1)
        torch.testing.assert_close(fast[b],slow)


def test_monte_carlo_uncertainty_is_not_zero_for_extreme_estimates():
    assert wilson(0,400)[1]>0
    assert wilson(400,400)[0]<1
