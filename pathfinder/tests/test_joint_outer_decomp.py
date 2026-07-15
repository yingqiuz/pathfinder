#!/usr/bin/env python

# tests.py - Unit tests for pathfinder
#
# Author: 
# Akina Ying-Qiu Zheng <ying-qiu.zheng@ndcn.ox.ac.uk>
# Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT


import numpy as np

from pathfinder import decomp, utils



def test_JointDecomp():
    data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
    data, alpha, beta = utils.DataTable_to_Lookup(data)
    from sklearn.linear_model import Ridge
    algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, dropout=-1, method=Ridge, method_kwargs={'alpha':1e3})
    algo.fit(data, alpha, beta)
    assert len(algo._A) == 3
    assert len(algo._S) == 2
    assert len(algo.predict())==len(data)
    # test with minibatch updates
    algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, dropout=-1, method=Ridge, method_kwargs={'alpha':1e3}, batch_size=3)
    algo.fit(data, alpha, beta)
    assert len(algo._A) == 3
    assert len(algo._S) == 2
    assert len(algo.decomp(0))==2
    # Test with DataDict
    data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
    from sklearn.linear_model import Ridge
    algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, dropout=-1, method=Ridge, method_kwargs={'alpha':1e3})
    algo.fit(data)
    assert len(algo._A) == 3
    assert len(algo._S) == 2
    assert len(algo.decomp(0))==2
    assert type(algo.predict(as_dict=True)) == dict
    # test random updates
    algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, dropout=-1, method=Ridge, method_kwargs={'alpha':1e3}, update_fraction=0.5)
    algo.fit(data)
    assert len(algo._A) == 3
    assert len(algo._S) == 2
    assert len(algo.decomp(0))==2
    assert type(algo.predict(as_dict=True)) == dict

def test_svd_init_gram_matches_full():
    """Minibatch Gram-accumulated 'svd' init spans the same subspace as the
    full-concatenation 'svd' init.

    With noise-free rank-r data the top-r singular subspace is exact, so the
    two inits must agree up to an in-subspace rotation. We compare the
    orthogonal projectors (rotation-invariant) and check orthonormality.
    """
    rng = np.random.default_rng(0)
    r = 3
    A_true = [rng.standard_normal((40, r)), rng.standard_normal((55, r))]
    S_true = [rng.standard_normal((30, r)), rng.standard_normal((35, r))]
    obs = [(0, 0), (0, 1), (1, 0), (1, 1)]  # full 2x2 grid
    Clist = [A_true[i] @ S_true[j].T for (i, j) in obs]
    alpha = [i for (i, j) in obs]
    beta = [j for (i, j) in obs]

    # n_iter=0 -> factors are left at their initialised values
    full = decomp.JointOuterDecomp(
        n_components=r, n_iter=0, init_type='svd', verbose=False)
    full.fit(Clist, alpha, beta)
    gram = decomp.JointOuterDecomp(
        n_components=r, n_iter=0, init_type='svd', batch_size=8, verbose=False)
    gram.fit(Clist, alpha, beta)

    for p in range(len(A_true)):
        Ag = gram._A[p]
        assert Ag.shape == (A_true[p].shape[0], r)
        assert np.allclose(Ag.T @ Ag, np.eye(r), atol=1e-8), 'A cols not orthonormal'
        # projectors onto the two initialised subspaces must coincide
        assert np.allclose(full._A[p] @ full._A[p].T, Ag @ Ag.T, atol=1e-6)
    for q in range(len(S_true)):
        Sg = gram._S[q]
        assert Sg.shape == (S_true[q].shape[0], r)
        assert np.allclose(Sg.T @ Sg, np.eye(r), atol=1e-8), 'S cols not orthonormal'
        assert np.allclose(full._S[q] @ full._S[q].T, Sg @ Sg.T, atol=1e-6)


def test_JointDecomp_ICA():
    """Test that ICA post-processing runs for all approach options."""
    data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
    data, alpha, beta = utils.DataTable_to_Lookup(data)
    from sklearn.linear_model import Ridge
    import warnings
    warnings.filterwarnings("ignore")
    for approach in ['left', 'right', 'both']:
        algo = decomp.JointOuterDecomp(
            n_components=5, n_iter=3, dropout=-1,
            method=Ridge, method_kwargs={'alpha': 1e3}, do_ica=approach
        )
        algo.fit(data, alpha, beta)
        assert len(algo._A) == 3
        assert len(algo._S) == 2


def test_JointDecomp_ICA_product_preservation():
    """Test that ICA post-processing preserves A @ S^T products."""
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(42)
    n_rows = [60, 80]
    n_cols = [50, 70]
    rank = 5

    # Bimodal ground truth (strongly non-Gaussian)
    def make_bimodal(n, r, rng):
        signs = rng.choice([-1, 1], size=(n, r))
        return signs * (2.0 + rng.standard_normal((n, r)) * 0.3)

    A_true = [make_bimodal(nr, rank, rng) for nr in n_rows]
    S_true = [make_bimodal(nc, rank, rng) for nc in n_cols]

    obs_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    X_noisy = {}
    for i in range(2):
        for j in range(2):
            C = A_true[i] @ S_true[j].T
            sig = np.linalg.norm(C, 'fro')
            noise = rng.standard_normal(C.shape)
            noise = noise / np.linalg.norm(noise, 'fro') * sig * 0.01
            X_noisy[(i, j)] = C + noise

    Clist = [X_noisy[p] for p in obs_pairs]
    alpha_idx = [p[0] for p in obs_pairs]
    beta_idx = [p[1] for p in obs_pairs]

    for approach in ['left', 'right', 'both']:
        # Fit without ICA first to get baseline predictions
        np.random.seed(0)
        algo_no_ica = decomp.JointOuterDecomp(
            n_components=rank, n_iter=100, alpha=1e-5,
            batch_size=10, init_type='random', verbose=False
        )
        algo_no_ica.fit(Clist, alpha_idx, beta_idx)
        preds_no_ica = algo_no_ica.predict()

        # Fit with ICA
        np.random.seed(0)
        algo_ica = decomp.JointOuterDecomp(
            n_components=rank, n_iter=100, alpha=1e-5,
            batch_size=10, init_type='random', do_ica=approach, verbose=False
        )
        algo_ica.fit(Clist, alpha_idx, beta_idx)
        preds_ica = algo_ica.predict()

        # Products should be (nearly) identical
        for k in range(len(Clist)):
            diff = np.max(np.abs(preds_no_ica[k] - preds_ica[k]))
            assert diff < 1e-6, (
                f"approach='{approach}', matrix {k}: "
                f"product not preserved (max diff = {diff:.2e})"
            )


def test_JointDecomp_ICA_factor_recovery():
    """Test that ICA recovers non-Gaussian ground truth factors."""
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(123)
    n_rows = [80, 100]
    n_cols = [60, 90]
    rank = 5
    n_restarts = 10

    # Bimodal ground truth
    def make_bimodal(n, r, rng):
        signs = rng.choice([-1, 1], size=(n, r))
        return signs * (2.0 + rng.standard_normal((n, r)) * 0.3)

    A_true = [make_bimodal(nr, rank, rng) for nr in n_rows]
    S_true = [make_bimodal(nc, rank, rng) for nc in n_cols]

    obs_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    X_noisy = {}
    for i in range(2):
        for j in range(2):
            C = A_true[i] @ S_true[j].T
            sig = np.linalg.norm(C, 'fro')
            noise = rng.standard_normal(C.shape)
            noise = noise / np.linalg.norm(noise, 'fro') * sig * 0.01
            X_noisy[(i, j)] = C + noise

    Clist = [X_noisy[p] for p in obs_pairs]
    alpha_idx = [p[0] for p in obs_pairs]
    beta_idx = [p[1] for p in obs_pairs]

    # Multi-restart to find good solution
    best_loss = np.inf
    best_decomp = None
    for restart in range(n_restarts):
        np.random.seed(int(rng.integers(1e9)) + restart)
        d = decomp.JointOuterDecomp(
            n_components=rank, n_iter=200, alpha=1e-5,
            batch_size=10, init_type='random', do_ica='left', verbose=False
        )
        d.fit(Clist, alpha_idx, beta_idx)
        final_loss = np.mean(d._loss[-1])
        if final_loss < best_loss:
            best_loss = final_loss
            best_decomp = d

    # Match recovered factors to ground truth by max |correlation|
    def match_and_correlate(true, rec):
        nc = true.shape[1]
        corr = np.zeros((nc, nc))
        for i in range(nc):
            for j in range(nc):
                corr[i, j] = np.corrcoef(true[:, i], rec[:, j])[0, 1]
        abs_corr = np.abs(corr)
        used = set()
        matched_corrs = []
        for i in range(nc):
            best_j, best_val = -1, -1
            for j in range(nc):
                if j not in used and abs_corr[i, j] > best_val:
                    best_val = abs_corr[i, j]
                    best_j = j
            matched_corrs.append(best_val)
            used.add(best_j)
        return matched_corrs

    # Check A factors
    for p in range(2):
        corrs = match_and_correlate(A_true[p], best_decomp._A[p])
        median_corr = np.median(corrs)
        assert median_corr > 0.9, (
            f"A{p} factor recovery too low: median |r| = {median_corr:.3f}"
        )

    # Check S factors
    for q in range(2):
        corrs = match_and_correlate(S_true[q], best_decomp._S[q])
        median_corr = np.median(corrs)
        assert median_corr > 0.9, (
            f"S{q} factor recovery too low: median |r| = {median_corr:.3f}"
        )
