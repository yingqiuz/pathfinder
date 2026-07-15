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

def test_simulate_JointSVD():
    Clist, alpha, beta = utils.simulate_JointSVD(K=3, num_U=2, num_V=2, rank=3, SNR = 50)
    assert len(alpha) == 3
    assert len(beta) == 3
    assert len(Clist) == 3
    jsvd = decomp.JointSVD(3, n_iter=10)
    jsvd.fit(Clist, alpha, beta)
    assert len(jsvd.predict())==3

def test_JointSVD():
    # Test with DataDict
    data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
    algo = decomp.JointSVD(n_components=5, n_iter=3)
    algo.fit(data)
    assert len(algo._Ulist) == 3
    assert len(algo._Vlist) == 2
    assert len(algo.decomp(0))==3
    assert type(algo.predict(as_dict=True)) == dict

    # Test with DataList
    data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
    data, alpha, beta = utils.DataTable_to_Lookup(data)
    algo = decomp.JointSVD(n_components=5, n_iter=3)
    algo.fit(data, alpha, beta)
    assert len(algo._Ulist) == 3
    assert len(algo._Vlist) == 2
    assert len(algo.predict()) == len(data)

    # test with minibatch updates
    algo = decomp.JointSVD(n_components=5, n_iter=3, batch_size=3)
    algo.fit(data, alpha, beta)
    assert len(algo._Ulist) == 3
    assert len(algo._Vlist) == 2
    assert len(algo.predict()) == len(data)


def test_JointSVD_init_type():
    """SVD init (full-batch and Gram/minibatch) yields orthonormal U/V and fits;
    invalid init_type raises."""
    import pytest
    Clist, alpha, beta = utils.simulate_JointSVD(K=6, num_U=2, num_V=3, rank=4, SNR=50)

    for kwargs in [dict(init_type='svd'),               # full-batch concat SVD
                   dict(init_type='svd', batch_size=16),  # streamed Gram accum
                   dict(init_type='random')]:
        algo = decomp.JointSVD(n_components=4, n_iter=5, verbose=False, **kwargs)
        algo.fit(Clist, alpha, beta)
        assert len(algo._Ulist) == 2 and len(algo._Vlist) == 3
        assert len(algo.predict()) == len(Clist)
        for U in algo._Ulist:
            assert np.allclose(U.T @ U, np.eye(4), atol=1e-8), 'U not orthonormal'
        for V in algo._Vlist:
            assert np.allclose(V.T @ V, np.eye(4), atol=1e-8), 'V not orthonormal'

    with pytest.raises(ValueError):
        decomp.JointSVD(n_components=4, init_type='bogus')

