#!/usr/bin/env python

# tests.py - Unit tests for pathfinder
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
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

def test_JointDecomp_ICA():
    data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
    data, alpha, beta = utils.DataTable_to_Lookup(data)
    from sklearn.linear_model import Ridge
    import warnings
    warnings.filterwarnings("ignore")
    algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, dropout=-1, method=Ridge, method_kwargs={'alpha':1e3}, do_ica='left')
    algo.fit(data, alpha, beta)
    assert len(algo._A) == 3
    assert len(algo._S) == 2
    algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, dropout=-1, method=Ridge, method_kwargs={'alpha':1e3}, do_ica='right')
    algo.fit(data, alpha, beta)
    assert len(algo._A) == 3
    assert len(algo._S) == 2
    algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, dropout=-1, method=Ridge, method_kwargs={'alpha':1e3}, do_ica='both')
    algo.fit(data, alpha, beta)
    assert len(algo._A) == 3
    assert len(algo._S) == 2
