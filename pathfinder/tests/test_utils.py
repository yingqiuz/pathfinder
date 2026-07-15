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

def test_simulate_data_grid():
    data = utils.simulate_data_grid(num_domains=4, num_modalities=2)
    assert len(data) == 4
    assert len(data['Dom0']) == 2

    data = utils.simulate_data_grid(missing=[(1,0),])
    assert data['Dom1']['Mod0'] is None
    assert data['Dom0']['Mod0'] is not None

    data, data_c = utils.simulate_data_grid(missing=[(1,0),], output_complete=True)
    assert (data['Dom1']['Mod0'] is None) and (data_c['Dom1']['Mod0'] is not None)
    assert (data['Dom0']['Mod0'] is not None) and (data_c['Dom0']['Mod0'] is not None)



def test_predicted_data_grid():
    data_dict = utils.simulate_data_grid(num_domains=3, num_modalities=2)
    algo = decomp.JointOuterDecomp(n_components=5, n_iter=3)
    data_list, alpha, beta = utils.DataTable_to_Lookup(data_dict)

    algo.fit(data_list, alpha, beta)

    grid = utils.predicted_data_grid(data_dict, algo._A, algo._S)
    assert grid.keys() == data_dict.keys()


def test_Lookup():
    # simulate data
    data, data_complete = utils.simulate_data_grid(num_domains=3,
                                               num_modalities=4,
                                               missing=[(0,1),(0,3),(1,3),(2,1),(2,2)],
                                               output_complete=True)



    Clist, alpha, beta = utils.DataTable_to_Lookup(data)
    data2              = utils.Lookup_to_DataTable(Clist, alpha, beta, list(data.keys()), list(data[list(data.keys())[0]].keys()))

    assert np.all(data2['Dom1']['Mod2'] == data['Dom1']['Mod2'])


def test_perform_ica_product_preservation():
    """perform_ica should preserve A @ S^T for all approach options."""
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(42)
    A = [rng.standard_normal((50, 5)) ** 3 for _ in range(3)]
    S = [rng.standard_normal((40, 5)) / 10. for _ in range(2)]

    for approach in ['left', 'right', 'both']:
        A_ica, S_ica = utils.perform_ica(A, S, approach=approach)
        assert len(A_ica) == len(A)
        assert len(S_ica) == len(S)
        # Check product preservation
        for p in range(len(A)):
            for q in range(len(S)):
                orig = A[p] @ S[q].T
                rotated = A_ica[p] @ S_ica[q].T
                diff = np.max(np.abs(orig - rotated))
                assert diff < 1e-6, (
                    f"approach='{approach}', ({p},{q}): "
                    f"product not preserved (max diff = {diff:.2e})"
                )


def test_perform_ica_return_ica():
    """perform_ica with return_ica=True returns the ICA object."""
    import warnings
    warnings.filterwarnings("ignore")

    rng = np.random.default_rng(0)
    A = [rng.standard_normal((50, 3)) ** 3 for _ in range(2)]
    S = [rng.standard_normal((40, 3)) / 10. for _ in range(2)]

    A_ica, S_ica, ica = utils.perform_ica(A, S, approach='left', return_ica=True)
    assert hasattr(ica, 'components_')
    assert ica.components_.shape == (3, 3)
