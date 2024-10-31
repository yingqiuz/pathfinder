#!/usr/bin/env python

# tests.py - Unit tests for pathfinder
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
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
