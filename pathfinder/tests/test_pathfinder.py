#!/usr/bin/env python

# tests.py - Unit tests for pathfinder
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

from pathfinder import decomp, utils

import numpy as np

def test_simulate_data_grid():
    data = utils.simulate_data_grid(num_domains=4, num_modalities=2)
    assert len(data) == 4
    assert len(data['Dom0']) == 2

    data = utils.simulate_data_grid(missing=[(1,0),])
    assert data['Dom1']['Mod0'] is None

def test_JointDecomp():
    data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
    from sklearn.linear_model import Ridge
    algo = decomp.JointDecomp(n_components=5, n_iter=3, dropout=-1, method=Ridge, method_kwargs={'alpha':1e3})
    A, S, err = algo.fit(data)
    assert len(A) == 3
    assert len(S) == 2
