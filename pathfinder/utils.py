#!/usr/bin/env python

# decomp.py - Utilities for pathfinder
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
import matplotlib.pyplot as plt

# SIMULATIONS
def simulate_data_grid(num_domains=3, num_modalities=2, rank=5, missing=None, noise=0.):
    """
    :param num_domains: number of domains (rows of data grid)
    :param num_modalities: number of modalities (columns of data grid)
    :param rank: integer
    :param missing: example: [ (0,1), (1,0) ] means missing Dom0, Mod1 and Dom1, Mod0
    :param noise: float
    :return: data dictionary
    """
    Domains    = [f'Dom{i}' for i in range(num_domains)]
    Modalities = [f'Mod{i}' for i in range(num_modalities)]
    Missing    = []
    if missing is not None:
        Missing = [(f'Dom{m[0]}',f'Mod{m[1]}') for m in missing]

    A, S = {}, {}
    DataSets = {}

    for d in Domains:
        n    = np.random.randint(1000, 2000)
        A[d] = np.random.randn(n, rank)**3
    for m in Modalities:
        n    = np.random.randint(100,200)
        S[m] = np.random.randn(n, rank)/10.

    DataSets = {}
    for d in Domains:
        DataSets[d] = {}
        for m in Modalities:
            if (d, m) in Missing:
                DataSets[d][m] = None
            else:
                DataSets[d][m] = A[d] @ S[m].T + noise*np.random.randn(len(A[d]), len(S[m]))

    return DataSets


# VISUALISATION
def plot_error(err, use_log=True):
    fig = plt.figure()
    ax = fig.add_subplot()
    err_np = np.reshape(np.array(err), (len(err),-1))
    if use_log:
        err_np = np.log10(err_np)
    ax.plot(err_np)
    return fig

def plot_data_grid(DataGrid):
    import matplotlib

    Domains = list(DataGrid.keys())
    Modalities = list(DataGrid[Domains[0]].keys())

    fig, axes = plt.subplots(len(Domains), len(Modalities))

    for ax, col in zip(axes[0], Modalities):
        ax.set_title(col, size='large')

    for ax, row in zip(axes[:,0], Domains):
        ax.set_ylabel(row, rotation=90, size='large')

    for i, d in enumerate(Domains):
        for j, m in enumerate(Modalities):
            if DataGrid[d][m] is not None:
                nr, nc=DataGrid[d][m].shape
                axes[i, j].add_patch(matplotlib.patches.Rectangle((0,0), nc, nr, color="blue"))
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])

    return fig
