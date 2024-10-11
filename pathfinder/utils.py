#!/usr/bin/env python

# decomp.py - Utilities for pathfinder
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT


# TODO:
#

import numpy as np
import matplotlib.pyplot as plt


# SIMULATIONS
def simulate_data_grid(num_domains=3, num_modalities=2, rank=5, missing=None, noise=0., output_complete=False):
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

    for d in Domains:
        n    = np.random.randint(1000, 2000)
        A[d] = np.random.randn(n, rank)**3
    for m in Modalities:
        n    = np.random.randint(100,200)
        S[m] = np.random.randn(n, rank)/10.

    DataSets = {}
    if output_complete:
        DataSetsComplete = {}
    for d in Domains:
        DataSets[d] = {}
        if output_complete:
            DataSetsComplete[d] = {}
        for m in Modalities:
            AS = A[d] @ S[m].T + noise*np.random.randn(len(A[d]), len(S[m]))
            if (d, m) in Missing:
                DataSets[d][m] = None
            else:
                DataSets[d][m] = AS
            if output_complete:
                    DataSetsComplete[d][m] = AS
    if output_complete:
        return DataSets, DataSetsComplete
    else:
        return DataSets

# Predict data
def predicted_data_grid(data, A, S):
    """Predict data grid using X=AS^T
    :param data: dictionary
    :param A: list
    :param S: list
    :return: dictionary
    """
    Domains = list(data.keys())
    Modalities = list(data[Domains[0]].keys())

    pred = {}
    for i, d in enumerate(Domains):
        pred[d] = {}
        for j, m in enumerate(Modalities):
            pred[d][m] = A[i] @ S[j].T

    return pred

# Lookup stuff Dict-List
def DataTable_to_Lookup(DataDict):
    """Produce Clist,alpha/beta so we can use jointSVD
    """
    Clist = []
    alpha = []
    beta  = []
    for i, row in enumerate(DataDict):
        for j, col in enumerate(DataDict[row]):
            if DataDict[row][col] is not None:
                Clist.append(DataDict[row][col])
                alpha.append(i)
                beta.append(j)
    return Clist, alpha, beta

def Lookup_to_DataTable(DataList, alpha, beta, row_names=None, col_names=None):
    """Produce Data Dictionary based on list and lookup (for backwards compatibility)
    """
    DataDict = {}
    n_rows, n_cols = max(alpha)+1, max(beta)+1
    if row_names is not None:
        assert n_rows == len(row_names), f'n_rows={n_rows} does not match len(row_names)={len(row_names)}'
    else:
        row_names = [f'row_{i}' for i in range(n_rows)]
    if col_names is not None:
        assert n_cols == len(col_names), f'n_cols={n_cols} does not match len(col_names)={len(col_names)}'
    else:
        col_names = [f'col_{i}' for i in range(n_cols)]

    # init with None
    for i, row in enumerate(row_names):
        DataDict[row] = {}
        for j, col in enumerate(col_names):
            DataDict[row][col] = None

    for k in range(len(DataList)):
        i, j = alpha[k], beta[k]
        DataDict[row_names[i]][col_names[j]] = DataList[k]

    return DataDict



# ICA
def perform_ica(A, S, approach='both'):
    """Do a concat ICA at the end of the fitting process

    Parameters
    ----------
    A: list of left-matrices
    S: list of right-matrices
    approach (str) : 'left', 'right', or 'both'

    :return: rotated A's and S's
    """
    X = []
    if approach in ['left', 'both']:
        X.extend(A)
    if approach in ['right', 'both']:
        X.extend(S)

    from sklearn.decomposition import FastICA
    ica = FastICA(whiten=False)
    X = np.concatenate(X, axis=0)
    ica.fit(X)
    # rotate all matrices
    A = [ica.transform(a) for a in A]
    S = [ica.transform(s) for s in S]
    return A, S


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

def plot_data_fit(data_dict, A, S, data_complete=None):

    Domains    = list(data_dict.keys())
    Modalities = list(data_dict[Domains[0]].keys())

    fig, axes  = plt.subplots(len(Domains),len(Modalities))
    for ax, col in zip(axes[0], Modalities):
        ax.set_title(col, size='large')

    for ax, row in zip(axes[:, 0], Domains):
        ax.set_ylabel(row, rotation=90, size='large')

    for i, d in enumerate(data_dict):
        for j, m in enumerate(data_dict[d]):
            if data_complete is not None:
                groudtruth = data_complete[d][m]
            else:
                groudtruth = data_dict[d][m]

            c = '#4c72b0ff' if data_dict[d][m] is not None else '#dd8452ff'
            axes[i, j].scatter(groudtruth, A[i]@S[j].T, c=c)
            axes[i, j].axline((0, 0), slope=1, color='k')
    return fig
