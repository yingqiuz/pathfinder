#!/usr/bin/env python

# decomp.py - Utilities for pathfinder
#
# Author: 
# Akina Ying-Qiu Zheng <ying-qiu.zheng@ndcn.ox.ac.uk>
# Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

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

# Simulate with jointSVD Model
def simulate_JointSVD(K, num_U, num_V, rank, SNR = 50):
    valid = False
    while not valid:
        alpha = np.random.choice(num_U, K)
        beta  = np.random.choice(num_V, K)
        if (len(np.unique(alpha)) == num_U) & (len(np.unique(beta)) == num_V):
            valid = True

    nrows = 100

    Ulist = [np.linalg.qr(np.random.randn(nrows,rank))[0] for _ in range(num_U)]
    Vlist = [np.linalg.qr(np.random.randn(nrows,rank))[0] for _ in range(num_V)]

    D = [ np.diag(np.random.rand(rank)) for _ in range(K)]
    Clist = []
    for k in range(K):
        U = Ulist[ alpha[k] ]
        V = Vlist[ beta[k] ]

        C = U@D[k]@V.T

        # add noise
        S   = np.max(C.flatten())
        SIG = S/SNR
        C   = C + SIG*np.random.randn(*C.shape)
        Clist.append(C)
    return Clist, alpha, beta



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

    TODO: Check for situations where data share both row and col, in which case the dict won't work
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
def perform_ica(A, S, approach='both', return_ica=False, max_iter=5000, tol=1e-6):
    """ICA post-processing to resolve rotational ambiguity.

    Given factors A and S such that C_k = A[p] @ S[q]^T, applies ICA to
    find a rotation that maximises non-Gaussianity of the components, while
    preserving the product A @ S^T.

    Parameters
    ----------
    A : list of left-matrices (n_p x r)
    S : list of right-matrices (n_q x r)
    approach : str
        Which side to run ICA on:
        - 'left' or 'rows': ICA on concatenated A's, derive S's via inverse.
        - 'right' or 'cols': ICA on concatenated S's, derive A's via inverse.
        - 'both': equivalent to 'left'.
    return_ica : bool
        If True, also return the fitted ICA object.
    max_iter : int
        Maximum iterations for FastICA.
    tol : float
        Convergence tolerance for FastICA.

    Returns
    -------
    A_new, S_new : lists of rotated factor matrices
        Product is preserved: A_new[p] @ S_new[q]^T == A[p] @ S[q]^T
    ica : FastICA (only if return_ica=True)
    """
    from sklearn.decomposition import FastICA

    if approach in ['left', 'rows', 'both']:
        # ICA on concatenated A's
        A_concat = np.concatenate(A, axis=0)
        ica = FastICA(whiten='unit-variance', max_iter=max_iter, tol=tol,
                      random_state=0)
        ica.fit(A_concat)
        # ica.components_ = W @ K (unmixing @ whitening), shape (r, r)
        # ica.transform(X) = (X - mean) @ M^T — includes centering.
        # For product preservation we apply the rotation WITHOUT centering:
        #   A_new = A @ M^T,  S_new = S @ M^{-1}
        #   => A_new @ S_new^T = A @ M^T @ M^{-T} @ S^T = A @ S^T  ✓
        M = ica.components_  # (r, r)
        M_inv = np.linalg.inv(M)
        A_new = [a @ M.T for a in A]
        S_new = [s @ M_inv for s in S]

    elif approach in ['right', 'cols']:
        # ICA on concatenated S's
        S_concat = np.concatenate(S, axis=0)
        ica = FastICA(whiten='unit-variance', max_iter=max_iter, tol=tol,
                      random_state=0)
        ica.fit(S_concat)
        M = ica.components_
        M_inv = np.linalg.inv(M)
        S_new = [s @ M.T for s in S]
        A_new = [a @ M_inv for a in A]

    else:
        raise ValueError(f"approach must be 'left', 'right', 'rows', 'cols', "
                         f"or 'both', got '{approach}'")

    if return_ica:
        return A_new, S_new, ica
    else:
        return A_new, S_new


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

def plot_data_fit(data_dict, data_pred, data_complete=None):

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
            axes[i, j].scatter(groudtruth, data_pred[d][m], c=c)
            axes[i, j].axline((0, 0), slope=1, color='k')
    return fig


# DECOMPOSITION INIT HELPER (shared by JointOuterDecomp and JointSVD)
def _gram_svd_init(Clist, indices, mode, n_components, batch_size):
    """Memory-light SVD initialisation via streamed Gram accumulation.

    Returns the top-n_components singular vectors of the (never materialised)
    concatenation of the matrices in `indices`, computed as the leading
    eigenvectors of a Gram matrix accumulated one matrix - and one batch of
    columns/rows - at a time. Peak memory is bounded by the factor dimension
    squared, independent of how many matrices share the factor. Accuracy of the
    small singular directions is reduced (the Gram squares the condition
    number), which is immaterial for an initialisation the iterations refine.
    The returned columns are orthonormal.

    Parameters
    ----------
    Clist : list of 2D arrays
    indices : list of int
        Data indices sharing this factor (a lookup-table entry).
    mode : str
        'left'  : left singular vectors of the horizontal concat [C_i | ...]
                  (matrices share rows); accumulates sum_i C_i C_i^T.
        'right' : right singular vectors of the vertical concat [C_i ; ...]
                  (matrices share cols); accumulates sum_i C_i^T C_i.
    n_components : int
        Number of leading singular vectors to return.
    batch_size : int
        Column/row chunk size for the streamed accumulation.
    """
    if mode == 'left':
        dim = Clist[indices[0]].shape[0]
        G = np.zeros((dim, dim))
        for i in indices:
            C = Clist[i]
            ncol = C.shape[1]
            for c0 in range(0, ncol, batch_size):
                blk = C[:, c0:min(c0 + batch_size, ncol)]
                G += blk @ blk.T
    elif mode == 'right':
        dim = Clist[indices[0]].shape[1]
        G = np.zeros((dim, dim))
        for i in indices:
            C = Clist[i]
            nrow = C.shape[0]
            for r0 in range(0, nrow, batch_size):
                blk = C[r0:min(r0 + batch_size, nrow), :]
                G += blk.T @ blk
    else:
        raise ValueError(f"mode must be 'left' or 'right', got '{mode}'")
    # leading eigenvectors (eigh returns eigenvalues in ascending order).
    # .copy() the k columns we keep so the full (dim x dim) eigenvector matrix
    # can be freed rather than kept alive by a slice-view.
    eigvecs = np.linalg.eigh(G)[1]
    return eigvecs[:, ::-1][:, :n_components].copy()
