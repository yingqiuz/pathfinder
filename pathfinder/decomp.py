#!/usr/bin/env python

# decomp.py - Matrix Decompositions Classes
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from tqdm import tqdm

from pathfinder import utils

class JointDecomp(object):
    def __init__(self, n_components, n_iter=100, dropout=-1, alpha=1e-5, method=None, method_kwargs=None, do_ica=None):
        """Decomposition of a set of data matrices into X=AS'

        :param n_components: number of components of low rank decomposition
        :param n_iter: number of iterations
        :param dropout: float between 0 and 1, proportion of dropped out rows/columns of X's. If <0, no dropout
        :param alpha: Regularisation parameter
        :param method: Regression method to use in sklearn (e.g. sklearn.linear_model.Ridge)
        :param method_kwargs: Keyword arguments to pass to sklearn method
        :param do_ica: Perform ICA rotation at the end. Can be 'rows', 'cols', or 'both'
        """
        self.n_components = n_components
        self.n_iter       = n_iter
        self.dropout      = dropout
        self.alpha        = alpha
        self.do_ica       = do_ica
        assert self.dropout<1, 'dropout should be between 0 and 1'
        # sklearn method to use for matrix decomp
        # default = Ridge
        self.method       = method
        self.kwargs       = method_kwargs
        if method is None:
            from sklearn.linear_model import Ridge
            self.method = Ridge
            self.kwargs = {'alpha':alpha}

    def init(self, DataSet, n_components):
        """ initialise A[d]'s and S[m]'s
        For now uses multivariate Gaussians

        :returns A, S (lists of 2D arrays)
        """
        A, S = [], []
        D, M = self.get_dataset_dimensions(DataSet)
        row_dims, col_dims =  self.get_dataset_matrix_dimensions(DataSet)
        row_dims = [np.unique(r[r>0])[0] for r in row_dims]
        col_dims = [np.unique(c[c>0])[0] for c in col_dims.T]

        for d in range(D):
            n = row_dims[d]
            A.append(np.random.randn(n,n_components))

        for m in range(M):
            n    = col_dims[m]
            S.append(np.random.randn(n,n_components))

        return A, S

    @staticmethod
    def DictToList(DataSetAsDict):
        """Turns the data dictionary into a list of lists
        :param DataSetAsDict: dict
        :return: list
        """
        return [list(d.values()) for d in list(DataSetAsDict.values())]

    @staticmethod
    def get_dataset_matrix_dimensions(DataSet):
        """Get rows and columns of matrices in data dict
        Input:
        DataSet (dict)
        Output:
        2D array : row dims
        2D array : column dims
        """
        row_dims = np.array([[DataSet[d][m].shape[0] if DataSet[d][m] is not None else -1 for m in DataSet[d]] for d in DataSet], dtype=int)
        col_dims = np.array([[DataSet[d][m].shape[1] if DataSet[d][m] is not None else -1 for m in DataSet[d]] for d in DataSet], dtype=int)
        return row_dims, col_dims

    def check_dataset_dimensions(self, DataSet):
        """Check that dimensions are compatible within modalities and domains
        """
        row_dims, col_dims =  self.get_dataset_matrix_dimensions(DataSet)

        assert len(np.unique([len(DataSets[d]) for d in DataSets]))==1, f"Incompatible number of matrices per domain"

        assert all([len(np.unique(r[r>0]))==1 for r in row_dims]), f"Incompatible row dimensions {row_dims}"
        assert all([len(np.unique(c[c>0]))==1 for c in col_dims.T]), f"Incompatible col dimensions {col_dims}"

    @staticmethod
    def get_dataset_dimensions(DataSet):
        """calc D and M (number of domains and modalities)

        :returns D, M (lists of row and column dimensions)
        """
        #check_dataset_dimensions(DataSet)
        D = len(DataSet)
        M = [len(DataSet[d]) for d in DataSet][0]
        return D, M

    def get_dataset_mask(self, DataSet):
        """Get mask of missing vs non-missing data (False=missing)

        :returns boolean 2D array
        """
        D, M = self.get_dataset_dimensions(DataSet)
        mask = np.empty((D,M),dtype=bool)
        for i,d in enumerate(DataSet):
            for j,m in enumerate(DataSet[d]):
                if DataSet[d][m] is None:
                    mask[i,j] = False
                else:
                    mask[i,j] = True
        return mask

    @staticmethod
    def concat_d(DataSet, d):
        """Concatenate matrices within domain d

        :returns numpy array
        """
        assert type(DataSet)==list, "DataSet must be a list"
        cmat = []
        for mat in DataSet[d]:
            if mat is not None:
                cmat.append(mat)
        return np.concatenate(cmat, axis=1)

    @staticmethod
    def concat_m(DataSet, m):
        """Concatenate matrices within modality m

        :returns numpy array
        """
        assert type(DataSet)==list, "DataSet must be a list"
        cmat = []
        for row in DataSet:
            mat = row[m]
            if mat is not None:
                cmat.append(mat)
        return np.concatenate(cmat, axis=0)


    def decomp(self, M, X, mode):
        """
        M (2D-array)
        X (2D-array)
        mode (str) : either 'left' or 'right'
        alpha (float) : L2 regularisation

        Solve M=AX -> find A (mode='left')
        or
        Solve M=XS -> find S (mode='right')

        :returns A or S
        """
        XX = X
        MM = M
        if self.dropout>0:
            if mode == 'left':
                # drop columns of X
                n   = int(X.shape[1]*self.dropout)
                idx = np.random.choice(X.shape[1], n, replace=False)
                XX = XX[:,idx]
                MM = MM[:,idx]
            else:
                # drop rows of X
                n   = int(X.shape[0]*self.dropout)
                idx = np.random.choice(X.shape[0], n, replace=False)
                XX = XX[idx,:]
                MM = MM[idx,:]

        # do the business
        reg = self.method(**self.kwargs)
        if mode == 'left':
            reg.fit(XX.T,MM.T)
            return reg.coef_
        elif mode == 'right':
            reg.fit(XX,MM)
            return reg.coef_
        else:
            raise(Exception(f'Unrecognised mode={mode}'))


    def _update_A(self, DataSet, A, S, mask):
        D,M = len(A),len(S)
        for d in range(D):
            # concat
            concat_mat = self.concat_d(DataSet, d)
            concat_S = []
            for col in range(M):
                if mask[d,col]:
                    concat_S.append(S[col])
            concat_S = np.concatenate(concat_S, axis=0)
            # calc decomp
            A_d = self.decomp(concat_mat, concat_S.T, mode='left')
            # update A_d
            A[d] = A_d
        return

    def _update_S(self, DataSet, A, S, mask):
        D,M = len(A),len(S)
        for m in range(M):
            # concat
            concat_mat = self.concat_m(DataSet, m)
            concat_A   = []
            for row in range(D):
                if mask[row,m]:
                    concat_A.append(A[row])
            concat_A = np.concatenate(concat_A, axis=0)
            # calc decomp
            S_m = self.decomp(concat_mat, concat_A, mode='right')
            # update A_d
            S[m] = S_m
        return


    def _perform_ica(self, A, S):
        """Do a concat ICA at the end of the fitting process

        :param A: list of left-matrices
        :param S: list of right-matrices
        :return: rotated A's and S's
        """
        X = []
        if self.do_ica in ['left', 'both']:
            X.extend(A)
        if self.do_ica in ['right', 'both']:
            X.extend(S)

        from sklearn.decomposition import FastICA
        ica = FastICA(whiten=False)
        X = np.concatenate(X, axis=0)
        ica.fit(X)
        # rotate all matrices
        A = [ica.transform(a) for a in A]
        S = [ica.transform(s) for s in S]
        return A, S

    @staticmethod
    def calc_err(DataSet, A, S):
        """Calculate fitting error norm(Data - AS')

        :param DataSet:
        :param A: list of matrices
        :param S: list of matrices
        :return: err as array
        """
        D,M = len(A),len(S)
        err = []
        for d in range(D):
            err_row = []
            A_d = A[d]
            for m in range(M):
                S_m = S[m]
                if DataSet[d][m] is not None:
                    err_row.append( np.linalg.norm(DataSet[d][m]-A_d@S_m.T, ord='fro') )
                else:
                    err_row.append(np.nan)
            err.append(err_row)
        return np.array(err)


    def fit(self, DataSet):
        """Run the algorithm

        :param DataSet:
        :return: A, S, err (all lists of lists)
        """
        # Possibly not the most efficient algorithm in the world, as it concatenates
        # potentially large matrices.
        # Get data dimensions
        D, M = self.get_dataset_dimensions(DataSet)
        mask = self.get_dataset_mask(DataSet)

        # run the algorithm
        A, S = self.init(DataSet, n_components=self.n_components)

        # begin loop
        # Begin by making the DataSet a list
        if type(DataSet) == dict:
            DataSet = self.DictToList(DataSet)
        err = [self.calc_err(DataSet, A, S)]
        for _ in tqdm(range(self.n_iter)):
            self._update_A(DataSet, A, S, mask)
            self._update_S(DataSet, A, S, mask)
            err.append(self.calc_err(DataSet, A, S))

        # Do ICA at the end?
        if self.do_ica is not None:
            A, S = self._perform_ica(A, S)

        return A, S, err

