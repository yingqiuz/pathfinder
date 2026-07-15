#!/usr/bin/env python

# joint_outer_decomp.py - Joint Outer Product Decomposition
#
# Author: 
# Akina Ying-Qiu Zheng <ying-qiu.zheng@ndcn.ox.ac.uk>
# Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

""" Joint Outer Product Decomposition

Decomposes a set of matrices Xk = A_{alpha(k)} S_{beta(k)}^T, sharing the
left matrices A across data that share a row-mode and the right matrices S
across data that share a column-mode. See the shared interface described in
pathfinder.decomp.
"""

import numpy as np
from numpy.linalg import svd

from pathfinder import utils


class JointOuterDecomp(object):
    def __init__(self, n_components, n_iter=100, dropout=-1,
                 alpha=1e-5, method=None, method_kwargs=None, do_ica=None,
                 batch_size=None, update_fraction=1.0,
                 init_type='svd', verbose=True):
        """Joint Outer Product Decomposition.

        Decomposes a set of matrices Xk = A_{alpha(k)} @ S_{beta(k)}^T

        Parameters
        ----------
        n_components : int
            Number of components of low rank decomposition.
        n_iter : int
            Number of iterations.
        dropout : float
            Float between 0 and 1, proportion of dropped out rows/columns of X's.
            If <0, no dropout.
        alpha : float
            Regularisation parameter.
        method : class
            Regression method to use in sklearn (e.g. sklearn.linear_model.Ridge).
        method_kwargs : dict
            Keyword arguments to pass to sklearn method.
        do_ica : str or None
            Perform ICA rotation at the end. Can be 'rows', 'cols', or 'both'.
        batch_size : int or None
            If None, use full batch updates, otherwise use minibatch updates.
        update_fraction : float
            Fraction of factors to update each iteration.
            1.0 means update all (default), <1.0 means random subset.
        init_type : str
            Type of initialisation. Can be 'random' or 'svd'.
        verbose : bool
            Whether to print progress during fitting.
        """

        self.n_components = n_components
        self.n_iter       = n_iter
        self.dropout      = dropout
        self.alpha        = alpha
        self.do_ica       = do_ica
        self.verbose      = verbose

        # mini batch attributes
        self.batch_size = batch_size
        self._use_minibatch = (self.batch_size is not None)
        self.update_fraction = update_fraction
        if init_type not in ('random', 'svd'):
            raise ValueError(f"init_type must be 'random' or 'svd', got '{init_type}'")
        self.init_type = init_type

        assert self.dropout<1, 'dropout should be between 0 and 1'
        # sklearn method to use for matrix decomp
        # default = Ridge
        self.method       = method
        self.kwargs       = method_kwargs
        if method is None and not self._use_minibatch:
            from sklearn.linear_model import Ridge
            self.method = Ridge
            self.kwargs = {'alpha':alpha}


        # internal parameters (fitted)
        self._P = None  # number of row-modes
        self._Q = None  # number of column-modes
        self._K = None  # number of datasets

        self._Alu = None # lookup for A
        self._Slu = None # lookup for S
        self._alpha = None
        self._beta  = None

        self._A = None
        self._S = None
        self._loss = None
        self._data_as_dict = False
        self._dict_keys = None

    def _create_lookup_tables(self):
        """Make backwards lookup tables

        Alu : list of lists such that Alu[p] = [list of k such that Ck has A[p] as a left matrix]
        Slu : list of lists such that Slu[q] = [list of k such that Ck has S[q] as a right matrix]
        """
        self._Alu = [[] for _ in range(self._P)]
        self._Slu = [[] for _ in range(self._Q)]

        for p in range(self._P):
            for k in range(self._K):
                if self._alpha[k] == p:
                    self._Alu[p].append(k)

        for q in range(self._Q):
            for k in range(self._K):
                if self._beta[k] == q:
                    self._Slu[q].append(k)

    def _check_dimensions(self, Clist):
        """Check that the dimensions of the input matrices match
        with what is required from alpha/beta
        """
        # loop through backwards lookups and check rows and cols
        for p in range(self._P):
            assert len(set([Clist[k].shape[0] for k in self._Alu[p]])) == 1 , 'Matrices have incompatible rows'
        for q in range(self._Q):
            assert len(set([Clist[k].shape[1] for k in self._Slu[q]])) == 1 , 'Matrices have incompatible cols'

    def init(self, Clist):
        """Initialise A[p]'s and S[q]'s based on init_type."""
        self._A = [[] for _ in range(self._P)]
        self._S = [[] for _ in range(self._Q)]
        if self.verbose:
            print(f'Initialising A and S with {self.init_type}...')
        for p in range(self._P):
            nrows = Clist[ self._Alu[p][0] ].shape[0]
            if self.init_type == 'random':
                self._A[p] = np.random.randn(nrows, self.n_components)
            elif self.init_type == 'svd':  # svd
                if self._use_minibatch:
                    # streamed Gram accumulation - no concatenation
                    self._A[p] = utils._gram_svd_init(Clist, self._Alu[p], 'left',
                                                      self.n_components, self.batch_size)
                else:
                    # full concatenation (memory-intensive for large matrices)
                    concat_mat = np.concatenate([Clist[i] for i in self._Alu[p]], axis=1)
                    self._A[p] = svd(concat_mat, full_matrices=False)[0][:, :self.n_components]
            else:
                raise ValueError(f'Unrecognised init_type={self.init_type}')
        for q in range(self._Q):
            ncols = Clist[ self._Slu[q][0] ].shape[1]
            if self.init_type == 'random':
                self._S[q] = np.random.randn(ncols, self.n_components)
            elif self.init_type == 'svd':  # svd
                if self._use_minibatch:
                    # streamed Gram accumulation - no concatenation
                    self._S[q] = utils._gram_svd_init(Clist, self._Slu[q], 'right',
                                                      self.n_components, self.batch_size)
                else:
                    concat_mat = np.concatenate([Clist[i] for i in self._Slu[q]], axis=0)
                    self._S[q] = svd(concat_mat, full_matrices=False)[2][:self.n_components, :].T
            else:
                raise ValueError(f'Unrecognised init_type={self.init_type}')

    def regress(self, M, X, mode):
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

    def _update_A(self, Clist, p):
        """Update p-th matrix A[p]"""
        if self._use_minibatch:
            self._update_A_minibatch(Clist, p)
        else:
            self._update_A_fullbatch(Clist, p)

    def _update_A_fullbatch(self, Clist, p):
        """Update p-th matrix A[p]
        Concatenate horizontally all matrices in Clist where alpha[k]=p
        Also concatenate the corresponding S's

        Parameters
        ----------
        Clist (list)
        p (int)

        """
        indices = self._Alu[p]
        concat_mat = np.concatenate([Clist[i] for i in indices], axis=1)
        concat_S   = np.concatenate([self._S[self._beta[i]] for i in indices], axis=0)

        # Update A
        self._A[p] = self.regress(concat_mat, concat_S.T, mode='left')

    def _update_A_minibatch(self, Clist, p):
        """Minibatch update for A[p] using closed-form solution
        solves: A = C @ S @ (S^T@S + alpha*I)^{-1} by accumulating S^T @ S and C @ S in mini-batches
        TODO: add support for full SGD?
        """
        indices = self._Alu[p]
        if len(indices) == 0:
            return

        nrows = Clist[indices[0]].shape[0]
        n_components = self.n_components

        # solve A^T = (S^T @ S + alpha*I)^{-1} @ (S^T @ C^T)
        S_gram = np.zeros((n_components, n_components)) # sum of S_k^T @ S_k
        C_S = np.zeros((nrows, n_components)) # sum of C_k @ S_k

        # for each matrix that uses A[p]
        for k in indices:
            C_k = Clist[k]
            S_k = self._S[self._beta[k]]

            # process columns in batches
            n_cols = C_k.shape[1]
            for col_start in range(0, n_cols, self.batch_size):
                col_end = min(col_start + self.batch_size, n_cols)
                C_batch = C_k[:, col_start:col_end] # (nrows, batch_size)
                S_batch = S_k[col_start:col_end, :] # (batch_size, n_components)
                S_gram += S_batch.T @ S_batch # (n_components, n_components)
                C_S += C_batch @ S_batch # (nrows, n_components)

        #L2 regularisation
        S_gram += self.alpha * np.eye(n_components)

        # solve A = C_S @ S_gram^{-1}
        try:
            self._A[p] = np.linalg.solve(S_gram, C_S.T).T
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix in A update for mode {p}. Using pseudo-inverse instead.")
            self._A[p] = C_S @ np.linalg.pinv(S_gram)

    def _update_S(self, Clist, q):
        """Update q-th matrix S[q]"""
        if self._use_minibatch:
            self._update_S_minibatch(Clist, q)
        else:
            self._update_S_fullbatch(Clist, q)

    def _update_S_fullbatch(self, Clist, q):
        """Update q-th matrix S[q].

        Concatenate vertically all matrices in Clist where beta[k]=q.
        Also concatenate the corresponding A's.

        Parameters
        ----------
        Clist : list
            List of data matrices.
        q : int
            Index of S matrix to update.
        """

        indices = self._Slu[q]
        concat_mat = np.concatenate([Clist[i] for i in indices], axis=0)
        concat_A   = np.concatenate([self._A[self._alpha[i]] for i in indices], axis=0)

        # Update S
        self._S[q] = self.regress(concat_mat, concat_A, mode='right')

    def _update_S_minibatch(self, Clist, q):
        """Minibatch update for S[q] using closed-form solution
        solves: S = C^T @ A @ (A^T @ A + alpha*I)^{-1} accumulating A^T @ A and C^T @ A in mini-batches
        """
        indices = self._Slu[q]
        if len(indices) == 0:
            return

        ncols = Clist[indices[0]].shape[1]
        n_components = self.n_components

        A_gram = np.zeros((n_components, n_components)) # sum of A_k^T @ A_k
        CT_A = np.zeros((ncols, n_components)) # sum of C_k^T @ A_k

        # for each matrix that uses S[q]
        for k in indices:
            C_k = Clist[k]
            A_k = self._A[self._alpha[k]]

            # process rows in batches
            n_rows = C_k.shape[0]
            for row_start in range(0, n_rows, self.batch_size):
                row_end = min(row_start + self.batch_size, n_rows)
                C_batch = C_k[row_start:row_end, :] # (batch_size, ncols)
                A_batch = A_k[row_start:row_end, :] # (batch_size, n_components)
                A_gram += A_batch.T @ A_batch # (n_components, n_components)
                CT_A += C_batch.T @ A_batch # (ncols, n_components)

        #L2 regularisation
        A_gram += self.alpha * np.eye(n_components)

        # solve S = CT_A @ A_gram^{-1}
        try:
            self._S[q] = np.linalg.solve(A_gram, CT_A.T).T
        except np.linalg.LinAlgError:
            print(f"Warning: Singular matrix in S update for mode {q}. Using pseudo-inverse instead.")
            self._S[q] = CT_A @ np.linalg.pinv(A_gram)

    def calc_loss(self, Clist):
        """Calculate fitting error norm(Data - AS')

        :param Clist:
        :return: err as array
        """
        return [ np.linalg.norm(C-Cpred) for C,Cpred in zip(Clist, self.predict()) ]


    def decomp(self, k):
        A = self._A[ self._alpha[k] ]
        S = self._S[ self._beta[k] ]
        return A, S

    def predict(self, k=None, as_dict=False):
        """Use internally stored parameters to make a prediction
        """
        if k is None:
            Cpred = [self.predict(k) for k in range(self._K)]
            if as_dict:
                assert self._data_as_dict == True, 'Cannot predict data as dict as it was provided as a list in the fitting'
                return utils.Lookup_to_DataTable(Cpred, self._alpha, self._beta, self._dict_keys[0], self._dict_keys[1])
            else:
                return Cpred
        else:
            assert as_dict == False, 'Can only output a dict if k=None'
            A = self._A[ self._alpha[k] ]
            S = self._S[ self._beta[k] ]
            return A@S.T

    def fit(self, Clist, alpha=None, beta=None, initial_state=None):
        """Fit list of matrices

        Clist : list of 2D arrays
        alpha : list of length len(Clist) indexing left matrices for C's
        beta  : list of length len(Clist) indexing right matrices for C's


        For example, if Clist = [C0, C1, C2], and we want:

                    C0=A0 @ S0^T
                    C1=A0 @ S1^T
                    C2=A1 @ S1^T

        then alpha = [0,0,1] and beta = [0,1,1]

        """
        # Possibly not the most efficient algorithm in the world, as it concatenates
        # potentially large matrices.
        # Checks data dimensions
        if (alpha is None) or (beta is None):
            # it is assumed that Clist is actually a dict
            assert type(Clist) == dict, 'Clist should be dict if alpha/beta are not provided'
            self._data_as_dict = True
            Domains    = list(Clist.keys())
            Modalities = list(Clist[Domains[0]].keys())
            self._dict_keys = (Domains, Modalities)
            Clist, alpha, beta = utils.DataTable_to_Lookup(Clist)


        assert (len(Clist)==len(alpha)) & (len(Clist)==len(beta)), 'alpha and beta must have the same length as Clist'
        self._alpha = alpha
        self._beta  = beta
        self._K     = len(Clist)
        self._P     = max(self._alpha)+1
        self._Q     = max(self._beta)+1
        # make backwards lookup tables
        self._create_lookup_tables()
        self._check_dimensions(Clist)

        # run the algorithm
        if initial_state is None:
            self.init(Clist)
        else:
            # check the dimension of the initial state
            assert len(initial_state['A']) == self._P, 'Initial A has incorrect shape'
            assert len(initial_state['S']) == self._Q, 'Initial S has incorrect shape'
            self._A = initial_state['A']
            self._S = initial_state['S']

        # begin loop
        loss = [self.calc_loss(Clist)]
        for it in range(self.n_iter):
            if self.update_fraction >= 1.0:
                # update all A and S matrices in each iteration
                for q in range(self._Q):
                    self._update_S(Clist, q)
                for p in range(self._P):
                    self._update_A(Clist, p)
            else:
                # randomly select a subset of p and q to update
                n_q = max(1, int(self._Q * self.update_fraction))
                n_p = max(1, int(self._P * self.update_fraction))
                for q in np.random.choice(self._Q, size=n_q, replace=False):
                    self._update_S(Clist, q)
                for p in np.random.choice(self._P, size=n_p, replace=False):
                    self._update_A(Clist, p)

            loss.append(self.calc_loss(Clist))

            if self.verbose:
                mean_loss = np.mean(loss[-1])
                progress = (it + 1) / self.n_iter * 100
                print(f'\rIteration {it+1:4d}/{self.n_iter} [{progress:5.1f}%] | Loss: {mean_loss:.6f}', end='')

        if self.verbose:
            print()  # newline after progress

        # Do ICA at the end?
        if self.do_ica is not None:
            self._A, self._S = utils.perform_ica(self._A, self._S, self.do_ica)

        # store loss
        self._loss = loss
