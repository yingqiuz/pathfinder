#!/usr/bin/env python

# decomp.py - Matrix Decompositions Classes
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from tqdm import tqdm

""" Decomposition classes - Interface

All classes in this file have a shared interface:

decomp = JointClass(n_components=n_comp, n_iter=n_iter, **kwargs)
decomp.fit(X, alpha, beta)

X     : list of matrices to be decomposed
alpha : mapping from data index to row-modes indices
beta  : mapping from data index to col-modes indices

For example : Xk = U_{alpha(k)} Dk V_{beta(k)}^T   --> JointSVD
            : Xk = A_{alpha(k)} S_{beta(k)}^T      --> JointOuterDecomp
            
            
decomp.predict() -> list of predicted matrices
decomp.decomp(k) -> decomposition of matrix number k
"""


from pathfinder import utils

class JointOuterDecomp(object):
    def __init__(self, n_components, n_iter=100, dropout=-1, alpha=1e-5, method=None, method_kwargs=None, do_ica=None):
        """

        Parameters
        ----------
        n_components (int) : number of components of low rank decomposition
        n_iter (int) : number of iterations
        dropout (float) :float between 0 and 1, proportion of dropped out rows/columns of X's. If <0, no dropout
        alpha (float) : Regularisation parameter
        method : Regression method to use in sklearn (e.g. sklearn.linear_model.Ridge)
        method_kwargs : Keyword arguments to pass to sklearn method
        do_ica : Perform ICA rotation at the end. Can be 'rows', 'cols', or 'both'
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
        """ initialise A[p]'s and S[q]'s
        Uses multivariate Gaussians

        """
        self._A = [[] for _ in range(self._P)]
        self._S = [[] for _ in range(self._Q)]

        for p in range(self._P):
            nrows = Clist[ self._Alu[p][0] ].shape[0]
            self._A[p] = np.random.randn(nrows, self.n_components)
        for q in range(self._Q):
            ncols = Clist[ self._Slu[q][0] ].shape[1]
            self._S[q] = np.random.randn(ncols, self.n_components)



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


    def _update_A(self, Clist, p):
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
        self._A[p] = self.decomp(concat_mat, concat_S.T, mode='left')

        return

    def _update_S(self, Clist, q):
        """Update q-th matrix S[p]
        Concatenate horizontally all matrices in Clist where beta[k]=p
        Also concatenate the corresponding A's

        Parameters
        ----------
        Clist (list)
        q (int)
        """

        indices = self._Slu[q]
        concat_mat = np.concatenate([Clist[i] for i in indices], axis=0)
        concat_A   = np.concatenate([self._A[self._alpha[i]] for i in indices], axis=0)

        # Update A
        self._S[q] = self.decomp(concat_mat, concat_A, mode='right')

        return


    def calc_loss(self, Clist):
        """Calculate fitting error norm(Data - AS')

        :param Clist:
        :return: err as array
        """
        return [ np.linalg.norm(C-Cpred) for C,Cpred in zip(Clist, self.predict()) ]

    def predict(self):
        """Use internally stored parameters to make a prediction
        """
        Cpred = []
        for k in range(self._K):
            A = self._A[ self._alpha[k] ]
            S = self._S[ self._beta[k] ]
            Cpred.append( A@S.T )
        return Cpred


    def fit(self, Clist, alpha=None, beta=None):
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
        self.init(Clist)

        # begin loop
        loss = [self.calc_loss(Clist)]
        for _ in tqdm(range(self.n_iter)):
            for p in range(self._P):
                self._update_A(Clist, p)
            for q in range(self._Q):
                self._update_S(Clist, q)
            loss.append(self.calc_loss(Clist))

        # Do ICA at the end?
        if self.do_ica is not None:
            self._A, self._S = utils.perform_ica(self._A, self._S, self.do_ica)

        # store loss
        self._loss = loss


