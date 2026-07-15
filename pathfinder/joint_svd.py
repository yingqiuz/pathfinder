#!/usr/bin/env python

# joint_svd.py - Joint Singular Value Decomposition
#
# Author: 
# Akina Ying-Qiu Zheng <ying-qiu.zheng@ndcn.ox.ac.uk>
# Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

""" Joint Singular Value Decomposition

Decomposes a set of matrices Xk = U_{alpha(k)} Dk V_{beta(k)}^T, with shared
orthonormal U's / V's and per-dataset diagonal D's. See the shared interface
described in pathfinder.decomp.
"""

import numpy as np
from scipy.sparse.linalg import svds
from scipy.linalg import qr
from numpy.linalg import svd

from pathfinder import utils


class JointSVD(object):
    def __init__(self, n_components, n_iter=10, batch_size=None,
                 n_power_iter=2, update_fraction=1.0, init_type='random',
                 verbose=True):
        """Joint Singular Value Decomposition.

        Given set of matrices C1, ..., CK, performs a joint SVD such that:

        Ck = U_{alpha(k)} Dk V_{beta(k)}^T

        Where the Dk's are diagonal matrices (singular values) and the U's and V's
        are sets of orthonormal matrices with n_components columns.

        The mappings alpha(k) and beta(k) map from the data indices to the
        respective left/right orthonormal matrices.

        Algorithm inspired by:
        Congedo M, et al. Approximate Joint Singular Value Decomposition of an
        Asymmetric Rectangular Matrix Set. IEEE TSP 2010. DOI: 10.1109/TSP.2010.2087018

        Parameters
        ----------
        n_components : int
            Number of components.
        n_iter : int
            Number of iterations.
        batch_size : int or None
            If None, use full batch updates, otherwise use minibatch updates.
        n_power_iter : int
            Number of power iterations for minibatch updates.
        update_fraction : float
            Fraction of factors to update each iteration.
            1.0 means update all (default), <1.0 means random subset.
        init_type : str
            Type of initialisation. 'random' (default) uses random orthonormal
            matrices; 'svd' uses the leading singular vectors of the
            concatenated data (streamed Gram accumulation when batch_size is
            set). Both yield orthonormal U/V.
        verbose : bool
            Whether to print progress during fitting.
        """
        self._ncomp  = n_components
        self._niter  = n_iter
        self.batch_size = batch_size
        self.n_power_iter = n_power_iter
        self._use_minibatch = (self.batch_size is not None)
        self.update_fraction = update_fraction
        if init_type not in ('random', 'svd'):
            raise ValueError(f"init_type must be 'random' or 'svd', got '{init_type}'")
        self.init_type = init_type
        self.verbose = verbose

        self._K      = None
        self._P      = None
        self._Q      = None
        self._Ulist  = None
        self._Vlist  = None
        self._Dlist  = None
        self._alpha  = None
        self._beta   = None
        self._Ulu    = None
        self._Vlu    = None

        self._data_as_dict = False
        self._dict_keys = None

    def _create_lookup_tables(self):
        """Make backwards lookup tables

        Ulu : list of lists such that Ulu[p] = [list of k such that Ck has U[p] as a left matrix]
        Vlu : list of lists such that Vlu[q] = [list of k such that Ck has V[q] as a right matrix]
        """
        self._Ulu = [[] for _ in range(self._P)]
        self._Vlu = [[] for _ in range(self._Q)]

        for p in range(self._P):
            for k in range(self._K):
                if self._alpha[k] == p:
                    self._Ulu[p].append(k)

        for q in range(self._Q):
            for k in range(self._K):
                if self._beta[k] == q:
                    self._Vlu[q].append(k)


    def _check_dimensions(self, Clist):
        """Check that the dimensions of the input matrices match
        with what is required from alpha/beta
        """
        # loop through backwards lookups and check rows and cols
        for p in range(self._P):
            assert len(set([Clist[k].shape[0] for k in self._Ulu[p]])) == 1 , 'Matrices have incompatible rows'
        for q in range(self._Q):
            assert len(set([Clist[k].shape[1] for k in self._Vlu[q]])) == 1 , 'Matrices have incompatible cols'



    def init(self, Clist):
        """Initialise the U's, the V's, and the D's based on init_type.

        U's and V's are orthonormal in both cases: 'random' uses random
        orthonormal matrices (QR of Gaussian noise); 'svd' uses the leading
        singular vectors of the concatenated data (streamed Gram accumulation
        when batch_size is set). The D's are then computed to minimise the loss.
        """
        self._Ulist = [[] for _ in range(self._P)]
        self._Vlist = [[] for _ in range(self._Q)]
        self._Dlist = [[] for _ in range(self._K)]
        if self.verbose:
            print(f'Initialising U and V with {self.init_type}...')

        for p in range(self._P):
            nrows = Clist[self._Ulu[p][0]].shape[0]
            if self.init_type == 'random':
                self._Ulist[p] = qr(np.random.randn(nrows, self._ncomp), mode='economic')[0]
            elif self._use_minibatch:
                # streamed Gram accumulation - no concatenation
                self._Ulist[p] = utils._gram_svd_init(Clist, self._Ulu[p], 'left',
                                                      self._ncomp, self.batch_size)
            else:
                concat_mat = np.concatenate([Clist[i] for i in self._Ulu[p]], axis=1)
                self._Ulist[p] = svd(concat_mat, full_matrices=False)[0][:, :self._ncomp]
        for q in range(self._Q):
            ncols = Clist[self._Vlu[q][0]].shape[1]
            if self.init_type == 'random':
                self._Vlist[q] = qr(np.random.randn(ncols, self._ncomp), mode='economic')[0]
            elif self._use_minibatch:
                # streamed Gram accumulation - no concatenation
                self._Vlist[q] = utils._gram_svd_init(Clist, self._Vlu[q], 'right',
                                                      self._ncomp, self.batch_size)
            else:
                concat_mat = np.concatenate([Clist[i] for i in self._Vlu[q]], axis=0)
                self._Vlist[q] = svd(concat_mat, full_matrices=False)[2][:self._ncomp, :].T
        for k in range(self._K):
            self._updateD(Clist, k)


    def _updateD(self, Clist, k):
        """Update the eigenvalues
        Dk = diag( UTCkV )  with the appropriate U and V for Ck using the lookups
        """
        U = self._Ulist[ self._alpha[k] ]
        V = self._Vlist[ self._beta[k] ]
        self._Dlist[k] = np.diag( U.T@np.dot(Clist[k], V) )


    def _updateU(self, Clist, p):
        """Update p-th matrix U[p]"""
        if self._use_minibatch:
            self._updateU_minibatch(Clist, p)
        else:
            self._updateU_fullbatch(Clist, p)

    def _updateU_fullbatch(self, Clist, p):
        # Un is main eigenvector of Mn(V)
        U = []
        for n in range(self._ncomp):
            MV = []
            for k in range(self._K):
                if k in self._Ulu[p]:
                    C = Clist[k]
                    v = self._Vlist[self._beta[k]][:,n]
                    MV.append( np.dot(C, v) )
            MV = np.asarray(MV).T
            if MV.shape[1]>1:
                u = svds(MV, k=1)[0].flatten()
            else:
                u  = MV.flatten() / np.linalg.norm(MV)

            U.append(u)
        U = np.asarray(U).T
        # orthogonalise U
        self._Ulist[p] = qr(U, mode='economic')[0]

    def _updateU_minibatch(self, Clist, p):
        """chunck rows within matrices"""
        indices = self._Ulu[p]
        if len(indices) == 0:
            return

        nrows = Clist[indices[0]].shape[0]
        U_new = []

        for n in range(self._ncomp):
            # Sample subset of matrices
            if len(indices) > self.batch_size:
                batch_indices = np.random.choice(indices, self.batch_size, replace=False)
            else:
                batch_indices = indices

            # randomised power iteration to find dominant eigenvector
            # this only requires matrix-vector products, which can be done in chunks
            u = np.random.randn(nrows)
            u = u / np.linalg.norm(u)

            for _ in range(self.n_power_iter):
                # Compute M @ M^T @ u without forming M explicitly
                # M @ M^T @ u = sum_k (C_k @ v_k) @ (C_k @ v_k)^T @ u
                MMT_u = np.zeros(nrows)

                for k in batch_indices:
                    C = Clist[k]
                    v = self._Vlist[self._beta[k]][:, n]

                    # process in row batches
                    Cv = np.zeros(nrows)
                    for row_start in range(0, nrows, self.batch_size):
                        row_end = min(row_start + self.batch_size, nrows)
                        Cv[row_start:row_end] = C[row_start:row_end, :] @ v

                    # (C@v) @ (C@v)^T @ u = (C@v) * dot(C@v, u)
                    weight = np.dot(Cv, u)
                    MMT_u += Cv * weight

                if np.linalg.norm(MMT_u) > 1e-10:
                    u = MMT_u / np.linalg.norm(MMT_u)

            U_new.append(u)

        U_new = np.asarray(U_new).T
        self._Ulist[p] = qr(U_new, mode='economic')[0]

    def _updateV(self, Clist, q):
        """Update q-th matrix V[q]"""
        if self._use_minibatch:
            self._updateV_minibatch(Clist, q)
        else:
            self._updateV_fullbatch(Clist, q)

    def _updateV_fullbatch(self, Clist, q):
        # Vn is main eigenvector of Mn(U)
        V = []
        for n in range(self._ncomp):
            MU = []
            for k in range(self._K):
                if k in self._Vlu[q]:
                    C = Clist[k]
                    u = self._Ulist[self._alpha[k]][:,n]
                    MU.append( np.dot(C.T, u) )
            MU = np.asarray(MU).T
            if MU.shape[1]>1:
                v  = svds(MU, k=1)[0].flatten()
            else:
                v  = MU.flatten() / np.linalg.norm(MU)

            V.append(v)
        V = np.asarray(V).T
        # orthogonalise V
        self._Vlist[q] = qr(V, mode='economic')[0]

    def predict(self, k=None, as_dict=False):
        if k is None:
            Cpred = [self.predict(k) for k in range(self._K)]
            if as_dict:
                assert self._data_as_dict == True, 'Cannot predict data as dict as it was provided as a list in the fitting'
                return utils.Lookup_to_DataTable(Cpred, self._alpha, self._beta, self._dict_keys[0], self._dict_keys[1])
            else:
                return Cpred
        else:
            assert as_dict == False, 'Can only output a dict if k=None'
            U = self._Ulist[ self._alpha[k] ]
            V = self._Vlist[ self._beta[k] ]
            return U@np.diag(self._Dlist[k])@V.T

    def _updateV_minibatch(self, Clist, q):
        """ chunk columns within matrices"""
        indices = self._Vlu[q]
        if len(indices) == 0:
            return

        ncols = Clist[indices[0]].shape[1]
        V_new = []

        for n in range(self._ncomp):
            # sample subset of matrices (if batch_size specified)
            if len(indices) > self.batch_size:
                batch_indices = np.random.choice(indices, self.batch_size, replace=False)
            else:
                batch_indices = indices

            # power iteration with column chunking
            v = np.random.randn(ncols)
            v = v / np.linalg.norm(v)

            for _ in range(self.n_power_iter):
                MTM_v = np.zeros(ncols)

                for k in batch_indices:
                    C = Clist[k]
                    u = self._Ulist[self._alpha[k]][:, n]

                    # process in column batches
                    CTu = np.zeros(ncols)
                    for col_start in range(0, ncols, self.batch_size):
                        col_end = min(col_start + self.batch_size, ncols)
                        CTu[col_start:col_end] = C[:, col_start:col_end].T @ u

                    weight = np.dot(CTu, v)
                    MTM_v += CTu * weight


                if np.linalg.norm(MTM_v) > 1e-10:
                    v = MTM_v / np.linalg.norm(MTM_v)

            V_new.append(v)

        V_new = np.asarray(V_new).T
        self._Vlist[q] = qr(V_new, mode='economic')[0]

    def decomp(self, k):
        """Get USV for a given input matrix from list
        """
        U = self._Ulist[ self._alpha[k] ]
        V = self._Vlist[ self._beta[k] ]
        return U, np.diag(self._Dlist[k]), V

    def fit(self, Clist, alpha=None, beta=None, initial_state=None):
        """Fit list of matrices.

        Parameters
        ----------
        Clist : list of 2D arrays
            List of matrices to decompose.
        alpha : list of int
            List of length len(Clist) indexing left matrices (U's) for each C.
        beta : list of int
            List of length len(Clist) indexing right matrices (V's) for each C.
        initial_state : dict or None
            Dict with 'U' and 'V' keys containing lists of initial matrices.
            If provided, overrides random initialisation.

        Example
        -------
        If Clist = [C0, C1, C2], and we want:
            C0 = U0 @ D0 @ V0^T
            C1 = U0 @ D1 @ V1^T
            C2 = U1 @ D2 @ V1^T

        then alpha = [0, 0, 1] and beta = [0, 1, 1]
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

        # Initialise
        if initial_state is None:
            self.init(Clist)
        else:
            assert len(initial_state['U']) == self._P, 'Initial U has incorrect length'
            assert len(initial_state['V']) == self._Q, 'Initial V has incorrect length'
            self._Ulist = initial_state['U']
            self._Vlist = initial_state['V']
            self._Dlist = [[] for _ in range(self._K)]
            for k in range(self._K):
                self._updateD(Clist, k)

        self._loss = np.zeros((self._niter+1, self._K))
        self._loss[0,:] = [ np.linalg.norm(C-Cpred) for C,Cpred in zip(Clist,self.predict()) ]

        # Main algorithm
        for it in range(self._niter):
            if self.update_fraction >= 1.0:
                # Update all U, V, D
                for p in range(self._P):
                    self._updateU(Clist, p)
                for q in range(self._Q):
                    self._updateV(Clist, q)
            else:
                # Randomly select a subset to update
                n_p = max(1, int(self._P * self.update_fraction))
                n_q = max(1, int(self._Q * self.update_fraction))
                for p in np.random.choice(self._P, size=n_p, replace=False):
                    self._updateU(Clist, p)
                for q in np.random.choice(self._Q, size=n_q, replace=False):
                    self._updateV(Clist, q)
            # Always update all D's
            for k in range(self._K):
                self._updateD(Clist, k)
            self._loss[it+1,:] = [ np.linalg.norm(C-Cpred) for C,Cpred in zip(Clist,self.predict()) ]

            if self.verbose:
                mean_loss = np.mean(self._loss[it+1,:])
                progress = (it + 1) / self._niter * 100
                print(f'\rIteration {it+1:4d}/{self._niter} [{progress:5.1f}%] | Loss: {mean_loss:.6f}', end='')

        if self.verbose:
            print()  # newline after progress
