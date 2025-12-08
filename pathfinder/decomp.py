#!/usr/bin/env python

# decomp.py - Matrix Decompositions Classes
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

import numpy as np
from tqdm import tqdm
from scipy.sparse.linalg import svds
from scipy.linalg import qr, svd

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
decomp.predict(k) -> predict matrix k
decomp.decomp(k) -> decomposition of matrix k
"""


from pathfinder import utils

class JointOuterDecomp(object):
    def __init__(self, n_components, n_iter=100, dropout=-1, 
                 alpha=1e-5, method=None, method_kwargs=None, do_ica=None, batch_size=None, learning_rate=None):
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
        batch_size (int or None): If None, use full batch updates, othewise use minibatch updates
        learning_rate (float or None): only used if batch_size is not None
        """

        self.n_components = n_components
        self.n_iter       = n_iter
        self.dropout      = dropout
        self.alpha        = alpha
        self.do_ica       = do_ica
        
        # mini batch attributes
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self._use_minibatch = (self.batch_size is not None)
        
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

        return
    
    def _update_A_minibatch(self, Clist, p):
        f"""Minibatch update for A[p] using closed-form solution
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
        self._S[q] = self.regress(concat_mat, concat_A, mode='right')

        return


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

    def fit(self, Clist, alpha=None, beta=None, random_update=None):
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
        self.init(Clist)

        # begin loop
        loss = [self.calc_loss(Clist)]
        for _ in tqdm(range(self.n_iter)):
            if not random_update:
                # update all A and S matrices in each iteration
                for p in range(self._P):
                    self._update_A(Clist, p)
                for q in range(self._Q):
                    self._update_S(Clist, q)
            else:
                # randomly select a subset of p and q to update
                for p in np.random.choice(self._P, size=np.floor(self._P * random_update), replace=False):
                    self._update_A(Clist, p)
                for q in np.random.choice(self._Q, size=np.floor(self._Q * random_update), replace=False):
                    self._update_S(Clist, q)
                
            loss.append(self.calc_loss(Clist))

        # Do ICA at the end?
        if self.do_ica is not None:
            self._A, self._S = utils.perform_ica(self._A, self._S, self.do_ica)

        # store loss
        self._loss = loss




# ANOTHER DECOMPOSITION APPROACH : JointSVD
class JointSVD(object):
    def __init__(self, n_components, n_iter=10, do_ica=None, batch_size=None, n_power_iter=2):
        """Joint Singular Value Decomposition

        Given set of matrices C1, ..., CK, performs a joint SVD such that:

        Ck = U_{alpha(k)} Dk V_{beta(k)}^T

        Where the Dk's are eigenvalue matrices (diagonal)
        And the Ua's and Vb's are a set of orthonormal matrices with n_components columns

        The mappings alpha(k) and beta(k) map from the data indices to the respective left/right ortho matrices

        Algorithm inspired by:
        Congedo M, et al. Approximate Joint Singular Value Decomposition of an Asymmetric Rectangular Matrix Set.
        ieee TSP 2010. DOI: 10.1109/TSP.2010.2087018
        """
        self._ncomp  = n_components
        self._niter  = n_iter
        self._do_ica = do_ica
        self.batch_size = batch_size
        self.n_power_iter = n_power_iter
        self._use_minibatch = (self.batch_size is not None)
        
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
        """Initialise the U's, the V's, and the D's
        U's and V's are random
        the D's are given by minimizing the loss
        """
        self._Ulist = [[] for _ in range(self._P)]
        self._Vlist = [[] for _ in range(self._Q)]
        self._Dlist = [[] for _ in range(self._K)]

        for p in range(self._P):
            nrows = Clist[ self._Ulu[p][0] ].shape[0]
            self._Ulist[p] = qr(np.random.randn(nrows, self._ncomp), mode='economic')[0]
        for q in range(self._Q):
#             self._updateV(Clist, q)
            ncols = Clist[ self._Vlu[q][0] ].shape[1]
            self._Vlist[q] = qr(np.random.randn(ncols, self._ncomp), mode='economic')[0]
        for k in range(self._K):
            self._updateD(Clist,k)


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

    def fit(self, Clist, alpha=None, beta=None):
        """Fit list of matrices

        Clist : list of 2D arrays
        alpha : list of length len(Clist) indexing left matrices for C's
        beta  : list of length len(Clist) indexing right matrices for C's


        For example, if Clist = [C0, C1, C2], and we want:

                    C0=U0D0V0^T
                    C1=U0D1V1^T
                    C2=U1D2V1^T

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

        # Initialise
        self.init(Clist)  # random for now

        self._loss = np.zeros((self._niter+1, self._K))
        self._loss[0,:] = [ np.linalg.norm(C-Cpred) for C,Cpred in zip(Clist,self.predict()) ]
        # Main algorithm
        for it in range(self._niter):
            # Update U
            for p in range(self._P):
                self._updateU(Clist, p)
            # Update V
            for q in range(self._Q):
                self._updateV(Clist, q)
            # Update D
            for k in range(self._K):
                self._updateD(Clist, k)
            self._loss[it+1,:] = [ np.linalg.norm(C-Cpred) for C,Cpred in zip(Clist,self.predict()) ]
        # Finish
        # Do ICA at the end?
        if self._do_ica is not None:
            self._Ulist, self._Vlist, ica = utils.perform_ica(self._Ulist, self._Vlist, self._do_ica, return_ica=True)
            # Need to do something with the D's otherwise predict() doesn't work
            # NEED TO FIX THIS AND WRITE A TEST

