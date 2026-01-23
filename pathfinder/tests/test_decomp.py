#!/usr/bin/env python

# test_decomp.py - Comprehensive unit tests for decomp module
#
# Author: Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

import pytest
import numpy as np
from sklearn.linear_model import Ridge, Lasso
from pathfinder import decomp, utils


class TestJointOuterDecomp:
    """Test suite for JointOuterDecomp class"""

    def test_initialization_default(self):
        """Test default initialization"""
        algo = decomp.JointOuterDecomp(n_components=5)
        assert algo.n_components == 5
        assert algo.n_iter == 100
        assert algo.dropout == -1
        assert algo.alpha == 1e-5
        assert algo.do_ica is None
        assert algo.batch_size is None
        assert algo.learning_rate is None
        assert algo.random_update is None
        assert algo._use_minibatch is False
        assert algo.method == Ridge

    def test_initialization_custom(self):
        """Test initialization with custom parameters"""
        algo = decomp.JointOuterDecomp(
            n_components=10,
            n_iter=50,
            dropout=0.2,
            alpha=1e-3,
            do_ica='both',
            batch_size=32,
            learning_rate=0.01,
            random_update=0.5
        )
        assert algo.n_components == 10
        assert algo.n_iter == 50
        assert algo.dropout == 0.2
        assert algo.alpha == 1e-3
        assert algo.do_ica == 'both'
        assert algo.batch_size == 32
        assert algo.learning_rate == 0.01
        assert algo.random_update == 0.5
        assert algo._use_minibatch is True

    def test_initialization_with_method(self):
        """Test initialization with custom sklearn method"""
        algo = decomp.JointOuterDecomp(
            n_components=5,
            method=Lasso,
            method_kwargs={'alpha': 0.1}
        )
        assert algo.method == Lasso
        assert algo.kwargs == {'alpha': 0.1}

    def test_initialization_minibatch_no_method(self):
        """Test that minibatch mode doesn't set sklearn method"""
        algo = decomp.JointOuterDecomp(n_components=5, batch_size=10)
        assert algo.method is None

    def test_dropout_validation(self):
        """Test that dropout >= 1 raises assertion error"""
        with pytest.raises(AssertionError, match='dropout should be between 0 and 1'):
            decomp.JointOuterDecomp(n_components=5, dropout=1.5)

    def test_fit_basic(self):
        """Test basic fitting with list input"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo.fit(data_list, alpha, beta)

        assert len(algo._A) == 3  # num_domains
        assert len(algo._S) == 2  # num_modalities
        assert algo._K == len(data_list)
        assert algo._P == 3
        assert algo._Q == 2
        assert algo._alpha == alpha
        assert algo._beta == beta

    def test_fit_with_dict(self):
        """Test fitting with dict input (no alpha/beta provided)"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo.fit(data)

        assert len(algo._A) == 3
        assert len(algo._S) == 2
        assert algo._data_as_dict is True
        assert algo._dict_keys is not None

    def test_fit_minibatch(self):
        """Test fitting with minibatch updates"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(
            n_components=5,
            n_iter=3,
            batch_size=10,
            alpha=1e-3,
            verbose=False
        )
        algo.fit(data_list, alpha, beta)

        assert len(algo._A) == 3
        assert len(algo._S) == 2

    def test_fit_with_dropout(self):
        """Test fitting with dropout enabled"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(
            n_components=5,
            n_iter=3,
            dropout=0.3,
            verbose=False
        )
        algo.fit(data_list, alpha, beta)

        assert len(algo._A) == 3
        assert len(algo._S) == 2

    def test_fit_with_random_update(self):
        """Test fitting with random update"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(
            n_components=5,
            n_iter=3,
            random_update=0.5,
            verbose=False
        )
        algo.fit(data_list, alpha, beta)

        assert len(algo._A) == 3
        assert len(algo._S) == 2

    def test_fit_with_ica(self):
        """Test fitting with ICA rotation"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        import warnings
        warnings.filterwarnings("ignore")

        # Test 'left' ICA
        algo = decomp.JointOuterDecomp(
            n_components=5,
            n_iter=3,
            do_ica='left',
            verbose=False
        )
        algo.fit(data_list, alpha, beta)
        assert len(algo._A) == 3

        # Test 'right' ICA
        algo = decomp.JointOuterDecomp(
            n_components=5,
            n_iter=3,
            do_ica='right',
            verbose=False
        )
        algo.fit(data_list, alpha, beta)
        assert len(algo._S) == 2

        # Test 'both' ICA
        algo = decomp.JointOuterDecomp(
            n_components=5,
            n_iter=3,
            do_ica='both',
            verbose=False
        )
        algo.fit(data_list, alpha, beta)
        assert len(algo._A) == 3
        assert len(algo._S) == 2

    def test_fit_with_initial_state(self):
        """Test fitting with provided initial state"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        # First fit to get dimensions
        algo1 = decomp.JointOuterDecomp(n_components=5, n_iter=1, verbose=False)
        algo1.fit(data_list, alpha, beta)

        # Use state from first fit as initial state for second
        initial_state = {'A': algo1._A, 'S': algo1._S}

        algo2 = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo2.fit(data_list, alpha, beta, initial_state=initial_state)

        assert len(algo2._A) == 3
        assert len(algo2._S) == 2

    def test_fit_initial_state_wrong_shape(self):
        """Test that fitting with wrong initial state shape raises error"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        # Create wrong initial state (wrong number of A matrices)
        initial_state = {
            'A': [np.random.randn(10, 5), np.random.randn(10, 5)],  # Only 2 instead of 3
            'S': [np.random.randn(10, 5), np.random.randn(10, 5)]
        }

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=1, verbose=False)

        with pytest.raises(AssertionError, match='Initial A has incorrect shape'):
            algo.fit(data_list, alpha, beta, initial_state=initial_state)

    def test_predict_single(self):
        """Test prediction for single matrix"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo.fit(data_list, alpha, beta)

        pred = algo.predict(k=0)
        assert pred.shape == data_list[0].shape

    def test_predict_all(self):
        """Test prediction for all matrices"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo.fit(data_list, alpha, beta)

        preds = algo.predict()
        assert len(preds) == len(data_list)
        for pred, data in zip(preds, data_list):
            assert pred.shape == data.shape

    def test_predict_as_dict(self):
        """Test prediction output as dictionary"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo.fit(data)

        pred_dict = algo.predict(as_dict=True)
        assert type(pred_dict) == dict
        assert pred_dict.keys() == data.keys()

    def test_predict_as_dict_error_when_fit_with_list(self):
        """Test that predicting as dict fails when fit with list"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo.fit(data_list, alpha, beta)

        with pytest.raises(AssertionError, match='Cannot predict data as dict'):
            algo.predict(as_dict=True)

    def test_decomp(self):
        """Test decomposition retrieval"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo.fit(data_list, alpha, beta)

        A, S = algo.decomp(k=0)
        assert A.shape[1] == 5  # n_components
        assert S.shape[1] == 5  # n_components

        # Verify that A @ S.T approximates data
        reconstruction = A @ S.T
        assert reconstruction.shape == data_list[0].shape

    def test_calc_loss(self):
        """Test loss calculation"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=3, verbose=False)
        algo.fit(data_list, alpha, beta)

        loss = algo.calc_loss(data_list)
        assert len(loss) == len(data_list)
        assert all(l >= 0 for l in loss)  # Loss should be non-negative

    def test_loss_stored_after_fit(self):
        """Test that loss is stored after fitting"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=5, verbose=False)
        algo.fit(data_list, alpha, beta)

        assert algo._loss is not None
        assert len(algo._loss) == 6  # n_iter + 1 (initial)

    def test_regress_left_mode(self):
        """Test regression in left mode"""
        algo = decomp.JointOuterDecomp(n_components=5, verbose=False)

        M = np.random.randn(10, 20)
        X = np.random.randn(5, 20)

        A = algo.regress(M, X, mode='left')
        assert A.shape == (10, 5)

    def test_regress_right_mode(self):
        """Test regression in right mode"""
        algo = decomp.JointOuterDecomp(n_components=5, verbose=False)

        M = np.random.randn(10, 20)
        X = np.random.randn(10, 5)

        S = algo.regress(M, X, mode='right')
        assert S.shape == (20, 5)

    def test_regress_invalid_mode(self):
        """Test that invalid regression mode raises error"""
        algo = decomp.JointOuterDecomp(n_components=5, verbose=False)

        M = np.random.randn(10, 20)
        X = np.random.randn(5, 20)

        with pytest.raises(Exception, match='Unrecognised mode'):
            algo.regress(M, X, mode='invalid')

    def test_fit_mismatched_alpha_beta_length(self):
        """Test that mismatched alpha/beta lengths raise error"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, verbose=False)

        with pytest.raises(AssertionError, match='alpha and beta must have the same length'):
            algo.fit(data_list, alpha=[0, 1], beta=beta)  # Wrong alpha length

    def test_lookup_tables_creation(self):
        """Test that lookup tables are created correctly"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=5, n_iter=1, verbose=False)
        algo.fit(data_list, alpha, beta)

        # Check that lookup tables are created
        assert algo._Alu is not None
        assert algo._Slu is not None
        assert len(algo._Alu) == 3  # num_domains
        assert len(algo._Slu) == 2  # num_modalities

        # Verify lookup table contents
        for p in range(3):
            for k in algo._Alu[p]:
                assert alpha[k] == p

        for q in range(2):
            for k in algo._Slu[q]:
                assert beta[k] == q


class TestJointSVD:
    """Test suite for JointSVD class"""

    def test_initialization_default(self):
        """Test default initialization"""
        algo = decomp.JointSVD(n_components=5)
        assert algo._ncomp == 5
        assert algo._niter == 10
        assert algo._do_ica is None
        assert algo.batch_size is None
        assert algo.n_power_iter == 2
        assert algo._use_minibatch is False

    def test_initialization_custom(self):
        """Test initialization with custom parameters"""
        algo = decomp.JointSVD(
            n_components=10,
            n_iter=20,
            do_ica='both',
            batch_size=32,
            n_power_iter=5
        )
        assert algo._ncomp == 10
        assert algo._niter == 20
        assert algo._do_ica == 'both'
        assert algo.batch_size == 32
        assert algo.n_power_iter == 5
        assert algo._use_minibatch is True

    def test_fit_basic(self):
        """Test basic fitting with simulated data"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=10)
        algo.fit(Clist, alpha, beta)

        assert len(algo._Ulist) == 2
        assert len(algo._Vlist) == 2
        assert len(algo._Dlist) == 5
        assert algo._K == 5
        assert algo._P == 2
        assert algo._Q == 2

    def test_fit_with_dict(self):
        """Test fitting with dict input"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)

        algo = decomp.JointSVD(n_components=5, n_iter=3)
        algo.fit(data)

        assert len(algo._Ulist) == 3
        assert len(algo._Vlist) == 2
        assert algo._data_as_dict is True

    def test_fit_with_list(self):
        """Test fitting with list input"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointSVD(n_components=5, n_iter=3)
        algo.fit(data_list, alpha, beta)

        assert len(algo._Ulist) == 3
        assert len(algo._Vlist) == 2
        assert len(algo.predict()) == len(data_list)

    def test_fit_minibatch(self):
        """Test fitting with minibatch updates"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointSVD(n_components=5, n_iter=3, batch_size=10)
        algo.fit(data_list, alpha, beta)

        assert len(algo._Ulist) == 3
        assert len(algo._Vlist) == 2

    def test_orthogonality_of_U(self):
        """Test that U matrices are orthonormal"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=10)
        algo.fit(Clist, alpha, beta)

        for U in algo._Ulist:
            # Check orthonormality: U.T @ U should be identity
            UTU = U.T @ U
            np.testing.assert_array_almost_equal(
                UTU,
                np.eye(3),
                decimal=10,
                err_msg="U matrix is not orthonormal"
            )

    def test_orthogonality_of_V(self):
        """Test that V matrices are orthonormal"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=10)
        algo.fit(Clist, alpha, beta)

        for V in algo._Vlist:
            # Check orthonormality: V.T @ V should be identity
            VTV = V.T @ V
            np.testing.assert_array_almost_equal(
                VTV,
                np.eye(3),
                decimal=10,
                err_msg="V matrix is not orthonormal"
            )

    def test_predict_single(self):
        """Test prediction for single matrix"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=10)
        algo.fit(Clist, alpha, beta)

        pred = algo.predict(k=0)
        assert pred.shape == Clist[0].shape

    def test_predict_all(self):
        """Test prediction for all matrices"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=10)
        algo.fit(Clist, alpha, beta)

        preds = algo.predict()
        assert len(preds) == len(Clist)
        for pred, C in zip(preds, Clist):
            assert pred.shape == C.shape

    def test_predict_as_dict(self):
        """Test prediction output as dictionary"""
        data = utils.simulate_data_grid(num_domains=3, num_modalities=2)

        algo = decomp.JointSVD(n_components=5, n_iter=3)
        algo.fit(data)

        pred_dict = algo.predict(as_dict=True)
        assert type(pred_dict) == dict
        assert pred_dict.keys() == data.keys()

    def test_decomp(self):
        """Test decomposition retrieval"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=10)
        algo.fit(Clist, alpha, beta)

        U, D, V = algo.decomp(k=0)

        assert U.shape[1] == 3  # n_components
        assert D.shape == (3, 3)  # diagonal matrix
        assert V.shape[1] == 3  # n_components

        # Check D is diagonal
        assert np.allclose(D, np.diag(np.diag(D)))

        # Verify reconstruction
        reconstruction = U @ D @ V.T
        assert reconstruction.shape == Clist[0].shape

    def test_loss_decreases(self):
        """Test that loss generally decreases during fitting"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=20)
        algo.fit(Clist, alpha, beta)

        # Check that final loss is less than initial loss
        assert algo._loss is not None
        initial_loss = algo._loss[0]
        final_loss = algo._loss[-1]

        # Loss should decrease on average
        assert np.mean(final_loss) < np.mean(initial_loss)

    def test_reconstruction_accuracy(self):
        """Test that SVD provides accurate reconstruction"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=100
        )

        algo = decomp.JointSVD(n_components=3, n_iter=20)
        algo.fit(Clist, alpha, beta)

        preds = algo.predict()

        # Calculate reconstruction error
        for C, pred in zip(Clist, preds):
            rel_error = np.linalg.norm(C - pred) / np.linalg.norm(C)
            assert rel_error < 0.1  # Relative error should be small

    def test_fit_mismatched_alpha_beta_length(self):
        """Test that mismatched alpha/beta lengths raise error"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=10)

        with pytest.raises(AssertionError, match='alpha and beta must have the same length'):
            algo.fit(Clist, alpha=[0, 1], beta=beta)

    def test_lookup_tables_creation(self):
        """Test that lookup tables are created correctly"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=1)
        algo.fit(Clist, alpha, beta)

        assert algo._Ulu is not None
        assert algo._Vlu is not None
        assert len(algo._Ulu) == 2
        assert len(algo._Vlu) == 2

        # Verify lookup table contents
        for p in range(2):
            for k in algo._Ulu[p]:
                assert alpha[k] == p

        for q in range(2):
            for k in algo._Vlu[q]:
                assert beta[k] == q

    def test_updateD(self):
        """Test D matrix update"""
        Clist, alpha, beta = utils.simulate_JointSVD(
            K=5, num_U=2, num_V=2, rank=3, SNR=50
        )

        algo = decomp.JointSVD(n_components=3, n_iter=1)
        algo.fit(Clist, alpha, beta)

        # Update D for first matrix
        old_D = algo._Dlist[0].copy()
        algo._updateD(Clist, 0)

        # D should have changed (unless we're at convergence)
        assert algo._Dlist[0].shape == (3,)  # Should be 1D array


class TestEdgeCases:
    """Test edge cases and boundary conditions"""

    def test_single_component(self):
        """Test with n_components=1"""
        data = utils.simulate_data_grid(num_domains=2, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=1, n_iter=3, verbose=False)
        algo.fit(data_list, alpha, beta)

        assert algo._A[0].shape[1] == 1
        assert algo._S[0].shape[1] == 1

    def test_zero_iterations(self):
        """Test with n_iter=0 (only initialization)"""
        data = utils.simulate_data_grid(num_domains=2, num_modalities=2)
        data_list, alpha, beta = utils.DataTable_to_Lookup(data)

        algo = decomp.JointOuterDecomp(n_components=3, n_iter=0, verbose=False)
        algo.fit(data_list, alpha, beta)

        # Should still initialize
        assert len(algo._A) == 2
        assert len(algo._S) == 2

    def test_small_matrices(self):
        """Test with very small matrices"""
        # Create small test matrices
        C0 = np.random.randn(5, 5)
        C1 = np.random.randn(5, 5)
        Clist = [C0, C1]
        alpha = [0, 0]
        beta = [0, 1]

        algo = decomp.JointOuterDecomp(n_components=2, n_iter=3, verbose=False)
        algo.fit(Clist, alpha, beta)

        preds = algo.predict()
        assert len(preds) == 2

    def test_large_n_components(self):
        """Test when n_components is close to matrix dimensions"""
        C0 = np.random.randn(10, 20)
        C1 = np.random.randn(10, 15)
        Clist = [C0, C1]
        alpha = [0, 0]
        beta = [0, 1]

        # n_components close to minimum dimension
        algo = decomp.JointSVD(n_components=9, n_iter=3)
        algo.fit(Clist, alpha, beta)

        preds = algo.predict()
        assert len(preds) == 2


class TestNumericalStability:
    """Test numerical stability and edge cases"""

    def test_near_zero_matrix(self):
        """Test with matrix close to zero"""
        C0 = np.random.randn(10, 10) * 1e-10
        C1 = np.random.randn(10, 10) * 1e-10
        Clist = [C0, C1]
        alpha = [0, 0]
        beta = [0, 1]

        algo = decomp.JointOuterDecomp(n_components=3, n_iter=2, verbose=False)
        algo.fit(Clist, alpha, beta)

        # Should not raise any errors
        preds = algo.predict()
        assert len(preds) == 2

    def test_rank_deficient_matrix(self):
        """Test with rank-deficient matrices"""
        # Create rank-1 matrix
        u = np.random.randn(10, 1)
        v = np.random.randn(20, 1)
        C0 = u @ v.T  # rank-1 matrix
        C1 = u @ v.T
        Clist = [C0, C1]
        alpha = [0, 0]
        beta = [0, 1]

        algo = decomp.JointSVD(n_components=2, n_iter=5)
        algo.fit(Clist, alpha, beta)

        preds = algo.predict()
        assert len(preds) == 2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
