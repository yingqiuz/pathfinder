#!/usr/bin/env python

# decomp.py - Matrix Decomposition Classes
#
# Author: 
# Akina Ying-Qiu Zheng <ying-qiu.zheng@ndcn.ox.ac.uk>
# Saad Jbabdi <saad@fmrib.ox.ac.uk>
#
# Copyright (C) 2024 University of Oxford
# SHBASECOPYRIGHT

""" Decomposition classes - Interface

All decomposition classes share a common interface:

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

The implementations live in separate modules; this module re-exports them so
that `from pathfinder.decomp import JointOuterDecomp, JointSVD` and
`decomp.JointOuterDecomp` keep working:
    pathfinder.joint_outer_decomp.JointOuterDecomp
    pathfinder.joint_svd.JointSVD
"""

from pathfinder.joint_outer_decomp import JointOuterDecomp
from pathfinder.joint_svd import JointSVD

__all__ = ['JointOuterDecomp', 'JointSVD']
