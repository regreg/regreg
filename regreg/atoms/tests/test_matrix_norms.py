from __future__ import print_function, division, absolute_import

import pytest

from itertools import chain

import numpy as np

import regreg.atoms.block_norms as B
import regreg.atoms.svd_norms as SVD
import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test


from .test_seminorms import Solver, all_close, SolverFactory, run_checks, check_lagrange_bound_duality

class MatrixSolverFactory(SolverFactory):

    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]
    shape = (5,4)

@pytest.mark.slow
@pytest.mark.parametrize('klass',
                         [k for k in chain(B.conjugate_block_pairs.keys(),
                                           SVD.conjugate_svd_pairs.keys())
                          if k not in [B.block_max, B.block_sum]],
                         ids=lambda k: k.__name__)
@set_seed_for_test(seed=20)
def test_proximal_maps(klass):
    factory = MatrixSolverFactory(klass, 'lagrange')
    for solver in factory:
        check_lagrange_bound_duality(solver, klass)
        run_checks(solver.all_tests())

    factory = MatrixSolverFactory(klass, 'bound')
    for solver in factory:
        run_checks(solver.all_tests())
