from __future__ import print_function, division, absolute_import

from itertools import chain

import numpy as np

import regreg.atoms.block_norms as B
import regreg.atoms.svd_norms as SVD
import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

import nose.tools as nt

from .test_seminorms import Solver, all_close, SolverFactory

class MatrixSolverFactory(SolverFactory):

    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]
    shape = (5,4)

@set_seed_for_test(seed=20)
@np.testing.dec.slow
def test_proximal_maps():
    for klass in chain(B.conjugate_block_pairs.keys(),
                       SVD.conjugate_svd_pairs.keys()):
        if klass not in [B.block_max, B.block_sum]:
            factory = MatrixSolverFactory(klass, 'lagrange')
            for solver in factory:
                penalty = solver.atom
                dual = penalty.conjugate 
                Z = solver.prox_center
                L = solver.L

                yield all_close, penalty.lagrange_prox(Z, lipschitz=L), Z-dual.bound_prox(Z*L)/L, 'testing lagrange_prox and bound_prox starting from atom %s ' % klass, None

            # some arguments of the constructor

                yield nt.assert_raises, AttributeError, setattr, penalty, 'bound', 4.
                yield nt.assert_raises, AttributeError, setattr, dual, 'lagrange', 4.

                yield nt.assert_raises, AttributeError, setattr, penalty, 'bound', 4.
                yield nt.assert_raises, AttributeError, setattr, dual, 'lagrange', 4.

            for t in solver.all_tests():
                yield t

            factory = MatrixSolverFactory(klass, 'bound')
            for solver in factory:
                for t in solver.all_tests():
                    yield t


