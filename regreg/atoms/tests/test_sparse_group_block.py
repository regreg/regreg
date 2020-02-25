import itertools
import numpy as np
import nose.tools as nt

import regreg.api as rr
from ..sparse_group_block import (sparse_group_block,
                                             sparse_group_block_dual,
                                             _inside_set,
                                             _gauge_function_dual)
from ..sparse_group_lasso import sparse_group_lasso
from regreg.tests.decorators import set_seed_for_test

from .test_seminorms import Solver, SolverFactory, all_close

def test_sparse_group_lasso_equivalent():
    """
    check agrees with equivalent group lasso
    """

    l1_weight = 1.
    l2_weight = 2.
    lagrange = 0.5
    pen = sparse_group_block((4, 5), l1_weight, l2_weight, lagrange=lagrange)
    pen2 = sparse_group_lasso(np.multiply.outer(np.arange(4), np.ones(5)).reshape(-1).astype(np.int),
                              l1_weight,
                              weights=dict([(j, l2_weight) for j in range(4)]),
                              lagrange=lagrange)

    Z = np.random.standard_normal(pen.shape) * 2
    np.testing.assert_allclose(pen.lagrange_prox(Z), pen2.lagrange_prox(Z.reshape(-1)).reshape(Z.shape))

    dual = pen.conjugate
    dual2 = pen2.conjugate
    np.testing.assert_allclose(Z, pen.lagrange_prox(Z) + dual.bound_prox(Z))
    np.testing.assert_allclose(dual.bound_prox(Z), dual2.bound_prox(Z.reshape(-1)).reshape(Z.shape))

def test_inside_set():
    """
    with 0 as lasso weights should be group lasso
    """
    l1_weight, l2_weight = 1, 2
    pen = sparse_group_block((4, 5),
                             l1_weight, 
                             l2_weight,
                             lagrange=0.4)
    point = np.zeros(pen.shape)
    assert _inside_set(point, pen.lagrange, l1_weight, l2_weight)

    Z = np.random.standard_normal(pen.shape) * 20
    proxZ = pen.lagrange_prox(Z)
    assert _inside_set(Z - proxZ, pen.lagrange, l1_weight, l2_weight)  # its gauge norm is 0.4, larger than 1

    assert np.fabs(_gauge_function_dual(Z - proxZ, l1_weight, l2_weight) - pen.lagrange) < 1.e-4

    pen2 = sparse_group_block((4,5),
                              l1_weight, 
                              l2_weight,
                              lagrange=1.2)
    point = np.zeros(pen2.shape)
    assert _inside_set(point, pen.lagrange, l1_weight, l2_weight) 

    Z = np.random.standard_normal(pen.shape) * 20
    proxZ = pen2.lagrange_prox(Z)
    print(Z, proxZ)
    assert np.fabs(_gauge_function_dual(Z - proxZ, l1_weight, l2_weight) - 1.2) < 1.e-4
    assert not _inside_set(Z - proxZ, 1., l1_weight, l2_weight) # its gauge norm is 1.2, larger than 1

class SparseBlockSolverFactory(SolverFactory):

    shape = (4, 5)
    l1_weight = 1.
    l2_weight = 2.
    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]

    def __init__(self, klass, mode):
        self.klass = klass
        self.mode = mode

    def __iter__(self):
        for offset, FISTA, coef_stop, L, q in itertools.product(self.offset_choices,
                                                                self.FISTA_choices,
                                                                self.coef_stop_choices,
                                                                self.L_choices,
                                                                self.quadratic_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            if self.mode == 'lagrange':
                atom = self.klass(self.shape,
                                  self.l1_weight,
                                  self.l2_weight, lagrange=self.lagrange)
            else:
                atom = self.klass(self.shape,
                                  self.l1_weight,
                                  self.l2_weight, bound=self.bound)

            if q: 
                atom.quadratic = rr.identity_quadratic(0, 0, np.random.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * np.random.standard_normal(atom.shape)

            solver = Solver(atom, interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)
            yield solver

@set_seed_for_test()
@np.testing.dec.slow
def test_proximal_maps():
    for klass, mode in zip([sparse_group_block,
                            sparse_group_block_dual], 
                           ['lagrange', 'bound']):
        factory = SparseBlockSolverFactory(klass, mode)
        for solver in factory:
            for t in solver.all_tests():
                yield t


