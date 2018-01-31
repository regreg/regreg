import itertools
import numpy as np
import nose.tools as nt

import regreg.api as rr
from regreg.atoms.sparse_group_lasso import (sparse_group_lasso,
                                             sparse_group_lasso_dual)
from regreg.tests.decorators import set_seed_for_test

from .test_seminorms import Solver, SolverFactory, all_close

def test_l1norm_equivalent():
    """
    with equal weights the prox is the same as l1 norm
    """
    pen1 = sparse_group_lasso([1,1,2,2,2], {1:0, 2:0},
                              np.ones(5), lagrange=0.4)
    pen2 = rr.l1norm(5, lagrange=0.4)

    Z = np.array([3,2,4,6,7])
    np.testing.assert_allclose(pen1.lagrange_prox(Z), pen2.lagrange_prox(Z))

    Z = np.random.standard_normal(5) * 100
    np.testing.assert_allclose(pen1.lagrange_prox(Z), pen2.lagrange_prox(Z))

    dual1 = pen1.conjugate
    dual2 = pen2.conjugate

    np.testing.assert_allclose(Z, pen1.lagrange_prox(Z) + dual1.bound_prox(Z))
    np.testing.assert_allclose(dual1.bound_prox(Z), dual2.bound_prox(Z))

def test_group_lasso_equivalent():
    """
    with 0 as lasso weights should be group lasso
    """
    pen1 = sparse_group_lasso([1,1,2,2,2], {1:0.2, 2:0.1},
                              np.zeros(5), lagrange=0.4)
    pen2 = rr.group_lasso([1,1,2,2,2], {1:0.2, 2:0.1}, lagrange=0.4)

    Z = np.array([3,2,4,6,7])
    np.testing.assert_allclose(pen1.lagrange_prox(Z), pen2.lagrange_prox(Z))

    Z = np.random.standard_normal(5) * 100
    np.testing.assert_allclose(pen1.lagrange_prox(Z), pen2.lagrange_prox(Z))

    dual1 = pen1.conjugate
    dual2 = pen2.conjugate

    np.testing.assert_allclose(Z, pen1.lagrange_prox(Z) + dual1.bound_prox(Z))
    np.testing.assert_allclose(dual1.bound_prox(Z), dual2.bound_prox(Z))

class SlopeSolverFactory(SolverFactory):

    groups = [[0]*12 + [1]*8]
    weights = [{0:0.1,1:0.04}]
    lasso_weights = [np.linspace(0, 1, 20), np.zeros(20)]
    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]

    def __init__(self, klass, mode, use_sklearn=True):
        self.klass = klass
        self.mode = mode
        self.use_sklearn = use_sklearn

    def __iter__(self):
        pen_choices = itertools.product(self.weights,
                                        self.lasso_weights,
                                        self.groups)
        for offset, FISTA, coef_stop, L, q, pen in itertools.product(self.offset_choices,
                                                                     self.FISTA_choices,
                                                                     self.coef_stop_choices,
                                                                     self.L_choices,
                                                                     self.quadratic_choices,
                                                                     pen_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            w, l, g = pen
            if self.mode == 'lagrange':
                atom = self.klass(g, w, l, lagrange=self.lagrange)
            else:
                atom = self.klass(g, w, l, bound=self.bound)

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
    for klass, mode in zip([sparse_group_lasso, 
                            sparse_group_lasso_dual], 
                           ['lagrange', 'bound']):
        factory = SlopeSolverFactory(klass, mode)
        for solver in factory:
            for t in solver.all_tests():
                yield t
        factory = SlopeSolverFactory(klass, mode, use_sklearn=False)
        for solver in factory:
            for t in solver.all_tests():
                yield t


