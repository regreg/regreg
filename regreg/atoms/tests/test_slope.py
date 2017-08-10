import itertools
import numpy as np
import nose.tools as nt

import regreg.api as rr
from regreg.atoms.slope import slope, slope_conjugate, have_sklearn_iso
from regreg.tests.decorators import set_seed_for_test

from .test_seminorms import Solver, SolverFactory, all_close

def test_kkt():

    Z = np.array([0.75529996,
                  0.87948569,
                  -0.2399563,
                  -0.69505665,
                  -0.47140008,
                  0.34947641,
                  1.61145729,
                  0.23933927,
                  0.48500263,
                  -1.65860471,
                  2.67991031,
                  -1.10352213,
                  1.81778261,
                  -0.55604734,
                  0.95771124,
                  0.81522068,
                  0.0643194,
                  -0.16640537,
                  -0.94070977,
                  -0.0535945]) 

    o = np.argsort(np.fabs(Z))
    Z = Z[o][::-1]
    pen1 = slope(0.14 * np.arange(20)[::-1] / 20 + .14, lagrange=1.)
    pen1.check_subgradient(pen1, Z)

def test_l1norm_equivalent():
    """
    with equal weights the prox is the same as l1 norm
    """
    pen1 = slope(np.ones(3) * 4, lagrange=0.3)
    pen2 = rr.l1norm(3, lagrange=1.2)

    Z = np.array([3,2,4])
    np.testing.assert_allclose(pen1.lagrange_prox(Z), pen2.lagrange_prox(Z))

    Z = np.random.standard_normal(3) * 100
    np.testing.assert_allclose(pen1.lagrange_prox(Z), pen2.lagrange_prox(Z))

    dual1 = pen1.conjugate
    dual2 = pen2.conjugate

    np.testing.assert_allclose(Z, pen1.lagrange_prox(Z) + dual1.bound_prox(Z))
    np.testing.assert_allclose(dual1.bound_prox(Z), dual2.bound_prox(Z))

def test_duality():
    """
    with equal weights the prox is the same as l1 norm
    """
    pen1 = slope(np.array([0.4,0.35,0.1]), lagrange=1.1)
    dual1 = pen1.conjugate
    Z = np.random.standard_normal(3) * 100
    np.testing.assert_allclose(Z, pen1.lagrange_prox(Z) + dual1.bound_prox(Z))

class SlopeSolverFactory(SolverFactory):

    weight_choices = [np.arange(20)[::-1]/20. + 1]
    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]

    def __init__(self, klass, mode, use_sklearn=True):
        self.klass = klass
        self.mode = mode
        self.use_sklearn = use_sklearn

    def __iter__(self):
        for offset, FISTA, coef_stop, L, q, w in itertools.product(self.offset_choices,
                                                                   self.FISTA_choices,
                                                                   self.coef_stop_choices,
                                                                   self.L_choices,
                                                                   self.quadratic_choices,
                                                                   self.weight_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            if self.mode == 'lagrange':
                atom = self.klass(w, lagrange=self.lagrange)
            else:
                atom = self.klass(w, bound=self.bound)
            atom.use_sklearn = self.use_sklearn and have_sklearn_iso # test out both prox maps if available

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
    for klass, mode in zip([slope, slope_conjugate], ['lagrange', 'bound']):
        factory = SlopeSolverFactory(klass, mode)
        for solver in factory:
            for t in solver.all_tests():
                yield t
        factory = SlopeSolverFactory(klass, mode, use_sklearn=False)
        for solver in factory:
            for t in solver.all_tests():
                yield t


