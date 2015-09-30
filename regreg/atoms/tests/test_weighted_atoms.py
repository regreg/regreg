from __future__ import print_function, division, absolute_import

import itertools

import numpy as np

import regreg.api as rr
import regreg.atoms.weighted_atoms as WA
from regreg.tests.decorators import set_seed_for_test

from numpy import testing as npt

from .test_seminorms import Solver, SolverFactory

w1 = np.ones(20) * 0.5
w2 = w1 * 0
w2[:10] = 2.

class WeightedSolverFactory(SolverFactory):

    weight_choices = [w1, w2]
    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]

    def __init__(self, klass, mode):
        self.klass = klass
        self.mode = mode

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

            if q: 
                atom.quadratic = rr.identity_quadratic(0,0,np.random.standard_normal(atom.shape)*0.02)

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
    for klass, mode in zip([WA.l1norm, WA.supnorm], ['lagrange', 'bound']):
        factory = WeightedSolverFactory(klass, mode)
        for solver in factory:
            for t in solver.all_tests():
                yield t

@set_seed_for_test()
def test_weighted_l1():
    a =rr.weighted_l1norm(2*np.ones(10), lagrange=0.5)
    b= rr.l1norm(10, lagrange=1)
    z = np.random.standard_normal(10)
    npt.assert_equal(b.lagrange_prox(z), a.lagrange_prox(z))
    npt.assert_equal(b.dual[1].bound_prox(z), a.dual[1].bound_prox(z))

@set_seed_for_test()
def test_weighted_l1_with_zero():
    z = np.random.standard_normal(5)
    a=rr.weighted_l1norm([0,1,1,1,1], lagrange=0.5)
    b=a.dual[1]
    c=rr.l1norm(4, lagrange=0.5)
    npt.assert_equal(a.lagrange_prox(z), z-b.bound_prox(z))
    npt.assert_equal(a.lagrange_prox(z)[0], z[0])
    npt.assert_equal(a.lagrange_prox(z)[1:], c.lagrange_prox(z[1:]))
