from __future__ import print_function, division, absolute_import

import pytest

import itertools

import numpy as np

import regreg.api as rr
import regreg.atoms.weighted_atoms as WA

from numpy import testing as npt

from .test_seminorms import Solver, SolverFactory, run_checks

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
        for (offset,
             FISTA,
             coef_stop,
             L,
             q,
             w) in itertools.product(self.offset_choices,
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
                atom.quadratic = rr.identity_quadratic(0,
                                                       0,
                                                       self.rng.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * self.rng.standard_normal(atom.shape)

            solver = Solver(atom, rng=self.rng,
                            interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)
            yield solver

@pytest.mark.slow
@pytest.mark.parametrize('mode', ['lagrange', 'bound'])
@pytest.mark.parametrize('klass', [WA.l1norm, WA.supnorm], ids=lambda k: k.__name__)
def test_proximal_maps(klass, mode):
    factory = WeightedSolverFactory(klass, mode)
    for solver in factory:
        run_checks(solver.all_tests())

def test_weighted_l1():
    rng = np.random.default_rng(100)
    a =rr.weighted_l1norm(2*np.ones(10), lagrange=0.5)
    b= rr.l1norm(10, lagrange=1)
    z = rng.standard_normal(10)
    npt.assert_equal(b.lagrange_prox(z), a.lagrange_prox(z))
    npt.assert_equal(b.dual[1].bound_prox(z), a.dual[1].bound_prox(z))

def test_weighted_l1_bound():
    rng = np.random.default_rng(101)
    a =rr.weighted_supnorm(2*np.ones(10), bound=2)
    b= rr.supnorm(10, bound=1)
    z = rng.standard_normal(10)
    npt.assert_equal(b.bound_prox(z), a.bound_prox(z))
    npt.assert_equal(b.dual[1].lagrange_prox(z), a.dual[1].lagrange_prox(z))

def test_weighted_l1_with_zero():
    rng = np.random.default_rng(102)
    z = rng.standard_normal(5)
    a=rr.weighted_l1norm([0,1,1,1,1], lagrange=0.5)
    b=a.dual[1]
    c=rr.l1norm(4, lagrange=0.5)
    npt.assert_equal(a.lagrange_prox(z), z-b.bound_prox(z))
    npt.assert_equal(a.lagrange_prox(z)[0], z[0])
    npt.assert_equal(a.lagrange_prox(z)[1:], c.lagrange_prox(z[1:]))

def test_weighted_l1_bound_loose():
    rng = np.random.default_rng(103)
    n, p = 100, 10
    X = rng.standard_normal((n, p))
    Y = rng.standard_normal(n)
    beta = np.linalg.pinv(X).dot(Y)
    bound = 2 * np.fabs(beta).sum()
    atom = rr.weighted_l1norm(np.ones(p), bound=bound)
    loss = rr.squared_error(X, Y)
    problem = rr.simple_problem(loss, atom)
    soln = problem.solve(tol=1.e-12, min_its=100)
    npt.assert_allclose(soln, beta)
    
