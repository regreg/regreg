import itertools

import numpy as np

import regreg.api as rr
import regreg.atoms.simplex as S
from regreg.tests.decorators import set_seed_for_test

import nose.tools as nt

from .test_cones import Solver, ConeSolverFactory


class SimplexSolverFactory(ConeSolverFactory):

    FISTA_choices = [True]
    L_choices = [1, 0.3]
    coef_stop_choices = [False]
    shape = 5
    
    def __init__(self, klass):
        self.klass = klass

    def __iter__(self):
        for offset, FISTA, coef_stop, L, q in itertools.product(self.offset_choices,
                                                                self.FISTA_choices,
                                                                self.coef_stop_choices,
                                                                self.L_choices,
                                                                self.quadratic_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            atom = self.klass(self.shape)

            # make sure certain lines of code are tested
            assert(atom == atom)
            atom.latexify(), atom.dual, atom.conjugate

            if q: 
                atom.quadratic = rr.identity_quadratic(0,0,np.random.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * np.random.standard_normal(atom.shape)

            solver = SimplexSolver(atom,
                                   interactive=self.interactive, 
                                   coef_stop=coef_stop,
                                   FISTA=FISTA,
                                   L=L)
            yield solver

class SimplexSolver(Solver):

    def all_tests(self):
        for group in [self.test_duality_of_projections,
                      self.test_simple_problem,
                      self.test_separable,
                      self.test_dual_problem,
                      self.test_container,
                      self.test_simple_problem_nonsmooth
                      ]:
            for t in group():
                yield t

@set_seed_for_test()
@np.testing.dec.slow
def test_proximal_maps():
    for klass in sorted(S.conjugate_simplex_pairs.keys(), key=str):
        factory = SimplexSolverFactory(klass)
        for solver in factory:
            for t in solver.all_tests():
                yield t
