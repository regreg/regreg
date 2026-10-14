from __future__ import print_function, division, absolute_import

import pytest

import itertools

import numpy as np

import regreg.atoms.linear_constraints as LC
import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test
from .test_seminorms import run_checks


from .test_cones import Solver, ConeSolverFactory

class ConstraintSolverFactory(ConeSolverFactory):

    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]
    shape = (20,)

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

            basis = np.linalg.svd(np.random.standard_normal((4,) + self.shape), full_matrices=0)[2]
            atom = self.klass(self.shape, basis)

            if q: 
                atom.quadratic = rr.identity_quadratic(0,0,np.random.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * np.random.standard_normal(atom.shape)

            solver = Solver(atom, interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)
            yield solver

@pytest.mark.slow
@pytest.mark.parametrize('klass', sorted(LC.conjugate_cone_pairs.keys(), key=str), ids=lambda k: k.__name__)
@set_seed_for_test()
def test_proximal_maps(klass):
    factory = ConstraintSolverFactory(klass)
    for solver in factory:
        run_checks(solver.all_tests())
