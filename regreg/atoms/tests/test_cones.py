import itertools

import numpy as np

import regreg.api as rr
import regreg.atoms.cones as C
import regreg.atoms.svd_norms as C_SVD
from regreg.tests.decorators import set_seed_for_test

import nose.tools as nt

from .test_seminorms import Solver, SolverFactory


class ConeSolverFactory(SolverFactory):

    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]

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

            solver = Solver(atom, interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)
            yield solver


@set_seed_for_test()
@np.testing.dec.slow
def test_proximal_maps():
    for klass in sorted(C.conjugate_cone_pairs.keys(), key=str):
        if klass in [C_SVD.nuclear_norm_epigraph,
                     C_SVD.nuclear_norm_epigraph_polar,
                     C_SVD.operator_norm_epigraph,
                     C_SVD.operator_norm_epigraph_polar]:
            ConeSolverFactory.shape = (5,4)
        else:
            ConeSolverFactory.shape = 20
        factory = ConeSolverFactory(klass)
        for solver in factory:
            for t in solver.all_tests():
                yield t
