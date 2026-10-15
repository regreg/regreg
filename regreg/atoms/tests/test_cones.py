import pytest
import itertools

import numpy as np

import regreg.api as rr
import regreg.atoms.cones as C
import regreg.atoms.svd_norms as C_SVD


from .test_seminorms import Solver, SolverFactory, run_checks


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
                atom.quadratic = rr.identity_quadratic(0,0,self.rng.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * self.rng.standard_normal(atom.shape)

            solver = Solver(atom, rng=self.rng, interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)
            yield solver


_svd_cones = [C_SVD.nuclear_norm_epigraph,
              C_SVD.nuclear_norm_epigraph_polar,
              C_SVD.operator_norm_epigraph,
              C_SVD.operator_norm_epigraph_polar]

@pytest.mark.slow
@pytest.mark.parametrize('klass', sorted(C.conjugate_cone_pairs.keys(), key=str), ids=lambda k: k.__name__)
def test_proximal_maps(klass):
    factory = ConeSolverFactory(klass)
    if klass in _svd_cones:
        factory.shape = (5,4)
    else:
        factory.shape = 20
    for solver in factory:
        run_checks(solver.all_tests())
