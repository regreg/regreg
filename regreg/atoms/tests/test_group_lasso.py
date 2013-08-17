
import numpy as np
import itertools
from copy import copy

import regreg.atoms.group_lasso as GL
import regreg.api as rr
import nose.tools as nt

from test_seminorms import Solver, all_close, SolverFactory
from test_cones import ConeSolverFactory

class GroupSolverFactory(SolverFactory):

    group_choices = [np.arange(10),
                     np.array([1,1,2,2,2,3,3,4,4,4,4,5,5,6,6,6,6])]
    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]

    def __init__(self, klass, mode):
        self.klass = klass
        self.mode = mode

    def __iter__(self):
        for offset, FISTA, coef_stop, L, q, groups in itertools.product(self.offset_choices,
                                                                        self.FISTA_choices,
                                                                        self.coef_stop_choices,
                                                                        self.L_choices,
                                                                        self.quadratic_choices,
                                                                        self.group_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            if self.mode == 'lagrange':
                atom = self.klass(groups, lagrange=self.lagrange)
            else:
                atom = self.klass(groups, bound=self.bound)

            if q: 
                atom.quadratic = rr.identity_quadratic(0,0,np.random.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * np.random.standard_normal(atom.shape)

            solver = Solver(atom, interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)
            yield solver

class GroupConeSolverFactory(ConeSolverFactory):

    group_choices = [np.arange(10),
                     np.array([1,1,2,2,2,3,3,4,4,4,4,5,5,6,6,6,6])]

    def __iter__(self):
        for offset, FISTA, coef_stop, L, q, groups in itertools.product(self.offset_choices,
                                                                        self.FISTA_choices,
                                                                        self.coef_stop_choices,
                                                                        self.L_choices,
                                                                        self.quadratic_choices,
                                                                        self.group_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            atom = self.klass(groups)

            if q: 
                atom.quadratic = rr.identity_quadratic(0,0,np.random.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * np.random.standard_normal(atom.shape)

            solver = Solver(atom, interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)
            yield solver

@np.testing.dec.slow
def test_proximal_maps(interactive=False):
    for klass in GL.conjugate_seminorm_pairs.keys(): 
        factory = GroupSolverFactory(klass, 'lagrange')
        for solver in factory:
            penalty = solver.atom
            dual = penalty.conjugate 
            Z = solver.prox_center
            L = solver.L

            yield all_close, penalty.lagrange_prox(Z, lipschitz=L), Z-dual.bound_prox(Z*L)/L, 'testing lagrange_prox and bound_prox starting from atom\n %s ' % klass, None

            # some arguments of the constructor

            nt.assert_raises(AttributeError, setattr, penalty, 'bound', 4.)
            nt.assert_raises(AttributeError, setattr, dual, 'lagrange', 4.)
        
            nt.assert_raises(AttributeError, setattr, penalty, 'bound', 4.)
            nt.assert_raises(AttributeError, setattr, dual, 'lagrange', 4.)

            for t in solver.all_tests():
                yield t

        factory = GroupSolverFactory(klass, 'bound')
        for solver in factory:
            for t in solver.all_tests():
                yield t

    for klass in GL.conjugate_cone_pairs.keys():
        factory = GroupConeSolverFactory(klass)
        for solver in factory:
            for t in solver.all_tests():
                yield t

