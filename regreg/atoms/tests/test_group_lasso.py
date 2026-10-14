from __future__ import print_function, division, absolute_import

import pytest

import itertools

import numpy as np

import regreg.atoms.group_lasso as GL
import regreg.api as rr

from .test_seminorms import Solver, all_close, SolverFactory, run_checks, check_lagrange_bound_duality
from .test_cones import ConeSolverFactory
from regreg.tests.decorators import set_seed_for_test

class GroupSolverFactory(SolverFactory):

    group_choices = [np.arange(10),
                     np.array([1,1,2,2,2,3,3,4,4,4,4,5,5,6,6,6,6])]
    weight_choices = [{}, {}]

    FISTA_choices = [True]
    L_choices = [0.3]
    coef_stop_choices = [False]

    def __init__(self, klass, mode):
        self.klass = klass
        self.mode = mode

    def __iter__(self):
        for offset, FISTA, coef_stop, L, q, groups, w in itertools.product(self.offset_choices,
                                                                           self.FISTA_choices,
                                                                           self.coef_stop_choices,
                                                                           self.L_choices,
                                                                           self.quadratic_choices,
                                                                           self.group_choices,
                                                                           self.weight_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            if self.mode == 'lagrange':
                atom = self.klass(groups, w, lagrange=self.lagrange)
            else:
                atom = self.klass(groups, w, bound=self.bound)

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
    weight_choices = [{}, {}]

    def __iter__(self):
        for offset, FISTA, coef_stop, L, q, groups, w in itertools.product(self.offset_choices,
                                                                           self.FISTA_choices,
                                                                           self.coef_stop_choices,
                                                                           self.L_choices,
                                                                           self.quadratic_choices,
                                                                           self.group_choices,
                                                                           self.weight_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            atom = self.klass(groups, w)

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
@pytest.mark.parametrize('klass', list(GL.conjugate_seminorm_pairs.keys()), ids=lambda k: k.__name__)
@set_seed_for_test()
def test_proximal_maps(klass):
    factory = GroupSolverFactory(klass, 'lagrange')
    for solver in factory:
        check_lagrange_bound_duality(solver, klass)
        run_checks(solver.all_tests())

    factory = GroupSolverFactory(klass, 'bound')
    for solver in factory:
        run_checks(solver.all_tests())

@pytest.mark.slow
@pytest.mark.parametrize('klass', list(GL.conjugate_cone_pairs.keys()), ids=lambda k: k.__name__)
@set_seed_for_test()
def test_proximal_maps_cones(klass):
    factory = GroupConeSolverFactory(klass)
    for solver in factory:
        run_checks(solver.all_tests())
