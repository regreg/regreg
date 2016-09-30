from __future__ import print_function, division, absolute_import

import itertools
from copy import copy
import nose.tools as nt

import numpy as np

import regreg.atoms.seminorms as S
import regreg.api as rr
from regreg.tests.decorators import set_seed_for_test

def all_close(x, y, msg, solver):
    """
    Check to see if x and y are close
    """
    try:
        v = np.linalg.norm(x-y) <= 1.0e-03 * max([1, np.linalg.norm(x), np.linalg.norm(y)])
    except:
        print("""
check_failed
============
msg: %s
x: %s
y: %s
""" % (msg, x, y))
        return False
    v = v or np.allclose(x,y)
    if not v:
        print("""
summary
=======
msg: %s
comparison: %0.3f
x : %s
y : %s
""" % (msg, np.linalg.norm(x-y) / max([1, np.linalg.norm(x), np.linalg.norm(y)]), x, y))
    if not hasattr(solver, 'interactive') or not solver.interactive:
        nt.assert_true(v)
    else:
        print(msg.split('\n')[0])

@set_seed_for_test()
@np.testing.dec.slow
def test_proximal_maps(interactive=False):
    for klass in [S.l1norm, S.supnorm, S.l2norm,
                  S.positive_part, S.constrained_max]:
        factory = SolverFactory(klass, 'lagrange')
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

            # call these to ensure coverage at least

            repr(penalty)
            repr(dual)

            penalty.seminorm(Z, lagrange=1)
            penalty.constraint(Z, bound=1)

            dual.seminorm(Z, lagrange=1)
            dual.constraint(Z, bound=1)

            for t in solver.all_tests():
                yield t

        factory = SolverFactory(klass, 'bound')
        for solver in factory:
            for t in solver.all_tests():
                yield t

    for klass in sorted(S.nonpaired_atoms):
        factory = SolverFactory(klass, 'lagrange')
        for solver in factory:

            penalty = solver.atom
            dual = penalty.conjugate 
            Z = solver.prox_center
            L = solver.L

            yield all_close, penalty.lagrange_prox(Z, lipschitz=L), Z-dual.bound_prox(Z*L)/L, 'testing lagrange_prox and bound_prox starting from atom %s\n ' % klass, None

            nt.assert_raises(AttributeError, setattr, penalty, 'bound', 4.)
            nt.assert_raises(AttributeError, setattr, dual, 'lagrange', 4.)
        
            nt.assert_raises(AttributeError, setattr, penalty, 'bound', 4.)
            nt.assert_raises(AttributeError, setattr, dual, 'lagrange', 4.)

            for t in solver.all_tests():
                yield t

class SolverFactory(object):

    offset_choices = [True, False]
    FISTA_choices =[True,False]
    coef_stop_choices = [True,False]
    lagrange = 0.13
    bound = 0.14
    L_choices = [0.1,0.5,1]
    quadratic_choices = [True, False]
    shape = (20,)
    interactive = False

    def __init__(self, klass, mode):
        self.klass = klass
        self.mode = mode

    def __iter__(self):
        for offset, FISTA, coef_stop, L, q in itertools.product(self.offset_choices,
                                                                self.FISTA_choices,
                                                                self.coef_stop_choices,
                                                                self.L_choices,
                                                                self.quadratic_choices):
            self.FISTA = FISTA
            self.coef_stop = coef_stop
            self.L = L

            if self.mode == 'lagrange':
                atom = self.klass(self.shape, lagrange=self.lagrange)
            else:
                atom = self.klass(self.shape, bound=self.bound)

            if q: 
                atom.quadratic = rr.identity_quadratic(0,0,np.random.standard_normal(atom.shape)*0.02)

            if offset:
                atom.offset = 0.02 * np.random.standard_normal(atom.shape)

            solver = Solver(atom, interactive=self.interactive, 
                            coef_stop=coef_stop,
                            FISTA=FISTA,
                            L=L)

            # make sure certain lines of code are tested
            assert(atom == atom)
            atom.latexify(), atom.dual, atom.conjugate

            yield solver


class Solver(object):

    def __iter__(self):
        factory = SolverFactory(self.atom.__class__)
        for solver in factory:
            yield solver

    def __repr__(self):
        return 'Solver(%s, L=%f, prox_center=%s)' % (repr(self.atom), self.L, repr(self.prox_center))

    def __init__(self, atom, interactive=False, coef_stop=False,
                 FISTA=True, L=1, prox_center=None):
        self.atom = atom
        self.interactive = interactive
        self.coef_stop = coef_stop
        self.FISTA = FISTA
        self.L = L

        if prox_center is None:
            self.prox_center = np.random.standard_normal(atom.shape)
        else:
            self.prox_center = prox_center

        self.q = rr.identity_quadratic(L, self.prox_center, 0, 0)
        self.loss = rr.quadratic_loss.shift(self.prox_center, coef=L)

    @set_seed_for_test()
    def test_duality_of_projections(self):
        if self.atom.quadratic == rr.identity_quadratic(0,0,0,0) or self.atom.quadratic is None:

            tests = []

            d = self.atom.conjugate
            q = rr.identity_quadratic(1, self.prox_center, 0, 0)
            tests.append((self.prox_center-self.atom.proximal(q), d.proximal(q), 'testing duality of projections starting from atom\n %s ' % str(self)))

            if hasattr(self.atom, 'check_subgradient') and self.atom.offset is None:
                # check subgradient condition
                v1, v2 = self.atom.check_subgradient(self.atom, self.prox_center)
                tests.append((v1, v2, 'checking subgradient condition\n %s' % str(self)))

            if not self.interactive:
                for test in tests:
                    yield (all_close,) + test + (self,)
            else:
                for test in tests:
                    yield all_close(*((test + (self,))))

    @set_seed_for_test()
    def test_simple_problem_nonsmooth(self):
        tests = []
        atom, q = self.atom, self.q
        loss = self.loss

        p2 = copy(atom)
        p2.quadratic = atom.quadratic + q
        problem = rr.simple_problem.nonsmooth(p2)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-14, FISTA=self.FISTA, coef_stop=self.coef_stop, min_its=100)

        gg = rr.gengrad(problem, 2.) # this lipschitz constant is based on knowing our loss...
        tests.append((atom.proximal(q), gg, 'solving prox with gengrad\n %s ' % str(self)))

        tests.append((atom.proximal(q), atom.solve(q), 'solving prox with solve method\n %s ' % str(self)))

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with monotonicity\n %s ' % str(self)))

        # use the solve method

        p3 = copy(atom)
        p3.quadratic = atom.quadratic + q
        soln = p3.solve(tol=1.e-14, min_its=10)
        tests.append((atom.proximal(q), soln, 'solving prox with solve method\n %s ' % str(self)))

        p4 = copy(atom)
        p4.quadratic = atom.quadratic + q
        problem = rr.simple_problem.nonsmooth(p4)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-14, monotonicity_restart=False, coef_stop=self.coef_stop,
                   FISTA=self.FISTA,
                   min_its=100)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem.nonsmooth with no monotonocity\n %s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    @set_seed_for_test()
    def test_simple_problem(self):
        tests = []
        atom, q, prox_center, L = self.atom, self.q, self.prox_center, self.L
        loss = self.loss

        problem = rr.simple_problem(loss, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=self.FISTA, coef_stop=self.coef_stop, min_its=100)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem with monotonicity\n %s' % str(self)))

        # write the loss in terms of a quadratic for the smooth loss and a smooth function...

        q = rr.identity_quadratic(L, prox_center, 0, 0)
        lossq = rr.quadratic_loss.shift(prox_center.copy(), coef=0.6*L)
        lossq.quadratic = rr.identity_quadratic(0.4*L, prox_center.copy(), 0, 0)
        problem = rr.simple_problem(lossq, atom)

        tests.append((atom.proximal(q), 
              problem.solve(coef_stop=self.coef_stop, 
                            FISTA=self.FISTA, 
                            tol=1.0e-12), 
               'solving prox with simple_problem ' +
               'with monotonicity  but loss has identity_quadratic %s\n ' % str(self)))

        problem = rr.simple_problem(loss, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, monotonicity_restart=False,
                   coef_stop=self.coef_stop, FISTA=self.FISTA, min_its=100)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving prox with simple_problem no monotonicity_restart\n %s' % str(self)))

        d = atom.conjugate
        problem = rr.simple_problem(loss, d)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, monotonicity_restart=False, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA, min_its=100)
        tests.append((d.proximal(q), problem.solve(tol=1.e-12,
                                                FISTA=self.FISTA,
                                                coef_stop=self.coef_stop,
                                                monotonicity_restart=False), 
               'solving dual prox with simple_problem no monotonocity\n %s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    @set_seed_for_test()
    def test_dual_problem(self):
        tests = []
        atom, q, prox_center, L = self.atom, self.q, self.prox_center, self.L
        loss = self.loss

        dproblem = rr.dual_problem.fromprimal(loss, atom)
        dcoef = dproblem.solve(coef_stop=self.coef_stop, tol=1.0e-14)
        tests.append((atom.proximal(q), dcoef, 'solving prox with dual_problem.fromprimal with monotonicity \n %s ' % str(self)))

        dproblem2 = rr.dual_problem(loss.conjugate, 
                                    rr.identity(loss.shape),
                                    atom.conjugate)
        dcoef2 = dproblem2.solve(coef_stop=self.coef_stop, tol=1.e-14)
        tests.append((atom.proximal(q), dcoef2, 'solving prox with dual_problem with monotonicity %s \n' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    @set_seed_for_test()
    def test_separable(self):
        tests = []
        atom, q, prox_center, L = self.atom, self.q, self.prox_center, self.L
        loss = self.loss

        problem = rr.separable_problem.singleton(atom, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA, min_its=100)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving atom prox with separable_atom.singleton \n%s ' % str(self)))


        d = atom.conjugate
        problem = rr.separable_problem.singleton(d, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA, min_its=100)

        tests.append((d.proximal(q), solver.composite.coefs, 'solving dual atom prox with separable_atom.singleton \n%s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

    @set_seed_for_test()
    def test_container(self):
        tests = []
        atom, q, prox_center, L = self.atom, self.q, self.prox_center, self.L
        loss = self.loss

        problem = rr.container(loss, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA, min_its=100)

        tests.append((atom.proximal(q), solver.composite.coefs, 'solving atom prox with container\n %s ' % str(self)))

        # write the loss in terms of a quadratic for the smooth loss and a smooth function...

        q = rr.identity_quadratic(L, prox_center, 0, 0)
        lossq = rr.quadratic_loss.shift(prox_center.copy(), coef=0.6*L)
        lossq.quadratic = rr.identity_quadratic(0.4*L, prox_center.copy(), 0, 0)
        problem = rr.container(lossq, atom)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, FISTA=self.FISTA, coef_stop=self.coef_stop)

        tests.append((atom.proximal(q), 
                      problem.solve(tol=1.e-12,FISTA=self.FISTA,coef_stop=self.coef_stop), 
                      'solving prox with container with monotonicity ' + 
                      'but loss has identity_quadratic\n %s ' % str(self)))

        d = atom.conjugate
        problem = rr.container(d, loss)
        solver = rr.FISTA(problem)
        solver.fit(tol=1.0e-12, 
                   coef_stop=self.coef_stop, FISTA=self.FISTA, min_its=100)
        tests.append((d.proximal(q), solver.composite.coefs, 'solving dual prox with container\n %s ' % str(self)))

        if not self.interactive:
            for test in tests:
                yield (all_close,) + test + (self,)
        else:
            for test in tests:
                yield all_close(*((test + (self,))))

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
