"""
A container combining several smooth and nonsmooth atoms into one problem.
"""

import numpy as np
from scipy import sparse

from ..algorithms import FISTA
from ..problems.composite import (composite, nonsmooth as nonsmooth_composite,
                        smooth as smooth_composite)
from ..affine import (vstack as afvstack, identity as afidentity, power_L,
                     selector as afselector)
from ..problems.separable import separable
from ..problems.dual_problem import dual_problem, stacked_dual
from ..atoms import affine_atom as nonsmooth_affine_atom
from ..atoms.cones import zero_constraint, zero as zero_nonsmooth, affine_cone

from ..identity_quadratic import identity_quadratic

class container(composite):
    """
    A container class for storing/combining atoms.

    The objective is the sum of the smooth and nonsmooth atoms. The proximal
    map is computed by solving a dual problem (see `dual_problem`) unless
    the combined transform is an identity or a selector.

    Parameters
    ----------
    atoms : sequence
        Smooth composites (`regreg.problems.composite.smooth`) and nonsmooth
        atoms (`regreg.problems.composite.nonsmooth`, affine atoms or affine
        cones). At least one smooth atom is required.
    """

    def __init__(self, *atoms, **keywords):
        self.nonsmooth_atoms = []
        self.smooth_atoms = []
        for atom in atoms:
            if (isinstance(atom, nonsmooth_composite) or 
                isinstance(atom, nonsmooth_affine_atom) or
                isinstance(atom, affine_cone)):
                self.nonsmooth_atoms.append(atom)
            elif isinstance(atom, smooth_composite):
                self.smooth_atoms.append(atom)
            else:
                raise ValueError('each atom should either be a smooth or nonsmooth composite')

        if len(self.nonsmooth_atoms) == 0 and len(self.smooth_atoms) == 0:
            raise ValueError('must specify some atoms')

        if len(self.nonsmooth_atoms) == 0:
            self.nonsmooth_atoms = [zero_nonsmooth(self.smooth_atoms[0].shape)]

        self.transform, self.atom = stacked_dual(self.smooth_atoms[0].shape, *self.nonsmooth_atoms)
        self.coefs = np.zeros(self.transform.input_shape)

        self.smoothq = identity_quadratic(0,0,0,0)
        for atom in self.smooth_atoms:
            self.smoothq += atom.quadratic

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Sum of the smooth objectives of the smooth atoms.

        The smooth_objective DOES NOT INCLUDE the identity
        quadratic of all the smooth atoms.

        Parameters
        ----------
        x : ndarray
            Point at which to evaluate.
        mode : str, optional
            One of ``['func', 'grad', 'both']``.
        check_feasibility : bool, optional
            Check whether `x` is feasible?

        Returns
        -------
        value : float or ndarray or tuple
            Function value, gradient, or both.
        """
        value, grad = 0, np.zeros(x.shape)
        if mode == 'func':
            for atom in self.smooth_atoms:
                value += atom.smooth_objective(x, mode=mode, 
                                               check_feasibility=check_feasibility)
            return value
        elif mode == 'both':
            for atom in self.smooth_atoms:
                v, g = atom.smooth_objective(x, mode=mode, 
                                             check_feasibility=check_feasibility)
                value += v
                grad += g
            return value, grad

        elif mode == 'grad':
            for atom in self.smooth_atoms:
                grad += atom.smooth_objective(x, mode=mode, 
                                              check_feasibility=check_feasibility)
            return grad
        else:
            raise ValueError("Mode specified incorrectly")

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        Sum of the nonsmooth objectives of all atoms plus the quadratic.
        """
        out = 0.
        for atom in self.nonsmooth_atoms + self.smooth_atoms:
            out += atom.nonsmooth_objective(x,
                                            check_feasibility=check_feasibility)
        out += self.quadratic.objective(x, 'func')
        return out

    default_solver = FISTA
    def proximal(self, proxq, prox_control=None):
        """
        The proximal function for the primal problem.

        Parameters
        ----------
        proxq : `regreg.identity_quadratic.identity_quadratic`
            Quadratic term of the proximal problem.
        prox_control : dict, optional
            Arguments for `FISTA.fit` when solving the dual problem.

        Returns
        -------
        argmin : ndarray
            Minimizer of the proximal problem. If the dual problem is
            solved and ``prox_control['return_objective_hist']`` is True, a
            tuple ``(argmin, objective_hist)`` is returned.
        """

        transform, atom = self.transform, self.atom

        if not (isinstance(transform, afidentity) or
                isinstance(transform, afselector)):
            #Default fitting parameters
            prox_defaults = {'max_its': 5000,
                             'min_its': 5,
                             'return_objective_hist': False,
                             'tol': 1e-14,
                             'debug':False}

            if prox_control is not None:
                prox_defaults.update(prox_control)
            prox_control = prox_defaults

            primal_objective = zero_nonsmooth(transform.input_shape)
            primal_objective.quadratic = proxq + self.quadratic + self.smoothq
            dual_objective = primal_objective.conjugate
            dualp = dual_problem(dual_objective,
                                 transform,
                                 atom)

            #Approximate Lipschitz constant
            if not 'dual_reference_lipschitz' in prox_control.keys():
                # shouldn't do this over and over
                self.dual_reference_lipschitz = 1.05*power_L(transform, debug=prox_control['debug'])
            else:
                self.dual_reference_lipschitz = prox_control['dual_reference_lipschitz']
                prox_control.pop('dual_reference_lipschitz')
                
            dualopt = container.default_solver(dualp)
            dualopt.perform_backtrack = False
            dualopt.debug = prox_control['debug']

            if dualopt.perform_backtrack:
                #If backtracking set start_step
                prox_control['start_step'] = proxq.coef / self.dual_reference_lipschitz 

            # the lipschitz estimate comes from the
            # fact that the conjugate of a quadratic
            # with coef lipschitz is quadratic with coef 1/lipschitz

            dualp.lipschitz = self.dual_reference_lipschitz / proxq.coef

            if hasattr(self, 'dual_minimizer'):
                dualopt.composite.coefs[:] = self.dual_minimizer
            history = dualopt.fit(**prox_control)
            dual_minimizer = dualopt.composite.coefs
            lipschitz, x, grad = proxq.coef, proxq.center, proxq.linear_term
            if prox_control['return_objective_hist']:
                return x - grad / lipschitz - transform.adjoint_map(dualopt.composite.coefs/lipschitz), history
            else:
                return x - grad / lipschitz - transform.adjoint_map(dualopt.composite.coefs/lipschitz)

        else:
            proxq = proxq + self.quadratic + self.smoothq
            primal = atom.conjugate
            lipschitz, x, grad = proxq.coef, proxq.center, proxq.linear_term
            if isinstance(transform, afselector):
                z = x - grad / lipschitz
                z[transform.index_obj] = primal.proximal(proxq[transform.index_obj])
                return z
            else:
                return primal.proximal(proxq)

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        """
        Solve the problem using `FISTA`.

        Parameters
        ----------
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic added to the objective.
        return_optimum : bool, optional
            Also return the optimal value?
        fit_args : dict
            Keyword arguments passed to `regreg.algorithms.FISTA.fit`.

        Returns
        -------
        optimal_value : float
            Only returned if `return_optimum` is True.
        coefs : ndarray
            The solution.
        """
        if quadratic is not None:
            oldq, self.quadratic = self.quadratic, self.quadratic + quadratic
        else:
            oldq = self.quadratic

        solver = FISTA(self)
        solver.fit(**fit_args)

        if return_optimum:
            value = (self.objective(self.coefs), self.coefs)
        else:
            value = self.coefs
        self.quadratic = oldq
        return value

