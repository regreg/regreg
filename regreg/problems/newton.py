"""
A proximal quasi-Newton solver for problems specified by a smooth function
and a single penalty.
"""
from __future__ import print_function, division, absolute_import

from copy import copy
from functools import partial

import numpy as np, warnings
from scipy.optimize import fmin_powell

from .composite import composite
from .simple import simple_problem
from ..affine import identity, scalar_multiply, astransform, adjoint
from ..atoms import atom
from ..smooth import sum as smooth_sum, affine_smooth
from ..smooth.quadratic import quadratic_loss
from ..identity_quadratic import identity_quadratic
from ..algorithms import FISTA

class quasi_newton(object):
    
    """
    Minimize an objective using a proximal quasi-Newton algorithm.

    The algorithm is described in `Lee, Sun and Saunders
    <https://arxiv.org/pdf/1206.1623.pdf>`_. Uses BFGS updates.

    Parameters
    ----------
    smooth_atom : `regreg.smooth.smooth_atom`
        Smooth objective function.
    proximal_atom : `regreg.atoms.atom`
        Nonsmooth part -- its proximal map is used in solving
        inner loop.
    initial_hessian : ndarray
        Guess at initial Hessian.
    initial_step : float, optional
        Guess at initial stepsize.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic term optionally added to overall objective (e.g. for a
        ridge penalty).
    """

    damping_term = 1.

    objective_vars = copy(composite.objective_vars)

    def __init__(self, 
                 smooth_atom, 
                 proximal_atom,
                 initial_hessian,
                 initial_step=1.,
                 quadratic=None, # e.g. for ridge 
                 ):
        self.smooth_atom = smooth_atom
        self.proximal_atom = proximal_atom
        self.coefs = self.smooth_atom.coefs = self.proximal_atom.coefs
        H0 = initial_hessian
        self.hessian = H0 + (self.damping_term *
                             (np.diag(H0).mean() *
                              np.identity(H0.shape[0])))
        self.quadratic_loss = quadratic_loss(self.coefs.shape,
                                             Q=self.hessian)
        if quadratic is None:
            quadratic = identity_quadratic(0, 0, 0, 0)
        self.quadratic = quadratic

        self.quadratic_problem = simple_problem(self.quadratic_loss,
                                                proximal_atom)
        self.stepsize = initial_step

    def objective(self,
                  coefs):
        """
        Total objective.

        Parameters
        ----------
        coefs : ndarray
            Where to evaluate the objective.

        Returns
        -------
        value : float
            Value of the objective.
        """
        return (self.smooth_objective(coefs, 'func') +
                self.nonsmooth_objective(coefs, check_feasibility=True))

    def update_hessian(self,
                       old_soln,
                       new_soln,
                       old_grad,
                       new_grad):
        """
        Update current Hessian using the BFGS updating rule.

        See `BFGS
        <https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm>`_.
        The update is done in place.

        Parameters
        ----------
        old_soln : ndarray
            Current position.
        new_soln : ndarray
            Next position.
        old_grad : ndarray
            Current gradient.
        new_grad : ndarray
            Next gradient.
        """
        # update the hessian, 
        # updates self.quadratic_loss in place

        # uses BFGS

        y = new_grad - old_grad
        s = new_soln - old_soln

        Bs = self.hessian.dot(s)

        if (y.dot(s) < -1e-5):
            raise ValueError('failure in BFGS update -- dot product is negative')

        # update is done in place
        # note that this means self.quadratic_problem.Q is also updated!
        
        self.hessian += np.multiply.outer(y, y) / y.dot(s) - np.multiply.outer(Bs, Bs) / s.dot(Bs)


    def smooth_objective(self, x, mode='both', check_feasibility=False):
        '''
        Compute value and / or gradient of our smooth part.

        Parameters
        ----------
        x : ndarray
            Argument to smooth objective.
        mode : str, optional
            One of ``['func', 'grad', 'both']``.
        check_feasibility : bool, optional
            Check whether `x` is feasible?

        Returns
        -------
        f : float, optional
            Value of smooth objective at `x`.
        g : ndarray, optional
            Gradient of smooth objective at `x`.
        '''
        vs = self.smooth_atom.smooth_objective(x, mode, check_feasibility)
        return vs

    def nonsmooth_objective(self, x, check_feasibility=False):
        '''
        Compute value of nonsmooth part, including the quadratic terms.

        Parameters
        ----------
        x : ndarray
            Argument to nonsmooth objective.
        check_feasibility : bool, optional
            Check whether `x` is feasible?

        Returns
        -------
        f : float
            Value of nonsmooth objective at `x`.
        '''
        vn = self.proximal_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        vs = self.smooth_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        return vn + vs + self.quadratic.objective(x, 'func')

    def proximal(self, proxq):
        """
        Proximal map of ``proximal_atom``, with the quadratic parts added
        to `proxq`.
        """
        proxq = proxq + self.smooth_atom.quadratic + self.quadratic
        return self.proximal_atom.solve(proxq)

    def latexify(self, var=None):
        """
        Return a LaTeX representation of the problem.
        """
        template_dict = self.objective_vars.copy()
        if var is not None:
            template_dict['var'] = var
        template_dict['prox'] =  self.proximal_atom.latexify(var=var,idx='2')
        template_dict['smooth'] = self.smooth_atom.latexify(var=var,idx='1')
        result = r'''
        \text{BFGS Quasi-Newton for}
        \begin{aligned}
        \text{minimize}_{%(var)s} & f(%(var)s) + g(%(var)s) \\
        f(%(var)s) &= %(smooth)s \\
        g(%(var)s) &= %(prox)s \\
        \end{aligned}
        ''' % template_dict
        result = '\n'.join([s.strip() for s in result.split('\n')])
        return result

    def solve_inner_loop(self, 
                         return_optimum=False, 
                         initial_step=None,
                         **fit_args):
        
        r'''
        Inner loop of quasi-Newton.

        Given :math:`x, G, H` our current position, gradient and
        Hessian of the smooth part we minimize the
        (quasi) 2nd order Taylor approximation of the
        smooth part plus the nonsmooth atom.

        .. math::

            \Delta \mapsto G^T\Delta + \frac{1}{2} \Delta^TH\Delta + {\cal P}(\Delta + x)

        Parameters
        ----------
        return_optimum : bool, optional
            Unused.
        initial_step : float, optional
            Unused.
        fit_args : dict
            Keyword arguments for the solver of the quadratic subproblem.

        Returns
        -------
        direction : ndarray
            Search direction :math:`\Delta`.
        grad : ndarray
            Gradient :math:`G` of the smooth part at the current position.
        quad_soln : ndarray
            Solution :math:`x + \Delta` of the quadratic subproblem.
        '''

        # return search direction 
        # loss is 
        # x -> (\nabla g(x_cur) - Q x_cur)^Tx + 1/2 x^TQx + P(x)
        #
        # this gives a direction x^* - x_cur (with x^* optimizer from above)

        Q = self.quadratic_loss.Q
        grad = self.smooth_objective(self.coefs,
                                     'grad')
        linear_term = (grad - Q.dot(self.coefs))
        quad = identity_quadratic(0, 0, linear_term, 0)

        # warm start the subproblem
        self.quadratic_problem.coefs[:] = self.coefs 

        quad_soln = self.quadratic_problem.solve(quad + self.quadratic,
                                                 **fit_args)
        direction = quad_soln - self.coefs
        return direction, grad, quad_soln 

    def solve(self,
              niter=30,
              maxfun=10,
              maxiter=5,
              **fit_args # for inner loop solver
              ):
        
        '''
        Run quasi-Newton for at most `niter` steps.

        Parameters
        ----------
        niter : int, optional
            Number of quasi-Newton steps to take.
        maxfun : int, optional
            Maximum function evaluations passed to `fmin_powell`.
        maxiter : int, optional
            Maximum iterations passed to `fmin_powell`.
        fit_args : dict
            Keyword arguments for the inner loop solver.

        Returns
        -------
        coefs : ndarray
            Optimal solution.
        '''

        self.solver_results = [np.inf]
        for i in range(niter):
            direction, grad, quad_soln = self.solve_inner_loop(**fit_args)
            fit_args['start_step'] = 1.5 * self.quadratic_problem.final_step
            stepsize, val = self.backtrack(direction,
                                           maxfun=maxfun,
                                           maxiter=maxiter)
            delta = stepsize * direction
            new_grad = self.smooth_objective(self.coefs + delta, 'grad')

            if self.converged(self.solver_results[-1],
                              val,
                              self.coefs,
                              self.coefs + delta,
                              grad,
                              new_grad):
                break

            self.update_hessian(self.coefs, self.coefs + delta, grad, new_grad)

            # update state 

            self.coefs = self.coefs + delta
            self.stepsize = stepsize
            self.solver_results.append(val)

        self.solver_results = np.array(self.solver_results)
        self.coefs[:] = quad_soln # ensures the last solution
                                  # will satisfy the constraint
                                  # if in bound form
        return self.coefs

    def converged(self,
                  old_value,
                  new_value,
                  old_soln,
                  new_soln,
                  old_grad,
                  new_grad):
        """
        Check convergence based on the change in position.

        Parameters
        ----------
        old_value : float
            Current value (unused).
        new_value : float
            Next value (unused).
        old_soln : ndarray
            Current position.
        new_soln : ndarray
            Next position.
        old_grad : ndarray
            Current gradient (unused).
        new_grad : ndarray
            Next gradient (unused).

        Returns
        -------
        converged : bool
            True if the maximum absolute change in position is small
            relative to the norm of `new_soln`.
        """
        return (np.fabs(old_soln - new_soln).max()
                < 1.e-6 * np.linalg.norm(new_soln))

    def backtrack(self, 
                  direction,
                  maxiter=10,
                  maxfun=6):
        '''
        Run a simple line search using `scipy.optimize.fmin_powell`
        to determine optimal stepsize.

        Parameters
        ----------
        direction : ndarray
            Direction for line search.
            Found by solving inner optimization problem
            in quasi-Newton algorithm.
        maxiter : int, optional
            Maximum number of iterations in `fmin_powell`.
        maxfun : int, optional
            Maximum number of function evaluations in `fmin_powell`.

        Returns
        -------
        stepsize : float
            Optimal stepsize.
        value : float
            Objective value at the new position.
        '''

        BIG = 1e16

        def restriction(direction, coefs, stepsize):
            val = (self.smooth_objective(coefs + stepsize * direction, 'func') +
                   self.nonsmooth_objective(coefs + stepsize * direction,
                                            check_feasibility=True))
            if np.isnan(val):
                val = BIG
            grad = self.smooth_objective(coefs + stepsize * direction, 'grad')
            if np.any(np.isnan(grad)):
                val += BIG
            return val

        restriction = partial(restriction, direction, self.coefs)
        step =  fmin_powell(restriction,
                            1, 
                            disp=False,
                            maxiter=maxiter,
                            maxfun=maxfun)
        return step, restriction(step)
