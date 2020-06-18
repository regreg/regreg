"""
This module has a class for specifying a problem from just
a smooth function and a single penalty.
"""
from __future__ import print_function, division, absolute_import

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
    Minimize an objective using proximal Newton algorithm
    as described [here](https://arxiv.org/pdf/1206.1623.pdf).

    Uses BFGS updates.

    """

    damping_term = 0.01

    def __init__(self, 
                 smooth_atom, 
                 proximal_atom,
                 initial_hessian,
                 initial_step=1.,
                 quadratic=None, # e.g. for ridge 
                 ):
        '''
        Parameters
        ----------

        smooth_objective : `smooth_atom`
            Smooth objective function.

        proximal_map : `atom`
            Nonsmooth part -- its proximal map is used in solving
            inner loop.

        initial_hessian : ndarray
            Guess at initial Hessian.

        initial_step : float
            Guess at initial stepsize.

        quadratic : `identity_quadratic`
            Quadratic term optionally
            added to overall objective.

        '''
        self.smooth_atom = smooth_atom
        self.proximal_atom = proximal_atom
        self.coefs = self.smooth_atom.coefs = self.proximal_atom.coefs
        self.hessian = initial_hessian.copy()
        self.diag_hessian = np.mean(np.diag(self.hessian)) * np.identity(self.hessian.shape[0])
        self.quadratic_loss = quadratic_loss(self.coefs.shape,
                                             Q=self.hessian +
                                               self.damping_term * self.diag_hessian)
        if quadratic is None:
            quadratic = identity_quadratic(0, 0, 0, 0)
        self.quadratic = quadratic

        self.quadratic_problem = simple_problem(self.quadratic_loss,
                                                proximal_atom)
        self.stepsize = initial_step

    def update_hessian(self, old_soln, new_soln, old_grad, new_grad):
        """
        Update current Hessian using 
        [BFGS](https://en.wikipedia.org/wiki/Broyden%E2%80%93Fletcher%E2%80%93Goldfarb%E2%80%93Shanno_algorithm)
        updating rule.

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

        self.hessian += np.multiply.outer(y, y) / y.dot(s) - np.multiply.outer(Bs, Bs) / s.dot(Bs)
        self.quadratic_loss.Q[:] = self.hessian + self.damping_term * self.diag_hessian

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        '''
        Compute value and / or gradient of our smooth part.

        Parameters
        ----------

        x : ndarray
            Argument to smooth objective.

        Returns
        -------

        f : float (optional)
            Value of smooth objective at `x`.

        g : ndarray (optional)
            Gradient of smooth objective at `x`.

        '''
        vs = self.smooth_atom.smooth_objective(x, mode, check_feasibility)
        return vs

    def nonsmooth_objective(self, x, check_feasibility=False):
        '''
        Compute value and / or gradient of nonsmooth part.

        Parameters
        ----------

        x : ndarray
            Argument to nonsmooth objective.

        Returns
        -------

        f : float (optional)
            Value of nonsmooth objective at `x`.

        '''
        vn = self.proximal_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        vs = self.smooth_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        return vn + vs + self.quadratic.objective(x, 'func')

    def proximal(self, proxq):
        proxq = proxq + self.smooth_atom.quadratic + self.quadratic
        return self.proximal_atom.solve(proxq)

    def latexify(self, var=None):
        template_dict = self.objective_vars.copy()
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
        
        '''
        Inner loop of quasi-Newton.
        Given `x, G, H` our current position, gradient and
        Hessian of the smooth part we minimize
        (quasi) 2nd order Taylor approximation of
        smooth part plus the nonsmooth atom.

        .. math::

            \Delta \mapsto G^T\Delta + \frac{1}{2} \Delta^TH\Delta + {\cal P}(\Delta + x)

        Parameters
        ----------

        niter : int
            Number of quasi-Newton steps to take.

        Returns
        -------

        coefs : ndarray
            Optimal solution.
          
        '''

        # return search direction 
        # loss is 
        # x -> (\nabla g(x_cur) - H x_cur)^Tx + 1/2 x^THx + P(x)

        grad = self.smooth_objective(self.coefs,
                                     'grad')
        linear_term = (grad - self.hessian.dot(self.coefs))
        quad = identity_quadratic(0, 0, linear_term, 0)
        self.quadratic_problem.coefs[:] = self.coefs # warm start the subproblem
        quad_soln = self.quadratic_problem.solve(quad + self.quadratic, **fit_args)
        direction = quad_soln - self.coefs
        return direction, grad 

    def solve(self,
              niter=30,
              maxfun=10,
              maxiter=5,
              **fit_args, # for inner loop solver
              ):
        
        '''
        Run quasi-Newton for `niter` steps.

        Parameters
        ----------

        niter : int
            Number of quasi-Newton steps to take.

        maxfun : int
            Maximum function evaluations passed to `fmin_powell`

        maxiter : int
            Maximum iterations passed to `fmin_powell`

        Returns
        -------

        coefs : ndarray
            Optimal solution.
          
        '''

        self.solver_results = []
        for i in range(niter):
            direction, grad = self.solve_inner_loop(**fit_args)
            fit_args['start_step'] = 1.5 * self.quadratic_problem.final_step
            stepsize, val = self.backtrack(direction,
                                           maxfun=maxfun,
                                           maxiter=maxiter)
            delta = stepsize * direction

            new_grad = self.smooth_objective(self.coefs + delta, 'grad')
            self.update_hessian(self.coefs, self.coefs + delta, grad, new_grad)

            # update state 

            self.coefs += delta
            self.stepsize = stepsize
            self.solver_results.append(val)

        self.solver_results = np.array(self.solver_results)
        return self.coefs

    def backtrack(self, 
                  direction,
                  maxiter=10,
                  maxfun=6):
        '''
        Run a simple backtracking using `scipy.optimize.fmin_powell`
        to determine optimal stepsize.

        Parameters
        ----------

        direction : ndarray
            Direction for backtracking search.
            Found by solving inner optimization problem
            in quasi-Newton algorithm.

        maxiter : int
            Maximum number of iterations in `fmin_powell`

        Returns
        -------

        stepsize : float
            Optimal stepsize.
          
        '''

        BIG = 1e16

        def restriction(stepsize):
            val = (self.smooth_objective(self.coefs + stepsize * direction, 'func') +
                   self.nonsmooth_objective(self.coefs + stepsize * direction,
                                            check_feasibility=True))
            if np.isnan(val):
                val = BIG
            grad = self.smooth_objective(self.coefs + stepsize * direction, 'grad')
            if np.any(np.isnan(grad)):
                val += BIG
            return val

        step =  fmin_powell(restriction, 
                            self.stepsize, 
                            disp=False,
                            maxiter=maxiter,
                            maxfun=maxfun)
        return step, restriction(step)
