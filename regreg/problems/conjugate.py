import numpy as np

from copy import copy

from ..algorithms import FISTA
from ..smooth.quadratic import quadratic, smooth_atom
from composite import composite
from simple import simple_problem

from ..identity_quadratic import identity_quadratic

class conjugate(composite):

    def __init__(self, atom, quadratic=None, **fit_args):

        # we copy the atom because we will modify its quadratic part
        self.atom = copy(atom)

        if self.atom.quadratic is None:
            self.atom.set_quadratic(0, 0, 0, 0)
        
        if quadratic is not None:
            totalq = self.atom.quadratic + quadratic
        else:
            totalq = self.atom.quadratic
        if totalq.coef in [0, None]:
            raise ValueError('quadratic coefficient must be non-zero')

        if isinstance(self.atom, smooth_atom):
            self.problem = simple_problem.smooth(self.atom)
        self.fit_args = fit_args

        #XXX we need a better way to pass around the Lipschitz constant
        # should go in the container class
#         if hasattr(self.atom, "lipschitz"):
#             self.fit_args['perform_backtrack'] = False
#         else:
#             self.fit_args['perform_backtrack'] = True

        self._have_solved_once = False
        self.shape = atom.shape
        self.quadratic = quadratic

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate the conjugate function and/or its gradient

        if mode == 'both', return both function value and gradient
        if mode == 'grad', return only the gradient
        if mode == 'func', return only the function value
        """

        old_lin = self.quadratic.linear_term
        if old_lin is not None:
            new_lin = old_lin - x
        else:
            new_lin = -x
        self.quadratic.linear_term = new_lin
        if isinstance(self.atom, smooth_atom):
            minimizer = self.problem.solve(self.quadratic, max_its=5000, **self.fit_args)
        else: # it better have a proximal method!
            minimizer = self.atom.proximal(self.quadratic)
        self.quadratic.linear_term = old_lin
    
        # retain a reference
        self.argmin = minimizer
        if mode == 'both':
            v = self.atom.objective(minimizer)
            return - v - self.quadratic.objective(minimizer, mode='func') + (x * minimizer).sum(), minimizer
        elif mode == 'func':
            v = self.atom.objective(minimizer)
            return - v - self.quadratic.objective(minimizer, mode='func') + (x * minimizer).sum()
        elif mode == 'grad':
            return minimizer
        else:
            raise ValueError("mode incorrectly specified")
