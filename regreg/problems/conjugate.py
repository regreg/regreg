"""
Convex conjugates of atoms, computed by solving an optimization problem.
"""

from __future__ import print_function, division, absolute_import

from copy import copy

import numpy as np

from ..algorithms import FISTA
from ..smooth import smooth_atom
from .composite import composite
from .simple import simple_problem

from ..identity_quadratic import identity_quadratic

class conjugate(composite):

    r"""
    The convex conjugate of an atom plus a quadratic.

    For an atom :math:`f` and quadratic :math:`Q` this represents the
    smooth function

    .. math::

        v \mapsto \sup_{\beta} \left(\beta^Tv - f(\beta) - Q(\beta)\right)

    where the atom's own quadratic is included in :math:`f`. Its gradient is
    the maximizer, computed with the atom's proximal map (or by solving a
    smooth problem if the atom is smooth).

    Parameters
    ----------
    atom : `regreg.atoms.atom` or `regreg.smooth.smooth_atom`
        The function to conjugate. It is copied.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic added to the atom; the total quadratic coefficient must
        be nonzero.
    negate : bool, optional
        If True, evaluate the conjugate at :math:`-v` instead of :math:`v`.
    fit_args : dict
        Keyword arguments for the solver, used if `atom` is smooth.
    """

    def __init__(self, atom, quadratic=None, negate=False, **fit_args):
        
        # we copy the atom because we will modify its quadratic part
        self.atom = copy(atom)

        if quadratic is None:
            quadratic = identity_quadratic(0, 0, 0, 0)
        self.conjugate_quadratic = quadratic

        totalq = self.atom.quadratic + quadratic
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

        # for various purposes,
        # we often want to evalute the conjugate of -x 
        # instead of x

        # negate saves composition with an affine transform

        self.negate = negate

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        Evaluate the conjugate function and/or its gradient.

        Parameters
        ----------
        x : ndarray
            Point at which to evaluate.
        mode : str, optional
            If ``'both'``, return both function value and gradient;
            if ``'grad'``, return only the gradient;
            if ``'func'``, return only the function value.
        check_feasibility : bool, optional
            Unused.

        Returns
        -------
        value : float or ndarray or tuple
            Function value, gradient, or both.
        """
        v = -x if self.negate else x

        old_lin = self.conjugate_quadratic.linear_term
        new_lin = -v if old_lin is None else old_lin - v
        self.conjugate_quadratic.linear_term = new_lin
        if isinstance(self.atom, smooth_atom):
            minimizer = self.problem.solve(self.conjugate_quadratic, max_its=5000, **self.fit_args)
        else: # it better have a proximal method!
            minimizer = self.atom.proximal(self.conjugate_quadratic)
        self.conjugate_quadratic.linear_term = old_lin

        # retain a reference
        self.argmin = minimizer

        if mode == 'both':
            val = self.atom.objective(minimizer)
            if not self.negate:
                return -val - self.conjugate_quadratic.objective(minimizer, mode='func') + (v * minimizer).sum(), minimizer
            else:
                return -val - self.conjugate_quadratic.objective(minimizer, mode='func') + (v * minimizer).sum(), -minimizer
        elif mode == 'func':
            val = self.atom.objective(minimizer)
            return -val - self.conjugate_quadratic.objective(minimizer, mode='func') + (v * minimizer).sum()
        elif mode == 'grad':
            return -minimizer if self.negate else minimizer
        raise ValueError("mode incorrectly specified")
