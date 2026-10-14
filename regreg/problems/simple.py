"""
Problems specified by a smooth function and a single penalty.

This module has a class for specifying a problem from just
a smooth function and a single penalty, as well as the
`gengrad`, `nesta` and `tfocs` solvers.
"""
from __future__ import print_function, division, absolute_import

import numpy as np, warnings

from ..problems.composite import composite
from ..affine import identity, scalar_multiply, astransform, adjoint
from ..atoms import atom
from ..atoms.cones import zero as zero_cone
from ..smooth import zero as zero_smooth, sum as smooth_sum, affine_smooth
from ..identity_quadratic import identity_quadratic
from ..algorithms import FISTA

class simple_problem(composite):

    """
    A problem with a smooth part and a single nonsmooth penalty.

    The objective is ``smooth_atom`` plus ``proximal_atom``, each
    including its quadratic part.

    Parameters
    ----------
    smooth_atom : `regreg.smooth.smooth_atom`
        The smooth part of the objective.
    proximal_atom : `regreg.atoms.atom`
        The nonsmooth part of the objective; it must have a proximal map.
        It is assumed to have 0 for its smooth objective.
    """
    
    def __init__(self, smooth_atom, proximal_atom):
        self.smooth_atom = smooth_atom
        self.proximal_atom = proximal_atom
        self.coefs = self.smooth_atom.coefs = self.proximal_atom.coefs

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """
        The smooth objective of ``smooth_atom``.

        This class explicitly assumes that
        the proximal_atom has 0 for smooth_objective.

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
        vs = self.smooth_atom.smooth_objective(x, mode, check_feasibility)
        return vs

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        Nonsmooth objective of ``proximal_atom`` plus the quadratics.
        """
        vn = self.proximal_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        vs = self.smooth_atom.nonsmooth_objective(x, check_feasibility=check_feasibility)
        return vn + vs + self.quadratic.objective(x, 'func')

    def proximal(self, proxq):
        """
        Proximal map of ``proximal_atom``, with the quadratic parts of
        ``smooth_atom`` and the problem added to `proxq`.
        """
        proxq = proxq + self.smooth_atom.quadratic + self.quadratic
        return self.proximal_atom.solve(proxq)

    @staticmethod
    def smooth(smooth_atom):
        """
        A problem with no nonsmooth part except possibly
        the quadratic of `smooth_atom`.

        The proximal function is (almost) a nullop.

        Parameters
        ----------
        smooth_atom : `regreg.smooth.smooth_atom`
            The smooth objective.

        Returns
        -------
        problem : `simple_problem`
            Problem with a zero penalty.
        """
        proximal_atom = zero_cone(smooth_atom.shape)
        return simple_problem(smooth_atom, proximal_atom)

    @staticmethod
    def nonsmooth(proximal_atom):
        """
        A problem with no smooth part.

        Parameters
        ----------
        proximal_atom : `regreg.atoms.atom`
            The nonsmooth objective.

        Returns
        -------
        problem : `simple_problem`
            Problem with a zero smooth part.
        """
        smooth_atom = zero_smooth(proximal_atom.shape)
        return simple_problem(smooth_atom, proximal_atom)

    def latexify(self, var=None):
        """
        Return a LaTeX representation of the problem.
        """
        template_dict = self.objective_vars.copy()
        template_dict['prox'] =  self.proximal_atom.latexify(var=var,idx='2')
        template_dict['smooth'] = self.smooth_atom.latexify(var=var,idx='1')
        result = r'''
        \begin{aligned}
        \text{minimize}_{%(var)s} & f(%(var)s) + g(%(var)s) \\
        f(%(var)s) &= %(smooth)s \\
        g(%(var)s) &= %(prox)s \\
        \end{aligned}
        ''' % template_dict
        result = '\n'.join([s.strip() for s in result.split('\n')])
        return result

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        """
        Solve the problem using `FISTA`.

        The final step size is stored as ``self.final_step``.

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
        solver.composite.coefs[:] = self.coefs
        self.solver_results = solver.fit(**fit_args)
        self.final_step = solver.step

        if return_optimum:
            value = (self.objective(self.coefs), self.coefs)
        else:
            value = self.coefs
        self.quadratic = oldq
        return value

    
def gengrad(simple_problem, lipschitz, tol=1.0e-8, max_its=1000, debug=False,
            coef_stop=False):
    """
    A simple generalized gradient solver.

    Parameters
    ----------
    simple_problem : `simple_problem`
        Problem to solve.
    lipschitz : float
        Lipschitz constant of the gradient; the step size is
        ``1 / lipschitz``.
    tol : float, optional
        Convergence tolerance.
    max_its : int, optional
        Maximum number of iterations.
    debug : bool, optional
        Print progress information?
    coef_stop : bool, optional
        Stop based on coefficient changes instead of objective value.

    Returns
    -------
    coef : ndarray
        The solution.
    """
    itercount = 0
    coef = simple_problem.coefs
    value = np.inf
    while True:
        vnew, g = simple_problem.smooth_objective(coef, 'both')
        vnew += simple_problem.nonsmooth_objective(coef)
        newcoef = simple_problem.solve(identity_quadratic(lipschitz, coef, g, 0))
        if coef_stop:
            coef_stop_check = (np.linalg.norm(coef-newcoef) <= tol * 
                               np.max([np.linalg.norm(coef),
                                       np.linalg.norm(newcoef), 
                                       1]))
            if coef_stop_check:
                break
        else:
            obj_stop_check = np.fabs(value - vnew) <= tol * np.max([vnew, 1])
            if obj_stop_check:
                break
        if itercount == max_its:
            break
        if debug:
            print(itercount, vnew, value, (vnew - value) / vnew)
        value = vnew
        itercount += 1
        coef = newcoef
    return coef

def nesta(smooth_atom, proximal_atom, conjugate_atom, epsilon=None,
          tol=1.e-06, max_iters=100,
          min_iters=5, 
          coef_tol=1.e-6,
          initial_primal=None,
          initial_dual=None,
          coef_stop=False,
          quadratic=None):
    '''
    Solve a problem with the NESTA algorithm.

    The objective is the sum of `smooth_atom`, `proximal_atom` and
    `conjugate_atom`, where `conjugate_atom` is smoothed by adding a
    quadratic to its conjugate, with decreasing smoothing parameters
    `epsilon`.

    Parameters
    ----------
    smooth_atom : `regreg.problems.composite.smooth`
        A smooth function, i.e. having a smooth_objective method. May be
        None.
    proximal_atom : `regreg.atoms.atom`
        An atom with a proximal method. May be None.
    conjugate_atom : `regreg.atoms.atom`
        An atom that will be smoothed, by adding a quadratic to its
        conjugate.
    epsilon : ndarray, optional
        A decreasing array of positive constants for Moreau-Yosida
        smoothing.
    tol : float, optional
        Tolerance to which each problem is solved is
        ``max(tol, epsilon)``.
    max_iters : int, optional
        Maximum number of iterations. If epsilon is not supplied,
        it is taken to be ``[1]*max_iters``.
    min_iters : int, optional
        Minimum number of iterations.
    coef_tol : float, optional
        Tolerance for assessing convergence.
    initial_primal : ndarray, optional
        Initial condition for the primal variable.
    initial_dual : ndarray, optional
        Initial condition for the dual variable.
    coef_stop : bool, optional
        Stop based on changes in the dual coefficients instead of the
        objective value.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        A quadratic term that is added to the total objective.

    Returns
    -------
    primal : ndarray
        Primal coefficients.
    dual : ndarray
        Dual coefficients.
    '''
    if epsilon is None:
        epsilon = [1]*max_iters # 2.**(-np.arange(20))

    transform, conjugate = conjugate_atom.dual
    if initial_dual is None:
        dual_coef = np.zeros(transform.output_shape) 
    else:
        dual_coef = initial_dual.copy()

    if quadratic is None:
        quadratic = identity_quadratic(0,0,0,0)

    primal_old = np.zeros(transform.output_shape)
    value = np.inf
    for idx, eps in enumerate(epsilon):
        smoothed = conjugate_atom.smoothed(identity_quadratic(eps, dual_coef, 0, 0))
        if smooth_atom is not None:
            final_smooth = smooth_sum([smooth_atom, smoothed])
        else:
            final_smooth = smoothed
        if proximal_atom is not None:
            problem = simple_problem(final_smooth, proximal_atom)
        else:
            problem = simple_problem.smooth(final_smooth)

        if idx == 0 and initial_primal is not None:
            problem.coefs[:] = initial_primal

        primal_coef = problem.solve(quadratic,tol=max(eps,tol))
        updated = problem.objective(primal_coef) + quadratic.objective(primal_coef, 'func')

        # should we have a better convergence criterion?
        if coef_stop:
            if (np.linalg.norm(dual_coef - smoothed.grad) < coef_tol * (np.linalg.norm(smoothed.grad) + 1)
                and idx >= min_iters):
                break
        else:
            if (np.fabs(updated - value) < coef_tol * (np.fabs(updated) + 1) and
                idx >= min_iters):
                break
        value = updated

        # when there's an affine transform involved
        dual_coef[:] = smoothed.grad + idx / (idx + 3.) * (smoothed.grad - dual_coef)
        if idx == max_iters-1:
            warnings.warn('problem not solved after %d iterations' % max_iters)

    return primal_coef, dual_coef

def tfocs(primal_atom, transform, dual_proximal_atom, epsilon=None,
          tol=1.e-06,
          max_iters=100,
          coef_tol=1.e-6,
          quadratic=None):
    r'''
    Solve a problem with a TFOCS-style smoothed dual method.

    This function is based on the setup of problems
    described in `TFOCS <http://tfocs.stanford.edu/>`_.
    Generally speaking, these are the same type of problems
    that `nesta` can handle, though without the additional smooth part.

    This solver is suited to solving problems of the form

    .. math::

        \text{minimize}_v f(v) + h(Dv+a)

    when both :math:`f` and :math:`h` (and hence :math:`f^*` and
    :math:`h^*`) have simple proximal operators.

    Parameters
    ----------
    primal_atom : `regreg.atoms.atom`
        The atom :math:`f`, which will be smoothed,
        then composed with the transform.
    transform : `regreg.affine.affine_transform`
        The affine transform :math:`v \mapsto Dv+a` for the composition.
    dual_proximal_atom : `regreg.atoms.atom`
        The atom :math:`h^*`, which must have a proximal method.
    epsilon : ndarray, optional
        A decreasing array of positive constants for Moreau-Yosida
        smoothing.
    tol : float, optional
        Tolerance to which each problem is solved is
        ``max(tol, epsilon)``.
    max_iters : int, optional
        Maximum number of iterations. If epsilon is not supplied,
        it is taken to be ``[1]*max_iters``.
    coef_tol : float, optional
        Tolerance for assessing convergence of coefficients.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        A quadratic term that is added to the total objective.

    Returns
    -------
    primal : ndarray
        Primal coefficients.
    dual : ndarray
        Dual coefficients.

    Examples
    --------
    Here is an example for minimum :math:`\ell_1` norm reconstruction.

    >>> import numpy as np, regreg.api as rr
    >>> np.random.seed(0)
    >>> n, p = 200, 5000

    The problem assumes Y=np.dot(X,beta) for some sparse beta.

    >>> X = np.random.standard_normal((n, p))
    >>> beta = np.zeros(p)
    >>> beta[:10] = np.arange(10)+1
    >>> Y = np.dot(X, beta)

    The problem is formally,

    .. math::

        \text{minimize}_v \|v\|_1 \quad \text{subject to} \quad Y=Xv

    The :math:`\ell_1` norm is described as:

    >>> l1 = rr.l1norm(p, lagrange=1)

    The constraint is specified as

    >>> constraint = rr.zero_constraint.affine(X,-Y)

    >>> transform, zero = constraint.dual
    >>> epsilon = [0.01]*50 + [0.001]*20
    >>> primal_tfocs, dual_tfocs = rr.tfocs(l1, transform, zero, epsilon=epsilon)
    >>> np.linalg.norm(primal_tfocs - beta) < 1.e-3 * np.linalg.norm(beta)
    True

    >>> np.linalg.norm(primal_tfocs[10:]) < 1.e-3 * np.linalg.norm(primal_tfocs)
    True
    '''
    transform = astransform(transform)

    #conjugate_atom needs a conjugate so it can be smoothed

    conjugate_atom = primal_atom.conjugate

    if epsilon is None:
        epsilon = [1]*max_iters # 2.**(-np.arange(20))

    offset = transform.affine_offset
    if offset is not None:
        dual_sq = identity_quadratic(0,0,-offset, 0)
    else:
        dual_sq = identity_quadratic(0,0,0,0)
        
    if quadratic is None:
        quadratic = identity_quadratic(0,0,0,0)

    primal_coef = np.zeros(conjugate_atom.shape)
    for idx, eps in enumerate(epsilon):
        smoothed = conjugate_atom.smoothed(identity_quadratic(eps, primal_coef, 0, 0) + quadratic)
        final_smooth = affine_smooth(smoothed, scalar_multiply(adjoint(transform), -1))
        problem = simple_problem(final_smooth, dual_proximal_atom)
        dual_coef = problem.solve(dual_sq, tol=max(eps,tol))

        # should we have a better convergence criterion?
        if (np.linalg.norm(primal_coef - final_smooth.grad) < 
            coef_tol * max(np.linalg.norm(final_smooth.grad),1)):
            break

        # when there's an affine transform involved
        primal_coef[:] = final_smooth.grad + idx / (idx + 3.) * (final_smooth.grad - primal_coef)
        if idx == max_iters-1:
            warnings.warn('problem not solved after %d iterations' % max_iters)
    
    return primal_coef, dual_coef
