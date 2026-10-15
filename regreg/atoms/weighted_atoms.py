r"""
Weighted versions of the :math:`\ell_1` and :math:`\ell_{\infty}` norms.
"""
import numpy as np
from scipy import sparse
from copy import copy
import warnings

from .seminorms import seminorm as unweighted_seminorm

from ..problems.composite import composite, nonsmooth, smooth_conjugate
from ..affine import (linear_transform, identity as identity_transform, 
                     affine_transform, selector)
from ..identity_quadratic import identity_quadratic
from ..atoms import _work_out_conjugate
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)
from .piecewise_linear import find_solution_piecewise_linear

class seminorm(unweighted_seminorm):

    """
    Base class for weighted seminorms.

    Parameters
    ----------
    weights : np.ndarray
        Nonnegative weights, one per coordinate. The shape of the atom
        is the shape of `weights`.
    lagrange : float, optional
        Lagrange parameter (exactly one of `lagrange` and `bound` must be
        specified).
    bound : float, optional
        Bound parameter.
    offset : np.ndarray, optional
        Offset subtracted from the argument before the seminorm is
        evaluated.
    quadratic : identity_quadratic, optional
        Quadratic term added to the seminorm.
    initial : np.ndarray, optional
        Initial value of the coefficients.
    """
    tol = 1.0e-05

    def __init__(self, weights, lagrange=None, bound=None, 
                 offset=None, 
                 quadratic=None,
                 initial=None):

        self.weights = np.asarray(weights, float)
        unweighted_seminorm.__init__(self, self.weights.shape,
                                     lagrange=lagrange,
                                     bound=bound,
                                     quadratic=quadratic,
                                     initial=initial,
                                     offset=offset)

        self.invweights = np.zeros_like(self.weights)
        zero_weight = self.weights == 0
        self.invweights[~zero_weight] = 1. / self.weights[~zero_weight]
        self.invweights[zero_weight] = np.inf
        
        if not np.all(self.weights >= 0):
            raise ValueError('weights should be nonnegative')

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            weights_equal = np.all(np.equal(self.weights, other.weights))
            if self.bound is not None:
                return self.bound == other.bound and weights_equal
            return self.lagrange == other.lagrange and weights_equal
        return False

    def __copy__(self):
        return self.__class__(self.weights.copy(),
                              quadratic=self.quadratic,
                              initial=self.coefs,
                              bound=copy(self.bound),
                              lagrange=copy(self.lagrange),
                              offset=copy(self.offset))
    
    def __repr__(self):
        if self.lagrange is not None:
            if self.quadratic.iszero:
                return "%s(%s, lagrange=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     str(self.weights),
                     self.lagrange,
                     str(self.offset))
            else:
                return "%s(%s, lagrange=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__, 
                     str(self.weights),
                     self.lagrange,
                     str(self.offset),
                     self.quadratic)
        else:
            if self.quadratic.iszero:
                return "%s(%s, bound=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     str(self.weights),
                     self.bound,
                     str(self.offset))
            else:
                return "%s(%s, bound=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     str(self.weights),
                     self.bound,
                     str(self.offset),
                     self.quadratic)

    @doc_template_user
    def get_conjugate(self):
        if self.quadratic.coef == 0:

            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            if self.bound is None:
                cls = conjugate_weighted_pairs[self.__class__]
                atom = cls(self.invweights, 
                           bound=self.lagrange, 
                           lagrange=None,
                           offset=offset,
                           quadratic=outq)
            else:
                cls = conjugate_weighted_pairs[self.__class__]
                atom = cls(self.invweights,
                           lagrange=self.bound, 
                           bound=None,
                           offset=offset,
                           quadratic=outq)
        else:
            atom = smooth_conjugate(self)

        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    
    def form_transform(self, subsample=False):
        '''
        Form the linear transform applied before the seminorm.

        By subsampling we can get rid of some variables that have 0 weights.

        Parameters
        ----------
        subsample : bool, optional
            If True and some weights are 0, the transform is a `selector`
            keeping only the coordinates with nonzero weights. Otherwise
            it is the identity.

        Returns
        -------
        transform : affine_transform
            The (cached) linear transform.
        '''
        if not hasattr(self, '_linear_transform'):
            if self.weights is not None:
                test = self.weights == 0
                if test.sum() and subsample:
                    self._linear_transform = selector(~test, self.shape)
                else:
                    self._linear_transform = identity_transform(self.shape)
            else:
                self._linear_transform = identity_transform(self.shape)
        return self._linear_transform
    linear_transform = property(form_transform)


@objective_doc_templater()
class l1norm(seminorm):

    r"""
    The weighted :math:`\ell_1` norm.

    This is :math:`\|W\beta\|_1 = \sum_i w_i |\beta_i|` where
    :math:`W` is the diagonal matrix of `weights`. Coordinates with weight
    0 are unpenalized; coordinates with infinite weight are constrained
    to be 0.
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|W%(var)s\|_1"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'weighted_l1norm'
    objective_vars['dualnormklass'] = 'weighted_supnorm'
    objective_vars['initargs'] = '[1., 2., 3.]'

    @doc_template_user
    def seminorm(self,
                 arg,
                 lagrange=None,
                 check_feasibility=False):
        lagrange = seminorm.seminorm(self,
                                     arg, 
                                     check_feasibility=check_feasibility, 
                                     lagrange=lagrange)
        finite = np.isfinite(self.weights)
        if check_feasibility:
            check_zero = ~finite * (arg != 0)
            if check_zero.sum():
                return np.inf
        arg = np.asarray(arg)
        return lagrange * np.fabs(arg[finite] * self.weights[finite]).sum()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inbox = self.seminorm(arg,
                              lagrange=1,
                              check_feasibility=True) <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        return np.sign(arg) * np.maximum(np.fabs(arg)-lagrange * self.weights
                                       / lipschitz, 0)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        _isinf = np.isinf(self.weights)
        _is0 = self.weights == 0
        _keep = ~(_isinf + _is0)
        _lagrange = find_solution_piecewise_linear(bound,
                                                   0,
                                                   np.fabs(arg[_keep]),
                                                   self.weights[_keep])
        cut = self.weights * (_lagrange + _isinf)
        value = np.sign(arg) * np.maximum(np.fabs(arg) - cut, 0)
        return value
    
    def terms(self, arg):
        """
        Return the terms that are summed in computing the seminorm.

        Parameters
        ----------
        arg : np.ndarray
            Argument of the seminorm.

        Returns
        -------
        terms : np.ndarray
            The values ``np.fabs(arg) * self.weights``.
        """
        arg = np.asarray(arg)
        return np.fabs(arg) * self.weights

@objective_doc_templater()
class supnorm(seminorm):

    r"""
    The weighted :math:`\ell_{\infty}` norm.

    This is :math:`\|W\beta\|_{\infty} = \max_i w_i |\beta_i|` where
    :math:`W` is the diagonal matrix of `weights`. It is the dual of the
    weighted `l1norm` with weights :math:`1/w_i`.
    """

    objective_template = r"""\|W%(var)s\|_{\infty}"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'weighted_supnorm'
    objective_vars['dualnormklass'] = 'weighted_l1norm'
    objective_vars['initargs'] = '[1., 2., 3.]'

    @doc_template_user
    def seminorm(self,
                 arg,
                 lagrange=None,
                 check_feasibility=False):
        lagrange = seminorm.seminorm(self,
                                     arg, 
                                     check_feasibility=check_feasibility, 
                                     lagrange=lagrange)
        finite = np.isfinite(self.weights)
        if check_feasibility:
            check_zero = ~finite * (arg != 0)
            if check_zero.sum():
                return np.inf
        return lagrange * np.fabs(arg[finite] * self.weights[finite]).max()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inbox = self.seminorm(arg, lagrange=1,
                              check_feasibility=True) <= bound * (1+self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        weights = self.invweights
        _isinf = np.isinf(weights)
        _is0 = weights == 0
        _keep = ~(_isinf + _is0)
        _lagrange = find_solution_piecewise_linear(lagrange / lipschitz,
                                                   0,
                                                   np.fabs(arg[_keep]),
                                                   weights[_keep])
        resid = np.sign(arg) * np.maximum(np.fabs(arg)-_lagrange * weights, 0)

        return arg - resid

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        return np.clip(arg, -bound * self.invweights, bound * self.invweights)

    def terms(self, arg, check_feasibility=True):
        """
        Return the terms that are maximized in computing the seminorm.

        Parameters
        ----------
        arg : np.ndarray
            Argument of the seminorm.
        check_feasibility : bool, optional
            If True, entries with weight 0 are set to ``np.inf`` where
            `arg` is nonzero (and 0 otherwise).

        Returns
        -------
        terms : np.ndarray
            The values ``np.fabs(arg) / self.weights``.
        """
        arg = np.asarray(arg)
        v = np.fabs(arg) / self.weights
        if check_feasibility:
            v[self.weights == 0] = np.where(np.fabs(arg[self.weights == 0]) != 0, np.inf, 0)
        return v

conjugate_weighted_pairs = {}
for n1, n2 in [(l1norm,supnorm)]:
    conjugate_weighted_pairs[n1] = n2
    conjugate_weighted_pairs[n2] = n1
