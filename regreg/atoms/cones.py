from copy import copy
import warnings

from scipy import sparse
import numpy as np

from ..problems.composite import nonsmooth, smooth_conjugate
from ..affine import linear_transform, identity as identity_transform
from ..identity_quadratic import identity_quadratic
from ..smooth import affine_smooth
from ..atoms import _work_out_conjugate, atom, affine_atom
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)

from .projl1_cython import projl1_epigraph

@objective_doc_templater()
class cone(atom):

    """
    A class that defines the API for cone constraints.
    """

    objective_template = r'\|%(var)s\|'
    objective_vars = {'var': r'\beta', 
                      'shape':'p', 
                      'linear':'D', 
                      'offset':r'\alpha',
                      'coneklass':'nonnegative',
                      'dualconeklass':'nonpositive',
                      'initargs':'(30,)', # args need to construct penalty
                      }
    tol = 1.0e-05

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            return self.shape == other.shape
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              offset=copy(self.offset),
                              initial=copy(self.coefs),
                              quadratic=copy(self.quadratic))
    
    def __repr__(self):
        if self.quadratic.iszero:
            return "%s(%s, offset=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.offset))
        else:
            return "%s(%s, offset=%s, quadratic=%s)" % \
                (self.__class__.__name__,
                 repr(self.shape), 
                 repr(self.offset),
                 repr(self.quadratic))

    @doc_template_user
    @doc_template_provider
    def get_conjugate(self):
        """
        Return the conjugate of an given atom.

        >>> import regreg.api as rr
        >>> penalty = rr.%(coneklass)s(%(initargs)s)
        >>> penalty.get_conjugate() # doctest: +SKIP
        %(dualconeklass)s(%(initargs)s, offset=None)

        """
        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, 
                                               self.quadratic)
            cls = conjugate_cone_pairs[self.__class__]
            atom = cls(self.shape, 
                       offset=offset,
                       quadratic=outq)
        else:
            atom = smooth_conjugate(self)
        self._conjugate = atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @doc_template_user
    @doc_template_provider
    def get_dual(self):
        r"""
        Return the dual of an atom. This dual is formed by making the  
        substitution $v=Ax$ where $A$ is the `self.linear_transform`.

        >>> import regreg.api as rr
        >>> penalty = rr.%(coneklass)s(%(initargs)s)
        >>> penalty # doctest: +SKIP
        %(coneklass)s(%(initargs)s, offset=None)
        >>> penalty.dual # doctest: +SKIP
        (<regreg.affine.identity object at 0x...>, %(dualconeklass)s(%(initargs)s, offset=None))

        If there is a linear part to the penalty, the linear_transform may not be identity:

        >>> D = (np.identity(4) + np.diag(-np.ones(3),1))[:-1]
        >>> D
        array([[ 1., -1.,  0.,  0.],
               [ 0.,  1., -1.,  0.],
               [ 0.,  0.,  1., -1.]])
        >>> linear_atom = rr.nonnegative.linear(D)
        >>> linear_atom # doctest: +ELLIPSIS, +NORMALIZE_WHITESPACE
        affine_cone(nonnegative((3,), offset=None), array([[ 1., -1.,  0.,  0.],
               [ 0.,  1., -1.,  0.],
               [ 0.,  0.,  1., -1.]]))
        >>> linear_atom.dual # doctest: +ELLIPSIS
        (<regreg.affine.linear_transform object at 0x...>, nonpositive((3,), offset=None))

        """
        return self.linear_transform, self.conjugate

    @property
    def linear_transform(self):
        if not hasattr(self, "_linear_transform"):
            self._linear_transform = identity_transform(self.shape)
        return self._linear_transform
    
    @doc_template_user
    @doc_template_provider
    def constraint(self, x):
        """
        The constraint

        .. math::

           %(objective)s
        """
        raise NotImplementedError

    @doc_template_user
    @doc_template_provider
    def nonsmooth_objective(self, x, check_feasibility=False):
        '''
        >>> import regreg.api as rr
        >>> cone = rr.nonnegative(4)
        >>> cone.nonsmooth_objective([3, 4, 5, 9])
        0.0
        '''
        arg = np.asarray(x)
        x_offset = self.apply_offset(x)
        if check_feasibility:
            v = self.constraint(x_offset)
        else:
            v = 0
        v += self.quadratic.objective(arg, 'func')
        return v

    @doc_template_user
    @doc_template_provider
    def proximal(self, quadratic, prox_control=None):
        r"""
        The proximal operator. 

        .. math::

           v^{\lambda}(x) = \text{argmin}_{v \in \mathbb{R}^{%(shape)s}} \frac{L}{2}
           \|x-\alpha - v\|^2_2 + %(objective)s + \langle v, \eta \rangle

        where :math:`\alpha` is `self.offset`,
        :math:`\eta` is `quadratic.linear_term`.

        >>> import regreg.api as rr
        >>> cone = rr.nonnegative((4,))
        >>> Q = rr.identity_quadratic(1.5, [3, -4, -1, 1], 0, 0)
        >>> np.allclose(cone.proximal(Q), [3, 0, 0, 1]) # doctest: +NORMALIZE_WHITESPACE
        True

        Parameters
        ----------

        quadratic : `regreg.identity_quadratic.identity_quadratic`

            A quadratic added to the atom before minimizing.

        prox_control : `[None, dict]`

            This argument is ignored for seminorms, but otherwise
            is passed to `regreg.algorithms.FISTA` if the atom
            needs to be solved iteratively.

        Returns
        -------

        Z : `np.ndarray(np.float)`
            The proximal map of the implied center of `quadratic`.

        """
        offset, totalq = (self.quadratic + quadratic).recenter(self.offset)
        if totalq.coef == 0:
            raise ValueError('lipschitz + quadratic coef must be positive')

        prox_arg = -totalq.linear_term / totalq.coef

        eta = self.cone_prox(prox_arg)
        if offset is None:
            return eta
        else:
            return eta + offset

    @doc_template_user
    @doc_template_provider
    def cone_prox(self, x):
        r"""
        Return (unique) minimizer

        .. math::

           %(var)s^{\lambda}(u) = \text{argmin}_{%(var)s \in \mathbb{R}^%(shape)s} 
           \frac{1}{2} \|%(var)s-u\|^2_2 + %(objective)s

        """
        raise NotImplementedError

    # the minus signs below for offset is there until affine transforms SUBTRACT 
    # their offset until add. 
    # for atoms, the offset is really the "center"

    @classmethod
    @doc_template_provider
    def linear(cls, linear_operator, diag=False,
               offset=None,
               quadratic=None):
        """
        Composition of a cone constraint and a linear
        transform.
        """

        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        if offset is None:
            offset = 0
        cone = cls(l.output_shape, 
                   offset=-offset,
                   quadratic=quadratic)
        return affine_cone(cone, l)

    @classmethod
    @doc_template_provider
    def affine(cls, linear_operator, offset, diag=False,
               quadratic=None):
        """
        Composition of a cone constraint and a linear
        transform.
        """
        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        if offset is None:
            offset = 0
        cone = cls(l.output_shape, 
                   offset=-offset,
                   quadratic=quadratic)
        return affine_cone(cone, l)

    @staticmethod
    def check_subgradient(atom, prox_center):
        r"""
        For a given seminorm, verify the KKT condition for
        the problem for the proximal problem

        .. math::

           \text{minimize}_u \frac{1}{2} \|u-z\|^2_2 + h(z)

        where $z$ is the `prox_center` and $h$ is `atom`.

        This should return two values that are 0, 
        one is the inner product of the minimizer and the residual, the
        other is just 0.

        Parameters
        ----------

        atom : `cone`
             A cone instance with a `proximal` method.

        prox_center : np.ndarray(np.float)
             Center for the proximal map.

        Returns
        -------

        v1, v2 : float
             Two values that should be equal if the proximal map is correct.

        """
        atom = copy(atom)
        atom.quadratic = identity_quadratic(0,0,0,0)
        atom.offset = None
        q = identity_quadratic(1, prox_center, 0, 0)
        U = atom.proximal(q)
        return ((prox_center - U) * U).sum(), 0


@objective_doc_templater()
class affine_cone(affine_atom):

    def __repr__(self):
        return "affine_cone(%s, %s)" % (repr(self.atom),
                                        repr(self.linear_transform.linear_operator))

@objective_doc_templater()
class nonnegative(cone):
    """
    The non-negative cone constraint (which is the support
    function of the non-positive cone constraint).
    """

    objective_template = r"""I^{\infty}(%(var)s \succeq 0)"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'nonnegative'
    objective_vars['dualconeklass'] = 'nonpositive'

    @doc_template_user
    def constraint(self, x):
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.greater_equal(x, -tol_lim))
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x):
        return np.maximum(x, 0)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class nonpositive(nonnegative):

    """
    The non-positive cone constraint (which is the support
    function of the non-negative cone constraint).
    """

    objective_template = r"""I^{\infty}(%(var)s \preceq 0)"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['dualconeklass'] = 'nonnegative'
    objective_vars['coneklass'] = 'nonpositive'

    @doc_template_user
    def constraint(self, x):
        tol_lim = np.fabs(x).max() * self.tol
        incone = np.all(np.less_equal(x, tol_lim))
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x):
        return np.minimum(x, 0)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class zero(cone):
    """
    The zero seminorm, support function of :math:\{0\}
    """

    objective_template = r"""{\cal Z}(%(var)s)"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'zero'
    objective_vars['dualconeklass'] = 'zero_constraint'

    @doc_template_user
    def constraint(self, x):
        return 0.

    @doc_template_user
    def cone_prox(self, x):
        return np.asarray(x)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class zero_constraint(cone):
    """
    The zero constraint, support function of :math:`\mathbb{R}`^p
    """

    objective_template = r"""I^{\infty}(%(var)s = 0)"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'zero_constraint'
    objective_vars['dualconeklass'] = 'zero'

    @doc_template_user
    def constraint(self, x):
        if not np.linalg.norm(x) <= self.tol:
            return np.inf
        return 0.

    @doc_template_user
    def cone_prox(self, x):
        return np.zeros(np.asarray(x).shape)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class l2_epigraph(cone):

    """
    The l2_epigraph constraint.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_2 \leq %(var)s[-1])"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'l2_epigraph'
    objective_vars['dualconeklass'] = 'l2_epigraph_polar'

    @doc_template_user
    def constraint(self, x):
        
        incone = np.linalg.norm(x[:-1]) <= (1 + self.tol) * x[-1]
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x):
        norm = x[-1]
        coef = x[:-1]
        norm_coef = np.linalg.norm(coef)
        thold = (norm_coef - norm) / 2.
        result = np.zeros_like(x)
        result[:-1] = coef / norm_coef * max(norm_coef - thold, 0)
        result[-1] = max(norm + thold, 0)
        return result

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class l2_epigraph_polar(cone):

    """
    The polar of the l2_epigraph constraint, which is the negative of the 
    l2 epigraph..
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_2 \in -%(var)s[-1])"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['dualconeklass'] = 'l2_epigraph'
    objective_vars['coneklass'] = 'l2_epigraph_polar'

    @doc_template_user
    def constraint(self, x):
        incone = np.linalg.norm(x[:-1]) <= (1 + self.tol) * (-x[-1])
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, arg):
        arg = -arg
        norm = arg[-1]
        coef = arg[:-1]
        norm_coef = np.linalg.norm(coef)
        thold = (norm_coef - norm) / 2.
        result = np.zeros_like(arg)
        result[:-1] = coef / norm_coef * max(norm_coef - thold, 0)
        result[-1] = max(norm + thold, 0)
        return -result

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class l1_epigraph(cone):

    """
    The l1_epigraph constraint.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_1 \leq %(var)s[-1])"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['dualconeklass'] = 'l1_epigraph'
    objective_vars['coneklass'] = 'l1_epigraph_polar'

    @doc_template_user
    def constraint(self, x):
        incone = np.fabs(x[:-1]).sum() <= (1 + self.tol) * x[-1]
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, x):
        return projl1_epigraph(x)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class l1_epigraph_polar(cone):

    """
    The polar l1_epigraph constraint.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_{\infty} \leq - %(var)s[-1])"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'l1_epigraph_polar'
    objective_vars['dualconeklass'] = 'l1_epigraph'

    @doc_template_user
    def constraint(self, x):
        
        incone = np.fabs(-x[:-1]).max() <= (1 + self.tol) * ( -x[-1])
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, arg):
        arg = np.asarray(arg, np.float).copy()
        return arg - projl1_epigraph(arg)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class linf_epigraph(cone):

    """
    The $\ell_{\nfty}$ epigraph constraint.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_{\infty} \leq %(var)s[-1])"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'linf_epigraph'
    objective_vars['dualconeklass'] = 'linf_epigraph_polar'

    @doc_template_user
    def constraint(self, x):
        incone = np.fabs(x[:-1]).max() <= (1 + self.tol) * x[-1]
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, arg):
        arg = np.asarray(arg, np.float)
        return arg + projl1_epigraph(-arg)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

@objective_doc_templater()
class linf_epigraph_polar(cone):

    """
    The polar linf_epigraph constraint which is just the
    negative of the l1_epigraph.
    """

    objective_template = r"""I^{\infty}(\|%(var)s[:-1]\|_1 \leq -%(var)s[-1])"""
    objective_vars = cone.objective_vars.copy()
    objective_vars['coneklass'] = 'linf_epigraph_polar'
    objective_vars['dualconeklass'] = 'linf_epigraph'

    @doc_template_user
    def constraint(self, x):
        incone = np.fabs(x[:-1]).sum() <= (1 + self.tol) * (-x[-1])
        if incone:
            return 0
        return np.inf

    @doc_template_user
    def cone_prox(self, arg):
        arg = np.asarray(arg, np.float)
        return -projl1_epigraph(-arg)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return cone.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_conjugate(self):
        return cone.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return cone.dual(self)

conjugate_cone_pairs = {}
for n1, n2 in [(nonnegative,nonpositive),
               (zero, zero_constraint),
               (l1_epigraph, l1_epigraph_polar),
               (l2_epigraph, l2_epigraph_polar),
               (linf_epigraph, linf_epigraph_polar)
               ]:
    conjugate_cone_pairs[n1] = n2
    conjugate_cone_pairs[n2] = n1

