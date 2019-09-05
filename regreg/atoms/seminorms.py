from __future__ import print_function, division, absolute_import

from copy import copy

import numpy as np

from ..identity_quadratic import identity_quadratic
from ..affine import (linear_transform, 
                      identity as identity_transform)
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, 
                            doc_template_provider)
from ..problems.composite import smooth_conjugate
from ..atoms import atom, _work_out_conjugate, affine_atom
from .projl1_cython import projl1
from .piecewise_linear import find_solution_piecewise_linear_c

@objective_doc_templater()
class seminorm(atom):
    """
    An atom that can be in lagrange or bound form.
    """

    objective_template = r'\|%(var)s\|'
    objective_vars = {'var': r'\beta', 
                      'shape':'p', 
                      'linear':'D', 
                      'offset':r'\alpha',
                      'normklass':'l1norm',
                      'dualnormklass':'supnorm',
                      'initargs':'(30,)', # args need to construct penalty
                      }

    def __init__(self, shape, lagrange=None, bound=None,
                 offset=None, quadratic=None, initial=None):

        atom.__init__(self, shape, offset,
                      quadratic, initial)

        if not (bound is None or lagrange is None):
            raise ValueError('An atom must be either in Lagrange form or ' 
                             + 'bound form. Only one of the parameters '
                             + 'in the constructor can not be None.')
        if bound is None and lagrange is None:
            raise ValueError('Atom must be in lagrange or bound form, '
                             + 'as specified by the choice of one of'
                             + 'the keyword arguments.')
        if bound is not None and bound < 0:
            raise ValueError('Bound on the seminorm should be non-negative')
        if lagrange is not None and lagrange < 0:
            raise ValueError('Lagrange multiplier should be non-negative')

        if lagrange is not None:
            self._lagrange = lagrange
            self._bound = None
        if bound is not None:
            self._bound = bound
            self._lagrange = None
    
    def latexify(self, var=None, idx=''):
        r'''
        Return a LaTeX representation of an object.

        >>> import regreg.api as rr
        >>> penalty = rr.l1norm(10, lagrange=0.9)
        >>> penalty.latexify(var=r'\gamma') 
        '\\lambda_{} \\|\\gamma\\|_1'

        Parameters
        ----------

        var : `string`
            Argument of the functions

        idx : `string`
            Optional subscript index.

        Returns
        -------

        L : `string`
            A LaTeX representation of the atom.

        '''
        template_dict = self.objective_vars.copy()
        template_dict['idx'] = idx
        if var is not None:
            template_dict['var'] = var
        if self.offset is not None and np.any(self.offset != 0):
            template_dict['var'] = template_dict['var'] + (r' - %(offset)s_{%(idx)s}' % template_dict) 

        obj = self.objective_template % template_dict
        template_dict['obj'] = obj
        if self.lagrange is not None:
            obj = r'\lambda_{%(idx)s} %(obj)s' % template_dict
        else:
            obj = r'I^{\infty}(%(obj)s \leq \delta_{%(idx)s})' % template_dict

        if not self.quadratic.iszero:
            return ' + '.join([obj, self.quadratic.latexify(var=template_dict['var'], idx=idx)])
        return obj

    @doc_template_provider
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        r"""
        Return :math:`\lambda \cdot %(objective)s`, where
        :math:`\lambda` is `lagrange`. If `check_feasibility`
        is `True`, and seminorm is unbounded, will return `np.inf`
        if appropriate.

        The class seminorm's seminorm just returns the appropriate lagrange
        parameter for use by the subclasses.
        """
        return get_lagrange(self, lagrange)

    @doc_template_provider
    def constraint(self, arg, bound=None):
        r"""
        Verify :math:`%(objective)s \leq \delta`, where :math:`\delta`
        is `bound`.

        If the result is `True`, returns 0, else returns `np.inf`.

        The class seminorm's constraint just returns the appropriate bound
        parameter for use by the subclasses.
        """
        return get_bound(self, bound)

    def __eq__(self, other):
        if self.__class__ == other.__class__:
            if self.bound is not None:
                return self.bound == other.bound
            return self.lagrange == other.lagrange
        return False

    def __copy__(self):
        return self.__class__(copy(self.shape),
                              bound=copy(self.bound),
                              lagrange=copy(self.lagrange),
                              offset=copy(self.offset),
                              quadratic=copy(self.quadratic))

    def __repr__(self):
        if self.lagrange is not None:
            if self.quadratic.iszero:
                return "%s(%s, lagrange=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     repr(self.shape), 
                     self.lagrange,
                     repr(self.offset))
            else:
                return "%s(%s, lagrange=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     repr(self.shape), 
                     self.lagrange,
                     repr(self.offset),
                     repr(self.quadratic))

        else:
            if self.quadratic.iszero:
                return "%s(%s, bound=%f, offset=%s)" % \
                    (self.__class__.__name__,
                     repr(self.shape), 
                     self.bound,
                     repr(self.offset))

            else:
                return "%s(%s, bound=%f, offset=%s, quadratic=%s)" % \
                    (self.__class__.__name__,
                     repr(self.shape), 
                     self.bound,
                     repr(self.offset),
                     repr(self.quadratic))

    @doc_template_user
    @doc_template_provider
    def get_conjugate(self):
        """
        Return the conjugate of an given atom.

        >>> import regreg.api as rr
        >>> penalty = rr.%(normklass)s(%(initargs)s, lagrange=3.4)
        >>> penalty.get_conjugate() # doctest: +ELLIPSIS
        %(dualnormklass)s(..., bound=3.4...)

        """

        if self.quadratic.coef == 0:
            offset, outq = _work_out_conjugate(self.offset, self.quadratic)

            cls = conjugate_seminorm_pairs[self.__class__]
            conjugate_atom = cls(self.shape,  \
                       bound=self.lagrange, 
                       lagrange=self.bound,
                       quadratic=outq,
                       offset=offset)
        else:
            conjugate_atom = smooth_conjugate(self)
        self._conjugate = conjugate_atom
        self._conjugate._conjugate = self
        return self._conjugate
    conjugate = property(get_conjugate)

    @doc_template_user
    @doc_template_provider
    def get_lagrange(self):
        """
        Get method of the lagrange property.

        >>> import regreg.api as rr
        >>> penalty = rr.%(normklass)s(%(initargs)s, lagrange=3.4)
        >>> penalty.lagrange
        3.4

        """
        return self._lagrange

    @doc_template_user
    @doc_template_provider
    def set_lagrange(self, lagrange):
        """
        Set method of the lagrange property.

        >>> import regreg.api as rr
        >>> penalty = rr.%(normklass)s(%(initargs)s, lagrange=3.4)
        >>> penalty.lagrange
        3.4
        >>> penalty.lagrange = 2.3
        >>> penalty.lagrange
        2.3
        >>> constraint = rr.%(normklass)s(%(initargs)s, bound=3.4)
        >>> constraint.lagrange = 3.4 #doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        AttributeError: atom is in bound mode

        """
        if self.bound is None:
            self._lagrange = lagrange
        else:
            raise AttributeError("atom is in bound mode")
    lagrange = property(get_lagrange, set_lagrange)

    @doc_template_user
    @doc_template_provider
    def get_bound(self):
        """
        Get method of the bound property.

        >>> import regreg.api as rr
        >>> constraint = rr.%(normklass)s(%(initargs)s, bound=2.3) 
        >>> constraint.bound
        2.3

        """
        return self._bound

    @doc_template_user
    @doc_template_provider
    def set_bound(self, bound):
        """
        Set method of the bound property.

        >>> import regreg.api as rr
        >>> constraint = rr.%(normklass)s(%(initargs)s, bound=3.4) 
        >>> constraint.bound
        3.4
        >>> constraint.bound = 2.3
        >>> constraint.bound
        2.3
        >>> penalty = rr.%(normklass)s(%(initargs)s, lagrange=2.3)
        >>> penalty.bound = 3.4 # doctest: +ELLIPSIS +IGNORE_EXCEPTION_DETAIL
        Traceback (most recent call last):
        ...
        AttributeError: atom is in lagrange mode
        """
        if self.lagrange is None:
            self._bound = bound
        else:
            raise AttributeError("atom is in lagrange mode")
    bound = property(get_bound, set_bound)

    @doc_template_user
    @doc_template_provider
    def proximal(self, quadratic, prox_control=None):
        r"""
        The proximal operator. If the atom is in
        Lagrange mode, this has the form

        .. math::

           %(var)s^{\lambda}(\theta) = \text{argmin}_{%(var)s \in
           \mathbb{R}^{%(shape)s}} \frac{L}{2} \|\theta-%(var)s\|^2_2
           + \lambda h(%(var)s-\alpha) + \langle %(var)s, \eta \rangle

        where :math:`\alpha` is `self.offset`,
        :math:`\eta` is `quadratic.linear_term`, $\theta$ is `quadratic.center` and 

        .. math::

           h(%(var)s) = %(objective)s

        If the atom is in bound mode, then this has the form

        .. math::

           %(var)s^{\delta}(\theta) = \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} \frac{L}{2}
           \|\theta-%(var)s\|^2_2 + \langle %(var)s, \eta \rangle \  \text{s.t.} \   
           h(%(var)s - \alpha) \leq \delta

        >>> import regreg.api as rr
        >>> penalty = rr.l1norm(4, lagrange=2)
        >>> Q = rr.identity_quadratic(1.5, [3, -4, -1, 1], 0, 0)
        >>> penalty.proximal(Q) # doctest: +ELLIPSIS
        array([ 1.6666..., -2.6666..., -0.        ,  0.        ])

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

        debug = False
        if debug:
            print('='*80)
            print('atom: ', self)
            print('quadratic: ', quadratic)
            print('proxarg: ', prox_arg)
            print('totalq: ', totalq)
            print('offset: ', offset)

        if self.bound is not None:
            eta = self.bound_prox(prox_arg, 
                                  bound=self.bound)
        else:
            eta = self.lagrange_prox(prox_arg, 
                                     lipschitz=totalq.coef, 
                                     lagrange=self.lagrange)

        if offset is None:
            return eta
        else:
            return eta + offset

    @doc_template_provider
    def lagrange_prox(self, arg, lipschitz=1, lagrange=None):
        r"""
        Return unique minimizer

        .. math::

           {%(var)s}^{\lambda}(\theta) =
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{L}{2}
           \|\theta-%(var)s\|^2_2 + \lambda %(objective)s 

        Above, :math:`\lambda` is the Lagrange parameter and :math:`L`
        is the Lipschitz parameter and $\theta$ is `arg`.

        If the argument `lagrange` is None and the atom is in lagrange mode,
        ``self.lagrange`` is used as the lagrange parameter, else an exception
        is raised.

        The class atom's lagrange_prox just returns the appropriate lagrange
        parameter for use by the subclasses.

        Parameters
        ----------

        arg : `np.ndarray(np.float)`
            Argument of the proximal map.

        lipschitz : `float`
            Coefficient in front of the quadratic.

        lagrange : `float` (optional)
            Lagrange factor in front of the seminorm. 
            Defaults to `self.lagrange`.

        Returns
        -------

        Z : `np.ndarray(np.float)`
            The proximal map of `arg`.

        """
        return get_lagrange(self, lagrange)

    @doc_template_provider
    def bound_prox(self, arg, bound=None):
        r"""
        Return unique minimizer

        .. math::

           {%(var)s}^{\delta}(\theta) =
           \text{argmin}_{%(var)s \in \mathbb{R}^{%(shape)s}} 
           \frac{1}{2}
           \|\theta-%(var)s\|^2_2  \ 
           \text{s.t.} \   %(objective)s \leq \delta

        where :math:`\delta` is the bound parameter and $\theta$ is `arg`. 

        If the argument `bound` is None and the atom is in bound mode,
        ``self.bound`` is used as the bound parameter, else an exception is
        raised.

        The class atom's bound_prox just returns the appropriate bound
        parameter for use by the subclasses.

        Parameters
        ----------

        arg : `np.ndarray(np.float)`
            Argument of the proximal map.

        bound : `float` (optional)
            Bound for the constraint on the seminorm.
            Defaults to `self.bound`.

        Returns
        -------

        Z : `np.ndarray(np.float)`
            The proximal map of `arg`.

        """
        return get_bound(self, bound)

    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        The nonsmooth objective function of the atom.
        Includes `self.quadratic.objective(arg)`.

        >>> import regreg.api as rr
        >>> penalty = rr.l1norm(4, lagrange=2)
        >>> penalty.nonsmooth_objective([3, 4, 5, 9])
        42.0
        >>> 2 * sum([3, 4, 5, 9])
        42


        Parameters
        ----------

        arg : `np.ndarray(np.float)`
            Argument of the seminorm.

        check_feasibility : `bool`
            If `True`, then return `np.inf` if appropriate.

        Returns
        -------

        value : `np.float`
            The seminorm of `arg`.

        """
        arg = np.asarray(arg)
        x_offset = self.apply_offset(arg)
        if self.bound is not None:
            if check_feasibility:
                value = self.constraint(x_offset)
            else:
                value = 0
        else:
            value = self.seminorm(x_offset, check_feasibility=check_feasibility)
        value += self.quadratic.objective(arg, 'func')
        return value

    @doc_template_user
    @doc_template_provider
    def get_dual(self):
        """
        Return the dual of an atom. This dual is formed by making introducing
        new variables $v=Ax$ where $A$ is `self.linear_transform`. 

        >>> import regreg.api as rr
        >>> penalty = rr.%(normklass)s(%(initargs)s, lagrange=2.3)
        >>> penalty # doctest: +ELLIPSIS
        %(normklass)s(..., lagrange=2.3...)
        >>> penalty.dual # doctest: +ELLIPSIS
        (<regreg.affine.identity object at 0x...>, %(dualnormklass)s(..., bound=2.3...))

        If there is a linear part to the penalty, the linear_transform may not be 
        identity. For example, the 1D fused LASSO penalty:

        >>> D = (np.identity(4) + np.diag(-np.ones(3),1))[:-1]
        >>> D
        array([[ 1., -1.,  0.,  0.],
               [ 0.,  1., -1.,  0.],
               [ 0.,  0.,  1., -1.]])
        >>> linear_atom = rr.l1norm.linear(D, lagrange=2.3)
        >>> linear_atom # doctest: +ELLIPSIS
        affine_atom(l1norm((3,), lagrange=2.3...), array([[ 1., -1.,  0.,  0.],
               [ 0.,  1., -1.,  0.],
               [ 0.,  0.,  1., -1.]]))
        >>> linear_atom.dual # doctest: +ELLIPSIS
        (<regreg.affine.linear_transform object at 0x...>, supnorm((3,), bound=2.3...))

        """
        return self.linear_transform, self.conjugate

    @classmethod
    def affine(cls, linear_operator, offset, lagrange=None, diag=False,
               bound=None, quadratic=None):
        """
        This is the same as the linear class method but with offset as a positional argument
        """
        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        # the minus signs below for offset is there until affine transforms SUBTRACT 
        # their offset until add. 
        # for atoms, the offset is really the "center"

        if offset is None:
            offset = 0
        new_atom = cls(l.output_shape, 
                       lagrange=lagrange, 
                       bound=bound,
                       offset=-offset,
                       quadratic=quadratic)
        return affine_atom(new_atom, l)

    @classmethod
    def linear(cls, linear_operator, lagrange=None, diag=False,
               bound=None, quadratic=None, offset=None):
        if not isinstance(linear_operator, linear_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        if offset is None:
            offset = 0
        new_atom = cls(l.output_shape, lagrange=lagrange, bound=bound,
                   quadratic=quadratic, offset=-offset)
        return affine_atom(new_atom, l)


    @classmethod
    def shift(cls, offset, lagrange=None, diag=False,
              bound=None, quadratic=None):
        new_atom = cls(offset.shape, lagrange=lagrange, bound=bound,
                       quadratic=quadratic, offset=-offset)
        return new_atom

    @staticmethod
    def check_subgradient(atom, prox_center):
        r"""
        For a given seminorm, verify the KKT condition for
        the problem for the proximal problem

        .. math::

            \text{minimize}_u \frac{1}{2} \|u-z\|^2_2 + h(z)

        where $z$ is the `prox_center` and $h$ is `atom`
        which may be in Lagrange or bound form.

        If the atom is in Lagrange form, this function should
        return two values equal to the seminorm of the 
        minimizer. If it is bound form it should return two values
        equal to the dual seminorm of the residual, i.e.
        the prox_center minus the minimizer.

        Parameters
        ----------

        atom : `seminorm`

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
        dual = atom.conjugate
        if atom.bound is not None: # atom is bound mode
            if not atom.seminorm(U, lagrange=1) <= (1 + 1.e-5) * atom.bound:
                raise ValueError('expecting residual to have norm less than or equal to bound parameter')
            return ((prox_center - U) * U).sum() / atom.bound, dual.seminorm(prox_center-U, lagrange=1)
        else:
            if not dual.seminorm(prox_center - U, lagrange=1) <= (1 + 1.e-5) * atom.lagrange:
                raise ValueError('expecting residual to have norm less than equal to lagrange parameter: (%f, %f)' % (dual.seminorm(prox_center - U, lagrange=1), atom.lagrange))
            return ((prox_center - U) * U).sum() / atom.lagrange, atom.seminorm(U, lagrange=1)

@objective_doc_templater()
class l1norm(seminorm):

    """
    The l1 norm
    """
    prox_tol = 1.0e-10

    objective_template = r"""\|%(var)s\|_1"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'l1norm'
    objective_vars['dualnormklass'] = 'supnorm'

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                     check_feasibility=check_feasibility, 
                                     lagrange=lagrange)
        return lagrange * np.fabs(arg).sum()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inbox = np.fabs(arg).sum() <= bound * (1 + self.tol)
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        return np.sign(arg) * np.maximum(np.fabs(arg)-lagrange/lipschitz, 0)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        arg = np.asarray(arg, np.float)
        absarg = np.fabs(arg)
        cut = find_solution_piecewise_linear_c(bound, 0, absarg)
        if cut < np.inf:
            return np.sign(arg) * (absarg - cut) * (absarg > cut)
        return arg

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class supnorm(seminorm):

    r"""
    The :math:`\ell_{\infty}` norm
    """

    objective_template = r"""\|%(var)s\|_{\infty}"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'supnorm'
    objective_vars['dualnormklass'] = 'l1norm'

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return lagrange * np.fabs(arg).max()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inbox = np.product(np.less_equal(np.fabs(arg), bound * (1+self.tol)))
        if inbox:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg, np.float)
        absarg = np.fabs(arg)
        cut = find_solution_piecewise_linear_c(lagrange / lipschitz, 0, absarg)
        if cut < np.inf:
            proj = np.sign(arg) * (absarg - cut) * (absarg > cut)
        else:
            proj = arg
        return arg - proj

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        return np.clip(arg, -bound, bound)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class l2norm(seminorm):

    """
    The l2 norm
    """

    objective_template = r"""\|%(var)s\|_2"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'l2norm'
    objective_vars['dualnormklass'] = 'l2norm'

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                     check_feasibility=check_feasibility, 
                                     lagrange=lagrange)
        return lagrange * np.linalg.norm(arg)

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inball = (np.linalg.norm(arg) <= bound * (1 + self.tol))
        if inball:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        n = np.linalg.norm(arg)
        if n <= lagrange / lipschitz:
            proj = arg
        else:
            proj = (lagrange / (lipschitz * n)) * arg
        return arg - proj

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        n = np.linalg.norm(arg)
        if n <= bound:
            return arg
        else:
            return (bound / n) * arg

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

def positive_part_lagrange(shape, lagrange,
                           offset=None, quadratic=None, initial=None):
    r'''
    The positive_part atom in lagrange form can be represented
    by an l1norm atom with the addition of a linear term
    and half the lagrange parameter. This reflects the fact that
    :math:`[0,1]^p = [-1/2,1/2]^p + 1/2 \pmb{1}`.

    '''
    lin = np.ones(shape) * .5 * lagrange
    linq = identity_quadratic(0,0,lin,0)
    if quadratic is not None:
        linq = linq + quadratic
    return l1norm(shape, lagrange=0.5*lagrange,
                  offset=offset, quadratic=linq,
                  initial=initial)

@objective_doc_templater()
class positive_part(seminorm):

    """
    The positive_part seminorm (which is the support
    function of [0,l]^p).
    """

    objective_template = r"""\left(\sum_{i=1}^{%(shape)s} (%(var)s)_i^+\right)"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'positive_part'
    objective_vars['dualnormklass'] = 'constrained_max'

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return lagrange * np.maximum(arg, 0).sum()

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        inside = np.maximum(arg, 0).sum() <= bound * (1 + self.tol)
        if inside:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg)
        v = arg.copy()
        pos = v > 0
        v = np.atleast_1d(v)
        v[pos] = np.maximum(v[pos] - lagrange/lipschitz, 0)
        return v.reshape(arg.shape)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        arg = np.asarray(arg)
        v = arg.copy().astype(np.float)
        v = np.atleast_1d(v)
        pos = v > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], bound)
        return v.reshape(arg.shape)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class constrained_max(seminorm):
    r"""
    The seminorm x.max() s.t. x geq 0.
    """

    objective_template = (r"""\left\|%(var)s\right\|_{\infty} + """
                          + r"""I^{\infty}\left(\min(%(var)s) \in [0,+\infty)\right) """)
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'constrained_max'
    objective_vars['dualnormklass'] = 'positive_part'

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        anyneg = np.any(arg < 0 + self.tol)
        v = lagrange * np.max(arg)
        if not anyneg or not check_feasibility:
            return v
        return np.inf

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        anyneg = np.any(arg < 0 + self.tol)
        inside = np.max(arg) <= bound * (1 + self.tol)
        if inside and not anyneg:
            return 0
        else:
            return np.inf

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg)
        v = arg.copy().astype(np.float)
        v = np.atleast_1d(v)
        pos = v > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], lagrange/lipschitz)
        return arg - v.reshape(arg.shape)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        return np.clip(arg, 0, bound)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class constrained_positive_part(seminorm):

    r"""
    Support function of $[-\infty,1]^p$
    """

    objective_template = (r"""\|%(var)s\|_{1} + \sum_{i=1}^{%(shape)s} """
                          + r"""\delta_{[0,+\infty]}(%(var)s_i)""")
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'constrained_positive_part'
    objective_vars['dualnormklass'] = 'max_positive_part'

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        anyneg = np.any(arg < 0 + self.tol)
        v = np.maximum(arg, 0).sum()
        if not anyneg or not check_feasibility:
            return v * lagrange
        return np.inf

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        value = self.seminorm(arg, lagrange=1, check_feasibility=True)
        if value >= bound * (1 + self.tol):
            return np.inf
        return 0

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg)
        v = np.zeros(arg.shape)
        v = np.atleast_1d(v)
        pos = arg > 0
        if np.any(pos):
            v[pos] = np.maximum(arg[pos] - lagrange/lipschitz, 0)
        return v.reshape(arg.shape)

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        arg = np.asarray(arg)
        v = np.zeros(arg.shape, np.float)
        v = np.atleast_1d(v)
        pos = arg > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], bound)
        return v.reshape(arg.shape)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

@objective_doc_templater()
class max_positive_part(seminorm):

    """
    support function of the standard simplex
    """

    objective_template = r"""\|%(var)s^+\|_{\infty}"""
    objective_vars = seminorm.objective_vars.copy()
    objective_vars['normklass'] = 'max_positive_part'
    objective_vars['dualnormklass'] = 'constrained_positive_part'

    @doc_template_user
    def seminorm(self, arg, lagrange=None, check_feasibility=False):
        lagrange = seminorm.seminorm(self, arg, 
                                 check_feasibility=check_feasibility, 
                                 lagrange=lagrange)
        return np.max(np.maximum(x,0)) * lagrange

    @doc_template_user
    def constraint(self, arg, bound=None):
        bound = seminorm.constraint(self, arg, bound=bound)
        v = np.max(np.maximum(arg,0))
        if v >= bound * (1 + self.tol):
            return np.inf
        return 0

    @doc_template_user
    def bound_prox(self, arg, bound=None):
        bound = seminorm.bound_prox(self, arg, bound)
        return np.clip(arg, -np.inf, bound)

    @doc_template_user
    def lagrange_prox(self, arg,  lipschitz=1, lagrange=None):
        lagrange = seminorm.lagrange_prox(self, arg, lipschitz, lagrange)
        arg = np.asarray(arg)
        v = np.zeros(arg.shape, np.float)
        v = np.atleast_1d(v)
        pos = arg > 0
        if np.any(pos):
            v[pos] = projl1(v[pos], lagrange / lipschitz)
        return arg - v.reshape(arg.shape)

    @doc_template_user
    def proximal(self, quadratic, prox_control=None):
        return seminorm.proximal(self, quadratic, prox_control)

    @doc_template_user
    def get_bound(self):
        return seminorm.get_bound(self)

    @doc_template_user
    def set_bound(self, bound):
        return seminorm.set_bound(self, bound)

    @doc_template_user
    def get_lagrange(self):
        return seminorm.get_lagrange(self)

    @doc_template_user
    def set_lagrange(self, lagrange):
        return seminorm.set_lagrange(self, lagrange)

    @doc_template_user
    def get_conjugate(self):
        return seminorm.get_conjugate(self)

    @doc_template_user
    def get_dual(self):
        return seminorm.dual(self)

conjugate_seminorm_pairs = {}
for n1, n2 in [(l1norm,supnorm),
               (l2norm,l2norm),
               (positive_part, constrained_max),
               (constrained_positive_part, max_positive_part)]:
    conjugate_seminorm_pairs[n1] = n2
    conjugate_seminorm_pairs[n2] = n1

nonpaired_atoms = [positive_part_lagrange]

def get_lagrange(atom, lagrange=None):
    """
    Return appropriate `lagrange` parameter.
    """
    if lagrange is None:
        lagrange = atom.lagrange
    if lagrange is None:
        raise ValueError('either atom must be in Lagrange '
                         + 'mode or a keyword "lagrange" '
                         + 'argument must be supplied')
    return lagrange

def get_bound(atom, bound=None):
    """
    Return appropriate `bound` parameter.
    """
    if bound is None:
        bound = atom.bound
    if bound is None:
        raise ValueError('either atom must be in bound '
                         + 'mode or a keyword "bound" '
                         + 'argument must be supplied')
    return bound
