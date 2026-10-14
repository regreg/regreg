"""Smooth atoms: smooth functions together with their gradients.

This module defines the base class `smooth_atom` for smooth (differentiable)
objectives, the composition `affine_smooth` of a smooth atom with an affine
map, and a few simple smooth atoms (`zero`, `sum`).
"""

import numpy as np
from scipy import sparse
import warnings
import inspect

from ..problems.composite import smooth as smooth_composite
from ..affine import affine_transform, linear_transform, astransform
from ..identity_quadratic import identity_quadratic

#TODO: create proximal methods for known smooth things
class smooth_atom(smooth_composite):

    """A class for representing a smooth function and its gradient.

    Parameters
    ----------
    shape : int or tuple
        Shape of the argument to `smooth_objective`.
    coef : float, optional
        Scalar multiple to be applied (must be nonnegative).
    offset : ndarray, optional
        Vector to be subtracted from the argument before evaluating
        `smooth_objective`.
    quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
        Quadratic term to be added to the overall objective.
    initial : ndarray, optional
        Initial value for the coefficients.
    """

    objective_template = r'''f(%(var)s)'''
    objective_vars = {'var':r'\beta',
                      'shape':'p',
                      'coef':'C',
                      'offset':r'\alpha+'}

    def __init__(self, shape, coef=1, offset=None,
                 quadratic=None, initial=None):
        smooth_composite.__init__(self, shape,
                                  offset=offset,
                                  quadratic=quadratic,
                                  initial=initial)
        self.coef = coef
        if coef < 0:
            raise ValueError('coefs must be nonnegative to ensure convexity (assuming all atoms are indeed convex)')
        self.coefs = np.zeros(self.shape)

    def smooth_objective(self, arg, mode='both', check_feasibility=False):
        """Evaluate the smooth objective, its gradient, or both.

        Abstract method: subclasses must implement it.

        Parameters
        ----------
        arg : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `arg` is not in the domain.

        Returns
        -------
        f : float
            Objective value at `arg`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `arg`; returned when `mode` is 'grad'
            or 'both'.
        """
        raise NotImplementedError("Abstract method.")
    
    @classmethod
    def affine(cls, linear_operator, offset, coef=1, diag=False,
               quadratic=None, **kws):
        r"""Compose the atom with an affine map.

        The returned smooth function is :math:`\beta \mapsto f(X\beta + b)`
        where :math:`X` is `linear_operator`, :math:`b` is `offset` and
        :math:`f` is an instance of `cls`.

        Parameters
        ----------
        linear_operator : ndarray or `regreg.affine.affine_transform`
            Linear part of the affine map.
        offset : ndarray
            Offset added to the output of `linear_operator`.
        coef : float, optional
            Scalar multiple to be applied (must be nonnegative).
        diag : bool, optional
            Whether `linear_operator` acts diagonally, i.e. is a rescaling.
            Passed to `regreg.affine.linear_transform`.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        **kws
            Further keyword arguments passed to the constructor of `cls`.
            They must be arguments of ``cls.__init__`` other than ``shape``,
            ``coef``, ``quadratic``, ``initial`` and ``offset``.

        Returns
        -------
        atom : `affine_smooth`
            The composition of the atom with the affine map.
        """
        if not isinstance(linear_operator, affine_transform):
            l = linear_transform(linear_operator, diag=diag)
        else:
            l = linear_operator
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)
        
        # the minus signs below for offset is there until affine transforms SUBTRACT 
        # their offset until add. 
        # for atoms, the offset is really the "center"

        atom = cls(l.output_shape, coef=coef, offset=-offset, quadratic=quadratic, **kws)
        
        return affine_smooth(atom, l)

    @classmethod
    def linear(cls, linear_operator, coef=1, diag=False,
               offset=None, 
               quadratic=None, **kws):
        r"""Compose the atom with a linear map.

        The returned smooth function is :math:`\beta \mapsto f(X\beta)` where
        :math:`X` is `linear_operator` and :math:`f` is an instance of `cls`.

        Parameters
        ----------
        linear_operator : ndarray or `regreg.affine.affine_transform`
            The linear map.
        coef : float, optional
            Scalar multiple to be applied (must be nonnegative).
        diag : bool, optional
            Whether `linear_operator` acts diagonally, i.e. is a rescaling.
        offset : ndarray, optional
            Offset passed to the constructor of `cls`, i.e. subtracted from
            the output of `linear_operator` before evaluating the atom.
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        **kws
            Further keyword arguments passed to the constructor of `cls`.
            They must be arguments of ``cls.__init__`` other than ``shape``,
            ``coef``, ``quadratic``, ``initial`` and ``offset``.

        Returns
        -------
        atom : `affine_smooth`
            The composition of the atom with the linear map.
        """
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)

        atransform = affine_transform(linear_operator, None, diag=diag)
        atom = cls(atransform.output_shape, coef=coef, quadratic=quadratic, offset=offset, **kws)
        
        return affine_smooth(atom, atransform)

    @classmethod
    def shift(cls, offset, coef=1, quadratic=None, **kws):
        r"""Create an atom whose argument is shifted by `offset`.

        The returned smooth function is :math:`\beta \mapsto f(\beta - \alpha)`
        where :math:`\alpha` is `offset` and :math:`f` is an instance of `cls`.

        Parameters
        ----------
        offset : ndarray
            Vector subtracted from the argument; also determines the shape.
        coef : float, optional
            Scalar multiple to be applied (must be nonnegative).
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Quadratic term to be added to the objective.
        **kws
            Further keyword arguments passed to the constructor of `cls`.
            They must be arguments of ``cls.__init__`` other than ``shape``,
            ``coef``, ``quadratic``, ``initial`` and ``offset``.

        Returns
        -------
        atom : `cls`
            The shifted atom.
        """
        if not acceptable_init_args(cls, kws):
            raise ValueError("Invalid arguments being passed to initialize " + cls.__name__)
        
        atom = cls(offset.shape, coef=coef, quadratic=quadratic, 
                   offset=offset, **kws)
        return atom

    def scale(self, obj, copy=False):
        """Multiply `obj` by the atom's `coef`.

        Parameters
        ----------
        obj : float or ndarray
            Quantity to be scaled.
        copy : bool, optional
            If True and `coef` is 1, return a copy of `obj` rather than `obj`
            itself.

        Returns
        -------
        scaled : float or ndarray
            ``obj * self.coef``.
        """
        if self.coef != 1:
            return obj * self.coef
        if copy:
            return obj.copy()
        return obj

    def get_conjugate(self):
        """Return the convex conjugate of the atom.

        To be implemented by subclasses.
        """
        raise NotImplementedError('each smooth loss should implement its own get_conjugate')

    @property
    def conjugate(self):
        """The convex conjugate of the atom."""
        return self.get_conjugate()


def combine_case_weights(case_weights, loss_case_weights, shape):
    """
    Combine case weights passed to a loss with the loss's own case weights.

    Parameters
    ----------

    case_weights : ndarray or None
        Case weights passed in by the caller (e.g. a `glm`).
        Treated as all ones if None. Never modified.

    loss_case_weights : ndarray or None
        Case weights stored on the loss itself. Ignored if None.

    shape : tuple
        Shape of the weights when `case_weights` is None.

    Returns
    -------

    cw : ndarray
        A new float array, the product of the two sets of weights.
    """
    if case_weights is None:
        cw = np.ones(shape)
    else:
        cw = np.array(case_weights, float) # always a copy
    if loss_case_weights is not None:
        cw *= loss_case_weights
    return cw

def acceptable_init_args(cls, proposed_keywords):
    """Check that keywords are valid extra arguments to ``cls.__init__``.

    Parameters
    ----------
    cls : type
        Class whose constructor will receive the keywords.
    proposed_keywords : dict
        Proposed keyword arguments.

    Returns
    -------
    acceptable : bool
        True if every key of `proposed_keywords` is an argument of
        ``cls.__init__`` and is not one of ``self``, ``shape``, ``coef``,
        ``quadratic``, ``initial`` or ``offset``.
    """
    args = inspect.getfullargspec(cls.__init__).args
    forbidden = ['self', 'shape', 'coef', 'quadratic', 'initial', 'offset']
    for kw in proposed_keywords.keys():
        if not kw in args:
            return False
        if kw in forbidden:
            return False
    return True

class affine_smooth(smooth_atom):

    r"""Composition of a smooth objective with an affine transform.

    Represents :math:`\beta \mapsto f(X\beta + b)` where :math:`f` is a
    smooth atom and :math:`\beta \mapsto X\beta + b` an affine transform.
    """

    force_reshape = True

    objective_vars = {'linear':'X'}

    def __init__(self, smooth_atom, atransform, store_grad=True, diag=False):
        """Initialize the composition.

        Parameters
        ----------
        smooth_atom : `regreg.smooth.smooth_atom`
            A smooth atom.
        atransform : `regreg.affine.affine_transform` or ndarray
            An affine transformation, or an object cast to one using
            `regreg.affine.linear_transform`.
        store_grad : bool, optional
            If True, when computing the gradient, store a reference to the
            gradient of `smooth_atom` in the attribute `grad`.
        diag : bool, optional
            Indicates whether `atransform` acts diagonally, i.e. is a
            rescaling. Passed to `regreg.affine.linear_transform`.
        """
        self.store_grad = store_grad
        self.atom = smooth_atom
        if not isinstance(atransform, affine_transform):
            atransform = linear_transform(atransform, diag=diag)
        self.affine_transform = atransform
        self.shape = atransform.input_shape
        self.coefs = np.zeros(self.shape)

    def _get_coef(self):
        return self.atom.coef

    def _set_coef(self, coef):
        self.atom.coef = coef
    coef = property(_get_coef, _set_coef)

    def smooth_objective(self, arg, mode='both', check_feasibility=False):
        """Evaluate the composed objective, its gradient, or both.

        The atom is evaluated at ``self.affine_transform.affine_map(arg)``
        and its gradient is pulled back by the adjoint of the transform.

        Parameters
        ----------
        arg : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `arg` is not in the domain.

        Returns
        -------
        f : float
            Objective value at `arg`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `arg`; returned when `mode` is 'grad'
            or 'both'.
        """

        eta = self.affine_transform.affine_map(arg)
        if mode == 'both':
            v, g = self.atom.smooth_objective(eta, mode='both')
            if self.store_grad:
                self.grad = g
            g = self.affine_transform.adjoint_map(g)
            if self.force_reshape:
                g = g.reshape(self.shape)
            return v, g
        elif mode == 'grad':
            g = self.atom.smooth_objective(eta, mode='grad')
            if self.store_grad:
                self.grad = g
            g = self.affine_transform.adjoint_map(g)
            if self.force_reshape:
                g = g.reshape(self.shape)
            return g 
        elif mode == 'func':
            v = self.atom.smooth_objective(eta, mode='func')
            return v 

    @property
    def dual(self):
        """The ``(affine_transform, conjugate)`` pair for the dual problem.

        None if the conjugate of the underlying atom is not available.
        """
        try: 
            conj = self.atom.conjugate
            return self.affine_transform, conj
        except:
            return None

    def __repr__(self):
        return ("affine_smooth(%s, %s, store_grad=%s)" % 
                (str(self.atom),
                str(self.affine_transform),
                self.store_grad))

    def latexify(self, var=None, idx=''):
        """Return a LaTeX representation of the composed objective.

        Parameters
        ----------
        var : str, optional
            LaTeX name of the argument.
        idx : str, optional
            Index appended as a subscript.

        Returns
        -------
        latex : str
        """
        template_dict = self.atom.objective_vars.copy()
        template_dict['linear'] = self.objective_vars['linear']
        if var is not None:
            template_dict['var'] = var
        template_dict['idx'] = idx

        obj_latex = self.atom.latexify(var='%(linear)s_{%(idx)s}%(var)s' % template_dict, idx=idx)
        if not self.quadratic.iszero:
            return ' + '.join([obj_latex, self.quadratic.latexify(var=template_dict['var'], idx=idx)])
        else:
            return obj_latex

class zero(smooth_atom):

    """The zero function."""

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        """Evaluate the zero function, its (zero) gradient, or both.

        Parameters
        ----------
        x : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `x` is not in the domain.

        Returns
        -------
        f : float
            Objective value at `x`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `x`; returned when `mode` is 'grad'
            or 'both'.
        """
        if mode == 'both':
            return 0., np.zeros(x.shape)
        elif mode == 'func':
            return 0.
        elif mode == 'grad':
            return np.zeros(x.shape)
        raise ValueError("Mode not specified correctly")

class sum(smooth_atom):
    """A weighted sum of smooth atoms."""
    def __init__(self, atoms, weights=None):
        """Initialize the weighted sum.

        Parameters
        ----------
        atoms : sequence
            A sequence of `regreg.smooth.smooth_atom` instances that will be
            summed to make a new atom.
        weights : ndarray, optional
            Non-negative weights appearing as coefficients in front of each
            atom. Defaults to all ones.
        """
        self.offset = None
        self.atoms = atoms
        if weights is None:
            weights = np.ones(len(self.atoms))
        self.weights = np.asarray(weights).reshape(-1)
        if self.weights.shape[0] != len(atoms):
            raise ValueError('weights and atoms have different lengths')
        if np.any(self.weights < 0):
            raise ValueError('weights should be non-negative to maintain convexity')
        self.coefs = self.atoms[0].coefs
        self.shape = self.coefs.shape

    def smooth_objective(self, x, mode='both', check_feasibility=False):

        """Evaluate the weighted sum of the atoms, its gradient, or both.

        Parameters
        ----------
        x : ndarray
            The current parameter values.
        mode : {'both', 'func', 'grad'}, optional
            Whether to return the objective value ('func'), its gradient
            ('grad') or both ('both').
        check_feasibility : bool, optional
            If True, return ``np.inf`` when the point is not feasible,
            i.e. when `x` is not in the domain.

        Returns
        -------
        f : float
            Objective value at `x`; returned when `mode` is
            'func' or 'both'.
        g : ndarray
            Gradient at `x`; returned when `mode` is 'grad'
            or 'both'.
        """

        x = self.apply_offset(x)
        f, g = 0, 0
        for w, atom in zip(self.weights, self.atoms):
            if mode == 'func':
                f += w * atom.smooth_objective(x, 'func')
            elif mode == 'grad':
                g += w * atom.smooth_objective(x, 'grad')
            elif mode == 'both':
                fa, ga = atom.smooth_objective(x, 'both')
                f += w * fa; g += w * ga

        if mode == 'func':
            return f
        elif mode == 'grad':
            return g
        elif mode == 'both':
            return f, g
        else:
            raise ValueError("mode incorrectly specified")

