from copy import copy
import warnings

import numpy as np

from ..identity_quadratic import identity_quadratic
from ..problems.composite import nonsmooth
from ..affine import (linear_transform, identity as identity_transform, 
                    affine_transform, aslinear)
from ..smooth import affine_smooth
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)


@objective_doc_templater()
class atom(nonsmooth):

    """
    Base class for atoms: simple nonsmooth functions with a known prox.

    An atom is a support function (or, more generally, a simple convex
    function) with an inexpensive proximal map. Subclasses implement
    `nonsmooth_objective`, `proximal` and `get_conjugate`.
    """

    objective_vars = nonsmooth.objective_vars.copy()
    objective_vars['klass'] = 'norm'
    objective_vars['dualklass'] = 'dualnorm'

    tol = 1.0e-05

    @doc_template_provider
    def get_conjugate(self):
        """
        Return the conjugate of a given atom.

        Abstract method: subclasses must implement.
        """
        return None
    # look up get_conjugate / get_dual on the instance, so that subclass
    # overrides are used
    conjugate = property(lambda self: self.get_conjugate(), None, None,
                         'The conjugate of an atom.')

    @doc_template_provider
    def get_dual(self):
        """
        Return the dual of the atom.

        Returns
        -------
        linear_transform : `regreg.affine.linear_transform`
            The linear transform of the atom.
        conjugate : `atom`
            The conjugate of the atom.
        """
        return self.linear_transform, self.conjugate
    dual = property(lambda self: self.get_dual(), None, None,
                    'The dual of an atom.')

    @property
    def linear_transform(self):
        """
        The linear transform applied before a penalty is computed.

        Defaults to `regreg.affine.identity`.

        Examples
        --------
        >>> from regreg.api import l1norm
        >>> penalty = l1norm(30, lagrange=3.4)
        >>> type(penalty.linear_transform)
        <class 'regreg.affine.identity'>

        """
        if not hasattr(self, "_linear_transform"):
            self._linear_transform = identity_transform(self.shape)
        return self._linear_transform

    @doc_template_provider
    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        The nonsmooth objective function of the atom.

        Includes the quadratic term of the atom.

        Abstract method: subclasses must implement.

        Parameters
        ----------
        arg : `np.ndarray`
            Argument of the objective.
        check_feasibility : bool, optional
            If `True`, return `np.inf` when `arg` is infeasible.
        """
        raise NotImplementedError

    def smoothed(self, smoothing_quadratic):
        '''
        Return a smoothed version of the atom.

        The smoothing is achieved by adding `smoothing_quadratic`
        to the conjugate of the atom and taking the conjugate again
        (Nesterov smoothing).

        Parameters
        ----------
        smoothing_quadratic : `regreg.identity_quadratic.identity_quadratic`
            Quadratic term added to the conjugate. Its coefficient
            must be nonzero.

        Returns
        -------
        smoothed_atom : `regreg.problems.composite.smooth_conjugate`
            A smooth approximation of the atom.
        '''
        conjugate_atom = copy(self.conjugate)
        sq = smoothing_quadratic
        if sq.coef in [None, 0]:
            raise ValueError('quadratic term of ' 
                             + 'smoothing_quadratic must be non 0')
        total_q = sq

        if conjugate_atom.quadratic is not None:
            total_q = sq + conjugate_atom.quadratic
        conjugate_atom.quadratic = total_q
        smoothed_atom = conjugate_atom.conjugate
        return smoothed_atom

    @doc_template_provider
    def proximal(self, proxq, prox_control=None):
        r"""
        The proximal operator.

        Abstract method: subclasses must implement.

        Parameters
        ----------
        proxq : `regreg.identity_quadratic.identity_quadratic`
            A quadratic added to the atom before minimizing.
        prox_control : dict, optional
            Optional parameters for the proximal procedure.
        """
        raise NotImplementedError

class affine_atom(object):
    r"""
    An atom composed with an affine transform.

    Given a seminorm on :math:`\mathbb{R}^p`, i.e. :math:`\beta \mapsto
    h_K(\beta)`, this class creates a new seminorm that evaluates
    :math:`h_K(D\beta+\alpha)`.

    This class does not have a prox, but its dual does. The prox of the dual is

    .. math::

       \text{minimize} \frac{1}{2} \|y-x\|^2_2 + x^T\alpha
       \ \text{s.t.} \ x \in \lambda K

    Parameters
    ----------
    atom_obj : `atom`
        The atom to compose with the affine transform. A copy is stored.
    atransform : `regreg.affine.affine_transform`
        The affine transform. Any affine offset is moved into the
        offset of the atom, so that the stored transform is linear.
    """

    objective_vars = {'linear':'X'}

    def __init__(self, atom_obj, atransform):
        self.atom = copy(atom_obj)
        # if the affine transform has an offset, put it into
        # the atom. in this way, the affine_transform is actually
        # always linear
        #
        # the atom's offset is its center, i.e. it is evaluated at
        # ``x - offset``, so the transform's offset is subtracted.
        #
        # The offset is computed from the map itself, as affine_map(0): the
        # `affine_offset` attribute leaves out the offsets of nested
        # transforms (an affine_transform built from an affine transform, or
        # a linear_transform wrapping one)
        offset = np.asarray(atransform.affine_map(np.zeros(atransform.input_shape)), float)
        if np.any(offset != 0):
            if self.atom.offset is not None:
                self.atom.offset = self.atom.offset - offset
            else:
                self.atom.offset = -offset
            if getattr(atransform, 'linear_operator', None) is not None:
                ltransform = linear_transform(atransform.linear_operator,
                                              diag=atransform.diagD)
            else: # e.g. a composition
                ltransform = aslinear(atransform)
        else:
            ltransform = atransform
        self.linear_transform = ltransform
        self.input_shape = self.linear_transform.input_shape
        self.output_shape = self.linear_transform.output_shape

    def _repr_latex_(self):
        return r'$$' + self.latexify() + r'$$'

    def latexify(self, var=None, idx=''):
        r'''
        Return a LaTeX representation of an object.

        Parameters
        ----------
        var : str, optional
            Argument of the function.
        idx : str, optional
            Optional subscript index.

        Returns
        -------
        L : str
            A LaTeX representation of the atom.

        Examples
        --------
        >>> from regreg.api import l1norm
        >>> penalty = l1norm(10, lagrange=0.9)
        >>> penalty.latexify(var=r'\gamma') 
        '\\lambda_{} \\|\\gamma\\|_1'
        '''
        template_dict = self.atom.objective_vars.copy()
        template_dict['linear'] = self.objective_vars['linear']
        if var is not None:
            template_dict['var'] = var
        template_dict['idx'] = idx
        return self.atom.latexify(var='%(linear)s_{%(idx)s}%(var)s' % template_dict, idx=idx)

    def __repr__(self):
        return "affine_atom(%s, %s)" % (repr(self.atom),
                                        repr(getattr(self.linear_transform,
                                                     'linear_operator',
                                                     self.linear_transform)))

    @property
    def dual(self):
        """
        The dual of the affine atom: a pair of the linear transform and the
        conjugate of the underlying atom.
        """
        tmpatom = copy(self.atom)
        tmpatom.shape = self.output_shape
        return self.linear_transform, tmpatom.conjugate

    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Evaluate the atom at the linear transform of `arg`.

        Returns ``self.atom.nonsmooth_objective(self.linear_transform.linear_map(arg))``.

        Parameters
        ----------
        arg : `np.ndarray`
            Argument of the objective.
        check_feasibility : bool, optional
            Passed to the underlying atom.
        """
        return self.atom.nonsmooth_objective( \
            self.linear_transform.linear_map(arg),
            check_feasibility=check_feasibility)

    def smoothed(self, smoothing_quadratic):
        '''
        Return a smoothed version of the affine atom.

        Parameters
        ----------
        smoothing_quadratic : `regreg.identity_quadratic.identity_quadratic`
            Quadratic term added to the conjugate of the atom. Its
            coefficient must be nonzero.

        Returns
        -------
        smoothed : `regreg.smooth.affine_smooth`
            The smoothed atom composed with the linear transform.
        '''
        ltransform, conjugate_atom = self.dual
        if conjugate_atom.quadratic is not None:
            total_q = smoothing_quadratic + conjugate_atom.quadratic
        else:
            total_q = smoothing_quadratic
        if total_q.coef in [None, 0]:
            raise ValueError('quadratic term of '
                             + 'smoothing_quadratic must be non 0')
        conjugate_atom.quadratic = total_q
        smoothed_atom = conjugate_atom.conjugate
        value = affine_smooth(smoothed_atom, ltransform)
        value.total_quadratic = (smoothed_atom.smoothing_quadratic +
                                 smoothed_atom.atom.quadratic)
        return value

def _work_out_conjugate(offset, quadratic):
    """
    Compute the offset and quadratic term of a conjugate.

    Based on a given offset and the constant and linear terms in
    `quadratic`, compute the offset and the linear term of the
    conjugate.
    """
    if offset is None:
        offset = 0
    else:
        offset = offset
    if quadratic.linear_term is not None:
        linear_term = quadratic.linear_term
    else:
        linear_term = 0
    outq = identity_quadratic(0,0,offset, \
          -quadratic.constant_term - 
          np.sum(offset * linear_term))

    if quadratic.linear_term is not None:
        outoffset = linear_term
    else:
        outoffset = None
    return outoffset, outq

