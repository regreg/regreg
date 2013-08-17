from copy import copy
import warnings

import numpy as np

from ..identity_quadratic import identity_quadratic
from ..problems.composite import nonsmooth
from ..affine import (linear_transform, identity as identity_transform, 
                    affine_transform)
from ..smooth import affine_smooth
from ..objdoctemplates import objective_doc_templater
from ..doctemplates import (doc_template_user, doc_template_provider)


@objective_doc_templater()
class atom(nonsmooth):

    """
    A class that defines the API for support functions.
    """

    objective_vars = nonsmooth.objective_vars.copy()
    objective_vars['klass'] = 'norm'
    objective_vars['dualklass'] = 'dualnorm'

    tol = 1.0e-05

    @doc_template_provider
    def get_conjugate(self):
        """
        Return the conjugate of an given atom.
        Abstract method: subclasses must implement.
        """
        return None
    conjugate = property(get_conjugate, None, None, 'The conjugate of an atom.')

    @doc_template_provider
    def get_dual(self):
        """
        Get the dual of the atom. 
        Abstract method.
        """
        return self.linear_transform, self.conjugate
    dual = property(get_dual)

    @property
    def linear_transform(self):
        """
        The linear transform applied before a penalty is computed. Defaults to regreg.affine.identity

        >>> from regreg.api import l1norm
        >>> import numpy as np
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
        """
        raise NotImplementedError

    def smoothed(self, smoothing_quadratic):
        '''
        Add quadratic smoothing term
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
        Abstract method -- subclasses must implement.
        """
        raise NotImplementedError

class affine_atom(object):
    r"""
    Given a seminorm on :math:`\mathbb{R}^p`, i.e.  :math:`\beta \mapsto
    h_K(\beta)` this class creates a new seminorm that evaluates
    :math:`h_K(D\beta+\alpha)`

    This class does not have a prox, but its dual does. The prox of the dual is

    .. math::

       \text{minimize} \frac{1}{2} \|y-x\|^2_2 + x^T\alpha
       \ \text{s.t.} \ x \in \lambda K

    """

    objective_vars = {'linear':'X'}

    def __init__(self, atom_obj, atransform):
        self.atom = copy(atom_obj)
        # if the affine transform has an offset, put it into
        # the atom. in this way, the affine_transform is actually
        # always linear
        if atransform.affine_offset is not None:
            if self.atom.offset is not None:
                self.atom.offset += atransform.affine_offset
            else:
                self.atom.offset = atransform.affine_offset
            ltransform = linear_transform(atransform.linear_operator,
                                          diag=atransform.diagD)
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

        >>> from regreg.api import l1norm
        >>> penalty = l1norm(10, lagrange=0.9)
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
        template_dict = self.atom.objective_vars.copy()
        template_dict['linear'] = self.objective_vars['linear']
        if var is not None:
            template_dict['var'] = var
        template_dict['idx'] = idx
        return self.atom.latexify(var='%(linear)s_{%(idx)s}%(var)s' % template_dict, idx=idx)

    def __repr__(self):
        return "affine_atom(%s, %s)" % (repr(self.atom),
                                        repr(self.linear_transform.linear_operator))

    @property
    def dual(self):
        tmpatom = copy(self.atom)
        tmpatom.shape = self.output_shape
        return self.linear_transform, tmpatom.conjugate

    def nonsmooth_objective(self, arg, check_feasibility=False):
        """
        Return self.atom.seminorm(self.linear_transform.linear_map(x))
        """
        return self.atom.nonsmooth_objective( \
            self.linear_transform.linear_map(arg),
            check_feasibility=check_feasibility)

    def smoothed(self, smoothing_quadratic):
        '''
        Add quadratic smoothing term
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
    Compute the linear term in the conjugate as well as the offset
    based on having a given offset and the constant and linear
    terms in quadratic.

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

