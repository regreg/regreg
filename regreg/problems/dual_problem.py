"""
Solving composite problems through their duals.
"""

import numpy as np
from scipy import sparse
from warnings import warn

from ..algorithms import FISTA
from ..problems.composite import (composite, nonsmooth as nonsmooth_composite,
                        smooth as smooth_composite)
from ..affine import (vstack as afvstack, identity as afidentity, power_L,
                     selector as afselector,
                     scalar_multiply, adjoint)
from ..problems.separable import separable
from ..smooth import smooth_atom, affine_smooth
from ..atoms import affine_atom as nonsmooth_affine_atom
from ..atoms.cones import zero_constraint, zero as zero_nonsmooth, affine_cone
from ..identity_quadratic import identity_quadratic

class dual_problem(composite):
    r"""
    A class for specifying a problem of the form

    .. math::

       \text{minimize}_{x} f(x) + g(Dx)

    which will be solved by a dual problem

    .. math::

       \text{minimize}_{u} f^*(-D^Tu) + g^*(u)

    while the primal variable is stored in the computation of the gradient of
    :math:`f^*`.

    Parameters
    ----------
    f_conjugate : `regreg.problems.composite.smooth`
        The conjugate :math:`f^*`.
    transform : `regreg.affine.affine_transform`
        The linear transform :math:`D`.
    atom : `regreg.atoms.atom`
        The conjugate :math:`g^*`.
    """

    def __init__(self, f_conjugate, transform, atom):
        self.offset = None
        self.f_conjugate = f_conjugate
        if not isinstance(self.f_conjugate, smooth_composite):
            warn('the conjugate of f should be a smooth_composite to solve with generalized gradient')

        self.transform = transform
        self.atom = atom

        # the dual problem has f^*(-D^Tu) as objective
        self.affine_fc = affine_smooth(self.f_conjugate, scalar_multiply(adjoint(self.transform), -1))
        self.coefs = np.zeros(self.affine_fc.shape)

    # the quadratic is delegated to the individual parts
    @property
    def quadratic(self):
        """
        Zero quadratic; the quadratic is delegated to the individual parts.
        """
        return identity_quadratic(0,0,0,0)

    @staticmethod
    def fromprimal(f, *g):
        r"""
        Construct the dual problem from a primal problem.

        Parameters
        ----------
        f : `regreg.problems.composite.composite`
            The function :math:`f`; must have a ``conjugate`` attribute.
        g : sequence of atoms
            Atoms whose sum is the function :math:`g(D\cdot)`.

        Returns
        -------
        problem : `dual_problem`
            The dual problem.
        """
        transform, separable_dual_atom = stacked_dual(f.shape, *g)
        return dual_problem(f.conjugate, transform, separable_dual_atom)

    def smooth_objective(self, x, mode='both', check_feasibility=False):
        r"""
        The smooth part :math:`f^*(-D^Tu)` of the dual objective.

        The smooth_objective DOES NOT INCLUDE the identity
        quadratic of all the smooth atoms. If the gradient is computed,
        the current primal variable is stored as ``self.primal``.
        """
        v = self.affine_fc.smooth_objective(x, mode=mode, check_feasibility=check_feasibility)
        # retain a reference
        if mode in ['both', 'grad']:
            self.primal = self.affine_fc.grad
        return v

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        The nonsmooth part :math:`g^*(u)` of the dual objective.
        """
        out = self.atom.nonsmooth_objective(x, 
                                            check_feasibility=check_feasibility)
        return out + self.affine_fc.nonsmooth_objective(x, 
                                                        check_feasibility=check_feasibility)

    def proximal(self, proxq, prox_control=None):
        """
        The proximal function for the dual problem.
        """
        return self.atom.proximal(proxq, prox_control=prox_control)

    def solve(self, quadratic=None, return_optimum=False, **fit_args):
        """
        Solve the dual problem using `FISTA`, returning the primal solution.

        Parameters
        ----------
        quadratic : `regreg.identity_quadratic.identity_quadratic`, optional
            Ignored.
        return_optimum : bool, optional
            Also return the objective at the primal solution?
        fit_args : dict
            Keyword arguments passed to `regreg.algorithms.FISTA.fit`.

        Returns
        -------
        optimal_value : float
            Only returned if `return_optimum` is True.
        primal : ndarray
            The primal solution.
        """
        solver = FISTA(self)
        solver.fit(**fit_args)
        if return_optimum:
            return self.objective(self.primal), self.primal
        return self.primal

    def latexify(self, var=None):
        """
        Return a LaTeX representation of the dual problem.
        """
        template_dict = self.objective_vars.copy()
        template_dict['prox'] =  self.atom.latexify(var=var,idx='2')
        template_dict['smooth'] = self.affine_fc.latexify(var=var,idx='1')
        template_dict
        result = r'''
        \begin{aligned}
        \text{minimize}_{%(var)s} & f(%(var)s) + g(%(var)s) \\
        f(%(var)s) &= %(smooth)s \\
        g(%(var)s) &= %(prox)s \\
        \end{aligned}
        ''' % template_dict
        result = '\n'.join([s.strip() for s in result.split('\n')])
        return result


def stacked_dual(shape, *primary_atoms):
    r'''
    Computes a dual of

    .. math::

       \sum_i g_i(D_i\beta)

    under the substitutions :math:`v_i=D_i\beta`.

    That is, it returns the following dual function after minimizing
    over :math:`(v_i,\beta_i)`:

    .. math::

       -\sum_i g_i^*(u_i)

    as well as the transform :math:`D : \mathbb{R}^p \to \prod_i
    \mathbb{R}^{m_i}` where :math:`p` is the primal shape and :math:`m_i` are
    the corresponding dual shapes.

    Parameters
    ----------
    shape : tuple
        The primal shape; used only if no atoms are given.
    primary_atoms : sequence of atoms
        Objects that have a ``dual`` attribute, which is a pair
        ``(ltransform, conjugate)``.

    Returns
    -------
    transform : `regreg.affine.affine_transform`
        The stacked transform :math:`D`.
    dual_atom : `regreg.atoms.atom` or `regreg.problems.separable.separable`
        The (separable) conjugate atom.
    '''
    if len(primary_atoms) == 0:
        primary_atoms = [zero_nonsmooth(shape)]

    duals = [atom.dual for atom in primary_atoms]
    transforms = [d[0] for d in duals]
    dual_atoms = [d[1] for d in duals]

    if len(transforms) > 1:
        transform = afvstack(transforms)
        separable_atom = separable(transform.output_shape, dual_atoms,
                                   transform.output_slices)
        _dual = transform, separable_atom
    else:
        _dual = (transforms[0], dual_atoms[0])
    return _dual

