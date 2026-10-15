"""
Separable penalties built from several atoms acting on disjoint slices.

This module implements the notion of a separable support function / constraint
regularizer or penalty.

The penalty is specified by a primal shape, a sequence of atoms and
a sequence of slicing objects.
"""

import numpy as np

from ..affine import selector
from ..atoms import atom, affine_atom
from ..problems.simple import simple_problem
from ..atoms.cones import zero
from .composite import smooth_conjugate

def _as_index(group):
    """
    Convert a group to something that can index an array.

    A list that contains a slice, ``Ellipsis`` or ``None`` is a
    multidimensional index and must be a tuple (NumPy no longer accepts
    such lists); other groups, e.g. lists of integers, are used as is.
    """
    if isinstance(group, list) and any(g is Ellipsis or g is None or
                                       isinstance(g, slice) for g in group):
        return tuple(group)
    return group

def has_overlap(shape, groups):
    """
    Determine whether the groups, viewed as slices of an array
    with given shape, have any overlap.

    Parameters
    ----------
    shape : tuple
        A tuple of integers representing a shape for an array.
    groups : sequence
        A sequence of objects that can be viewed as slices of
        an ndarray with shape==shape.

    Returns
    -------
    res : bool
        True if the slices overlap, else False.

    Examples
    --------
    >>> from regreg.problems.separable import has_overlap
    >>> has_overlap((4,5), [slice(2,3), slice(4,5)])
    False
    >>> has_overlap((4,5), [slice(2,3), [Ellipsis, slice(4,5)]])
    True
    """
    indices = np.arange(np.prod(shape)).reshape(shape)
    subsets = []
    for group in groups:
        subsets.append(set(list(indices[_as_index(group)].reshape(-1))))
    subsets = tuple(subsets)
    for i in range(len(subsets)):
        for j in range(i):
            subset1 = subsets[i]
            subset2 = subsets[j]
            if subset1.intersection(subset2):
                    return True
    return False

class separable(atom):

    r"""
    A separable penalty: a sum of atoms, each applied to a slice of the
    argument.

    .. math::

        x \mapsto \sum_i {\cal P}_i(x[g_i])

    Coordinates not covered by any group are unpenalized.

    Parameters
    ----------
    shape : tuple
        Shape of the argument.
    atoms : sequence of `regreg.atoms.atom`
        The atoms :math:`{\cal P}_i`.
    groups : sequence
        Slicing objects :math:`g_i`, one for each atom.
    test_for_overlap : bool, optional
        If True, raise ValueError if the groups overlap.
    shapes : sequence, optional
        Shapes to which each slice is reshaped before being passed to its
        atom; None means no reshaping.
    initial : ndarray, optional
        Initial value of ``coefs``. Defaults to zeros.
    """

    def __init__(self,
                 shape,
                 atoms,
                 groups,
                 test_for_overlap=False,
                 shapes=[],
                 initial=None):
        if test_for_overlap and has_overlap(shape, groups):
            raise ValueError('groups are not separable')
        self.shape = shape
        self.groups = [_as_index(group) for group in groups]
        self.atoms = atoms
        if shapes == []:
            shapes = [None] * len(atoms)
        self.shapes = shapes
        test = np.empty(self.shape)
        self.assign_shapes = [test[group].shape for group in self.groups]
        self.zero_atom = zero(shape)

        if initial is None:
            self.coefs = np.zeros(shape)
        else:
            self.coefs = np.asarray(initial, float).copy()

    def latexify(self, var=None, idx=''):
        """
        Return a LaTeX representation of the separable penalty.
        """
        # BUG idx is unused
        strs = []
        for i, atom in enumerate(self.atoms):
            template_dict = atom.objective_vars.copy()
            template_dict['linear'] = atom.objective_vars['linear']
            if var is not None:
                template_dict['var'] = "%s[g%d]" % (var, i)
            else:
                template_dict['var'] = "%s[g%d]" % (template_dict['var'], i)
            template_dict['idx'] = idx
            if isinstance(atom, affine_atom):
                strs.append(atom.latexify(var='%(linear)s_{%(idx)s}%(var)s' % template_dict, idx=i))
            else:
                strs.append(atom.latexify(var='%(var)s' % template_dict, idx=i))
        return ' + '.join(strs)

    def seminorm(self, x, lagrange=None, check_feasibility=False):
        """
        Sum of the seminorms of each atom on its slice of `x`.
        """
        value = 0.
        for atom, group, shape in zip(self.atoms,
                                      self.groups,
                                      self.shapes):
            x_group = x[group]
            if shape is not None:
                x_group.shape = shape
            value += atom.seminorm(x_group,
                                   lagrange=lagrange,
                                   check_feasibility=check_feasibility)
        return value

    def constraint(self, x, bound=None):
        """
        Sum of the constraints of each atom on its slice of `x`.
        """
        value = 0.
        for atom, group, shape in zip(self.atoms,
                                      self.groups,
                                      self.shapes):
            x_group = x[group]
            if shape is not None:
                x_group.shape = shape
            value += atom.constraint(x_group, bound=bound)
        return value

    def nonsmooth_objective(self, x, check_feasibility=False):
        """
        Sum of the nonsmooth objectives of each atom on its slice of `x`.
        """
        value = 0
        for atom, group, shape in zip(self.atoms,
                                      self.groups,
                                      self.shapes):
            x_group = x[group]
            if shape is not None:
                x_group.shape = shape
            value += atom.nonsmooth_objective(x_group,
                                              check_feasibility=check_feasibility)
        return value

    def proximal(self, proxq, prox_control=None):
        """
        Proximal map, computed separately for each atom on its slice.

        Coordinates not in any group are mapped by the proximal map of
        the zero penalty.
        """
        # This allows separable to not have every
        # coefficient penalized in a natural way
        # because this instantiates it at the prox map of the zero penalty
        v = self.zero_atom.proximal(proxq)
        # collapse the center into the linear term, so that only the linear
        # term needs reshaping for each block
        proxq = proxq.collapsed()
        for atom, group, shape, assign_shape in zip(self.atoms,
                                                    self.groups,
                                                    self.shapes,
                                                    self.assign_shapes):
            proxq_group = proxq[group]
            if shape is not None:
                linear_term = np.asarray(proxq_group.linear_term, float)
                if linear_term.ndim > 0:
                    # a reshaped copy: setting .shape would change proxq
                    proxq_group.linear_term = linear_term.reshape(shape)
            v[group] = atom.proximal(proxq_group, prox_control).reshape(assign_shape)
        return v

    @property
    def conjugate(self):
        """
        The conjugate: separable in the conjugates of the atoms if there is
        no quadratic, otherwise a `smooth_conjugate`.
        """
        if self.quadratic.coef == 0:
            conjugate_atom = separable(self.shape,
                                       [atom.conjugate 
                                        for atom in self.atoms],
                                       self.groups,
                                       shapes=self.shapes)
        else:
            conjugate_atom = smooth_conjugate(self)
        self._conjugate = conjugate_atom
        self._conjugate._conjugate = self
        return self._conjugate

    def __repr__(self):
        return "separable(%s, %s, %s, %s)" % (repr(self.shape),
                                              repr(self.atoms),
                                              repr(self.groups),
                                              repr(self.shapes))

    @property
    def selectors(self):
        """
        Selectors for each group.
        """
        return [selector(group, self.shape, reshape=shape)
                for group, shape in zip(self.groups, self.shapes)]

class separable_problem(simple_problem):

    """
    A `simple_problem` whose nonsmooth part is a `separable` penalty.

    Parameters
    ----------
    smooth_atom : `regreg.smooth.smooth_atom`
        The smooth part of the objective.
    shape : tuple
        Shape of the argument.
    atoms : sequence of `regreg.atoms.atom`
        Atoms making up the separable penalty.
    groups : sequence
        Slicing objects, one for each atom.
    test_for_overlap : bool, optional
        If True, raise ValueError if the groups overlap.
    shapes : sequence, optional
        Shapes to which each slice is reshaped; see `separable`.
    """
    
    def __init__(self,
                 smooth_atom,
                 shape,
                 atoms,
                 groups,
                 test_for_overlap=False,
                 shapes=[]):

        nonsmooth_atom = separable(shape,
                                   atoms,
                                   groups,
                                   test_for_overlap,
                                   shapes=shapes)
        simple_problem.__init__(self, smooth_atom, nonsmooth_atom)
        
    @staticmethod
    def singleton(atom, smooth_f):
        """
        A separable problem with a single atom applied to all coordinates.
        """
        return separable_problem(smooth_f,
                                 atom.shape,
                                 [atom],
                                 [slice(None)],
                                 False)

    @staticmethod
    def fromatom(separable_atom, smooth_f):
        """
        A separable problem from a `separable` atom and a smooth function.
        """
        return separable_problem(smooth_f, separable_atom.shape, 
                                 separable_atom.atoms,
                                 separable_atom.groups,
                                 False,
                                 shapes=separable_atom.shapes)
    @property
    def selectors(self):
        """
        Selectors for each group of the separable penalty.
        """
        return self.proximal_atom.selectors
