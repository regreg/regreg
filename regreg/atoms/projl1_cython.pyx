r"""
Projections onto the :math:`\ell_1` ball and the :math:`\ell_1` epigraph.

Implements (expected) linear time projections onto the :math:`\ell_1`
ball as described in Duchi, Shalev-Shwartz, Singer and Chandra,
"Efficient projections onto the l1-ball for learning in high dimensions".
"""

import numpy as np, sys
cimport numpy as np

from .piecewise_linear import (find_solution_piecewise_linear,
                               find_solution_piecewise_linear_c)

DTYPE_float = float
ctypedef np.float_t DTYPE_float_t

def projl1(np.ndarray[DTYPE_float_t, ndim=1]  x, 
           DTYPE_float_t bound=1.):
    r"""
    Project `x` onto the :math:`\ell_1` ball of radius `bound`.

    Parameters
    ----------
    x : np.ndarray
        One-dimensional float array to project.
    bound : float, optional
        Radius of the :math:`\ell_1` ball.

    Returns
    -------
    projection : np.ndarray
        The projection of `x` onto :math:`\{v : \|v\|_1 \leq
        \text{bound}\}`.
    """

    cdef np.ndarray[DTYPE_float_t, ndim=1] sorted_x = np.sort(np.fabs(x))
    cdef int p = x.shape[0]
    cdef double cut = find_solution_piecewise_linear_c(bound, 0, np.fabs(x))

    if cut < np.inf:
        return soft_threshold(x,cut)
    else:
        return x

cdef soft_threshold(np.ndarray[DTYPE_float_t, ndim=1] x,
                    DTYPE_float_t lagrange):

    cdef int p = x.shape[0]
    cdef np.ndarray[DTYPE_float_t, ndim=1] y = np.empty(p)
    cdef DTYPE_float_t xi
    cdef int i
    for i in range(p):
        xi = x[i]
        if xi > 0:
            if xi < lagrange:
                y[i] = 0.
            else:
                y[i] = xi - lagrange
        else:
            if xi > -lagrange:
                y[i] = 0.
            else:
                y[i] = xi + lagrange
    return y


def projl1_epigraph(np.ndarray[DTYPE_float_t, ndim=1] center):
    r"""
    Project `center` onto the :math:`\ell_1` epigraph.

    The norm term is ``center[-1]`` and the coefficient term is
    ``center[:-1]``. The :math:`\ell_1` epigraph is the collection of
    points :math:`(v, u)` with :math:`\|v\|_1 \leq u`, i.e.
    ``np.fabs(center[:-1]).sum() <= center[-1]``.

    Parameters
    ----------
    center : np.ndarray
        One-dimensional float array to project.

    Returns
    -------
    projection : np.ndarray
        The projection of `center` onto the epigraph.
    """

    cdef np.ndarray[DTYPE_float_t, ndim=1] x = center[:-1]
    cdef DTYPE_float_t norm = center[-1]
    cdef double cut = find_solution_piecewise_linear_c(norm, 1, np.fabs(x))

    cdef np.ndarray[DTYPE_float_t, ndim=1] result = np.zeros_like(center)
    if cut < np.inf:
        if (norm + cut >= 0):
            result[-1] = norm + cut
            result[:-1] = soft_threshold(x, cut)
    else:
        result = center
    return result


