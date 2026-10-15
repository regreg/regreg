r"""
Solvers for piecewise linear equations arising in projections onto
:math:`\ell_1` balls, :math:`\ell_1` epigraphs and simplices.
"""

import numpy as np, sys
cimport numpy as cnp


DTYPE_float = float
ctypedef cnp.float_t DTYPE_float_t
DTYPE_int = int
ctypedef cnp.intp_t DTYPE_int_t
ctypedef cnp.intp_t DTYPE_intp_t

def find_solution_piecewise_linear(DTYPE_float_t b,
                                   DTYPE_float_t slope,
                                   cnp.ndarray[DTYPE_float_t, ndim=1] norms,
                                   cnp.ndarray[DTYPE_float_t, ndim=1] weights):
    """
    Solve a weighted piecewise linear equation.

    Given a (decreasing) piecewise linear function of the form::

       f(t) = (weights * (weights * t < norms) * (norms - weights * t)).sum()

    with ``weights >= 0``, return the ``t >= 0`` such that
    ``f(t) = slope * t + b``. That is, it returns::

       inf {t >= 0 : f(t) <= slope * t + b}

    If ``f(0) < b`` this is 0.

    This function is used in projecting onto weighted
    l1 balls of size s and epigraphs. The ball projection uses
    ``slope=0, b=s``, and if this returns 0, one is already within the
    ball. The epigraph projection uses ``slope=1`` and ``b`` the norm
    part of the epigraph.

    Parameters
    ----------
    b : float
        Intercept of the linear function.
    slope : float
        Slope of the linear function.
    norms : np.ndarray
        Nonnegative knots (norms) of the piecewise linear function.
    weights : np.ndarray
        Nonnegative weights, same shape as `norms`.

    Returns
    -------
    t : float
        The solution.
    """

    cdef int stop = 0

    if (weights * norms).sum() < b:
        return 0

    # terms with weight 0 contribute nothing, but their knots (norm / 0)
    # would start the sweep at a huge value
    keep = weights > 0
    norms = np.ascontiguousarray(norms[keep])
    weights = np.ascontiguousarray(weights[keep])
    cdef int q = norms.shape[0]
    if q == 0:
        return 0

    cdef cnp.ndarray[DTYPE_float_t, ndim=1] knots = norms / weights
    cdef cnp.ndarray[DTYPE_intp_t, ndim=1] order = np.argsort(knots)

    slope = - slope # move the linear piece to the LHS
    cdef double curX = knots[order[q-1]]
    cdef double curV = slope * curX
    
    cdef double nextX, nextV
    cdef double solution = np.inf
    
    # if f(0) < b, then the set is empty
    # \inf of empty set is +\infty
    
    for j in range(q-1):
        slope -= weights[order[q-j-1]]**2
        nextX = knots[order[q-j-2]]
        nextV = curV + slope * (nextX - curX)
        stop = nextV > b
        if stop:
            intercept = curV - curX * slope
            break
        curV, curX = nextV, nextX
        curX = nextX
         
    if stop:
        solution = (b - intercept) / slope
    else:
        slope -= weights[order[0]]**2
        nextX = 0
        nextV = curV + slope * (nextX - curX)
        intercept = curV - curX * slope
        solution = (b - intercept) / slope
    return solution

def find_solution_piecewise_linear_c(DTYPE_float_t b,
                                     DTYPE_float_t slope,
                                     cnp.ndarray[DTYPE_float_t, ndim=1] norms):
    """
    Solve an unweighted piecewise linear equation.

    Given a (decreasing) piecewise linear function of the form::

       f(t) = ((t < norms) * (norms - t)).sum()

    return the ``t >= 0`` such that ``f(t) = slope * t + b``. That is,
    it returns::

       inf {t >= 0 : f(t) <= slope * t + b}

    If ``f(0) < b`` this is 0.

    This function is used in projecting onto
    l1 balls of size s and epigraphs. The ball projection uses
    ``slope=0, b=s``, and if this returns 0, one is already within the
    ball. The epigraph projection uses ``slope=1`` and ``b`` the norm
    part of the epigraph.

    Parameters
    ----------
    b : float
        Intercept of the linear function.
    slope : float
        Slope of the linear function.
    norms : np.ndarray
        Nonnegative knots (norms) of the piecewise linear function.

    Returns
    -------
    t : float
        The solution.
    """

    cdef int q = norms.shape[0]
    cdef int stop = 0

    if (norms).sum() < b:
        return 0

    cdef cnp.ndarray[DTYPE_float_t, ndim=1] knots = norms
    cdef cnp.ndarray[DTYPE_intp_t, ndim=1] order = np.argsort(knots)

    slope = - slope # move the linear piece to the LHS
    cdef double curX = knots[order[q-1]]
    cdef double curV = slope * curX
    
    cdef double nextX, nextV
    cdef double solution = np.inf
    
    # if f(0) < b, then the set is empty
    # \inf of empty set is +\infty
    
    for j in range(q-1):
        slope -= 1.
        nextX = knots[order[q-j-2]]
        nextV = curV + slope * (nextX - curX)
        stop = nextV > b
        if stop:
            intercept = curV - curX * slope
            break
        curV, curX = nextV, nextX
        curX = nextX
         
    if stop:
        solution = (b - intercept) / slope
    else:
        slope -= 1.
        nextX = 0
        nextV = curV + slope * (nextX - curX)
        intercept = curV - curX * slope
        solution = (b - intercept) / slope
    return solution

def find_solution_piecewise_linear_uncon(DTYPE_float_t b,
                                         cnp.ndarray[DTYPE_float_t, ndim=1] values):
    """
    Solve an unconstrained piecewise linear equation.

    Given a piecewise linear function of the form::

       f(t) = ((t < values) * (values - t)).sum()

    return the ``t`` such that ``f(t) = b`` (``t`` is not restricted to
    be nonnegative).

    This function is used in projecting onto the simplex.

    Parameters
    ----------
    b : float
        Target value, assumed positive.
    values : np.ndarray
        Knots of the piecewise linear function.

    Returns
    -------
    t : float
        The solution.
    """

    cdef int q = values.shape[0]
    cdef int stop = 0

    cdef cnp.ndarray[DTYPE_float_t, ndim=1] knots = values
    cdef cnp.ndarray[DTYPE_intp_t, ndim=1] order = np.argsort(knots)

    slope = 0
    cdef double curX = knots[order[q-1]]
    cdef double curV = slope * curX
    
    cdef double nextX, nextV
    cdef double solution = np.inf
    
    # if f(0) < b, then the set is empty
    # \inf of empty set is +\infty
    
    for j in range(q-1):
        slope -= 1.
        nextX = knots[order[q-j-2]]
        nextV = curV + slope * (nextX - curX)
        stop = nextV > b
        if stop:
            intercept = curV - curX * slope
            break
        curV, curX = nextV, nextX
        curX = nextX
         
    if stop:
        solution = (b - intercept) / slope
    else:
        slope -= 1.
        nextX = 0
        nextV = curV + slope * (nextX - curX)
        intercept = curV - curX * slope
        solution = (b - intercept) / slope
    return solution

