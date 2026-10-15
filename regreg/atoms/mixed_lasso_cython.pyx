"""
Cython implementations for the mixed LASSO penalty.

Implements the proximal maps of the mixed LASSO penalty and of its dual,
the seminorm and dual seminorm, the strong set rule, KKT checks and the
projection onto the epigraph of the penalty.
"""

import numpy as np, sys
cimport numpy as cnp

from .piecewise_linear import find_solution_piecewise_linear


DTYPE_float = float
ctypedef cnp.float_t DTYPE_float_t
DTYPE_int = int
ctypedef cnp.intp_t DTYPE_int_t
ctypedef cnp.intp_t DTYPE_intp_t

def mixed_lasso_lagrange_prox(cnp.ndarray[DTYPE_float_t, ndim=1] prox_center, 
                              DTYPE_float_t lagrange, 
                              DTYPE_float_t lipschitz,
                              cnp.ndarray[DTYPE_intp_t, ndim=1] l1_penalty, 
                              cnp.ndarray[DTYPE_intp_t, ndim=1] unpenalized,
                              cnp.ndarray[DTYPE_intp_t, ndim=1] positive_part, 
                              cnp.ndarray[DTYPE_intp_t, ndim=1] nonnegative, 
                              cnp.ndarray[DTYPE_intp_t, ndim=1] groups,
                              cnp.ndarray[DTYPE_float_t, ndim=1] weights,
                              cnp.ndarray[DTYPE_float_t, ndim=1] norms,
                              cnp.ndarray[DTYPE_float_t, ndim=1] factors,
                              cnp.ndarray[DTYPE_float_t, ndim=1] projection
                              ):
    r"""
    Proximal map of the mixed LASSO penalty in Lagrange form.

    Parameters
    ----------
    prox_center : np.ndarray
        Center of the proximal map.
    lagrange : float
        Lagrange parameter.
    lipschitz : float
        Coefficient of the quadratic term.
    l1_penalty, unpenalized, positive_part, nonnegative : np.ndarray
        Indices of the coordinates with an :math:`\ell_1` penalty, no
        penalty, a positive part penalty and a nonnegativity constraint.
    groups : np.ndarray
        Group index of each coordinate, or a negative value for
        coordinates that are not in a group.
    weights : np.ndarray
        Weight of each group.
    norms, factors, projection : np.ndarray
        Work buffers, assumed to be zero on entry; `norms` and `factors`
        have one entry per group and `projection` one per coordinate.

    Returns
    -------
    prox : np.ndarray
        Minimizer of :math:`\frac{L}{2}\|x - c\|^2_2 + \lambda h(x)`
        where :math:`c` is `prox_center`, :math:`L` is `lipschitz` and
        :math:`h` is the mixed LASSO penalty.
    """
    
    cdef int i, j
    cdef int p = groups.shape[0]
    
    cdef implied_bound = lagrange / lipschitz
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + prox_center[i]**2
    
    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        if norms[j] > 0:
            factors[j] = min(1., implied_bound * weights[j] / norms[j])
        else:
            factors[j] = 1.
    
    for i in range(p):
        if groups[i] >= 0:
            projection[i] = prox_center[i] * factors[groups[i]]

    projection[l1_penalty] = (prox_center[l1_penalty] * 
                              np.minimum(1, implied_bound / 
                                         np.fabs(prox_center[l1_penalty])))
    projection[unpenalized] = 0
    # the dual set of lambda * max(x, 0) is [0, lambda] (gh-103)
    projection[positive_part] = np.clip(prox_center[positive_part],
                                        0, implied_bound)
    projection[nonnegative] = np.minimum(prox_center[nonnegative], 0)

    return prox_center - projection

def mixed_lasso_dual_bound_prox(cnp.ndarray[DTYPE_float_t, ndim=1] prox_center, 
                                DTYPE_float_t bound, 
                                cnp.ndarray[DTYPE_intp_t, ndim=1] l1_penalty, 
                                cnp.ndarray[DTYPE_intp_t, ndim=1] unpenalized,
                                cnp.ndarray[DTYPE_intp_t, ndim=1] positive_part, 
                                cnp.ndarray[DTYPE_intp_t, ndim=1] nonnegative, 
                                cnp.ndarray[DTYPE_intp_t, ndim=1] groups,
                                cnp.ndarray[DTYPE_float_t, ndim=1] weights,
                                cnp.ndarray[DTYPE_float_t, ndim=1] norms,
                                cnp.ndarray[DTYPE_float_t, ndim=1] factors,
                                cnp.ndarray[DTYPE_float_t, ndim=1] projection):
    r"""
    Projection onto the set where the dual mixed LASSO seminorm is at most `bound`.

    Parameters
    ----------
    prox_center : np.ndarray
        Point to project.
    bound : float
        Bound on the dual seminorm.
    l1_penalty, unpenalized, positive_part, nonnegative : np.ndarray
        Indices of the coordinates with an :math:`\ell_1` penalty, no
        penalty, a positive part penalty and a nonnegativity constraint.
    groups : np.ndarray
        Group index of each coordinate, or a negative value for
        coordinates that are not in a group.
    weights : np.ndarray
        Weight of each group.
    norms, factors, projection : np.ndarray
        Work buffers, assumed to be zero on entry; `norms` and `factors`
        have one entry per group and `projection` one per coordinate.

    Returns
    -------
    projection : np.ndarray
        The projection of `prox_center`.
    """
    
    cdef int i, j
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + prox_center[i]**2
    
    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        if norms[j] > 0:
            factors[j] = min(1., bound * weights[j] / norms[j])
        else:
            factors[j] = 1.
    
    for i in range(p):
        if groups[i] >= 0:
            projection[i] = prox_center[i] * factors[groups[i]]

    projection[l1_penalty] = prox_center[l1_penalty] * np.minimum(1, bound / np.fabs(prox_center[l1_penalty]))
    projection[unpenalized] = 0
    projection[positive_part] = np.clip(prox_center[positive_part], 0, bound)
    projection[nonnegative] = np.minimum(prox_center[nonnegative], 0)

    # a copy: `projection` is a work buffer reused by the next call
    return projection.copy()

def seminorm_mixed_lasso(cnp.ndarray[DTYPE_float_t, ndim=1] x, 
                         cnp.ndarray[DTYPE_intp_t, ndim=1] l1_penalty, 
                         cnp.ndarray[DTYPE_intp_t, ndim=1] unpenalized,
                         cnp.ndarray[DTYPE_intp_t, ndim=1] positive_part, 
                         cnp.ndarray[DTYPE_intp_t, ndim=1] nonnegative, 
                         cnp.ndarray[DTYPE_intp_t, ndim=1] groups,
                         cnp.ndarray[DTYPE_float_t, ndim=1] weights,
                         cnp.ndarray[DTYPE_float_t, ndim=1] norms,
                         DTYPE_int_t check_feasibility,
			 DTYPE_float_t nntol=1.e-5):
    r"""
    Evaluate the mixed LASSO seminorm (with Lagrange parameter 1).

    Parameters
    ----------
    x : np.ndarray
        Argument of the seminorm.
    l1_penalty, unpenalized, positive_part, nonnegative : np.ndarray
        Indices of the coordinates with an :math:`\ell_1` penalty, no
        penalty, a positive part penalty and a nonnegativity constraint.
    groups : np.ndarray
        Group index of each coordinate, or a negative value for
        coordinates that are not in a group.
    weights : np.ndarray
        Weight of each group.
    norms : np.ndarray
        Work buffer with one entry per group, assumed to be zero on entry.
    check_feasibility : int
        If nonzero, return `np.inf` if a positive part or nonnegative
        coordinate is less than a small tolerance.
    nntol : float, optional
        Tolerance for the nonnegativity check.

    Returns
    -------
    value : float
        Value of the seminorm.
    """
    
    cdef int i, j
    cdef DTYPE_float_t value
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + x[i]**2
    
    value = np.fabs(x[l1_penalty]).sum()
    value += np.maximum(x[positive_part], 0).sum()

    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        if weights[j] < np.inf:
            value += weights[j] * norms[j]
        elif norms[j] != 0:
            return np.inf

    if check_feasibility:
        # POSITIVE_PART coordinates are penalized, not constrained (gh-103)
        xnn = x[nonnegative]
        if tuple(xnn.shape) not in [(),(0,)] and xnn.min() < -nntol:
            value = np.inf

    return value


def strong_set_mixed_lasso(cnp.ndarray[DTYPE_float_t, ndim=1] x, 
                           DTYPE_float_t lagrange_new,
                           DTYPE_float_t lagrange_cur,
                           DTYPE_float_t slope_estimate,
                           cnp.ndarray[DTYPE_intp_t, ndim=1] l1_penalty, 
                           cnp.ndarray[DTYPE_intp_t, ndim=1] unpenalized,
                           cnp.ndarray[DTYPE_intp_t, ndim=1] positive_part, 
                           cnp.ndarray[DTYPE_intp_t, ndim=1] nonnegative, 
                           cnp.ndarray[DTYPE_intp_t, ndim=1] groups,
                           cnp.ndarray[DTYPE_float_t, ndim=1] weights):
    r"""
    Strong set rule for the mixed LASSO penalty.

    Parameters
    ----------
    x : np.ndarray
        Gradient of the smooth part of the objective at the solution
        for `lagrange_cur`.
    lagrange_new : float
        New value of the Lagrange parameter.
    lagrange_cur : float
        Current value of the Lagrange parameter.
    slope_estimate : float
        Estimate of the slope used by the strong rule.
    l1_penalty, unpenalized, positive_part, nonnegative : np.ndarray
        Indices of the coordinates with an :math:`\ell_1` penalty, no
        penalty, a positive part penalty and a nonnegativity constraint.
    groups : np.ndarray
        Group index of each coordinate, or a negative value for
        coordinates that are not in a group.
    weights : np.ndarray
        Weight of each group.

    Returns
    -------
    value : np.ndarray
        Array that is 1 for the coordinates guessed to be active at
        `lagrange_new` and 0 otherwise.
    """
    
    cdef cnp.ndarray value = np.zeros_like(x)
    cdef cnp.ndarray norms = np.zeros_like(weights)
    cdef cnp.ndarray mixed_value = np.zeros_like(weights)
    cdef int i, j
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + x[i]**2
    
    value[l1_penalty] = np.fabs(x[l1_penalty]) < (slope_estimate+1)*lagrange_new - slope_estimate*lagrange_cur
    value[positive_part] = -x[positive_part] < (slope_estimate+1) * lagrange_new - slope_estimate*lagrange_cur
    value[nonnegative] = -x[nonnegative] < -np.fabs(lagrange_new - lagrange_cur) * slope_estimate

    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        mixed_value[j] = norms[j] < weights[j] * ((slope_estimate+1) * lagrange_new - slope_estimate*lagrange_cur)

    for i in range(p):
        if groups[i] >= 0:
            value[i] = mixed_value[groups[i]]

    return 1 - value

def check_KKT_mixed_lasso(cnp.ndarray[DTYPE_float_t, ndim=1] grad, 
                          cnp.ndarray[DTYPE_float_t, ndim=1] solution, 
                          DTYPE_float_t lagrange,
                          cnp.ndarray[DTYPE_intp_t, ndim=1] l1_penalty, 
                          cnp.ndarray[DTYPE_intp_t, ndim=1] unpenalized,
                          cnp.ndarray[DTYPE_intp_t, ndim=1] positive_part, 
                          cnp.ndarray[DTYPE_intp_t, ndim=1] nonnegative, 
                          cnp.ndarray[DTYPE_intp_t, ndim=1] groups,
                          cnp.ndarray[DTYPE_float_t, ndim=1] weights,
			  DTYPE_float_t tol=1.e-2,
			  DTYPE_float_t nntol=1.e-2):
    r"""
    Check the KKT conditions of a mixed LASSO problem.

    Parameters
    ----------
    grad : np.ndarray
        Gradient of the smooth part of the objective at `solution`.
    solution : np.ndarray
        Candidate solution.
    lagrange : float
        Lagrange parameter.
    l1_penalty, unpenalized, positive_part, nonnegative : np.ndarray
        Indices of the coordinates with an :math:`\ell_1` penalty, no
        penalty, a positive part penalty and a nonnegativity constraint.
    groups : np.ndarray
        Group index of each coordinate, or a negative value for
        coordinates that are not in a group.
    weights : np.ndarray
        Weight of each group.
    tol : float, optional
        Relative tolerance for the checks.
    nntol : float, optional
        Tolerance for the checks of nonnegative coordinates.

    Returns
    -------
    failing : np.ndarray
        Number of failed checks for each coordinate.
    """
    
    cdef cnp.ndarray failing = np.zeros_like(grad)
    cdef cnp.ndarray norms = np.zeros_like(weights)
    cdef cnp.ndarray snorms = np.zeros_like(weights)
    cdef int i, j
    cdef int p = groups.shape[0]
    
    # L1 check

    cdef int debug = 0
    g_l1 = grad[l1_penalty]
    if g_l1.shape not in [(), (0,)]:

        failing[l1_penalty] += np.fabs(g_l1) > lagrange * (1 + tol)
        if debug:
            print('l1 (dual) feasibility:', np.fabs(g_l1) > lagrange * (1 + tol), np.fabs(g_l1), lagrange * (1 + tol))

        # Check that active coefficients are on the boundary 
        soln_l1 = solution[l1_penalty]
        active_l1 = soln_l1 != 0

        failing_l1 = np.zeros(g_l1.shape, np.intp)
        failing_l1[active_l1] = np.fabs(-g_l1[active_l1] / lagrange - np.sign(soln_l1[active_l1])) >= tol 
        failing[l1_penalty] += failing_l1

        if debug:
            print('l1 (dual) tightness:', failing_l1, np.fabs(-g_l1[active_l1] / lagrange - np.sign(soln_l1[active_l1])))

    # Positive part

    # Check subgradient is feasible
            
    g_pp = grad[positive_part]
    if g_pp.shape not in [(), (0,)]:
        failing[positive_part] += -g_pp > lagrange * (1 + tol)
        if debug:
            print('positive part (dual) feasibility:', -g_pp > lagrange * (1 + tol))

        # Check that active coefficients are on the boundary 
        soln_pp = solution[positive_part]
        active_pp = soln_pp != 0

        failing_pp = np.zeros(g_pp.shape, np.intp)
        failing_pp[active_pp] += np.fabs(-g_pp[active_pp] / lagrange - 1) >= tol 
        if debug:
            print('positive part (dual) tightness:', -g_pp[active_pp] / lagrange - 1)
        failing[positive_part] += failing_pp

    # Nonnegative

    # Check subgradient is feasible
            
    g_nn = grad[nonnegative]
    if g_nn.shape not in [(), (0,)]:
        failing[nonnegative] += -g_nn > lagrange * tol
        if debug:
            print('nonnegative (dual) feasibility:', -g_nn > nntol)

        # Check that active coefficients are on the boundary 
        soln_nn = solution[nonnegative]
        active_nn = soln_nn != 0

        failing_nn = np.zeros(g_nn.shape, np.intp)
        failing_nn[active_nn] += -g_nn[active_nn] < -nntol
        if debug:
            print('nonnegative (dual) tightness:', -g_nn[active_nn] / lagrange - 1)
        failing[nonnegative] += failing_nn

    # group norms

    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + grad[i]**2
            snorms[groups[i]] = snorms[groups[i]] + solution[i]**2

    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])

    for i in range(p):
        if groups[i] >= 0:
            j = groups[i]

	    # check that the subgradient is feasible 
            failing[i] += norms[j] > weights[j] * lagrange * (1 + tol)

            # check that the active groups have a tight subgradient
            if snorms[j] != 0:
                failing[i] += norms[j] < weights[j] * lagrange * (1 - tol)

    return failing
   
def seminorm_mixed_lasso_dual(cnp.ndarray[DTYPE_float_t, ndim=1] x, 
                              cnp.ndarray[DTYPE_intp_t, ndim=1] l1_penalty, 
                              cnp.ndarray[DTYPE_intp_t, ndim=1] unpenalized,
                              cnp.ndarray[DTYPE_intp_t, ndim=1] positive_part, 
                              cnp.ndarray[DTYPE_intp_t, ndim=1] nonnegative, 
                              cnp.ndarray[DTYPE_intp_t, ndim=1] groups,
                              cnp.ndarray[DTYPE_float_t, ndim=1] weights,
                              cnp.ndarray[DTYPE_float_t, ndim=1] norms,
                              DTYPE_float_t nntol=1.e-5,
                              DTYPE_int_t check_feasibility=0):
    r"""
    Evaluate the dual of the mixed LASSO seminorm.

    Parameters
    ----------
    x : np.ndarray
        Argument of the dual seminorm.
    l1_penalty, unpenalized, positive_part, nonnegative : np.ndarray
        Indices of the coordinates with an :math:`\ell_1` penalty, no
        penalty, a positive part penalty and a nonnegativity constraint.
    groups : np.ndarray
        Group index of each coordinate, or a negative value for
        coordinates that are not in a group.
    weights : np.ndarray
        Weight of each group.
    norms : np.ndarray
        Work buffer with one entry per group, assumed to be zero on entry.
    nntol : float, optional
        Tolerance for the feasibility checks.
    check_feasibility : int, optional
        If nonzero, return `np.inf` if a group with weight 0 has a
        nonzero norm or a nonnegative coordinate is negative.

    Returns
    -------
    value : float
        Value of the dual seminorm.
    """
    
    cdef int i
    cdef DTYPE_float_t value
    cdef int p = groups.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + x[i]**2
    
    xl1 = x[l1_penalty]
    if xl1.shape not in [(), (0,)]:
        value = np.fabs(xl1).max()
    else:
        value = -np.inf

    xpos = x[positive_part]
    if xpos.shape not in [(), (0,)]:
        value = max(value, np.maximum(xpos, 0).max())

    for j in range(weights.shape[0]):
        norms[j] = np.sqrt(norms[j])
        if weights[j] > 0:
            value = max(value, norms[j] / weights[j])
        else:
            if check_feasibility and norms[j] > nntol:
                value = np.inf

    if check_feasibility:
        # the dual set: u = 0 for unpenalized coordinates, u >= 0 for
        # positive-part ones and u <= 0 for nonnegative ones (the polar
        # of the nonnegative orthant)
        xun = x[unpenalized]
        if xun.shape not in [(), (0,)] and np.fabs(xun).max() > nntol:
            value = np.inf
        if xpos.shape not in [(), (0,)] and xpos.min() < -nntol:
            value = np.inf
        xnn = x[nonnegative]
        if xnn.shape not in [(), (0,)] and xnn.max() > nntol:
            value = np.inf

    return value

def mixed_lasso_bound_prox(cnp.ndarray[DTYPE_float_t, ndim=1] prox_center,
                           DTYPE_float_t bound,
                           cnp.ndarray[DTYPE_intp_t, ndim=1] l1_penalty,
                           cnp.ndarray[DTYPE_intp_t, ndim=1] unpenalized,
                           cnp.ndarray[DTYPE_intp_t, ndim=1] positive_part,
                           cnp.ndarray[DTYPE_intp_t, ndim=1] nonnegative,
                           cnp.ndarray[DTYPE_intp_t, ndim=1] groups, 
                           cnp.ndarray[DTYPE_float_t, ndim=1] weights,
                           cnp.ndarray[DTYPE_float_t, ndim=1] norms,
                           cnp.ndarray[DTYPE_float_t, ndim=1] factors,
                           cnp.ndarray[DTYPE_float_t, ndim=1] projection):
    r"""
    Proximal map of the mixed LASSO penalty in bound form.

    Computes the projection of `prox_center` onto the set where the mixed
    LASSO seminorm is at most `bound`.

    Parameters
    ----------
    prox_center : np.ndarray
        Point to project.
    bound : float
        Bound on the seminorm.
    l1_penalty, unpenalized, positive_part, nonnegative : np.ndarray
        Indices of the coordinates with an :math:`\ell_1` penalty, no
        penalty, a positive part penalty and a nonnegativity constraint.
    groups : np.ndarray
        Group index of each coordinate, or a negative value for
        coordinates that are not in a group.
    weights : np.ndarray
        Weight of each group.
    norms, factors, projection : np.ndarray
        Work buffers, assumed to be zero on entry; `norms` and `factors`
        have one entry per group and `projection` one per coordinate.

    Returns
    -------
    projection : np.ndarray
        The projection of `prox_center`.
    """

    cdef int p = groups.shape[0]
    cdef int stop, i, j
    cdef float curV = 0.
    cdef double nextV, slope, intercept, curX, nextX
    
    cdef int q = norms.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + prox_center[i]**2

    for j in range(q):
        norms[j] = np.sqrt(norms[j])
    
    cdef cnp.ndarray[DTYPE_float_t, ndim=1] l1term = prox_center[l1_penalty]
    l1term = l1term[l1term != 0]
    
    cdef cnp.ndarray[DTYPE_float_t, ndim=1] ppterm = prox_center[positive_part]
    ppterm = ppterm[ppterm > 0]
    
    cdef cnp.ndarray[DTYPE_float_t, ndim=1] fnorms = np.hstack([norms,
                                                               l1term,
                                                               ppterm])
    cdef cnp.ndarray[DTYPE_float_t, ndim=1] fweights = np.ones_like(fnorms)
    fweights[:q] = weights

    cdef double cut = find_solution_piecewise_linear(bound, 
                                                     0,
                                                     fnorms,
                                                     fweights)
    if cut < np.inf:
        for j in range(weights.shape[0]):
            if norms[j] > 0:
                factors[j] = min(1., cut * weights[j] / norms[j])
            else:
                factors[j] = 1.

        for i in range(p):
            if groups[i] >= 0:
                projection[i] = prox_center[i] * factors[groups[i]]

        projection[l1_penalty] = prox_center[l1_penalty] * np.minimum(1., cut / np.fabs(prox_center[l1_penalty]))
        projection[unpenalized] = 0
        projection[positive_part] = np.clip(prox_center[positive_part], 0, cut)
        projection[nonnegative] = np.minimum(prox_center[nonnegative], 0)

        return prox_center - projection
    else:
        return prox_center

def mixed_lasso_epigraph(cnp.ndarray[DTYPE_float_t, ndim=1] center,
                         cnp.ndarray[DTYPE_intp_t, ndim=1] l1_penalty,
                         cnp.ndarray[DTYPE_intp_t, ndim=1] unpenalized,
                         cnp.ndarray[DTYPE_intp_t, ndim=1] positive_part,
                         cnp.ndarray[DTYPE_intp_t, ndim=1] nonnegative,
                         cnp.ndarray[DTYPE_intp_t, ndim=1] groups, 
                         cnp.ndarray[DTYPE_float_t, ndim=1] weights,
                         cnp.ndarray[DTYPE_float_t, ndim=1] norms,
                         cnp.ndarray[DTYPE_float_t, ndim=1] factors,
                         cnp.ndarray[DTYPE_float_t, ndim=1] projection):
    r"""
    Projection onto the epigraph of the mixed LASSO seminorm.

    Parameters
    ----------
    center : np.ndarray
        Point to project: all entries but the last are the
        coefficients, the last entry is the value of the epigraph
        variable.
    l1_penalty, unpenalized, positive_part, nonnegative : np.ndarray
        Indices of the coordinates with an :math:`\ell_1` penalty, no
        penalty, a positive part penalty and a nonnegativity constraint.
    groups : np.ndarray
        Group index of each coordinate, or a negative value for
        coordinates that are not in a group.
    weights : np.ndarray
        Weight of each group.
    norms, factors, projection : np.ndarray
        Work buffers, assumed to be zero on entry; `norms` and `factors`
        have one entry per group and `projection` one per coefficient.

    Returns
    -------
    result : np.ndarray
        The projection of `center` onto the epigraph.
    """

    cdef cnp.ndarray[DTYPE_float_t, ndim=1] x = center[:-1]
    cdef DTYPE_float_t norm = center[-1]

    cdef int p = groups.shape[0]
    cdef int stop, i, j
    cdef float curV = 0.
    cdef double nextV, slope, intercept, curX, nextX
    
    cdef int q = norms.shape[0]
    
    for i in range(p):
        if groups[i] >= 0:
            norms[groups[i]] = norms[groups[i]] + x[i]**2

    for j in range(q):
        norms[j] = np.sqrt(norms[j])
    
    cdef cnp.ndarray[DTYPE_float_t, ndim=1] l1term = x[l1_penalty]
    l1term = l1term[l1term != 0]
    
    cdef cnp.ndarray[DTYPE_float_t, ndim=1] ppterm = x[positive_part]
    ppterm = ppterm[ppterm > 0]
    
    cdef cnp.ndarray[DTYPE_float_t, ndim=1] fnorms = np.hstack([norms,
                                                               l1term,
                                                               ppterm])
    cdef cnp.ndarray[DTYPE_float_t, ndim=1] fweights = np.ones_like(fnorms)
    fweights[:q] = weights

    cdef double cut = find_solution_piecewise_linear(norm, 
                                                     1,
                                                     fnorms,
                                                     fweights)

    cdef cnp.ndarray[DTYPE_float_t, ndim=1] result = np.zeros_like(center)
    if cut < np.inf:
        if norm + cut >= 0:
            for j in range(weights.shape[0]):
                if norms[j] > 0:
                    factors[j] = min(1., cut * weights[j] / norms[j])
                else:
                    factors[j] = 1.

            for i in range(p):
                if groups[i] >= 0:
                    projection[i] = x[i] * factors[groups[i]]

            projection[l1_penalty] = x[l1_penalty] * np.minimum(1., cut / np.fabs(x[l1_penalty]))
            projection[unpenalized] = 0
            projection[positive_part] = np.clip(x[positive_part], 0, cut)
            projection[nonnegative] = np.minimum(x[nonnegative], 0)

            result[:-1] = x - projection
            result[-1] = norm + cut
        # else, it is zero from initialization
    else:
        result[:] = center

    return result
