"""
Isotonic regression by the pool adjacent violators algorithm (PAVA).

Closely based on

https://gist.github.com/ajtulloch/9447957#file-_isotonic-pyx

which has been merged into sklearn.
"""

import numpy as np, sys
cimport numpy as np

DTYPE_float = float
ctypedef np.float_t DTYPE_float_t

def _isotonic_regression(np.ndarray[DTYPE_float_t, ndim=1] y,
                         np.ndarray[DTYPE_float_t, ndim=1] weight,
                         np.ndarray[DTYPE_float_t, ndim=1] solution):
    """
    Weighted isotonic (non-decreasing) regression of `y`.

    Parameters
    ----------
    y : np.ndarray
        One-dimensional array of values.
    weight : np.ndarray
        Positive weights, same shape as `y`.
    solution : np.ndarray
        Output array, same shape as `y`; overwritten in place.

    Returns
    -------
    solution : np.ndarray
        The fitted non-decreasing values (the same object as the
        `solution` argument).
    """

    cdef int i, pooled, n, k
    cdef double numerator, denominator

    n = y.shape[0]

    # The algorithm proceeds by iteratively updating the solution
    # array.

    for i in range(n):
        solution[i] = y[i]

    if n <= 1:
        return solution

    n -= 1
    while 1:
        # repeat until there are no more adjacent violators.
        i = 0
        pooled = 0
        while i < n:
            k = i
            while k < n and solution[k] >= solution[k + 1]:
                k += 1
            if solution[i] != solution[k]:
                # solution[i:k + 1] is a decreasing subsequence, so
                # replace each point in the subsequence with the
                # weighted average of the subsequence.
                numerator = 0.0
                denominator = 0.0
                for j in range(i, k + 1):
                    numerator += solution[j] * weight[j]
                    denominator += weight[j]
                for j in range(i, k + 1):
                    solution[j] = numerator / denominator
                pooled = 1
            i = k + 1
        # Check for convergence
        if pooled == 0:
            break

    return solution
