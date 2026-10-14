"""
Pure Python isotonic regression by the pool adjacent violators algorithm.

Based on https://gist.github.com/fabianp/3081831
"""
import numpy as np

# Author : Alexandre Gramfort
# license : BSD

def pav(y):
    """
    Isotonic (non-decreasing) regression of `y`.

    Uses the pool adjacent violators method to produce a monotonic
    smoothing of `y`.

    Parameters
    ----------
    y : array-like
        One-dimensional array of values.

    Returns
    -------
    v : np.ndarray
        The non-decreasing sequence closest to `y` in least squares.

    Notes
    -----
    Translated from Matlab by Sean Collins (2006) as part of the EMAP
    toolbox.

    Examples
    --------
    >>> from regreg.atoms._isotonic_py import pav
    >>> pav([1., 3., 2., 4.])
    array([1. , 2.5, 2.5, 4. ])
    """
    y = np.asarray(y)
    assert y.ndim == 1
    n_samples = len(y)
    v = y.copy()
    lvls = np.arange(n_samples)
    lvlsets = np.c_[lvls, lvls]
    flag = 1
    while flag:
        deriv = np.diff(v)
        if np.all(deriv >= 0):
            break

        viol = np.where(deriv < 0)[0]
        start = lvlsets[viol[0], 0]
        last = lvlsets[viol[0] + 1, 1]
        s = 0
        n = last - start + 1
        for i in range(start, last + 1):
            s += v[i]

        val = s / n
        for i in range(start, last + 1):
            v[i] = val
            lvlsets[i, 0] = start
            lvlsets[i, 1] = last
    return v
