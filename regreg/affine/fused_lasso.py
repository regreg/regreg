import numpy as np
from scipy import sparse

from ..affine import astransform, affine_transform


def difference_transform(X, order=1, sorted=False,
                         transform=False):
    """
    Compute the divided difference matrix for `X`
    after sorting `X`.

    Parameters
    ----------
    X : ndarray
        1D array of coordinates where discrete derivative is computed.
    order : int
        What order of difference should we compute?
    sorted : bool
        Is `X` sorted?
    transform : bool
        If True, return a linear_transform rather
        than a sparse matrix.

    Returns
    -------
    D : scipy.sparse.csr_matrix or linear_transform
        Matrix of divided differences of sorted `X`, of
        shape ``(n-order, n)``.
    """
    if not sorted:
        X = np.sort(X)
    X = np.asarray(X)
    n = X.shape[0]
    Dfinal = np.identity(n)
    for j in range(1, order+1):
        D = (-np.identity(n-j+1)+np.diag(np.ones(n-j),k=1))[:-1]
        steps = X[j:]-X[:-j]
        inv_steps = np.zeros(steps.shape)
        inv_steps[steps != 0] = 1. / steps[steps != 0]
        D = np.dot(np.diag(inv_steps), D)
        Dfinal = np.dot(D, Dfinal)
    if not transform:
        return sparse.csr_matrix(Dfinal)
    return astransform(Dfinal)

class trend_filter(affine_transform):
    """
    Linear transform computing divided differences of a given order
    on a set of knots, as used in trend filtering.

    Parameters
    ----------
    knots : ndarray
        1D array of knot locations.
    order : int, optional
        Order of the divided differences.
    sorted : bool, optional
        Are the knots sorted?
    """

    def __init__(self, knots, order=1, sorted=False):
        self.order = order
        self.knots = np.sort(knots)
        self.steps = self.knots[1:] - self.knots[:-1]

        self.linear_transform = difference_transform(self.knots, order=order, sorted=True,
                                                     transform=True)
        self.affine_offset = None
        self.input_shape = self.linear_transform.input_shape
        self.output_shape = self.linear_transform.output_shape

    @classmethod
    def grid(cls, m, order=1, sorted=False):
        """
        Trend filter transform on the equally spaced knots ``np.arange(m)``.
        """
        return cls(np.arange(m), order=order, sorted=sorted)

    def linear_map(self, x):
        return self.linear_transform.linear_map(x)

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        return self.linear_transform.adjoint_map(x)


class trend_filter_inverse(affine_transform):
    """
    Pseudo-inverse of a first order `trend_filter` transform.

    Maps divided differences back to a mean-centered sequence
    on the knots (via cumulative sums).

    Parameters
    ----------
    knots : ndarray
        1D array of knot locations.
    order : int, optional
        Order of the divided differences. Only ``order=1`` is implemented.
    sorted : bool, optional
        Are the knots sorted?
    """

    def __init__(self, knots, order=1, sorted=False):
        self.order = order
        self.knots = np.sort(knots)
        self.steps = self.knots[1:] - self.knots[:-1]
        if order != 1:
            raise ValueError('pseudo inverse only worked out for first order')
            
        dtransform = difference_transform(self.knots, order=order, sorted=True,
                                          transform=True)

        self.affine_offset = None
        self.output_shape = dtransform.input_shape
        self.input_shape = dtransform.output_shape

    @classmethod
    def grid(cls, m, order=1, sorted=False):
        """
        Inverse trend filter transform on the equally spaced
        knots ``np.arange(m)``.
        """
        return cls(np.arange(m), order=order, sorted=sorted)

    def linear_map(self, x):
        if x.ndim == 1:
            v = np.zeros(self.output_shape)
            v[1:] = np.cumsum(x * self.steps)
            v -= v.mean()
            return v
        elif x.ndim == 2:
            # assuming m is the first axis
            v = np.zeros((self.output_shape[0], x.shape[1]))
            v[1:] = np.cumsum(x * self.steps[:,np.newaxis], axis=0)
            v -= v.mean(0)
            return v

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, x):
        if x.ndim == 1:
            x = x - x.mean(0)
            C = np.cumsum(x[1:][::-1])[::-1]
            return C * self.steps
        if x.ndim == 2:
            # assuming m is the first axis
            x = x - x.mean(0)[np.newaxis,:]
            C = np.cumsum(x[1:][::-1], 1)[::-1]
            return C * self.steps[:,np.newaxis]


