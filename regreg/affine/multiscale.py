from copy import copy

import numpy as np
from ..affine import affine_transform

class multiscale(affine_transform):

    """
    An affine transform representing the
    multiscale changepoint transform.

    This transform centers the input
    and then computes the average over
    all intervals of a minimum size (or over a given
    list of intervals).
    """

    dtype = np.dtype([('start', int),
                      ('end', int)])

    def __init__(self, p, minsize=None, slices=[],
                 scaling=None):
        """
        Parameters
        ----------
        p : int
            Length of signal.
        minsize : int, optional
            Smallest interval to consider.
            Defaults to ``p**(1/3.)``.
        slices : sequence, optional
            A list of ``(start, end)`` intervals to use in the transform.
            Will be coerced to have
            ``dtype([('start', int), ('end', int)])``.
            If empty, all intervals of length at least `minsize`
            are used.
        scaling : ndarray, optional
            An optional scaling, of shape ``slices.shape``, to apply after
            computing mean over each interval.
        """
        self.p = p
        self.minsize = minsize or int(np.around(p**(1/3.)))
        self.slices = np.asarray(slices, self.dtype)
        self.scaling = scaling
        if self.slices.shape in [(), (0,)]:
            self._all = True # if computing all differences of a certain size things can be sped up
            S = np.indices((p,p)).reshape((2,-1))
            self._mask = (S[1]-S[0] >= self.minsize)
            slices = S[:,self._mask].T
            self.slices = np.empty(slices.shape[0], self.dtype)
            self.slices['start'] = slices[:,0]
            self.slices['end'] = slices[:,1]
        else:
            self._all = False
        self.sizes = self.slices['end'] - self.slices['start'] * 1.
        self.input_shape = (p,)
        self.output_shape = (len(self.slices),)

    def __copy__(self):
        M = multiscale(self.p)
        M.minsize = self.minsize
        M.slices = copy(self.slices)
        M.scaling = copy(self.scaling)
        if hasattr(self, "_mask"):
            M._mask = copy(self._mask)
        M._all = self._all
        M.sizes = copy(self.sizes)
        M.input_shape = self.input_shape
        M.output_shape = self.output_shape
        return M

    def update_slices(self, slices, scaling=None):
        """
        Change the intervals computed by the multiscale
        transform.

        Parameters
        ----------
        slices : sequence
            List of ``(i,j)`` intervals to compute.
            Will be converted to have
            ``dtype([('start', int), ('end', int)])``.
        scaling : ndarray, optional
            An optional scaling to apply after computing the
            mean over each interval.
        """
        slices = np.asarray(slices)
        if slices.dtype != self.dtype:
            self.slices = np.empty(slices.shape[0], self.dtype)
            self.slices['start'] = slices[:,0]
            self.slices['end'] = slices[:,1]
            self._all = False
        else:
            self.slices = slices
            self._all = False
        self.sizes = self.slices['end'] - self.slices['start'] * 1.
        self.output_shape = (len(self.slices),)
        self.scaling = scaling

    def linear_map(self, x):
        """
        Compute the (scaled) average of ``x - x.mean()`` over each interval.

        If no intervals were specified, these are all intervals
        of size at least ``self.minsize``.

        Parameters
        ----------
        x : ndarray
            Signal of shape ``self.input_shape``.

        Returns
        -------
        v : ndarray
            Interval averages, of shape ``self.output_shape``.
        """
        x_centered = x - x.mean()
        output = np.zeros(self.output_shape)
        cumsum = np.cumsum(x_centered)
        if not self._all:
            for k, ij in enumerate(self.slices):
                i, j = ij
                if i >= 1:
                    output[k] = cumsum[j-1] - cumsum[i-1] 
                else:
                    output[k] = cumsum[j-1]
                output[k] /= j - i
        else:
            _cumsum = np.hstack([np.zeros(1), cumsum])
            D = np.subtract.outer(_cumsum, _cumsum)[:-1,:-1].reshape(-1)[self._mask]
            output = -D / self.sizes
        if self.scaling is not None:
            output *= self.scaling
        return output

    def affine_map(self, x):
        return self.linear_map(x)

    def adjoint_map(self, v):
        """
        Apply the adjoint of the multiscale transform to `v`.

        Parameters
        ----------
        v : ndarray
            Array of shape ``self.output_shape``.

        Returns
        -------
        output : ndarray
            Array of shape ``self.input_shape``.
        """

        if v.ndim == 1:
            v_scaled = v / self.sizes
        else:
            v_scaled = v / self.sizes[:,None]
        if self.scaling is not None:
            v_scaled *= self.scaling
        if not self._all:
            output = np.zeros(self.input_shape)
            non0 = np.nonzero(v_scaled)[0]
            if non0.shape != ():
                for k in non0:
                    i, j = self.slices[k]
                    output[i:j] += v_scaled[k]
        else:
            _output1 = np.zeros((self.input_shape[0], self.input_shape[0])).reshape(-1)
            _output1[self._mask] = v_scaled
            _output1.shape = (self.input_shape[0], self.input_shape[0])
            _output2 = _output1.sum(0) - _output1.sum(1)
            output = -np.cumsum(_output2)
        return output - output.mean()
        
    def form_matrix(self, slices):
        """
        Form a matrix with given slices of multiscale transform.

        Parameters
        ----------
        slices : sequence
            List of ``(i,j)`` intervals.

        Returns
        -------
        X : ndarray
            Matrix of shape ``(p, len(slices))`` whose columns are the
            centered indicators of each interval divided by the square
            root of its length.
        length : ndarray
            Length of each interval.
        """
        p = self.input_shape[0]
        X_E = np.zeros((p, len(slices)))
        length = np.zeros(len(slices))
        for k, ij in enumerate(slices):
            i,j = ij
            X_E[i:j, k] = 1.
            length[k] = j-i

        X_hat = X_E - X_E.mean(0)[None,:]
        return X_hat / np.sqrt(length), length

def SMUCE_stat(M, Z, sigma=None):
    """
    Compute multiscale test statistic of Chan and Walther.

    Used in the SMUCE algorithm and in the multiscale change point
    algorithm in the square-root LASSO selective inference paper.

    Parameters
    ----------
    M : multiscale
        Multiscale transform defining the intervals.
    Z : ndarray
        Data on which to compute the statistic.
    sigma : float, optional
        Noise level. Defaults to ``np.std(Z)``.

    Returns
    -------
    T : float
        Value of the statistic.
    slice : np.void
        The ``(start, end)`` interval achieving the maximum.

    Notes
    -----
    M's scaling is ignored, even if not None: ``np.sqrt(M.sizes)`` is used
    instead, and M's scaling is restored afterwards.
    """

    p = Z.shape[0]
    M.scaling, old = np.sqrt(M.sizes), M.scaling
    try:
        Z0 = Z - Z.mean()
        if sigma is None:
            sigma_hat = np.std(Z)
        else:
            sigma_hat = sigma
        T = np.fabs(M.linear_map(Z0) / sigma_hat) - np.sqrt(2 * np.log(p / M.sizes))
    finally:
        M.scaling = old
    idx = np.argmax(T)
    return T[idx], M.slices[idx]

def choose_tuning_parameter(M, ndraw=100, quantile=0.95, sigma=None, rng=None):
    """
    Choose tuning parameter by Monte Carlo.

    Used in the SMUCE algorithm and in the multiscale change point
    algorithm in the square-root LASSO selective inference paper.

    Parameters
    ----------
    M : multiscale
        Multiscale transform defining the intervals.
    ndraw : int
        How many Monte Carlo draws.
    quantile : float
        Which quantile of Monte Carlo draws to choose.
    sigma : float, optional
        Noise level passed to `SMUCE_stat`.
    rng : int or `np.random.Generator`, optional
        Seed or generator for the Monte Carlo draws, passed to
        `np.random.default_rng`.

    Returns
    -------
    value : float
        The `quantile` of `SMUCE_stat` over `ndraw` draws of
        standard normal noise.

    Notes
    -----
    M's scaling is ignored, even if not None.
    """
    rng = np.random.default_rng(rng)
    V = []
    p = M.input_shape[0]
    for i in range(ndraw):
        Z = rng.standard_normal(p)
        V.append(SMUCE_stat(M, Z, sigma=sigma)[0])
    return np.percentile(V, 100*quantile)

