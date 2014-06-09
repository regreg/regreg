import numpy as np
from ..affine import affine_transform

class multiscale(affine_transform):

    """
    An affine transform representing the
    multiscale changepoint transform.

    This transform centers the input
    and then computes the average over
    all intervals of a minimum size.
    """

    def __init__(self, p, minsize=None):
        """
        Parameters
        ----------

        p : int
            Length of signal.

        minsize : int
            Smallest interval to consider.
            Defaults to p**(1/3.)

        """
        self.p = p
        self.minsize = minsize or int(np.around(p**(1/3.)))
        self._slices = []
        self._sizes = []
        for i in range(p):
            for j in range(i,p):
                if j - i >= self.minsize:
                    self._slices.append((i,j))
                    self._sizes.append(j-i)
        self._sizes = np.array(self._sizes)
        self.input_shape = (p,)
        self.output_shape = (len(self._slices),)

    def linear_map(self, x):
        """
        Given a p-vector `x` compute the average of
        `x - x.mean()` over each interval
        of size greater than `self.minsize`.

        Parameters
        ----------

        x : np.float(self.input_shape)

        Returns
        -------

        v : np.float(self.output_shape)

        """
        x_centered = x - x.mean()
        output = np.zeros(self.output_shape)
        for k, ij in enumerate(self._slices):
            i, j = ij
            output[k] = x_centered[i:j].mean()
        return output

    def affine_map(self, x):
        return self.linear_map(x)

    def offset_map(self, x):
        return x

    def adjoint_map(self, v):
        """
        Parameters
        ----------

        v : np.float(self.output_shape)

        Returns
        -------

        v : np.float(self.input_shape)
        """

        v_scaled = v / self._sizes
        output = np.zeros(self.input_shape)
        non0 = np.nonzero(v_scaled)[0]
        if non0.shape != ():
            for k in non0:
                i, j = self._slices[k]
                size = (j-i)
                output[i:j] += v_scaled[k]
        return output - output.mean()



