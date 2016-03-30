"""
A module for commonly used losses that are not in `regreg.smooth.glm`. Only
current example is Huberized SVM.

"""

import numpy as np

from . import smooth_atom
from ..affine import astransform
from ..identity_quadratic import identity_quadratic
from ..atoms.seminorms import positive_part

class huberized_svm(smooth_atom):

    objective_template = r"""\ell^{\text{Huber SVM}}\left(%(var)s\right)"""

    def __init__(self, 
                 X, 
                 labels,
                 smoothing_parameter,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None):

        shape = astransform(X).input_shape
        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)
        
        if len(np.unique(labels)) > 2:
            raise ValueError('labels should be binary')
        # force labels to be \pm 1
        Y = 2 * (labels == np.max(labels)) - 1
        atom = positive_part.affine(-Y[:, None] * X, np.ones_like(Y), lagrange=1.)
        Q = identity_quadratic(smoothing_parameter, 0, 0, 0)
        self.smoothed_atom = atom.smoothed(Q)
                                         
    def smooth_objective(self, beta, mode='both', check_feasibility=False):
        """

        Parameters
        ----------

        beta : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `beta` is not
            in the domain.

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `beta`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        return self.smoothed_atom.smooth_objective(beta,
                                                   mode=mode,
                                                   check_feasibility=check_feasibility)

