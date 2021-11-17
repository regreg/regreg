from copy import copy

import numpy as np
from scipy.stats import rankdata

from . import smooth_atom, affine_smooth
from .cox_utils import (cox_objective,
                        cox_gradient,
                        cox_hessian)

class cox_loglike(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    objective_template = r"""\ell^{\text{Cox}}\left(%(var)s\right)"""

    def __init__(self, 
                 shape, 
                 event_times,
                 censoring,
                 coef=1., 
                 offset=None,
                 quadratic=None,
                 initial=None,
                 case_weights=None):

        smooth_atom.__init__(self,
                             shape,
                             offset=offset,
                             quadratic=quadratic,
                             initial=initial,
                             coef=coef)

        self.data = np.asarray([event_times, censoring]).T

        self._ordering = np.argsort(self.event_times).astype(np.intp)
        self._rankmax = (rankdata(self.event_times, method='max') - 1).astype(np.intp)
        self._rankmin = (rankdata(self.event_times, method='min') - 1).astype(np.intp)

        if case_weights is not None:
            case_weights = np.asarray(case_weights).astype(np.float)
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
        else:
            self.case_weights = np.ones(self.event_times.shape, np.float)

        # buffers to store results used by C code
        self._G = np.zeros(self.event_times.shape, np.float) # gradient 
        self._exp_buffer = np.zeros(self.event_times.shape, np.float) # exp(eta)
        self._exp_accum = np.zeros(self.event_times.shape, np.float) # accum of exp(eta)
        self._expZ_accum = np.zeros(self.event_times.shape, np.float) # accum of Z*exp(eta)
        self._outer_1st = np.zeros(self.event_times.shape, np.float) # for log(W)
        self._outer_2nd = np.zeros(self.event_times.shape, np.float) # used in Hessian

    def smooth_objective(self, 
                         natural_param, 
                         mode='both', 
                         check_feasibility=False,
                         case_weights=None):
        """

        Evaluate the smooth objective, computing its value, gradient or both.

        Parameters
        ----------

        natural_param : ndarray
            The current parameter values.

        mode : str
            One of ['func', 'grad', 'both']. 

        check_feasibility : bool
            If True, return `np.inf` when
            point is not feasible, i.e. when `natural_param` is not
            in the domain.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `natural_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        eta = natural_param # shorthand

        eta = self.apply_offset(eta)

        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        censoring = self.censoring

        if mode in ['both', 'grad']:
            G = cox_gradient(self._G,
                             eta,
                             self._exp_buffer,
                             self._exp_accum,
                             self._outer_1st,
                             cw,
                             censoring,
                             self._ordering,
                             self._rankmin,
                             self._rankmax,
                             eta.shape[0])

        if mode in ['both', 'func']:
            F = cox_objective(eta,
                              self._exp_buffer,
                              self._exp_accum,
                              self._outer_1st,
                              cw,
                              censoring,
                              self._ordering,
                              self._rankmin,
                              self._rankmax,
                              eta.shape[0])

        if mode == 'both':
            return self.scale(F), self.scale(G)
        elif mode == 'grad':
            return self.scale(G)
        elif mode == 'func':
            return self.scale(F)
        else:
            raise ValueError("mode incorrectly specified")

    # Begin loss API

    def hessian_mult(self, 
                     natural_param, 
                     right_vector,
                     case_weights=None):
        """
        Evaluate Hessian of the loss at a pair of vectors.

        Parameters
        ----------

        natural_param : ndarray
            Parameters where Hessian will be evaluated.

        left_vec : ndarray
            Vector on the left in Hessian evaluation.

        right_vec : ndarray
            Vector on the left in Hessian evaluation.

        case_weights : ndarray
            Non-negative case weights

        Returns
        -------

        value : float
            Hessian evaluated at this pair of left and right vectors
        """

        if case_weights is None:
            case_weights = np.ones_like(natural_param)
        cw = case_weights
        if self.case_weights is not None:
            cw *= self.case_weights

        eta = natural_param # shorthand
        eta = self.apply_offset(eta)
        censoring = self.censoring

        H = np.zeros(eta.shape, np.float)

        return cox_hessian(H,
                           eta,
                           right_vector,
                           self._exp_buffer,
                           self._exp_accum,
                           self._expZ_accum,
                           self._outer_1st,
                           self._outer_2nd,
                           cw,
                           censoring,
                           self._ordering,
                           self._rankmin,
                           self._rankmax,
                           eta.shape[0])

    def get_data(self):
        return np.array([self.event_times, self.censoring]).T

    def set_data(self, data):
        event_times, censoring = np.asarray(data).T
        self.event_times, self.censoring = (np.asarray(event_times),
                                            np.asarray(censoring).astype(np.intp))

    data = property(get_data, set_data)

    def __copy__(self):
        event_times, censoring = self.event_times, self.censoring
        return cox_loglike(self.shape,
                           copy(event_times),
                           copy(censoring),
                           coef=self.coef,
                           offset=copy(self.offset),
                           quadratic=copy(self.quadratic),
                           initial=copy(self.coefs),
                           case_weights=copy(self.case_weights))

    def subsample(self, case_idx):
        """
        Create a saturated loss using a subsample of the data.
        Makes a copy of the loss and 
        multiplies case_weights by the indicator for
        `idx`.

        Parameters
        ----------

        idx : index
            Indices of np.arange(n) to keep.

        Returns
        -------

        subsample_loss : `smooth_atom`
            Loss after discarding all
            cases not in `idx.

        """
        loss_cp = copy(self)
        if loss_cp.case_weights is None:
            case_weights = loss_cp.case_weights = np.ones(self.shape[0])
        else:
            case_weights = loss_cp.case_weights

        idx_bool = np.zeros_like(case_weights, np.bool)
        idx_bool[case_idx] = 1

        case_weights *= idx_bool
        return loss_cp

    def latexify(self, var=None, idx=''):
        # a trick to get latex representation looking right
        # coxph should be written similar to logistic
        # composed with a linear transform (TODO)
        return smooth_atom.latexify(self, var=var, idx=idx)
