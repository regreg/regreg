import numpy as np
from scipy.stats import rankdata

from . import smooth_atom, affine_smooth

class cox_loglike(smooth_atom):

    """
    A class for combining the logistic log-likelihood with a general seminorm
    """

    objective_template = r"""\ell^{\text{logit}}\left(%(var)s\right)"""
    #TODO: Make init more standard, replace np.dot with shape friendly alternatives in case successes.shape is (n,1)

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

        self.data = (event_times, censoring)

        self._ordering = np.argsort(self.event_times)
        self._rev_ordering = self._ordering[::-1]
        self._ordered_times = self.event_times[self._ordering]
        self._ordered_censoring = self.censoring[self._ordering]
        self._ranking_max = rankdata(self._ordered_times, method='max') - 1
        self._ranking_min = rankdata(self._ordered_times, method='min') - 1

        if case_weights is not None:
            if not np.all(case_weights >= 0):
                raise ValueError('case_weights should be non-negative')
            self.case_weights = np.asarray(case_weights)
            if self.case_weights.shape != self.successes.shape:
                raise ValueError('case_weights should have same shape as successes')
        else:
            self.case_weights = None

    def smooth_objective(self, natural_param, mode='both', check_feasibility=False):
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

        Returns
        -------

        If `mode` is 'func' returns just the objective value 
        at `natural_param`, else if `mode` is 'grad' returns the gradient
        else returns both.
        """
        
        eta = natural_param # shorthand

        eta = self.apply_offset(eta)

        exp_w = np.exp(eta)
        risk_dens = np.cumsum(exp_w[self._rev_ordering])[::-1][self._ranking_min]
        
        if mode in ['both', 'grad']:
            grad_o = np.cumsum(self._ordered_censoring / risk_dens)[self._ranking_max]
            G = np.zeros_like(grad_o)
            G[self._ordering] = self._ordered_censoring - exp_w[self._ordering] * grad_o

        if mode in ['both', 'func']:
            F = np.sum(self._ordered_censoring * (eta[self._ordering] - np.log(risk_dens)))

        if mode == 'both':
            return self.scale(F), self.scale(G)
        elif mode == 'grad':
            return self.scale(G)
        elif mode == 'func':
            return self.scale(F)
        else:
            raise ValueError("mode incorrectly specified")

    # Begin loss API

    def hessian(self, natural_param):
        """
        Hessian of the loss.

        Parameters
        ----------

        natural_param : ndarray
            Parameters where Hessian will be evaluated.

        Returns
        -------

        hess : ndarray
            A 1D-array representing the diagonal of the Hessian
            evaluated at `natural_param`.
        """
        raise NotImplementedError('cox hessian not worked out: not a diagonal matrix')

    def get_data(self):
        return self.event_times, self.censoring

    def set_data(self, data):
        event_times, censoring = data
        self.event_times, self.censoring = (np.asarray(event_times),
                                            np.asarray(censoring))

    data = property(get_data, set_data)

    def __copy__(self):
        event_times, censoring = self.data
        return cox_loglike(self.shape,
                           copy(event_times),
                           copy(censoring),
                           coef=self.coef,
                           offset=copy(self.offset),
                           quadratic=copy(self.quadratic),
                           initial=copy(self.coefs),
                           case_weights=copy(self.case_weights))


