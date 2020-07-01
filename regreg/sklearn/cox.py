import numpy as np

have_sklearn = True
try:
    from .base import (sklearn_regression,
                       sklearn_regression_lagrange)
except ImportError:
    have_sklearn = False
from ..smooth.glm import (glm, 
                          cox_loglike)

if have_sklearn:
    class sklearn_cox(sklearn_regression):

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

        """

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)

            event_times = response[:,0]
            censoring = response[:,1]
            return glm.cox(X, 
                           event_times,
                           censoring,
                           case_weights=case_weights_,
                           coef=self.coef,
                           saturated_offset=offset_)

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            event_times = response[:,0]
            censoring = response[:,1]

            loss = lambda yhat: cox_loglike(event_times.shape,
                                            event_times,
                                            censoring,
                                            case_weights=case_weights).smooth_objective(yhat, 'func')

            # factor of 2 to form proper deviance (default is negative log-likelihood,
            # while deviance is 2 * negative log-likelihood
            # negative sign is to align with sklearn's maximizing a score with grid search

            if self.score_method == 'deviance':
                return -2 * loss(predictions)
            elif self.score_method == 'mean_deviance':
                return -2 * loss(predictions) / predictions.shape[0]
            elif self.score_method == 'R2':
                SSE = 2 * loss(predictions)
                SST = loss(response.mean() * np.ones(response.shape[0])) # X: correct for Cox?
                return 1 - SSE / SST
            elif self.score_method == 'C-index':
                return np.nan
            else:
                return np.nan

    class sklearn_cox_lagrange(sklearn_regression_lagrange, sklearn_cox):

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

        """
        pass

