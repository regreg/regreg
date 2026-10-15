"""sklearn estimators for Gaussian (least squares) regression."""

import numpy as np

have_sklearn = True
try:
    from .base import (sklearn_regression,
                       sklearn_regression_lagrange)
except ImportError:
    have_sklearn = False
from ..smooth.glm import (glm, 
                          gaussian_loglike)

if have_sklearn:
    class sklearn_gaussian(sklearn_regression):

        """Regularized Gaussian (least squares) regression for sklearn.

        Any atom can be used as a regularizer.
        """

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)
            return glm.gaussian(X, 
                                response, 
                                case_weights=case_weights_,
                                coef=self.coef,
                                saturated_offset=offset_)

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            loss = lambda yhat: gaussian_loglike(response.shape,
                                                 response,
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
                # the null model fits the (weighted) mean
                SST = 2 * loss(np.average(response, weights=case_weights) * np.ones_like(response))
                return 1 - SSE / SST
            else:
                return np.nan

    class sklearn_gaussian_lagrange(sklearn_regression_lagrange, sklearn_gaussian):

        """Lagrange-form Gaussian regression for sklearn."""
        pass

