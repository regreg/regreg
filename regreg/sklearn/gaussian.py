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

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

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

            if self.score_method == 'deviance':
                return loss(predictions)
            elif self.score_method == 'mean_deviance':
                return loss(predictions) / predictions.shape[0]
            elif self.score_method == 'R2':
                SSE = np.sum(loss(predictions))
                SST = np.sum(loss(response.mean() * np.ones_like(response)))
                return 1 - SSE / SST
            else:
                return np.nan

    class sklearn_gaussian_lagrange(sklearn_regression_lagrange, sklearn_gaussian):

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

        """
        pass

