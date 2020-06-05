import numpy as np

have_sklearn = True
try:
    from .base import (sklearn_regression,
                       sklearn_regression_lagrange,
                       sklearn_classifier,
                       sklearn_classifier_lagrange)
except ImportError:
    have_sklearn = False

from ..smooth.mglm import (mglm, 
                           multinomial_loglike)

if have_sklearn:
    class sklearn_multinomial(sklearn_regression):

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

        """

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)

            successes = response

            return mglm.multinomial(X, 
                                    successes,
                                    case_weights=case_weights_,
                                    coef=self.coef,
                                    saturated_offset=offset_)

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            successes = response

            loss = lambda yhat: multinomial_loglike(successes.shape,
                                                    successes,
                                                    case_weights=case_weights).smooth_objective(yhat, 'func')

            if self.score_method == 'deviance':
                return np.sum(loss(predictions))
            elif self.score_method == 'mean_deviance':
                return np.mean(loss(predictions))
            elif self.score_method == 'R2':
                SSE = np.sum(loss(predictions))
                SST = np.sum(loss(response.mean() * np.ones_like(response)))
                return 1 - SSE / SST
            elif self.score_method == 'accuracy':
                labels = predictions > 0
                return np.mean(labels == response)
            else:
                return np.nan

    class sklearn_multinomial_lagrange(sklearn_regression_lagrange, sklearn_multinomial):
        pass

    class sklearn_multinomial_classifier(sklearn_classifier):

        """

        A simple regression mixin for sklearn
        that allows any atom to be used as a regularizer.

        """

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)

            successes = response

            return mglm.multinomial(X, 
                                    successes,
                                    case_weights=case_weights_,
                                    coef=self.coef,
                                    saturated_offset=offset_ # should be None
                                    )

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            successes = response

            if self.score_method == 'accuracy':
                return np.mean(predictions == successes)
            return np.nan

        def predict(self, X):
            """
            Predict labels in classification setting.

            Parameters
            ----------

            X : np.ndarray((n, p))
                Feature matrix.

            Returns
            -------

            probs : np.ndarray(n)
                Predictions from classification model.

            """
            linpred = X.dot(self._coefs)
            return np.argmax(linpred, 1)

        def predict_proba(self, X):
            """
            Predict new probabilities in classification setting.

            Parameters
            ----------

            X : np.ndarray((n, p))
                Feature matrix.

            Returns
            -------

            probs : np.ndarray(n)
                Predictions from classification model.

            """
            linpred = X.dot(self._coefs)
            exp_lin = np.exp(linpred)
            return exp_lin / exp_lin.sum(1)

    class sklearn_multinomial_classifier_lagrange(sklearn_multinomial_classifier):
        pass

