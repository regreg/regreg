import numpy as np

have_sklearn = True
try:
    from .base import (sklearn_regression,
                       sklearn_regression_lagrange)
except ImportError:
    have_sklearn = False

from ..smooth.glm import (glm, 
                          huber_svm)

if have_sklearn:
    class sklearn_huber_svm(sklearn_regression):

        def __init__(self, 
                     smoothing_parameter,
                     atom_constructor, 
                     atom_params, 
                     case_weights=False,
                     offset=False,
                     coef=1., 
                     score_method='deviance'):

            """

            Parameters
            ----------

            smoothing_parameter : float
                Smoothing parameter for Huber SVM loss.

            atom_constructor : callable
                Atom constructor that is to to be used as a regularizer.
                Final atom will be constructed as `atom_constructor(**atom_params)`.

            atom_params : dict
                Dictionary of arguments to construct atom.

            offset : bool
                While `y` include an offset column when fitting?

            case_weights : bool
                Will `y` include a case weight column when fitting?

            coef : float
                Scaling to be put in front of loss.

            score_method : str
                Which score to use as default `score`?
                One of ['deviance', 'mean_deviance', 'R2']

            Notes
            -----

            If `case_weights` or `offset` is True, then the `fit`
            method should pass a 2-d array for `y` rather than just a
            response. If only one of these is not None, then the
            quantity is assumed to be the last column.  If both are
            not None, then the 2nd to last are to be the case weights
            and the last is to be the offset.

            The `offset` argument, if not None
            is *subtracted* from the linear predictor
            before evaluating the loss -- this sign is different
            than behavior in e.g. `R` where it is *added* instead.

            """

            self.smoothing_parameter = smoothing_parameter
            sklearn_regression.__init__(self,
                                        atom_constructor,
                                        atom_params,
                                        case_weights=case_weights,
                                        coef=coef,
                                        offset=offset,
                                        score_method=score_method)

        def _loglike_factory(self, X, y):
            response, case_weights_, offset_ = self._check_y_arg(y)
            return glm.huber_svm(X, 
                                 response, 
                                 self.smoothing_parameter,
                                 case_weights=case_weights_,
                                 coef=self.coef,
                                 saturated_offset=offset_)

        def _saturated_score(self,
                             predictions,
                             response,
                             case_weights=None):

            loss = lambda yhat: huber_svm(response.shape,
                                          response,
                                          self.smoothing_parameter,
                                          case_weights=case_weights).smooth_objective(yhat, 'func')

            # negative sign is to align with sklearn's maximizing a score with grid search

            if self.score_method == 'deviance':
                return -loss(predictions)
            elif self.score_method == 'mean_deviance':
                return -loss(predictions) / predictions.shape[0]
            elif self.score_method == 'R2':
                SSE = loss(predictions)
                SST = loss(response.mean() * np.ones_like(response)) # X: right for huber_svm?
                return 1 - SSE / SST
            else:
                return np.nan

    class sklearn_huber_svm_lagrange(sklearn_regression_lagrange, sklearn_huber_svm):

        def __init__(self, 
                     smoothing_parameter,
                     atom_constructor, 
                     atom_params, 
                     case_weights=False,
                     offset=False,
                     enet_alpha=0.,
                     coef=1., 
                     score_method='deviance'):

            """

            Parameters
            ----------

            smoothing_parameter : float
                Smoothing parameter for Huber SVM loss.

            atom_constructor : callable
                Atom constructor that is to to be used as a regularizer.
                Final atom will be constructed as `atom_constructor(**atom_params)`.

            atom_params : dict
                Dictionary of arguments to construct atom.

            offset : bool
                While `y` include an offset column when fitting?

            case_weights : bool
                Will `y` include a case weight column when fitting?

            enet_alpha : float
                Weighting in [0, 1] between ridge and regularizer.

            coef : float
                Scaling to be put in front of loss.

            score_method : str
                Which score to use as default `score`?
                One of ['deviance', 'mean_deviance', 'R2']

            Notes
            -----

            If `case_weights` or `offset` is True, then the `fit`
            method should pass a 2-d array for `y` rather than just a
            response. If only one of these is not None, then the
            quantity is assumed to be the last column.  If both are
            not None, then the 2nd to last are to be the case weights
            and the last is to be the offset.

            The `offset` argument, if not None
            is *subtracted* from the linear predictor
            before evaluating the loss -- this sign is different
            than behavior in e.g. `R` where it is *added* instead.

            """

            self.smoothing_parameter = smoothing_parameter
            sklearn_regression_lagrange.__init__(self,
                                                 atom_constructor,
                                                 atom_params,
                                                 enet_alpha=enet_alpha,
                                                 case_weights=case_weights,
                                                 coef=coef,
                                                 offset=offset,
                                                 score_method=score_method)

