have_sklearn = True
try:
    from sklearn.base import (BaseEstimator, 
                              RegressorMixin,
                              ClassifierMixin)
except ImportError:
    have_sklearn = False

if have_sklearn:

    from .gaussian import (sklearn_gaussian as gaussian,
                           sklearn_gaussian_lagrange as gaussian_lagrange)
    from .logistic import (sklearn_logistic as logistic,
                           sklearn_logistic_lagrange as logistic_lagrange)
    from .huber import (sklearn_huber as huber,
                        sklearn_huber_lagrange as huber_lagrange)
    from .cox import (sklearn_cox as cox,
                      sklearn_cox_lagrange as cox_lagrange)

