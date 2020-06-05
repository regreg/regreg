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
    from .huber import (sklearn_huber as huber,
                        sklearn_huber_lagrange as huber_lagrange)
    from .cox import (sklearn_cox as cox,
                      sklearn_cox_lagrange as cox_lagrange)

    from .logistic import (sklearn_logistic as logistic,
                           sklearn_logistic_lagrange as logistic_lagrange,
                           sklearn_logistic_classifier as logistic_classifier,
                           sklearn_logistic_classifier_lagrange as logistic_classifier_lagrange)
    from .multinomial import (sklearn_multinomial as multinomial,
                              sklearn_multinomial_lagrange as multinomial_lagrange,
                              sklearn_multinomial_classifier as multinomial_classifier,
                              sklearn_multinomial_classifier_lagrange as multinomial_classifier_lagrange)
    from .probit import (sklearn_probit as probit,
                         sklearn_probit_lagrange as probit_lagrange,
                         sklearn_probit_classifier as probit_classifier,
                         sklearn_probit_classifier_lagrange as probit_classifier_lagrange)
    from .cloglog import (sklearn_cloglog as cloglog,
                          sklearn_cloglog_lagrange as cloglog_lagrange,
                          sklearn_cloglog_classifier as cloglog_classifier,
                          sklearn_cloglog_classifier_lagrange as cloglog_classifier_lagrange)
    from .huber_svm import (sklearn_huber_svm as huber_svm,
                            sklearn_huber_svm_lagrange as huber_svm_lagrange)
    
