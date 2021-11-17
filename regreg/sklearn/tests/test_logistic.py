import numpy as np
try:
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from ..api import (logistic,
                       logistic_lagrange,
                       logistic_classifier,
                       logistic_classifier_lagrange)
    have_sklearn = True
except ImportError:
    have_sklearn = False

from ...api import l1norm
from ...tests.decorators import set_seed_for_test

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_logistic():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)
    y = y > 0

    atom = l1norm
    atom_args = {'shape':p, 'bound':3}

    logistic_lasso = logistic(atom,
                              atom_args)

    logistic_lasso.fit(X, y)
    print(cross_validate(logistic_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'bound': b} for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(logistic_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    logistic_lasso_offset = logistic(atom,
                                     atom_args,
                                     offset=True)
    y_offset = np.array([y, np.random.standard_normal(n)]).T
    logistic_lasso_offset.fit(X, y_offset)

    logistic_lasso_weights = logistic(atom,
                                      atom_args,
                                      case_weights=True,
                                      score_method='accuracy')

    y_weights = np.array([y, np.ones_like(y)]).T
    logistic_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(logistic_lasso_weights._coefs,
                               logistic_lasso._coefs)

    GridSearchCV(logistic_lasso_offset, params, cv=3).fit(X, y_offset)
    GridSearchCV(logistic_lasso_weights, params, cv=3).fit(X, y_weights)

    logistic_lasso_both = logistic(atom,
                                   atom_args,
                                   offset=True,
                                   case_weights=True,
                                   score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(logistic_lasso_both, params, cv=3).fit(X, y_both)

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_logistic_classifier():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)
    y = y > 0

    atom = l1norm
    atom_args = {'shape':p, 'bound':3}

    logistic_lasso = logistic_classifier(atom,
                                         atom_args)

    logistic_lasso.fit(X, y)
    print(cross_validate(logistic_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'bound': b} for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(logistic_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    logistic_lasso_weights = logistic_classifier(atom,
                                                 atom_args,
                                                 case_weights=True)

    y_weights = np.array([y, np.ones_like(y)]).T
    logistic_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(logistic_lasso_weights._coefs,
                               logistic_lasso._coefs)

    GridSearchCV(logistic_lasso_weights, params, cv=3).fit(X, y_weights)




