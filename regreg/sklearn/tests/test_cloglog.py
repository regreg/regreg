import pytest
import numpy as np
try:
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from ..api import (cloglog,
                       cloglog_lagrange,
                       cloglog_classifier,
                       cloglog_classifier_lagrange) 
    have_sklearn = True
except ImportError:
    have_sklearn = False

from ...api import l1norm

@pytest.mark.skipif(not have_sklearn, reason="scikit-learn not available")
def test_cloglog():

    rng = np.random.default_rng(5500)
    n, p = 100, 20
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    y = y > 0

    atom = l1norm
    atom_args = {'shape':p, 'bound':3}

    cloglog_lasso = cloglog(atom,
                            atom_args)

    cloglog_lasso.fit(X, y)
    print(cross_validate(cloglog_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'bound': b} for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(cloglog_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    cloglog_lasso_offset = cloglog(atom,
                                   atom_args,
                                   offset=True)
    y_offset = np.array([y, rng.standard_normal(n)]).T
    cloglog_lasso_offset.fit(X, y_offset)

    cloglog_lasso_weights = cloglog(atom,
                                    atom_args,
                                    case_weights=True,
                                    score_method='accuracy')

    y_weights = np.array([y, np.ones_like(y)]).T
    cloglog_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(cloglog_lasso_weights._coefs,
                               cloglog_lasso._coefs)

    GridSearchCV(cloglog_lasso_offset, params, cv=3).fit(X, y_offset)
    GridSearchCV(cloglog_lasso_weights, params, cv=3).fit(X, y_weights)

    cloglog_lasso_both = cloglog(atom,
                                 atom_args,
                                 offset=True,
                                 case_weights=True,
                                 score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(cloglog_lasso_both, params, cv=3).fit(X, y_both)

@pytest.mark.skipif(not have_sklearn, reason="scikit-learn not available")
def test_cloglog_classifier():

    rng = np.random.default_rng(5501)
    n, p = 100, 20
    X = rng.standard_normal((n, p))
    y = rng.standard_normal(n)
    y = y > 0

    atom = l1norm
    atom_args = {'shape':p, 'bound':3}

    cloglog_lasso = cloglog_classifier(atom,
                                       atom_args)

    cloglog_lasso.fit(X, y)
    print(cross_validate(cloglog_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'bound': b} for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(cloglog_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    cloglog_lasso_weights = cloglog_classifier(atom,
                                               atom_args,
                                               case_weights=True)

    y_weights = np.array([y, np.ones_like(y)]).T
    cloglog_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(cloglog_lasso_weights._coefs,
                               cloglog_lasso._coefs)

    GridSearchCV(cloglog_lasso_weights, params, cv=3).fit(X, y_weights)




