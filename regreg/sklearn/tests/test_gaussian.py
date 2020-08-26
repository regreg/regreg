import numpy as np
try:
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from ..api import (gaussian,
                       gaussian_lagrange)
    have_sklearn = True
except ImportError:
    have_sklearn = False

from ...api import l1norm, group_lasso
from ...tests.decorators import set_seed_for_test

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_regression_gaussian():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)

    atom = l1norm
    atom_args = {'shape':p, 'lagrange':2*np.sqrt(n)}

    gaussian_lasso = gaussian(atom, atom_args)
    gaussian_lasso.fit(X, y)
    print(cross_validate(gaussian_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'lagrange': alpha * 2 * np.sqrt(n)} for alpha in [0.5, 1, 1.5]]}
    lasso_cv = GridSearchCV(gaussian_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    gaussian_lasso_offset = gaussian(atom, atom_args,
                                             offset=True)
    y_offset = np.array([y, np.random.standard_normal(n)]).T
    gaussian_lasso_offset.fit(X, y_offset)

    gaussian_lasso_weights = gaussian(atom, atom_args,
                                      case_weights=True,
                                      score_method='mean_deviance')

    y_weights = np.array([y, np.ones_like(y)]).T
    gaussian_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(gaussian_lasso_weights._coefs,
                               gaussian_lasso._coefs)

    GridSearchCV(gaussian_lasso_offset, params, cv=3).fit(X, y_offset)
    GridSearchCV(gaussian_lasso_weights, params, cv=3).fit(X, y_weights)

    gaussian_lasso_both = gaussian(atom, atom_args,
                                   offset=True,
                                   case_weights=True,
                                   score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(gaussian_lasso_both, params, cv=3).fit(X, y_both)

    def atom_constructor(null_grad, **args):
        return l1norm(**args)
    gaussian_lasso_enet = gaussian_lagrange(atom_constructor,
                                            atom_args,
                                            offset=True,
                                            enet_alpha=0.5,
                                            case_weights=True,
                                            score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(gaussian_lasso_enet, params, cv=3).fit(X, y_both)

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_regression_group_lasso():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)

    atom = group_lasso
    atom_args = {'groups':[0]*10+[1]*10, 'lagrange':2*np.sqrt(n)}

    gaussian_lasso = gaussian(atom, atom_args)
    gaussian_lasso.fit(X, y)
    print(cross_validate(gaussian_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'groups':[0]*10+[1]*10,
                              'lagrange': alpha * 2 * np.sqrt(n)} for alpha in [0.5, 1, 1.5]]}
    lasso_cv = GridSearchCV(gaussian_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    gaussian_lasso_offset = gaussian(atom, atom_args,
                                             offset=True)
    y_offset = np.array([y, np.random.standard_normal(n)]).T
    gaussian_lasso_offset.fit(X, y_offset)

    gaussian_lasso_weights = gaussian(atom, atom_args,
                                      case_weights=True,
                                      score_method='mean_deviance')

    y_weights = np.array([y, np.ones_like(y)]).T
    gaussian_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(gaussian_lasso_weights._coefs,
                               gaussian_lasso._coefs)

    GridSearchCV(gaussian_lasso_offset, params, cv=3).fit(X, y_offset)
    GridSearchCV(gaussian_lasso_weights, params, cv=3).fit(X, y_weights)

    gaussian_lasso_both = gaussian(atom, atom_args,
                                   offset=True,
                                   case_weights=True,
                                   score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(gaussian_lasso_both, params, cv=3).fit(X, y_both)

    def atom_constructor(null_grad, **args):
        return group_lasso(**args)
    gaussian_lasso_enet = gaussian_lagrange(atom_constructor,
                                            atom_args,
                                            offset=True,
                                            enet_alpha=0.5,
                                            case_weights=True,
                                            score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(gaussian_lasso_enet, params, cv=3).fit(X, y_both)

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_regression_gaussian_bound():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)

    atom = l1norm
    atom_args = {'shape':p, 'bound':3}

    gaussian_lasso = gaussian(atom,
                              atom_args)

    gaussian_lasso.fit(X, y)
    print(cross_validate(gaussian_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'bound': b,
                              } for b in [3, 4, 5]]}
    lasso_cv = GridSearchCV(gaussian_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    gaussian_lasso_offset = gaussian(atom,
                                     atom_args,
                                     offset=True)
    y_offset = np.array([y, np.random.standard_normal(n)]).T
    gaussian_lasso_offset.fit(X, y_offset)

    gaussian_lasso_weights = gaussian(atom,
                                      atom_args,
                                      case_weights=True,
                                      score_method='mean_deviance')

    y_weights = np.array([y, np.ones_like(y)]).T
    gaussian_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(gaussian_lasso_weights._coefs,
                               gaussian_lasso._coefs)

    GridSearchCV(gaussian_lasso_offset, params, cv=3).fit(X, y_offset)
    GridSearchCV(gaussian_lasso_weights, params, cv=3).fit(X, y_weights)

    gaussian_lasso_both = gaussian(atom,
                                   atom_args,
                                   offset=True,
                                   case_weights=True,
                                   score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(gaussian_lasso_both, params, cv=3).fit(X, y_both)




