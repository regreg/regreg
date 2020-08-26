import numpy as np
try:
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import GridSearchCV
    from ..api import (huber,
                       huber_lagrange)
    have_sklearn = True
except ImportError:
    have_sklearn = False

from ...api import l1norm
from ...tests.decorators import set_seed_for_test

@np.testing.dec.skipif(not have_sklearn)
@set_seed_for_test()
def test_regression_huber():

    n, p = 100, 20
    X = np.random.standard_normal((n, p))
    y = np.random.standard_normal(n)

    atom = l1norm
    atom_args = {'shape': p,
                 'lagrange': 2 * np.sqrt(n)}
    huber_lasso = huber(0.5, atom, atom_args)
    huber_lasso.fit(X, y)
    print(cross_validate(huber_lasso, X, y, cv=10))

    # grid search
    params = {'atom_params':[{'shape':p,
                              'lagrange': alpha * 2 * np.sqrt(n)} for alpha in [0.5, 1, 1.5]]}
    lasso_cv = GridSearchCV(huber_lasso, params, cv=3)
    lasso_cv.fit(X, y)

    huber_lasso_offset = huber(0.5, atom, atom_args,
                               offset=True)
    y_offset = np.array([y, np.random.standard_normal(n)]).T
    huber_lasso_offset.fit(X, y_offset)

    huber_lasso_weights = huber(0.5, atom, atom_args,
                                case_weights=True,
                                score_method='mean_deviance')
    
    y_weights = np.array([y, np.ones_like(y)]).T
    huber_lasso_weights.fit(X, y_weights)

    np.testing.assert_allclose(huber_lasso_weights._coefs,
                               huber_lasso._coefs)

    GridSearchCV(huber_lasso_offset, params, cv=3).fit(X, y_offset)
    GridSearchCV(huber_lasso_weights, params, cv=3).fit(X, y_weights)

    huber_lasso_both = huber(0.5, atom, atom_args,
                             offset=True,
                             case_weights=True,
                             score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(huber_lasso_both, params, cv=3).fit(X, y_both)

    def atom_constructor(null_grad, **args):
        return l1norm(**args)
    huber_lasso_enet = huber_lagrange(0.5,
                                      atom_constructor,
                                      atom_args,
                                      offset=True,
                                      enet_alpha=0.5,
                                      case_weights=True,
                                      score_method='R2')
    y_both = np.array([y, np.ones_like(y), y_offset[:,1]]).T
    GridSearchCV(huber_lasso_enet, params, cv=3).fit(X, y_both)



