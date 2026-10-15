---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# scikit-learn interface

`regreg.sklearn.api` wraps penalized GLM-type problems as scikit-learn estimators, so that
`GridSearchCV`, `cross_val_score`, `clone`, pipelines and so on can be used to tune and
assess them. Each estimator pairs a loss with an arbitrary `regreg` atom as the regularizer.

| family | estimators |
|---|---|
| Gaussian (least squares) | `gaussian`, `gaussian_lagrange` |
| Huber | `huber`, `huber_lagrange` |
| smoothed hinge | `huber_svm`, `huber_svm_lagrange` |
| logistic | `logistic`, `logistic_lagrange`, `logistic_classifier`, `logistic_classifier_lagrange` |
| probit | `probit`, `probit_lagrange`, `probit_classifier`, `probit_classifier_lagrange` |
| complementary log-log | `cloglog`, `cloglog_lagrange`, `cloglog_classifier`, `cloglog_classifier_lagrange` |
| multinomial | `multinomial`, `multinomial_lagrange`, `multinomial_classifier`, `multinomial_classifier_lagrange` |
| Cox | `cox`, `cox_lagrange` |

There is no Poisson estimator (although `regreg.paths` has Poisson paths).
The plain and `_lagrange` estimators are regressors: `predict` returns the linear
predictor $X\hat\beta$. The `_classifier` estimators return labels from `predict`
(and probabilities from `predict_proba`) and are scored by accuracy.

```{code-cell} ipython3
import numpy as np

import regreg.api as rr
from regreg.sklearn import api as rsk

from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import Lasso, LogisticRegression
```

## The constructor

An estimator is built from an atom constructor and a dictionary of its arguments.
For the plain estimators the penalty is `atom_constructor(**atom_params)`, so the atom can be
in bound or Lagrange form. The objective is

$$
\text{minimize}_\beta \; \ell(X\beta; y) + \mathcal{P}(\beta)
$$

where $\ell$ is the negative log-likelihood (for the Gaussian, $\frac12\|y-X\beta\|_2^2$; no
division by $n$) and $\mathcal{P}$ is the atom. There is no intercept: add a column of ones
to `X` if you want one (and, if it should be unpenalized, use an atom that leaves it
unpenalized, e.g. `rr.weighted_l1norm` with weight 0).

`solve_args` is passed to the solver. Its defaults are loose (coefficients are typically
accurate to about $10^{-2}$), which is fine for model selection; pass e.g.
`solve_args={'tol': 1e-12, 'min_its': 200}` when the coefficients themselves matter.

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, p = 150, 20
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:4] = [2, -1.5, 1, 0.5]
y = X @ beta + rng.standard_normal(n)

bound_lasso = rsk.gaussian(rr.l1norm, {'shape': p, 'bound': 4})   # ||beta||_1 <= 4
bound_lasso.fit(X, y)
print(np.round(bound_lasso._coefs[:6], 3), np.fabs(bound_lasso._coefs).sum())
```

Since `atom_params` is an ordinary hyperparameter, it is what `GridSearchCV` searches over:

```{code-cell} ipython3
cv = KFold(5, shuffle=True, random_state=0)
grid = {'atom_params': [{'shape': p, 'bound': b} for b in [1, 2, 3, 4, 5, 6]]}
gs_bound = GridSearchCV(rsk.gaussian(rr.l1norm, {'shape': p, 'bound': 1}), grid, cv=cv).fit(X, y)
print('best bound:', gs_bound.best_params_['atom_params']['bound'])
```

## Lagrange form: penalties relative to $\lambda_{\max}$

A Lagrange parameter is only meaningful relative to the scale of the data, and the scale
differs between training folds. The `_lagrange` estimators therefore build the atom as

```
atom_constructor(score, **atom_params)
```

where `score` is the gradient of the loss at the null solution ($\beta=0$, or the fit with
only the unpenalized variables when `unpenalized=True`). For the lasso,
$\lambda_{\max} = \|\text{score}\|_\infty$ is the smallest $\lambda$ giving $\hat\beta=0$, so
the natural tuning parameter is a fraction of it, recomputed on every training fold:

```{code-cell} ipython3
def lasso_fraction(score, frac=0.5):
    return rr.l1norm(score.shape, lagrange=frac * np.fabs(score).max())

lagrange_lasso = rsk.gaussian_lagrange(lasso_fraction, {'frac': 0.1},
                                       solve_args={'tol': 1e-12, 'min_its': 200})
lagrange_lasso.fit(X, y);
```

As a check, scikit-learn's `Lasso` minimizes $\frac{1}{2n}\|y-X\beta\|_2^2 + a\|\beta\|_1$,
so it should agree with $a = \lambda / n$:

```{code-cell} ipython3
lam = 0.1 * np.fabs(X.T @ y).max()
sk_lasso = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, y)
print('max |regreg - sklearn| = %.1e' % np.fabs(lagrange_lasso._coefs - sk_lasso.coef_).max())
np.testing.assert_allclose(lagrange_lasso._coefs, sk_lasso.coef_, atol=1e-6)
```

Now tune the fraction by 5-fold CV. The default score is the negated deviance (larger is
better, as `GridSearchCV` maximizes); `score_method` selects others:

* `'deviance'`: $-2\times$ negative log-likelihood on the test fold (the default for regressors);
* `'mean_deviance'`: the same divided by the number of test cases;
* `'R2'`: $1 - D(\hat\beta)/D_0$, with $D_0$ the deviance of the null (intercept-only) model;
* `'accuracy'`: for the binary families, the (weighted) proportion of correctly predicted
  labels (the default, and only option, for the classifiers).

```{code-cell} ipython3
fracs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4]
est = rsk.gaussian_lagrange(lasso_fraction, {'frac': 0.5}, score_method='mean_deviance')
gs = GridSearchCV(est, {'atom_params': [{'frac': f} for f in fracs]}, cv=cv).fit(X, y)
for f, s in zip(fracs, gs.cv_results_['mean_test_score']):
    print('frac = %.2f   CV mean deviance = %.3f' % (f, -s))
print('selected:', gs.best_params_)
print('nonzero coefficients of the refit:', np.nonzero(gs.best_estimator_._coefs)[0])
```

### Elastic net and extra quadratic terms

For the Lagrange estimators, `enet_alpha` $\in[0,1]$ splits the penalty between the atom and
a ridge term: the atom's Lagrange parameter $\lambda$ becomes $(1-\texttt{enet\_alpha})\lambda$
and $\frac{\texttt{enet\_alpha}\,\lambda}{2}\|\beta\|_2^2$ is added. (Note that
`enet_alpha=0` is the pure lasso, the opposite convention from `alpha` in `regreg.paths`.)
Every estimator also accepts `quadratic`, an `rr.identity_quadratic` added to the objective;
in Lagrange form its gradient at the null solution is included in `score`.

```{code-cell} ipython3
enet = rsk.gaussian_lagrange(lasso_fraction, {'frac': 0.1}, enet_alpha=0.5,
                             solve_args={'tol': 1e-12, 'min_its': 200}).fit(X, y)
# the same problem written out by hand
loss = rr.squared_error(X, y)
direct = rr.simple_problem(loss, rr.l1norm(p, lagrange=0.5 * lam)).solve(
    rr.identity_quadratic(0.5 * lam, 0, 0, 0), tol=1e-12, min_its=200)
print('max |enet - direct| = %.1e' % np.fabs(enet._coefs - direct).max())
np.testing.assert_allclose(enet._coefs, direct, atol=1e-6)
```

## Case weights, offsets and `sample_weight`

scikit-learn passes only `X` and `y` through its splitters, so case weights and offsets are
carried as extra columns of `y`. With `case_weights=True` the last column of `y` holds the
weights; with `offset=True` the last column holds the offset; with both, the weights are
second to last. The offset is *subtracted* from the linear predictor (the opposite of R's
convention). Because the extra columns travel with `y`, cross-validation subsets them along
with the response.

`score(X, y, sample_weight=w)` uses `w` as case weights for scoring (multiplied into the
weight column if there is one).

```{code-cell} ipython3
w = np.random.default_rng(1).uniform(0.5, 2, n)
weighted = rsk.gaussian(rr.l1norm, {'shape': p, 'lagrange': 5.}, case_weights=True,
                       solve_args={'tol': 1e-12, 'min_its': 200})
weighted.fit(X, np.column_stack([y, w]))

s1 = weighted.score(X, np.column_stack([y, w]))
s2 = weighted.score(X, np.column_stack([y, np.ones(n)]), sample_weight=w)
print(s1, s2)
np.testing.assert_allclose(s1, s2)

# the same weighted fit by hand
direct = rr.simple_problem(rr.glm.gaussian(X, y, case_weights=w),
                           rr.l1norm(p, lagrange=5.)).solve(tol=1e-12, min_its=200)
print('max |weighted - direct| = %.1e' % np.fabs(weighted._coefs - direct).max())
np.testing.assert_allclose(weighted._coefs, direct, atol=1e-4)
```

## Classification: comparison with `LogisticRegression`

The `_classifier` estimators predict labels, so `cross_val_score` uses stratified folds and
the accuracy score. Here an $\ell_1$-penalized logistic classifier with
$\lambda=5$ is compared with scikit-learn's `LogisticRegression` with `l1_ratio=1` and
$C = 1/\lambda$ (scikit-learn minimizes $C\sum_i \text{logloss}_i + \|\beta\|_1$) on the same
folds.

```{code-cell} ipython3
rng = np.random.default_rng(2)
eta = X[:, :3] @ np.array([1.5, -1, 1])
yb = rng.uniform(size=n) < 1 / (1 + np.exp(-eta))

lam_logit = 5.
l1_atom = lambda score, lagrange: rr.l1norm(score.shape, lagrange=lagrange)
clf = rsk.logistic_classifier_lagrange(l1_atom, {'lagrange': lam_logit},
                                       solve_args={'tol': 1e-12, 'min_its': 200})
sk_clf = LogisticRegression(C=1 / lam_logit, l1_ratio=1.0, solver='saga',
                            fit_intercept=False, tol=1e-12, max_iter=100000)

folds = StratifiedKFold(5, shuffle=True, random_state=0)
acc_rr = cross_val_score(clf, X, yb, cv=folds)
acc_sk = cross_val_score(sk_clf, X, yb, cv=folds)
print('regreg  accuracy by fold:', np.round(acc_rr, 3))
print('sklearn accuracy by fold:', np.round(acc_sk, 3))
np.testing.assert_allclose(acc_rr, acc_sk)

clf.fit(X, yb)
sk_clf.fit(X, yb)
print('max |coef difference| = %.1e' % np.fabs(clf._coefs - sk_clf.coef_[0]).max())
np.testing.assert_allclose(clf._coefs, sk_clf.coef_[0], atol=1e-6)
print('predicted probabilities:', np.round(clf.predict_proba(X[:4]), 3))
```

Here the Lagrange form is used with a fixed $\lambda$ (the constructor ignores `score`)
because only the `_lagrange` classifiers accept `solve_args`; the plain `logistic_classifier`
uses the solver's default tolerance, which agrees with scikit-learn only to about $10^{-2}$
([gh-104](https://github.com/regreg/regreg/issues/104)).

The same accuracy can be had from the regression estimator `logistic` with
`score_method='accuracy'`, whose `predict` returns the linear predictor instead of labels.

## A group lasso through the Gaussian estimator

Any atom works. Here the 20 features form five groups of four and `rr.group_lasso` is the
penalty; its Lagrange parameter is tuned by CV.

```{code-cell} ipython3
groups = np.repeat(np.arange(5), 4)
glasso = rsk.gaussian(rr.group_lasso, {'groups': groups, 'lagrange': 20.},
                      solve_args={'tol': 1e-12, 'min_its': 200})
grid = {'atom_params': [{'groups': groups, 'lagrange': l} for l in [5, 10, 20, 50, 100]]}
gs_group = GridSearchCV(glasso, grid, cv=cv).fit(X, y)
best = gs_group.best_estimator_
print('best lagrange:', gs_group.best_params_['atom_params']['lagrange'])
print('group norms:', np.round([np.linalg.norm(best._coefs[groups == g]) for g in range(5)], 3))
```

The first group, which holds all the signal, has by far the largest norm. As a check, the
estimator's coefficients solve the KKT conditions of the group lasso: for active groups
$-X_g^T(y - X\hat\beta) = -\lambda w_g \hat\beta_g / \|\hat\beta_g\|_2$, and for inactive
groups $\|X_g^T(y-X\hat\beta)\|_2 \le \lambda w_g$, with default weights $w_g=\sqrt{|g|}$.

```{code-cell} ipython3
lam_g = gs_group.best_params_['atom_params']['lagrange']
grad = -X.T @ (y - X @ best._coefs)
for g in range(5):
    bg, gg = best._coefs[groups == g], grad[groups == g]
    if np.linalg.norm(bg) > 0:
        resid = np.fabs(gg + lam_g * 2 * bg / np.linalg.norm(bg)).max()
        print('group %d active,   KKT residual %.1e' % (g, resid))
        assert resid < 1e-5
    else:
        print('group %d inactive, ||grad|| / (lambda w) = %.3f' % (g, np.linalg.norm(gg) / (lam_g * 2)))
        assert np.linalg.norm(gg) <= lam_g * 2 * (1 + 1e-6)
```

## `clone`, `get_params` and `set_params`

The estimators are ordinary scikit-learn estimators: all constructor arguments are
hyperparameters, so `clone` copies an unfitted estimator and `set_params` changes any of
them (this is what `GridSearchCV` does internally).

```{code-cell} ipython3
est = rsk.gaussian_lagrange(lasso_fraction, {'frac': 0.5}, score_method='R2')
print(sorted(est.get_params()))

est2 = clone(est).set_params(atom_params={'frac': 0.05}, enet_alpha=0.25)
print(est2)
assert est.get_params()['atom_params'] == {'frac': 0.5}       # the original is unchanged
assert est2.fit(X, y).score(X, y) > 0.5                        # R2 on the training data
```
