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

# Regularization paths

The `regreg.paths` subpackage computes solutions of penalized problems of the form

$$
\hat\beta(\lambda) = \text{argmin}_{\beta} \; \ell(X\beta) + \lambda \alpha \, P(\beta)
 + \frac{\lambda(1-\alpha)}{2} \sum_j c_j \beta_j^2
$$

over a decreasing sequence $\lambda_1 > \lambda_2 > \dots > \lambda_K$. Here $\ell$ is a
loss of the linear predictor (Gaussian, logistic, probit, cloglog, Poisson, Cox, Huber, ...),
$P$ is a (weighted) $\ell_1$, group lasso or sparse group lasso penalty, $\alpha \in (0,1]$
is the elastic net mixing parameter and $c_j$ are the ridge weights
(`elastic_net_param`). With $\alpha=1$ (the default) this is the lasso, group lasso, etc.

A path computation has three ingredients:

1. a **path object**, which describes the loss, design and penalty. These live in
   `regreg.paths.lasso`, `group_lasso`, `sparse_group_lasso`, `sparse_group_block` and
   `mixed_lasso`, and are built by module-level constructors such as
   `lasso.gaussian(X, Y, weights)` or `group_lasso.logistic(X, Y, groups, weights=...)`;
2. a **sequence of Lagrange parameters**, usually from `default_lagrange_sequence`;
3. a **driver**: `strong_rules` or `warm_start` (in `regreg.paths`) or `basil`
   (in `regreg.paths.basil`).

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

import regreg.api as rr
from regreg.paths import (lasso,
                          group_lasso,
                          strong_rules,
                          warm_start,
                          default_lagrange_sequence)
from regreg.paths.basil import basil
from regreg.paths.cross_validate import cross_validate
```

Note the import locations: `regreg.paths.basil` and `regreg.paths.cross_validate` are
submodules, so the functions must be imported from them.

## A lasso path

We simulate a sparse linear model with five nonzero coefficients.

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, p = 200, 30
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:5] = [3, -2, 1.5, 1, -0.5]
Y = X @ beta + rng.standard_normal(n)

path = lasso.gaussian(X, Y, np.ones(p))   # weights: one l1 weight per feature
```

The path object stores the solution at $\lambda = \infty$ (the "null" solution, here
$\beta=0$) as `path.solution`, and the gradient of the loss there as
`path.grad_solution`. The smallest $\lambda$ for which the null solution is optimal,
$\lambda_{\max}$, is the dual norm of this gradient; for the lasso with squared error loss
$\frac{1}{2}\|Y-X\beta\|_2^2$ it is $\|X^TY\|_\infty$.
`default_lagrange_sequence` returns `nstep` log-spaced values from $\lambda_{\max}$ down to
`lagrange_proportion` $\times \lambda_{\max}$.

```{code-cell} ipython3
lagrange = default_lagrange_sequence(path.penalty,
                                     path.grad_solution,
                                     lagrange_proportion=0.01,
                                     nstep=30)
np.testing.assert_allclose(lagrange[0], np.fabs(X.T @ Y).max())
lagrange[:3], lagrange[-1] / lagrange[0]
```

Every driver takes the path object, the sequence and the initial `(solution, gradient)`
pair at `lagrange[0]`. `strong_rules` returns a dictionary; its `'beta'` entry has one
row per value of $\lambda$.

```{code-cell} ipython3
initial = (path.solution.copy(), path.grad_solution.copy())
sr = strong_rules(path, lagrange, initial, inner_tol=1e-10)
B_sr = sr['beta']
B_sr.shape
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4.5))
loglam = np.log(lagrange)
for j in range(5, p):
    ax.plot(loglam, B_sr[:, j], color='0.7', lw=1)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
for j in range(5):
    ax.plot(loglam, B_sr[:, j], color=colors[j], lw=2, label=r'$\beta_{%d}=%g$' % (j + 1, beta[j]))
ax.axhline(0, color='k', lw=0.5)
ax.invert_xaxis()
ax.set_xlabel(r'$\log \lambda$')
ax.set_ylabel('coefficient')
ax.set_title('Lasso path (true signals in color, nulls in gray)')
ax.legend(frameon=False, fontsize=9);
```

## Three drivers, one path

* `warm_start` solves the full problem at each $\lambda$, starting from the previous
  solution. It is the simplest and the reference implementation.
* `strong_rules` solves each problem on a small working set chosen by the sequential strong
  rule of Tibshirani et al. (2012), then checks the KKT conditions on all variables and
  enlarges the working set until they hold.
* `basil` (batch screening iterative lasso) adds the `step_nvar` largest KKT violators
  at a time and solves `step_lagrange` values of $\lambda$ per batch.

All three should give the same path. We also compare with a direct solve of each problem
using `rr.simple_problem`.

```{code-cell} ipython3
path = lasso.gaussian(X, Y, np.ones(p))
B_ws = warm_start(path, lagrange, initial, inner_tol=1e-10)['beta']

path = lasso.gaussian(X, Y, np.ones(p))
B_basil = basil(path, lagrange, initial, inner_tol=1e-10, step_nvar=10, step_lagrange=5)

loss = rr.squared_error(X, Y)
B_direct = np.array([rr.simple_problem(loss, rr.l1norm(p, lagrange=lam)).solve(tol=1e-13, min_its=500)
                     for lam in lagrange])

for name, B in [('strong_rules', B_sr), ('warm_start', B_ws), ('basil', B_basil)]:
    print('%-13s max |beta - direct| = %.2e' % (name, np.fabs(B - B_direct).max()))
print('max |strong_rules - warm_start| = %.2e' % np.fabs(B_sr - B_ws).max())
print('max |basil - warm_start|        = %.2e' % np.fabs(B_basil - B_ws).max())
```

Before a step is accepted, the drivers check the KKT conditions on all variables. An
inactive variable fails the check when $|\nabla_j \ell(X\hat\beta)| > (1 +
\epsilon)\,\lambda$, and then joins the working set. The tolerance $\epsilon$ is
`kkt_tol`, with default `1e-6`. It is an attribute of the path object (`path.kkt_tol`), and the drivers
also take it as an argument. A looser value saves little time and can leave a variable
that is just about to enter at zero for a step. Earlier versions used 1%, which
gave errors of order $10^{-3}$ near entry points
([gh-99](https://github.com/regreg/regreg/issues/99)). The KKT checks of the active
variables (`check_active=True`) use their own, looser, tolerance, as they depend on how
accurately each subproblem is solved.

`strong_rules` agrees with the direct solves to the solver tolerance at every
$\lambda$. The $10^{-5}$-level errors of `warm_start` and `basil` come from the
stopping rule of their solves, which is not as strict as `inner_tol` suggests.

```{code-cell} ipython3
print(np.array2string(np.fabs(B_sr - B_direct).max(1), precision=1))
```

```{code-cell} ipython3
np.testing.assert_allclose(B_sr, B_direct, atol=1e-6)
np.testing.assert_allclose(B_basil, B_direct, atol=5e-4)
np.testing.assert_allclose(B_ws, B_direct, atol=5e-4)
```

## Elastic net

Set `alpha < 1` for the elastic net. Pass the same `alpha` to `default_lagrange_sequence`,
which divides the sequence by $\alpha$ so that it still starts at the first $\lambda$
where a variable enters.

```{code-cell} ipython3
enet = lasso.gaussian(X, Y, np.ones(p), alpha=0.5)
lagrange_enet = default_lagrange_sequence(enet.penalty, enet.grad_solution,
                                          lagrange_proportion=0.05, nstep=15, alpha=enet.alpha)
init_enet = (enet.solution.copy(), enet.grad_solution.copy())
B_enet = strong_rules(enet, lagrange_enet, init_enet, inner_tol=1e-10)['beta']

# direct solve at the smallest lambda: 0.5 * ||Y - X b||^2 + lam * (0.5 ||b||_1 + 0.25 ||b||_2^2)
lam = lagrange_enet[-1]
ridge = rr.identity_quadratic(lam * 0.5, 0, 0, 0)
direct = rr.simple_problem(loss, rr.l1norm(p, lagrange=0.5 * lam)).solve(ridge, tol=1e-13, min_its=500)
print('elastic net, max |path - direct| = %.2e' % np.fabs(B_enet[-1] - direct).max())
np.testing.assert_allclose(B_enet[-1], direct, atol=1e-6)
```

## Unpenalized variables: a group lasso path

Variables with penalty weight 0 are unpenalized. They are fitted in the null solution
(by solving the problem with only those variables) and kept in every subproblem. Here the
first group of three variables is unpenalized; the other five groups get the usual weight
$\sqrt{|g|}$.

```{code-cell} ipython3
rng = np.random.default_rng(1)
n = 150
groups = np.repeat(np.arange(6), 3)
p = groups.shape[0]
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:3] = 1
beta[3:6] = [2, -1, 0.5]
Y = X @ beta + rng.standard_normal(n)

weights = {g: np.sqrt(3) for g in range(1, 6)}
weights[0] = 0.

# with unpenalized variables, give them ridge weight 0 (otherwise a warning is issued)
enet_param = (groups != 0).astype(float)
gpath = group_lasso.gaussian(X, Y, groups, weights=weights, elastic_net_param=enet_param)
```

The null solution is the least squares fit on the unpenalized group:

```{code-cell} ipython3
ols = np.linalg.lstsq(X[:, :3], Y, rcond=None)[0]
print(gpath.solution[:3], ols)
np.testing.assert_allclose(gpath.solution[:3], ols, atol=1e-3)
```

```{code-cell} ipython3
lagrange_g = default_lagrange_sequence(gpath.penalty, gpath.grad_solution,
                                       lagrange_proportion=0.02, nstep=20)
B_g = strong_rules(gpath, lagrange_g, (gpath.solution.copy(), gpath.grad_solution.copy()),
                   inner_tol=1e-10)['beta']

# which groups are active at each lambda?
active = np.array([[np.any(b[groups == g] != 0) for g in range(6)] for b in B_g])
print(active.astype(int)[::4])

pen = rr.group_lasso(groups, weights=weights, lagrange=lagrange_g[-1])
direct = rr.simple_problem(rr.squared_error(X, Y), pen).solve(tol=1e-13, min_its=500)
print('group lasso, max |path - direct| = %.2e' % np.fabs(B_g[-1] - direct).max())
np.testing.assert_allclose(B_g[-1], direct, atol=1e-6)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4.5))
loglam = np.log(lagrange_g)
for j in range(p):
    g = groups[j]
    style = dict(color='#1f77b4', lw=2, ls='--') if g == 0 else \
            dict(color='#d62728', lw=2) if g == 1 else dict(color='0.7', lw=1)
    ax.plot(loglam, B_g[:, j], **style)
ax.plot([], [], color='#1f77b4', ls='--', lw=2, label='group 0 (unpenalized)')
ax.plot([], [], color='#d62728', lw=2, label='group 1 (signal)')
ax.plot([], [], color='0.7', lw=1, label='groups 2-5 (null)')
ax.invert_xaxis()
ax.set_xlabel(r'$\log \lambda$')
ax.set_ylabel('coefficient')
ax.set_title('Group lasso path with an unpenalized group')
ax.legend(frameon=False, fontsize=9);
```

## A logistic lasso path

The other losses work the same way. To include an intercept, add a column of ones and
give it weight 0.

```{code-cell} ipython3
rng = np.random.default_rng(2)
n, p = 200, 20
X = rng.standard_normal((n, p))
eta = X[:, :3] @ np.array([1.5, -1, 1])
Yb = (rng.uniform(size=n) < 1 / (1 + np.exp(-eta))).astype(float)

X1 = np.column_stack([np.ones(n), X])
w = np.ones(p + 1)
w[0] = 0
enet_param = (w > 0).astype(float)
lpath = lasso.logistic(X1, Yb, w, elastic_net_param=enet_param)

# the null solution is the intercept-only fit: logit of the success proportion
print(lpath.solution[0], np.log(Yb.mean() / (1 - Yb.mean())))
np.testing.assert_allclose(lpath.solution[0], np.log(Yb.mean() / (1 - Yb.mean())), atol=1e-4)
```

```{code-cell} ipython3
lagrange_l = default_lagrange_sequence(lpath.penalty, lpath.grad_solution,
                                       lagrange_proportion=0.02, nstep=20)
res = strong_rules(lpath, lagrange_l, (lpath.solution.copy(), lpath.grad_solution.copy()),
                   inner_tol=1e-10)
B_l = res['beta']
print('fraction of deviance explained at smallest lambda: %.3f' % res['dev explained'])

pen = rr.weighted_l1norm(w, lagrange=lagrange_l[-1])
direct = rr.simple_problem(rr.glm.logistic(X1, Yb), pen).solve(tol=1e-13, min_its=500)
print('logistic, max |path - direct| = %.2e' % np.fabs(B_l[-1] - direct).max())
np.testing.assert_allclose(B_l[-1], direct, atol=1e-6)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4.5))
loglam = np.log(lagrange_l)
for j in range(4, p + 1):
    ax.plot(loglam, B_l[:, j], color='0.7', lw=1)
for j, c in zip(range(1, 4), colors):
    ax.plot(loglam, B_l[:, j], color=c, lw=2, label=r'$\beta_{%d}$' % j)
ax.invert_xaxis()
ax.set_xlabel(r'$\log \lambda$')
ax.set_ylabel('coefficient')
ax.set_title('Logistic lasso path (intercept not shown)')
ax.legend(frameon=False, fontsize=9);
```

## Cross-validation

`cross_validate(path_obj, lagrange_seq, ...)` fits the path on each training fold (by
default with `strong_rules`; change it with `path_fitter`) and scores the linear
predictor on the held-out fold. The default score is the mean squared error; pass
`score=f(linpred, y)` for anything else. It returns the mean and standard deviation of the
score over folds, for each $\lambda$.

The folds can be given as

* an integer `cv`: random folds drawn from `rng` (a seed or `np.random.Generator`);
* any scikit-learn splitter, e.g. `KFold(5, shuffle=True, random_state=0)`;
* an integer `cv` together with `group_labels`, which uses `GroupKFold` so that no group
  of cases is split across folds.

The training loss is rescaled by $n / n_{\text{train}}$ so that the same $\lambda$ sequence
is comparable across folds and with the full-data fit.

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, p = 200, 30
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:5] = [3, -2, 1.5, 1, -0.5]
Y = X @ beta + rng.standard_normal(n)

path = lasso.gaussian(X, Y, np.ones(p))
lagrange = default_lagrange_sequence(path.penalty, path.grad_solution,
                                     lagrange_proportion=0.005, nstep=30)
cv_mean, cv_sd = cross_validate(path, lagrange, cv=5, rng=np.random.default_rng(3),
                                n_jobs=1, verbose=False)
imin = np.argmin(cv_mean)
print('lambda minimizing CV error: %.2f (lambda_max = %.1f), CV MSE %.3f' %
      (lagrange[imin], lagrange[0], cv_mean[imin]))
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4.5))
loglam = np.log(lagrange)
se = cv_sd / np.sqrt(5)
ax.fill_between(loglam, cv_mean - se, cv_mean + se, color='#1f77b4', alpha=0.2, lw=0)
ax.plot(loglam, cv_mean, color='#1f77b4', lw=2, marker='o', ms=4)
ax.axvline(loglam[imin], color='k', ls='--', lw=1)
ax.annotate(r'$\hat\lambda_{\min}$', (loglam[imin], cv_mean.max() * 0.9),
            xytext=(5, 0), textcoords='offset points')
ax.invert_xaxis()
ax.set_yscale('log')
ax.set_xlabel(r'$\log \lambda$')
ax.set_ylabel('CV mean squared error (± 1 s.e.)')
ax.set_title('5-fold cross-validation of the lasso path');
```

The same call with a scikit-learn splitter, and with grouped cases (say, ten cases per
subject):

```{code-cell} ipython3
from sklearn.model_selection import KFold

cv_mean_kf, _ = cross_validate(path, lagrange, cv=KFold(5, shuffle=True, random_state=0),
                               n_jobs=1, verbose=False)
subjects = np.repeat(np.arange(20), 10)
cv_mean_grp, _ = cross_validate(path, lagrange, cv=5, group_labels=subjects,
                                n_jobs=1, verbose=False)
print('KFold:      lambda_min = %.2f' % lagrange[np.argmin(cv_mean_kf)])
print('GroupKFold: lambda_min = %.2f' % lagrange[np.argmin(cv_mean_grp)])
```

Finally, refit at the selected $\lambda$: the path's row `imin` is the solution there.

```{code-cell} ipython3
B = strong_rules(path, lagrange, (path.solution.copy(), path.grad_solution.copy()),
                 inner_tol=1e-10)['beta']
print('selected variables:', np.nonzero(B[imin])[0])
print('estimates:', np.round(B[imin][B[imin] != 0], 2))
```
