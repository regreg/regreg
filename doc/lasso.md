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

# Getting started: the LASSO

The LASSO is the prototypical problem `regreg` is built for:

$$
\hat\beta(\lambda) = \text{argmin}_{\beta \in \mathbb{R}^p} \; \frac{1}{2}\|Y - X\beta\|^2_2 + \lambda \|\beta\|_1 .
$$

The objective splits into a *smooth* part (the loss, whose gradient we can
evaluate) and a *nonsmooth* part (the penalty, whose proximal map we can
evaluate). `regreg` represents each part by an object and hands the pair to a
proximal gradient solver (FISTA). This page walks through each piece and checks
the answer against first-order optimality conditions and against
scikit-learn.

```{code-cell} ipython3
import numpy as np
import regreg.api as rr

rng = np.random.default_rng(0)
n, p = 100, 20
X = rng.standard_normal((n, p))
beta_true = np.zeros(p)
beta_true[:4] = [3, -2, 1.5, 1]
Y = X @ beta_true + rng.standard_normal(n)
```

## The loss

`rr.squared_error(X, Y)` is the function $\beta \mapsto \frac{1}{2}\|Y - X\beta\|^2_2$.
It is a smooth atom (the quadratic $\mu \mapsto \frac{1}{2}\|\mu - Y\|_2^2$)
composed with the linear map $\beta \mapsto X\beta$.

```{code-cell} ipython3
loss = rr.squared_error(X, Y)
print(type(loss).__name__, loss.shape)
```

Every smooth object exposes `smooth_objective(beta, mode)`, which returns the
value (`mode='func'`), the gradient (`mode='grad'`) or both (`mode='both'`).
The solver calls only this method, so any function implementing it can be used
as a loss.

```{code-cell} ipython3
beta0 = np.zeros(p)
value = loss.smooth_objective(beta0, mode='func')
grad = loss.smooth_objective(beta0, mode='grad')
value_both, grad_both = loss.smooth_objective(beta0, mode='both')

np.testing.assert_allclose(value, 0.5 * np.sum(Y**2))
np.testing.assert_allclose(grad, X.T @ (X @ beta0 - Y))
np.testing.assert_allclose(grad_both, grad)
value
```

The same function is available as a Gaussian GLM, `rr.glm.gaussian(X, Y)`,
which also accepts case weights and offsets (see the losses page). Its value
and gradient coincide with `squared_error`.

```{code-cell} ipython3
gauss = rr.glm.gaussian(X, Y)
b = rng.standard_normal(p)
f1, g1 = loss.smooth_objective(b, 'both')
f2, g2 = gauss.smooth_objective(b, 'both')
np.testing.assert_allclose([f1], [f2])
np.testing.assert_allclose(g1, g2)
```

## The penalty

`rr.l1norm(p, lagrange=lam)` is the function $\beta \mapsto \lambda\|\beta\|_1$
on $\mathbb{R}^p$. What makes it an *atom* is that its proximal map,

$$
\text{prox}(z) = \text{argmin}_\beta \; \frac{L}{2}\|\beta - z\|^2_2 + \lambda\|\beta\|_1,
$$

is available in closed form: soft-thresholding at $\lambda/L$. The quadratic
term is represented by an `rr.identity_quadratic(L, center, linear_term,
constant)`, i.e. $\frac{L}{2}\|\beta - \mu\|^2_2 + \eta^T\beta + \gamma$.

```{code-cell} ipython3
lam = 20.
penalty = rr.l1norm(p, lagrange=lam)

z = rng.standard_normal(p) * 30
soft = np.sign(z) * np.maximum(np.abs(z) - lam, 0)
np.testing.assert_allclose(penalty.proximal(rr.identity_quadratic(1, z, 0, 0)), soft)
penalty
```

## Solving the problem

`rr.simple_problem(loss, penalty)` pairs one smooth function with one atom, and
`solve` runs FISTA with backtracking. The main controls are `tol` (relative
change in objective at which to stop) and `min_its`/`max_its`. The solution is
returned and also stored in `problem.coefs`, which is used as the starting point
of the next call to `solve`.

```{code-cell} ipython3
problem = rr.simple_problem(loss, penalty)
beta_hat = problem.solve(tol=1e-12, min_its=100)
np.round(beta_hat, 3)
```

### Checking the KKT conditions

The solution is characterized by $0 \in \nabla \ell(\hat\beta) + \lambda\,\partial\|\hat\beta\|_1$,
i.e. with $g = X^T(X\hat\beta - Y)$,

$$
g_j = -\lambda\, \text{sign}(\hat\beta_j) \text{ if } \hat\beta_j \neq 0, \qquad |g_j| \le \lambda \text{ if } \hat\beta_j = 0.
$$

```{code-cell} ipython3
def check_lasso_kkt(grad, beta, lam, tol=1e-5):
    active = beta != 0
    eq_err = np.max(np.abs(grad[active] + lam * np.sign(beta[active])))
    ineq = np.max(np.abs(grad[~active])) if np.any(~active) else 0.
    print(f'active set size {active.sum()}, '
          f'max |g_j + lam sign(b_j)| on active set = {eq_err:.2e}, '
          f'max |g_j| off active set = {ineq:.3f} (lam = {lam})')
    assert eq_err < tol and ineq <= lam + tol

check_lasso_kkt(loss.smooth_objective(beta_hat, 'grad'), beta_hat, lam)
```

## The bound (constrained) form

Every seminorm atom can instead be used as a constraint. `rr.l1norm(p, bound=delta)`
is the indicator of the ball $\{\beta: \|\beta\|_1 \le \delta\}$, and the problem

$$
\text{minimize}_\beta \; \frac{1}{2}\|Y - X\beta\|^2_2 \quad \text{subject to } \|\beta\|_1 \le \delta
$$

is solved the same way; the proximal map is now projection onto the
$\ell_1$ ball.

The two forms trace out the same set of solutions. If $\hat\beta$ solves the
Lagrange form at $\lambda$, it solves the bound form with
$\delta = \|\hat\beta\|_1$. Conversely, if $\hat\beta$ solves the bound form
(with the constraint active), the corresponding Lagrange multiplier is
$\lambda = \|X^T(Y - X\hat\beta)\|_\infty$, read off from the KKT conditions
above.

```{code-cell} ipython3
delta = np.abs(beta_hat).sum()
constraint = rr.l1norm(p, bound=delta)
beta_bound = rr.simple_problem(loss, constraint).solve(tol=1e-12, min_its=100)

print('max |beta_lagrange - beta_bound| =', np.max(np.abs(beta_hat - beta_bound)))
np.testing.assert_allclose(beta_bound, beta_hat, atol=1e-6)

lam_recovered = np.max(np.abs(loss.smooth_objective(beta_bound, 'grad')))
print('recovered lagrange parameter:', lam_recovered)
np.testing.assert_allclose(lam_recovered, lam, rtol=1e-5)
```

## Duality and the conjugate

The convex conjugate of $\lambda\|\cdot\|_1$ is the indicator of the
$\ell_\infty$ ball of radius $\lambda$; more generally the conjugate of a
seminorm in Lagrange form is its dual norm in bound form, and vice versa.
`regreg` returns this as `penalty.conjugate`.

```{code-cell} ipython3
print(penalty.conjugate)
print(constraint.conjugate)
```

Fenchel duality gives the LASSO dual

$$
\text{maximize}_r \; Y^T r - \frac{1}{2}\|r\|^2_2 \quad \text{subject to } \|X^T r\|_\infty \le \lambda,
$$

whose solution is the residual $\hat r = Y - X\hat\beta$. The constraint is
exactly the conjugate atom, so we can check dual feasibility with
`penalty.conjugate.nonsmooth_objective(..., check_feasibility=True)` (which
returns 0 inside the set and $\infty$ outside), and check that the duality gap
vanishes.

```{code-cell} ipython3
r = Y - X @ beta_hat
dual_constraint = penalty.conjugate
print('dual feasible:',
      dual_constraint.nonsmooth_objective(X.T @ r, check_feasibility=True) == 0)
print('scaled up residual feasible:',
      dual_constraint.nonsmooth_objective(1.01 * X.T @ r, check_feasibility=True) == 0)

primal = 0.5 * np.sum(r**2) + lam * np.abs(beta_hat).sum()
dual = Y @ r - 0.5 * np.sum(r**2)
print(f'primal value {primal:.8f}, dual value {dual:.8f}, |gap| {abs(primal - dual):.1e}')
np.testing.assert_allclose(primal, problem.objective(beta_hat))
assert abs(primal - dual) < 1e-5 * primal
```

(The residual sits on the boundary of the dual feasible set, since some
constraints $|x_j^T \hat r| \le \lambda$ are active; scaling it by $1.01$
leaves the set. The gap is zero only up to the solver tolerance, so it can come
out slightly negative.)

## Adding a ridge term: the elastic net

A quadratic $\frac{\kappa}{2}\|\beta\|^2_2$ is an `identity_quadratic(kappa, 0, 0, 0)`.
Quadratics of this form are handled exactly inside the proximal map, so adding
one costs nothing. It can be passed to `solve` (it is used for that call only)
or attached to the atom (or loss) when constructing it. Both give the elastic
net

$$
\text{minimize}_\beta \; \frac{1}{2}\|Y - X\beta\|^2_2 + \lambda \|\beta\|_1 + \frac{\kappa}{2}\|\beta\|^2_2 .
$$

```{code-cell} ipython3
kappa = 5.
ridge = rr.identity_quadratic(kappa, 0, 0, 0)

beta_enet = problem.solve(quadratic=ridge, tol=1e-12, min_its=100)

penalty_enet = rr.l1norm(p, lagrange=lam, quadratic=ridge)
beta_enet2 = rr.simple_problem(loss, penalty_enet).solve(tol=1e-12, min_its=100)
np.testing.assert_allclose(beta_enet, beta_enet2, atol=1e-8)

# KKT: the gradient of the smooth part now includes kappa * beta
check_lasso_kkt(loss.smooth_objective(beta_enet, 'grad') + kappa * beta_enet,
                beta_enet, lam)
np.round(beta_enet, 3)
```

## Comparison with scikit-learn

`sklearn.linear_model.Lasso` minimizes

$$
\frac{1}{2n}\|Y - X\beta\|^2_2 + \alpha\|\beta\|_1 ,
$$

which is the `regreg` objective divided by $n$, so $\alpha = \lambda / n$.
Likewise `ElasticNet` minimizes
$\frac{1}{2n}\|Y - X\beta\|^2_2 + \alpha\rho\|\beta\|_1 + \frac{\alpha(1-\rho)}{2}\|\beta\|^2_2$,
so $\alpha\rho = \lambda/n$ and $\alpha(1-\rho) = \kappa/n$. We turn off the
intercept, since our problem has none.

```{code-cell} ipython3
from sklearn.linear_model import Lasso, ElasticNet

sk_lasso = Lasso(alpha=lam / n, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, Y)
print('lasso: max |regreg - sklearn| =', np.max(np.abs(sk_lasso.coef_ - beta_hat)))
np.testing.assert_allclose(sk_lasso.coef_, beta_hat, atol=1e-6)

alpha = (lam + kappa) / n
sk_enet = ElasticNet(alpha=alpha, l1_ratio=lam / (lam + kappa), fit_intercept=False,
                     tol=1e-12, max_iter=100000).fit(X, Y)
print('elastic net: max |regreg - sklearn| =', np.max(np.abs(sk_enet.coef_ - beta_enet)))
np.testing.assert_allclose(sk_enet.coef_, beta_enet, atol=1e-6)
```

## A solution path by warm starts

Solutions change continuously in $\lambda$, so solving over a decreasing grid
of $\lambda$ values, starting each fit from the previous solution, is cheap.
For $\lambda \ge \lambda_{\max} = \|X^TY\|_\infty$ the solution is zero. The
`lagrange` attribute of the atom can be changed in place, and `problem.coefs`
carries the warm start from one call to the next.

```{code-cell} ipython3
import matplotlib.pyplot as plt

lam_max = np.max(np.abs(X.T @ Y))
lam_grid = lam_max * np.logspace(0, -3, 50)

path_penalty = rr.l1norm(p, lagrange=lam_max)
path_problem = rr.simple_problem(loss, path_penalty)
path = []
for l in lam_grid:
    path_penalty.lagrange = l
    path.append(path_problem.solve(tol=1e-10).copy())
path = np.array(path)

# spot check one point of the path against sklearn
k = 25
sk = Lasso(alpha=lam_grid[k] / n, fit_intercept=False, tol=1e-12, max_iter=100000).fit(X, Y)
print('path point', k, ': max |regreg - sklearn| =', np.max(np.abs(sk.coef_ - path[k])))
np.testing.assert_allclose(sk.coef_, path[k], atol=1e-5)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(np.log10(lam_grid), path, lw=1.2)
ax.set_xlabel(r'$\log_{10} \lambda$')
ax.set_ylabel(r'$\hat\beta_j(\lambda)$')
ax.set_title('LASSO solution path')
ax.invert_xaxis()
ax.axhline(0, color='gray', lw=0.5)
plt.tight_layout()
```
