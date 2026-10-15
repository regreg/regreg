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

# Losses and GLMs

Most smooth losses in regression depend on the coefficients only through
the linear predictor $\eta = X\beta$:

$$
\beta \mapsto \ell(X\beta), \qquad \ell(\eta) = \sum_{i=1}^n w_i\, \ell_i(\eta_i).
$$

`regreg` builds such losses from two pieces: a *saturated loss* $\ell$,
a smooth function of the $n$-vector $\eta$ that knows about the response (and
case weights $w_i$), and a design matrix $X$. The class `rr.glm` composes the
two, and its class-method constructors build the common models:

| constructor | saturated loss $\ell_i(\eta_i)$ (negative log-likelihood up to constants) |
|---|---|
| `rr.glm.gaussian(X, y)` | $\frac{1}{2}(y_i - \eta_i)^2$ |
| `rr.glm.logistic(X, y, trials=None)` | $n_i \log(1 + e^{\eta_i}) - y_i\eta_i$ |
| `rr.glm.probit(X, y)` | $-y_i \log\Phi(\eta_i) - (n_i - y_i)\log(1 - \Phi(\eta_i))$ |
| `rr.glm.cloglog(X, y)` | binomial with $\pi_i = 1 - \exp(-e^{\eta_i})$ |
| `rr.glm.poisson(X, y)` | $e^{\eta_i} - y_i\eta_i$ |
| `rr.glm.huber(X, y, eps)` | $h_\epsilon(\eta_i - y_i)$, $h_\epsilon(r) = r^2/(2\epsilon)$ for $\lvert r\rvert\le\epsilon$, $\lvert r\rvert - \epsilon/2$ otherwise |
| `rr.glm.huber_svm(X, y, eps)` | a Huberized hinge loss in $(2y_i - 1)\eta_i$ |
| `rr.glm.cox(X, times, status)` | Cox negative log partial likelihood (not separable in $i$) |

Each constructor also takes `case_weights`, `coef` (a nonnegative multiplier
of the whole loss), `saturated_offset`, `offset` and `quadratic`. Because they
all expose the same `smooth_objective` method, any of them can be paired with
any penalty in `rr.simple_problem`.

```{code-cell} ipython3
import numpy as np
from scipy.stats import norm
import regreg.api as rr

rng = np.random.default_rng(0)
n, p = 300, 6
X = rng.standard_normal((n, p))
beta_true = np.array([1., -0.8, 0.5, 0, 0, 0])
eta_true = X @ beta_true
```

## Value and gradient checks

For each binary-response and regression loss we compare the value with the
formula above and the gradient with central finite differences. The gradient
of $\beta \mapsto \ell(X\beta)$ is $X^T \nabla\ell(X\beta)$, computed by the
`glm` object from the saturated loss's gradient.

```{code-cell} ipython3
y_bin = rng.binomial(1, 1 / (1 + np.exp(-eta_true)))
y_cont = eta_true + rng.standard_normal(n)
y_count = rng.poisson(np.exp(0.5 * eta_true))

losses = {
    'gaussian': (rr.glm.gaussian(X, y_cont),
                 lambda e: 0.5 * np.sum((y_cont - e)**2)),
    'logistic': (rr.glm.logistic(X, y_bin),
                 lambda e: np.sum(np.logaddexp(0, e) - y_bin * e)),
    'probit': (rr.glm.probit(X, y_bin),
               lambda e: -np.sum(y_bin * norm.logcdf(e) + (1 - y_bin) * norm.logsf(e))),
    'cloglog': (rr.glm.cloglog(X, y_bin),
                lambda e: -np.sum(y_bin * np.log(-np.expm1(-np.exp(e)))
                                  - (1 - y_bin) * np.exp(e))),
    'poisson': (rr.glm.poisson(X, y_count),
                lambda e: np.sum(np.exp(e) - y_count * e)),
    'huber': (rr.glm.huber(X, y_cont, 0.5), None),
    'huber_svm': (rr.glm.huber_svm(X, y_bin, 0.5), None),
}

def fd_grad(f, b, h=1e-6):
    E = np.eye(len(b))
    return np.array([(f(b + h * E[j]) - f(b - h * E[j])) / (2 * h) for j in range(len(b))])

b = 0.3 * rng.standard_normal(p)
for name, (loss, formula) in losses.items():
    value, grad = loss.smooth_objective(b, 'both')
    fd = fd_grad(lambda v: loss.smooth_objective(v, 'func'), b)
    grad_err = np.max(np.abs(grad - fd)) / max(1, np.max(np.abs(grad)))
    msg = f'{name:10s} value {value:10.4f}  rel. grad error {grad_err:.1e}'
    if formula is not None:
        np.testing.assert_allclose(value, formula(X @ b), rtol=1e-10)
        msg += '  (value matches formula)'
    print(msg)
    assert grad_err < 1e-6
```

The `glm` object is literally the composition of a saturated loss with $X$; the
saturated losses are available on their own (e.g. `rr.logistic_loglike`,
`rr.poisson_loglike`) and `rr.glm(X, Y, saturated_loss)` puts them together.
The design can also be any `regreg` affine transform (for instance
`rr.normalize(X)`, which centers and scales columns without forming a new
matrix).

```{code-cell} ipython3
sat = rr.logistic_loglike(y_bin.shape, y_bin)
by_hand = rr.glm(X, y_bin, sat)
np.testing.assert_allclose(by_hand.smooth_objective(b, 'func'),
                           losses['logistic'][0].smooth_objective(b, 'func'))
print(type(losses['logistic'][0].saturated_loss).__name__)
```

## Case weights and offsets

`case_weights` multiplies each observation's contribution, $\ell(\eta) = \sum_i w_i \ell_i(\eta_i)$.
There are two kinds of offset, distinguished by the space they live in:

* `saturated_offset` $o \in \mathbb{R}^n$ is subtracted from the linear
  predictor: the loss is $\ell(X\beta - o)$.
* `offset` $\alpha \in \mathbb{R}^p$ is subtracted from the coefficients:
  the loss is $\ell(X(\beta - \alpha))$.

Note the sign convention. A classical GLM offset enters as
$\eta = X\beta + o$ (e.g. $o = \log(\text{exposure})$ in a Poisson rate
model), so it is passed as `saturated_offset=-o`.

```{code-cell} ipython3
w = rng.uniform(0.5, 2, n)
o = 0.2 * rng.standard_normal(n)
alpha = 0.1 * rng.standard_normal(p)
e = X @ b

def logistic_nll(eta, weights=1):
    return np.sum(weights * (np.logaddexp(0, eta) - y_bin * eta))

checks = [
    (rr.glm.logistic(X, y_bin, case_weights=w), logistic_nll(e, w)),
    (rr.glm.logistic(X, y_bin, saturated_offset=o), logistic_nll(e - o)),
    (rr.glm.logistic(X, y_bin, offset=alpha), logistic_nll(X @ (b - alpha))),
]
for loss, expected in checks:
    np.testing.assert_allclose(loss.smooth_objective(b, 'func'), expected, rtol=1e-12)
print('case weights and offsets behave as described')
```

## Penalized logistic regression vs scikit-learn

The $\ell_1$-penalized logistic regression

$$
\text{minimize}_\beta \; \sum_i \left[\log(1 + e^{x_i^T\beta}) - y_i x_i^T\beta\right] + \lambda\|\beta\|_1
$$

is `rr.simple_problem(rr.glm.logistic(X, y), rr.l1norm(p, lagrange=lam))`.
scikit-learn's `LogisticRegression` with an $\ell_1$ penalty minimizes
$\|\beta\|_1 + C \sum_i (\text{log-loss})_i$, which is the same problem divided by
$\lambda$, so $C = 1/\lambda$. (In scikit-learn $\ge 1.8$ the $\ell_1$ penalty is
requested with `l1_ratio=1`; older versions use `penalty='l1'`.) We use the
`saga` solver with no intercept.

```{code-cell} ipython3
import warnings
from sklearn.linear_model import LogisticRegression

def sk_l1_logistic(X, y, C, **kw):
    import sklearn
    major, minor = map(int, sklearn.__version__.split('.')[:2])
    pen = dict(l1_ratio=1) if (major, minor) >= (1, 8) else dict(penalty='l1')
    return LogisticRegression(C=C, solver='saga', fit_intercept=False,
                              tol=1e-12, max_iter=100000, **pen, **kw).fit(X, y)

lam = 8.
logistic_loss = rr.glm.logistic(X, y_bin)
problem = rr.simple_problem(logistic_loss, rr.l1norm(p, lagrange=lam))
beta_logit = problem.solve(tol=1e-12, min_its=200)

sk = sk_l1_logistic(X, y_bin, C=1 / lam)
print(np.round(beta_logit, 4))
print('max |regreg - sklearn| =', np.max(np.abs(sk.coef_.ravel() - beta_logit)))
np.testing.assert_allclose(sk.coef_.ravel(), beta_logit, atol=1e-6)
```

The same KKT check as for the LASSO applies, with the logistic score in
place of the least squares gradient.

```{code-cell} ipython3
g = logistic_loss.smooth_objective(beta_logit, 'grad')
active = beta_logit != 0
print('active-set KKT error', np.max(np.abs(g[active] + lam * np.sign(beta_logit[active]))))
print('max |score| off the active set', np.max(np.abs(g[~active])), '<= lam =', lam)
assert np.max(np.abs(g[active] + lam * np.sign(beta_logit[active]))) < 1e-5
assert np.all(np.abs(g[~active]) <= lam)
```

## Unpenalized Poisson regression vs IRLS

With no penalty FISTA is just accelerated gradient descent on the smooth
loss; `rr.simple_problem.smooth(loss)` builds that problem. Here we fit a
Poisson rate model with exposures $t_i$, i.e. $\log E[y_i] = \log t_i + x_i^T\beta$,
and compare with Newton's method (equivalently IRLS) written directly in numpy:

$$
\beta \leftarrow \beta + (X^T W X)^{-1} X^T (y - \mu), \qquad \mu = t \odot e^{X\beta},\; W = \text{diag}(\mu).
$$

```{code-cell} ipython3
Xp = 0.5 * X[:, :4]
exposure = rng.uniform(1, 5, n)
counts = rng.poisson(exposure * np.exp(Xp @ np.array([0.5, -0.5, 0.25, 0.])))

pois = rr.glm.poisson(Xp, counts, saturated_offset=-np.log(exposure))
beta_pois = rr.simple_problem.smooth(pois).solve(tol=1e-14, min_its=200)

beta_newton = np.zeros(Xp.shape[1])
for _ in range(10):
    mu = exposure * np.exp(Xp @ beta_newton)
    beta_newton += np.linalg.solve(Xp.T @ (mu[:, None] * Xp), Xp.T @ (counts - mu))

print(np.round(beta_pois, 5))
print('max |FISTA - Newton| =', np.max(np.abs(beta_pois - beta_newton)))
np.testing.assert_allclose(beta_pois, beta_newton, atol=1e-6)
```

`glm` objects also provide `hessian(beta)`, here $X^T W X$, which is what
second-order methods need:

```{code-cell} ipython3
mu = exposure * np.exp(Xp @ beta_pois)
np.testing.assert_allclose(pois.hessian(beta_pois), Xp.T @ (mu[:, None] * Xp), rtol=1e-8)
```

## The Cox model

`rr.glm.cox(X, times, status)` is the negative log partial likelihood of the
proportional hazards model, with `status` equal to 1 for an observed failure
and 0 for a censored time. Its value, gradient and Hessian are computed by the
[`coxdev`](https://github.com/jonathan-taylor/coxdev) package. Tied failure
times are handled by Breslow's approximation (the default) or Efron's,
selected by `tie_breaking='breslow'` or `'efron'`. With Breslow's method

$$
\ell(\eta) = -\sum_{i: \delta_i = 1}\left[\eta_i - \log \sum_{j: T_j \ge T_i} e^{\eta_j}\right],
$$

and Efron's method replaces the risk-set sum for a time with $d$ tied failures
$D$ by $\sum_{j \in R} e^{\eta_j} - \frac{k}{d}\sum_{j \in D} e^{\eta_j}$ for
$k = 0, \dots, d - 1$. We round the times to create plenty of ties and check
both against direct implementations.

```{code-cell} ipython3
T = rng.exponential(np.exp(-eta_true))
cens = rng.exponential(2, n)
times = np.round(np.minimum(T, cens), 1) + 0.1
status = (T <= cens).astype(int)
print('number of tied times:', n - len(np.unique(times)), ' failures:', status.sum())

def breslow_nll(eta):
    return -sum(eta[i] - np.log(np.exp(eta[times >= times[i]]).sum())
                for i in range(n) if status[i])

def efron_nll(eta):
    val = 0.
    for t in np.unique(times[status == 1]):
        D = (times == t) & (status == 1)
        d = D.sum()
        risk, tied = np.exp(eta[times >= t]).sum(), np.exp(eta[D]).sum()
        val -= eta[D].sum() - sum(np.log(risk - k / d * tied) for k in range(d))
    return val

for tb, ref in [('breslow', breslow_nll), ('efron', efron_nll)]:
    cox = rr.glm.cox(X, times, status, tie_breaking=tb)
    v, g = cox.smooth_objective(b, 'both')
    fd = fd_grad(lambda u: cox.smooth_objective(u, 'func'), b)
    grad_err = np.max(np.abs(g - fd)) / np.max(np.abs(g))
    print(f'{tb:8s} value {v:.6f}  direct {ref(X @ b):.6f}  rel. grad error {grad_err:.1e}')
    np.testing.assert_allclose(v, ref(X @ b), rtol=1e-10)
    assert grad_err < 1e-6
```

A lasso-penalized Cox model is again a `simple_problem`. We check it against
the KKT conditions, and check the unpenalized fit against Newton's method using
the Hessian from `coxdev`.

```{code-cell} ipython3
cox = rr.glm.cox(X, times, status, tie_breaking='efron')
# a fraction of the smallest lambda giving the zero solution, ||grad at 0||_inf
lam_cox = 0.3 * np.max(np.abs(cox.smooth_objective(np.zeros(p), 'grad')))
beta_cox = rr.simple_problem(cox, rr.l1norm(p, lagrange=lam_cox)).solve(tol=1e-14, min_its=500)
print(f'lambda = {lam_cox:.2f}; lasso Cox:', np.round(beta_cox, 4))

g = cox.smooth_objective(beta_cox, 'grad')
active = beta_cox != 0
print('active-set KKT error', np.max(np.abs(g[active] + lam_cox * np.sign(beta_cox[active]))))
assert np.max(np.abs(g[active] + lam_cox * np.sign(beta_cox[active]))) < 1e-6
assert np.all(np.abs(g[~active]) <= lam_cox)

beta_mle = rr.simple_problem.smooth(cox).solve(tol=1e-14, min_its=200)
beta_nr = np.zeros(p)
for _ in range(10):
    beta_nr -= np.linalg.solve(cox.hessian(beta_nr), cox.smooth_objective(beta_nr, 'grad'))
print('unpenalized: max |FISTA - Newton| =', np.max(np.abs(beta_mle - beta_nr)))
np.testing.assert_allclose(beta_mle, beta_nr, atol=1e-6)
```

## Multinomial regression

For a response with $J$ categories the coefficients form a $p \times J$
matrix $B$ and the linear predictor is the $n \times J$ matrix $XB$.
`rr.mglm.multinomial(X, Y)` takes $Y$ as an $n \times J$ matrix of counts (one-hot
rows for a single draw per case) and uses the overparametrized loss

$$
\ell(XB) = -\sum_i \left[\sum_j y_{ij} (XB)_{ij} - n_i \log \sum_j e^{(XB)_{ij}}\right]
$$

(`baseline=True` instead fixes the first category's coefficients at zero).
Without a penalty $B$ is identified only up to adding a common vector to every
column, but an $\ell_1$ penalty on all entries picks a unique solution, and this
is the same problem scikit-learn solves for multinomial $\ell_1$-penalized
logistic regression, again with $C = 1/\lambda$.

```{code-cell} ipython3
J = 3
Xm = X[:, :4]
B_true = rng.standard_normal((4, J))
prob = np.exp(Xm @ B_true)
prob /= prob.sum(1, keepdims=True)
labels = np.array([rng.choice(J, p=pr) for pr in prob])
Y = np.eye(J)[labels]

# zero counts trigger a harmless log(0) warning when the saturated
# log-likelihood is computed
with np.errstate(divide='ignore', invalid='ignore'):
    multi = rr.mglm.multinomial(Xm, Y)
print('coefficient shape', multi.shape)

lam_m = 5.
B_hat = rr.simple_problem(multi, rr.l1norm(multi.shape, lagrange=lam_m)).solve(tol=1e-12, min_its=200)
print(np.round(B_hat, 4))

sk_m = sk_l1_logistic(Xm, labels, C=1 / lam_m)
print('max |regreg - sklearn| =', np.max(np.abs(sk_m.coef_.T - B_hat)))
np.testing.assert_allclose(sk_m.coef_.T, B_hat, atol=1e-6)
```
