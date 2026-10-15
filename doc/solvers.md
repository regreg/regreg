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

# Solvers

Every problem in `regreg` is put in composite form,

$$
\text{minimize}_\beta \ f(\beta) + g(\beta),
$$

with $f$ smooth (its gradient is Lipschitz) and $g$ convex with a cheap
*proximal map*

$$
\text{prox}_{g/L}(z) = \text{argmin}_\beta \ \frac{L}{2}\|\beta - z\|_2^2 + g(\beta).
$$

The main solver is FISTA, the accelerated proximal gradient method of Beck
and Teboulle (2009). Each iteration takes a gradient step on $f$ from an
extrapolated point and then applies the proximal map of $g$. This page
describes the problem containers and solver options, how to plug in your
own problem, and the alternatives for problems that do not fit this
pattern directly: dual solves for `container`, ADMM, TFOCS/NESTA-style
smoothing, and a proximal quasi-Newton method.

```{code-cell} ipython3
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog, lsq_linear

import regreg.api as rr
```

## `simple_problem`: one smooth part, one atom

`rr.simple_problem(loss, atom)` pairs a smooth loss with a single atom whose
proximal map is known exactly. Its `solve` method runs FISTA and returns
the solution; keyword arguments go to `FISTA.fit`:

- `tol` (default `1e-5`): stop when the relative change in objective
  (or, with `coef_stop=True`, in coefficients) falls below `tol`;
- `min_its` (default 5) and `max_its` (default 10000);
- `FISTA` (default `True`): use Nesterov momentum; `False` gives plain
  proximal gradient descent (ISTA);
- `start_step` (default 1): the initial step size for backtracking;
- `restart` (default never): reset momentum every `restart` iterations;
- `monotonicity_restart` (default `True`): reset momentum whenever the
  objective increases (adaptive restart);
- `coef_stop` (default `False`): use the coefficient-change criterion;
- `debug`: print the objective, step size and relative change at each
  iteration.

The step size is chosen by backtracking: it is decreased by a factor
$1.1$ until the usual quadratic upper bound on $f$ holds, and tentatively
increased every 100 iterations. After `solve`, the history of objective
values is in `problem.solver_results` and the final step in
`problem.final_step`. Because the backtracking condition is only checked
along the iterates, the accepted step reflects the local curvature of $f$
and can be much larger than $1/L$, where $L$ is the global Lipschitz
constant of $\nabla f$. In the example below $L = \|X\|_2^2$ is dominated
by the factor shared by all columns, and the final step is about ten times
$1/L$.

```{code-cell} ipython3
rng = np.random.default_rng(0)
n, p = 100, 400
X = 0.3 * rng.standard_normal((n, p)) + rng.standard_normal((n, 1))  # correlated columns
beta = np.zeros(p)
beta[:10] = 3
Y = X @ beta + rng.standard_normal(n)
lam = 0.05 * np.abs(X.T @ Y).max()

def lasso():
    return rr.simple_problem(rr.squared_error(X, Y), rr.l1norm(p, lagrange=lam))

problem = lasso()
beta_hat = problem.solve(tol=1e-10)
print('iterations:', len(problem.solver_results))
print('final step: %.3g   1/L: %.3g' % (problem.final_step, 1 / np.linalg.norm(X, 2)**2))
print('nonzero coefficients:', (beta_hat != 0).sum())
```

## Using `FISTA` directly

`rr.FISTA(problem)` creates the solver; `fit` takes the same arguments as
above and returns the vector of objective values, one per iteration. The
solution is written into `problem.coefs` (FISTA modifies the problem's
coefficients in place, so a second `fit` warm starts from the previous
solution).

Momentum matters. With correlated columns the problem is badly
conditioned and ISTA converges at rate $O(1/k)$, against $O(1/k^2)$ for
FISTA. Without adaptive restart FISTA's objective is not monotone and it
oscillates; restarting the momentum whenever the objective increases
(the default) removes the oscillation and typically converges fastest.

```{code-cell} ipython3
f_star = lasso().objective(lasso().solve(tol=1e-15, min_its=20000, max_its=20000))

histories = {}
for label, kwargs in [('FISTA (default)', {}),
                      ('FISTA, no restart', {'monotonicity_restart': False}),
                      ('ISTA', {'FISTA': False})]:
    solver = rr.FISTA(lasso())
    histories[label] = solver.fit(tol=1e-12, max_its=2000, **kwargs)
    print('%-18s iterations: %4d   f - f*: %.1e' % (label, len(histories[label]),
                                                    histories[label][-1] - f_star))

assert histories['FISTA (default)'][-1] - f_star < 1e-6 * f_star
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
for label, hist in histories.items():
    ax.semilogy(np.maximum(hist - f_star, 1e-12) / f_star, label=label)
ax.set_xlabel('iteration')
ax.set_ylabel(r'$(F(\beta_k) - F^*) / F^*$')
ax.legend();
```

### Fixed step size

Backtracking costs extra evaluations of $f$. If the Lipschitz constant
is known, set `solver.perform_backtrack = False` and give the problem a
`lipschitz` attribute; FISTA then uses step $1/L$ throughout.
`rr.power_L(X)` estimates $\|X\|_2^2$, which is $L$ for least squares.
For this problem the global constant is pessimistic, and the fixed step
needs more iterations than backtracking.

```{code-cell} ipython3
problem = lasso()
problem.lipschitz = rr.power_L(X)
solver = rr.FISTA(problem)
solver.perform_backtrack = False
hist = solver.fit(tol=1e-12)
print('iterations:', len(hist), '  f - f*: %.1e' % (hist[-1] - f_star))
assert hist[-1] - f_star < 1e-6 * f_star
```

## Defining your own problem: the `composite` interface

FISTA only needs four things from a problem: a `coefs` array (the starting
point, overwritten with the solution), `smooth_objective(x, mode)` returning
the value (`mode='func'`), gradient (`'grad'`) or both (`'both'`) of $f$,
`nonsmooth_objective(x)` returning $g(x)$, and `proximal(q)`. The argument
`q` of `proximal` is an `rr.identity_quadratic`, the function
$\beta \mapsto \frac{L}{2}\|\beta - \mu\|_2^2 + \eta^T\beta + \gamma$, and
`proximal` must return the minimizer of $q(\beta) + g(\beta)$. Its
attribute `q.true_center` is $\mu - \eta / L$, so the proximal map is
evaluated at `q.true_center` with step $1/L$ where $L$ is `q.coef`.

Subclassing `rr.composite` provides `coefs`, `objective` and the plumbing.
As an example, here is least squares under box constraints
$\ell \le \beta \le u$, whose proximal map is clipping, compared with
`scipy.optimize.lsq_linear`.

```{code-cell} ipython3
class box_least_squares(rr.composite):

    def __init__(self, X, Y, lower, upper):
        self.X, self.Y, self.lower, self.upper = X, Y, lower, upper
        rr.composite.__init__(self, X.shape[1])

    def smooth_objective(self, beta, mode='both', check_feasibility=False):
        resid = self.X @ beta - self.Y
        f, g = 0.5 * resid @ resid, self.X.T @ resid
        return {'func': f, 'grad': g, 'both': (f, g)}[mode]

    def nonsmooth_objective(self, beta, check_feasibility=False):
        inside = np.all((beta >= self.lower - 1e-10) & (beta <= self.upper + 1e-10))
        return 0. if inside else np.inf

    def proximal(self, q, prox_control=None):
        return np.clip(q.true_center, self.lower, self.upper)

rng = np.random.default_rng(1)
Xb = rng.standard_normal((50, 10))
Yb = 3 * rng.standard_normal(50)

box = box_least_squares(Xb, Yb, -0.2, 0.2)
rr.FISTA(box).fit(tol=1e-14, min_its=100)
reference = lsq_linear(Xb, Yb, bounds=(-0.2, 0.2), tol=1e-12).x

print(np.round(box.coefs, 3))
print('max |regreg - lsq_linear|:', np.abs(box.coefs - reference).max())
np.testing.assert_allclose(box.coefs, reference, atol=1e-7)
```

`regreg`'s own atoms and smooth losses implement the same interface (with
the quadratic term handled for you), so they can be mixed with
hand-written pieces: for instance `rr.simple_problem(my_smooth, atom)`
works for any object with `smooth_objective` and `coefs`.

## `container`: several atoms, solved through the dual

`rr.container(loss, atom_1, ..., atom_k)` minimizes
$f(\beta) + \sum_j {\cal P}_j(D_j\beta + \alpha_j)$ where the atoms are
possibly composed with affine maps (see
[constraints and linear transforms](constraints.md)). The proximal map of
the sum has no closed form, so at every FISTA step the container solves the
dual of the proximal problem,

$$
\text{minimize}_\beta \ \frac{L}{2}\|\beta - z\|_2^2 + \sum_j {\cal P}_j(D_j\beta + \alpha_j)
\iff
\text{minimize}_{u_1, \dots, u_k} \ \frac{1}{2L}\Big\|\sum_j D_j^Tu_j - Lz\Big\|_2^2
- \sum_j \alpha_j^T u_j + \sum_j {\cal P}_j^*(u_j),
$$

by an inner FISTA run (with fixed step from `power_L` of the stacked
$D_j$), and recovers $\beta = z - \sum_j D_j^Tu_j / L$. The conjugates
${\cal P}_j^*$ are again atoms with exact proximal maps. The inner solver
can be controlled by passing `prox_control`, a dictionary of `fit`
arguments (defaults `tol=1e-14`, `max_its=5000`), to `solve` or `fit`.
Because the inner solve is iterative, the outer solve is accurate only to
roughly the precision of the inner one.

Below is a generalized lasso, $\frac{1}{2}\|Y - X\beta\|_2^2 + \lambda\|D\beta\|_1$
with a random $D$.

```{code-cell} ipython3
rng = np.random.default_rng(2)
n, p, m = 200, 100, 30
X = rng.standard_normal((n, p))
Y = X @ np.r_[3 * np.ones(10), np.zeros(p - 10)] + rng.standard_normal(n)
D = rng.standard_normal((m, p))
lam = 0.2 * np.abs(X.T @ Y).max()

genlasso = rr.container(rr.squared_error(X, Y), rr.l1norm.linear(D, lagrange=lam))
tic = time.time()
beta_container = genlasso.solve(tol=1e-12)
print('container: %.2fs, objective %.6f' % (time.time() - tic,
                                            genlasso.objective(beta_container)))
```

## ADMM

The alternating direction method of multipliers handles the same
generalized problem $\ell(\beta) + {\cal P}(D\beta + \alpha)$ by splitting
$z = D\beta + \alpha$ and alternating

$$
\beta \leftarrow \text{argmin}_\beta \ \ell(\beta) + \frac{\rho}{2}\|D\beta + \alpha - z + y/\rho\|_2^2,
\qquad
z \leftarrow \text{prox}_{{\cal P}/\rho}(D\beta + \alpha + y/\rho),
\qquad
y \leftarrow y + \rho(D\beta + \alpha - z).
$$

`regreg.problems.admm.admm_problem(loss, atom, D, rho)` implements this; the
$\beta$-update is a smooth problem solved by FISTA (with arguments from
`fit_args`), and `solve(niter)` runs a fixed number of iterations. The
primal iterate is `loss_coefs`. Each iteration is cheap but ADMM converges
slowly to high accuracy; here 200 iterations agree with the container to
about $10^{-5}$.

```{code-cell} ipython3
from regreg.problems.admm import admm_problem

admm = admm_problem(rr.squared_error(X, Y), rr.l1norm(m, lagrange=lam), D, 1.)
tic = time.time()
admm.solve(niter=200)
beta_admm = admm.loss_coefs
print('ADMM: %.2fs, objective %.6f' % (time.time() - tic, genlasso.objective(beta_admm)))
print('max |ADMM - container|:', np.abs(beta_admm - beta_container).max())
np.testing.assert_allclose(beta_admm, beta_container, atol=1e-3)
```

## Smoothing: TFOCS and NESTA

Some problems have no smooth part at all, for instance basis pursuit,

$$
\text{minimize}_\beta \ \|\beta\|_1 \quad \text{subject to} \quad X\beta = Y,
$$

or the Dantzig selector,
$\text{minimize} \ \|\beta\|_1$ subject to $\|X^T(Y - X\beta)\|_\infty \le \delta$.
Both are $h(\beta) + k(A\beta + a)$ with $h$ and $k$ having simple proximal
maps. The TFOCS approach (Becker, Candès and Grant, 2011) adds a proximal
term $\frac{\epsilon}{2}\|\beta - \beta_0\|_2^2$ to $h$. This makes the
dual problem smooth,

$$
\text{minimize}_u \ h_\epsilon^*(-A^Tu) - a^Tu + k^*(u),
$$

with $h_\epsilon^*$ the conjugate of the regularized $h$, so the dual can
be solved by FISTA. The primal solution is $\nabla h_\epsilon^*(-A^T\hat u)$.
The regularized problem is then re-solved with $\beta_0$ moved to
(an extrapolation of) the new primal solution. This outer loop is a
proximal point method, so it converges to the solution of the original
problem for any fixed $\epsilon$.

`rr.tfocs(h, transform, k_conjugate, epsilon=[...])` runs this scheme, one
outer iteration per entry of `epsilon`. The pair `(transform, k_conjugate)`
is the `dual` attribute of the composed atom $k(A\beta + a)$. For basis
pursuit we check the result against a linear-programming solve.

```{code-cell} ipython3
rng = np.random.default_rng(3)
n, p = 100, 300
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:8] = np.arange(8) + 1
Y = X @ beta                           # noiseless

l1 = rr.l1norm(p, lagrange=1)
transform, zero = rr.zero_constraint.affine(X, -Y).dual
tic = time.time()
beta_tfocs, u_tfocs = rr.tfocs(l1, transform, zero, epsilon=[1.] * 50)
print('tfocs: %.2fs' % (time.time() - tic))

# basis pursuit as a linear program in (beta^+, beta^-)
lp = linprog(np.ones(2 * p), A_eq=np.hstack([X, -X]), b_eq=Y, bounds=(0, None))
beta_lp = lp.x[:p] - lp.x[p:]
print('max |tfocs - LP|:', np.abs(beta_tfocs - beta_lp).max())
print('max |X beta - Y|:', np.abs(X @ beta_tfocs - Y).max())
np.testing.assert_allclose(beta_tfocs, beta_lp, atol=1e-2)
```

The accuracy of `rr.tfocs` is limited because it solves each smoothed dual
problem only to relative tolerance $\max(\epsilon, \text{tol})$, which
for moderate $\epsilon$ is loose, and the error compounds over the outer
iterations ([gh-104](https://github.com/regreg/regreg/issues/104)). The loop itself is short, and written out with a
tight inner tolerance it converges well. Here it is for the Dantzig selector, whose
constraint is `rr.supnorm.affine(X.T @ X, -X.T @ Y, bound=delta)`.

```{code-cell} ipython3
from regreg.affine import adjoint, scalar_multiply

def tfocs_loop(h, transform, k_conjugate, epsilon, niter, inner_tol=1e-10):
    h_conjugate = h.conjugate
    primal = np.zeros(h.shape)
    for i in range(niter):
        # h_eps^*: conjugate of h(beta) + eps/2 ||beta - primal||^2
        smoothed = h_conjugate.smoothed(rr.identity_quadratic(epsilon, primal, 0, 0))
        dual_loss = rr.affine_smooth(smoothed, scalar_multiply(adjoint(transform), -1))
        rr.simple_problem(dual_loss, k_conjugate).solve(tol=inner_tol, max_its=5000)
        new_primal = dual_loss.grad          # gradient of h_eps^* at -A^T u
        primal = new_primal + i / (i + 3.) * (new_primal - primal)
    return new_primal

Y_noisy = Y + 0.5 * rng.standard_normal(n)
G, g = X.T @ X, X.T @ Y_noisy
delta = 0.5 * np.sqrt(2 * np.log(p) * n)

dantzig = rr.supnorm.affine(G, -g, bound=delta)
transform, k_conjugate = dantzig.dual
tic = time.time()
beta_ds = tfocs_loop(rr.l1norm(p, lagrange=1), transform, k_conjugate, 1., 50)
print('Dantzig selector: %.2fs' % (time.time() - tic))

lp = linprog(np.ones(2 * p),
             A_ub=np.vstack([np.hstack([G, -G]), np.hstack([-G, G])]),
             b_ub=np.r_[delta + g, delta - g], bounds=(0, None))
beta_ds_lp = lp.x[:p] - lp.x[p:]
print('||X^T(Y - X beta)||_inf / delta: %.5f' % (np.abs(g - G @ beta_ds).max() / delta))
print('l1 norm: tfocs %.5f   LP %.5f' % (np.abs(beta_ds).sum(), np.abs(beta_ds_lp).sum()))
print('max |tfocs - LP|:', np.abs(beta_ds - beta_ds_lp).max())
np.testing.assert_allclose(beta_ds, beta_ds_lp, atol=1e-3)
```

NESTA (Becker, Bobin and Candès, 2011) smooths the other way: it keeps the
proximal map of $h$ and replaces the constraint $k(A\beta + a)$ by its
Moreau-Yosida smoothing, obtained by adding $\frac{\epsilon}{2}\|u - u_0\|_2^2$
to its conjugate (the atom method `smoothed`). For the equality constraint
this smoothing is the augmented-Lagrangian penalty
$u_0^T(X\beta - Y) + \frac{1}{2\epsilon}\|X\beta - Y\|_2^2$, and updating
$u_0$ to the gradient of the smoothed term is the multiplier update.
`rr.nesta(smooth, h, k_atom, epsilon=[...])` runs this scheme, with an
optional extra smooth term `smooth`. It ties the inner tolerance to
$\epsilon$ in the same way as `rr.tfocs`, so again we write the loop out.

```{code-cell} ipython3
def nesta_loop(h, k_atom, epsilon, niter, inner_tol=1e-12):
    transform, _ = k_atom.dual
    u = np.zeros(transform.output_shape)
    for i in range(niter):
        smoothed = k_atom.smoothed(rr.identity_quadratic(epsilon, u, 0, 0))
        primal = rr.simple_problem(smoothed, h).solve(tol=inner_tol, max_its=5000)
        u = smoothed.grad                   # multiplier update
    return primal

tic = time.time()
beta_nesta = nesta_loop(rr.l1norm(p, lagrange=1), rr.zero_constraint.affine(X, -Y), 1., 50)
print('NESTA: %.2fs' % (time.time() - tic))
print('max |NESTA - LP|:', np.abs(beta_nesta - beta_lp).max())
np.testing.assert_allclose(beta_nesta, beta_lp, atol=1e-6)
```

## Proximal quasi-Newton

For smooth losses with strong curvature, such as logistic regression,
proximal Newton methods (Lee, Sun and Saunders, 2014) take fewer, more
expensive steps. At each step they minimize the quadratic model of $f$
plus $g$, $\nabla f(\beta)^T\Delta + \frac{1}{2}\Delta^TH\Delta + g(\beta + \Delta)$,
and then line search along $\Delta$. `regreg.problems.newton.quasi_newton`
does this with BFGS updates of $H$, starting from a user-supplied
`initial_hessian`. The quadratic subproblems are solved by FISTA, and the
keyword arguments of `solve` are passed to that inner solver.

```{code-cell} ipython3
from regreg.problems.newton import quasi_newton

rng = np.random.default_rng(4)
n, p = 500, 50
X = rng.standard_normal((n, p))
eta = X @ np.r_[np.ones(5), np.zeros(p - 5)]
Y = rng.binomial(1, 1 / (1 + np.exp(-eta)))
lam = 0.1 * np.abs(X.T @ (Y - Y.mean())).max()

fista_problem = rr.simple_problem(rr.glm.logistic(X, Y), rr.l1norm(p, lagrange=lam))
beta_fista = fista_problem.solve(tol=1e-12, min_its=50)

qn = quasi_newton(rr.glm.logistic(X, Y), rr.l1norm(p, lagrange=lam),
                  initial_hessian=X.T @ X / 4)
beta_qn = qn.solve(niter=30, tol=1e-12)
print('quasi-Newton steps:', len(qn.solver_results) - 1)
print('max |quasi-Newton - FISTA|:', np.abs(beta_qn - beta_fista).max())
np.testing.assert_allclose(beta_qn, beta_fista, atol=1e-5)
```

## Which solver?

For a smooth loss plus one atom (possibly `separable`), use
`simple_problem`, whose proximal step is exact. Add atoms composed with
linear maps through `container`, which is convenient but solves a dual
problem at every step. ADMM is an alternative when the $\beta$-update is
cheap or can be reused. For problems with no smooth part, such as
constrained $\ell_1$ minimization, use TFOCS- or NESTA-style smoothing.
