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

# Atoms, duality and proximal maps

regreg solves composite problems

$$
\min_\beta \; \ell(\beta) + \mathcal{P}(\beta)
$$

where $\ell$ is smooth and $\mathcal{P}$ is nonsmooth but *simple*: its
proximal map can be computed exactly and cheaply. The simple nonsmooth pieces
are called **atoms**. This page explains what an atom is, how its proximal map
is parametrized, and how atoms relate to each other through convex duality.

```{code-cell} ipython3
import numpy as np
import regreg.api as rr
from IPython.display import Math, display
```

## Seminorms in Lagrange and bound form

Most atoms are built from a seminorm $h$ (the $\ell_1$ norm, the $\ell_2$ norm,
the group lasso norm, the nuclear norm, ...). Every seminorm atom comes in two
forms, chosen by which keyword is given to the constructor:

* **Lagrange form** `lagrange=λ`: the function $\beta \mapsto \lambda\, h(\beta)$;
* **bound form** `bound=δ`: the constraint $\beta \mapsto I^\infty(h(\beta) \le \delta)$,
  which is $0$ when $h(\beta) \le \delta$ and $+\infty$ otherwise.

Exactly one of the two must be specified.

```{code-cell} ipython3
p = 6
penalty = rr.l1norm(p, lagrange=0.5)      # 0.5 * ||beta||_1
constraint = rr.l1norm(p, bound=2.0)      # I(||beta||_1 <= 2)
penalty, constraint
```

The method `seminorm` evaluates $\lambda h(\beta)$ (or $h$ with a different
multiplier, via `lagrange=`), while `nonsmooth_objective` evaluates the atom as
a function, including any offset and quadratic term attached to it (see below).
For a bound-form atom `nonsmooth_objective` returns $0$, or $+\infty$ for
infeasible points when `check_feasibility=True`.

```{code-cell} ipython3
beta = np.array([3., -1., 0., 0.5, 0., -2.])
print(penalty.seminorm(beta), 0.5 * np.abs(beta).sum())
print(penalty.nonsmooth_objective(beta))
print(constraint.nonsmooth_objective(beta, check_feasibility=True),   # ||beta||_1 = 6.5 > 2
      constraint.nonsmooth_objective(beta / 5, check_feasibility=True))
```

Besides seminorms there are **cone constraints**, the indicator functions of
convex cones: `rr.nonnegative`, `rr.nonpositive`, `rr.zero`,
`rr.l2_epigraph` ($\|\beta_{1:p-1}\|_2 \le \beta_p$), and so on. They have no
Lagrange parameter.

```{code-cell} ipython3
rr.nonnegative(p), rr.l2_epigraph(p)
```

## Displaying atoms

Every atom has a `latexify()` method returning a LaTeX string, and a
`_repr_latex_` method so that Jupyter renders an atom as mathematics when it is
the last expression in a cell. (The empty subscripts `_{}` are placeholders for
an optional index argument `idx`.)

```{code-cell} ipython3
print(penalty.latexify())
print(constraint.latexify(var=r'\gamma'))
display(Math(rr.group_lasso([0, 0, 1, 1, 1], lagrange=1.).latexify()))
```

## Proximal maps

The single operation an atom must support is its proximal map. In regreg the
proximal map is parametrized by an `identity_quadratic`,

$$
q(\beta) = \frac{L}{2}\|\beta - \mu\|_2^2 + \langle \eta, \beta \rangle + \gamma,
\qquad
\texttt{rr.identity\_quadratic(coef=}L\texttt{, center=}\mu\texttt{, linear\_term=}\eta\texttt{, constant\_term=}\gamma\texttt{)}.
$$

`atom.proximal(q)` returns the minimizer of $q(\beta) + \mathcal{P}(\beta)$.
Inside FISTA the quadratic is $\frac{L}{2}\|\beta - \beta_k\|^2 + \langle \nabla
\ell(\beta_k), \beta\rangle$, so a proximal gradient step is literally
`atom.proximal(rr.identity_quadratic(L, beta_k, grad_k, 0))`.

Two lower-level methods solve the "standard" problems with no linear term:
`lagrange_prox(θ, lipschitz=L)` returns
$\arg\min_\beta \frac{L}{2}\|\beta-\theta\|^2 + \lambda h(\beta)$ and
`bound_prox(θ)` returns the projection
$\arg\min_\beta \frac{1}{2}\|\beta-\theta\|^2$ subject to $h(\beta)\le \delta$.
`proximal` reduces to one of these by completing the square: the minimizer of
$q$ is $\theta = \mu - \eta / L$.

For the $\ell_1$ norm the Lagrange prox is soft-thresholding at $\lambda/L$.

```{code-cell} ipython3
rng = np.random.default_rng(0)
L = 2.0
theta = rng.standard_normal(p) * 2
eta = rng.standard_normal(p)
mu = theta

soft = lambda z, t: np.sign(z) * np.maximum(np.abs(z) - t, 0)

q = rr.identity_quadratic(L, mu, eta, 0)
prox = penalty.proximal(q)
expected = soft(mu - eta / L, penalty.lagrange / L)
np.testing.assert_allclose(prox, expected)
np.testing.assert_allclose(penalty.lagrange_prox(mu - eta / L, lipschitz=L), expected)
prox
```

The bound-form prox is a projection onto the $\ell_1$ ball, which does not
depend on $L$:

```{code-cell} ipython3
proj = constraint.proximal(rr.identity_quadratic(L, theta, 0, 0))
np.testing.assert_allclose(proj, constraint.bound_prox(theta))
print(np.abs(proj).sum())   # on the boundary of the l1 ball of radius 2
```

`solve` is a convenience for nonsmooth atoms: `atom.solve(q)` is the same
prox, and `return_optimum=True` also returns the optimal value.

### Offsets and attached quadratics

An atom may carry an **offset** $\alpha$, in which case it is evaluated at
$\beta - \alpha$, and a **quadratic** term that is added to it. With both,

$$
\mathcal{P}(\beta) = \lambda h(\beta - \alpha) + \frac{L_0}{2}\|\beta-\mu_0\|^2 + \langle \eta_0, \beta\rangle + \gamma_0 .
$$

These are how regreg represents shifted penalties, ridge terms attached to a
penalty, and the linear terms that appear in duals.

```{code-cell} ipython3
alpha = np.ones(p)
q0 = rr.identity_quadratic(0.3, 0, 0.2 * np.ones(p), 1.0)
shifted = rr.l1norm(p, lagrange=0.5, offset=alpha, quadratic=q0)
display(Math(shifted.latexify()))

value = shifted.nonsmooth_objective(beta)
direct = 0.5 * np.abs(beta - alpha).sum() + q0.objective(beta, 'func')
print(value, direct)
np.testing.assert_allclose(value, direct)
```

The proximal map accounts for both: it minimizes $q(\beta) + \mathcal{P}(\beta)$.
Substituting $u = \beta - \alpha$ and adding the two quadratics, the minimizer is
$\alpha$ plus a soft-thresholding of the new center.

```{code-cell} ipython3
total = (q + q0).collapsed()            # a single quadratic L/2 |b|^2 + <eta, b> + c
Ltot, eta_tot = total.coef, total.linear_term
expected = alpha + soft(-eta_tot / Ltot - alpha, 0.5 / Ltot)
np.testing.assert_allclose(shifted.proximal(q), expected)

# an attached quadratic is the same as passing it to proximal
plain = rr.l1norm(p, lagrange=0.5, offset=alpha)
np.testing.assert_allclose(shifted.proximal(q), plain.proximal(q + q0))
```

## Conjugates and duals

The convex conjugate of $f$ is $f^*(u) = \sup_\beta \langle u, \beta\rangle - f(\beta)$.
For a seminorm $h$ with dual norm $h_*$ (here $h(\beta) = \sup_{h_*(u)\le 1}
\langle u, \beta \rangle$),

$$
(\lambda h)^* = I^\infty(h_*(\cdot) \le \lambda),
\qquad
\big(I^\infty(h(\cdot)\le\delta)\big)^* = \delta\, h_*,
$$

so conjugation swaps the norm with its dual norm *and* the Lagrange form with
the bound form. `atom.conjugate` returns this atom:

```{code-cell} ipython3
print(penalty.conjugate)                                  # l1norm  <-> supnorm
print(constraint.conjugate)
print(rr.group_lasso([0, 0, 1, 1, 1], lagrange=1.).conjugate)
print(rr.nuclear_norm((4, 3), lagrange=1.).conjugate)     # nuclear <-> operator norm
print(rr.l2norm(p, lagrange=1.).conjugate)                # l2 is self-dual
print(rr.nonnegative(p).conjugate)                        # cone <-> polar cone
```

Offsets and linear terms are exchanged by conjugation: the conjugate of
$\beta \mapsto \lambda h(\beta - \alpha)$ is
$u \mapsto I^\infty(h_*(u) \le \lambda) + \langle \alpha, u\rangle$.

```{code-cell} ipython3
c = rr.l1norm(p, lagrange=0.5, offset=alpha).conjugate
print(c)
display(Math(c.latexify()))
```

If the atom carries a quadratic with positive coefficient, its conjugate is a
smooth function, represented by a `smooth_conjugate` object (see smoothing
below).

The `dual` property (equivalently `get_dual()`) returns the pair
`(linear_transform, conjugate)`. For a plain atom the transform is the
identity; for an atom composed with a linear map $D$ it is $D$. This is what
the dual problem solvers use: $\min_\beta \ell(\beta) + h(D\beta)$ has dual
$\min_u \ell^*(-D^T u) + h^*(u)$.

```{code-cell} ipython3
penalty.dual
```

### The Moreau decomposition

The proximal maps of $f$ and $f^*$ are linked by the Moreau decomposition: for
any $L > 0$,

$$
x = \operatorname{prox}_{f/L}(x) + \frac{1}{L} \operatorname{prox}_{L f^*}(L x).
$$

In regreg notation, $\operatorname{prox}_{f/L}(x)$ is
`atom.proximal(rr.identity_quadratic(L, x, 0, 0))` and
$\operatorname{prox}_{Lf^*}(Lx)$ is
`atom.conjugate.proximal(rr.identity_quadratic(1/L, L*x, 0, 0))`. For a
Lagrange-form seminorm the second term is a projection onto
$\lambda \cdot\{h_* \le 1\}$, so the decomposition says that soft-thresholding is
the residual from projecting onto the dual ball. We check it for a range of
atoms, in both forms, including one with an offset:

```{code-cell} ipython3
L = 2.5
groups = [0, 0, 1, 1, 1, 2]
atoms = [rr.l1norm(6, lagrange=0.7),
         rr.l1norm(6, bound=1.2),
         rr.l2norm(6, lagrange=0.7),
         rr.supnorm(6, bound=0.4),
         rr.weighted_l1norm(np.linspace(0, 1, 6), lagrange=0.5),
         rr.group_lasso(groups, lagrange=0.7),
         rr.sparse_group_lasso(groups, np.ones(6), lagrange=0.3),
         rr.slope(np.linspace(2, 1, 6), lagrange=0.5),
         rr.nuclear_norm((4, 3), lagrange=0.7),
         rr.l1_l2((3, 4), lagrange=0.5),
         rr.nonnegative(6),
         rr.positive_part(6, lagrange=0.5),
         rr.l1norm(6, lagrange=0.7, offset=np.ones(6))]

rng = np.random.default_rng(1)
for atom in atoms:
    x = rng.standard_normal(atom.shape)
    conj = atom.conjugate
    recon = (atom.proximal(rr.identity_quadratic(L, x, 0, 0)) +
             conj.proximal(rr.identity_quadratic(1 / L, L * x, 0, 0)) / L)
    err = np.abs(recon - x).max()
    name = lambda a: type(a).__module__.split('.')[-1] + '.' + type(a).__name__
    print(f'{name(atom):>32s} <-> {name(conj):<36s} max |error| = {err:.1e}')
    np.testing.assert_allclose(recon, x, atol=1e-10)
```

## Smoothing an atom

Adding a strongly convex quadratic to the conjugate and conjugating back gives
a smooth approximation (Nesterov smoothing):

$$
h_\mu(\beta) = \sup_u \; \langle u, \beta\rangle - h^*(u) - \frac{\mu}{2}\|u\|_2^2 .
$$

$h_\mu$ is differentiable with $\nabla h_\mu(\beta) = \hat u(\beta)$, the maximizer,
which is $1/\mu$-Lipschitz; and $h - \frac{\mu}{2}\sup_{u \in \mathrm{dom}\, h^*}\|u\|^2
\le h_\mu \le h$. `atom.smoothed(rr.identity_quadratic(mu, 0, 0, 0))` returns
$h_\mu$ as a smooth atom, usable as a loss. For $\lambda\|\cdot\|_1$ it is the
Huber function, with gradient $\mathrm{clip}(\beta/\mu, -\lambda, \lambda)$.

```{code-cell} ipython3
mu = 0.2
lam = 0.5
l1 = rr.l1norm(p, lagrange=lam)
huber = l1.smoothed(rr.identity_quadratic(mu, 0, 0, 0))
print(huber)
print('Lipschitz constant of the gradient:', huber.lipschitz)

rng = np.random.default_rng(2)
x = rng.standard_normal(p)
f, g = huber.smooth_objective(x, mode='both')
np.testing.assert_allclose(g, np.clip(x / mu, -lam, lam))

# finite-difference check of the gradient
h = 1e-6
fd = np.array([(huber.smooth_objective(x + h * e, 'func') -
                huber.smooth_objective(x - h * e, 'func')) / (2 * h)
               for e in np.identity(p)])
print('max |finite difference - gradient|:', np.abs(fd - g).max())
np.testing.assert_allclose(fd, g, atol=1e-6)

# sandwich bound: sup ||u||^2 over the l_inf ball of radius lam is p * lam^2
f_exact = l1.seminorm(x)
print(f_exact - mu / 2 * p * lam**2 <= f <= f_exact)
```

## Composing atoms with linear maps

Many penalties are an atom evaluated at a linear function of $\beta$, e.g. the
fused lasso $\lambda \|D\beta\|_1$ with $D$ the first-difference matrix. The class
methods

* `rr.l1norm.linear(D, lagrange=λ)` — $\lambda\, h(D\beta)$,
* `rr.l1norm.affine(D, a, lagrange=λ)` — $\lambda\, h(D\beta + a)$,
* `rr.l1norm.shift(a, lagrange=λ)` — $\lambda\, h(\beta + a)$,

construct these (the same methods exist on every seminorm class). `shift`
returns an ordinary atom with offset $-a$. `linear` and `affine` return an
`affine_atom`: it can be evaluated, displayed, dualized and smoothed, but it has
no cheap proximal map in general. Such penalties are handled either by smoothing
or by solving a dual problem (e.g. `rr.dual_problem`), in which the conjugate
atom, which does have a cheap prox, appears and $D^T$ is moved into the smooth
part.

```{code-cell} ipython3
p = 8
D = (np.identity(p) - np.diag(np.ones(p - 1), 1))[:-1]   # (D beta)_i = beta_i - beta_{i+1}
fused = rr.l1norm.linear(D, lagrange=1.)
shifted_fused = rr.l1norm.affine(D, np.ones(p - 1), lagrange=1.)
shift = rr.l1norm.shift(np.ones(p), lagrange=1.)

rng = np.random.default_rng(3)
b = rng.standard_normal(p)
np.testing.assert_allclose(fused.nonsmooth_objective(b), np.abs(D @ b).sum())
np.testing.assert_allclose(shifted_fused.nonsmooth_objective(b), np.abs(D @ b + 1).sum())
np.testing.assert_allclose(shift.nonsmooth_objective(b), np.abs(b + 1).sum())
display(Math(fused.latexify()))
display(Math(shifted_fused.latexify()))
fused.dual
```

Smoothing an `affine_atom` gives a smooth function of $\beta$,
$\beta \mapsto h_\mu(D\beta)$, with gradient $D^T \hat u(D\beta)$:

```{code-cell} ipython3
mu = 0.2
smooth_fused = fused.smoothed(rr.identity_quadratic(mu, 0, 0, 0))
f, g = smooth_fused.smooth_objective(b, mode='both')
np.testing.assert_allclose(g, D.T @ np.clip(D @ b / mu, -1, 1))

h = 1e-6
fd = np.array([(smooth_fused.smooth_objective(b + h * e, 'func') -
                smooth_fused.smooth_objective(b - h * e, 'func')) / (2 * h)
               for e in np.identity(p)])
print('max |finite difference - gradient|:', np.abs(fd - g).max())
np.testing.assert_allclose(fd, g, atol=1e-6)
```

The smoothed penalty can be added to a loss and minimized with any of the
problem containers. Smaller $\mu$ gives a closer approximation at the price of a
larger Lipschitz constant ($\|D\|_{\mathrm{op}}^2/\mu$) and slower convergence.
