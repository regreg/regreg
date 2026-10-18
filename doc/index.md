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

# regreg

`regreg` is a Python framework for regularized regression and, more
generally, for convex problems of the form

$$
\text{minimize}_{\beta} \quad \ell(X\beta) + \sum_j h_j(D_j \beta),
$$

where $\ell$ is a smooth loss (squared error, a GLM negative
log-likelihood, the Cox partial likelihood, a Huber loss, …) and each $h_j$
is a nonsmooth penalty or constraint (the $\ell_1$ norm, group norms,
SLOPE, the nuclear norm, nonnegativity, linear constraints, …), possibly
composed with a linear map $D_j$. Problems are solved by proximal gradient
methods (FISTA), through the dual when there are several nonsmooth terms,
or by ADMM and TFOCS-style smoothing.

The emphasis is on building problems out of pieces rather than on any one
model:

- **Atoms** are the nonsmooth pieces. Each is a seminorm or a cone
  constraint, in Lagrange form ($\lambda h(\beta)$) or bound form
  ($h(\beta) \le c$), with a proximal map and a conjugate, so that duality
  comes for free. See [Atoms, duality and proximal maps](atoms.md).
- **Smooth losses** know their value and gradient, and compose with a design
  matrix or any linear operator. See [Losses and GLMs](losses.md).
- **Affine transforms** are matrices, sparse matrices or matrix-free
  operators, and atoms can be composed with them: the fused lasso, trend
  filtering and isotonic regression are an atom of a difference operator.
  See [Constraints and linear transforms](constraints.md).
- **Problems** pair them up and are solved by [Solvers](solvers.md).

On top of these, `regreg.paths` computes [regularization paths](paths.md)
with strong rules and cross-validation, and `regreg.sklearn` wraps models as
[scikit-learn estimators](sklearn.md).

## Installation

Install the latest release from PyPI:

```
pip install regreg
```

There are wheels for Linux (x86_64), macOS (arm64) and Windows (64-bit); on
other platforms pip builds the Cython extensions from the source
distribution, which needs a C compiler. These pages are built from the
development version, which installs from the repository:

```
pip install git+https://github.com/regreg/regreg
```

The optional extras are `regreg[sklearn]` (scikit-learn, for the
`regreg.sklearn` estimators and cross-validation splitters) and
`regreg[test]`.

## A first example

The LASSO, $\frac{1}{2}\|Y - X\beta\|_2^2 + \lambda \|\beta\|_1$, is a loss and
an atom:

```{code-cell} ipython3
import numpy as np
import regreg.api as rr

rng = np.random.default_rng(0)
n, p = 100, 20
X = rng.standard_normal((n, p))
beta = np.zeros(p)
beta[:3] = [3, -2, 1.5]
Y = X @ beta + rng.standard_normal(n)

loss = rr.squared_error(X, Y)
penalty = rr.l1norm(p, lagrange=20.)
problem = rr.simple_problem(loss, penalty)
coef = problem.solve(tol=1e-10)
np.round(coef, 3)
```

Swapping the atom changes the model: `rr.group_lasso(groups, lagrange=...)`
gives the group LASSO, `rr.l1norm(p, bound=...)` the constrained form,
and `rr.nonnegative(p)` nonnegative least squares. The
[Getting started](lasso.md) page goes through this example in detail.

## Links

- Source code and issues: <https://github.com/regreg/regreg>
- [API reference](https://regreg.github.io/regreg/api/)
