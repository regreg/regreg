

```python
import numpy as np
import regreg.api as rr
%pylab inline
from sklearn import datasets
iris = datasets.load_iris()
digits = datasets.load_digits()
```

## Hinge loss

The SVM can be parametrized various ways, one way to write
it as a regression problem is to use the hinge loss:
$$
\ell(r) = \max(1-x, 0)
$$


```python
hinge = lambda x: np.maximum(1-x, 0)
fig = plt.figure(figsize=(9,6))
ax = fig.gca()
r = np.linspace(-1,2,100)
ax.plot(r, hinge(r))
```

The SVM loss is then
$$
\ell(\beta) = C \sum_{i=1}^n h(Y_i X_i^T\beta) + \frac{1}{2} \|\beta\|^2_2
)
$$
where $Y_i \in \{-1,1\}$ and $X_i \in \mathbb{R}^p$ is one of the feature vectors. 

In regreg, the hinge loss can be  represented by composition of
some of the basic atoms. Specifcally, let $g:\mathbb{R}^n \rightarrow \mathbb{R}$ be the sum of positive part function
$$
g(z) = \sum_{i=1}^n\max(z_i, 0).
$$
Then,
$$
\ell(\beta) = g\left(Y \cdot X\beta \right)
$$
where the product in the parentheses is elementwise multiplication.


```python
linear_part = np.array([[-1.]])
offset = np.array([1.])
hinge_rep = rr.positive_part.affine(linear_part, offset, lagrange=1.)
hinge_rep
```

Let's plot the loss to be sure it agrees with our original hinge.


```python
ax.plot(r, [hinge_rep.nonsmooth_objective(v) for v in r])
fig
```

Here is a vectorized version.


```python
N = 1000
P = 200

Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
X = np.random.standard_normal((N,P))
#X[Y==1] += np.array([30,-20] + (P-2)*[0])[np.newaxis,:]
X -= X.mean(0)[np.newaxis, :]
hinge_vec = rr.positive_part.affine(-Y[:, None] * X, np.ones_like(Y), lagrange=1.)
```


```python
beta = np.ones(X.shape[1])
hinge_vec.nonsmooth_objective(beta), np.maximum(1 - Y * X.dot(beta), 0).sum()
```

## Smoothed hinge

For optimization, the hinge loss is not differentiable so it is often
smoothed first.

The smoothing is applicable to general functions of the form
$$
g(X\beta-\alpha) = g_{\alpha}(X\beta)
$$
where $g_{\alpha}(z) = g(z-\alpha)$ 
and is determined by a small quadratic term
$$
q(z) = \frac{C_0}{2} \|z-x_0\|^2_2 + v_0^Tz + c_0.
$$


```python
epsilon = 0.5
smoothing_quadratic = rr.identity_quadratic(epsilon, 0, 0, 0)
smoothing_quadratic
```

The quadratic terms are determined by four parameters with $(C_0, x_0, v_0, c_0)$.

Smoothing of the function by the quadratic $q$ is performed by Moreau smoothing:
$$
S(g_{\alpha},q)(\beta) = \sup_{z \in \mathbb{R}^p} z^T\beta - g^*_{\alpha}(z) - q(z)
$$
where
$$
g^*_{\alpha}(z) = \sup_{\beta \in \mathbb{R}^p} z^T\beta - g_{\alpha}(\beta)
$$
is the convex (Fenchel) conjugate of the composition $g$ with the translation by
$-\alpha$.

The basic atoms in `regreg` know what their conjugate is. Our hinge loss, `hinge_rep`,
is the composition of an `atom`, and an affine transform. This affine transform is split
into two pieces, the linear part, stored as `linear_transform` and its offset
stored as `atom.offset`. It is stored with `atom` as `atom` needs knowledge of
this when computing proximal maps.


```python
hinge_rep.atom
```


```python
hinge_rep.atom.offset
```


```python
hinge_rep.linear_transform.linear_operator
```

As we said before, `hinge_rep.atom` knows what its conjugate is


```python
hinge_conj = hinge_rep.atom.conjugate
hinge_conj
```

The notation $I^{\infty}$ denotes a constraint. The expression can therefore be parsed as
a linear function $\eta^T\beta$ plus the function
$$
g^*(z) = \begin{cases}
0 & 0 \leq z_i \leq \delta \, \forall i \\
\infty & \text{otherwise.}
\end{cases}
$$

The term $\eta$ is derived from `hinge_rep.atom.offset` and is stored in `hinge_conj.quadratic`.


```python
hinge_conj.quadratic.linear_term
```

Now, let's look at the smoothed hinge loss. 


```python
smoothed_hinge_loss = hinge_rep.smoothed(smoothing_quadratic)
smoothed_hinge_loss
```

It is now a smooth function and its objective value and gradient can be computed with
`smooth_objective`.


```python
ax.plot(r, [smoothed_hinge_loss.smooth_objective(v, 'func') for v in r])
fig
```


```python
less_smooth = hinge_rep.smoothed(rr.identity_quadratic(5.e-2, 0, 0, 0))
ax.plot(r, [less_smooth.smooth_objective(v, 'func') for v in r])
fig
```

## Fitting the SVM

We can now minimize this objective.


```python
smoothed_vec = hinge_vec.smoothed(rr.identity_quadratic(0.2, 0, 0, 0))
soln = smoothed_vec.solve(tol=1.e-12, min_its=100)
```

## Sparse SVM

We might want to fit a sparse version, adding a sparsifying penalty like the LASSO.
This yields the problem
$$
\text{minimize}_{\beta} \ell(\beta) + \lambda \|\beta\|_1
$$


```python
penalty = rr.l1norm(smoothed_vec.shape, lagrange=20)
problem = rr.simple_problem(smoothed_vec, penalty)
problem
```


```python
sparse_soln = problem.solve(tol=1.e-12)
sparse_soln
```

What value of $\lambda$ should we use? For the $\ell_1$ penalty in Lagrange form,
the smallest $\lambda$ such that the solution is zero can be found by taking
the dual norm, the $\ell_{\infty}$ norm, of the gradient of the smooth part at 0.


```python
linf_norm = penalty.conjugate
linf_norm
```

Just computing the conjugate will yield an $\ell_{\infty}$ constraint, but this
object can still be used to compute the desired value of $\lambda$.


```python
score_at_zero = smoothed_vec.smooth_objective(np.zeros(smoothed_vec.shape), 'grad')
lam_max = linf_norm.seminorm(score_at_zero, lagrange=1.)
lam_max
```


```python
penalty.lagrange = lam_max * 1.001
problem.solve(tol=1.e-12, min_its=200)
```


```python
penalty.lagrange = lam_max * 0.99
problem.solve(tol=1.e-12, min_its=200)
```

### Path of solutions

If we want a path of solutions, we can simply take multiples of `lam_max`. This is similar
to the strategy that packages like `glmnet` use


```python
path = []
lam_vals = (np.linspace(0.05, 1.01, 50) * lam_max)[::-1]
for lam_val in lam_vals:
    penalty.lagrange = lam_val
    path.append(problem.solve(min_its=200).copy())
fig = plt.figure(figsize=(12,8))
ax = fig.gca()
path = np.array(path)
ax.plot(path);
```

## Changing the penalty

We may not want to penalize features the same. We may want some features to be unpenalized.
This can be achieved by introducing possibly non-zero feature weights to the $\ell_1$ norm
$$
\beta \mapsto \sum_{j=1}^p w_j|\beta_j|
$$




```python
weights = np.random.sample(P) + 1.
weights[:5] = 0.
weighted_penalty = rr.weighted_l1norm(weights, lagrange=1.)
weighted_penalty
```


```python
weighted_dual = weighted_penalty.conjugate
weighted_dual
```


```python
lam_max_weight = weighted_dual.seminorm(score_at_zero, lagrange=1.)
lam_max_weight

```


```python
weighted_problem = rr.simple_problem(smoothed_vec, weighted_penalty)
path = []
lam_vals = (np.linspace(0.05, 1.01, 50) * lam_max_weight)[::-1]
for lam_val in lam_vals:
    weighted_penalty.lagrange = lam_val
    path.append(weighted_problem.solve(min_its=200).copy())
fig = plt.figure(figsize=(12,8))
ax = fig.gca()
path = np.array(path)
ax.plot(path);
```

Note that there are 5 coefficients that are not penalized hence they are nonzero the entire path.

#### Group LASSO

Variables may come in groups. A common penalty for this setting is the group LASSO.
Let $$
\{1, \dots, p\} = \cup_{g \in G} g
$$
be a partition of the set of features and $w_g$ a weight for each group. The 
group LASSO penalty is
$$
\beta \mapsto \sum_{g \in G} w_g \|\beta_g\|_2.
$$


```python
groups = []
for i in range(P/5):
    groups.extend([i]*5)
weights = dict([g, np.random.sample()+1] for g in np.unique(groups))
group_penalty = rr.group_lasso(groups, weights=weights, lagrange=1.)

```


```python
group_dual = group_penalty.conjugate
lam_max_group = group_dual.seminorm(score_at_zero, lagrange=1.)
```


```python
group_problem = rr.simple_problem(smoothed_vec, group_penalty)
path = []
lam_vals = (np.linspace(0.05, 1.01, 50) * lam_max_group)[::-1]
for lam_val in lam_vals:
    group_penalty.lagrange = lam_val
    path.append(group_problem.solve(min_its=200).copy())
fig = plt.figure(figsize=(12,8))
ax = fig.gca()
path = np.array(path)
ax.plot(path);
```

As expected, variables enter in groups here.

### Bound form

The common norm atoms also have a bound form. That is, we can just as easily solve the 
problem
$$
\text{minimize}_{\beta: \|\beta\|_1 \leq \delta}\ell(\beta)
$$



```python
bound_l1 = rr.l1norm(P, bound=2.)
bound_l1
```


```python
bound_problem = rr.simple_problem(smoothed_vec, bound_l1)
bound_problem
```


```python
bound_soln = bound_problem.solve()
np.fabs(bound_soln).sum()
```


```python

```
