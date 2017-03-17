
# Huberized lasso tutorial

The Huberized lasso minimizes the following objective
$$
H_{\delta}(Y - X\beta) + \lambda \|\beta\|_1
$$
where $H_{\delta}(\cdot)$ is a function applied element-wise,
$$
 H_{\delta}(r) = \left\{\begin{array}{ll} r^2/2 & \mbox{ if } |r| \leq \delta \\ \delta r - \delta^2/2 & \mbox{ else}\end{array} \right.
$$
To solve this problem using RegReg we begin by loading the necessary numerical libraries


```python
import numpy as np
import regreg.api as rr
```

Next, let's generate some example data,


```python
X = np.random.normal(0,1,500000).reshape((500,1000))
Y = np.random.randint(0,2,500)
```

Now we can create the problem object, beginning with the loss function


```python
penalty = rr.l1norm(1000,lagrange=5.)
loss = rr.l1norm.affine(X,-Y, lagrange=1.).smoothed(rr.identity_quadratic(1,0,0,0))
```

The penalty contains the regularization parameter that can be easily accessed and changed,


```python
penalty.lagrange
```

Now we can create the final problem object


```python
problem = rr.simple_problem(loss, penalty)
```

Next, we can select our algorithm of choice and use it solve the problem,


```python
solver = rr.FISTA(problem)
obj_vals = solver.fit(max_its=200, tol=1e-6)
solution = solver.composite.coefs
(solution != 0).sum()
```

Here max_its represents primal iterations, and tol is the primal tolerance.


```python
obj_vals
```

# Poisson regression tutorial

The Poisson regression problem minimizes the objective
$$
-2 \left(Y^TX\beta - \sum_{i=1}^n \mbox{exp}(x_i^T\beta) \right), \qquad Y_i \in {0,1,2,\ldots}
$$
which corresponds to the usual Poisson regression model
$$
P(Y_i=j) = \frac{\mbox{exp}(jx_i^T\beta-\mbox{exp}(x_i^T\beta))}{j!}
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries


```python
import numpy as np
import regreg.api as rr
```

The only code needed to add Poisson regression is a class
with one method which computes the objective and its gradient.

Next, let's generate some example data,


```python
n = 1000
p = 50
X = np.random.standard_normal((n,p))
Y = np.random.randint(0,100,n)
```

Now we can create the problem object, beginning with the loss function


```python
loss = rr.poisson_deviance.linear(X, counts=Y)
```

Next, we can fit this model in the usual way


```python
solver = rr.FISTA(loss)
obj_vals = solver.fit()
solution = solver.composite.coefs
```

# Regularized logistic regression tutorial

The $\ell_2$ regularized logistic regression problem minimizes the objective

$$
-2\left(Y^TX\beta - \sum_i \log \left[ 1 + \exp(x_i^T\beta) \right] \right) + \lambda \|\beta\|_2^2
$$
which corresponds to the usual logistic regression model

$$
P(Y_i=1) = \mbox{logit}(x_i^T\beta) = \frac{1}{1 + \mbox{exp}(-x_i^T\beta)}
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries


```python
import numpy as np
import regreg.api as rr
```

The only code needed to add logistic regression is a class
with one method which computes the objective and its gradient.

.. literalinclude:: ../code/regreg/smooth.py
..   :pyobject: logistic_deviance
   

Next, let's generate some example data,


```python
X = np.random.normal(0,1,500000).reshape((500,1000))
Y = np.random.randint(0,2,500)
```

Now we can create the problem object, beginning with the loss function


```python
loss = rr.logistic_deviance.linear(X,successes=Y)
penalty = rr.identity_quadratic(1., 0., 0., 0.)
loss.quadratic = penalty
loss
```

The logistic log-likelihood function is written without a matrix :math:`X`. We use the ".linear" to specify the linear composition :math:`X\beta`. Similarly, we could use ".affine" to specify an offset :math:`X\beta + \alpha`.
The penalty contains the regularization parameter that can be easily accessed and changed,


```python
penalty.coef
```

Next, we can select our algorithm of choice and use it solve the problem,


```python
solver = rr.FISTA(loss)
obj_vals = solver.fit(max_its=100, tol=1e-5)
solution = solver.composite.coefs
```

Here max_its represents primal iterations, and tol is the primal tolerance.


```python
obj_vals
```

# Multinomial regression

The multinomial regression problem minimizes the objective

$$
-2\left[ \sum_{j=1}^{J-1} \sum_{k=1}^p \beta_{jk}\sum_{i=1}^n x_{ik}y_{ij}
 - \sum_{i=1}^n \log \left(1 + \mbox{exp}(x_i^T\beta_j) \right)\right]
$$
which corresponds to a baseline category logit model for $J$ nominal categories (e.g. Agresti, p.g. 272). For $i \ne J$ the probabilities are measured relative to a baseline category $J$
$$
\frac{P(\mbox{Category } i)}{P(\mbox{Category } J)} = \mbox{logit}(x^T\beta_i) = \frac{1}{1 + \mbox{exp}(-x^T\beta_i)}
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries


```python
import numpy as np
import regreg.api as rr
```

The only code needed to add multinomial regression to RegReg is a class
with one method which computes the objective and its gradient.


Next, let's generate some example data. The multinomial counts will be stored in a $n \times J$ array


```python
J = 5
n = 1000
p = 50
X = np.random.standard_normal((n,p))
Y = np.random.randint(0,10,n*J).reshape((n,J))
```

Now we can create the problem object, beginning with the loss function. The coefficients will be stored in a $p \times J-1$ array, and we need to let RegReg know that the coefficients will be a 2d array instead of a vector. We can do this by defining the input_shape in a linear_transform object that multiplies by X,


```python
multX = rr.linear_transform(X, input_shape=(p,J-1))
loss = rr.multinomial_deviance.linear(multX, counts=Y)
```

Next, we can solve the problem


```python
solver = rr.FISTA(loss)
solver.fit()
```

When $J=2$ this model should reduce to logistic regression. We can easily check that this is the case by first fitting the multinomial model


```python
J = 2
Y = np.random.randint(0,10,n*J).reshape((n,J))
multX = rr.linear_transform(X, input_shape=(p,J-1))	
loss = rr.multinomial_deviance.linear(multX, counts=Y)
solver = rr.FISTA(loss)
solver.fit(tol=1e-6)
multinomial_coefs = solver.composite.coefs.flatten()
```

and then the equivalent logistic regresison model


```python
successes = Y[:,0]
trials = np.sum(Y, axis=1)
loss = rr.logistic_deviance.linear(X, successes=successes, trials=trials)
solver = rr.FISTA(loss)
solver.fit(tol=1e-6)
logistic_coefs = solver.composite.coefs
```

Finally we can check that the two models gave the same coefficients


```python
print np.linalg.norm(multinomial_coefs - logistic_coefs) / np.linalg.norm(logistic_coefs)
```


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

# Support vector machine

This tutorial illustrates one version of the support vector machine, a linear
example. 
The minimization problem for the support vector machine,
following *ESL* is 
$$
\text{minimize}_{\beta,\gamma} \sum_{i=1}^n (1- y_i(x_i^T\beta+\gamma))^+ \frac{\lambda}{2} \|\beta\|^2_2
$$
We use the $C$ parameterization in (12.25) of *ESL*

$$
\text{minimize}_{\beta,\gamma} C \sum_{i=1}^n (1- y_i(x_i^T\beta+\gamma))^+ \frac{1}{2} \|\beta\|^2_2
$$
This is an example of the positive part atom combined with a smooth
quadratic penalty. Above, the $x_i$ are rows of a matrix of features
and the $y_i$ are labels coded as $\pm 1$.

Let's generate some data appropriate for this problem.


```python
import numpy as np

np.random.seed(400) # for reproducibility
N = 500
P = 2

Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
X = np.random.standard_normal((N,P))
X[Y==1] += np.array([3,-2])[np.newaxis,:]
X -= X.mean(0)[np.newaxis,:]
```

We now specify the hinge loss part of the problem


```python
import regreg.api as rr
X_1 = np.hstack([X, np.ones((N,1))])
transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
C = 0.2 # = 1/\lambda
hinge = rr.positive_part(N, lagrange=C)
hinge_loss = rr.linear_atom(hinge, transform)
```

and the quadratic penalty


```python
quadratic = rr.quadratic.linear(rr.selector(slice(0,P), (P+1,)), coef=0.5)
```

Now, let's solve it


```python
problem = rr.simple_problem(quadratic, hinge_loss)
solver = rr.FISTA(problem)
vals = solver.fit()
solver.composite.coefs
```

This determines a line in the plane, specified as $\beta_1 \cdot x + \beta_2 \cdot y + \gamma = 0$ and the classifications are determined by which
side of the line a point is on.


```python
fits = np.dot(X_1, problem.coefs)
labels = 2 * (fits > 0) - 1
accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
accuracy
```


```python
import numpy as np
import regreg.api as rr

np.random.seed(400)

N = 500
P = 2

Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
X = np.random.standard_normal((N,P))
X[Y==1] += np.array([3,-2])[np.newaxis,:]

X_1 = np.hstack([X, np.ones((N,1))])
X_1_signs = -Y[:,np.newaxis] * X_1
transform = rr.affine_transform(X_1_signs, np.ones(N))
C = 0.2
hinge = rr.positive_part(N, lagrange=C)
hinge_loss = rr.linear_atom(hinge, transform)

quadratic = rr.quadratic.linear(rr.selector(slice(0,P), (P+1,)), coef=0.5)
problem = rr.simple_problem(quadratic, hinge_loss)
solver = rr.FISTA(problem)
solver.fit()

import pylab
pylab.clf()
pylab.scatter(X[Y==1,0],X[Y==1,1], facecolor='red')
pylab.scatter(X[Y==-1,0],X[Y==-1,1], facecolor='blue')

fits = np.dot(X_1, problem.coefs)
labels = 2 * (fits > 0) - 1

pointX = [X[:,0].min(), X[:,0].max()]
pointY = [-(pointX[0]*problem.coefs[0]+problem.coefs[2])/problem.coefs[1],
          -(pointX[1]*problem.coefs[0]+problem.coefs[2])/problem.coefs[1]]
pylab.plot(pointX, pointY, linestyle='--', label='Separating hyperplane')
pylab.title("Accuracy = %0.1f %%" % (100-100 * np.fabs(labels - Y).sum() / (2 * N)))
#pylab.show()
```

Sparse SVM
~~~~~~~~~~

We can also fit a sparse SVM by adding a sparsity penalty to the original problem, solving the problem

$$
\text{minimize}_{\beta,\gamma} C \sum_{i=1}^n (1- y_i(x_i^T\beta+\gamma))^+ \frac{1}{2} \|\beta\|^2_2 + \lambda \|\beta\|_1
$$

Let's generate a bigger dataset


```python
N = 1000
P = 200

Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
X = np.random.standard_normal((N,P))
X[Y==1] += np.array([30,-20] + (P-2)*[0])[np.newaxis,:]
X -= X.mean(0)[np.newaxis,:]
```

The hinge loss is defined similarly, and we only need to add a sparsity penalty


```python
X_1 = np.hstack([X, np.ones((N,1))])
transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
C = 0.2
hinge = rr.positive_part(N, lagrange=C)
hinge_loss = rr.linear_atom(hinge, transform)

s = rr.selector(slice(0,P), (P+1,))
sparsity = rr.l1norm.linear(s, lagrange=0.2)
quadratic = rr.quadratic.linear(s, coef=0.5)
```


```python
problem = rr.dual_problem.fromprimal(quadratic, hinge_loss, sparsity)
solver = rr.FISTA(problem)
solver.fit()
solver.composite.coefs
```

In high dimensions, it becomes easier to separate
points.


```python
fits = np.dot(X_1, problem.coefs)
labels = 2 * (fits > 0) - 1
accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
accuracy
```


```python
import numpy as np
import regreg.api as rr

np.random.seed(400)

N = 1000
P = 200

Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
X = np.random.standard_normal((N,P))
X[Y==1] += np.array([30,-20] + (P-2)*[0])[np.newaxis,:]
X -= X.mean(0)[np.newaxis,:]

X_1 = np.hstack([X, np.ones((N,1))])
transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
C = 0.2
hinge = rr.positive_part(N, lagrange=C)
hinge_loss = rr.linear_atom(hinge, transform)

s = rr.selector(slice(0,P), (P+1,))
sparsity = rr.l1norm.linear(s, lagrange=0.2)
quadratic = rr.quadratic.linear(s, coef=0.5)
problem = rr.dual_problem.fromprimal(loss, hinge_loss, sparsity)
solver = rr.FISTA(problem)
solver.fit()
solver.composite.coefs


fits = np.dot(X_1, problem.coefs)
labels = 2 * (fits > 0) - 1
accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
print accuracy
```

Sparse Huberized SVM
~~~~~~~~~~~~~~~~~~~~

We can also smooth the hinge loss to yield a Huberized version of SVM.
In fact, it is easier to write the python code to specify the problem then
to write it out formally.

The hinge loss is defined similarly, and we only need to add a sparsity penalty


```python
X_1 = np.hstack([X, np.ones((N,1))])
transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
C = 0.2
hinge = rr.positive_part(N, lagrange=C)
hinge_loss = rr.linear_atom(hinge, transform)
epsilon = 0.04
Q = rr.identity_quadratic(epsilon, 0., 0., 0.)
smoothed_hinge_loss = hinge_loss.smoothed(Q)

s = rr.selector(slice(0,P), (P+1,))
sparsity = rr.l1norm.linear(s, lagrange=0.2)
quadratic = rr.quadratic.linear(s, coef=0.5)
```

Now, let's fit it. For this problem, we can use a known bound for the Lipschitz
constant. We'll first get a bound on the largest squared singular value of X


```python
from regreg.affine import power_L
singular_value_sq = power_L(X)
# the other smooth piece is a quadratic with identity
# for quadratic form, so its lipschitz constant is 1

lipschitz = 1.05 * singular_value_sq / epsilon + 1
```

Now, we can solve the problem without having to backtrack.


```python
problem = rr.dual_problem.fromprimal(quadratic, 
	                             smoothed_hinge_loss, 
                                     sparsity)
solver = rr.FISTA(problem)
solver.composite.lipschitz = lipschitz
solver.perform_backtrack = False
vals = solver.fit()
solver.composite.coefs
```

In high dimensions, it becomes easier to separate
points.


```python
fits = np.dot(X_1, problem.coefs)
labels = 2 * (fits > 0) - 1
accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
accuracy
```


```python
import numpy as np
import regreg.api as rr

np.random.seed(400)

N = 1000
P = 200

Y = 2 * np.random.binomial(1, 0.5, size=(N,)) - 1.
X = np.random.standard_normal((N,P))
X[Y==1] += np.array([30,-20] + (P-2)*[0])[np.newaxis,:]
X -= X.mean(0)[np.newaxis, :]

X_1 = np.hstack([X, np.ones((N,1))])
transform = rr.affine_transform(-Y[:,np.newaxis] * X_1, np.ones(N))
C = 0.2
hinge = rr.positive_part(N, lagrange=C)
hinge_loss = rr.linear_atom(hinge, transform)
epsilon = 0.04
Q = rr.identity_quadratic(epsilon, 0., 0., 0.)
smoothed_hinge_loss = hinge_loss.smoothed(Q)

s = rr.selector(slice(0,P), (P+1,))
sparsity = rr.l1norm.linear(s, lagrange=3.)
quadratic = rr.quadratic.linear(s, coef=0.5)


from regreg.affine import power_L
ltransform = rr.linear_transform(X_1)
singular_value_sq = power_L(X_1)
# the other smooth piece is a quadratic with identity
# for quadratic form, so its lipschitz constant is 1

lipschitz = 1.05 * singular_value_sq / epsilon + 1.1


problem = rr.dual_problem.fromprimal(quadratic, 
                                     smoothed_hinge_loss, 
                                     sparsity)
solver = rr.FISTA(problem)
solver.composite.lipschitz = lipschitz
solver.debug = True
solver.perform_backtrack = False
solver.fit()
solver.composite.coefs


fits = np.dot(X_1, problem.coefs)
labels = 2 * (fits > 0) - 1
accuracy = (1 - np.fabs(Y-labels).sum() / (2. * N))
print accuracy
```
