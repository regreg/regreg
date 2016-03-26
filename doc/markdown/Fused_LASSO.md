
## Fused LASSO with sparsity

This notebook provides some ways to solve the sparse fused LASSO problem

$$
\frac{1}{2}\|y - \beta\|^2_2 + \lambda_{1}\|D\beta\|_{1} + \lambda_2 \|\beta\|_1
$$


```python
# third party imports
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt
from scipy import sparse
%load_ext rpy2.ipython

import regreg.api as rr
from regreg.affine.fused_lasso import difference_transform
```

We will use the CGH data from the R package cghFLasso. You will have to have installed the package 'cghFLasso' in R, as well as installed rpy2.


```python
%%R -o Y
library(cghFLasso)
data(CGH)
Y = CGH$GBM.y
```


```python
n = Y.shape[0]
plt.figure(figsize=(20,10))
plt.scatter(np.arange(n), Y)
```

Let's specify two penalties, one for "smoothness", i.e. piecewise constant behaviour, the other for sparsity. For "smoothness" we need to create the
matrix of first order differences. There is a class of affine transforms in {\bf regreg.affine} that can compute this and return it as a 
sparse matrix.


```python
loss = rr.signal_approximator(Y)
sparsity = rr.l1norm(n, lagrange=0.3)
D = difference_transform(np.arange(n))
fused = rr.l1norm.linear(D, lagrange=2.8)
```

Finally, we will create the problem. We will solve this as a dual problem where the primal objective is {\bf loss}=${\cal L}$ so the dual objective is ${\cal L}^*(-u_s-D^Tu_f)$ with $u_s$ dual variables for the sparsity
term and $u_f$ dual variables for the fused term.


```python
problem = rr.dual_problem.fromprimal(loss, sparsity, fused)
smooth_and_sparse_coefs = problem.solve(tol=1.e-14)
plt.figure(figsize=(20,10))
plt.plot(np.arange(n), smooth_and_sparse_coefs, linewidth=3, c='r')
plt.scatter(np.arange(n), Y)
```

## Fused LASSO without sparsity

Of course, we can also solve the problem without the sparsity penalty.


```python
problem = rr.dual_problem.fromprimal(loss, fused)
smooth_coefs = problem.solve(tol=1.e-14)
plt.figure(figsize=(20,10))
plt.plot(np.arange(n), smooth_coefs, linewidth=3, c='r')
plt.scatter(np.arange(n), Y)
```

## Fused LASSO via smoothing the TV term

Rather than solve the dual problem, we might also consider smoothing the {\bf fused} penalty and keeping a sparsity penalty. The smoothing is accomplished by using
a small quadratic term. This results in a separable problem with just the $\ell_1$ penalty. There are now several smooth terms, we lump
them together in the container.


```python
sq = rr.identity_quadratic(0.001, 0, 0, 0)
fused_smooth = fused.smoothed(sq)
problem = rr.container(loss, fused_smooth, sparsity)
smoothed_smooth_and_sparse_coefs = problem.solve()
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smoothed_smooth_and_sparse_coefs, linewidth=4, c='r', label='Smoothed')
pylab.plot(np.arange(n), smooth_and_sparse_coefs, linewidth=2, c='yellow', linestyle='--', label='Unsmoothed')
pylab.scatter(np.arange(n), Y, facecolor='gray')
pylab.legend()
```

We can also solve the problem without the sparsity


```python

fused_smooth = fused.smoothed(sq)
problem = rr.container(loss, fused_smooth)
smoothed_smooth_coefs = problem.solve()
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smoothed_smooth_coefs, linewidth=4, c='r', label='Smoothed')
pylab.plot(np.arange(n), smooth_coefs, linewidth=2, c='yellow', linestyle='--', label='Unsmoothed')
pylab.scatter(np.arange(n), Y, facecolor='gray')
pylab.legend()
```

## Adding offsets to penalties

In some cases, we may want to shrink to some value other than zero. This can be achieved by adding an offset to the seminorm. For instance, we might
want to shrink towards 2. We might do this by solving
$$ \frac{1}{2}||y - \beta||^2_2 + \lambda_{1}||D\beta||_{1} + \lambda_2 \|\beta-2\|_1 $$
This is an offset of -2.


```python
sparsity_2 = rr.l1norm(n, offset=-2*np.ones(n), lagrange=1)
problem = rr.dual_problem.fromprimal(loss, sparsity_2, fused)
smooth_and_sparse2_coefs = problem.solve(tol=1.e-14)
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smooth_and_sparse2_coefs, linewidth=3, c='r')
pylab.scatter(np.arange(n), Y)
```

By increasing the penalty on the sparsity_2 term, we shrink all the way to the constant solution $2  \cdot \pmb{1}$.


```python
sparsity_2.lagrange = 3
problem = rr.dual_problem.fromprimal(loss, sparsity_2, fused)
smooth_and_sparse2_coefs = problem.solve(tol=1.e-14)
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smooth_and_sparse2_coefs, linewidth=3, c='r')
pylab.scatter(np.arange(n), Y)
```

Of course, we can set the offset to be some other vector as well


```python
sparsity_2.offset = -np.arange(n) * 2. / n
problem = rr.dual_problem.fromprimal(loss, sparsity_2, fused)
smooth_and_sparse2_coefs = problem.solve(tol=1.e-14)
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smooth_and_sparse2_coefs, linewidth=3, c='r')
pylab.scatter(np.arange(n), Y)
```

## Constraint form

We might also try a constrained form of the problem, rather than the Lagrange form. For instance, we might
constrain the $\ell_1$ norm to be less than or equal to 50.


```python
sparsity_bound = rr.l1norm(n, bound=50)
problem = rr.dual_problem.fromprimal(loss, sparsity_bound, fused)
smooth_and_sparse_coefs = problem.solve(tol=1.e-14)
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smooth_and_sparse_coefs, linewidth=3, c='r')
pylab.scatter(np.arange(n), Y)
print 'sparsity %0.1f' % np.fabs(smooth_and_sparse_coefs).sum()
```

Alternatively, we might constrain the $\ell_1$ norm of $D\hat{\beta}$ to be less than 10, and keep the Lagrange form for the sparsity.


```python

fused_bound = rr.l1norm.linear(D, bound=10)
problem = rr.dual_problem.fromprimal(loss, sparsity, fused_bound)
smooth_and_sparse_coefs = problem.solve(tol=1.e-14)
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smooth_and_sparse_coefs, linewidth=3, c='r')
pylab.scatter(np.arange(n), Y)
print 'fused %0.1f' % np.fabs(D*smooth_and_sparse_coefs).sum()
```

We can also keep them both in bound form. At a solution, of course, they may not be both tight.


```python
problem = rr.dual_problem.fromprimal(loss, sparsity_bound, fused_bound)
smooth_and_sparse_coefs = problem.solve(tol=1.e-14)
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smooth_and_sparse_coefs, linewidth=3, c='r')
pylab.scatter(np.arange(n), Y)
print 'sparsity: %0.1f' % np.fabs(smooth_and_sparse_coefs).sum()
print 'fused: %0.1f' % np.fabs(D*smooth_and_sparse_coefs).sum()
```


```python
sparsity_bound.bound = 500
problem = rr.dual_problem.fromprimal(loss, sparsity_bound, fused_bound)
smooth_and_sparse_coefs = problem.solve(tol=1.e-14)
pylab.figure(figsize=(20,10))
pylab.plot(np.arange(n), smooth_and_sparse_coefs, linewidth=3, c='r')
pylab.scatter(np.arange(n), Y)
print 'sparsity: %0.1f' % np.fabs(smooth_and_sparse_coefs).sum()
print 'fused: %0.1f' % np.fabs(D*smooth_and_sparse_coefs).sum()
```

# Sparse fused lasso tutorial

The sparse fused lasso minimizes the objective
$$
\frac{1}{2}||y - \beta||^{2}_{2} + \lambda_{1}||D\beta||_{1} + \lambda_2 \|\beta\|_1
$$
with
$$
D = \left(\begin{array}{rrrrrr} -1 & 1 & 0 & 0 & \cdots & 0 \\ 
0 & -1 & 1 & 0 & \cdots & 0 \\ &&&&\cdots &\\ 0 &0&0&\cdots & -1 & 1 \end{array}\right)
$$

To solve this problem using RegReg we begin by loading the necessary numerical libraries


```python
# third party imports
%pylab inline
import numpy as np
from scipy import sparse

import regreg.api as rr
```

Next, let's generate an example signal,


```python
Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
```

which looks like


```python
import numpy as np
import pylab
Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
pylab.scatter(np.arange(Y.shape[0]), Y)
```

Now we can create the problem object, beginning with the loss function


```python
loss = rr.signal_approximator(Y)
n = Y.shape[0]
```

there are other loss functions (squared error, logistic, etc) and any differentiable function can be specified. Next, we specifiy the seminorm for this problem by instantiating two l1norm objects,


```python
sparsity = rr.l1norm(Y.shape[0], lagrange=0.8)
```

which creates an l1norm object with :math:`\lambda_2=0.8`. The first argument specifies the length of the coefficient vector. The object sparsity now has a coefficient associated with it that we can access and change,


```python
sparsity
sparsity.lagrange += 1
sparsity.lagrange
```

Next, we create the fused lasso matrix and the associated l1norm object,


```python
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
print (D)
D = sparse.csr_matrix(D)
fused = rr.l1norm.linear(D, lagrange=25.5)
```

Here we first created D, converted it a sparse matrix, and then created an l1norm object with the sparse version of D and :math:`\lambda_1 = 25.5`. We can now combine the two l1norm objects and the loss function using the  container class


```python
problem = rr.container(loss, sparsity, fused)
problem # TODO: fix the latexify for container
```

We could still easily access the penalty parameter


```python
problem.nonsmooth_atoms
problem.nonsmooth_atoms[0].lagrange
```

Next, we can select our algorithm of choice and use it solve the problem,


```python
solver = rr.FISTA(problem)
solver.fit(max_its=100, tol=1e-10)
solution = problem.coefs
```

Here max_its represents primal (outer) iterations, and tol is the primal tolerance. 

We can then plot solution to see the result of the regression,


```python

import numpy as np
import pylab	
from scipy import sparse

import regreg.api as rr

Y = np.random.standard_normal(500); Y[100:150] += 7; Y[250:300] += 14
loss = rr.signal_approximator(Y) 
sparsity = rr.l1norm(len(Y), lagrange=0.8)
sparsity.lagrange += 1
D = (np.identity(500) + np.diag([-1]*499,k=1))[:-1]
D = sparse.csr_matrix(D)
fused = rr.l1norm.linear(D, lagrange=25.5)
problem = rr.container(sparsity, fused, loss)
solver = rr.FISTA(problem)
solver.fit(max_its=100, tol=1e-10)
solution = problem.coefs
pylab.plot(solution, c='g', linewidth=3)	
pylab.scatter(np.arange(Y.shape[0]), Y)
```
